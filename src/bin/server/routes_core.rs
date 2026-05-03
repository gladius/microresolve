//! Primary routing endpoint: query → scoring pipeline → multi-intent → response.

use crate::log_store::LogRecord;
use crate::routes_events::emit_queued;
use crate::state::*;
use axum::{extract::State, http::HeaderMap, routing::post, Json};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new().route("/api/resolve", post(resolve))
}

#[derive(serde::Deserialize)]
pub struct ResolveRequest {
    pub query: String,
    /// Per-request threshold override. If absent, falls back to the namespace's
    /// `default_threshold`, then to the compile-time default (0.3).
    #[serde(default)]
    pub threshold: Option<f32>,
    #[serde(default = "default_gap")]
    pub gap: f32,
    /// If false, skip logging to review queue (use for UI test/explore)
    #[serde(default = "default_log")]
    pub log: bool,
    /// If 1 or true, include per-round trace in response.
    #[serde(default)]
    pub trace: Option<serde_json::Value>,
}

impl ResolveRequest {
    fn wants_trace(&self) -> bool {
        match &self.trace {
            Some(serde_json::Value::Bool(b)) => *b,
            Some(serde_json::Value::Number(n)) => n.as_i64().unwrap_or(0) != 0,
            Some(serde_json::Value::String(s)) => s == "1" || s == "true",
            _ => false,
        }
    }
}

fn default_threshold() -> f32 {
    0.3
}
fn default_gap() -> f32 {
    1.5
}
fn default_log() -> bool {
    true
}

/// Multi-intent classification.
pub async fn resolve(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ResolveRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let t0 = std::time::Instant::now();
    let wants_trace = req.wants_trace();

    let (resolve_result, opt_trace) = match state.engine.try_namespace(&app_id) {
        Some(h) => h.resolve_with_options(
            &req.query,
            req.threshold,
            req.gap,
            default_threshold(),
            wants_trace,
        ),
        None => (microresolve::ResolveResult::default(), None),
    };

    let latency_us = t0.elapsed().as_micros() as u64;

    if resolve_result.intents.is_empty() {
        if req.log {
            let log_id = log_query(
                &state,
                LogRecord {
                    id: 0,
                    query: req.query.clone(),
                    app_id: app_id.clone(),
                    detected_intents: vec![],
                    confidence: "none".to_string(),
                    session_id: None,
                    timestamp_ms: now_ms(),
                    router_version: state
                        .engine
                        .try_namespace(&app_id)
                        .map(|h| h.version())
                        .unwrap_or(0),
                    source: "none".to_string(),
                },
            );
            emit_queued(&state, log_id, &req.query, &app_id);
        }

        let trace_val = opt_trace.map(|t| build_trace_json(&t));
        let mut resp = serde_json::json!({
            "intents": [],
            "disposition": "NoMatch",
            "routing_us": latency_us,
        });
        if let Some(tv) = trace_val {
            resp["trace"] = tv;
        }
        return Json(resp);
    }

    // ── Cross-provider disambiguation ─────────────────────────────────
    let mut scored: Vec<(String, f32)> = resolve_result
        .intents
        .iter()
        .map(|m| (m.id.clone(), m.score))
        .collect();
    if scored.len() > 1 {
        if let Some(h) = state.engine.try_namespace(&app_id) {
            h.deduplicate_by_provider(&mut scored, &req.query);
        }
    }

    // Rebuild result after dedup (may have fewer entries).
    let tokens: Vec<String> = opt_trace
        .as_ref()
        .map(|t| t.tokens.clone())
        .unwrap_or_default();
    let confidences: Vec<f32> = if let Some(h) = state.engine.try_namespace(&app_id) {
        scored
            .iter()
            .map(|(id, score)| h.confidence_for(*score, &tokens, id))
            .collect()
    } else {
        vec![0.0; scored.len()]
    };

    let threshold = req.threshold.unwrap_or_else(|| {
        state
            .engine
            .try_namespace(&app_id)
            .map(|h| h.resolve_threshold(None, default_threshold()))
            .unwrap_or_else(default_threshold)
    });
    let candidate_threshold = (threshold * 0.2_f32).max(0.05);

    let intents: Vec<serde_json::Value> = scored
        .iter()
        .zip(confidences.iter())
        .map(|((id, score), conf)| {
            let band = if *score >= threshold {
                "High"
            } else if *score >= candidate_threshold {
                "Medium"
            } else {
                "Low"
            };
            serde_json::json!({
                "id": id,
                "score": (*score * 100.0).round() / 100.0,
                "confidence": (*conf * 1000.0).round() / 1000.0,
                "band": band,
            })
        })
        .collect();

    let has_high = intents.iter().any(|i| i["band"].as_str() == Some("High"));
    let disposition = if has_high {
        "Confident"
    } else {
        "LowConfidence"
    };

    let detected_ids: Vec<String> = intents
        .iter()
        .map(|i| i["id"].as_str().unwrap_or("").to_string())
        .collect();

    if req.log {
        let best_confidence = if has_high { "high" } else { "low" };
        let log_id = log_query(
            &state,
            LogRecord {
                id: 0,
                query: req.query.clone(),
                app_id: app_id.clone(),
                detected_intents: detected_ids,
                confidence: best_confidence.to_string(),
                session_id: None,
                timestamp_ms: now_ms(),
                router_version: state
                    .engine
                    .try_namespace(&app_id)
                    .map(|h| h.version())
                    .unwrap_or(0),
                source: "router".to_string(),
            },
        );
        emit_queued(&state, log_id, &req.query, &app_id);
    }

    let mut resp = serde_json::json!({
        "intents": intents,
        "disposition": disposition,
        "routing_us": latency_us,
    });

    if let Some(t) = opt_trace {
        resp["trace"] = build_trace_json(&t);
    }

    Json(resp)
}

fn build_trace_json(t: &microresolve::ResolveTrace) -> serde_json::Value {
    serde_json::json!({
        "tokens": t.tokens,
        "all_scores": t.all_scores.iter().take(10).map(|(id, s)|
            serde_json::json!({"id": id, "score": (*s * 100.0).round() / 100.0})).collect::<Vec<_>>(),
        "multi": {
            "rounds": t.multi_round_trace.rounds,
            "stop_reason": t.multi_round_trace.stop_reason,
            "has_negation": t.negated,
        },
        "negated": t.negated,
        "threshold_applied": t.threshold_applied,
    })
}
