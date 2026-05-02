//! Primary routing endpoint: query → L2 (IDF + Hebbian) → multi-intent → response.

use crate::log_store::LogRecord;
use crate::routes_events::emit_queued;
use crate::state::*;
use axum::{extract::State, http::HeaderMap, routing::post, Json};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new().route("/api/route_multi", post(route_multi))
}

#[derive(serde::Deserialize)]
pub struct RouteMultiRequest {
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

/// Multi-intent classification via L2 (IDF + Hebbian).
pub async fn route_multi(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteMultiRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let t0 = std::time::Instant::now();

    let (intent_graph_results, raw_ranked, query_has_negation, tokens, multi_trace) =
        match state.engine.try_namespace(&app_id) {
            Some(h) => {
                let p = h.route_multi(
                    &req.query,
                    req.threshold,
                    req.gap,
                    true,
                    default_threshold(),
                );
                (Some(p.multi), p.raw, p.negated, p.tokens, p.trace)
            }
            None => (None, vec![], false, vec![], None),
        };

    let latency_us = t0.elapsed().as_micros() as u64;

    if let Some(mut scored) = intent_graph_results.filter(|s| !s.is_empty()) {
        // ── Cross-provider disambiguation ─────────────────────────────────
        // When the same action appears from multiple providers (e.g.,
        // shopify:list_customers + stripe:list_customers), pick the provider
        // whose unique query words match best. Only affects duplicates —
        // different actions are never touched.
        if scored.len() > 1 {
            if let Some(h) = state.engine.try_namespace(&app_id) {
                h.disambiguate_cross_provider(&mut scored, &req.query);
            }
        }

        // ── Per-intent confidence (normalized 0-1) ───────────────────────────
        // Raw scores are unbounded sums of `weight × IDF` — fine for ranking,
        // useless for human comparison or disposition gates. Confidence is
        // `raw_score / intent_max_score(query, intent)` clamped to [0,1] —
        // "what fraction of THIS intent's relevant content matched the query."
        // Stable across namespace sizes; what disposition + UI consume.
        let confidences: Vec<f32> = if let Some(h) = state.engine.try_namespace(&app_id) {
            scored
                .iter()
                .map(|(id, score)| h.confidence_for(*score, &tokens, id))
                .collect()
        } else {
            vec![0.0; scored.len()]
        };
        let top_confidence = confidences.first().copied().unwrap_or(0.0);

        // ── L5 Disposition: now driven by normalized confidence ─────────────
        // confident:      top intent's normalized confidence ≥ 0.5
        // low_confidence: 0 < confidence < 0.5 — ranked candidates exist but the
        //                 best-fit intent only matched a fraction of its training
        let disposition = if top_confidence >= 0.5 {
            "confident"
        } else {
            "low_confidence"
        };

        let intents: Vec<serde_json::Value> = scored
            .iter()
            .zip(confidences.iter())
            .map(|((id, score), conf)| {
                // Categorical band derived from numeric confidence — UI hint.
                let band = if *conf >= 0.7 {
                    "high"
                } else if *conf >= 0.4 {
                    "medium"
                } else {
                    "low"
                };
                serde_json::json!({
                    "id": id,
                    "score": (*score * 100.0).round() / 100.0,
                    "confidence": (*conf * 1000.0).round() / 1000.0,  // 3-decimal rounded
                    "band": band,
                    "source": "router",
                    "position": 0,
                    "span": [0, req.query.len()],
                    "intent_type": "Action",
                    "negated": query_has_negation,
                })
            })
            .collect();

        let confirmed: Vec<&serde_json::Value> = intents
            .iter()
            .filter(|i| i["band"].as_str() != Some("low"))
            .collect();
        let candidates: Vec<&serde_json::Value> = intents
            .iter()
            .filter(|i| i["band"].as_str() == Some("low"))
            .collect();

        let detected_ids: Vec<String> = intents
            .iter()
            .map(|i| i["id"].as_str().unwrap_or("").to_string())
            .collect();
        let best_confidence = if !confirmed.is_empty() { "high" } else { "low" };

        if req.log {
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

        // Top-N ranked list from raw IDF (before token consumption)
        let ranked: Vec<serde_json::Value> = raw_ranked.iter().take(5).map(|(id, score)| {
            serde_json::json!({"id": id, "score": (*score * 100.0).round() / 100.0})
        }).collect();

        let trace = serde_json::json!({
            "tokens": tokens,
            "all_scores": raw_ranked.iter().take(10).map(|(id, s)|
                serde_json::json!({"id": id, "score": (*s * 100.0).round() / 100.0})).collect::<Vec<_>>(),
            "multi": multi_trace.as_ref().map(|t| serde_json::json!({
                "rounds": t.rounds,
                "stop_reason": t.stop_reason,
                "has_negation": query_has_negation,
            })),
        });

        return Json(serde_json::json!({
            "confirmed": confirmed,
            "candidates": candidates,
            "ranked": ranked,
            "disposition": disposition,
            "relations": [],
            "routing_us": latency_us,
            "source": "router",
            "trace": trace,
        }));
    }

    // ── No match — log and return empty (triggers auto-learn in "auto" mode) ──
    let latency_us = t0.elapsed().as_micros() as u64;
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

    let trace = serde_json::json!({
        "tokens": tokens,
        "all_scores": [],
        "multi": multi_trace.as_ref().map(|t| serde_json::json!({
            "rounds": t.rounds,
            "stop_reason": t.stop_reason,
            "has_negation": query_has_negation,
        })),
    });

    Json(serde_json::json!({
        "confirmed": [],
        "candidates": [],
        "disposition": "no_match",
        "relations": [],
        "routing_us": latency_us,
        "source": "none",
        "trace": trace,
    }))
}
