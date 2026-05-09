//! Primary routing endpoint: query → scoring pipeline → multi-intent → response.

use crate::audit_log::hash_query;
use crate::log_store::LogRecord;
use crate::routes_events::emit_queued;
use crate::state::*;
use axum::{extract::State, http::HeaderMap, routing::post, Extension, Json};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new().route("/api/resolve", post(resolve))
}

#[derive(serde::Deserialize)]
pub struct ResolveRequest {
    pub query: String,
    /// Per-request threshold override. If absent, falls back to the namespace's
    /// `default_threshold`, then to the compile-time default (1.0).
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
    Extension(KeyName(kid)): Extension<KeyName>,
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

        // Audit: record the no-match decision too (compliance buyers
        // need to see "the system saw this query and declined to fire").
        let no_match_trace = opt_trace.as_ref().map(build_compact_audit_trace);
        audit_resolve(
            &state,
            &kid,
            &app_id,
            &req.query,
            &[],
            0.0,
            latency_us,
            no_match_trace,
        );

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
                detected_intents: detected_ids.clone(),
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

    // ── Audit log: tamper-evident decision record ────────────────────
    // When the caller asked for a trace, embed a compact summary in the
    // audit payload too — this is what makes Art. 13 interpretive
    // transparency real (you can defend not just "we routed" but "we
    // routed because tokens X, Y, Z").
    let compact_trace = opt_trace.as_ref().map(build_compact_audit_trace);
    audit_resolve(
        &state,
        &kid,
        &app_id,
        &req.query,
        &intents,
        threshold,
        latency_us,
        compact_trace,
    );

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

/// Append a `resolve` event to the audit log when the effective audit
/// mode is enabled. Caller already computed everything; this just
/// shapes the payload and serializes the chain write. The query is
/// stored as a SHA-256 hash (PII-friendly) — auditors can verify
/// "decision X happened for query Y" by hashing Y and looking it up,
/// without the operator retaining raw queries. When `compact_trace`
/// is supplied, it lands inside the payload — surfaces *why* a routing
/// happened, not just *that* it happened (Art. 13 interpretive
/// transparency in the audit chain).
fn audit_resolve(
    state: &AppState,
    kid: &str,
    app_id: &str,
    query: &str,
    intents: &[serde_json::Value],
    threshold: f32,
    latency_us: u64,
    compact_trace: Option<serde_json::Value>,
) {
    if !state.audit_log.mode().enabled() {
        return;
    }
    let mut payload = serde_json::json!({
        "ns": app_id,
        "query_hash": hash_query(query),
        "intents": intents,
        "threshold_applied": threshold,
        "latency_us": latency_us,
    });
    if let Some(t) = compact_trace {
        payload["trace"] = t;
    }
    state.audit_log.record(kid, app_id, "resolve", payload);
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
        "per_token": t.per_token.iter().map(|c| serde_json::json!({
            "token": c.token,
            "intent": c.intent,
            "weight": (c.weight * 1000.0).round() / 1000.0,
            "idf": (c.idf * 1000.0).round() / 1000.0,
            "delta": (c.delta * 1000.0).round() / 1000.0,
            "negated": c.negated,
        })).collect::<Vec<_>>(),
        "per_intent": t.per_intent.iter().map(|s| serde_json::json!({
            "intent": s.intent,
            "raw_score": (s.raw_score * 100.0).round() / 100.0,
            "voting_tokens": s.voting_tokens,
            "voting_multiplier": (s.voting_multiplier * 100.0).round() / 100.0,
            "policy_overrides_bonus": (s.policy_overrides_bonus * 100.0).round() / 100.0,
            "policy_overrides_fired": s.policy_overrides_fired,
        })).collect::<Vec<_>>(),
        "explanation": t.explanation,
    })
}

/// Compact trace summary for audit log entries: top intents (with voting state
/// + any conjunctions that fired) + top 5 token contributions. Designed to be
/// small enough to live inside every resolve audit event without bloating the
/// chain. Full trace stays in the API response only when requested.
fn build_compact_audit_trace(t: &microresolve::ResolveTrace) -> serde_json::Value {
    let top_intents: Vec<serde_json::Value> = t
        .per_intent
        .iter()
        .take(3)
        .map(|s| {
            serde_json::json!({
                "intent": s.intent,
                "raw_score": (s.raw_score * 100.0).round() / 100.0,
                "voting_tokens": s.voting_tokens,
                "policy_overrides_fired": s.policy_overrides_fired,
            })
        })
        .collect();
    // Top 5 by absolute delta, picking the highest-impact contributions only.
    let mut sorted_contrib: Vec<&microresolve::scoring::TokenContribution> =
        t.per_token.iter().collect();
    sorted_contrib.sort_by(|a, b| b.delta.abs().partial_cmp(&a.delta.abs()).unwrap_or(std::cmp::Ordering::Equal));
    let top_tokens: Vec<serde_json::Value> = sorted_contrib
        .iter()
        .take(5)
        .map(|c| {
            serde_json::json!({
                "token": c.token,
                "intent": c.intent,
                "delta": (c.delta * 1000.0).round() / 1000.0,
            })
        })
        .collect();
    serde_json::json!({
        "top_intents": top_intents,
        "top_tokens": top_tokens,
        "explanation": t.explanation,
    })
}
