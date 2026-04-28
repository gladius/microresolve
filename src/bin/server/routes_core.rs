//! Primary routing endpoints — Hebbian L1+L2 is the sole router.

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
    /// Skip L1 morphology/abbreviation rewriting entirely.
    #[serde(default)]
    pub disable_l1: bool,
    /// Vestigial — synonym expansion was removed; both flags now produce
    /// identical L1 behaviour. Kept for wire-format compatibility.
    #[serde(default)]
    pub grounded_l1: bool,
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

/// Multi-intent classification via Hebbian L1+L2.
pub async fn route_multi(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteMultiRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let t0 = std::time::Instant::now();

    // ── Full L0→L1→L2 pipeline via Resolver ────────────────────────────────────
    // Run all three layers inside a single read lock on `routers`. Multi-intent
    // scoring captures a trace of rounds so the UI can render per-layer cards
    // without a second API call.
    type PipelineOut = (
        Option<Vec<(String, f32)>>, // confirmed (token-consumed)
        Vec<(String, f32)>,         // raw_ranked (single-pass)
        bool,                       // has_negation
        String,                     // l0_corrected
        String,                     // l1_normalized
        String,                     // l1_expanded (= processed)
        Vec<String>,                // l1_injected
        Vec<String>,                // l2_tokens
        Option<microresolve::scoring::MultiIntentTrace>,
        f32, // effective_threshold (cascade-resolved)
    );
    let pipeline: PipelineOut = match state.engine.try_namespace(&app_id) {
        Some(h) => h.with_resolver(|router| {
            let effective_threshold = router.resolve_threshold(req.threshold, default_threshold());
            let q0 = router.l0().correct_query(&req.query);
            let preprocessed = if req.disable_l1 {
                microresolve::scoring::PreprocessResult {
                    original: q0.clone(),
                    normalized: q0.clone(),
                    expanded: q0.clone(),
                    injected: vec![],
                    semantic_hits: vec![],
                    was_modified: false,
                }
            } else if req.grounded_l1 {
                let known: std::collections::HashSet<&str> =
                    router.l2().word_intent.keys().map(|s| s.as_str()).collect();
                router.l1().preprocess_grounded(&q0, &known)
            } else {
                router.l1().preprocess(&q0)
            };
            if preprocessed.was_modified {
                eprintln!(
                    "[hebbian/L1] {} | {:?} → {:?} (injected: {:?})",
                    app_id, preprocessed.original, preprocessed.normalized, preprocessed.injected
                );
            }
            let processed = preprocessed.expanded.clone();
            let injected = preprocessed.injected.clone();
            let normalized = preprocessed.normalized.clone();
            let tokens: Vec<String> = microresolve::tokenizer::tokenize(&processed);
            let (raw, neg) = router.l2().score_normalized(&processed);
            let (consumed, _neg2, trace) = router.l2().score_multi_normalized_traced(
                &processed,
                effective_threshold,
                req.gap,
                true,
            );
            (
                Some(consumed),
                raw,
                neg,
                q0,
                normalized,
                processed,
                injected,
                tokens,
                trace,
                effective_threshold,
            )
        }),
        None => (
            None,
            vec![],
            false,
            req.query.clone(),
            req.query.clone(),
            req.query.clone(),
            vec![],
            vec![],
            None,
            default_threshold(),
        ),
    };
    let (
        intent_graph_results,
        raw_ranked,
        query_has_negation,
        l0_corrected,
        l1_normalized,
        processed_query,
        hebbian_injected,
        l2_tokens,
        multi_trace,
        effective_threshold,
    ) = pipeline;

    let latency_us = t0.elapsed().as_micros() as u64;

    if let Some(mut scored) = intent_graph_results.filter(|s| !s.is_empty()) {
        // ── Cross-provider disambiguation ─────────────────────────────────
        // When the same action appears from multiple providers (e.g.,
        // shopify:list_customers + stripe:list_customers), pick the provider
        // whose unique query words match best. Only affects duplicates —
        // different actions are never touched.
        if scored.len() > 1 {
            if let Some(h) = state.engine.try_namespace(&app_id) {
                h.with_resolver(|r| r.disambiguate_cross_provider(&mut scored, &processed_query));
            }
        }

        let top_score = scored[0].1;
        let max_score = top_score;

        // ── L5 Disposition: score distribution shape ─────────────────────────
        // "confident"      — top score solidly above threshold (single OR multi-intent)
        // "low_confidence" — top score barely above threshold; verify before acting
        //
        // Multi-intent queries ("cancel and refund") legitimately fire several
        // intents at similar scores — that is correct behaviour, not ambiguity.
        // Caller can iterate `confirmed[]` to act on each.
        let disposition = if top_score < effective_threshold * 2.0 {
            "low_confidence"
        } else {
            "confident"
        };

        let intents: Vec<serde_json::Value> = scored
            .iter()
            .map(|(id, score)| {
                let confidence = if *score >= max_score * 0.8 {
                    "high"
                } else if *score >= max_score * 0.5 {
                    "medium"
                } else {
                    "low"
                };
                serde_json::json!({
                    "id": id,
                    "score": (*score * 100.0).round() / 100.0,
                    "confidence": confidence,
                    "source": "hebbian_l2",
                    "position": 0,
                    "span": [0, req.query.len()],
                    "intent_type": "Action",
                    "negated": query_has_negation,
                })
            })
            .collect();

        let confirmed: Vec<&serde_json::Value> = intents
            .iter()
            .filter(|i| i["confidence"].as_str() != Some("low"))
            .collect();
        let candidates: Vec<&serde_json::Value> = intents
            .iter()
            .filter(|i| i["confidence"].as_str() == Some("low"))
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
                        .map(|h| h.with_resolver(|r| r.version()))
                        .unwrap_or(0),
                    source: "hebbian_l2".to_string(),
                },
            );
            emit_queued(&state, log_id, &req.query, &app_id);
        }

        // Top-N ranked list from raw IDF (before token consumption)
        let ranked: Vec<serde_json::Value> = raw_ranked.iter().take(5).map(|(id, score)| {
            serde_json::json!({"id": id, "score": (*score * 100.0).round() / 100.0})
        }).collect();

        let trace = serde_json::json!({
            "l0_corrected": l0_corrected,
            "l1_normalized": l1_normalized,
            "l1_expanded": processed_query,
            "l1_injected": hebbian_injected,
            "l1_disabled": req.disable_l1,
            "tokens": l2_tokens,
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
            "source": "hebbian_l2",
            "hebbian": if hebbian_injected.is_empty() { serde_json::json!(null) }
                       else { serde_json::json!({"injected": hebbian_injected, "processed_query": processed_query}) },
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
                    .map(|h| h.with_resolver(|r| r.version()))
                    .unwrap_or(0),
                source: "none".to_string(),
            },
        );
        emit_queued(&state, log_id, &req.query, &app_id);
    }

    let trace = serde_json::json!({
        "l0_corrected": l0_corrected,
        "l1_normalized": l1_normalized,
        "l1_expanded": processed_query,
        "l1_injected": hebbian_injected,
        "l1_disabled": req.disable_l1,
        "tokens": l2_tokens,
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
        "hebbian": if hebbian_injected.is_empty() { serde_json::json!(null) }
                   else { serde_json::json!({"injected": hebbian_injected, "processed_query": processed_query}) },
        "trace": trace,
    }))
}
