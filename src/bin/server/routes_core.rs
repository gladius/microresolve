//! Primary routing endpoints — Hebbian L1+L2 is the sole router.

use axum::{
    extract::State,
    http::HeaderMap,
    routing::post,
    Json,
};
use crate::state::*;
use crate::log_store::LogRecord;
use crate::routes_events::emit_queued;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/route_multi", post(route_multi))
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
    /// Apply char-ngram Jaccard tiebreaker when top-1/top-2 scores are close.
    /// Safe default is off for backward compatibility.
    #[serde(default)]
    pub tiebreaker: bool,
    /// Skip L1 synonym/morphology expansion entirely.
    #[serde(default)]
    pub disable_l1: bool,
    /// Run L1 morphological normalization only — no synonym expansion.
    #[serde(default)]
    pub morphology_only: bool,
    /// Vocabulary-grounded synonym expansion: only expand tokens NOT already in L2.
    #[serde(default)]
    pub grounded_l1: bool,
    /// Return per-layer debug trace in the response under "debug".
    #[serde(default)]
    pub debug: bool,
    /// Enable the optional entity-detection layer (PoC).
    /// When true, runs hybrid PII detector before L0 and appends entity-type
    /// tokens (e.g., `[CC]`, `[SSN]`) to the query so intents trained on those
    /// tokens score appropriately. Default off.
    #[serde(default)]
    pub enable_entity_layer: bool,
}

fn default_threshold() -> f32 { 0.3 }
fn default_gap() -> f32 { 1.5 }
fn default_log() -> bool { true }

/// Multi-intent routing via Hebbian L1+L2.
pub async fn route_multi(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteMultiRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let t0 = std::time::Instant::now();

    // ── Full L0→L1→L2 pipeline via Router ────────────────────────────────────
    // Run all three layers inside a single read lock on `routers`. Multi-intent
    // scoring captures a trace of rounds so the UI can render per-layer cards
    // without a second API call.
    type PipelineOut = (
        Option<Vec<(String, f32)>>,              // confirmed (token-consumed)
        Vec<(String, f32)>,                      // raw_ranked (single-pass)
        bool,                                    // has_negation
        String,                                  // l0_corrected
        String,                                  // l1_normalized
        String,                                  // l1_expanded (= processed)
        Vec<String>,                             // l1_injected
        Vec<String>,                             // l2_tokens
        Option<microresolve::scoring::MultiIntentTrace>,
        f32,                                     // effective_threshold (cascade-resolved)
    );
    let pipeline: PipelineOut = {
        let routers = state.routers.read().unwrap();
        match routers.get(&app_id) {
            Some(router) => {
                // Threshold cascade: request override > namespace default > compile-time default.
                // Logic centralized in Router so all bindings (Node/Python/WASM) stay in sync.
                let effective_threshold = router.resolve_threshold(req.threshold, default_threshold());

                // Entity layer (optional): augments query with mr_pii_<label>
                // tokens for any detected entities. Cached per-namespace —
                // typical cost ~3µs, p99 ~20µs (CPU-limited).
                let query_for_l0 = if let Some(layer) = router.entity_layer() {
                    layer.augment(&req.query)
                } else if req.enable_entity_layer {
                    microresolve::entity::EntityLayer::recommended().augment(&req.query)
                } else {
                    req.query.clone()
                };

                // L0: typo correction
                let q0 = router.l0().correct_query(&query_for_l0);

                // L1: normalize + expand (mode selected by request flags)
                let preprocessed = if req.disable_l1 {
                    microresolve::scoring::PreprocessResult {
                        original: q0.clone(),
                        normalized: q0.clone(),
                        expanded: q0.clone(),
                        injected: vec![],
                        semantic_hits: vec![],
                        was_modified: false,
                    }
                } else if req.morphology_only {
                    router.l1().preprocess_morphonly(&q0)
                } else if req.grounded_l1 {
                    let known: std::collections::HashSet<&str> =
                        router.l2().word_intent.keys().map(|s| s.as_str()).collect();
                    router.l1().preprocess_grounded(&q0, &known)
                } else {
                    router.l1().preprocess(&q0)
                };
                if preprocessed.was_modified {
                    eprintln!("[hebbian/L1] {} | {:?} → {:?} (injected: {:?})",
                        app_id, preprocessed.original, preprocessed.normalized, preprocessed.injected);
                }
                let processed = preprocessed.expanded.clone();
                let injected = preprocessed.injected.clone();
                let normalized = preprocessed.normalized.clone();
                let tokens: Vec<String> = microresolve::tokenizer::tokenize(&processed);

                // L2 raw scores (single-pass, pre-token-consumption) for top-N ranking
                let (raw, neg) = router.l2().score_normalized(&processed);
                let raw = if req.tiebreaker {
                    router.l2().apply_char_ngram_tiebreaker(&processed, raw, 0.65, 0.5)
                } else { raw };

                // L2 multi-intent with trace (rounds)
                let (consumed, _neg2, trace) = router.l2().score_multi_normalized_traced(
                    &processed, effective_threshold, req.gap, true,
                );
                let consumed = if req.tiebreaker {
                    router.l2().apply_char_ngram_tiebreaker(&processed, consumed, 0.65, 0.5)
                } else { consumed };

                (Some(consumed), raw, neg, q0, normalized, processed, injected, tokens, trace, effective_threshold)
            }
            None => (None, vec![], false, req.query.clone(), req.query.clone(),
                    req.query.clone(), vec![], vec![], None, default_threshold()),
        }
    };
    let (intent_graph_results, raw_ranked, query_has_negation,
         l0_corrected, l1_normalized, processed_query, hebbian_injected, l2_tokens, multi_trace,
         effective_threshold) = pipeline;

    let latency_us = t0.elapsed().as_micros() as u64;

    if let Some(mut scored) = intent_graph_results.filter(|s| !s.is_empty()) {
        // ── Cross-provider disambiguation ─────────────────────────────────
        // When the same action appears from multiple providers (e.g.,
        // shopify:list_customers + stripe:list_customers), pick the provider
        // whose unique query words match best. Only affects duplicates —
        // different actions are never touched.
        if scored.len() > 1 {
            disambiguate_cross_provider(&mut scored, &processed_query, &state, &app_id);
        }

        let top_score = scored[0].1;
        let max_score = top_score;

        // ── L5 Disposition: score distribution shape ─────────────────────────
        // "confident"      — clear winner; system is sure
        // "low_confidence" — top score barely above threshold; verify before acting
        // "escalate"       — 3+ intents at similar scores; query is genuinely ambiguous
        let disposition = if scored.len() >= 3 && scored[2].1 / top_score >= 0.75 {
            "escalate"        // tight cluster of ≥3 — can't rank, need clarification
        } else if top_score < effective_threshold * 2.0 {
            "low_confidence"  // marginally above detection threshold
        } else {
            "confident"
        };

        let intents: Vec<serde_json::Value> = scored.iter().map(|(id, score)| {
            let confidence = if *score >= max_score * 0.8 { "high" }
                else if *score >= max_score * 0.5 { "medium" }
                else { "low" };
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
        }).collect();

        let confirmed: Vec<&serde_json::Value> = intents.iter()
            .filter(|i| i["confidence"].as_str() != Some("low"))
            .collect();
        let candidates: Vec<&serde_json::Value> = intents.iter()
            .filter(|i| i["confidence"].as_str() == Some("low"))
            .collect();

        let detected_ids: Vec<String> = intents.iter()
            .map(|i| i["id"].as_str().unwrap_or("").to_string())
            .collect();
        let best_confidence = if !confirmed.is_empty() { "high" } else { "low" };
        let flag = LogRecord::compute_flag(&detected_ids, best_confidence);

        if req.log {
            let log_id = log_query(&state, LogRecord {
                id: 0,
                query: req.query.clone(),
                app_id: app_id.clone(),
                detected_intents: detected_ids,
                confidence: best_confidence.to_string(),
                flag: flag.clone(),
                session_id: None,
                timestamp_ms: now_ms(),
                router_version: {
                    state.routers.read().unwrap()
                        .get(&app_id).map(|r| r.version()).unwrap_or(0)
                },
                source: "hebbian_l2".to_string(),
            });
            emit_queued(&state, log_id, &req.query, &app_id, flag);
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
    let flag = LogRecord::compute_flag(&[], "none");
    if req.log {
        let log_id = log_query(&state, LogRecord {
            id: 0,
            query: req.query.clone(),
            app_id: app_id.clone(),
            detected_intents: vec![],
            confidence: "none".to_string(),
            flag: flag.clone(),
            session_id: None,
            timestamp_ms: now_ms(),
            router_version: state.routers.read().unwrap()
                .get(&app_id).map(|r| r.version()).unwrap_or(0),
            source: "none".to_string(),
        });
        emit_queued(&state, log_id, &req.query, &app_id, flag);
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

/// Cross-provider disambiguation: when the same action name appears from multiple
/// providers (e.g., shopify:list_customers + stripe:list_customers), use query
/// word exclusivity to pick the best provider. Only affects duplicates.
fn disambiguate_cross_provider(
    scored: &mut Vec<(String, f32)>,
    query: &str,
    state: &ServerState,
    app_id: &str,
) {
    use std::collections::{HashMap, HashSet};

    if scored.len() < 2 { return; }

    // Group by action name (part after ':')
    let mut action_groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, (id, _)) in scored.iter().enumerate() {
        let action = id.split(':').nth(1).unwrap_or(id.as_str());
        action_groups.entry(action).or_default().push(i);
    }

    // Find groups with duplicates (same action, different provider)
    let duplicate_groups: Vec<Vec<usize>> = action_groups.values()
        .filter(|indices| indices.len() > 1)
        .cloned()
        .collect();

    if duplicate_groups.is_empty() { return; }

    // Get query tokens
    let tokens = microresolve::tokenizer::tokenize(query);

    // For each token, find which intents it maps to
    let routers = state.routers.read().unwrap();
    let router = match routers.get(app_id) {
        Some(r) => r,
        None => return,
    };

    let scored_ids: HashSet<&str> = scored.iter().map(|(id, _)| id.as_str()).collect();

    // Count unique words per intent: words that map to THIS intent but not others in the result set
    let mut unique_count: HashMap<&str, usize> = HashMap::new();
    for token in &tokens {
        let base = token.strip_prefix("not_").unwrap_or(token.as_str());
        if let Some(activations) = router.l2().word_intent.get(base) {
            let matching: Vec<&str> = activations.iter()
                .filter(|(id, _)| scored_ids.contains(id.as_str()))
                .map(|(id, _)| id.as_str())
                .collect();
            if matching.len() == 1 {
                *unique_count.entry(matching[0]).or_insert(0) += 1;
            }
        }
    }

    // For each duplicate group, keep only the intent with most unique words
    let mut to_remove: HashSet<usize> = HashSet::new();
    for group in &duplicate_groups {
        let best = group.iter()
            .max_by_key(|&&i| unique_count.get(scored[i].0.as_str()).copied().unwrap_or(0));
        if let Some(&best_idx) = best {
            let best_unique = unique_count.get(scored[best_idx].0.as_str()).copied().unwrap_or(0);
            if best_unique > 0 {
                // Remove all others in this group
                for &i in group {
                    if i != best_idx { to_remove.insert(i); }
                }
            }
            // If no unique words for any candidate, keep all (genuinely ambiguous)
        }
    }

    if !to_remove.is_empty() {
        let mut i = 0;
        scored.retain(|_| { let keep = !to_remove.contains(&i); i += 1; keep });
    }
}

