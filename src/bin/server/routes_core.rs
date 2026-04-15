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
        .route("/api/execute", post(execute))
}

#[derive(serde::Deserialize)]
pub struct RouteMultiRequest {
    pub query: String,
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default = "default_gap")]
    pub gap: f32,
    /// If false, skip logging to review queue (use for UI test/explore)
    #[serde(default = "default_log")]
    pub log: bool,
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

    // ── Layer 0: N-gram typo correction ──────────────────────────────────────
    let l0_query = {
        let ngram_map = state.ngram.read().unwrap();
        if let Some(ng) = ngram_map.get(&app_id) {
            ng.correct_query(&req.query)
        } else {
            req.query.clone()
        }
    };

    // ── Layer 1: Hebbian normalize + expand ──────────────────────────────────
    let (processed_query, hebbian_injected) = {
        let hebbian = state.hebbian.read().unwrap();
        if let Some(graph) = hebbian.get(&app_id) {
            let r = graph.preprocess(&l0_query);
            if r.was_modified {
                eprintln!("[hebbian/L1] {} | {:?} → {:?} (injected: {:?})",
                    app_id, r.original, r.normalized, r.injected);
            }
            (r.expanded, r.injected)
        } else {
            (l0_query, vec![])
        }
    };

    // ── Layer 2: Intent graph (IDF scoring + token consumption) ─────────────
    // Two passes: raw scores (for top-N ranking) and token-consumed (for confirmed).
    let (intent_graph_results, raw_ranked, query_has_negation): (Option<Vec<(String, f32)>>, Vec<(String, f32)>, bool) = {
        let ig_map = state.intent_graph.read().unwrap();
        match ig_map.get(&app_id) {
            Some(ig) => {
                // Raw single-pass scores (no token consumption) — for top-N ranking
                let (raw, neg) = ig.score_normalized(&processed_query);
                // Token-consumed scores — for confirmed intents
                let (consumed, _) = ig.score_multi_normalized(&processed_query, req.threshold, req.gap);
                (Some(consumed), raw, neg)
            }
            None => (None, vec![], false),
        }
    };

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
        } else if top_score < req.threshold * 2.0 {
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

        return Json(serde_json::json!({
            "confirmed": confirmed,
            "candidates": candidates,
            "ranked": ranked,
            "disposition": disposition,
            "relations": [],
            "metadata": {},
            "routing_us": latency_us,
            "source": "hebbian_l2",
            "hebbian": if hebbian_injected.is_empty() { serde_json::json!(null) }
                       else { serde_json::json!({"injected": hebbian_injected, "processed_query": processed_query}) },
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

    Json(serde_json::json!({
        "confirmed": [],
        "candidates": [],
        "disposition": "no_match",
        "relations": [],
        "metadata": {},
        "routing_us": latency_us,
        "source": "none",
        "hebbian": if hebbian_injected.is_empty() { serde_json::json!(null) }
                   else { serde_json::json!({"injected": hebbian_injected, "processed_query": processed_query}) },
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
    let tokens = asv_router::tokenizer::tokenize(query);

    // For each token, find which intents it maps to
    let ig_map = state.intent_graph.read().unwrap();
    let ig = match ig_map.get(app_id) {
        Some(ig) => ig,
        None => return,
    };

    let scored_ids: HashSet<&str> = scored.iter().map(|(id, _)| id.as_str()).collect();

    // Count unique words per intent: words that map to THIS intent but not others in the result set
    let mut unique_count: HashMap<&str, usize> = HashMap::new();
    for token in &tokens {
        let base = token.strip_prefix("not_").unwrap_or(token.as_str());
        if let Some(activations) = ig.word_intent.get(base) {
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

// ── Intent Programming: execute endpoint ──────────────────────────────────────

#[derive(serde::Deserialize)]
struct ExecuteRequest {
    query: String,
    #[serde(default)]
    history: Vec<serde_json::Value>,
}

/// Execute a conversation turn using Intent Programming.
///
/// Routes the query, resolves the active intent (new or continued),
/// assembles a focused system prompt, calls the LLM, and returns
/// the response with routing info and audit remark.
pub async fn execute(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ExecuteRequest>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let t0 = std::time::Instant::now();

    // Resolve turn using library
    let turn = {
        let ig_map = state.intent_graph.read().unwrap();
        let heb_map = state.hebbian.read().unwrap();
        let routers = state.routers.read().unwrap();
        match (ig_map.get(&app_id), routers.get(&app_id)) {
            (Some(ig), Some(router)) => {
                let l1 = heb_map.get(&app_id);
                asv_router::execute::resolve_turn(
                    ig, l1, router, &req.query, &req.history, ig.default_threshold(),
                )
            }
            _ => return Err((axum::http::StatusCode::NOT_FOUND,
                format!("namespace '{}' not found", app_id))),
        }
    };

    let routing_us = t0.elapsed().as_micros() as u64;

    let intent = match &turn.intent {
        Some(id) => id.clone(),
        None => return Ok(Json(serde_json::json!({
            "response": null,
            "routing": { "intent": null, "disposition": "no_match", "routing_us": routing_us },
        }))),
    };

    // Call LLM with assembled messages
    let llm_response = crate::pipeline::call_llm_with_messages(&state, &turn.messages, 1024).await;
    let total_us = t0.elapsed().as_micros() as u64;

    match llm_response {
        Ok(text) => {
            let (clean_response, remark) = asv_router::execute::extract_remark(&text);
            Ok(Json(serde_json::json!({
                "response": clean_response,
                "remark": remark,
                "routing": {
                    "intent": intent,
                    "is_transition": turn.is_transition,
                    "disposition": if turn.is_transition {
                        &turn.route_result.disposition
                    } else { "continued" },
                    "routing_us": routing_us,
                    "total_us": total_us,
                },
            })))
        }
        Err((_status, msg)) => Ok(Json(serde_json::json!({
            "response": null,
            "error": msg,
            "routing": { "intent": intent, "is_transition": turn.is_transition },
        }))),
    }
}
