//! Primary routing endpoints — Hebbian L1+L3 is the sole router.

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
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default = "default_gap")]
    pub gap: f32,
}

fn default_threshold() -> f32 { 0.3 }
fn default_gap() -> f32 { 1.5 }

/// Multi-intent routing via Hebbian L1+L3.
pub async fn route_multi(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteMultiRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let t0 = std::time::Instant::now();

    // ── Layer 1: Hebbian normalize + expand ──────────────────────────────────
    let (processed_query, hebbian_injected) = {
        let hebbian = state.hebbian.read().unwrap();
        if let Some(graph) = hebbian.get(&app_id) {
            let r = graph.preprocess(&req.query);
            if r.was_modified {
                eprintln!("[hebbian/L1] {} | {:?} → {:?} (injected: {:?})",
                    app_id, r.original, r.normalized, r.injected);
            }
            (r.expanded, r.injected)
        } else {
            (req.query.clone(), vec![])
        }
    };

    // ── Layer 3: Intent graph (spreading activation) ──────────────────────────
    let (intent_graph_results, query_has_negation): (Option<Vec<(String, f32)>>, bool) = {
        let ig_map = state.intent_graph.read().unwrap();
        match ig_map.get(&app_id) {
            Some(ig) => {
                let (scores, neg) = ig.score_multi_normalized(&processed_query, req.threshold, req.gap);
                (Some(scores), neg)
            }
            None => (None, false),
        }
    };

    let latency_us = t0.elapsed().as_micros() as u64;

    if let Some(scored) = intent_graph_results.filter(|s| !s.is_empty()) {
        let max_score = scored.iter().map(|(_, s)| *s).fold(0f32, f32::max);

        let intents: Vec<serde_json::Value> = scored.iter().map(|(id, score)| {
            let confidence = if *score >= max_score * 0.8 { "high" }
                else if *score >= max_score * 0.5 { "medium" }
                else { "low" };
            serde_json::json!({
                "id": id,
                "score": (*score * 100.0).round() / 100.0,
                "confidence": confidence,
                "source": "hebbian_l3",
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
            source: "hebbian_l3".to_string(),
        });

        emit_queued(&state, log_id, &req.query, &app_id, flag);

        return Json(serde_json::json!({
            "confirmed": confirmed,
            "candidates": candidates,
            "relations": [],
            "metadata": {},
            "routing_us": latency_us,
            "source": "hebbian_l3",
            "hebbian": if hebbian_injected.is_empty() { serde_json::json!(null) }
                       else { serde_json::json!({"injected": hebbian_injected, "processed_query": processed_query}) },
        }));
    }

    // ── No match — log and return empty (triggers auto-learn in "auto" mode) ──
    let latency_us = t0.elapsed().as_micros() as u64;
    let flag = LogRecord::compute_flag(&[], "none");
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

    Json(serde_json::json!({
        "confirmed": [],
        "candidates": [],
        "relations": [],
        "metadata": {},
        "routing_us": latency_us,
        "source": "none",
        "hebbian": if hebbian_injected.is_empty() { serde_json::json!(null) }
                   else { serde_json::json!({"injected": hebbian_injected, "processed_query": processed_query}) },
    }))
}
