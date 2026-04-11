//! Routing endpoints.

use axum::{
    extract::{State, Query},
    http::{StatusCode, HeaderMap},
    routing::{get, post, delete},
    Json,
};
use asv_router::Router;
use crate::state::*;
use crate::log_store::{LogRecord};
use crate::llm::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/route", post(route_query))
        .route("/api/route_multi", post(route_multi))
}

#[derive(serde::Deserialize)]
pub struct RouteRequest {
    query: String,
}

pub async fn route_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
    };
    let results = router.route(&req.query);
    let out: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "id": r.id,
                "score": (r.score * 100.0).round() / 100.0
            })
        })
        .collect();
    Json(serde_json::json!(out))
}

#[derive(serde::Deserialize)]
pub struct RouteMultiRequest {
    query: String,
    #[serde(default = "default_threshold")]
    threshold: f32,
}

pub fn default_threshold() -> f32 {
    0.3
}

pub async fn route_multi(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteMultiRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let t0 = std::time::Instant::now();
    let output = {
        let routers = state.routers.read().unwrap();
        let router = match routers.get(&app_id) {
            Some(r) => r,
            None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
        };
        router.route_multi(&req.query, req.threshold)
    };
    let latency_us = t0.elapsed().as_micros() as u64;

    let intents: Vec<serde_json::Value> = output
        .intents
        .iter()
        .map(|i| {
            serde_json::json!({
                "id": i.id,
                "score": (i.score * 100.0).round() / 100.0,
                "position": i.position,
                "span": [i.span.0, i.span.1],
                "intent_type": i.intent_type,
                "confidence": i.confidence,
                "source": i.source,
                "negated": i.negated
            })
        })
        .collect();
    let relations: Vec<serde_json::Value> = output
        .relations
        .iter()
        .map(|r| {
            use asv_router::IntentRelation;
            match r {
                IntentRelation::Parallel => serde_json::json!({"type": "Parallel"}),
                IntentRelation::Sequential { first, then } => {
                    serde_json::json!({"type": "Sequential", "first": first, "then": then})
                }
                IntentRelation::Conditional { primary, fallback } => {
                    serde_json::json!({"type": "Conditional", "primary": primary, "fallback": fallback})
                }
                IntentRelation::Reverse {
                    stated_first,
                    execute_first,
                } => {
                    serde_json::json!({"type": "Reverse", "stated_first": stated_first, "execute_first": execute_first})
                }
                IntentRelation::Negation { do_this, not_this } => {
                    serde_json::json!({"type": "Negation", "do_this": do_this, "not_this": not_this})
                }
            }
        })
        .collect();

    // Split into confirmed (high + medium confidence) and candidates (low confidence)
    let confirmed: Vec<&serde_json::Value> = intents.iter()
        .filter(|i| i["confidence"].as_str() != Some("low"))
        .collect();
    let candidates: Vec<&serde_json::Value> = intents.iter()
        .filter(|i| i["confidence"].as_str() == Some("low"))
        .collect();

    let result = serde_json::json!({
        "confirmed": confirmed,
        "candidates": candidates,
        "relations": relations,
        "metadata": output.metadata,
        "routing_us": latency_us,
    });

    // Determine best confidence across all detected intents
    let best_confidence = output.intents.iter()
        .map(|i| i.confidence.as_str())
        .min_by_key(|c| match *c { "high" => 0, "medium" => 1, "low" => 2, _ => 3 })
        .unwrap_or("none");

    let detected_ids: Vec<String> = output.intents.iter().map(|i| i.id.clone()).collect();
    let flag = LogRecord::compute_flag(&detected_ids, best_confidence);

    log_query(&state, LogRecord {
        id: 0, // assigned by log_store
        query: req.query.clone(),
        app_id: app_id.clone(),
        detected_intents: detected_ids,
        confidence: best_confidence.to_string(),
        flag,
        session_id: None,
        timestamp_ms: now_ms(),
        router_version: {
            let routers = state.routers.read().unwrap();
            routers.get(&app_id).map(|r| r.version()).unwrap_or(0)
        },
        source: "local".to_string(),
    });

    Json(result)
}

