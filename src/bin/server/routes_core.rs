//! Routing endpoints.

use axum::{
    extract::{State, Query},
    http::{StatusCode, HeaderMap},
    routing::{get, post, delete},
    Json,
};
use std::collections::HashMap;
use asv_router::{Router, IntentType};
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

    // Record intent sequence (co-occurrence + temporal order + full sequence)
    if output.intents.len() > 1 {
        let ids: Vec<&str> = output.intents.iter().map(|i| i.id.as_str()).collect();
        if let Ok(mut routers) = state.routers.write() {
            if let Some(router) = routers.get_mut(&app_id) {
                router.record_intent_sequence(&ids);
                maybe_persist(&state, &app_id, router);
            }
        }
    }

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

    // Compute projected_context from co-occurrence
    let projected_context = {
        let routers = state.routers.read().unwrap();
        let router = match routers.get(&app_id) {
            Some(r) => r,
            None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
        };
        let co_pairs = router.get_co_occurrence();
        let matched_ids: std::collections::HashSet<&str> = output.intents.iter().map(|i| i.id.as_str()).collect();

        // For each matched action intent, find context intents that co-occur but aren't already in results
        let mut context_scores: HashMap<String, (u32, u32)> = HashMap::new(); // id -> (co_count, total_action_count)

        for intent in &output.intents {
            if intent.intent_type != asv_router::IntentType::Action {
                continue;
            }
            // Count total co-occurrences for this action (denominator for strength)
            let mut action_total: u32 = 0;
            for &(a, b, count) in &co_pairs {
                if a == intent.id || b == intent.id {
                    action_total += count;
                }
            }
            if action_total == 0 {
                continue;
            }
            // Find context partners not already in results
            for &(a, b, count) in &co_pairs {
                let partner = if a == intent.id { b } else if b == intent.id { a } else { continue };
                if matched_ids.contains(partner) {
                    continue; // already in results, don't project
                }
                if router.get_intent_type(partner) != asv_router::IntentType::Context {
                    continue; // only project context intents
                }
                let entry = context_scores.entry(partner.to_string()).or_insert((0, 0));
                entry.0 += count;
                entry.1 += action_total;
            }
        }

        let mut projected: Vec<serde_json::Value> = context_scores
            .into_iter()
            .map(|(id, (co_count, total))| {
                let strength = co_count as f64 / total as f64;
                serde_json::json!({
                    "id": id,
                    "co_occurrence": co_count,
                    "strength": (strength * 100.0).round() / 100.0
                })
            })
            .filter(|v| v["strength"].as_f64().unwrap_or(0.0) >= 0.1) // min 10% strength
            .collect();
        projected.sort_by(|a, b| {
            b["strength"].as_f64().unwrap_or(0.0)
                .partial_cmp(&a["strength"].as_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        projected
    };

    // Split into confirmed (high + medium confidence) and candidates (low confidence)
    let confirmed: Vec<&serde_json::Value> = intents.iter()
        .filter(|i| i["confidence"].as_str() != Some("low"))
        .collect();
    let candidates: Vec<&serde_json::Value> = intents.iter()
        .filter(|i| i["confidence"].as_str() == Some("low"))
        .collect();

    let suggestions: Vec<serde_json::Value> = output.suggestions.iter()
        .map(|s| serde_json::json!({
            "id": s.id,
            "probability": (s.probability * 100.0).round() / 100.0,
            "observations": s.observations,
            "because_of": s.because_of
        }))
        .collect();

    let result = serde_json::json!({
        "confirmed": confirmed,
        "candidates": candidates,
        "relations": relations,
        "metadata": output.metadata,
        "projected_context": projected_context,
        "suggestions": suggestions
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

