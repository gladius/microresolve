//! Primary routing endpoints — concept system is primary, term index is fallback.

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
        .route("/api/route", post(route_query))
        .route("/api/route_multi", post(route_multi))
}

#[derive(serde::Deserialize)]
pub struct RouteRequest {
    pub query: String,
}

/// Simple single-best routing (backwards compat).
pub async fn route_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);

    // Try concept system first
    {
        let concepts = state.concepts.read().unwrap();
        if let Some(reg) = concepts.get(&app_id) {
            let results = reg.score_query(&req.query);
            if let Some((id, score)) = results.into_iter().next() {
                return Json(serde_json::json!([{"id": id, "score": (score * 100.0).round() / 100.0}]));
            }
        }
    }

    // Fallback: term index
    let routers = state.routers.read().unwrap();
    match routers.get(&app_id) {
        None => Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
        Some(router) => {
            let out: Vec<serde_json::Value> = router.route(&req.query).iter()
                .map(|r| serde_json::json!({"id": r.id, "score": (r.score * 100.0).round() / 100.0}))
                .collect();
            Json(serde_json::json!(out))
        }
    }
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

/// Multi-intent routing. Concept system primary, term index fallback.
/// Returns same JSON shape as before so UI works unchanged.
pub async fn route_multi(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteMultiRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let t0 = std::time::Instant::now();

    // ── Concept system (primary) ──────────────────────────────────────────────
    let concept_results: Option<Vec<(String, f32)>> = {
        let concepts = state.concepts.read().unwrap();
        concepts.get(&app_id).map(|reg| {
            reg.score_query_multi(&req.query, req.threshold, req.gap)
        })
    };

    let latency_us = t0.elapsed().as_micros() as u64;

    if let Some(scored) = concept_results {
        let max_score = scored.iter().map(|(_, s)| *s).fold(0f32, f32::max);

        let intents: Vec<serde_json::Value> = scored.iter().map(|(id, score)| {
            let confidence = if *score >= max_score * 0.8 { "high" }
                else if *score >= max_score * 0.5 { "medium" }
                else { "low" };
            serde_json::json!({
                "id": id,
                "score": (*score * 100.0).round() / 100.0,
                "confidence": confidence,
                "source": "concept",
                "position": 0,
                "span": [0, req.query.len()],
                "intent_type": "Action",
                "negated": false,
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
            source: "concept".to_string(),
        });

        emit_queued(&state, log_id, &req.query, &app_id, flag);

        return Json(serde_json::json!({
            "confirmed": confirmed,
            "candidates": candidates,
            "relations": [],
            "metadata": {},
            "routing_us": latency_us,
            "source": "concept",
        }));
    }

    // ── Fallback: term index ──────────────────────────────────────────────────
    let output = {
        let routers = state.routers.read().unwrap();
        match routers.get(&app_id) {
            None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
            Some(router) => router.route_multi(&req.query, req.threshold),
        }
    };
    let latency_us = t0.elapsed().as_micros() as u64;

    let intents: Vec<serde_json::Value> = output.intents.iter().map(|i| {
        serde_json::json!({
            "id": i.id,
            "score": (i.score * 100.0).round() / 100.0,
            "position": i.position,
            "span": [i.span.0, i.span.1],
            "intent_type": i.intent_type,
            "confidence": i.confidence,
            "source": "term_index",
            "negated": i.negated
        })
    }).collect();

    let confirmed: Vec<&serde_json::Value> = intents.iter()
        .filter(|i| i["confidence"].as_str() != Some("low"))
        .collect();
    let candidates: Vec<&serde_json::Value> = intents.iter()
        .filter(|i| i["confidence"].as_str() == Some("low"))
        .collect();

    let detected_ids: Vec<String> = output.intents.iter().map(|i| i.id.clone()).collect();
    let best_confidence = output.intents.iter()
        .map(|i| i.confidence.as_str())
        .min_by_key(|c| match *c { "high" => 0, "medium" => 1, "low" => 2, _ => 3 })
        .unwrap_or("none");
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
        source: "local".to_string(),
    });

    emit_queued(&state, log_id, &req.query, &app_id, flag);

    Json(serde_json::json!({
        "confirmed": confirmed,
        "candidates": candidates,
        "relations": output.relations.iter().map(|r| {
            use asv_router::IntentRelation;
            match r {
                IntentRelation::Parallel => serde_json::json!({"type": "Parallel"}),
                IntentRelation::Sequential { first, then } =>
                    serde_json::json!({"type": "Sequential", "first": first, "then": then}),
                IntentRelation::Conditional { primary, fallback } =>
                    serde_json::json!({"type": "Conditional", "primary": primary, "fallback": fallback}),
                IntentRelation::Reverse { stated_first, execute_first } =>
                    serde_json::json!({"type": "Reverse", "stated_first": stated_first, "execute_first": execute_first}),
                IntentRelation::Negation { do_this, not_this } =>
                    serde_json::json!({"type": "Negation", "do_this": do_this, "not_this": not_this}),
            }
        }).collect::<Vec<_>>(),
        "metadata": output.metadata,
        "routing_us": latency_us,
        "source": "term_index",
    }))
}
