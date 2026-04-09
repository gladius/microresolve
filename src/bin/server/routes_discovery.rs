//! Discovery endpoints.

use axum::{
    extract::{State, Query},
    http::{StatusCode, HeaderMap},
    routing::{get, post, delete},
    Json,
};
use std::collections::HashMap;
use asv_router::{Router, IntentType};
use crate::state::*;
use crate::llm::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/discover", post(discover))
        .route("/api/discover/apply", post(discover_apply))
}


#[derive(serde::Deserialize)]
pub struct DiscoverRequest {
    queries: Vec<String>,
    #[serde(default)]
    expected_intents: usize,
}

pub async fn discover(
    State(state): State<AppState>,
    Json(req): Json<DiscoverRequest>,
) -> Json<serde_json::Value> {
    let config = asv_router::discovery::DiscoveryConfig {
        expected_intents: req.expected_intents,
        ..Default::default()
    };
    let clusters = asv_router::discovery::discover_intents(&req.queries, &config);

    // LLM naming: send representative queries to Claude for each cluster
    let mut clusters_json: Vec<serde_json::Value> = Vec::new();
    for c in &clusters {
        let samples: Vec<&str> = c.representative_queries.iter().take(5).map(|s| s.as_str()).collect();
        let mut name = c.suggested_name.clone();
        let mut description = String::new();

        if state.llm_key.is_some() {
            let prompt = format!(
                "These user queries all belong to the same intent category:\n{}\n\n\
                 Respond with ONLY a JSON object (no markdown, no explanation):\n\
                 {{\"name\": \"snake_case_intent_name\", \"description\": \"one sentence description of what the user wants\"}}",
                samples.iter().enumerate()
                    .map(|(i, q)| format!("{}. {}", i + 1, q))
                    .collect::<Vec<_>>().join("\n")
            );
            if let Ok(response) = call_llm(&state, &prompt, 100).await {
                // Parse the JSON response
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(extract_json(&response)) {
                    if let Some(n) = parsed["name"].as_str() {
                        name = n.to_string();
                    }
                    if let Some(d) = parsed["description"].as_str() {
                        description = d.to_string();
                    }
                }
            }
        }

        clusters_json.push(serde_json::json!({
            "suggested_name": name,
            "description": description,
            "top_terms": c.top_terms,
            "representative_queries": c.representative_queries,
            "size": c.size,
            "confidence": (c.confidence * 100.0).round() / 100.0,
        }));
    }

    let total: usize = clusters.iter().map(|c| c.size).sum();
    Json(serde_json::json!({
        "clusters": clusters_json,
        "total_clusters": clusters.len(),
        "total_assigned": total,
        "total_queries": req.queries.len(),
    }))
}

#[derive(serde::Deserialize)]
pub struct DiscoverApplyRequest {
    clusters: Vec<DiscoverApplyCluster>,
}

#[derive(serde::Deserialize)]
pub struct DiscoverApplyCluster {
    name: String,
    representative_queries: Vec<String>,
}

pub async fn discover_apply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DiscoverApplyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or((StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;

    let mut created = Vec::new();
    router.begin_batch();
    for cluster in &req.clusters {
        let seeds: Vec<&str> = cluster.representative_queries.iter().map(|s| s.as_str()).collect();
        router.add_intent(&cluster.name, &seeds);
        created.push(cluster.name.clone());
    }
    router.end_batch();

    maybe_persist(&state, &app_id, router);

    Ok(Json(serde_json::json!({
        "created": created,
        "count": created.len(),
    })))
}

// ============================================================================
