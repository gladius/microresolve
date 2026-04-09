//! Seed generation prompt endpoints.

use axum::{
    extract::{State, Query},
    http::{StatusCode, HeaderMap},
    routing::{get, post, delete},
    Json,
};
use std::collections::HashMap;
use asv_router::{Router, IntentType};
use crate::state::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/seed/prompt", post(build_seed_prompt))
        .route("/api/seed/parse", post(parse_seed_response))
}

#[derive(serde::Deserialize)]
pub struct BuildPromptRequest {
    intent_id: String,
    description: String,
    languages: Vec<String>,
}

pub async fn build_seed_prompt(Json(req): Json<BuildPromptRequest>) -> Json<serde_json::Value> {
    let prompt = asv_router::seed::build_prompt(&req.intent_id, &req.description, &req.languages);
    Json(serde_json::json!({ "prompt": prompt }))
}

#[derive(serde::Deserialize)]
pub struct ParseResponseRequest {
    response_text: String,
    languages: Vec<String>,
}

pub async fn parse_seed_response(
    Json(req): Json<ParseResponseRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let result = asv_router::seed::parse_response(&req.response_text, &req.languages)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let val: serde_json::Value =
        serde_json::from_str(&result).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(val))
}

