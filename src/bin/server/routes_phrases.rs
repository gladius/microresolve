//! Phrase management endpoints.
//!
//! - `/api/phrase/prompt` — return the LLM prompt for a given intent (client calls LLM itself)
//! - `/api/phrase/parse` — parse an LLM phrase response server-side
//! - `/api/phrase/generate` — server-side LLM call: generate phrases for an intent

use axum::{
    extract::State,
    http::StatusCode,
    routing::post,
    Json,
};
use crate::state::AppState;
use crate::pipeline::call_llm;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/phrase/prompt",    post(build_phrase_prompt))
        .route("/api/phrase/parse",     post(parse_phrase_response))
        .route("/api/phrase/generate",  post(generate_phrases))
}

#[derive(serde::Deserialize)]
pub struct BuildPromptRequest {
    intent_id: String,
    description: String,
    languages: Vec<String>,
}

pub async fn build_phrase_prompt(Json(req): Json<BuildPromptRequest>) -> Json<serde_json::Value> {
    let prompt = asv_router::phrase::build_prompt(&req.intent_id, &req.description, &req.languages);
    Json(serde_json::json!({ "prompt": prompt }))
}

#[derive(serde::Deserialize)]
pub struct ParseResponseRequest {
    response_text: String,
    languages: Vec<String>,
}

pub async fn parse_phrase_response(
    Json(req): Json<ParseResponseRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let result = asv_router::phrase::parse_response(&req.response_text, &req.languages)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let val: serde_json::Value =
        serde_json::from_str(&result).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(val))
}

#[derive(serde::Deserialize)]
pub struct GeneratePhrasesRequest {
    intent_id: String,
    description: String,
    languages: Vec<String>,
}

/// Server-side LLM phrase generation — used by the Intents page to seed a new intent.
pub async fn generate_phrases(
    State(state): State<AppState>,
    Json(req): Json<GeneratePhrasesRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let prompt = asv_router::phrase::build_prompt(&req.intent_id, &req.description, &req.languages);
    let text = call_llm(&state, &prompt, 2048).await?;
    let result = asv_router::phrase::parse_response(&text, &req.languages)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Failed to parse phrases: {}", e)))?;
    let val: serde_json::Value = serde_json::from_str(&result)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(val))
}
