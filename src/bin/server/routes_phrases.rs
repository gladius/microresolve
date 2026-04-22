//! Phrase generation endpoint.
//!
//! - `/api/phrase/generate` — server-side LLM call: generate phrases for an intent
//!
//! Earlier client-driven variants `/api/phrase/prompt` and `/api/phrase/parse`
//! were removed in the dead-endpoint sweep — no UI or external caller used them.

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
        .route("/api/phrase/generate",  post(generate_phrases))
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
    let prompt = microresolve::phrase::build_prompt(&req.intent_id, &req.description, &req.languages);
    let text = call_llm(&state, &prompt, 2048).await?;
    let result = microresolve::phrase::parse_response(&text, &req.languages)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Failed to parse phrases: {}", e)))?;
    let val: serde_json::Value = serde_json::from_str(&result)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(val))
}
