//! Phrase generation prompt endpoints.

use axum::{
    http::StatusCode,
    routing::post,
    Json,
};

pub fn routes() -> axum::Router<crate::state::AppState> {
    axum::Router::new()
        .route("/api/phrase/prompt", post(build_phrase_prompt))
        .route("/api/phrase/parse", post(parse_phrase_response))
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

