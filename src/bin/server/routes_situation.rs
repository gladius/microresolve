//! Situation pattern generation endpoints.
//!
//! Two modes:
//! 1. Server-side LLM (when API key configured): POST /api/situation/generate — one click
//! 2. Manual copy-paste: POST /api/situation/prompt + POST /api/situation/parse

use axum::{
    extract::State,
    http::{StatusCode, HeaderMap},
    routing::post,
    Json,
};
use asv_router::phrase::{build_situation_prompt, parse_situation_response};
use crate::state::*;
use crate::llm::call_llm;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/situation/generate", post(generate))
        .route("/api/situation/prompt", post(build_prompt))
        .route("/api/situation/parse", post(parse_and_apply))
}

#[derive(serde::Deserialize)]
pub struct SituationGenerateRequest {
    intent_id: String,
    description: String,
    /// Current training phrases — the LLM uses these to know what NOT to generate.
    #[serde(default)]
    phrases: Vec<String>,
    /// Language codes to include CJK hints (e.g. ["en", "zh"]).
    #[serde(default)]
    languages: Vec<String>,
}

/// Full server-side LLM roundtrip: build prompt → call LLM → parse → apply patterns.
pub async fn generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SituationGenerateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    // Build phrases list from router if not supplied
    let phrases = if req.phrases.is_empty() {
        let routers = state.routers.read().unwrap();
        routers.get(&app_id)
            .and_then(|r| r.get_training(&req.intent_id))
            .unwrap_or_default()
    } else {
        req.phrases.clone()
    };

    let prompt = build_situation_prompt(&req.intent_id, &req.description, &phrases, &req.languages);
    let text = call_llm(&state, &prompt, 512).await?;
    let patterns = parse_situation_response(&text)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Failed to parse patterns: {}", e)))?;

    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
    if router.get_training(&req.intent_id).is_none() {
        return Err((StatusCode::NOT_FOUND, format!("intent '{}' not found", req.intent_id)));
    }
    let pairs: Vec<(&str, f32)> = patterns.iter().map(|p| (p.pattern.as_str(), p.weight)).collect();
    router.add_situation_patterns(&req.intent_id, &pairs);
    maybe_persist(&state, &app_id, router);

    Ok(Json(serde_json::json!({
        "applied": patterns.len(),
        "patterns": patterns,
    })))
}

pub async fn build_prompt(
    Json(req): Json<SituationGenerateRequest>,
) -> Json<serde_json::Value> {
    let prompt = build_situation_prompt(&req.intent_id, &req.description, &req.phrases, &req.languages);
    Json(serde_json::json!({ "prompt": prompt }))
}

#[derive(serde::Deserialize)]
pub struct ParseSituationRequest {
    intent_id: String,
    response_text: String,
}

/// Parse LLM response and bulk-apply the generated situation patterns to the intent.
pub async fn parse_and_apply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ParseSituationRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let patterns = parse_situation_response(&req.response_text)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;

    if router.get_training(&req.intent_id).is_none() {
        return Err((StatusCode::NOT_FOUND, format!("intent '{}' not found", req.intent_id)));
    }

    let count = patterns.len();
    let pairs: Vec<(&str, f32)> = patterns.iter().map(|p| (p.pattern.as_str(), p.weight)).collect();
    router.add_situation_patterns(&req.intent_id, &pairs);
    maybe_persist(&state, &app_id, router);

    Ok(Json(serde_json::json!({
        "applied": count,
        "patterns": patterns,
    })))
}
