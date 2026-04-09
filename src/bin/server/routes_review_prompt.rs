//! Review prompt endpoint.

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
        .route("/api/review/prompt", post(build_review_prompt))
}

#[derive(serde::Deserialize)]
pub struct ReviewPromptRequest {
    pub query: String,
    pub results: Vec<serde_json::Value>,
    pub threshold: f32,
}

pub async fn build_review_prompt(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReviewPromptRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
    };

    // Build intent definitions for the prompt
    let mut intent_defs = Vec::new();
    let mut ids = router.intent_ids();
    ids.sort();
    for id in &ids {
        let seeds = router.get_training(id).unwrap_or_default();
        let intent_type = router.get_intent_type(id);
        intent_defs.push(format!(
            "- {} (type: {:?}): seeds: {:?}",
            id, intent_type,
            seeds.iter().take(5).cloned().collect::<Vec<_>>()
        ));
    }

    let results_json = serde_json::to_string_pretty(&req.results).unwrap_or_default();

    let prompt = format!(
r#"You are reviewing intent routing results from ASV Router, a model-free intent classification system.

## Current intents and their seed phrases:
{}

## Query:
"{}"

## ASV routing result (threshold: {}):
{}

## Your task:
Analyze whether ASV's routing is correct. Return a JSON object with this exact structure:
{{
  "correct": ["intent_id", ...],
  "false_positives": [
    {{"id": "intent_id", "reason": "why this is wrong"}}
  ],
  "missed": [
    {{"id": "intent_id", "reason": "why this should have matched"}}
  ],
  "suggestions": [
    {{
      "action": "learn" | "correct" | "add_seed",
      "query": "the query text",
      "intent_id": "target intent",
      "wrong_intent": "only for correct action",
      "seed": "only for add_seed action",
      "reason": "why this helps"
    }}
  ],
  "confidence": "high" | "medium" | "low",
  "summary": "one sentence summary"
}}

Rules:
- A score below 30% of the best score is likely a false positive
- Context-type intents with low scores may be valid context suggestions, not false positives
- If the routing is perfect, return empty arrays for false_positives, missed, and suggestions
- Be conservative with suggestions — only suggest changes you're confident about
- Return ONLY the JSON object, no other text"#,
        intent_defs.join("\n"),
        req.query,
        req.threshold,
        results_json,
    );

    Json(serde_json::json!({ "prompt": prompt }))
}

