//! Server-side LLM review and seed generation.

use axum::{
    extract::{State, Query},
    http::{StatusCode, HeaderMap},
    routing::{get, post, delete},
    Json,
};
use std::collections::HashMap;
use asv_router::{Router, IntentType};
use crate::state::*;
use crate::routes_review_prompt::ReviewPromptRequest;
use crate::llm::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/review", post(review))
        .route("/api/seed/generate", post(generate_seeds))
}

pub async fn review(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReviewPromptRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let prompt = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
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

        format!(
r#"You are reviewing intent routing results from ASV Router, a model-free intent classification system.

## Current intents and their seed phrases:
{}

## Query:
"{}"

## ASV routing result (threshold: {}):

The router returns two tiers:
- **Confirmed** (high confidence, dual-source verified): the orchestrator will act on these directly
- **Candidates** (low confidence, routing-only): detected but not yet verified

Results:
{}

## Your task:
Analyze whether ASV's routing is correct. Consider:
- Confirmed intents are high-confidence — only flag as false positive if clearly wrong
- Candidates are low-confidence — they're correctly detected but need promotion via training
- If a candidate is correct for the query, suggest an add_seed to promote it

Return a JSON object with this exact structure:
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
      "action": "add_seed",
      "intent_id": "target intent",
      "seed": "short focused phrase (3-8 words) from the query relevant to this intent only",
      "reason": "why this helps"
    }}
  ],
  "confidence": "high" | "medium" | "low",
  "summary": "one sentence summary"
}}

{}

Rules:
- ONLY use action "add_seed". No learn or correct actions.
- Suggest 1-2 seeds per intent that fill the vocabulary gap — words in the query not covered by existing seeds
- Never use the full query as a seed
- Do NOT suggest seeds for intents already confirmed — they don't need it
- For candidates that are correct: suggest 1 seed phrase to promote them
- For true misses: suggest 1-2 seed phrases to teach the router
- Be conservative — only suggest changes you're confident about
- Return ONLY the JSON object, no other text"#,
            intent_defs.join("\n"),
            req.query,
            req.threshold,
            results_json,
            asv_router::seed::SEED_QUALITY_RULES,
        )
    };

    let text = call_llm(&state, &prompt, 1024).await?;

    // Extract JSON from response
    let json_str = text.find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No JSON in LLM response".to_string()))?;

    let review_val: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Invalid JSON from LLM: {}", e)))?;

    Ok(Json(review_val))
}

// --- Seed generation: server-side LLM call ---

#[derive(serde::Deserialize)]
pub struct GenerateSeedsRequest {
    intent_id: String,
    description: String,
    languages: Vec<String>,
}

pub async fn generate_seeds(
    State(state): State<AppState>,
    Json(req): Json<GenerateSeedsRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let prompt = asv_router::seed::build_prompt(&req.intent_id, &req.description, &req.languages);
    let text = call_llm(&state, &prompt, 2048).await?;
    let result = asv_router::seed::parse_response(&text, &req.languages)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Failed to parse seeds: {}", e)))?;
    let val: serde_json::Value = serde_json::from_str(&result)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(val))
}

