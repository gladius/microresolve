//! Simulate and training arena endpoints.
//!
//! - `/api/simulate/turn` — generate one simulated customer message (LLM, with conversation history)
//! - `/api/simulate/respond` — generate agent response for a routed message
//! - `/api/training/generate` — batch-generate a full conversation (LLM)
//! - `/api/training/run` — route a batch of turns, measure accuracy (L0→L1→L2+L3 pipeline)
//! - `/api/training/review` — review a failure: delegates to `full_review` (Turn 1 skipped via GT set math)
//! - `/api/training/apply` — apply a `FullReviewResult`: delegates to `apply_review` (full L0–L3 pipeline)

use crate::pipeline::*;
use crate::state::*;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json,
};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/simulate/turn", post(simulate_turn))
        .route("/api/simulate/respond", post(simulate_respond))
        .route(
            "/api/simulate/history",
            get(sim_history_get).post(sim_history_save),
        )
        .route("/api/training/generate", post(training_generate))
        .route("/api/training/run", post(training_run))
        .route("/api/training/review", post(training_review))
        .route("/api/training/apply", post(training_apply))
}

#[derive(serde::Deserialize)]
pub struct SimulateTurnRequest {
    personality: String,             // e.g. "frustrated", "polite", "terse"
    sophistication: String,          // e.g. "low", "medium", "high"
    verbosity: String,               // e.g. "short", "medium", "long"
    history: Vec<serde_json::Value>, // previous turns [{role, message}]
    intents: Vec<String>,            // available intent IDs
    mode: String,                    // "normal" or "adversarial"
    #[serde(default)]
    language: String, // e.g. "English", "Spanish", "Chinese", "French", "German" — defaults to "English"
}

pub async fn simulate_turn(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SimulateTurnRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let intent_defs = {
        let h = state
            .engine
            .try_namespace(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
        let mut defs = Vec::new();
        let filter: std::collections::HashSet<&str> =
            req.intents.iter().map(|s| s.as_str()).collect();
        let mut ids = h.intent_ids();
        ids.sort();
        for id in &ids {
            if !filter.is_empty() && !filter.contains(id.as_str()) {
                continue;
            }
            let seeds = h.training(id).unwrap_or_default();
            defs.push(format!(
                "- {}: {}",
                id,
                seeds.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
            ));
        }
        defs.join("\n")
    };

    let history_text = if req.history.is_empty() {
        "This is the first message in the conversation.".to_string()
    } else {
        let turns: Vec<String> = req
            .history
            .iter()
            .map(|t| {
                format!(
                    "{}: {}",
                    t["role"].as_str().unwrap_or("?"),
                    t["message"].as_str().unwrap_or("")
                )
            })
            .collect();
        turns.join("\n")
    };

    let adversarial_instructions = if req.mode == "adversarial" {
        r#"
ADVERSARIAL MODE: Deliberately try to break the classification engine:
- Use unusual synonyms and slang the engine may not know
- Be vague and describe things indirectly instead of using exact terms
- Mix multiple intents in confusing ways
- Use negations ambiguously ("I don't NOT want a refund")
- Switch topics mid-sentence
- Use typos or informal spelling"#
    } else {
        ""
    };

    let language = if req.language.is_empty() {
        "English".to_string()
    } else {
        req.language.clone()
    };
    let prompt = format!(
        r#"You are simulating a customer interacting with a support system. Generate the next customer message.

## Your persona:
- Personality: {personality}
- Sophistication: {sophistication} (how technical/precise your language is)
- Verbosity: {verbosity}
- Language: {language} (write the customer message entirely in this language)
{adversarial}

## Available intents in the system:
{intents}

## Conversation so far:
{history}

## Instructions:
Generate a realistic customer message in {language}. You must also specify which intent(s) you are trying to express as ground truth.

{turn_guidance}

Return ONLY a JSON object:
{{
  "message": "the customer message text (in {language})",
  "ground_truth": ["intent_id_1", "intent_id_2"],
  "intent_description": "brief note on what the customer wants (in English)"
}}

Rules:
- ground_truth must use exact intent IDs from the list above
- Use 1-3 intents per message (multi-intent is encouraged)
- Stay in character for your persona throughout
- Write the message field entirely in {language}
- intent_description is always in English regardless of language
- If this is a follow-up turn, react naturally to the agent's previous response
- Return ONLY the JSON object"#,
        personality = req.personality,
        sophistication = req.sophistication,
        verbosity = req.verbosity,
        language = language,
        adversarial = adversarial_instructions,
        intents = intent_defs,
        history = history_text,
        turn_guidance = if req.history.is_empty() {
            "This is the opening message. Start a new conversation topic."
        } else {
            "Continue the conversation naturally. You may stick with the same topic, follow up, or pivot to a new request."
        },
    );

    let text = call_llm(&state, &prompt, 512).await?;

    let json_str = text
        .find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| {
            (
                StatusCode::BAD_GATEWAY,
                "No JSON in LLM response".to_string(),
            )
        })?;

    let val: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Invalid JSON from LLM: {}", e),
        )
    })?;

    Ok(Json(val))
}

#[derive(serde::Deserialize)]
pub struct SimulateRespondRequest {
    query: String,
    routed_intents: Vec<serde_json::Value>,
    history: Vec<serde_json::Value>,
}

pub async fn simulate_respond(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SimulateRespondRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let intent_defs = {
        let h = state
            .engine
            .try_namespace(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
        let mut defs = Vec::new();
        for intent in &req.routed_intents {
            let id = intent["id"].as_str().unwrap_or("");
            let intent_type = h
                .intent(id)
                .map(|i| i.intent_type)
                .unwrap_or(microresolve::IntentType::Action);
            defs.push(format!(
                "- {} ({:?}, score: {})",
                id,
                intent_type,
                intent["score"].as_f64().unwrap_or(0.0)
            ));
        }
        defs.join("\n")
    };

    let history_text = if req.history.is_empty() {
        String::new()
    } else {
        let turns: Vec<String> = req
            .history
            .iter()
            .map(|t| {
                format!(
                    "{}: {}",
                    t["role"].as_str().unwrap_or("?"),
                    t["message"].as_str().unwrap_or("")
                )
            })
            .collect();
        format!("\n## Previous conversation:\n{}", turns.join("\n"))
    };

    let prompt = format!(
        r#"You are a helpful customer support agent. Respond to the customer's message based on the routing results.

## Customer message:
"{query}"

## Routing detected these intents:
{intents}
{history}

## Instructions:
- Respond naturally and helpfully to ALL detected intents
- Keep your response concise (2-4 sentences)
- If multiple intents were detected, address each one
- Use a professional but friendly tone

Return ONLY a JSON object:
{{
  "message": "your response to the customer"
}}"#,
        query = req.query,
        intents = intent_defs,
        history = history_text,
    );

    let text = call_llm(&state, &prompt, 512).await?;

    let json_str = text
        .find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| {
            (
                StatusCode::BAD_GATEWAY,
                "No JSON in respond response".to_string(),
            )
        })?;

    let respond_val: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Invalid JSON from respond: {}", e),
        )
    })?;

    Ok(Json(respond_val))
}

// =============================================================================
// Training Arena endpoints
// =============================================================================

#[derive(serde::Deserialize)]
pub struct TrainingGenerateRequest {
    personality: String,
    sophistication: String,
    verbosity: String,
    turns: usize,
    scenario: Option<String>,
    #[serde(default)]
    language: String, // e.g. "English", "Spanish", "Chinese" — defaults to "English"
}

pub async fn training_generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainingGenerateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let intent_defs = {
        let h = state
            .engine
            .try_namespace(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
        let mut defs = Vec::new();
        let mut ids = h.intent_ids();
        ids.sort();
        for id in &ids {
            let seeds = h.training(id).unwrap_or_default();
            let intent_type = h
                .intent(id)
                .map(|i| i.intent_type)
                .unwrap_or(microresolve::IntentType::Action);
            defs.push(format!(
                "- {} ({:?}): {}",
                id,
                intent_type,
                seeds.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
            ));
        }
        defs.join("\n")
    };

    let scenario_section = if let Some(ref scenario) = req.scenario {
        format!("\n## Scenario description:\n{}\nGenerate a conversation that follows this scenario. The customer's messages should naturally express the intents described in the scenario.\n", scenario)
    } else {
        "\nGenerate a random customer support conversation. Pick different intents across turns to test variety.\n".to_string()
    };

    let language = if req.language.is_empty() {
        "English".to_string()
    } else {
        req.language.clone()
    };
    let prompt = format!(
        r#"You are generating a simulated customer support conversation for testing an intent classification engine.

## Customer persona:
- Personality: {personality}
- Sophistication: {sophistication} (how technical/precise their language is)
- Verbosity: {verbosity}
- Language: {language} (write ALL customer messages in this language)
{scenario}
## Available intents in the system:
{intents}

## Instructions:
Generate a {turns}-turn conversation in {language}. For each turn, provide the customer message, what intents they are expressing (ground truth), and a brief agent response.

Return ONLY a JSON object:
{{
  "turns": [
    {{
      "customer_message": "the customer's message (in {language})",
      "ground_truth": ["intent_id_1", "intent_id_2"],
      "intent_description": "brief note on what the customer wants (always in English)",
      "agent_response": "the agent's helpful response (2-3 sentences, in {language})"
    }}
  ]
}}

Rules:
- ground_truth must use exact intent IDs from the list above
- Use 1-3 intents per turn (multi-intent is encouraged)
- Stay in character for the persona throughout ALL turns
- Write customer_message and agent_response in {language}; intent_description always in English
- Each turn should build on or react to the previous agent response
- Make conversations realistic — customers don't always state things clearly
- Return ONLY the JSON object, no other text"#,
        personality = req.personality,
        sophistication = req.sophistication,
        verbosity = req.verbosity,
        language = language,
        scenario = scenario_section,
        intents = intent_defs,
        turns = req.turns,
    );

    let text = call_llm(&state, &prompt, 8192).await?;

    let json_str = text
        .find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| {
            (
                StatusCode::BAD_GATEWAY,
                "No JSON in generate response".to_string(),
            )
        })?;

    let gen_val: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Invalid JSON from generate: {}", e),
        )
    })?;

    Ok(Json(gen_val))
}

#[derive(serde::Deserialize)]
pub struct TrainingRunRequest {
    turns: Vec<TrainingTurn>,
}

#[derive(serde::Deserialize, Clone)]
pub struct TrainingTurn {
    message: String,
    ground_truth: Vec<String>,
}

pub async fn training_run(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainingRunRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut results = Vec::new();

    for turn in &req.turns {
        // Route via L0→L1→L2+L3 — same pipeline as production /api/route_multi.
        let scored = state
            .engine
            .try_namespace(&app_id)
            .map(|h| h.resolve(&turn.message))
            .unwrap_or_default();
        let max_score = scored.iter().map(|m| m.score).fold(0f32, f32::max);
        let confirmed: Vec<String> = scored
            .iter()
            .filter(|m| m.score >= max_score * 0.5)
            .map(|m| m.id.clone())
            .collect();
        let candidates: Vec<String> = scored
            .iter()
            .filter(|m| m.score < max_score * 0.5)
            .map(|m| m.id.clone())
            .collect();

        let ground_set: std::collections::HashSet<&str> =
            turn.ground_truth.iter().map(|s| s.as_str()).collect();
        let confirmed_set: std::collections::HashSet<&str> =
            confirmed.iter().map(|s| s.as_str()).collect();
        let candidate_set: std::collections::HashSet<&str> =
            candidates.iter().map(|s| s.as_str()).collect();

        // Pass/fail: confirmed matches ground truth
        let matched: Vec<&str> = turn
            .ground_truth
            .iter()
            .map(|s| s.as_str())
            .filter(|s| confirmed_set.contains(s))
            .collect();
        // Candidates that match GT — auto-promotable, not true misses
        let promotable: Vec<&str> = turn
            .ground_truth
            .iter()
            .map(|s| s.as_str())
            .filter(|s| !confirmed_set.contains(s) && candidate_set.contains(s))
            .collect();
        // True misses — not in confirmed OR candidates
        let missed: Vec<&str> = turn
            .ground_truth
            .iter()
            .map(|s| s.as_str())
            .filter(|s| !confirmed_set.contains(s) && !candidate_set.contains(s))
            .collect();
        let extra: Vec<&str> = confirmed
            .iter()
            .map(|s| s.as_str())
            .filter(|s| !ground_set.contains(s))
            .collect();

        // Pass = all GT in confirmed, no extras. Promotable candidates don't count as misses.
        let status = if missed.is_empty() && promotable.is_empty() && extra.is_empty() {
            "pass"
        } else if missed.is_empty() && extra.is_empty() {
            // All GT found (confirmed + candidates), just needs promotion
            "promotable"
        } else if !matched.is_empty() {
            "partial"
        } else {
            "fail"
        };

        results.push(serde_json::json!({
            "message": turn.message,
            "ground_truth": turn.ground_truth,
            "confirmed": confirmed,
            "candidates": candidates,
            "matched": matched,
            "promotable": promotable,
            "missed": missed,
            "extra": extra,
            "status": status,
            "details": scored.iter().map(|m| serde_json::json!({
                "id": &m.id,
                "score": (m.score * 100.0).round() / 100.0,
                "confidence": if m.score >= max_score * 0.8 { "high" } else if m.score >= max_score * 0.5 { "medium" } else { "low" },
                "source": "hebbian_l2",
                "negated": false,
            })).collect::<Vec<_>>(),
        }));
    }

    let pass_count = results.iter().filter(|r| r["status"] == "pass").count();
    let promotable_count = results
        .iter()
        .filter(|r| r["status"] == "promotable")
        .count();
    let detected_count = pass_count + promotable_count; // router found the right intents
    let total = results.len();
    Ok(Json(serde_json::json!({
        "results": results,
        "pass_count": detected_count,
        "confirmed_count": pass_count,
        "promotable_count": promotable_count,
        "total": total,
        "accuracy": if total == 0 { 0.0 } else { detected_count as f64 / total as f64 },
        "confirmed_rate": if total == 0 { 0.0 } else { pass_count as f64 / total as f64 },
    })))
}

/// Training review: delegates entirely to the unified auto-learn pipeline.
///
/// With ground truth available (known correct intents), Turn 1 LLM is skipped —
/// correct/missed/wrong sets are computed by set math. Turn 2 LLM then generates
/// phrases for any missed intents. Returns a `FullReviewResult` ready for `training_apply`.
#[derive(serde::Deserialize)]
pub struct TrainingReviewRequest {
    message: String,
    /// Intent IDs the engine returned (confirmed + false positives).
    #[serde(default)]
    detected: Vec<String>,
    ground_truth: Vec<String>,
}

pub async fn training_review(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainingReviewRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let result = full_review(
        &state,
        &app_id,
        &req.message,
        &req.detected,
        Some(&req.ground_truth),
    )
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(serde_json::to_value(&result).unwrap()))
}

/// Training apply: delegates entirely to the unified auto-learn pipeline.
///
/// Accepts the `FullReviewResult` from `training_review` plus the original query
/// (needed for L2 Hebbian context and L1 synonym reinforcement).
/// Runs phrase_pipeline → L2 update → L3 inhibition → L1 synonym/morphology.
#[derive(serde::Deserialize)]
pub struct TrainingApplyRequest {
    query: String,
    result: FullReviewResult,
}

pub async fn training_apply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainingApplyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    if !state.engine.has_namespace(&app_id) {
        return Err((StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)));
    }
    let applied = apply_review(&state, &app_id, &req.result, &req.query).await;
    Ok(Json(serde_json::json!({ "applied": applied })))
}

// ─── Simulation history ───────────────────────────────────────────────────────
// Persists as {data_dir}/{app_id}/simulations.json — at most 20 records.

pub async fn sim_history_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let runs = load_sim_history(&state, &app_id);
    Json(serde_json::json!({ "runs": runs }))
}

pub async fn sim_history_save(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(record): Json<serde_json::Value>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut runs = load_sim_history(&state, &app_id);
    runs.insert(0, record);
    runs.truncate(20);
    save_sim_history(&state, &app_id, &runs);
    StatusCode::OK
}

fn sim_history_path(state: &crate::state::AppState, app_id: &str) -> Option<std::path::PathBuf> {
    state.data_dir.as_ref().map(|d| {
        std::path::PathBuf::from(d)
            .join(app_id)
            .join("simulations.json")
    })
}

fn load_sim_history(state: &crate::state::AppState, app_id: &str) -> Vec<serde_json::Value> {
    let Some(path) = sim_history_path(state, app_id) else {
        return vec![];
    };
    let Ok(json) = std::fs::read_to_string(&path) else {
        return vec![];
    };
    serde_json::from_str::<Vec<serde_json::Value>>(&json).unwrap_or_default()
}

fn save_sim_history(state: &crate::state::AppState, app_id: &str, runs: &[serde_json::Value]) {
    let Some(path) = sim_history_path(state, app_id) else {
        return;
    };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(json) = serde_json::to_string(runs) {
        let _ = std::fs::write(path, json);
    }
}
