//! Training and simulation endpoints.

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
        .route("/api/simulate/turn", post(simulate_turn))
        .route("/api/simulate/respond", post(simulate_respond))
        .route("/api/training/generate", post(training_generate))
        .route("/api/training/run", post(training_run))
        .route("/api/training/review", post(training_review))
        .route("/api/training/apply", post(training_apply))
}

#[derive(serde::Deserialize)]
pub struct SimulateTurnRequest {
    personality: String,     // e.g. "frustrated", "polite", "terse"
    sophistication: String,  // e.g. "low", "medium", "high"
    verbosity: String,       // e.g. "short", "medium", "long"
    history: Vec<serde_json::Value>, // previous turns [{role, message}]
    intents: Vec<String>,    // available intent IDs
    mode: String,            // "normal" or "adversarial"
}

pub async fn simulate_turn(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SimulateTurnRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let intent_defs = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
        let mut defs = Vec::new();
        // Use client-provided intent list to scope simulation
        let filter: std::collections::HashSet<&str> = req.intents.iter().map(|s| s.as_str()).collect();
        let mut ids = router.intent_ids();
        ids.sort();
        for id in &ids {
            if !filter.is_empty() && !filter.contains(id.as_str()) { continue; }
            let seeds = router.get_training(id).unwrap_or_default();
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
        let turns: Vec<String> = req.history.iter().map(|t| {
            format!("{}: {}", t["role"].as_str().unwrap_or("?"), t["message"].as_str().unwrap_or(""))
        }).collect();
        turns.join("\n")
    };

    let adversarial_instructions = if req.mode == "adversarial" {
        r#"
ADVERSARIAL MODE: Deliberately try to break the routing system:
- Use unusual synonyms and slang the router may not know
- Be vague and describe things indirectly instead of using exact terms
- Mix multiple intents in confusing ways
- Use negations ambiguously ("I don't NOT want a refund")
- Switch topics mid-sentence
- Use typos or informal spelling"#
    } else {
        ""
    };

    let prompt = format!(
r#"You are simulating a customer interacting with a support system. Generate the next customer message.

## Your persona:
- Personality: {personality}
- Sophistication: {sophistication} (how technical/precise your language is)
- Verbosity: {verbosity}
{adversarial}

## Available intents in the system:
{intents}

## Conversation so far:
{history}

## Instructions:
Generate a realistic customer message. You must also specify which intent(s) you are trying to express as ground truth.

{turn_guidance}

Return ONLY a JSON object:
{{
  "message": "the customer message text",
  "ground_truth": ["intent_id_1", "intent_id_2"],
  "intent_description": "brief note on what the customer wants"
}}

Rules:
- ground_truth must use exact intent IDs from the list above
- Use 1-3 intents per message (multi-intent is encouraged)
- Stay in character for your persona throughout
- If this is a follow-up turn, react naturally to the agent's previous response
- Return ONLY the JSON object"#,
        personality = req.personality,
        sophistication = req.sophistication,
        verbosity = req.verbosity,
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

    let json_str = text.find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No JSON in LLM response".to_string()))?;

    let val: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Invalid JSON from LLM: {}", e)))?;

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
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
        let mut defs = Vec::new();
        for intent in &req.routed_intents {
            let id = intent["id"].as_str().unwrap_or("");
            let intent_type = router.get_intent_type(id);
            defs.push(format!("- {} ({:?}, score: {})", id, intent_type,
                intent["score"].as_f64().unwrap_or(0.0)));
        }
        defs.join("\n")
    };

    let history_text = if req.history.is_empty() {
        String::new()
    } else {
        let turns: Vec<String> = req.history.iter().map(|t| {
            format!("{}: {}", t["role"].as_str().unwrap_or("?"), t["message"].as_str().unwrap_or(""))
        }).collect();
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

    let json_str = text.find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No JSON in respond response".to_string()))?;

    let respond_val: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Invalid JSON from respond: {}", e)))?;

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
}

pub async fn training_generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainingGenerateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let intent_defs = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
        let mut defs = Vec::new();
        let mut ids = router.intent_ids();
        ids.sort();
        for id in &ids {
            let seeds = router.get_training(id).unwrap_or_default();
            let intent_type = router.get_intent_type(id);
            defs.push(format!(
                "- {} ({:?}): {}",
                id, intent_type,
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

    let prompt = format!(
r#"You are generating a simulated customer support conversation for testing an intent routing system.

## Customer persona:
- Personality: {personality}
- Sophistication: {sophistication} (how technical/precise their language is)
- Verbosity: {verbosity}
{scenario}
## Available intents in the system:
{intents}

## Instructions:
Generate a {turns}-turn conversation. For each turn, provide the customer message, what intents they are expressing (ground truth), and a brief agent response.

Return ONLY a JSON object:
{{
  "turns": [
    {{
      "customer_message": "the customer's message",
      "ground_truth": ["intent_id_1", "intent_id_2"],
      "intent_description": "brief note on what the customer wants",
      "agent_response": "the agent's helpful response (2-3 sentences)"
    }}
  ]
}}

Rules:
- ground_truth must use exact intent IDs from the list above
- Use 1-3 intents per turn (multi-intent is encouraged)
- Stay in character for the persona throughout ALL turns
- Each turn should build on or react to the previous agent response
- Make conversations realistic — customers don't always state things clearly
- Return ONLY the JSON object, no other text"#,
        personality = req.personality,
        sophistication = req.sophistication,
        verbosity = req.verbosity,
        scenario = scenario_section,
        intents = intent_defs,
        turns = req.turns,
    );

    let text = call_llm(&state, &prompt, 8192).await?;

    let json_str = text.find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No JSON in generate response".to_string()))?;

    let gen_val: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Invalid JSON from generate: {}", e)))?;

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
    let routers = state.routers.read().unwrap();
    let router = routers.get(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
    let mut results = Vec::new();

    for turn in &req.turns {
        let output = router.route_multi(&turn.message, 0.3);

        // Confirmed = high + medium confidence (paraphrase-confirmed)
        // Candidates = low confidence (routing-only, needs LLM verification)
        let confirmed: Vec<String> = output.intents.iter()
            .filter(|i| i.confidence != "low")
            .map(|i| i.id.clone())
            .collect();
        let candidates: Vec<String> = output.intents.iter()
            .filter(|i| i.confidence == "low")
            .map(|i| i.id.clone())
            .collect();

        let ground_set: std::collections::HashSet<&str> = turn.ground_truth.iter().map(|s| s.as_str()).collect();
        let confirmed_set: std::collections::HashSet<&str> = confirmed.iter().map(|s| s.as_str()).collect();
        let candidate_set: std::collections::HashSet<&str> = candidates.iter().map(|s| s.as_str()).collect();

        // Pass/fail: confirmed matches ground truth
        let matched: Vec<&str> = turn.ground_truth.iter().map(|s| s.as_str()).filter(|s| confirmed_set.contains(s)).collect();
        // Candidates that match GT — auto-promotable, not true misses
        let promotable: Vec<&str> = turn.ground_truth.iter().map(|s| s.as_str()).filter(|s| !confirmed_set.contains(s) && candidate_set.contains(s)).collect();
        // True misses — not in confirmed OR candidates
        let missed: Vec<&str> = turn.ground_truth.iter().map(|s| s.as_str()).filter(|s| !confirmed_set.contains(s) && !candidate_set.contains(s)).collect();
        let extra: Vec<&str> = confirmed.iter().map(|s| s.as_str()).filter(|s| !ground_set.contains(s)).collect();

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
            "details": output.intents.iter().map(|i| serde_json::json!({
                "id": i.id,
                "score": (i.score * 100.0).round() / 100.0,
                "confidence": i.confidence,
                "source": i.source,
                "negated": i.negated,
            })).collect::<Vec<_>>(),
        }));
    }

    let pass_count = results.iter().filter(|r| r["status"] == "pass").count();
    let promotable_count = results.iter().filter(|r| r["status"] == "promotable").count();
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

#[derive(serde::Deserialize)]
pub struct TrainingReviewRequest {
    message: String,
    detected: Vec<serde_json::Value>,
    ground_truth: Vec<String>,
}

pub async fn training_review(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainingReviewRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let intent_seeds = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
        let mut relevant_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for gt in &req.ground_truth {
            relevant_ids.insert(gt.clone());
        }
        for d in &req.detected {
            if let Some(id) = d["id"].as_str() {
                relevant_ids.insert(id.to_string());
            }
        }
        let mut defs = Vec::new();
        for id in &relevant_ids {
            let seeds = router.get_training(id).unwrap_or_default();
            defs.push(format!(
                "- {}: {}",
                id,
                seeds.iter().take(5).cloned().collect::<Vec<_>>().join(" | ")
            ));
        }
        defs.join("\n")
    };

    let detected_str: Vec<String> = req.detected.iter().map(|d| {
        format!("{} (score: {})", d["id"].as_str().unwrap_or("?"), d["score"].as_f64().unwrap_or(0.0))
    }).collect();

    let prompt = format!(
r#"You are reviewing a failed intent routing result from ASV Router, a keyword-based intent classifier.

## Customer message:
"{message}"

## Ground truth intents (what the customer actually wants):
{ground_truth}

## What the router detected:
{detected}

## Relevant intent seeds (existing phrases the router knows):
{seeds}

## Your task:
For each MISSED intent (in ground truth but not detected), generate a NEW seed phrase that captures the GENERAL PATTERN of what the customer is asking for. Do NOT copy phrases from the customer message directly — create clean, reusable pattern phrases.

Seed quality guidelines:
{quality}

CRITICAL RULES:
- ONLY use action "add_seed". No other action types.
- NEVER use the customer's exact words as a seed. Create a generalized pattern.
- Do NOT suggest corrections for false positives (extra detected intents). Ignore them.
- Only suggest seeds for MISSED intents, not for intents already detected.
- Use exact intent IDs from the lists above.

Example: If the message is "I got the wrong item and I want my money back and someone needs to call me"
and missed intents are [refund, contact_human]:
- add_seed "get my money back for wrong item" → refund
- add_seed "need someone to call me" → contact_human

Return ONLY a JSON object:
{{
  "analysis": "brief explanation of what was missed and why",
  "corrections": [
    {{
      "action": "add_seed",
      "phrase": "short focused phrase from the message",
      "intent": "missed_intent_id"
    }}
  ]
}}"#,
        message = req.message,
        ground_truth = req.ground_truth.join(", "),
        detected = if detected_str.is_empty() { "nothing detected".to_string() } else { detected_str.join(", ") },
        seeds = intent_seeds,
        quality = asv_router::phrase::PHRASE_QUALITY_RULES,
    );

    let text = call_llm(&state, &prompt, 1024).await?;

    let json_str = text.find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No JSON in review response".to_string()))?;

    let review_result: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Invalid JSON from review: {}", e)))?;

    Ok(Json(review_result))
}

#[derive(serde::Deserialize)]
pub struct TrainingApplyRequest {
    corrections: Vec<serde_json::Value>,
}

pub async fn training_apply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainingApplyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
    router.begin_batch();

    let mut applied = 0;
    let mut errors = Vec::new();

    for correction in &req.corrections {
        let action = correction["action"].as_str().unwrap_or("");
        match action {
            "add_seed" => {
                let phrase = correction["phrase"].as_str().unwrap_or("");
                let intent = correction["intent"].as_str().unwrap_or("");
                if !phrase.is_empty() && !intent.is_empty() {
                    router.learn(phrase, intent);
                    applied += 1;
                } else {
                    errors.push("add_seed: missing phrase or intent".to_string());
                }
            }
            _ => {
                errors.push(format!("ignored action: {} (only add_seed allowed)", action));
            }
        }
    }

    router.end_batch();
    maybe_persist(&state, &app_id, router);

    Ok(Json(serde_json::json!({
        "applied": applied,
        "errors": errors,
    })))
}

