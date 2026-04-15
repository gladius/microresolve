//! Conversational Agent Builder — agentic loop endpoint.
//!
//! `POST /api/build` — user describes their business, LLM creates intents.
//! The LLM has tools: create_intent, update_intent, list_intents, test_query.
//! Server executes tool calls, feeds results back to LLM, returns final response.

use axum::{extract::State, http::HeaderMap, routing::post, Json};
use crate::state::*;
use crate::pipeline;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/build", post(build))
}

#[derive(serde::Deserialize)]
struct BuildRequest {
    message: String,
    #[serde(default)]
    history: Vec<serde_json::Value>,
}

const BUILDER_SYSTEM: &str = r#"You are an AI agent builder. You help users create intent-based agents through conversation.

You have these tools (call them by responding with JSON tool calls):

1. create_intent: Create a new intent with phrases and instructions.
   {"tool": "create_intent", "id": "intent_name", "phrases": ["phrase1", "phrase2", ...], "instructions": "what the agent should do", "guardrails": ["constraint1"], "persona": "tone description"}

2. update_intent: Update an existing intent's field.
   {"tool": "update_intent", "id": "intent_name", "field": "instructions|guardrails|persona|phrases", "value": "new value or array"}

3. list_intents: Show all current intents.
   {"tool": "list_intents"}

4. test_query: Test how a query would route.
   {"tool": "test_query", "query": "test message"}

Workflow:
1. Ask what the user's business does
2. Identify the main categories of customer requests → create intents with phrases
3. For each intent, ask how it should be handled → set instructions
4. Ask about constraints/rules → add guardrails
5. Ask about tone → set persona

Always create intents with at least 5 diverse phrases. Write instructions as clear paragraphs that tell an AI assistant exactly what to do, including conditional logic.

When you call a tool, respond with ONLY the JSON tool call. Do not mix tool calls with text.
When you want to respond to the user (no tool call), just respond normally."#;

const MAX_TOOL_ROUNDS: usize = 5;

pub async fn build(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<BuildRequest>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    // Ensure namespace exists
    if !state.routers.read().unwrap().contains_key(&app_id) {
        return Err((axum::http::StatusCode::NOT_FOUND,
            format!("namespace '{}' not found", app_id)));
    }

    // Build initial context: show existing intents so builder knows the state
    let existing_intents = {
        let routers = state.routers.read().unwrap();
        if let Some(router) = routers.get(&app_id) {
            let ids = router.intent_ids();
            if ids.is_empty() {
                "No intents created yet. This is a fresh workspace.".to_string()
            } else {
                let mut desc = format!("Current intents ({}):\n", ids.len());
                for id in &ids {
                    let d = router.get_description(id);
                    desc.push_str(&format!("  - {} {}\n", id,
                        if d.is_empty() { String::new() } else { format!("({})", d) }));
                }
                desc
            }
        } else {
            "Fresh workspace.".to_string()
        }
    };

    // Build messages: system + context + history + user message
    let mut messages = vec![
        serde_json::json!({"role": "system", "content": format!(
            "{}\n\nWorkspace state:\n{}", BUILDER_SYSTEM, existing_intents
        )}),
    ];
    for msg in &req.history {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
        messages.push(serde_json::json!({"role": role, "content": content}));
    }
    messages.push(serde_json::json!({"role": "user", "content": req.message}));

    // Agentic loop: LLM may call tools, we execute and feed back
    let mut actions: Vec<serde_json::Value> = Vec::new();

    for _round in 0..MAX_TOOL_ROUNDS {
        let response = pipeline::call_llm_with_messages(&state, &messages, 2048).await
            .map_err(|e| e)?;

        // Check if response is a tool call (JSON with "tool" field)
        if let Some(tool_call) = try_parse_tool_call(&response) {
            let tool_name = tool_call.get("tool").and_then(|v| v.as_str()).unwrap_or("");
            let result = execute_tool(&state, &app_id, &tool_call).await;

            actions.push(serde_json::json!({
                "tool": tool_name,
                "result": result,
            }));

            // Feed tool result back to LLM
            messages.push(serde_json::json!({"role": "assistant", "content": response}));
            messages.push(serde_json::json!({"role": "user", "content": format!(
                "[Tool result: {}]", result
            )}));

            continue; // LLM may call another tool
        }

        // No tool call — this is the final response to the user
        return Ok(Json(serde_json::json!({
            "response": response,
            "actions": actions,
        })));
    }

    // Max rounds reached
    Ok(Json(serde_json::json!({
        "response": "I've completed the setup. Let me know if you'd like to make any changes.",
        "actions": actions,
    })))
}

fn try_parse_tool_call(text: &str) -> Option<serde_json::Value> {
    let trimmed = text.trim();
    // Try to find JSON in the response
    let json_str = if trimmed.starts_with('{') {
        trimmed
    } else if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            &trimmed[start..=end]
        } else {
            return None;
        }
    } else {
        return None;
    };

    if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
        if val.get("tool").is_some() {
            return Some(val);
        }
    }
    None
}

async fn execute_tool(
    state: &AppState,
    app_id: &str,
    tool_call: &serde_json::Value,
) -> String {
    let tool = tool_call.get("tool").and_then(|v| v.as_str()).unwrap_or("");

    match tool {
        "create_intent" => {
            let id = tool_call.get("id").and_then(|v| v.as_str()).unwrap_or("unnamed");
            let phrases: Vec<String> = tool_call.get("phrases")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let instructions = tool_call.get("instructions").and_then(|v| v.as_str()).unwrap_or("");
            let guardrails: Vec<String> = tool_call.get("guardrails")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let persona = tool_call.get("persona").and_then(|v| v.as_str()).unwrap_or("");

            // Create intent with phrases
            {
                let mut routers = state.routers.write().unwrap();
                if let Some(router) = routers.get_mut(app_id) {
                    let mut by_lang = std::collections::HashMap::new();
                    by_lang.insert("en".to_string(), phrases.clone());
                    router.add_intent_multilingual(id, by_lang);
                    router.set_description(id, instructions);

                    // Set metadata per key
                    if !instructions.is_empty() {
                        router.set_metadata(id, "instructions", vec![instructions.to_string()]);
                    }
                    if !guardrails.is_empty() {
                        router.set_metadata(id, "guardrails", guardrails);
                    }
                    if !persona.is_empty() {
                        router.set_metadata(id, "persona", vec![persona.to_string()]);
                    }
                }
            }

            // Seed into L2
            let seeds: Vec<(String, String)> = phrases.iter()
                .map(|p| (id.to_string(), p.clone())).collect();
            crate::routes_import::seed_into_l2(state, app_id, &seeds);

            format!("Created intent '{}' with {} phrases.", id, phrases.len())
        }

        "update_intent" => {
            let id = tool_call.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let field = tool_call.get("field").and_then(|v| v.as_str()).unwrap_or("");
            let value = tool_call.get("value").unwrap_or(&serde_json::Value::Null);

            let mut routers = state.routers.write().unwrap();
            if let Some(router) = routers.get_mut(app_id) {
                let vals = match value {
                    serde_json::Value::String(s) => vec![s.clone()],
                    serde_json::Value::Array(arr) => arr.iter()
                        .filter_map(|v| v.as_str().map(String::from)).collect(),
                    _ => vec![],
                };
                if !vals.is_empty() {
                    router.set_metadata(id, field, vals);
                    format!("Updated '{}' field '{}' successfully.", id, field)
                } else {
                    format!("No valid value provided for '{}'.", field)
                }
            } else {
                "Namespace not found.".to_string()
            }
        }

        "list_intents" => {
            let routers = state.routers.read().unwrap();
            if let Some(router) = routers.get(app_id) {
                let ids = router.intent_ids();
                if ids.is_empty() {
                    "No intents created yet.".to_string()
                } else {
                    let mut result = format!("{} intents:\n", ids.len());
                    for id in &ids {
                        let desc = router.get_description(id);
                        let meta = router.get_metadata(id);
                        let has_instructions = meta.map(|m| m.contains_key("instructions")).unwrap_or(false);
                        let has_guardrails = meta.map(|m| m.contains_key("guardrails")).unwrap_or(false);
                        result.push_str(&format!("  - {} {}{}{}\n", id,
                            if desc.is_empty() { "" } else { desc },
                            if has_instructions { " [instructions ✓]" } else { " [needs instructions]" },
                            if has_guardrails { " [guardrails ✓]" } else { "" },
                        ));
                    }
                    result
                }
            } else {
                "Namespace not found.".to_string()
            }
        }

        "test_query" => {
            let query = tool_call.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let ig_map = state.intent_graph.read().unwrap();
            let heb_map = state.hebbian.read().unwrap();
            if let Some(ig) = ig_map.get(app_id) {
                let l1 = heb_map.get(app_id);
                let result = ig.route(l1, query, ig.default_threshold(), 3);
                if result.confirmed.is_empty() {
                    format!("Query '{}' → no match. May need more phrases.", query)
                } else {
                    let intents: Vec<String> = result.confirmed.iter()
                        .map(|(id, score)| format!("{} ({:.1})", id, score)).collect();
                    format!("Query '{}' → [{}] disposition={}", query, intents.join(", "), result.disposition)
                }
            } else {
                "No scoring model loaded yet.".to_string()
            }
        }

        _ => format!("Unknown tool: {}", tool),
    }
}
