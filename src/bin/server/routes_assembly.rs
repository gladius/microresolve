//! Intent Programming: route-and-assemble endpoint.
//!
//! Routes a query, loads matched intents' metadata payloads (instructions,
//! guardrails, schema, context), merges them, and returns a ready-to-use
//! system prompt fragment alongside the intent IDs.
//!
//! Metadata keys (stored per intent via /api/metadata):
//!   ip_instructions  — what the LLM should do when this intent is active
//!   ip_guardrails    — what the LLM must NOT do
//!   ip_schema        — required/optional fields ("required:field_name" or "optional:field_name")
//!   ip_context       — background knowledge the LLM needs
//!   ip_next_if       — conditional branching: "if X, activate intent_id"
//!   ip_execute       — tool execution template: "POST https://api.example.com/path"

use axum::{
    extract::State,
    http::{StatusCode, HeaderMap},
    routing::post,
    Json,
};
use crate::state::*;
use crate::pipeline::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/route/assemble", post(route_assemble))
        .route("/api/route/execute", post(route_execute))
}

#[derive(serde::Deserialize)]
pub struct RouteAssembleRequest {
    query: String,
    #[serde(default = "default_threshold")]
    threshold: f32,
    /// Optional conversation history for context
    #[serde(default)]
    history: Vec<serde_json::Value>,
}

fn default_threshold() -> f32 { 0.3 }

pub async fn route_assemble(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteAssembleRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    // Route via Hebbian L2
    let scored = {
        let ig_map = state.intent_graph.read().unwrap();
        let heb_map = state.hebbian.read().unwrap();
        if let (Some(ig), Some(heb)) = (ig_map.get(&app_id), heb_map.get(&app_id)) {
            let pre = heb.preprocess(&req.query);
            let (scores, _) = ig.score_multi_normalized(&pre.expanded, req.threshold, 1.5);
            scores
        } else {
            vec![]
        }
    };

    let max_score = scored.iter().map(|(_, s)| *s).fold(0f32, f32::max);
    let confirmed: Vec<String> = scored.iter()
        .filter(|(_, s)| *s >= max_score * 0.5)
        .map(|(id, _)| id.clone())
        .collect();
    let candidates: Vec<String> = scored.iter()
        .filter(|(_, s)| *s < max_score * 0.5)
        .map(|(id, _)| id.clone())
        .collect();
    let mut active_ids: Vec<String> = scored.iter().map(|(id, _)| id.clone()).collect();

    // Merge metadata payloads from all active intents
    let mut all_instructions: Vec<String> = Vec::new();
    let mut all_guardrails: Vec<String> = Vec::new();
    let mut required_fields: Vec<String> = Vec::new();
    let mut optional_fields: Vec<String> = Vec::new();
    let mut all_context: Vec<String> = Vec::new();
    let mut all_next_if: Vec<String> = Vec::new();
    let mut execute_template: Option<String> = None;
    let mut intent_descriptions: Vec<(String, String)> = Vec::new();

    // Extract intent metadata from the term-index registry (storage only).
    // Each intent's metadata, description, and situation patterns come from here.
    struct IntentData {
        desc: String,
        meta: std::collections::HashMap<String, Vec<String>>,
    }
    let all_intent_data: Vec<(String, IntentData)> = {
        let routers = state.routers.read().unwrap();
        if let Some(router) = routers.get(&app_id) {
            active_ids.iter().map(|id| {
                (id.clone(), IntentData {
                    desc: router.get_description(id).to_string(),
                    meta: router.get_metadata(id).cloned().unwrap_or_default(),
                })
            }).collect()
        } else {
            vec![]
        }
    };

    // ── __clarify__ fallback ──────────────────────────────────────────────────
    if active_ids.is_empty() {
        active_ids.push("__clarify__".to_string());
        all_instructions.push(
            "The user's request did not match any known workflow.".to_string()
        );
        all_instructions.push(
            "Greet them warmly, list what you can help with from the available capabilities, \
             and ask a clarifying question to understand their need.".to_string()
        );
        let routers = state.routers.read().unwrap();
        if let Some(router) = routers.get(&app_id) {
            let available = router.intent_ids();
            if !available.is_empty() {
                let cap_list = available.iter().take(20).cloned().collect::<Vec<_>>().join(", ");
                all_context.push(format!("Available capabilities: {}", cap_list));
            }
        }
    } else {
        for (id, data) in &all_intent_data {
            if !data.desc.is_empty() {
                intent_descriptions.push((id.clone(), data.desc.clone()));
            }
            if let Some(instructions) = data.meta.get("ip_instructions") {
                all_instructions.extend(instructions.clone());
            }
            if let Some(guardrails) = data.meta.get("ip_guardrails") {
                all_guardrails.extend(guardrails.clone());
            }
            if let Some(schema) = data.meta.get("ip_schema") {
                for field in schema {
                    if field.starts_with("required:") {
                        required_fields.push(field.trim_start_matches("required:").to_string());
                    } else if field.starts_with("optional:") {
                        optional_fields.push(field.trim_start_matches("optional:").to_string());
                    }
                }
            }
            if let Some(context) = data.meta.get("ip_context") {
                all_context.extend(context.clone());
            }
            if let Some(next_if) = data.meta.get("ip_next_if") {
                all_next_if.extend(next_if.clone());
            }
            if execute_template.is_none() {
                if let Some(exec) = data.meta.get("ip_execute") {
                    if let Some(first) = exec.first() {
                        execute_template = Some(first.clone());
                    }
                }
            }
        }
    }

    // Build assembled system prompt fragment
    let assembled_prompt = build_system_prompt(
        &active_ids,
        &intent_descriptions,
        &all_instructions,
        &all_guardrails,
        &required_fields,
        &optional_fields,
        &all_context,
        &all_next_if,
        &req.history,
    );

    let routing_details: Vec<serde_json::Value> = scored.iter().map(|(id, score)| {
        serde_json::json!({
            "id": id,
            "score": (*score * 100.0).round() / 100.0,
            "source": "hebbian_l2",
        })
    }).collect();

    Ok(Json(serde_json::json!({
        "query": req.query,
        "confirmed": confirmed,
        "candidates": candidates,
        "routing": routing_details,
        "payload": {
            "instructions": all_instructions,
            "guardrails": all_guardrails,
            "schema": {
                "required": required_fields,
                "optional": optional_fields,
            },
            "context": all_context,
            "next_if": all_next_if,
            "execute": execute_template,
        },
        "system_prompt": assembled_prompt,
        "has_payload": !all_instructions.is_empty() || !all_guardrails.is_empty(),
        "is_clarify": active_ids == vec!["__clarify__"],
    })))
}

fn build_system_prompt(
    intent_ids: &[String],
    descriptions: &[(String, String)],
    instructions: &[String],
    guardrails: &[String],
    required: &[String],
    optional: &[String],
    context: &[String],
    next_if: &[String],
    history: &[serde_json::Value],
) -> String {
    let mut parts: Vec<String> = Vec::new();

    parts.push("You are a helpful assistant handling a customer request.".to_string());

    // Active intents
    let non_clarify: Vec<_> = intent_ids.iter().filter(|id| *id != "__clarify__").collect();
    if !non_clarify.is_empty() {
        parts.push(format!("\n## Active intents:\n{}",
            non_clarify.iter().map(|id| format!("- {}", id)).collect::<Vec<_>>().join("\n")
        ));
    }

    // Descriptions
    if !descriptions.is_empty() {
        parts.push(format!("\n## What these intents mean:\n{}",
            descriptions.iter()
                .map(|(id, desc)| format!("- {}: {}", id, desc.lines().next().unwrap_or(desc)))
                .collect::<Vec<_>>().join("\n")
        ));
    }

    // Background context
    if !context.is_empty() {
        parts.push(format!("\n## Context:\n{}", context.join("\n")));
    }

    // Instructions
    if !instructions.is_empty() {
        parts.push(format!("\n## Instructions:\n{}",
            instructions.iter().map(|i| format!("- {}", i)).collect::<Vec<_>>().join("\n")
        ));
    }

    // Schema
    if !required.is_empty() || !optional.is_empty() {
        let mut schema_lines = Vec::new();
        if !required.is_empty() {
            schema_lines.push(format!("Required: {}", required.join(", ")));
        }
        if !optional.is_empty() {
            schema_lines.push(format!("Optional: {}", optional.join(", ")));
        }
        parts.push(format!(
            "\n## Data needed:\n{}\nIf required fields are missing, ask for them before proceeding.",
            schema_lines.join("\n")
        ));
    }

    // Guardrails
    if !guardrails.is_empty() {
        parts.push(format!("\n## Rules (strictly follow):\n{}",
            guardrails.iter().map(|g| format!("- {}", g)).collect::<Vec<_>>().join("\n")
        ));
    }

    // Conditional branching
    if !next_if.is_empty() {
        parts.push(format!("\n## Branching rules:\n{}",
            next_if.iter().map(|r| format!("- {}", r)).collect::<Vec<_>>().join("\n")
        ));
        parts.push(
            "\nWhen a branching condition applies, output [ACTIVATE: intent_id] on its own line \
             before your response.".to_string()
        );
    }

    // Execution readiness
    parts.push(
        "\nWhen you have collected all required fields and are ready to execute, output \
         [EXECUTE: {\"field\": \"value\", ...}] on its own line before confirming to the user."
        .to_string()
    );

    // Conversation history summary
    if !history.is_empty() {
        let turns: Vec<String> = history.iter().map(|t| {
            format!("{}: {}",
                t["role"].as_str().unwrap_or("?"),
                t["message"].as_str().unwrap_or(""))
        }).collect();
        parts.push(format!("\n## Conversation so far:\n{}", turns.join("\n")));
    }

    parts.push("\nRespond naturally and helpfully. Follow all instructions and rules above.".to_string());

    parts.join("")
}

// ---------------------------------------------------------------------------
// route_execute: route + assemble + LLM call in one request
//
// Features:
//   - Reflective routing: routes on [last assistant turn window + user query]
//     so follow-up turns (no keywords) still find the right intent
//   - __clarify__ fallback: if nothing routes, graceful recovery with intent list
//   - ip_next_if: conditional branching rules passed to LLM
//   - ip_execute: when LLM signals [EXECUTE: {...}], calls the tool URL and
//     injects the result back into the conversation
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct RouteExecuteRequest {
    query: String,
    #[serde(default = "default_threshold")]
    threshold: f32,
    #[serde(default)]
    history: Vec<serde_json::Value>,
}

pub async fn route_execute(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteExecuteRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {

    // ── Reflective routing ────────────────────────────────────────────────────
    // Append the last 150 chars of the previous assistant turn to the routing
    // query. The LLM's response vocabulary carries the intent forward — no
    // session state needed.
    let routing_query = {
        let last_assistant = req.history.iter().rev()
            .find(|t| t.get("role").and_then(|r| r.as_str()) == Some("assistant"))
            .and_then(|t| t.get("message").and_then(|m| m.as_str()))
            .unwrap_or("");
        if last_assistant.is_empty() {
            req.query.clone()
        } else {
            let chars: Vec<char> = last_assistant.chars().collect();
            let start = chars.len().saturating_sub(150);
            let window: String = chars[start..].iter().collect();
            format!("{} {}", window, req.query)
        }
    };

    // ── Assemble ──────────────────────────────────────────────────────────────
    let assemble_req = RouteAssembleRequest {
        query: routing_query,
        threshold: req.threshold,
        history: req.history.clone(),
    };
    let assembled = route_assemble(
        State(state.clone()),
        headers.clone(),
        Json(assemble_req),
    ).await?.0;

    let system_prompt = assembled["system_prompt"].as_str().unwrap_or("").to_string();
    let execute_url = assembled["payload"]["execute"].as_str().map(|s| s.to_string());
    let is_clarify = assembled["is_clarify"].as_bool().unwrap_or(false);

    // ── LLM call ──────────────────────────────────────────────────────────────
    let llm_prompt = format!(
        "{}\n\n---\nUser message: {}",
        system_prompt,
        req.query
    );
    let mut response = call_llm(&state, &llm_prompt, 512).await?;

    // ── ip_execute: tool execution ────────────────────────────────────────────
    // If LLM signals [EXECUTE: {...}] and we have an execute URL, call the tool
    // and inject the result back so LLM can respond with the outcome.
    let mut tool_result: Option<String> = None;

    if let Some(ref url) = execute_url {
        if let Some(exec_start) = response.find("[EXECUTE:") {
            if let Some(exec_end) = response[exec_start..].find(']') {
                let raw = response[exec_start + 9..exec_start + exec_end].trim().to_string();
                if let Ok(fields) = serde_json::from_str::<serde_json::Value>(&raw) {
                    // Parse method and URL from template: "POST https://..."
                    let parts: Vec<&str> = url.splitn(2, ' ').collect();
                    let (method, target_url) = if parts.len() == 2 {
                        (parts[0], parts[1])
                    } else {
                        ("POST", url.as_str())
                    };

                    let http_result = match method.to_uppercase().as_str() {
                        "GET" => state.http.get(target_url).send().await,
                        _ => state.http.post(target_url).json(&fields).send().await,
                    };

                    match http_result {
                        Ok(resp) => {
                            let status = resp.status().as_u16();
                            let body = resp.text().await.unwrap_or_default();
                            tool_result = Some(format!("status={} body={}", status, body));
                        }
                        Err(e) => {
                            tool_result = Some(format!("error={}", e));
                        }
                    }

                    // Re-call LLM with tool result injected
                    if let Some(ref result_text) = tool_result {
                        let result_prompt = format!(
                            "{}\n\n---\nTool execution result:\n{}\n\nNow respond to the user confirming the outcome.",
                            system_prompt, result_text
                        );
                        if let Ok(final_response) = call_llm(&state, &result_prompt, 512).await {
                            response = final_response;
                        }
                    }
                }
            }
        }
    }

    // ── ip_next_if: activation signal ─────────────────────────────────────────
    // If LLM outputs [ACTIVATE: intent_id], extract and return it so client
    // can track intent transitions.
    let activated_intent: Option<String> = {
        if let Some(act_start) = response.find("[ACTIVATE:") {
            if let Some(act_end) = response[act_start..].find(']') {
                let intent_id = response[act_start + 10..act_start + act_end].trim().to_string();
                Some(intent_id)
            } else { None }
        } else { None }
    };

    Ok(Json(serde_json::json!({
        "query": req.query,
        "confirmed": assembled["confirmed"],
        "candidates": assembled["candidates"],
        "has_payload": assembled["has_payload"],
        "is_clarify": is_clarify,
        "payload": assembled["payload"],
        "response": response,
        "system_prompt": system_prompt,
        "tool_result": tool_result,
        "activated_intent": activated_intent,
    })))
}
