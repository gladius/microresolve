//! Spec import endpoints (OpenAPI, Postman).

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
        .route("/api/import/parse", post(import_parse))
        .route("/api/import/apply", post(import_apply))
        .route("/api/import/mcp/search", get(mcp_search))
        .route("/api/import/mcp/fetch", get(mcp_fetch))
        .route("/api/import/mcp/parse", post(mcp_parse))
        .route("/api/import/mcp/apply", post(mcp_apply))
}

// ============================================================
// MCP Discovery via Smithery Registry
// ============================================================

#[derive(serde::Deserialize)]
pub struct McpSearchParams {
    q: String,
    #[serde(default = "default_page_size")]
    limit: usize,
}

pub fn default_page_size() -> usize { 20 }

/// Search MCP servers on Smithery registry.
pub async fn mcp_search(
    State(state): State<AppState>,
    Query(params): Query<McpSearchParams>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let url = format!(
        "https://api.smithery.ai/servers?q={}&pageSize={}",
        urlencoding(&params.q), params.limit
    );

    let resp = state.http.get(&url)
        .send().await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Smithery fetch failed: {}", e)))?;

    if !resp.status().is_success() {
        return Err((StatusCode::BAD_GATEWAY, format!("Smithery returned {}", resp.status())));
    }

    let data: serde_json::Value = resp.json().await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Smithery parse failed: {}", e)))?;

    // Extract servers with relevant info
    let servers = data.get("servers").or(data.as_array().map(|_| &data));

    Ok(Json(data))
}

/// Fetch full tool definitions for a specific MCP server from Smithery.
#[derive(serde::Deserialize)]
pub struct McpFetchParams {
    name: String,
}

pub async fn mcp_fetch(
    State(state): State<AppState>,
    Query(params): Query<McpFetchParams>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let url = format!("https://api.smithery.ai/servers/{}", urlencoding(&params.name));

    let resp = state.http.get(&url)
        .send().await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Smithery fetch failed: {}", e)))?;

    if !resp.status().is_success() {
        return Err((StatusCode::BAD_GATEWAY, format!("Smithery returned {} for '{}'", resp.status(), params.name)));
    }

    let data: serde_json::Value = resp.json().await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Smithery parse failed: {}", e)))?;

    Ok(Json(data))
}

fn urlencoding(s: &str) -> String {
    s.chars().map(|c| match c {
        ' ' => "%20".to_string(),
        '/' => "%2F".to_string(),
        '@' => "%40".to_string(),
        _ if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' => c.to_string(),
        _ => format!("%{:02X}", c as u8),
    }).collect()
}

#[derive(serde::Deserialize)]
pub struct ImportParseRequest {
    spec: String,
}

/// Step 1: Parse spec, return operations for user to select.
pub async fn import_parse(
    Json(req): Json<ImportParseRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let parsed = asv_router::import::parse_spec(&req.spec)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let operations: Vec<serde_json::Value> = parsed.operations.iter().map(|op| {
        serde_json::json!({
            "id": op.id,
            "name": op.name,
            "method": op.method,
            "path": op.path,
            "summary": op.summary,
            "description": op.description,
            "tags": op.tags,
            "parameters": op.parameters.iter().map(|p| serde_json::json!({
                "name": p.name,
                "in": p.location,
                "required": p.required,
            })).collect::<Vec<_>>(),
            "has_body": op.request_body.is_some(),
        })
    }).collect();

    // Collect tags from operations if top-level tags are empty
    let tags = if parsed.tags.is_empty() {
        let mut t: Vec<String> = parsed.operations.iter()
            .flat_map(|op| op.tags.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        t.sort();
        t
    } else {
        parsed.tags.clone()
    };

    Ok(Json(serde_json::json!({
        "title": parsed.title,
        "version": parsed.version,
        "description": parsed.description,
        "total_operations": parsed.operations.len(),
        "tags": tags,
        "operations": operations,
    })))
}

/// Step 2: Import selected operations with LLM seed generation.
#[derive(serde::Deserialize)]
pub struct ImportApplyRequest {
    /// The raw spec (needed for full operation data)
    spec: String,
    /// Operation IDs to import (from the parse step)
    selected: Vec<String>,
    /// Optional domain prefix — imported intent IDs become "domain:intent_name"
    #[serde(default)]
    domain: String,
    /// Languages to generate seeds for (e.g. ["en", "zh", "es"])
    #[serde(default)]
    languages: Vec<String>,
}

pub async fn import_apply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ImportApplyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    let parsed = asv_router::import::parse_spec(&req.spec)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    // Filter to selected operations only
    let selected_set: std::collections::HashSet<&str> = req.selected.iter().map(|s| s.as_str()).collect();
    let selected_ops: Vec<&asv_router::import::openapi::ParsedOperation> = parsed.operations.iter()
        .filter(|op| selected_set.contains(op.id.as_str()))
        .collect();

    if selected_ops.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No operations selected".to_string()));
    }

    // Create intents with description-based seeds first
    {
        let mut routers = state.routers.write().unwrap();
        let router = routers.entry(app_id.clone()).or_insert_with(Router::new);

        for op in &selected_ops {
            let base_name = asv_router::import::to_snake_case(
                op.operation_id.as_deref().unwrap_or(&op.id)
            );
            let intent_name = if req.domain.is_empty() {
                base_name
            } else {
                format!("{}:{}", req.domain, base_name)
            };

            // Only use operation name as minimal placeholder seed.
            // Real seeds are generated by LLM below using the description as context.
            let name_words = op.name.to_lowercase();
            if name_words.is_empty() { continue; }
            router.add_intent(&intent_name, &[name_words.as_str()]);

            // Set description from operation summary/name
            let description = op.summary.as_deref()
                .or(Some(op.name.as_str()))
                .unwrap_or("");
            if !description.is_empty() {
                router.set_description(&intent_name, description);
            }

            // Set type based on method
            let intent_type = match op.method.as_str() {
                "GET" | "HEAD" => asv_router::IntentType::Context,
                _ => asv_router::IntentType::Action,
            };
            router.set_intent_type(&intent_name, intent_type);

            // Store full operation spec as metadata for LLM tool calling
            let endpoint = format!("{} {}", op.method, op.path);
            router.set_metadata(&intent_name, "endpoint", vec![endpoint]);
            if let Some(ref op_id) = op.operation_id {
                router.set_metadata(&intent_name, "operation_id", vec![op_id.clone()]);
            }
            if !op.tags.is_empty() {
                router.set_metadata(&intent_name, "tags", op.tags.clone());
            }

            // Store full operation JSON for LLM context
            if let Ok(op_json) = serde_json::to_string(op) {
                router.set_metadata(&intent_name, "operation_spec", vec![op_json]);
            }

            // Store parameter info
            if !op.parameters.is_empty() {
                let param_names: Vec<String> = op.parameters.iter()
                    .map(|p| format!("{}({}{})", p.name, p.location, if p.required { ",required" } else { "" }))
                    .collect();
                router.set_metadata(&intent_name, "parameters", param_names);
            }
            if op.request_body.is_some() {
                router.set_metadata(&intent_name, "has_body", vec!["true".to_string()]);
            }
        }

        maybe_persist(&state, &app_id, router);
    }

    // Generate LLM seeds through shared pipeline in batches
    let mut total_added = 0usize;
    let mut total_blocked = 0usize;

    if state.llm_key.is_some() {
        for batch in selected_ops.chunks(10) {
            let ops_desc: Vec<String> = batch.iter().map(|op| {
                let base = asv_router::import::to_snake_case(op.operation_id.as_deref().unwrap_or(&op.id));
                let intent_name = if req.domain.is_empty() { base } else { format!("{}:{}", req.domain, base) };
                format!("- {} ({}): {} — {}",
                    intent_name, op.method,
                    op.summary.as_deref().unwrap_or(&op.name),
                    if op.description.len() > 100 { &op.description[..100] } else { &op.description })
            }).collect();

            let existing_seeds: String = {
                let routers = state.routers.read().unwrap();
                if let Some(router) = routers.get(&app_id) {
                    batch.iter().map(|op| {
                        let base = asv_router::import::to_snake_case(op.operation_id.as_deref().unwrap_or(&op.id));
                        let name = if req.domain.is_empty() { base } else { format!("{}:{}", req.domain, base) };
                        let seeds = router.get_training(&name).unwrap_or_default();
                        format!("  {}: {:?}", name, seeds.iter().take(3).collect::<Vec<_>>())
                    }).collect::<Vec<_>>().join("\n")
                } else { String::new() }
            };

            let langs: Vec<&str> = if req.languages.is_empty() {
                vec!["en"]
            } else {
                req.languages.iter().map(|s| s.as_str()).collect()
            };
            let is_multilang = langs.len() > 1;

            let (lang_instruction, response_format) = if is_multilang {
                let lang_list = langs.join(", ");
                (
                    format!("Generate training phrases in ALL of these languages: {}. Include phrases for EACH language.\n\n", lang_list),
                    format!(
                        "Respond with ONLY JSON:\n\
                         {{\"phrases_by_intent\": {{\"intent_name\": {{\"{}\": [\"phrase1\", \"phrase2\"], \"{}\": [\"phrase1\", \"phrase2\"]}}}}}}\n",
                        langs[0], langs.get(1).copied().unwrap_or("zh")
                    ),
                )
            } else {
                (String::new(), "Respond with ONLY JSON:\n{\"phrases_by_intent\": {\"intent_name\": [\"phrase1\", \"phrase2\", \"phrase3\", \"phrase4\", \"phrase5\"]}}\n".to_string())
            };

            let prompt = format!(
                "Generate 10 diverse training phrases for each API operation below.\n\
                 These train a keyword-matching router. VOCABULARY DIVERSITY is critical.\n\n\
                 {}Operations:\n{}\n\n\
                 Current phrases (avoid duplicating these):\n{}\n\n\
                 {}\n\n\
                 For each operation, generate phrases a developer or user would say when they want this action.\n\
                 Mix: short commands, questions, and situational phrases.\n\n\
                 {}",
                lang_instruction, ops_desc.join("\n"), existing_seeds, asv_router::phrase::PHRASE_QUALITY_RULES, response_format
            );

            if let Ok(response) = call_llm(&state, &prompt, 2000).await {
                if let Ok(seeds_json) = serde_json::from_str::<serde_json::Value>(extract_json(&response)) {
                    if let Some(sbi) = seeds_json.get("phrases_by_intent").or_else(|| seeds_json.get("seeds_by_intent")).and_then(|v| v.as_object()) {
                        if is_multilang {
                            // Grouped format: {"intent": {"en": [...], "zh": [...]}}
                            for lang in &langs {
                                let lang_map: HashMap<String, Vec<String>> = sbi.iter()
                                    .filter_map(|(intent, by_lang)| {
                                        by_lang.get(lang)
                                            .and_then(|v| v.as_array())
                                            .map(|arr| (intent.clone(), arr.iter().filter_map(|s| s.as_str().map(String::from)).collect()))
                                    }).collect();
                                if !lang_map.is_empty() {
                                    let result = phrase_pipeline(&state, &app_id, &lang_map, true, lang).await;
                                    total_added += result.added.len();
                                    total_blocked += result.blocked.len();
                                }
                            }
                        } else {
                            // Flat format: {"intent": [...]}
                            let phrases_map: HashMap<String, Vec<String>> = sbi.iter()
                                .filter_map(|(k, v)| {
                                    v.as_array().map(|arr| {
                                        (k.clone(), arr.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                                    })
                                }).collect();
                            let result = phrase_pipeline(&state, &app_id, &phrases_map, true, langs[0]).await;
                            total_added += result.added.len();
                            total_blocked += result.blocked.len();
                        }
                    }
                }
            }
        }
    }

    let intent_names: Vec<String> = selected_ops.iter().map(|op| {
        let base = asv_router::import::to_snake_case(op.operation_id.as_deref().unwrap_or(&op.id));
        if req.domain.is_empty() { base } else { format!("{}:{}", req.domain, base) }
    }).collect();

    Ok(Json(serde_json::json!({
        "title": parsed.title,
        "version": parsed.version,
        "imported": intent_names.len(),
        "phrases_added": total_added,
        "phrases_blocked": total_blocked,
        "intents": intent_names,
    })))
}

// ============================================================
// MCP Tools Import
// ============================================================

#[derive(serde::Deserialize)]
pub struct McpParseRequest {
    /// Raw JSON from MCP tools/list response
    tools_json: String,
}

/// Parse MCP tools/list response — return tools for selection.
pub async fn mcp_parse(
    Json(req): Json<McpParseRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // MCP tools/list response is either:
    // { "tools": [...] }  (standard MCP response)
    // or just [...] (plain array)
    let parsed: serde_json::Value = serde_json::from_str(&req.tools_json)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)))?;

    let tools_array = if let Some(arr) = parsed.as_array() {
        arr.clone()
    } else if let Some(arr) = parsed.get("tools").and_then(|t| t.as_array()) {
        arr.clone()
    } else {
        return Err((StatusCode::BAD_REQUEST, "Expected array of tools or {\"tools\": [...]}".to_string()));
    };

    let tools: Vec<serde_json::Value> = tools_array.iter().map(|tool| {
        let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("unnamed");
        let description = tool.get("description").and_then(|v| v.as_str()).unwrap_or("");
        let has_input = tool.get("inputSchema").is_some();

        // Extract parameter names from inputSchema
        let params: Vec<String> = tool.get("inputSchema")
            .and_then(|s| s.get("properties"))
            .and_then(|p| p.as_object())
            .map(|props| props.keys().cloned().collect())
            .unwrap_or_default();

        let required: Vec<String> = tool.get("inputSchema")
            .and_then(|s| s.get("required"))
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let read_only = tool.get("annotations")
            .and_then(|a| a.get("readOnlyHint"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        serde_json::json!({
            "name": name,
            "description": description,
            "has_input": has_input,
            "params": params,
            "required_params": required,
            "read_only": read_only,
        })
    }).collect();

    Ok(Json(serde_json::json!({
        "total_tools": tools.len(),
        "tools": tools,
    })))
}

#[derive(serde::Deserialize)]
pub struct McpApplyRequest {
    /// Raw JSON from MCP tools/list response
    tools_json: String,
    /// Tool names to import
    selected: Vec<String>,
    /// Optional domain prefix — imported intent IDs become "domain:tool_name"
    #[serde(default)]
    domain: String,
    /// Languages to generate seeds for (e.g. ["en", "zh", "es"])
    #[serde(default)]
    languages: Vec<String>,
}

/// Import selected MCP tools as intents with LLM seed generation.
pub async fn mcp_apply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<McpApplyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    let parsed: serde_json::Value = serde_json::from_str(&req.tools_json)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)))?;

    let tools_array = if let Some(arr) = parsed.as_array() {
        arr.clone()
    } else if let Some(arr) = parsed.get("tools").and_then(|t| t.as_array()) {
        arr.clone()
    } else {
        return Err((StatusCode::BAD_REQUEST, "Expected array of tools".to_string()));
    };

    let selected_set: std::collections::HashSet<&str> = req.selected.iter().map(|s| s.as_str()).collect();

    let selected_tools: Vec<&serde_json::Value> = tools_array.iter()
        .filter(|t| {
            t.get("name").and_then(|n| n.as_str())
                .map(|n| selected_set.contains(n))
                .unwrap_or(false)
        })
        .collect();

    if selected_tools.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No tools selected".to_string()));
    }

    // Create intents from MCP tools
    {
        let mut routers = state.routers.write().unwrap();
        let router = routers.entry(app_id.clone()).or_insert_with(Router::new);

        for tool in &selected_tools {
            let base_name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("unnamed");
            let name = if req.domain.is_empty() {
                base_name.to_string()
            } else {
                format!("{}:{}", req.domain, base_name)
            };
            let description = tool.get("description").and_then(|v| v.as_str()).unwrap_or("");

            let name_words = base_name.replace('_', " ");
            router.add_intent(&name, &[name_words.as_str()]);

            // Set description
            if !description.is_empty() {
                router.set_description(&name, description);
            }

            // Set type based on readOnlyHint
            let read_only = tool.get("annotations")
                .and_then(|a| a.get("readOnlyHint"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            router.set_intent_type(&name, if read_only { IntentType::Context } else { IntentType::Action });

            // Store full tool spec as metadata
            if let Ok(tool_json) = serde_json::to_string(tool) {
                router.set_metadata(&name, "mcp_tool", vec![tool_json]);
            }

            // Store input schema params as metadata
            let params: Vec<String> = tool.get("inputSchema")
                .and_then(|s| s.get("properties"))
                .and_then(|p| p.as_object())
                .map(|props| props.keys().cloned().collect())
                .unwrap_or_default();
            if !params.is_empty() {
                router.set_metadata(&name, "parameters", params);
            }
        }

        maybe_persist(&state, &app_id, router);
    }

    // Generate LLM seeds through shared pipeline
    let mut total_added = 0usize;
    let mut total_blocked = 0usize;

    if state.llm_key.is_some() {
        for batch in selected_tools.chunks(10) {
            let tools_desc: Vec<String> = batch.iter().map(|t| {
                let base = t.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let name = if req.domain.is_empty() { base.to_string() } else { format!("{}:{}", req.domain, base) };
                let desc = t.get("description").and_then(|v| v.as_str()).unwrap_or("");
                format!("- {} : {}", name, &desc[..desc.len().min(100)])
            }).collect();

            let existing_seeds: String = {
                let routers = state.routers.read().unwrap();
                if let Some(router) = routers.get(&app_id) {
                    batch.iter().map(|t| {
                        let base = t.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                        let name = if req.domain.is_empty() { base.to_string() } else { format!("{}:{}", req.domain, base) };
                        let seeds = router.get_training(&name).unwrap_or_default();
                        format!("  {}: {:?}", name, seeds.iter().take(3).collect::<Vec<_>>())
                    }).collect::<Vec<_>>().join("\n")
                } else { String::new() }
            };

            let langs: Vec<&str> = if req.languages.is_empty() {
                vec!["en"]
            } else {
                req.languages.iter().map(|s| s.as_str()).collect()
            };
            let is_multilang = langs.len() > 1;

            let (lang_instruction, response_format) = if is_multilang {
                let lang_list = langs.join(", ");
                (
                    format!("Generate training phrases in ALL of these languages: {}. Include phrases for EACH language.\n\n", lang_list),
                    format!(
                        "Respond with ONLY JSON:\n\
                         {{\"phrases_by_intent\": {{\"tool_name\": {{\"{}\": [\"phrase1\", \"phrase2\"], \"{}\": [\"phrase1\", \"phrase2\"]}}}}}}\n",
                        langs[0], langs.get(1).copied().unwrap_or("zh")
                    ),
                )
            } else {
                (String::new(), "Respond with ONLY JSON:\n{\"phrases_by_intent\": {\"tool_name\": [\"phrase1\", \"phrase2\", \"phrase3\", \"phrase4\", \"phrase5\"]}}\n".to_string())
            };

            let prompt = format!(
                "Generate 10 diverse training phrases for each MCP tool below.\n\
                 These train a keyword-matching router. VOCABULARY DIVERSITY is critical.\n\n\
                 {}Tools:\n{}\n\n\
                 Current phrases (avoid duplicating):\n{}\n\n\
                 {}\n\n\
                 For each tool, generate phrases a user would say when they want this action.\n\
                 Mix: short commands, questions, and situational phrases.\n\n\
                 {}",
                lang_instruction, tools_desc.join("\n"), existing_seeds, asv_router::phrase::PHRASE_QUALITY_RULES, response_format
            );

            if let Ok(response) = call_llm(&state, &prompt, 2000).await {
                if let Ok(seeds_json) = serde_json::from_str::<serde_json::Value>(extract_json(&response)) {
                    if let Some(sbi) = seeds_json.get("phrases_by_intent").or_else(|| seeds_json.get("seeds_by_intent")).and_then(|v| v.as_object()) {
                        if is_multilang {
                            // Grouped format: {"tool": {"en": [...], "zh": [...]}}
                            for lang in &langs {
                                let lang_map: HashMap<String, Vec<String>> = sbi.iter()
                                    .filter_map(|(intent, by_lang)| {
                                        by_lang.get(lang)
                                            .and_then(|v| v.as_array())
                                            .map(|arr| (intent.clone(), arr.iter().filter_map(|s| s.as_str().map(String::from)).collect()))
                                    }).collect();
                                if !lang_map.is_empty() {
                                    let result = phrase_pipeline(&state, &app_id, &lang_map, true, lang).await;
                                    total_added += result.added.len();
                                    total_blocked += result.blocked.len();
                                }
                            }
                        } else {
                            // Flat format: {"tool": [...]}
                            let phrases_map: HashMap<String, Vec<String>> = sbi.iter()
                                .filter_map(|(k, v)| {
                                    v.as_array().map(|arr| {
                                        (k.clone(), arr.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                                    })
                                }).collect();
                            let result = phrase_pipeline(&state, &app_id, &phrases_map, true, langs[0]).await;
                            total_added += result.added.len();
                            total_blocked += result.blocked.len();
                        }
                    }
                }
            }
        }
    }

    let tool_names: Vec<String> = selected_tools.iter()
        .filter_map(|t| t.get("name").and_then(|v| v.as_str()).map(|base| {
            if req.domain.is_empty() { base.to_string() } else { format!("{}:{}", req.domain, base) }
        }))
        .collect();

    Ok(Json(serde_json::json!({
        "imported": tool_names.len(),
        "phrases_added": total_added,
        "phrases_blocked": total_blocked,
        "intents": tool_names,
    })))
}
