//! Spec import endpoints (OpenAPI, Postman).

use crate::pipeline::*;
use crate::state::*;
use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json,
};
use microresolve::IntentType;
use std::collections::HashMap;

/// Feed accepted import phrases into the namespace's L2 (IntentIndex).
///
/// Called once after all batch phrase-generation is complete.
pub fn seed_into_l2(state: &AppState, app_id: &str, accepted: &[(String, String)]) {
    if accepted.is_empty() {
        return;
    }
    let Some(h) = state.engine.try_namespace(app_id) else {
        return;
    };

    for (intent_id, phrase) in accepted {
        h.index_phrase(intent_id, phrase);
    }
    h.rebuild_idf();
    eprintln!(
        "[import/L2] seeded {} phrases into count model for '{}'",
        accepted.len(),
        app_id
    );
    if let Err(e) = h.flush() {
        eprintln!("[import/L2] flush error for {}: {}", app_id, e);
    }
}

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/import/parse", post(import_parse))
        .route("/api/import/apply", post(import_apply))
        .route("/api/import/params", get(import_params))
        .route("/api/import/mcp/search", get(mcp_search))
        .route("/api/import/mcp/fetch", get(mcp_fetch))
        .route("/api/import/mcp/parse", post(mcp_parse))
        .route("/api/import/mcp/apply", post(mcp_apply))
}

/// Calculate optimal batch_size and max_tokens for LLM seed generation.
///
/// Formula:
///   phrases_per_intent  = 5  (focused, spec-driven — quality over quantity)
///   tokens_per_phrase   = 10 (avg across en/zh: short cmds ~4, sentences ~15)
///   json_overhead       = 30 (keys, brackets, commas per intent)
///   tokens_per_intent   = num_langs × 5 × 10 + 30 = num_langs × 50 + 30
///
///   batch_size = min(10, floor(8192 / tokens_per_intent))
///
/// Returns (batch_size, max_tokens, tokens_per_tool).
pub fn seed_gen_params(num_langs: usize) -> (usize, u32, u32) {
    let n = num_langs.max(1) as u32;
    let tokens_per_intent = n * 50 + 30; // num_langs × 5 phrases × 10 tokens + JSON overhead
    let batch_size = ((8192 / tokens_per_intent) as usize).max(1).min(10);
    let max_tokens = 8192;
    (batch_size, max_tokens, tokens_per_intent)
}

#[derive(serde::Deserialize)]
pub struct ImportParamsQuery {
    num_langs: usize,
    #[serde(default = "default_tool_count")]
    num_tools: usize,
}
fn default_tool_count() -> usize {
    10
}

/// GET /api/import/params?num_langs=N&num_tools=M
/// Returns the generation plan so the UI can show the user exactly what will happen.
pub async fn import_params(Query(q): Query<ImportParamsQuery>) -> Json<serde_json::Value> {
    let (batch_size, max_tokens, tokens_per_tool) = seed_gen_params(q.num_langs);
    let total_batches = (q.num_tools + batch_size - 1) / batch_size;
    // Expected output per batch: batch_size tools × tokens_per_tool (max_tokens=8192 is the ceiling)
    let expected_output_per_batch = batch_size as u32 * tokens_per_tool;
    let total_output_tokens = total_batches as u32 * expected_output_per_batch;
    // Input: ~300 tokens/tool (prompt context) + ~200 overhead per batch
    let total_input_tokens = q.num_tools as u32 * 300 + total_batches as u32 * 200;
    let total_tokens = total_input_tokens + total_output_tokens;

    Json(serde_json::json!({
        "batch_size": batch_size,
        "max_tokens_per_call": max_tokens,
        "tokens_per_tool": tokens_per_tool,
        "total_batches": total_batches,
        "total_output_tokens": total_output_tokens,
        "total_input_tokens": total_input_tokens,
        "total_tokens": total_tokens,
        "phrases_per_tool": q.num_langs * 5,
    }))
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

pub fn default_page_size() -> usize {
    20
}

/// Search MCP servers on Smithery registry.
pub async fn mcp_search(
    State(state): State<AppState>,
    Query(params): Query<McpSearchParams>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let url = format!(
        "https://api.smithery.ai/servers?q={}&pageSize={}",
        urlencoding(&params.q),
        params.limit
    );

    let resp = state.http.get(&url).send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Smithery fetch failed: {}", e),
        )
    })?;

    if !resp.status().is_success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("Smithery returned {}", resp.status()),
        ));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Smithery parse failed: {}", e),
        )
    })?;

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
    let url = format!(
        "https://api.smithery.ai/servers/{}",
        urlencoding(&params.name)
    );

    let resp = state.http.get(&url).send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Smithery fetch failed: {}", e),
        )
    })?;

    if !resp.status().is_success() {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("Smithery returned {} for '{}'", resp.status(), params.name),
        ));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("Smithery parse failed: {}", e),
        )
    })?;

    Ok(Json(data))
}

fn urlencoding(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' => "%20".to_string(),
            '/' => "%2F".to_string(),
            '@' => "%40".to_string(),
            _ if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' => c.to_string(),
            _ => format!("%{:02X}", c as u8),
        })
        .collect()
}

#[derive(serde::Deserialize)]
pub struct ImportParseRequest {
    spec: String,
}

/// Step 1: Parse spec, return operations for user to select.
pub async fn import_parse(
    Json(req): Json<ImportParseRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let parsed =
        microresolve::import::parse_spec(&req.spec).map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    let operations: Vec<serde_json::Value> = parsed
        .operations
        .iter()
        .map(|op| {
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
        })
        .collect();

    // Collect tags from operations if top-level tags are empty
    let tags = if parsed.tags.is_empty() {
        let mut t: Vec<String> = parsed
            .operations
            .iter()
            .flat_map(|op| op.tags.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
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
}

pub async fn import_apply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ImportApplyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    let parsed =
        microresolve::import::parse_spec(&req.spec).map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    // Filter to selected operations only
    let selected_set: std::collections::HashSet<&str> =
        req.selected.iter().map(|s| s.as_str()).collect();
    let selected_ops: Vec<&microresolve::import::openapi::ParsedOperation> = parsed
        .operations
        .iter()
        .filter(|op| selected_set.contains(op.id.as_str()))
        .collect();

    if selected_ops.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "No operations selected".to_string(),
        ));
    }

    // Create intents with description-based seeds first
    {
        let h = state.engine.namespace(&app_id);
        for op in &selected_ops {
            let base_name =
                microresolve::import::to_snake_case(op.operation_id.as_deref().unwrap_or(&op.id));
            let intent_name = if req.domain.is_empty() {
                base_name
            } else {
                format!("{}:{}", req.domain, base_name)
            };

            let name_words = op.name.to_lowercase();
            if name_words.is_empty() {
                continue;
            }
            let _ = h.add_intent(&intent_name, &[name_words.as_str()][..]);

            let description = op
                .summary
                .as_deref()
                .or(Some(op.name.as_str()))
                .unwrap_or("");
            let intent_type = match op.method.as_str() {
                "GET" | "HEAD" => microresolve::IntentType::Context,
                _ => microresolve::IntentType::Action,
            };
            let endpoint = format!("{} {}", op.method, op.path);
            let schema = serde_json::json!({
                "method": op.method,
                "path": op.path,
                "operation_id": op.operation_id,
                "summary": op.summary,
                "tags": op.tags,
                "parameters": op.parameters.iter().map(|p| serde_json::json!({
                    "name": p.name,
                    "in": p.location,
                    "required": p.required,
                })).collect::<Vec<_>>(),
                "has_body": op.request_body.is_some(),
            });

            let _ = h.update_intent(
                &intent_name,
                microresolve::IntentEdit {
                    description: if description.is_empty() {
                        None
                    } else {
                        Some(description.to_string())
                    },
                    intent_type: Some(intent_type),
                    source: Some(
                        microresolve::IntentSource::new("openapi").with_label(parsed.title.clone()),
                    ),
                    schema: Some(schema),
                    target: Some(
                        microresolve::IntentTarget::new("api_endpoint")
                            .with_handler(endpoint.clone()),
                    ),
                    ..Default::default()
                },
            );
        }
        maybe_commit(&state, &app_id);
    }

    // Generate LLM seeds through shared pipeline in batches
    let mut total_added = 0usize;
    let mut total_blocked = 0usize;
    let mut per_intent_added: HashMap<String, usize> = HashMap::new();
    let mut per_intent_blocked: HashMap<String, usize> = HashMap::new();
    let mut per_intent_recovered: HashMap<String, usize> = HashMap::new();
    // Collect all accepted phrases to feed into L2 count model after all batches
    let mut all_accepted: Vec<(String, String)> = Vec::new();

    if state.llm_key.is_some() {
        let setting_langs = state.ui_settings.read().unwrap().languages.clone();
        let (batch_size, max_tokens, _) = seed_gen_params(setting_langs.len());

        for batch in selected_ops.chunks(batch_size) {
            let ops_desc: Vec<String> = batch
                .iter()
                .map(|op| {
                    let base = microresolve::import::to_snake_case(
                        op.operation_id.as_deref().unwrap_or(&op.id),
                    );
                    let intent_name = if req.domain.is_empty() {
                        base
                    } else {
                        format!("{}:{}", req.domain, base)
                    };
                    format!(
                        "- {} ({}): {} — {}",
                        intent_name,
                        op.method,
                        op.summary.as_deref().unwrap_or(&op.name),
                        if op.description.len() > 100 {
                            &op.description[..100]
                        } else {
                            &op.description
                        }
                    )
                })
                .collect();

            let existing_seeds: String = state
                .engine
                .try_namespace(&app_id)
                .map(|h| {
                    batch
                        .iter()
                        .map(|op| {
                            let base = microresolve::import::to_snake_case(
                                op.operation_id.as_deref().unwrap_or(&op.id),
                            );
                            let name = if req.domain.is_empty() {
                                base
                            } else {
                                format!("{}:{}", req.domain, base)
                            };
                            let seeds = h.training(&name).unwrap_or_default();
                            format!("  {}: {:?}", name, seeds.iter().take(3).collect::<Vec<_>>())
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();

            let langs: Vec<&str> = setting_langs.iter().map(|s| s.as_str()).collect();
            let is_multilang = langs.len() > 1;

            // Build an example using the first real intent name from this batch
            let example_intent = {
                let op = &batch[0];
                let base = microresolve::import::to_snake_case(
                    op.operation_id.as_deref().unwrap_or(&op.id),
                );
                if req.domain.is_empty() {
                    base
                } else {
                    format!("{}:{}", req.domain, base)
                }
            };

            let (lang_instruction, response_format) = if is_multilang {
                let lang_list = langs.join(", ");
                let lang_keys: String = langs
                    .iter()
                    .map(|l| format!("\"{}\": [\"phrase1\", \"phrase2\"]", l))
                    .collect::<Vec<_>>()
                    .join(", ");
                (
                    format!("Generate training phrases in ALL of these languages: {}. Include phrases for EACH language.\n\n", lang_list),
                    format!("Respond with ONLY valid JSON (no extra text):\n{{\"phrases_by_intent\": {{\"{}\": {{{}}}}}}}\n", example_intent, lang_keys),
                )
            } else {
                (String::new(), format!("Respond with ONLY valid JSON (no extra text):\n{{\"phrases_by_intent\": {{\"{}\": [\"phrase1\", \"phrase2\", \"phrase3\"]}}}}\n", example_intent))
            };

            let prompt = format!(
                "Generate 5 diverse training phrases for each API operation below.\n\
                 You have the full operation context — use it to generate TARGETED phrases.\n\
                 These phrases train both a keyword router and a statistical count model.\n\
                 Phrases must use vocabulary specific to THIS operation, not generic API terms.\n\n\
                 {}Operations:\n{}\n\n\
                 Current phrases (avoid duplicating these):\n{}\n\n\
                 {}\n\n\
                 For each operation, generate phrases a developer or user would say when they want this action.\n\
                 Use imperative commands and short tool-distinctive requests. Avoid leading with generic conversational stems.\n\n\
                 {}",
                lang_instruction, ops_desc.join("\n"), existing_seeds, microresolve::phrase::PHRASE_QUALITY_RULES, response_format
            );

            if let Ok(response) = call_llm(&state, &prompt, max_tokens).await {
                if let Ok(seeds_json) =
                    serde_json::from_str::<serde_json::Value>(extract_json(&response))
                {
                    if let Some(sbi) = seeds_json
                        .get("phrases_by_intent")
                        .or_else(|| seeds_json.get("seeds_by_intent"))
                        .and_then(|v| v.as_object())
                    {
                        if is_multilang {
                            // Grouped format: {"intent": {"en": [...], "zh": [...]}}
                            for lang in &langs {
                                let lang_map: HashMap<String, Vec<String>> = sbi
                                    .iter()
                                    .filter_map(|(intent, by_lang)| {
                                        by_lang.get(lang).and_then(|v| v.as_array()).map(|arr| {
                                            (
                                                intent.clone(),
                                                arr.iter()
                                                    .filter_map(|s| s.as_str().map(String::from))
                                                    .collect(),
                                            )
                                        })
                                    })
                                    .collect();
                                if !lang_map.is_empty() {
                                    let result =
                                        phrase_pipeline(&state, &app_id, &lang_map, true, lang)
                                            .await;
                                    for (id, _) in &result.added {
                                        *per_intent_added.entry(id.clone()).or_default() += 1;
                                    }
                                    for (id, _, _) in &result.blocked {
                                        *per_intent_blocked.entry(id.clone()).or_default() += 1;
                                    }
                                    if result.recovered_by_retry > 0 {
                                        *per_intent_recovered
                                            .entry(
                                                lang_map.keys().next().cloned().unwrap_or_default(),
                                            )
                                            .or_default() += result.recovered_by_retry;
                                    }
                                    all_accepted.extend(result.added.iter().cloned());
                                    total_added += result.added.len();
                                    total_blocked += result.blocked.len();
                                }
                            }
                        } else {
                            // Flat format: {"intent": [...]}
                            let phrases_map: HashMap<String, Vec<String>> = sbi
                                .iter()
                                .filter_map(|(k, v)| {
                                    v.as_array().map(|arr| {
                                        (
                                            k.clone(),
                                            arr.iter()
                                                .filter_map(|s| s.as_str().map(String::from))
                                                .collect(),
                                        )
                                    })
                                })
                                .collect();
                            let result =
                                phrase_pipeline(&state, &app_id, &phrases_map, true, langs[0])
                                    .await;
                            for (id, _) in &result.added {
                                *per_intent_added.entry(id.clone()).or_default() += 1;
                            }
                            for (id, _, _) in &result.blocked {
                                *per_intent_blocked.entry(id.clone()).or_default() += 1;
                            }
                            all_accepted.extend(result.added.iter().cloned());
                            total_added += result.added.len();
                            total_blocked += result.blocked.len();
                        }
                    }
                }
            }
        }
    }

    // Feed accepted phrases into L2 count model — zero extra LLM cost
    seed_into_l2(&state, &app_id, &all_accepted);

    let intent_names: Vec<String> = selected_ops
        .iter()
        .map(|op| {
            let base =
                microresolve::import::to_snake_case(op.operation_id.as_deref().unwrap_or(&op.id));
            if req.domain.is_empty() {
                base
            } else {
                format!("{}:{}", req.domain, base)
            }
        })
        .collect();

    let per_intent: Vec<serde_json::Value> = intent_names.iter().map(|name| {
        let added = per_intent_added.get(name).copied().unwrap_or(0);
        let blocked = per_intent_blocked.get(name).copied().unwrap_or(0);
        let recovered = per_intent_recovered.get(name).copied().unwrap_or(0);
        serde_json::json!({ "name": name, "phrases_added": added, "blocked": blocked, "recovered": recovered })
    }).collect();

    let l2_words = state
        .engine
        .try_namespace(&app_id)
        .map(|h| h.vocab_size())
        .unwrap_or(0);

    Ok(Json(serde_json::json!({
        "title": parsed.title,
        "version": parsed.version,
        "imported": intent_names.len(),
        "phrases_added": total_added,
        "phrases_blocked": total_blocked,
        "l2_unique_words": l2_words,
        "intents": intent_names,
        "per_intent": per_intent,
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

/// Normalize a tool definition into MCP shape {name, description, inputSchema, annotations}.
///
/// Handles three input formats:
///   MCP:      { "name": "...", "description": "...", "inputSchema": {...} }
///   OpenAI:   { "type": "function", "function": { "name": "...", "description": "...", "parameters": {...} } }
///   LangChain:{ "name": "...", "description": "...", "args_schema": {...} }
fn normalize_tool(tool: &serde_json::Value) -> serde_json::Value {
    // OpenAI function calling: unwrap the "function" wrapper
    if tool.get("type").and_then(|t| t.as_str()) == Some("function") {
        if let Some(func) = tool.get("function") {
            return serde_json::json!({
                "name": func.get("name").cloned().unwrap_or(serde_json::Value::Null),
                "description": func.get("description").cloned().unwrap_or(serde_json::json!("")),
                "inputSchema": func.get("parameters").cloned().unwrap_or(serde_json::Value::Null),
            });
        }
    }
    // LangChain tool: args_schema → inputSchema
    if tool.get("args_schema").is_some() {
        return serde_json::json!({
            "name": tool.get("name").cloned().unwrap_or(serde_json::Value::Null),
            "description": tool.get("description").cloned().unwrap_or(serde_json::json!("")),
            "inputSchema": tool.get("args_schema").cloned().unwrap_or(serde_json::Value::Null),
        });
    }
    // Already MCP shape — pass through
    tool.clone()
}

/// Detect source type from raw (pre-normalization) tool entry.
fn detect_source_type(tool: &serde_json::Value) -> &'static str {
    if tool.get("type").and_then(|t| t.as_str()) == Some("function") {
        return "function";
    }
    if tool.get("args_schema").is_some() {
        return "langchain";
    }
    "mcp"
}

/// Extract and normalize the tools array from any supported input format.
/// Returns (normalized_tools, source_types).
fn extract_tools(
    parsed: &serde_json::Value,
) -> Result<Vec<serde_json::Value>, (StatusCode, String)> {
    let raw = if let Some(arr) = parsed.as_array() {
        arr.clone()
    } else if let Some(arr) = parsed.get("tools").and_then(|t| t.as_array()) {
        arr.clone()
    } else {
        return Err((StatusCode::BAD_REQUEST,
            "Expected array of tools or {\"tools\": [...]}. Supports MCP, OpenAI function calling, and LangChain tool formats.".to_string()));
    };
    Ok(raw.iter().map(normalize_tool).collect())
}

/// Extract tools retaining the original source type per tool.
fn extract_tools_with_source(
    parsed: &serde_json::Value,
) -> Result<Vec<(serde_json::Value, &'static str)>, (StatusCode, String)> {
    let raw = if let Some(arr) = parsed.as_array() {
        arr.clone()
    } else if let Some(arr) = parsed.get("tools").and_then(|t| t.as_array()) {
        arr.clone()
    } else {
        return Err((StatusCode::BAD_REQUEST,
            "Expected array of tools or {\"tools\": [...]}. Supports MCP, OpenAI function calling, and LangChain tool formats.".to_string()));
    };
    Ok(raw
        .iter()
        .map(|t| (normalize_tool(t), detect_source_type(t)))
        .collect())
}

/// Parse MCP / OpenAI / LangChain tools — return normalized list for selection.
pub async fn mcp_parse(
    Json(req): Json<McpParseRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let parsed: serde_json::Value = serde_json::from_str(&req.tools_json)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid JSON: {}", e)))?;

    let tools_array = extract_tools(&parsed)?;

    let tools: Vec<serde_json::Value> = tools_array
        .iter()
        .map(|tool| {
            let name = tool
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unnamed");
            let description = tool
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let has_input = tool.get("inputSchema").is_some();

            // Extract parameter names from inputSchema
            let params: Vec<String> = tool
                .get("inputSchema")
                .and_then(|s| s.get("properties"))
                .and_then(|p| p.as_object())
                .map(|props| props.keys().cloned().collect())
                .unwrap_or_default();

            let required: Vec<String> = tool
                .get("inputSchema")
                .and_then(|s| s.get("required"))
                .and_then(|r| r.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let read_only = tool
                .get("annotations")
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
        })
        .collect();

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

    let tools_with_source = extract_tools_with_source(&parsed)?;

    let selected_set: std::collections::HashSet<&str> =
        req.selected.iter().map(|s| s.as_str()).collect();

    let selected_tools: Vec<(&serde_json::Value, &str)> = tools_with_source
        .iter()
        .filter(|(t, _)| {
            t.get("name")
                .and_then(|n| n.as_str())
                .map(|n| selected_set.contains(n))
                .unwrap_or(false)
        })
        .map(|(t, s)| (t, *s))
        .collect();

    if selected_tools.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No tools selected".to_string()));
    }

    // Create intents from tools
    {
        let h = state.engine.namespace(&app_id);
        for (tool, source_type) in &selected_tools {
            let base_name = tool
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unnamed");
            let name = if req.domain.is_empty() {
                base_name.to_string()
            } else {
                format!("{}:{}", req.domain, base_name)
            };
            let description = tool
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let name_words = base_name.replace('_', " ");
            let _ = h.add_intent(&name, &[name_words.as_str()][..]);

            let read_only = tool
                .get("annotations")
                .and_then(|a| a.get("readOnlyHint"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let _ = h.update_intent(
                &name,
                microresolve::IntentEdit {
                    description: if description.is_empty() {
                        None
                    } else {
                        Some(description.to_string())
                    },
                    intent_type: Some(if read_only {
                        IntentType::Context
                    } else {
                        IntentType::Action
                    }),
                    source: Some(microresolve::IntentSource::new(*source_type)),
                    schema: Some((*tool).clone()),
                    target: Some(microresolve::IntentTarget::new(if *source_type == "mcp" {
                        "mcp_server"
                    } else {
                        "handler"
                    })),
                    ..Default::default()
                },
            );
        }
        maybe_commit(&state, &app_id);
    }

    // Generate LLM seeds through shared pipeline
    let mut total_added = 0usize;
    let mut total_blocked = 0usize;
    let mut per_intent_added: HashMap<String, usize> = HashMap::new();
    let mut per_intent_blocked: HashMap<String, usize> = HashMap::new();
    let mut per_intent_recovered: HashMap<String, usize> = HashMap::new();
    let mut all_accepted: Vec<(String, String)> = Vec::new();

    if state.llm_key.is_some() {
        let setting_langs = state.ui_settings.read().unwrap().languages.clone();
        let (batch_size, max_tokens, _) = seed_gen_params(setting_langs.len());

        for batch in selected_tools.chunks(batch_size) {
            let tools_desc: Vec<String> = batch
                .iter()
                .map(|(t, _)| {
                    let base = t.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                    let name = if req.domain.is_empty() {
                        base.to_string()
                    } else {
                        format!("{}:{}", req.domain, base)
                    };
                    let desc = t.get("description").and_then(|v| v.as_str()).unwrap_or("");
                    format!("- {} : {}", name, &desc[..desc.len().min(100)])
                })
                .collect();

            let existing_seeds: String = state
                .engine
                .try_namespace(&app_id)
                .map(|h| {
                    batch
                        .iter()
                        .map(|(t, _)| {
                            let base = t.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                            let name = if req.domain.is_empty() {
                                base.to_string()
                            } else {
                                format!("{}:{}", req.domain, base)
                            };
                            let seeds = h.training(&name).unwrap_or_default();
                            format!("  {}: {:?}", name, seeds.iter().take(3).collect::<Vec<_>>())
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();

            let langs: Vec<&str> = setting_langs.iter().map(|s| s.as_str()).collect();
            let is_multilang = langs.len() > 1;

            // Build example using the first real tool name from this batch
            let example_intent = {
                let base = batch[0]
                    .0
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tool");
                if req.domain.is_empty() {
                    base.to_string()
                } else {
                    format!("{}:{}", req.domain, base)
                }
            };

            let (lang_instruction, response_format) = if is_multilang {
                let lang_list = langs.join(", ");
                let lang_keys: String = langs
                    .iter()
                    .map(|l| format!("\"{}\": [\"phrase1\", \"phrase2\"]", l))
                    .collect::<Vec<_>>()
                    .join(", ");
                (
                    format!("Generate training phrases in ALL of these languages: {}. Include phrases for EACH language.\n\n", lang_list),
                    format!("Respond with ONLY valid JSON (no extra text):\n{{\"phrases_by_intent\": {{\"{}\": {{{}}}}}}}\n", example_intent, lang_keys),
                )
            } else {
                (String::new(), format!("Respond with ONLY valid JSON (no extra text):\n{{\"phrases_by_intent\": {{\"{}\": [\"phrase1\", \"phrase2\", \"phrase3\"]}}}}\n", example_intent))
            };

            let prompt = format!(
                "Generate 5 diverse training phrases for each MCP tool below.\n\
                 You have the full tool description — use it to generate TARGETED phrases.\n\
                 These phrases train both a keyword router and a statistical count model.\n\
                 Phrases must use vocabulary specific to THIS tool, not generic action terms.\n\n\
                 {}Tools:\n{}\n\n\
                 Current phrases (avoid duplicating):\n{}\n\n\
                 {}\n\n\
                 For each tool, generate phrases a user would say when they want this action.\n\
                 Use imperative commands and short tool-distinctive requests. Avoid leading with generic conversational stems.\n\n\
                 {}",
                lang_instruction, tools_desc.join("\n"), existing_seeds, microresolve::phrase::PHRASE_QUALITY_RULES, response_format
            );

            if let Ok(response) = call_llm(&state, &prompt, max_tokens).await {
                if let Ok(seeds_json) =
                    serde_json::from_str::<serde_json::Value>(extract_json(&response))
                {
                    if let Some(sbi) = seeds_json
                        .get("phrases_by_intent")
                        .or_else(|| seeds_json.get("seeds_by_intent"))
                        .and_then(|v| v.as_object())
                    {
                        if is_multilang {
                            // Grouped format: {"tool": {"en": [...], "zh": [...]}}
                            for lang in &langs {
                                let lang_map: HashMap<String, Vec<String>> = sbi
                                    .iter()
                                    .filter_map(|(intent, by_lang)| {
                                        by_lang.get(lang).and_then(|v| v.as_array()).map(|arr| {
                                            (
                                                intent.clone(),
                                                arr.iter()
                                                    .filter_map(|s| s.as_str().map(String::from))
                                                    .collect(),
                                            )
                                        })
                                    })
                                    .collect();
                                if !lang_map.is_empty() {
                                    let result =
                                        phrase_pipeline(&state, &app_id, &lang_map, true, lang)
                                            .await;
                                    for (id, _) in &result.added {
                                        *per_intent_added.entry(id.clone()).or_default() += 1;
                                    }
                                    for (id, _, _) in &result.blocked {
                                        *per_intent_blocked.entry(id.clone()).or_default() += 1;
                                    }
                                    if result.recovered_by_retry > 0 {
                                        *per_intent_recovered
                                            .entry(
                                                lang_map.keys().next().cloned().unwrap_or_default(),
                                            )
                                            .or_default() += result.recovered_by_retry;
                                    }
                                    all_accepted.extend(result.added.iter().cloned());
                                    total_added += result.added.len();
                                    total_blocked += result.blocked.len();
                                }
                            }
                        } else {
                            // Flat format: {"tool": [...]}
                            let phrases_map: HashMap<String, Vec<String>> = sbi
                                .iter()
                                .filter_map(|(k, v)| {
                                    v.as_array().map(|arr| {
                                        (
                                            k.clone(),
                                            arr.iter()
                                                .filter_map(|s| s.as_str().map(String::from))
                                                .collect(),
                                        )
                                    })
                                })
                                .collect();
                            let result =
                                phrase_pipeline(&state, &app_id, &phrases_map, true, langs[0])
                                    .await;
                            for (id, _) in &result.added {
                                *per_intent_added.entry(id.clone()).or_default() += 1;
                            }
                            for (id, _, _) in &result.blocked {
                                *per_intent_blocked.entry(id.clone()).or_default() += 1;
                            }
                            all_accepted.extend(result.added.iter().cloned());
                            total_added += result.added.len();
                            total_blocked += result.blocked.len();
                        }
                    }
                }
            }
        }
    }

    // Feed accepted phrases into L2 count model — zero extra LLM cost
    seed_into_l2(&state, &app_id, &all_accepted);

    let tool_names: Vec<String> = selected_tools
        .iter()
        .filter_map(|(t, _)| {
            t.get("name").and_then(|v| v.as_str()).map(|base| {
                if req.domain.is_empty() {
                    base.to_string()
                } else {
                    format!("{}:{}", req.domain, base)
                }
            })
        })
        .collect();

    let per_intent: Vec<serde_json::Value> = tool_names.iter().map(|name| {
        let added = per_intent_added.get(name).copied().unwrap_or(0);
        let blocked = per_intent_blocked.get(name).copied().unwrap_or(0);
        let recovered = per_intent_recovered.get(name).copied().unwrap_or(0);
        serde_json::json!({ "name": name, "phrases_added": added, "blocked": blocked, "recovered": recovered })
    }).collect();

    let l2_words = state
        .engine
        .try_namespace(&app_id)
        .map(|h| h.vocab_size())
        .unwrap_or(0);

    Ok(Json(serde_json::json!({
        "imported": tool_names.len(),
        "phrases_added": total_added,
        "phrases_blocked": total_blocked,
        "l2_unique_words": l2_words,
        "intents": tool_names,
        "per_intent": per_intent,
    })))
}
