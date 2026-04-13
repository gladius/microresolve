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

/// Seed L1 (LexicalGraph) using the accepted phrases as canonical vocabulary context.
///
/// Phrases are passed raw (no stop-word filtering) grouped by intent — multilingual safe.
/// The LLM sees the actual vocabulary L2 learned and decides what abbreviations,
/// morphological variants, and synonyms are worth mapping TO those words.
///
/// Already-covered source words are skipped — safe for incremental imports.
/// Saves to `_hebbian.json` so `load_hebbian()` picks it up automatically on restart.
async fn seed_into_l1(state: &AppState, app_id: &str, accepted: &[(String, String)]) {
    eprintln!("[import/L1] seed_into_l1 called: {} accepted phrases, llm_key={}", accepted.len(), state.llm_key.is_some());
    if accepted.is_empty() || state.llm_key.is_none() { return; }

    // Group phrases by intent_id
    let mut by_intent: std::collections::HashMap<&str, Vec<&str>> = Default::default();
    for (id, phrase) in accepted {
        by_intent.entry(id.as_str()).or_default().push(phrase.as_str());
    }
    eprintln!("[import/L1] {} unique intents", by_intent.len());

    // Build intent blocks: description + phrases (raw, no filtering — multilingual safe)
    let intent_blocks: Vec<String> = {
        let routers = state.routers.read().unwrap();
        eprintln!("[import/L1] router lookup for '{}'", app_id);
        let router = match routers.get(app_id) { Some(r) => r, None => { eprintln!("[import/L1] ERROR: router not found for '{}'", app_id); return; } };

        let mut blocks: Vec<String> = by_intent.iter().map(|(&id, phrases)| {
            let desc = router.get_description(id);
            // 2 phrases per intent — enough vocabulary context, keeps prompt compact
            let phrase_list = phrases.iter().take(2)
                .map(|p| format!("  \"{}\"", p))
                .collect::<Vec<_>>().join("\n");
            if desc.is_empty() {
                format!("{}:\n{}", id, phrase_list)
            } else {
                format!("{} ({}): \n{}", id, desc, phrase_list)
            }
        }).collect();
        blocks.sort(); // deterministic order
        blocks
    };

    // Source words already in L1 — skip to avoid regenerating (incremental safety)
    let existing_from_words: std::collections::HashSet<String> = {
        let heb_map = state.hebbian.read().unwrap();
        heb_map.get(app_id)
            .map(|h| h.edges.keys().cloned().collect())
            .unwrap_or_default()
    };

    let skip_hint = if existing_from_words.is_empty() {
        String::new()
    } else {
        format!("\nAlready mapped (skip FROM these): {}\n",
            existing_from_words.iter().take(60).cloned().collect::<Vec<_>>().join(", "))
    };

    let prompt = format!(
        "You are building a lexical graph for an intent router.\n\
         Below are the intents and the EXACT phrases already learned (these are the canonical vocabulary).\n\
         Your job: generate mappings FROM words a real user would type → TO words already in the phrases.\n\
         The target word MUST appear in the phrases shown — do not invent new canonical forms.\n\
         Works for any language — generate cross-lingual mappings where relevant.\n\n\
         {}\
         {}\n\
         Edge kinds:\n\
         - \"abbreviation\": shorthand → canonical phrase word (pr→pull request, sub→subscription, w=0.97-0.99)\n\
         - \"morphological\": inflected form → base form in phrases (cancellation→cancel, merging→merge, w=0.97-0.99)\n\
           Skip trivial English inflections (gets→get, using→use) unless non-obvious\n\
         - \"synonym\": different word same meaning → phrase word (terminate→cancel, w=0.80-0.96)\n\n\
         Respond with ONLY valid JSON:\n\
         {{\"edges\": [\n\
           {{\"from\": \"pr\", \"to\": \"pull request\", \"kind\": \"abbreviation\", \"weight\": 0.99}},\n\
           {{\"from\": \"cancellation\", \"to\": \"cancel\", \"kind\": \"morphological\", \"weight\": 0.98}},\n\
           {{\"from\": \"terminate\", \"to\": \"cancel\", \"kind\": \"synonym\", \"weight\": 0.88}}\n\
         ]}}",
        intent_blocks.join("\n\n"), skip_hint
    );

    // ── Turn 1: generate candidate edges ────────────────────────────────────
    eprintln!("[import/L1] Turn 1: generating candidates ({} intent blocks, prompt ~{} chars)", intent_blocks.len(), prompt.len());
    let t1_response = match call_llm(state, &prompt, 4096).await {
        Ok(r) => r,
        Err((_, e)) => { eprintln!("[import/L1] Turn 1 failed: {}", e); return; }
    };
    let t1_json = match serde_json::from_str::<serde_json::Value>(extract_json(&t1_response)) {
        Ok(j) => j,
        Err(e) => { eprintln!("[import/L1] Turn 1 parse failed: {} — raw: {}", e, &t1_response[..t1_response.len().min(200)]); return; }
    };
    let Some(candidates) = t1_json.get("edges").and_then(|e| e.as_array()) else {
        eprintln!("[import/L1] Turn 1: no 'edges' array");
        return;
    };
    eprintln!("[import/L1] Turn 1: {} candidate edges", candidates.len());

    // ── Turn 2: fix direction errors and obvious duds only ───────────────────
    let candidates_json = serde_json::to_string_pretty(candidates).unwrap_or_default();
    let verify_prompt = format!(
        "Review these lexical graph edges. Keep good ones, fix or remove bad ones.\n\n\
         REMOVE only if clearly wrong:\n\
         1. Morphological direction is reversed: FROM must be the longer/inflected form, TO must be\n\
            the shorter base form. 'listing→list' is CORRECT. 'list→listing' is WRONG — remove it.\n\
         2. TO word is longer than 4 words — too long for a routing target, remove it.\n\
         3. FROM and TO are the same word.\n\
         4. The mapping is clearly wrong domain-wise (e.g. 'terms→subscription' is too vague).\n\n\
         KEEP everything else — synonyms, cross-lingual mappings, abbreviations, morphological variants.\n\
         Do NOT remove edges just because FROM word is common — 'terminate→cancel' is valuable.\n\n\
         Candidate edges:\n{}\n\n\
         Respond with ONLY valid JSON:\n\
         {{\"edges\": [{{\"from\": \"...\", \"to\": \"...\", \"kind\": \"...\", \"weight\": 0.0}}]}}",
        candidates_json
    );

    eprintln!("[import/L1] Turn 2: verifying {} candidates", candidates.len());
    let t2_response = match call_llm(state, &verify_prompt, 4096).await {
        Ok(r) => r,
        Err((_, e)) => { eprintln!("[import/L1] Turn 2 failed: {}", e); return; }
    };
    let t2_json = match serde_json::from_str::<serde_json::Value>(extract_json(&t2_response)) {
        Ok(j) => j,
        Err(e) => { eprintln!("[import/L1] Turn 2 parse failed: {}", e); return; }
    };
    let Some(edges) = t2_json.get("edges").and_then(|e| e.as_array()) else {
        eprintln!("[import/L1] Turn 2: no 'edges' array");
        return;
    };
    eprintln!("[import/L1] Turn 2: {} edges after verification (removed {})", edges.len(), candidates.len().saturating_sub(edges.len()));

    let mut heb_map = state.hebbian.write().unwrap();
    let heb = heb_map.entry(app_id.to_string())
        .or_insert_with(asv_router::hebbian::LexicalGraph::new);

    let mut n_abbrev = 0usize;
    let mut n_morph  = 0usize;
    let mut n_syn    = 0usize;

    for edge in edges {
        let from   = edge.get("from").and_then(|v| v.as_str()).unwrap_or_default().trim();
        let to     = edge.get("to").and_then(|v| v.as_str()).unwrap_or_default().trim();
        let weight = edge.get("weight").and_then(|v| v.as_f64()).unwrap_or(0.9) as f32;
        let kind_s = edge.get("kind").and_then(|v| v.as_str()).unwrap_or("synonym");
        if from.is_empty() || to.is_empty() || from == to { continue; }
        if heb.edges.contains_key(from) { continue; }

        let kind = match kind_s {
            "abbreviation"  => { n_abbrev += 1; asv_router::hebbian::EdgeKind::Abbreviation }
            "morphological" => { n_morph  += 1; asv_router::hebbian::EdgeKind::Morphological }
            _               => { n_syn    += 1; asv_router::hebbian::EdgeKind::Synonym }
        };
        eprintln!("[import/L1] {:>14}: {} → {} (w={:.2})", kind_s, from, to, weight);
        heb.add(from, to, weight, kind);
    }

    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}/_hebbian.json", dir, app_id);
        std::fs::create_dir_all(format!("{}/{}", dir, app_id)).ok();
        heb.save(&path).ok();
    }
    eprintln!("[import/L1] total: {} abbreviations + {} morphological + {} synonyms for '{}'",
        n_abbrev, n_morph, n_syn, app_id);
}

/// Feed accepted import phrases into the L2 count model (IntentGraph).
///
/// Called once after all batch phrase-generation is complete.
/// Creates the IntentGraph if it doesn't exist yet for this app.
/// Uses L1 preprocessing if available — same tokenisation path as routing.
fn seed_into_l2(state: &AppState, app_id: &str, accepted: &[(String, String)]) {
    if accepted.is_empty() { return; }

    let l1 = state.hebbian.read().unwrap().get(app_id).cloned().unwrap_or_default();

    let mut ig_map = state.intent_graph.write().unwrap();
    let ig = ig_map.entry(app_id.to_string()).or_insert_with(asv_router::hebbian::IntentGraph::new);

    for (intent_id, phrase) in accepted {
        let preprocessed = l1.preprocess(phrase);
        let words = asv_router::tokenizer::tokenize(&preprocessed.expanded);
        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        if !word_refs.is_empty() {
            ig.learn_phrase(&word_refs, intent_id);
        }
    }

    // Persist
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}/_intent_graph.json", dir, app_id);
        std::fs::create_dir_all(format!("{}/{}", dir, app_id)).ok();
        ig.save(&path).ok();
        eprintln!("[import/L2] seeded {} phrases into count model for '{}'", accepted.len(), app_id);
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
fn default_tool_count() -> usize { 10 }

/// GET /api/import/params?num_langs=N&num_tools=M
/// Returns the generation plan so the UI can show the user exactly what will happen.
pub async fn import_params(
    Query(q): Query<ImportParamsQuery>,
) -> Json<serde_json::Value> {
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
    let mut per_intent_added: HashMap<String, usize> = HashMap::new();
    let mut per_intent_blocked: HashMap<String, usize> = HashMap::new();
    let mut per_intent_recovered: HashMap<String, usize> = HashMap::new();
    // Collect all accepted phrases to feed into L2 count model after all batches
    let mut all_accepted: Vec<(String, String)> = Vec::new();

    if state.llm_key.is_some() {
        let setting_langs = state.ui_settings.read().unwrap().languages.clone();
        let (batch_size, max_tokens, _) = seed_gen_params(setting_langs.len());

        for batch in selected_ops.chunks(batch_size) {
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

            let langs: Vec<&str> = setting_langs.iter().map(|s| s.as_str()).collect();
            let is_multilang = langs.len() > 1;

            // Build an example using the first real intent name from this batch
            let example_intent = {
                let op = &batch[0];
                let base = asv_router::import::to_snake_case(op.operation_id.as_deref().unwrap_or(&op.id));
                if req.domain.is_empty() { base } else { format!("{}:{}", req.domain, base) }
            };

            let (lang_instruction, response_format) = if is_multilang {
                let lang_list = langs.join(", ");
                let lang_keys: String = langs.iter().map(|l| format!("\"{}\": [\"phrase1\", \"phrase2\"]", l)).collect::<Vec<_>>().join(", ");
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
                 Mix: short commands, questions, and situational phrases.\n\n\
                 {}",
                lang_instruction, ops_desc.join("\n"), existing_seeds, asv_router::phrase::PHRASE_QUALITY_RULES, response_format
            );

            if let Ok(response) = call_llm(&state, &prompt, max_tokens).await {
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
                                    for (id, _) in &result.added { *per_intent_added.entry(id.clone()).or_default() += 1; }
                                    for (id, _, _) in &result.blocked { *per_intent_blocked.entry(id.clone()).or_default() += 1; }
                                    if result.recovered_by_retry > 0 {
                                        *per_intent_recovered.entry(lang_map.keys().next().cloned().unwrap_or_default()).or_default() += result.recovered_by_retry;
                                    }
                                    all_accepted.extend(result.added.iter().cloned());
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
                            for (id, _) in &result.added { *per_intent_added.entry(id.clone()).or_default() += 1; }
                            for (id, _, _) in &result.blocked { *per_intent_blocked.entry(id.clone()).or_default() += 1; }
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
    // Seed L1 lexical graph with synonym/morphology edges for new domain vocabulary
    seed_into_l1(&state, &app_id, &all_accepted).await;

    let intent_names: Vec<String> = selected_ops.iter().map(|op| {
        let base = asv_router::import::to_snake_case(op.operation_id.as_deref().unwrap_or(&op.id));
        if req.domain.is_empty() { base } else { format!("{}:{}", req.domain, base) }
    }).collect();

    let per_intent: Vec<serde_json::Value> = intent_names.iter().map(|name| {
        let added = per_intent_added.get(name).copied().unwrap_or(0);
        let blocked = per_intent_blocked.get(name).copied().unwrap_or(0);
        let recovered = per_intent_recovered.get(name).copied().unwrap_or(0);
        serde_json::json!({ "name": name, "phrases_added": added, "blocked": blocked, "recovered": recovered })
    }).collect();

    let l2_words = state.intent_graph.read().unwrap()
        .get(&app_id).map(|ig| ig.counts.len()).unwrap_or(0);
    let l1_edges = state.hebbian.read().unwrap()
        .get(&app_id).map(|h| h.edges.len()).unwrap_or(0);

    Ok(Json(serde_json::json!({
        "title": parsed.title,
        "version": parsed.version,
        "imported": intent_names.len(),
        "phrases_added": total_added,
        "phrases_blocked": total_blocked,
        "l1_lexical_edges": l1_edges,
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
    let mut per_intent_added: HashMap<String, usize> = HashMap::new();
    let mut per_intent_blocked: HashMap<String, usize> = HashMap::new();
    let mut per_intent_recovered: HashMap<String, usize> = HashMap::new();
    let mut all_accepted: Vec<(String, String)> = Vec::new();

    if state.llm_key.is_some() {
        let setting_langs = state.ui_settings.read().unwrap().languages.clone();
        let (batch_size, max_tokens, _) = seed_gen_params(setting_langs.len());

        for batch in selected_tools.chunks(batch_size) {
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

            let langs: Vec<&str> = setting_langs.iter().map(|s| s.as_str()).collect();
            let is_multilang = langs.len() > 1;

            // Build example using the first real tool name from this batch
            let example_intent = {
                let base = batch[0].get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                if req.domain.is_empty() { base.to_string() } else { format!("{}:{}", req.domain, base) }
            };

            let (lang_instruction, response_format) = if is_multilang {
                let lang_list = langs.join(", ");
                let lang_keys: String = langs.iter().map(|l| format!("\"{}\": [\"phrase1\", \"phrase2\"]", l)).collect::<Vec<_>>().join(", ");
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
                 Mix: short commands, questions, and situational phrases.\n\n\
                 {}",
                lang_instruction, tools_desc.join("\n"), existing_seeds, asv_router::phrase::PHRASE_QUALITY_RULES, response_format
            );

            if let Ok(response) = call_llm(&state, &prompt, max_tokens).await {
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
                                    for (id, _) in &result.added { *per_intent_added.entry(id.clone()).or_default() += 1; }
                                    for (id, _, _) in &result.blocked { *per_intent_blocked.entry(id.clone()).or_default() += 1; }
                                    if result.recovered_by_retry > 0 {
                                        *per_intent_recovered.entry(lang_map.keys().next().cloned().unwrap_or_default()).or_default() += result.recovered_by_retry;
                                    }
                                    all_accepted.extend(result.added.iter().cloned());
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
                            for (id, _) in &result.added { *per_intent_added.entry(id.clone()).or_default() += 1; }
                            for (id, _, _) in &result.blocked { *per_intent_blocked.entry(id.clone()).or_default() += 1; }
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
    // Seed L1 lexical graph with synonym/morphology edges for new domain vocabulary
    seed_into_l1(&state, &app_id, &all_accepted).await;

    let tool_names: Vec<String> = selected_tools.iter()
        .filter_map(|t| t.get("name").and_then(|v| v.as_str()).map(|base| {
            if req.domain.is_empty() { base.to_string() } else { format!("{}:{}", req.domain, base) }
        }))
        .collect();

    let per_intent: Vec<serde_json::Value> = tool_names.iter().map(|name| {
        let added = per_intent_added.get(name).copied().unwrap_or(0);
        let blocked = per_intent_blocked.get(name).copied().unwrap_or(0);
        let recovered = per_intent_recovered.get(name).copied().unwrap_or(0);
        serde_json::json!({ "name": name, "phrases_added": added, "blocked": blocked, "recovered": recovered })
    }).collect();

    let l2_words = state.intent_graph.read().unwrap()
        .get(&app_id).map(|ig| ig.counts.len()).unwrap_or(0);
    let l1_edges = state.hebbian.read().unwrap()
        .get(&app_id).map(|h| h.edges.len()).unwrap_or(0);

    Ok(Json(serde_json::json!({
        "imported": tool_names.len(),
        "phrases_added": total_added,
        "phrases_blocked": total_blocked,
        "l1_lexical_edges": l1_edges,
        "l2_unique_words": l2_words,
        "intents": tool_names,
        "per_intent": per_intent,
    })))
}
