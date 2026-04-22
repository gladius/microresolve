//! Entity-detection management endpoints.
//!
//! Three groups of endpoints:
//!
//! 1. **Discovery** — list available built-in patterns (`GET /api/entities/builtin`).
//! 2. **Per-namespace config** — get/update which patterns are enabled
//!    (`GET /api/entities`, `PATCH /api/entities`, custom CRUD).
//! 3. **Operations** — apply the configured layer to text
//!    (`POST /api/entities/detect`, `/extract`, `/mask`).
//! 4. **LLM distillation** — propose patterns from a plain-English description
//!    (`POST /api/entities/distill`). Returns proposed JSON, does not save.
//!
//! All per-namespace operations use `X-Workspace-ID` header to identify the
//! namespace. Pattern operations always reflect the *current* persisted config
//! for that namespace.

use axum::{
    extract::State,
    http::{StatusCode, HeaderMap},
    routing::{get, post, patch, delete},
    Json,
};
use microresolve::entity::{
    BUILTIN_PATTERNS, EntityConfig, EntityLayer, CustomEntity, get_builtin, all_categories,
};
use crate::state::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/entities/builtin", get(list_builtin))
        .route("/api/entities",         get(get_config))
        .route("/api/entities",         patch(update_config))
        .route("/api/entities/custom",  post(add_custom))
        .route("/api/entities/custom",  delete(delete_custom))
        .route("/api/entities/detect",  post(detect))
        .route("/api/entities/extract", post(extract))
        .route("/api/entities/mask",    post(mask))
        .route("/api/entities/distill", post(distill))
}

// ─── Discovery ────────────────────────────────────────────────────────────────

/// List all available built-in entity patterns, grouped by category.
/// Includes which are recommended for the default preset.
pub async fn list_builtin() -> Json<serde_json::Value> {
    let categories: Vec<serde_json::Value> = all_categories().iter().map(|cat| {
        let patterns: Vec<serde_json::Value> = BUILTIN_PATTERNS.iter()
            .filter(|p| p.category == *cat)
            .map(|p| serde_json::json!({
                "label": p.label,
                "display_name": p.display_name,
                "description": p.description,
                "recommended": p.recommended,
                "regex_patterns": p.regex_patterns,
                "context_phrases": p.context_phrases,
            }))
            .collect();
        serde_json::json!({
            "category": cat,
            "patterns": patterns,
        })
    }).collect();

    Json(serde_json::json!({
        "total": BUILTIN_PATTERNS.len(),
        "categories": categories,
    }))
}

// ─── Per-namespace config ─────────────────────────────────────────────────────

/// Get the entity config for the current namespace. Returns `null` enabled
/// state if the layer is disabled (no config saved).
pub async fn get_config(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = routers.get(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("namespace '{}' not found", app_id)))?;

    let config = router.entity_config().cloned();
    Ok(Json(serde_json::json!({
        "enabled": config.is_some(),
        "config": config,
    })))
}

#[derive(serde::Deserialize)]
pub struct UpdateConfigRequest {
    /// `Some(true)` to enable entity layer (using the recommended preset if
    /// no `enabled_builtins` is provided), `Some(false)` to disable, omit
    /// to leave the on/off state unchanged.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Replace the list of enabled built-in pattern labels.
    #[serde(default)]
    pub enabled_builtins: Option<Vec<String>>,
    /// Apply the recommended preset (overrides enabled_builtins).
    #[serde(default)]
    pub use_recommended: Option<bool>,
}

pub async fn update_config(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<UpdateConfigRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("namespace '{}' not found", app_id)))?;

    let mut current = router.entity_config().cloned();

    // Handle enable/disable first.
    if let Some(enabled) = req.enabled {
        if !enabled {
            router.set_entity_config(None);
            maybe_persist(&state, &app_id, router);
            return Ok(Json(serde_json::json!({
                "enabled": false,
                "config": serde_json::Value::Null,
            })));
        } else if current.is_none() {
            current = Some(EntityConfig::recommended());
        }
    }

    let mut config = current.unwrap_or_else(EntityConfig::recommended);

    if req.use_recommended == Some(true) {
        let custom = config.custom.clone();
        config = EntityConfig::recommended();
        config.custom = custom;
    } else if let Some(builtins) = req.enabled_builtins {
        // Validate every label exists in the registry.
        for label in &builtins {
            if get_builtin(label).is_none() {
                return Err((StatusCode::BAD_REQUEST,
                    format!("unknown built-in pattern label: '{}'", label)));
            }
        }
        config.enabled_builtins = builtins;
    }

    router.set_entity_config(Some(config.clone()));
    maybe_persist(&state, &app_id, router);

    Ok(Json(serde_json::json!({
        "enabled": true,
        "config": config,
    })))
}

// ─── Custom entity CRUD ────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct AddCustomRequest {
    pub label: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub description: String,
    pub regex_patterns: Vec<String>,
    #[serde(default)]
    pub context_phrases: Vec<String>,
    #[serde(default)]
    pub examples: Vec<String>,
    #[serde(default = "default_source_manual")]
    pub source: String,
}

fn default_source_manual() -> String { "manual".to_string() }

pub async fn add_custom(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddCustomRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    // Validate every regex compiles before saving — better to reject at
    // save time than silently drop at runtime.
    for pat in &req.regex_patterns {
        regex::Regex::new(pat).map_err(|e| (StatusCode::BAD_REQUEST,
            format!("regex {:?} won't compile: {}", pat, e)))?;
    }

    let entity = CustomEntity {
        label: req.label.clone(),
        display_name: if req.display_name.is_empty() { req.label.clone() } else { req.display_name },
        description: req.description,
        regex_patterns: req.regex_patterns,
        context_phrases: req.context_phrases.into_iter().map(|s| s.to_lowercase()).collect(),
        examples: req.examples,
        source: req.source,
    };

    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("namespace '{}' not found", app_id)))?;

    let mut config = router.entity_config().cloned()
        .unwrap_or_else(EntityConfig::recommended);

    // Replace existing entity with same label, otherwise append.
    config.custom.retain(|c| c.label != req.label);
    config.custom.push(entity.clone());

    router.set_entity_config(Some(config));
    maybe_persist(&state, &app_id, router);

    Ok(Json(serde_json::json!({"added": req.label, "entity": entity})))
}

#[derive(serde::Deserialize)]
pub struct DeleteCustomRequest {
    pub label: String,
}

pub async fn delete_custom(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DeleteCustomRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("namespace '{}' not found", app_id)))?;

    let mut config = router.entity_config().cloned()
        .ok_or_else(|| (StatusCode::NOT_FOUND, "entity layer not enabled".to_string()))?;

    let before = config.custom.len();
    config.custom.retain(|c| c.label != req.label);
    if config.custom.len() == before {
        return Err((StatusCode::NOT_FOUND, format!("custom entity '{}' not found", req.label)));
    }

    router.set_entity_config(Some(config));
    maybe_persist(&state, &app_id, router);

    Ok(Json(serde_json::json!({"deleted": req.label})))
}

// ─── Detection / extraction / masking ─────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct EntityOpRequest {
    pub text: String,
}

/// Get a layer for entity operations. Prefer the namespace's cached layer;
/// fall back to building a "recommended" layer if the namespace has no
/// config (so /api/entities/detect works even before the user enables the
/// layer for that namespace).
fn build_layer_for(state: &AppState, app_id: &str) -> Result<EntityLayer, (StatusCode, String)> {
    let routers = state.routers.read().unwrap();
    let router = routers.get(app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("namespace '{}' not found", app_id)))?;
    if let Some(layer) = router.entity_layer() {
        // Cached layer can't be borrowed past the lock release; rebuild from
        // the cached config (still fast — config is small) for the response.
        // For higher-throughput needs, expose a borrowed-layer API later.
        if let Some(cfg) = router.entity_config() {
            return Ok(cfg.build_layer());
        }
        let _ = layer; // silence unused warning when fallback path taken
    }
    Ok(EntityConfig::recommended().build_layer())
}

pub async fn detect(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<EntityOpRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let layer = build_layer_for(&state, &app_id)?;
    let t0 = std::time::Instant::now();
    let labels = layer.detect(&req.text);
    let elapsed_us = t0.elapsed().as_micros();
    Ok(Json(serde_json::json!({
        "text": req.text,
        "labels": labels,
        "latency_us": elapsed_us as u64,
    })))
}

pub async fn extract(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<EntityOpRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let layer = build_layer_for(&state, &app_id)?;
    let t0 = std::time::Instant::now();
    let extracted = layer.extract(&req.text);
    let elapsed_us = t0.elapsed().as_micros();
    Ok(Json(serde_json::json!({
        "text": req.text,
        "entities": extracted,
        "latency_us": elapsed_us as u64,
    })))
}

#[derive(serde::Deserialize)]
pub struct MaskRequest {
    pub text: String,
    /// Format string for the placeholder. Use `{label}` for the entity label.
    /// Default: `<{label}>` → `<EMAIL>`, `<CC>`, etc.
    #[serde(default = "default_placeholder")]
    pub placeholder: String,
}

fn default_placeholder() -> String { "<{label}>".to_string() }

pub async fn mask(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<MaskRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let layer = build_layer_for(&state, &app_id)?;
    let t0 = std::time::Instant::now();
    let template = req.placeholder.clone();
    let masked = layer.mask(&req.text, |label| template.replace("{label}", label));
    let elapsed_us = t0.elapsed().as_micros();
    Ok(Json(serde_json::json!({
        "original": req.text,
        "masked": masked,
        "latency_us": elapsed_us as u64,
    })))
}

// ─── LLM distillation ─────────────────────────────────────────────────────────
//
// POST /api/entities/distill — sends an entity description to the LLM, runs
// auto-repair on the response (drops regexes that won't compile or that don't
// match their own examples), and returns the proposed entity for the user
// to review. Does NOT save — caller must POST to /api/entities/custom to
// persist.

#[derive(serde::Deserialize)]
pub struct DistillRequest {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub examples: Vec<String>,
}

const DISTILL_SYSTEM_PROMPT: &str = "You design pattern-based detection rules for an embedded \
intent-routing engine called MicroResolve. The engine has no LLM at runtime — only the patterns \
you produce. Return strict JSON with: label, regex_patterns, context_phrases, examples, notes.\n\n\
Critical rules:\n\
- regex_patterns match the entity VALUE alone — NOT surrounding context.\n\
  Bad:  \\bpassport\\s+number\\s+[A-Z][0-9]{8}\\b   (this is a context+value regex)\n\
  Good: \\b[A-Z][0-9]{8}\\b                          (this matches the value alone)\n\
- Use \\b word boundaries.\n\
- Be specific — broad regexes like \\d{8,} cause false positives.\n\
- Each regex MUST match at least one of the examples below. Verify mentally.\n\
- context_phrases go in their own field (Aho-Corasick layer), NOT in regex.\n\
- Each example must be matched by at least one regex.\n\
- Never include real PII; generate plausible synthetic data only.\n\n\
Return ONLY the JSON object. No prose, no markdown, no code fences.";

pub async fn distill(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DistillRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let _app_id = app_id_from_headers(&headers); // not used for storage; just for logging
    if req.description.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "description is required".to_string()));
    }

    let mut user_msg = format!("Entity name: {}\nEntity description: {}",
        req.name, req.description);
    if !req.examples.is_empty() {
        user_msg.push_str(&format!("\nUser-provided examples: {:?}", req.examples));
    }

    // call_llm takes a single prompt; we concatenate the system rules and the user message.
    // Haiku follows the leading rule block reliably for structured-output tasks.
    let combined_prompt = format!("{}\n\n=== Request ===\n{}", DISTILL_SYSTEM_PROMPT, user_msg);
    let response = crate::pipeline::call_llm(&state, &combined_prompt, 2500).await
        .map_err(|(code, msg)| (code, format!("LLM call failed: {}", msg)))?;

    let json_str = crate::pipeline::extract_json(&response);
    let mut parsed: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::BAD_GATEWAY,
            format!("LLM returned invalid JSON: {}\nResponse: {}", e,
                &response[..response.len().min(400)])))?;

    // Extract fields.
    let label = parsed.get("label").and_then(|v| v.as_str())
        .unwrap_or(&req.name).to_string();
    let regex_patterns: Vec<String> = parsed.get("regex_patterns")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let context_phrases: Vec<String> = parsed.get("context_phrases")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_lowercase())).collect())
        .unwrap_or_default();
    let examples: Vec<String> = parsed.get("examples")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    // ── Auto-repair pass ──────────────────────────────────────────────────────
    // 1) drop regexes that won't compile
    // 2) drop regexes that match none of the examples
    let mut surviving_patterns: Vec<String> = Vec::new();
    let mut repairs: Vec<String> = Vec::new();
    for pat in &regex_patterns {
        match regex::Regex::new(pat) {
            Err(e) => {
                repairs.push(format!("DROPPED (won't compile): {} -- {}", pat, e));
            }
            Ok(rx) => {
                let matches_any = examples.iter().any(|ex| rx.is_match(ex));
                if matches_any {
                    surviving_patterns.push(pat.clone());
                } else {
                    repairs.push(format!("DROPPED (matches no examples): {}", pat));
                }
            }
        }
    }

    // Attach repairs and survivor count to the response.
    parsed["regex_patterns"] = serde_json::json!(surviving_patterns);
    parsed["label"] = serde_json::json!(label);
    parsed["context_phrases"] = serde_json::json!(context_phrases);
    parsed["examples"] = serde_json::json!(examples);

    Ok(Json(serde_json::json!({
        "proposed": parsed,
        "repairs": repairs,
        "usable": !surviving_patterns.is_empty(),
        "next_step": "POST /api/entities/custom to save (review fields first)",
    })))
}
