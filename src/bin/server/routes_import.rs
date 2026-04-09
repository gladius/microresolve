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
            let intent_name = asv_router::import::to_snake_case(
                op.operation_id.as_deref().unwrap_or(&op.id)
            );

            // Build seeds from summary + description
            let mut seeds: Vec<String> = Vec::new();
            if let Some(ref summary) = op.summary {
                let s = summary.trim().to_lowercase();
                if !s.is_empty() { seeds.push(s); }
            }
            if !op.description.is_empty() {
                for sent in op.description.split(". ") {
                    let s = sent.trim().to_lowercase().trim_end_matches('.').to_string();
                    if s.len() > 10 && seeds.len() < 10 { seeds.push(s); }
                }
            }
            let name_lower = op.name.to_lowercase();
            if !seeds.contains(&name_lower) && !name_lower.is_empty() {
                seeds.push(name_lower);
            }
            if seeds.is_empty() { continue; }

            let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();
            router.add_intent(&intent_name, &seed_refs);

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
                let intent_name = asv_router::import::to_snake_case(
                    op.operation_id.as_deref().unwrap_or(&op.id)
                );
                format!("- {} ({}): {} — {}",
                    intent_name, op.method,
                    op.summary.as_deref().unwrap_or(&op.name),
                    if op.description.len() > 100 { &op.description[..100] } else { &op.description })
            }).collect();

            let existing_seeds: String = {
                let routers = state.routers.read().unwrap();
                if let Some(router) = routers.get(&app_id) {
                    batch.iter().map(|op| {
                        let name = asv_router::import::to_snake_case(
                            op.operation_id.as_deref().unwrap_or(&op.id)
                        );
                        let seeds = router.get_training(&name).unwrap_or_default();
                        format!("  {}: {:?}", name, seeds.iter().take(3).collect::<Vec<_>>())
                    }).collect::<Vec<_>>().join("\n")
                } else { String::new() }
            };

            let prompt = format!(
                "Generate 5 diverse seed phrases for each API operation below.\n\
                 These train a keyword-matching router. VOCABULARY DIVERSITY is critical.\n\n\
                 Operations:\n{}\n\n\
                 Current seeds (avoid duplicating these):\n{}\n\n\
                 {}\n\n\
                 For each operation, generate phrases a developer or user would say when they want this action.\n\
                 Mix: short commands, questions, and situational phrases.\n\n\
                 Respond with ONLY JSON:\n\
                 {{\"seeds_by_intent\": {{\"intent_name\": [\"seed1\", \"seed2\", \"seed3\", \"seed4\", \"seed5\"]}}}}\n",
                ops_desc.join("\n"), existing_seeds, asv_router::seed::SEED_QUALITY_RULES
            );

            if let Ok(response) = call_llm(&state, &prompt, 2000).await {
                if let Ok(seeds_json) = serde_json::from_str::<serde_json::Value>(extract_json(&response)) {
                    if let Some(sbi) = seeds_json.get("seeds_by_intent").and_then(|v| v.as_object()) {
                        let seeds_map: HashMap<String, Vec<String>> = sbi.iter()
                            .filter_map(|(k, v)| {
                                v.as_array().map(|arr| {
                                    (k.clone(), arr.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                                })
                            }).collect();

                        let result = seed_pipeline(&state, &app_id, &seeds_map, true).await;
                        total_added += result.added.len();
                        total_blocked += result.blocked.len();
                    }
                }
            }
        }
    }

    let intent_names: Vec<String> = selected_ops.iter().map(|op| {
        asv_router::import::to_snake_case(op.operation_id.as_deref().unwrap_or(&op.id))
    }).collect();

    Ok(Json(serde_json::json!({
        "title": parsed.title,
        "version": parsed.version,
        "imported": intent_names.len(),
        "seeds_added": total_added,
        "seeds_blocked": total_blocked,
        "intents": intent_names,
    })))
}


