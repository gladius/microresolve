//! Intent management endpoints.

use crate::state::*;
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::{get, patch, post},
    Json,
};
use microresolve::IntentType;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/intents", get(list_intents).post(add_intent))
        .route(
            "/api/intents/{id}",
            patch(patch_intent).delete(delete_intent_by_id),
        )
        .route(
            "/api/intents/{id}/phrases",
            post(add_phrase_to_intent).delete(remove_phrase_from_intent),
        )
}

// ── New RESTful handlers ────────────────────────────────────────────────────

/// Partial update of an intent. Any subset of fields may be provided.
#[derive(serde::Deserialize)]
pub struct PatchIntentRequest {
    #[serde(default)]
    pub intent_type: Option<IntentType>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub persona: Option<String>,
    #[serde(default)]
    pub guardrails: Option<Vec<String>>,
    #[serde(default)]
    pub target: Option<microresolve::IntentTarget>,
}

pub async fn patch_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PatchIntentRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let h = state.engine.try_namespace(&app_id).ok_or((
        StatusCode::NOT_FOUND,
        format!("namespace '{}' not found", app_id),
    ))?;

    let edit = microresolve::IntentEdit {
        intent_type: req.intent_type,
        description: req.description,
        instructions: req.instructions,
        persona: req.persona,
        guardrails: req.guardrails,
        target: req.target,
        ..Default::default()
    };
    h.update_intent(&id, edit)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    maybe_commit(&state, &app_id);
    Ok(StatusCode::NO_CONTENT)
}

pub async fn delete_intent_by_id(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let Some(h) = state.engine.try_namespace(&app_id) else {
        return StatusCode::NOT_FOUND;
    };
    h.remove_intent(&id);
    maybe_commit(&state, &app_id);
    StatusCode::NO_CONTENT
}

#[derive(serde::Deserialize)]
pub struct PhrasePayload {
    pub phrase: String,
    #[serde(default = "default_lang")]
    pub lang: String,
}

pub fn default_lang() -> String {
    "en".to_string()
}

pub async fn add_phrase_to_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PhrasePayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let h = state
        .engine
        .try_namespace(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;

    let exists = h.training(&id).is_some();
    if !exists {
        return Err((StatusCode::NOT_FOUND, format!("intent '{}' not found", id)));
    }

    let result = h.add_phrase(&id, &req.phrase, &req.lang);

    if result.added {
        maybe_commit(&state, &app_id);
    }

    let counts: std::collections::HashMap<String, usize> = h
        .training_by_lang(&id)
        .map(|m| {
            m.iter()
                .map(|(lang, ps)| (lang.clone(), ps.len()))
                .collect()
        })
        .unwrap_or_default();

    Ok(Json(serde_json::json!({
        "added": result.added,
        "counts": counts,
        "redundant": result.redundant,
        "reason": result.warning,
    })))
}

pub async fn remove_phrase_from_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PhrasePayload>,
) -> Result<StatusCode, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let h = state
        .engine
        .try_namespace(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
    let removed = h.remove_phrase(&id, &req.phrase);
    if removed {
        maybe_commit(&state, &app_id);
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((StatusCode::NOT_FOUND, "phrase not found".to_string()))
    }
}

pub async fn list_intents(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    ensure_app(&state, &app_id);
    let h = state.engine.namespace(&app_id);
    let mut ids = h.intent_ids();
    ids.sort();
    let intents: Vec<serde_json::Value> = ids
        .iter()
        .filter_map(|id| h.intent(id))
        .map(|info| {
            let seeds: Vec<String> = info.training.values().flatten().cloned().collect();
            serde_json::json!({
                "id": info.id,
                "description": info.description,
                "phrases": seeds,
                "phrases_by_lang": info.training,
                "learned_count": 0usize,
                "intent_type": info.intent_type,
                "instructions": info.instructions,
                "persona": info.persona,
                "source": info.source,
                "target": info.target,
                "schema": info.schema,
                "guardrails": info.guardrails,
            })
        })
        .collect();
    Json(serde_json::json!(intents))
}

/// Create an intent. Accepts either a flat phrase list (English) or a
/// `phrases_by_lang` map for multilingual seeding. When both are provided,
/// `phrases_by_lang` wins.
#[derive(serde::Deserialize)]
pub struct AddIntentRequest {
    id: String,
    #[serde(default)]
    phrases: Vec<String>,
    #[serde(default)]
    phrases_by_lang: Option<std::collections::HashMap<String, Vec<String>>>,
    #[serde(default)]
    intent_type: Option<IntentType>,
    #[serde(default)]
    description: Option<String>,
}

pub async fn add_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddIntentRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let h = state.engine.namespace(&app_id);

    if let Some(by_lang) = req.phrases_by_lang {
        let _ = h.add_intent(&req.id, by_lang);
    } else {
        let seed_refs: Vec<&str> = req.phrases.iter().map(|s| s.as_str()).collect();
        let _ = h.add_intent(&req.id, seed_refs.as_slice());
    }

    if req.intent_type.is_some() || req.description.is_some() {
        let _ = h.update_intent(
            &req.id,
            microresolve::IntentEdit {
                intent_type: req.intent_type,
                description: req.description.filter(|d| !d.is_empty()),
                ..Default::default()
            },
        );
    }
    maybe_commit(&state, &app_id);
    StatusCode::CREATED
}
