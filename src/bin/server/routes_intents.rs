//! Intent management endpoints.

use axum::{
    extract::{State, Query},
    http::{StatusCode, HeaderMap},
    routing::{get, post, delete},
    Json,
};
use std::collections::HashMap;
use asv_router::{Router, IntentType};
use crate::state::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/intents", get(list_intents))
        .route("/api/intents", post(add_intent))
        .route("/api/intents/delete", post(delete_intent))
        .route("/api/intents/phrase", post(add_phrase))
        .route("/api/intents/phrase/remove", post(remove_phrase))
        .route("/api/intents/multilingual", post(add_intent_multilingual))
        .route("/api/intents/type", post(set_intent_type))
        .route("/api/intents/description", post(set_intent_description))
}

pub async fn list_intents(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    ensure_app(&state, &app_id);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!([])),
    };
    let mut ids = router.intent_ids();
    ids.sort();
    let intents: Vec<serde_json::Value> = ids
        .iter()
        .map(|id| {
            let seeds = router.get_training(id).unwrap_or_default();
            let by_lang = router.get_training_by_lang(id).cloned().unwrap_or_default();
            let learned = 0usize; // routing vectors removed; count always 0
            let intent_type = router.get_intent_type(id);
            let metadata = router.get_metadata(id).cloned().unwrap_or_default();
            let description = router.get_description(id);
            serde_json::json!({
                "id": id,
                "description": description,
                "phrases": seeds,
                "phrases_by_lang": by_lang,
                "learned_count": learned,
                "intent_type": intent_type,
                "metadata": metadata,
            })
        })
        .collect();
    Json(serde_json::json!(intents))
}

#[derive(serde::Deserialize)]
pub struct AddIntentRequest {
    id: String,
    #[serde(default)]
    phrases: Vec<String>,
    #[serde(default)]
    intent_type: Option<IntentType>,
    #[serde(default)]
    metadata: Option<HashMap<String, Vec<String>>>,
}

pub async fn add_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddIntentRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.entry(app_id.clone()).or_insert_with(Router::new);
    let seed_refs: Vec<&str> = req.phrases.iter().map(|s| s.as_str()).collect();
    router.add_intent(&req.id, &seed_refs);
    if let Some(t) = req.intent_type {
        router.set_intent_type(&req.id, t);
    }
    if let Some(meta) = req.metadata {
        for (key, values) in meta {
            router.set_metadata(&req.id, &key, values);
        }
    }
    maybe_persist(&state, &app_id, router);
    StatusCode::CREATED
}

#[derive(serde::Deserialize)]
pub struct AddPhraseRequest {
    intent_id: String,
    phrase: String,
    #[serde(default = "default_lang")]
    lang: String,
}

pub fn default_lang() -> String { "en".to_string() }

pub async fn add_phrase(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddPhraseRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;

    if router.get_training(&req.intent_id).is_none() {
        return Err((StatusCode::NOT_FOUND, format!("intent '{}' not found", req.intent_id)));
    }

    let result = router.add_phrase_checked(&req.intent_id, &req.phrase, &req.lang);

    if result.added {
        maybe_persist(&state, &app_id, router);
    }

    let counts = router.seed_counts_by_lang(&req.intent_id);
    Ok(Json(serde_json::json!({
        "added": result.added,
        "counts": counts,
        "new_terms": result.new_terms,
        "conflicts": result.conflicts.iter().map(|c| serde_json::json!({
            "term": c.term,
            "competing_intent": c.competing_intent,
            "severity": c.severity,
        })).collect::<Vec<_>>(),
        "redundant": result.redundant,
        "reason": result.warning,
    })))
}

#[derive(serde::Deserialize)]
pub struct RemovePhraseRequest {
    intent_id: String,
    phrase: String,
}

pub async fn remove_phrase(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RemovePhraseRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
    if router.remove_phrase(&req.intent_id, &req.phrase) {
        maybe_persist(&state, &app_id, router);
        Ok(StatusCode::OK)
    } else {
        Err((StatusCode::NOT_FOUND, "phrase not found".to_string()))
    }
}

#[derive(serde::Deserialize)]
pub struct AddIntentMultilingualRequest {
    id: String,
    phrases_by_lang: HashMap<String, Vec<String>>,
    #[serde(default)]
    intent_type: Option<IntentType>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    metadata: Option<HashMap<String, Vec<String>>>,
}

pub async fn add_intent_multilingual(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddIntentMultilingualRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    ensure_app(&state, &app_id);

    // Collect (intent_id, phrase) pairs for L2 seeding before the borrow
    let l2_seeds: Vec<(String, String)> = req.phrases_by_lang.values()
        .flat_map(|phrases| phrases.iter().map(|p| (req.id.clone(), p.clone())))
        .collect();

    {
        let mut routers = state.routers.write().unwrap();
        let router = routers.get_mut(&app_id).unwrap();
        router.add_intent_multilingual(&req.id, req.phrases_by_lang);
        if let Some(t) = req.intent_type {
            router.set_intent_type(&req.id, t);
        }
        if let Some(desc) = req.description {
            if !desc.is_empty() {
                router.set_description(&req.id, &desc);
            }
        }
        if let Some(meta) = req.metadata {
            for (key, values) in meta {
                router.set_metadata(&req.id, &key, values);
            }
        }
        maybe_persist(&state, &app_id, router);
    }

    // Seed L2 (IntentGraph) so routing works immediately without bootstrap
    crate::routes_import::seed_into_l2(&state, &app_id, &l2_seeds);

    StatusCode::CREATED
}

#[derive(serde::Deserialize)]
pub struct SetIntentTypeRequest {
    intent_id: String,
    intent_type: IntentType,
}

pub async fn set_intent_type(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetIntentTypeRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.set_intent_type(&req.intent_id, req.intent_type);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
pub struct SetDescriptionRequest {
    intent_id: String,
    description: String,
}

pub async fn set_intent_description(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetDescriptionRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.set_description(&req.intent_id, &req.description);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
pub struct DeleteIntentRequest {
    id: String,
}

pub async fn delete_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DeleteIntentRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.remove_intent(&req.id);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

