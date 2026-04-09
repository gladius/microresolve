//! Learn, correct, and metadata endpoints.

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
        .route("/api/learn", post(learn))
        .route("/api/correct", post(correct))
        .route("/api/metadata", post(set_metadata))
        .route("/api/metadata/get", post(get_metadata))
}

#[derive(serde::Deserialize)]
pub struct LearnRequest {
    query: String,
    intent_id: String,
}

pub async fn learn(State(state): State<AppState>, headers: HeaderMap, Json(req): Json<LearnRequest>) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.learn(&req.query, &req.intent_id);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
pub struct CorrectRequest {
    query: String,
    wrong_intent: String,
    correct_intent: String,
}

pub async fn correct(State(state): State<AppState>, headers: HeaderMap, Json(req): Json<CorrectRequest>) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.correct(&req.query, &req.wrong_intent, &req.correct_intent);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

// --- Metadata ---

#[derive(serde::Deserialize)]
pub struct SetMetadataRequest {
    intent_id: String,
    key: String,
    values: Vec<String>,
}

pub async fn set_metadata(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetMetadataRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.set_metadata(&req.intent_id, &req.key, req.values);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
pub struct GetMetadataRequest {
    intent_id: String,
}

pub async fn get_metadata(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<GetMetadataRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
    };
    let meta = router.get_metadata(&req.intent_id).cloned().unwrap_or_default();
    Json(serde_json::json!(meta))
}

