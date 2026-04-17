//! Learn and correct endpoints.

use axum::{
    extract::State,
    http::{StatusCode, HeaderMap},
    routing::post,
    Json,
};
use crate::state::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/learn", post(learn))
        .route("/api/correct", post(correct))
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
