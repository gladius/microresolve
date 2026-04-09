//! App management endpoints: list, create, delete.

use asv_router::Router;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post, delete},
    Json,
};
use crate::state::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/apps", get(list_apps))
        .route("/api/apps", post(create_app))
        .route("/api/apps", delete(delete_app))
}

async fn list_apps(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let routers = state.routers.read().unwrap();
    let apps: Vec<&String> = routers.keys().collect();
    Json(serde_json::json!(apps))
}

#[derive(serde::Deserialize)]
struct CreateAppRequest {
    app_id: String,
}

async fn create_app(
    State(state): State<AppState>,
    Json(req): Json<CreateAppRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut routers = state.routers.write().unwrap();
    if routers.contains_key(&req.app_id) {
        return Err((StatusCode::CONFLICT, format!("app '{}' already exists", req.app_id)));
    }
    let router = Router::new();
    maybe_persist(&state, &req.app_id, &router);
    routers.insert(req.app_id.clone(), router);
    Ok(Json(serde_json::json!({"created": req.app_id})))
}

#[derive(serde::Deserialize)]
struct DeleteAppRequest {
    app_id: String,
}

async fn delete_app(
    State(state): State<AppState>,
    Json(req): Json<DeleteAppRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if req.app_id == "default" {
        return Err((StatusCode::BAD_REQUEST, "cannot delete default app".to_string()));
    }
    let mut routers = state.routers.write().unwrap();
    if routers.remove(&req.app_id).is_none() {
        return Err((StatusCode::NOT_FOUND, format!("app '{}' not found", req.app_id)));
    }
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}.json", dir, req.app_id);
        let _ = std::fs::remove_file(&path);
    }
    Ok(Json(serde_json::json!({"deleted": req.app_id})))
}
