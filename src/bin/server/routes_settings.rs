//! Settings: reset, defaults, export/import, languages, analytics data.

use axum::{
    extract::State,
    http::{StatusCode, HeaderMap},
    routing::{get, post, delete},
    Json,
};
use std::collections::HashMap;
use microresolve::Router;
use crate::state::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/export", get(export_state))
        .route("/api/import", post(import_state))
        .route("/api/languages", get(get_languages))
        .route("/api/data/all", delete(delete_all_data))
}

// --- Export / Import ---

pub async fn export_state(State(state): State<AppState>, headers: HeaderMap) -> String {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    match routers.get(&app_id) {
        Some(router) => router.export_json(),
        None => format!("{{\"error\": \"app '{}' not found\"}}", app_id),
    }
}

pub async fn import_state(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Result<StatusCode, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let new_router =
        Router::import_json(&body).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    maybe_persist(&state, &app_id, &new_router);
    let mut routers = state.routers.write().unwrap();
    routers.insert(app_id, new_router);
    Ok(StatusCode::OK)
}

// --- Languages ---

pub async fn get_languages() -> Json<serde_json::Value> {
    let json_str = microresolve::phrase::supported_languages_json();
    let val: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_default();
    Json(val)
}

// --- Clear all data ---

pub async fn delete_all_data(State(state): State<AppState>) -> StatusCode {
    // 1. Clear all namespace directories from data_dir (intents, domains, ns metadata)
    if let Some(ref dir) = state.data_dir {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                // Skip _settings.json and logs/ dir — we handle them separately
                if name == "_settings.json" || name == "logs" { continue; }
                if p.is_dir() {
                    let _ = std::fs::remove_dir_all(&p);
                } else if p.extension().map(|e| e == "json").unwrap_or(false) {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
    }

    // 2. Clear log store (deletes .bin files and resets memory)
    state.log_store.lock().unwrap().clear_all();

    // 3. Reset routers to just default namespace
    {
        let mut routers = state.routers.write().unwrap();
        *routers = HashMap::from([("default".to_string(), Router::new())]);
    }

    // 4. Clear per-namespace review modes
    state.review_mode.write().unwrap().clear();

    // 5. Persist the fresh default namespace
    {
        let routers = state.routers.read().unwrap();
        if let Some(router) = routers.get("default") {
            maybe_persist(&state, "default", router);
        }
    }

    StatusCode::OK
}


