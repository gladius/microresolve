//! Settings: reset, defaults, languages, analytics data.

use crate::state::*;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{delete, get},
    Json,
};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/languages", get(get_languages))
        .route("/api/data/all", delete(delete_all_data))
}

// --- Languages ---

pub async fn get_languages() -> Json<serde_json::Value> {
    let json_str = microresolve::phrase::supported_languages_json();
    let val: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_default();
    Json(val)
}

// --- Clear all data ---

pub async fn delete_all_data(State(state): State<AppState>) -> StatusCode {
    // 1. Clear all namespace directories from data_dir
    if let Some(ref dir) = state.data_dir {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name == "_settings.json" || name == "logs" {
                    continue;
                }
                if p.is_dir() {
                    let _ = std::fs::remove_dir_all(&p);
                } else if p.extension().map(|e| e == "json").unwrap_or(false) {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
    }

    // 2. Clear log store
    state.log_store.lock().unwrap().clear_all();

    // 3. Drop all namespaces then re-create default
    for id in state.engine.namespaces() {
        state.engine.remove_namespace(&id);
    }
    let _ = state.engine.namespace("default");

    // 4. Clear per-namespace review modes
    state.review_mode.write().unwrap().clear();

    // 5. Persist fresh default namespace
    maybe_commit(&state, "default");

    StatusCode::OK
}
