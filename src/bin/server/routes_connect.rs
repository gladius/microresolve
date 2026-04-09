//! Connected-mode endpoints: version sync + log ingest.
//!
//!   GET  /api/sync?version=N  — library polls for model updates
//!   POST /api/ingest          — library ships bulk query logs

use axum::{
    extract::{State, Query},
    http::HeaderMap,
    Json,
};
use crate::state::*;
use crate::log_store::LogRecord;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/sync",   axum::routing::get(sync_pull))
        .route("/api/ingest", axum::routing::post(ingest_logs))
}

// ─── Sync ────────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct SyncParams {
    #[serde(default)]
    version: u64,
}

pub async fn sync_pull(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<SyncParams>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();

    match routers.get(&app_id) {
        None => Json(serde_json::json!({"up_to_date": true, "version": 0})),
        Some(router) => {
            let server_version = router.version();
            if server_version == params.version {
                Json(serde_json::json!({"up_to_date": true, "version": server_version}))
            } else {
                Json(serde_json::json!({
                    "up_to_date": false,
                    "version": server_version,
                    "export": router.export_json(),
                }))
            }
        }
    }
}

// ─── Log ingest ──────────────────────────────────────────────────────────────

pub async fn ingest_logs(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(entries): Json<Vec<LogRecord>>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let count = entries.len();
    let mut store = state.log_store.lock().unwrap();
    for mut record in entries {
        record.app_id = app_id.clone();
        record.source = "connected".to_string();
        store.append(record);
    }
    Json(serde_json::json!({"accepted": count}))
}
