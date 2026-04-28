//! Connected-mode endpoints: version sync + log ingest + explicit correction.
//!
//!   GET  /api/sync?version=N  — library polls for model updates
//!   POST /api/ingest          — library ships bulk query logs
//!   POST /api/correct         — library pushes an explicit correction
//!
//! All three are gated behind the `X-Api-Key` middleware when the server has
//! API keys configured. Empty key set = open mode for local dev.

use axum::{
    extract::{State, Query},
    http::{HeaderMap, StatusCode},
    Json,
};
use crate::state::*;
use crate::log_store::LogRecord;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/sync",    axum::routing::get(sync_pull))
        .route("/api/ingest",  axum::routing::post(ingest_logs))
        .route("/api/correct", axum::routing::post(correct))
}

/// Auth check for connected-mode endpoints.
/// Returns `Err(401)` if a key is required but missing/invalid.
/// Returns `Ok(Some(key_name))` on valid key, `Ok(None)` in open mode.
/// Callers use the key name to attribute requests in audit logs.
fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<Option<String>, StatusCode> {
    let store = state.key_store.read().unwrap();
    if !store.is_enabled() {
        return Ok(None); // open mode
    }
    let provided = headers.get("X-Api-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    match store.validate(provided) {
        Some(name) => Ok(Some(name)),
        None => Err(StatusCode::UNAUTHORIZED),
    }
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
) -> Result<Json<serde_json::Value>, StatusCode> {
    let _key_name = check_auth(&state, &headers)?;
    let app_id = app_id_from_headers(&headers);

    Ok(match state.engine.try_namespace(&app_id) {
        None => Json(serde_json::json!({"up_to_date": true, "version": 0})),
        Some(h) => {
            let server_version = h.with_resolver(|r| r.version());
            if server_version == params.version {
                Json(serde_json::json!({"up_to_date": true, "version": server_version}))
            } else {
                let export = h.with_resolver(|r| r.export_json());
                Json(serde_json::json!({
                    "up_to_date": false,
                    "version": server_version,
                    "export": export,
                }))
            }
        }
    })
}

// ─── Log ingest ──────────────────────────────────────────────────────────────

pub async fn ingest_logs(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(entries): Json<Vec<LogRecord>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let key_name = check_auth(&state, &headers)?;
    let app_id = app_id_from_headers(&headers);
    let count = entries.len();
    // Source attribution: "connected:<key>" if auth was used, else "connected"
    let source = match key_name {
        Some(name) => format!("connected:{}", name),
        None => "connected".to_string(),
    };
    {
        let mut store = state.log_store.lock().unwrap();
        for mut record in entries {
            record.app_id = app_id.clone();
            record.source = source.clone();
            store.append(record);
        }
    }
    // Wake the auto-learn worker so ingested queries actually get reviewed.
    // Without this, queries sit in the log_store unprocessed.
    if count > 0 {
        state.worker_notify.notify_one();
    }
    Ok(Json(serde_json::json!({"accepted": count})))
}

// ─── Explicit correction ─────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct CorrectRequest {
    pub query: String,
    pub wrong_intent: String,
    pub right_intent: String,
}

/// Apply an explicit correction sent by a connected library.
///
/// This is the user-driven path (someone clicked "this routing was wrong").
/// LLM-driven corrections happen via the auto-learn worker against ingested
/// query logs. Both paths converge in the same `Resolver::correct()` call.
pub async fn correct(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CorrectRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let _key_name = check_auth(&state, &headers).map_err(|c| (c, "auth failed".to_string()))?;
    let app_id = app_id_from_headers(&headers);

    let h = state.engine.try_namespace(&app_id)
        .ok_or((StatusCode::NOT_FOUND, format!("namespace '{}' not found", app_id)))?;

    h.with_resolver_mut(|r| r.correct(&req.query, &req.wrong_intent, &req.right_intent))
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    let new_version = h.with_resolver(|r| r.version());
    maybe_commit(&state, &app_id);

    Ok(Json(serde_json::json!({
        "applied": true,
        "version": new_version,
    })))
}
