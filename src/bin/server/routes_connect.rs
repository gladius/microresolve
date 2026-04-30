//! Connected-mode endpoint: a single unified sync.
//!
//!   POST /api/sync — one round-trip per tick carrying the client's
//!                    buffered logs + corrections + per-namespace local
//!                    versions; server applies everything and returns
//!                    deltas for each namespace.
//!
//! Gated behind the `X-Api-Key` middleware when the server has API keys
//! configured. Empty key set = open mode for local dev.

use crate::log_store::LogRecord;
use crate::state::*;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use std::collections::HashMap;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new().route("/api/sync", axum::routing::post(sync))
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
    let provided = headers
        .get("X-Api-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    match store.validate(provided) {
        Some(name) => Ok(Some(name)),
        None => Err(StatusCode::UNAUTHORIZED),
    }
}

// ─── Unified sync ────────────────────────────────────────────────────────────

/// A pending correction carried in a sync request.
#[derive(serde::Deserialize)]
pub struct BatchCorrection {
    pub namespace: String,
    pub query: String,
    pub wrong_intent: String,
    pub right_intent: String,
}

/// A log entry as sent by a connected client in a batch sync request.
/// Fields mirror `LogEntry` in `connect/mod.rs`; `id` and `source` are
/// assigned server-side and must not be trusted from the client.
#[derive(serde::Deserialize)]
pub struct BatchLogEntry {
    pub query: String,
    pub app_id: String,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub detected_intents: Vec<String>,
    #[serde(default = "default_confidence")]
    pub confidence: String,
    pub timestamp_ms: u64,
    #[serde(default)]
    pub router_version: u64,
}

fn default_confidence() -> String {
    "none".to_string()
}

/// Request body for `POST /api/sync`.
#[derive(serde::Deserialize)]
pub struct SyncBatchRequest {
    /// Map of namespace_id → local model version the client currently holds.
    #[serde(default)]
    pub local_versions: HashMap<String, u64>,
    /// Buffered query log entries since the last tick.
    #[serde(default)]
    pub logs: Vec<BatchLogEntry>,
    /// Buffered explicit corrections since the last tick.
    #[serde(default)]
    pub corrections: Vec<BatchCorrection>,
}

/// Unified single-round-trip sync.
///
/// Processing order:
///   1. Apply corrections (so version advances before export)
///   2. Ingest query logs (woken into the auto-learn worker)
///   3. For each namespace in `local_versions`, compute delta and include export if stale
pub async fn sync(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SyncBatchRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let key_name = check_auth(&state, &headers)?;
    let source = match key_name {
        Some(name) => format!("connected:{}", name),
        None => "connected".to_string(),
    };

    // 1. Apply corrections first so the version bump is reflected in the export.
    let mut corrections_applied: usize = 0;
    for correction in &req.corrections {
        if let Some(h) = state.engine.try_namespace(&correction.namespace) {
            if h.with_resolver_mut(|r| {
                r.correct(
                    &correction.query,
                    &correction.wrong_intent,
                    &correction.right_intent,
                )
            })
            .is_ok()
            {
                corrections_applied += 1;
                maybe_commit(&state, &correction.namespace);
            }
        }
    }

    // 2. Ingest query logs.
    let logs_accepted = req.logs.len();
    if !req.logs.is_empty() {
        let mut store = state.log_store.lock().unwrap();
        for entry in req.logs {
            let record = LogRecord {
                id: 0, // assigned by log store
                query: entry.query,
                app_id: entry.app_id,
                detected_intents: entry.detected_intents,
                confidence: entry.confidence,
                session_id: entry.session_id,
                timestamp_ms: entry.timestamp_ms,
                router_version: entry.router_version,
                source: source.clone(),
            };
            store.append(record);
        }
        drop(store);
        state.worker_notify.notify_one();
    }

    // 3. Compute per-namespace deltas.
    let mut namespaces = serde_json::Map::new();
    for (ns_id, local_version) in &req.local_versions {
        let entry = match state.engine.try_namespace(ns_id) {
            None => serde_json::json!({"up_to_date": true, "version": 0}),
            Some(h) => {
                let server_version = h.with_resolver(|r| r.version());
                if server_version == *local_version {
                    serde_json::json!({"up_to_date": true, "version": server_version})
                } else {
                    let export = h.with_resolver(|r| r.export_json());
                    serde_json::json!({
                        "up_to_date": false,
                        "version": server_version,
                        "export": export,
                    })
                }
            }
        };
        namespaces.insert(ns_id.clone(), entry);
    }

    Ok(Json(serde_json::json!({
        "namespaces": namespaces,
        "logs_accepted": logs_accepted,
        "corrections_applied": corrections_applied,
    })))
}
