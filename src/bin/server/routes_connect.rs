//! Connected-mode endpoint: a single unified sync, plus a read-only
//! roster endpoint for the Studio UI.
//!
//!   POST /api/sync — one round-trip per tick carrying the client's
//!                    buffered logs + corrections + per-namespace local
//!                    versions; server applies everything and returns
//!                    deltas for each namespace.
//!   GET  /api/connected_clients — list of currently-active library
//!                    clients (keyed by API key name). Lazy-GC'd on read.
//!
//! Gated behind the `X-Api-Key` middleware when the server has API keys
//! configured. Empty key set = open mode for local dev (no per-client
//! tracking — you'd have nothing to identify clients by).

use crate::log_store::LogRecord;
use crate::state::*;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use std::collections::HashMap;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/sync", axum::routing::post(sync))
        .route(
            "/api/connected_clients",
            axum::routing::get(connected_clients),
        )
}

/// Auth check for connected-mode endpoints — strict.
/// Returns `Err(401)` if the `X-Api-Key` header is missing or invalid.
/// Returns `Ok(name)` on success — the key's embedded label, used for
/// audit attribution and the connected-clients roster.
///
/// There is no open-mode bypass. The server bootstraps a `default` key on
/// first start (see main.rs), so a fresh install can still talk to itself
/// — the operator just has to copy the generated key into the library
/// config. Strict-by-default closes the "operator forgot to set up auth
/// and shipped logs to a public remote" failure class.
fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<String, StatusCode> {
    let provided = headers
        .get("X-Api-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    state
        .key_store
        .read()
        .unwrap()
        .validate(provided)
        .map(|(name, _scope)| name)
        .ok_or(StatusCode::UNAUTHORIZED)
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
    /// Library's polling interval in seconds. The server uses
    /// `2 × tick_interval_secs` as the freshness window for connected-
    /// client tracking. Defaults to 30 if missing.
    #[serde(default)]
    pub tick_interval_secs: Option<u32>,
    /// Library version string for telemetry ("microresolve-py/0.1.6").
    /// Surfaced in /api/connected_clients for "who's still on the old client?"
    #[serde(default)]
    pub library_version: Option<String>,
    /// v0.2.0 delta-sync: client opts in to receiving ops instead of full export.
    #[serde(default)]
    pub supports_delta: Option<bool>,
    /// v0.2.0 delta-sync: oldest op version the client can still apply.
    /// Server falls back to full export if its oplog doesn't reach this far back.
    #[serde(default)]
    pub oplog_min_version: Option<u64>,
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

    // Track this client in the connected-clients roster. Strict-auth means
    // every successful sync has an identity, so this branch is unconditional.
    {
        let mut subscribed: Vec<String> = req.local_versions.keys().cloned().collect();
        subscribed.sort();
        let entry = ConnectedClient {
            name: key_name.clone(),
            namespaces: subscribed,
            tick_interval_secs: req.tick_interval_secs.unwrap_or(30),
            library_version: req.library_version.clone(),
            last_seen_ms: now_ms(),
        };
        state
            .connected_clients
            .write()
            .unwrap()
            .insert(key_name.clone(), entry);
    }

    let source = format!("connected:{}", key_name);

    // 1. Apply corrections first so the version bump is reflected in the export.
    let mut corrections_applied: usize = 0;
    for correction in &req.corrections {
        if let Some(h) = state.engine.try_namespace(&correction.namespace) {
            if h.correct(
                &correction.query,
                &correction.wrong_intent,
                &correction.right_intent,
            )
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
    let supports_delta = req.supports_delta.unwrap_or(false);
    let oplog_min_version = req.oplog_min_version.unwrap_or(0);
    let mut namespaces = serde_json::Map::new();
    for (ns_id, local_version) in &req.local_versions {
        let entry = match state.engine.try_namespace(ns_id) {
            None => serde_json::json!({"up_to_date": true, "version": 0}),
            Some(h) => {
                let server_version = h.version();
                if server_version == *local_version {
                    serde_json::json!({"up_to_date": true, "version": server_version})
                } else if supports_delta {
                    // Try to serve delta ops.
                    let oldest = h.with_resolver(|r| r.oplog.front().map(|(v, _)| *v));
                    let client_version = *local_version;
                    let client_too_far_behind = oldest.map_or(true, |o| client_version < o);
                    if client_too_far_behind
                        || (oplog_min_version > 0 && client_version < oplog_min_version)
                    {
                        // Fall back to full export.
                        let export = h.export_json();
                        serde_json::json!({
                            "version": server_version,
                            "export": export,
                        })
                    } else {
                        let ops: Vec<serde_json::Value> = h.with_resolver(|r| {
                            r.oplog
                                .iter()
                                .filter(|(v, _)| *v > client_version && *v <= server_version)
                                .map(|(v, op)| {
                                    let mut entry = serde_json::to_value(op).unwrap_or_default();
                                    entry["version"] = serde_json::json!(*v);
                                    entry
                                })
                                .collect()
                        });
                        serde_json::json!({ "version": server_version, "ops": ops })
                    }
                } else {
                    // v0.2.0 baseline client — full export.
                    let export = h.export_json();
                    serde_json::json!({
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

// ─── Connected-clients roster (read-only) ────────────────────────────────────

/// `GET /api/connected_clients` — current set of library clients that have
/// hit `/api/sync` recently. Lazy-GC'd on read: any entry older than
/// `2 × its tick_interval_secs` is dropped before responding.
///
/// Auth-on mode only: open mode (no API keys) returns an empty list because
/// there's no identity to attribute connections to.
pub async fn connected_clients(State(state): State<AppState>) -> Json<serde_json::Value> {
    let now = now_ms();
    let mut clients = state.connected_clients.write().unwrap();
    clients.retain(|_, c| {
        let stale_after_ms = (c.tick_interval_secs as u64) * 2 * 1000;
        now.saturating_sub(c.last_seen_ms) <= stale_after_ms
    });

    let items: Vec<serde_json::Value> = clients
        .values()
        .map(|c| {
            let age_ms = now.saturating_sub(c.last_seen_ms);
            let stale_after_ms = (c.tick_interval_secs as u64) * 2 * 1000;
            let expires_in_ms = stale_after_ms.saturating_sub(age_ms);
            serde_json::json!({
                "name": c.name,
                "namespaces": c.namespaces,
                "tick_interval_secs": c.tick_interval_secs,
                "library_version": c.library_version,
                "last_seen_ms": c.last_seen_ms,
                "age_ms": age_ms,
                "expires_in_ms": expires_in_ms,
            })
        })
        .collect();

    Json(serde_json::json!({
        "count": items.len(),
        "clients": items,
    }))
}
