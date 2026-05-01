//! Connected-mode internals for [`crate::MicroResolve`].
//!
//! When `MicroResolveConfig::server` is set, the engine pulls each subscribed
//! namespace from the server on startup, spawns a single background thread
//! that ticks every `tick_interval_secs`, and on each tick:
//!   1. Flushes the buffered log entries to `/api/ingest`
//!   2. Polls each subscribed namespace's `/api/sync?version=N` and
//!      hot-swaps the local resolver if the server has a newer version.
//!
//! All types here are `pub(crate)` — library users never see them directly;
//! they interact only with [`crate::MicroResolve`] / [`crate::NamespaceHandle`].

#![allow(clippy::duplicated_attributes)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::{Resolver, ServerConfig};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct LogEntry {
    pub query: String,
    pub app_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub detected_intents: Vec<String>,
    pub confidence: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flag: Option<String>,
    pub timestamp_ms: u64,
    pub router_version: u64,
}

/// A correction buffered for the next sync tick.
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct PendingCorrection {
    pub namespace: String,
    pub query: String,
    pub wrong_intent: String,
    pub right_intent: String,
}

/// Per-namespace sync result from `POST /api/sync/batch`.
#[derive(Debug, serde::Deserialize)]
struct BatchNsResult {
    up_to_date: bool,
    version: u64,
    #[serde(default)]
    export: Option<String>,
}

/// Top-level response from `POST /api/sync`.
#[derive(Debug, serde::Deserialize)]
struct BatchSyncResponse {
    #[serde(default)]
    namespaces: HashMap<String, BatchNsResult>,
    #[allow(dead_code)]
    logs_accepted: Option<usize>,
    #[allow(dead_code)]
    corrections_applied: Option<usize>,
}

/// Shared state between the Engine and its background sync thread.
///
/// `namespaces` is a clone of the Engine's namespace map (Arc<RwLock<...>>),
/// so the sync thread can hot-swap namespace resolvers atomically.
pub(crate) struct ConnectState {
    pub server: ServerConfig,
    pub log_buf: Arc<Mutex<Vec<LogEntry>>>,
    /// Corrections buffered for the next batch sync tick. Each correction is
    /// applied locally immediately; the server learns about them on the next tick.
    pub correction_buf: Arc<Mutex<Vec<PendingCorrection>>>,
    pub versions: Arc<RwLock<HashMap<String, u64>>>,
    pub http: reqwest::blocking::Client,
}

impl ConnectState {
    pub fn new(server: ServerConfig) -> Result<Self, crate::Error> {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| crate::Error::Connect(format!("HTTP client: {}", e)))?;
        Ok(Self {
            server,
            log_buf: Arc::new(Mutex::new(Vec::new())),
            correction_buf: Arc::new(Mutex::new(Vec::new())),
            versions: Arc::new(RwLock::new(HashMap::new())),
            http,
        })
    }

    /// Fetch the full list of namespace IDs visible on the server.
    /// Used when `ServerConfig::subscribe` is empty: the library auto-pulls
    /// every namespace the server exposes.
    pub fn list_remote_namespaces(&self) -> Result<Vec<String>, crate::Error> {
        let url = format!("{}/api/namespaces", self.server.url);
        let mut req = self.http.get(&url);
        if let Some(ref key) = self.server.api_key {
            req = req.header("X-Api-Key", key);
        }
        let resp = req
            .send()
            .map_err(|e| crate::Error::Connect(format!("list namespaces: {}", e)))?;
        if !resp.status().is_success() {
            return Err(crate::Error::Connect(format!(
                "list namespaces: HTTP {}",
                resp.status()
            )));
        }
        let arr: Vec<serde_json::Value> = resp
            .json()
            .map_err(|e| crate::Error::Connect(format!("list namespaces parse: {}", e)))?;
        Ok(arr
            .iter()
            .filter_map(|v| v.get("id").and_then(|x| x.as_str()).map(|s| s.to_string()))
            .collect())
    }

    /// Pull a single namespace from the server via the unified sync endpoint.
    /// Returns `(resolver, version)` on success, or `None` if the server has
    /// no data for this namespace yet.
    pub fn pull(&self, app_id: &str) -> Result<Option<(Resolver, u64)>, crate::Error> {
        let url = format!("{}/api/sync", self.server.url);
        let mut versions = HashMap::new();
        versions.insert(app_id.to_string(), 0u64);
        let body = serde_json::json!({
            "local_versions": versions,
            "logs": Vec::<LogEntry>::new(),
            "corrections": Vec::<PendingCorrection>::new(),
            "tick_interval_secs": self.server.tick_interval_secs,
            "library_version": format!("microresolve-rust/{}", env!("CARGO_PKG_VERSION")),
        });
        let mut req = self.http.post(&url).json(&body);
        if let Some(ref key) = self.server.api_key {
            req = req.header("X-Api-Key", key);
        }
        let resp = req
            .send()
            .map_err(|e| crate::Error::Connect(format!("pull {}: {}", app_id, e)))?;
        if !resp.status().is_success() {
            return Err(crate::Error::Connect(format!(
                "pull {}: HTTP {}",
                app_id,
                resp.status()
            )));
        }
        let parsed: BatchSyncResponse = resp
            .json()
            .map_err(|e| crate::Error::Connect(e.to_string()))?;
        match parsed.namespaces.get(app_id) {
            Some(ns) if !ns.up_to_date => {
                if let Some(json) = ns.export.as_ref() {
                    let r = Resolver::import_json(json)?;
                    self.versions
                        .write()
                        .unwrap()
                        .insert(app_id.to_string(), ns.version);
                    Ok(Some((r, ns.version)))
                } else {
                    self.versions.write().unwrap().insert(app_id.to_string(), 0);
                    Ok(None)
                }
            }
            _ => {
                self.versions.write().unwrap().insert(app_id.to_string(), 0);
                Ok(None)
            }
        }
    }

    /// Buffer a correction for the next batch sync tick.
    ///
    /// The correction is applied to the local resolver by the caller before
    /// this method is invoked. The server will learn about it on the next tick
    /// via `POST /api/sync/batch`. This avoids the stampede risk of N clients
    /// all firing synchronous HTTP pushes the moment a user clicks "correct".
    pub fn push_correct(
        &self,
        app_id: &str,
        query: &str,
        wrong_intent: &str,
        right_intent: &str,
    ) -> Result<(), crate::Error> {
        let mut buf = self.correction_buf.lock().unwrap();
        buf.push(PendingCorrection {
            namespace: app_id.to_string(),
            query: query.to_string(),
            wrong_intent: wrong_intent.to_string(),
            right_intent: right_intent.to_string(),
        });
        Ok(())
    }

    pub fn buffer_log(&self, entry: LogEntry) {
        let mut buf = self.log_buf.lock().unwrap();
        if buf.len() >= self.server.log_buffer_max && !buf.is_empty() {
            buf.remove(0); // drop-oldest
        }
        buf.push(entry);
    }
}

/// Background tick: send a single `POST /api/sync` carrying buffered
/// logs + corrections + local version map, then apply any returned exports.
///
/// Holds an `Arc<ConnectState>` and a weak handle into the MicroResolve's namespace
/// map. Runs forever; the only termination signal is the MicroResolve instance being dropped
/// (which drops the strong references and the OS reclaims the thread).
pub(crate) fn run_background<F>(state: Arc<ConnectState>, apply_pull: F)
where
    F: Fn(&str, Resolver, u64) + Send + 'static,
{
    let tick = Duration::from_secs(state.server.tick_interval_secs.max(1));
    loop {
        std::thread::sleep(tick);
        match batch_sync(&state) {
            Ok(resp) => {
                for (app_id, ns_result) in resp.namespaces {
                    if !ns_result.up_to_date {
                        if let Some(json) = ns_result.export {
                            match Resolver::import_json(&json) {
                                Ok(resolver) => {
                                    state
                                        .versions
                                        .write()
                                        .unwrap()
                                        .insert(app_id.clone(), ns_result.version);
                                    apply_pull(&app_id, resolver, ns_result.version);
                                    eprintln!(
                                        "[microresolve-connect] reloaded {} → v{}",
                                        app_id, ns_result.version
                                    );
                                }
                                Err(e) => eprintln!(
                                    "[microresolve-connect] import error {}: {}",
                                    app_id, e
                                ),
                            }
                        }
                    }
                }
            }
            Err(e) => eprintln!("[microresolve-connect] batch sync error: {}", e),
        }
    }
}

/// Fire `POST /api/sync/batch`: drain log + correction buffers, ship them
/// together with local version map, return parsed response.
/// On failure the buffers are restored (logs re-prepended, corrections re-queued).
fn batch_sync(state: &ConnectState) -> Result<BatchSyncResponse, crate::Error> {
    // Drain buffers under lock, then release before the HTTP call.
    let logs: Vec<LogEntry> = {
        let mut buf = state.log_buf.lock().unwrap();
        std::mem::take(&mut *buf)
    };
    let corrections: Vec<PendingCorrection> = {
        let mut buf = state.correction_buf.lock().unwrap();
        std::mem::take(&mut *buf)
    };
    let local_versions: HashMap<String, u64> = state.versions.read().unwrap().clone();

    let url = format!("{}/api/sync", state.server.url);
    // Self-describing: include tick interval (so server knows the client's
    // expected freshness window) and library version (so the Studio's
    // connected-clients panel can flag stale clients). Both are advisory —
    // server treats them as optional metadata.
    let body = serde_json::json!({
        "local_versions": local_versions,
        "logs": logs,
        "corrections": corrections,
        "tick_interval_secs": state.server.tick_interval_secs,
        "library_version": format!("microresolve-rust/{}", env!("CARGO_PKG_VERSION")),
    });
    let mut req = state.http.post(&url).json(&body);
    if let Some(ref key) = state.server.api_key {
        req = req.header("X-Api-Key", key);
    }
    let resp = req.send().map_err(|e| {
        // Re-queue on send failure.
        let mut log_buf = state.log_buf.lock().unwrap();
        let mut c_buf = state.correction_buf.lock().unwrap();
        let mut restored = logs.clone();
        restored.extend(log_buf.drain(..));
        restored.truncate(state.server.log_buffer_max);
        *log_buf = restored;
        let mut rc = corrections.clone();
        rc.extend(c_buf.drain(..));
        *c_buf = rc;
        crate::Error::Connect(format!("batch sync send: {}", e))
    })?;

    if !resp.status().is_success() {
        // Re-queue on HTTP error too.
        let status = resp.status();
        let mut log_buf = state.log_buf.lock().unwrap();
        let mut c_buf = state.correction_buf.lock().unwrap();
        let mut restored = logs;
        restored.extend(log_buf.drain(..));
        restored.truncate(state.server.log_buffer_max);
        *log_buf = restored;
        let mut rc = corrections;
        rc.extend(c_buf.drain(..));
        *c_buf = rc;
        return Err(crate::Error::Connect(format!("batch sync HTTP {}", status)));
    }

    resp.json()
        .map_err(|e| crate::Error::Connect(format!("batch sync parse: {}", e)))
}

pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
