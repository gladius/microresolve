//! Connected-mode internals for [`crate::Engine`].
//!
//! When `EngineConfig::server` is set, the engine pulls each subscribed
//! namespace from the server on startup, spawns a single background thread
//! that ticks every `tick_interval_secs`, and on each tick:
//!   1. Flushes the buffered log entries to `/api/ingest`
//!   2. Polls each subscribed namespace's `/api/sync?version=N` and
//!      hot-swaps the local resolver if the server has a newer version.
//!
//! All types here are `pub(crate)` — library users never see them directly;
//! they interact only with [`crate::Engine`] / [`crate::NamespaceHandle`].

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

#[derive(Debug, serde::Deserialize)]
struct SyncResponse {
    up_to_date: bool,
    version: u64,
    #[serde(default)]
    export: Option<String>,
}

/// Shared state between the Engine and its background sync thread.
///
/// `namespaces` is a clone of the Engine's namespace map (Arc<RwLock<...>>),
/// so the sync thread can hot-swap namespace resolvers atomically.
pub(crate) struct ConnectState {
    pub server: ServerConfig,
    pub log_buf: Arc<Mutex<Vec<LogEntry>>>,
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
            versions: Arc::new(RwLock::new(HashMap::new())),
            http,
        })
    }

    /// Pull a single namespace from the server. Returns `(resolver, version)`
    /// on success, or `None` if the server has no data for this namespace yet.
    pub fn pull(&self, app_id: &str) -> Result<Option<(Resolver, u64)>, crate::Error> {
        let url = format!("{}/api/sync?version=0", self.server.url);
        let mut req = self.http.get(&url).header("X-Namespace-ID", app_id);
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
        let sync: SyncResponse = resp
            .json()
            .map_err(|e| crate::Error::Connect(e.to_string()))?;
        match sync.export {
            Some(json) => {
                let r = Resolver::import_json(&json)?;
                self.versions
                    .write()
                    .unwrap()
                    .insert(app_id.to_string(), sync.version);
                Ok(Some((r, sync.version)))
            }
            None => {
                self.versions.write().unwrap().insert(app_id.to_string(), 0);
                Ok(None)
            }
        }
    }

    /// Push an explicit correction to the server.
    pub fn push_correct(
        &self,
        app_id: &str,
        query: &str,
        wrong_intent: &str,
        right_intent: &str,
    ) -> Result<(), crate::Error> {
        let url = format!("{}/api/correct", self.server.url);
        let body = serde_json::json!({
            "query": query,
            "wrong_intent": wrong_intent,
            "right_intent": right_intent,
        });
        let mut req = self
            .http
            .post(&url)
            .header("X-Namespace-ID", app_id)
            .json(&body);
        if let Some(ref key) = self.server.api_key {
            req = req.header("X-Api-Key", key);
        }
        let resp = req
            .send()
            .map_err(|e| crate::Error::Connect(format!("server push: {}", e)))?;
        if !resp.status().is_success() {
            return Err(crate::Error::Connect(format!(
                "server returned {}",
                resp.status()
            )));
        }
        Ok(())
    }

    pub fn buffer_log(&self, entry: LogEntry) {
        let mut buf = self.log_buf.lock().unwrap();
        if buf.len() >= self.server.log_buffer_max && !buf.is_empty() {
            buf.remove(0); // drop-oldest
        }
        buf.push(entry);
    }

    pub fn version_of(&self, app_id: &str) -> u64 {
        self.versions
            .read()
            .unwrap()
            .get(app_id)
            .copied()
            .unwrap_or(0)
    }
}

/// Background tick: flush logs, then check each subscribed namespace for
/// updates and hot-swap.
///
/// Holds an `Arc<ConnectState>` and a weak handle into the Engine's namespace
/// map. Runs forever; the only termination signal is the Engine being dropped
/// (which drops the strong references and the OS reclaims the thread).
pub(crate) fn run_background<F>(state: Arc<ConnectState>, apply_pull: F)
where
    F: Fn(&str, Resolver, u64) + Send + 'static,
{
    let tick = Duration::from_secs(state.server.tick_interval_secs.max(1));
    loop {
        std::thread::sleep(tick);
        flush_logs(&state);
        // Iterate a snapshot to avoid holding the lock during HTTP calls.
        let app_ids: Vec<String> = state.versions.read().unwrap().keys().cloned().collect();
        for app_id in app_ids {
            let local_v = state.version_of(&app_id);
            match check_and_apply(&state, &app_id, local_v) {
                Ok(Some((resolver, version))) => {
                    apply_pull(&app_id, resolver, version);
                    eprintln!("[microresolve-connect] reloaded {} → v{}", app_id, version);
                }
                Ok(None) => {}
                Err(e) => eprintln!("[microresolve-connect] sync error {}: {}", app_id, e),
            }
        }
    }
}

fn check_and_apply(
    state: &ConnectState,
    app_id: &str,
    local_version: u64,
) -> Result<Option<(Resolver, u64)>, crate::Error> {
    let url = format!("{}/api/sync?version={}", state.server.url, local_version);
    let mut req = state.http.get(&url).header("X-Namespace-ID", app_id);
    if let Some(ref key) = state.server.api_key {
        req = req.header("X-Api-Key", key);
    }
    let resp = req
        .send()
        .map_err(|e| crate::Error::Connect(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(crate::Error::Connect(format!("HTTP {}", resp.status())));
    }
    let sync: SyncResponse = resp
        .json()
        .map_err(|e| crate::Error::Connect(e.to_string()))?;
    if sync.up_to_date {
        return Ok(None);
    }
    let json = sync
        .export
        .ok_or_else(|| crate::Error::Connect("no export in response".into()))?;
    let resolver = Resolver::import_json(&json)?;
    Ok(Some((resolver, sync.version)))
}

fn flush_logs(state: &ConnectState) {
    let entries: Vec<LogEntry> = {
        let mut buf = state.log_buf.lock().unwrap();
        std::mem::take(&mut *buf)
    };
    if entries.is_empty() {
        return;
    }

    let mut by_app: HashMap<String, Vec<&LogEntry>> = HashMap::new();
    for e in &entries {
        by_app.entry(e.app_id.clone()).or_default().push(e);
    }

    let mut failed: Vec<LogEntry> = Vec::new();
    for (app_id, batch) in by_app {
        let url = format!("{}/api/ingest", state.server.url);
        let mut req = state
            .http
            .post(&url)
            .header("X-Namespace-ID", &app_id)
            .json(&batch);
        if let Some(ref key) = state.server.api_key {
            req = req.header("X-Api-Key", key);
        }
        if let Err(e) = req.send() {
            eprintln!("[microresolve-connect] log flush {}: {}", app_id, e);
            failed.extend(batch.into_iter().cloned());
        }
    }

    if !failed.is_empty() {
        let mut buf = state.log_buf.lock().unwrap();
        failed.extend(buf.drain(..));
        failed.truncate(state.server.log_buffer_max);
        *buf = failed;
    }
}

pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
