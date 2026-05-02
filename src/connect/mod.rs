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

/// Per-namespace sync result from `POST /api/sync`.
#[derive(Debug, serde::Deserialize)]
struct BatchNsResult {
    #[serde(default)]
    up_to_date: bool,
    version: u64,
    /// Delta ops — present when the server can cover the gap from the client's version.
    #[serde(default)]
    ops: Option<Vec<crate::oplog::Op>>,
    /// Set to true when the client is too far behind for delta; must call `/api/snapshot`.
    #[serde(default)]
    cold_start_required: bool,
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

/// Shared state between the `MicroResolve` engine and its background sync thread.
///
/// `namespaces` is a clone of the `MicroResolve` namespace map (Arc<RwLock<...>>),
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

    /// Fetch a full snapshot for the given namespace IDs via `POST /api/snapshot`.
    /// Returns a map of namespace_id → (Resolver, version).
    /// Namespaces absent from the server response are not included in the result.
    pub fn fetch_snapshot(
        &self,
        ns_ids: &[String],
    ) -> Result<HashMap<String, (Resolver, u64)>, crate::Error> {
        let url = format!("{}/api/snapshot", self.server.url);
        let body = serde_json::json!({ "namespace_ids": ns_ids });
        let mut req = self.http.post(&url).json(&body);
        if let Some(ref key) = self.server.api_key {
            req = req.header("X-Api-Key", key);
        }
        let resp = req
            .send()
            .map_err(|e| crate::Error::Connect(format!("snapshot: {}", e)))?;
        if !resp.status().is_success() {
            return Err(crate::Error::Connect(format!(
                "snapshot: HTTP {}",
                resp.status()
            )));
        }
        #[derive(serde::Deserialize)]
        struct SnapshotNs {
            version: u64,
            export: String,
        }
        #[derive(serde::Deserialize)]
        struct SnapshotResponse {
            namespaces: HashMap<String, SnapshotNs>,
        }
        let parsed: SnapshotResponse = resp
            .json()
            .map_err(|e| crate::Error::Connect(format!("snapshot parse: {}", e)))?;
        let mut result = HashMap::new();
        for (id, ns) in parsed.namespaces {
            match Resolver::import_json(&ns.export) {
                Ok(r) => {
                    self.versions
                        .write()
                        .unwrap()
                        .insert(id.clone(), ns.version);
                    result.insert(id, (r, ns.version));
                }
                Err(e) => eprintln!("[microresolve-connect] snapshot import error {}: {}", id, e),
            }
        }
        Ok(result)
    }

    pub fn buffer_log(&self, entry: LogEntry) {
        let mut buf = self.log_buf.lock().unwrap();
        if buf.len() >= self.server.log_buffer_max && !buf.is_empty() {
            buf.remove(0); // drop-oldest
        }
        buf.push(entry);
    }
}

/// Apply a list of delta-sync ops to a resolver in one atomic write-lock acquisition.
///
/// Idempotent by construction: structural ops (add/remove) deduplicate, numeric
/// ops (WeightUpdates) overwrite to post-values.
/// Apply a list of delta-sync ops to a resolver. Delegates to the engine's
/// canonical implementation. Exposed for use by connected-mode clients.
pub fn apply_ops(resolver: &mut Resolver, ops: &[crate::oplog::Op]) -> Result<(), crate::Error> {
    // Delegate to the engine's canonical apply_ops_inner (defined in engine.rs).
    // We can't call it directly (it's private), so we replicate the match here.
    // Both stay in sync via the shared Op enum — any new variant causes a compile error.
    use crate::oplog::Op;
    for op in ops {
        match op {
            Op::IntentAdded {
                id,
                phrases_by_lang,
                ..
            } => {
                let seeds = crate::IntentSeeds::Multi(
                    phrases_by_lang
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                );
                let _ = resolver.add_intent(id, seeds);
            }
            Op::IntentRemoved { id } => resolver.remove_intent(id),
            Op::PhraseAdded {
                intent_id,
                phrase,
                lang,
            } => {
                resolver.add_phrase(intent_id, phrase, lang);
            }
            Op::PhraseRemoved { intent_id, phrase } => {
                resolver.remove_phrase(intent_id, phrase);
            }
            Op::WeightUpdates { changes } => {
                for (token, intent_id, post_weight) in changes {
                    resolver
                        .index_mut()
                        .set_weight(token, intent_id, *post_weight);
                }
            }
            Op::IntentMetadataUpdated { id, edit_json } => {
                let edit: crate::IntentEdit = serde_json::from_str(edit_json)
                    .map_err(|e| crate::Error::Parse(format!("intent edit parse: {}", e)))?;
                let _ = resolver.update_intent(id, edit);
            }
            Op::NamespaceMetadataUpdated { edit_json } => {
                let edit: crate::NamespaceEdit = serde_json::from_str(edit_json)
                    .map_err(|e| crate::Error::Parse(format!("namespace edit parse: {}", e)))?;
                let _ = resolver.update_namespace(edit);
            }
            Op::DomainDescription {
                domain,
                description,
            } => match description {
                Some(d) => resolver.set_domain_description(domain, d),
                None => resolver.remove_domain_description(domain),
            },
        }
    }
    Ok(())
}

/// Background tick: send a single `POST /api/sync` carrying buffered
/// logs + corrections + local version map, then apply any returned exports.
///
/// Holds an `Arc<ConnectState>` and a weak handle into the MicroResolve's namespace
/// map. Runs forever; the only termination signal is the MicroResolve instance being dropped
/// (which drops the strong references and the OS reclaims the thread).
///
/// `apply_pull` — called for full-export syncs: replaces the resolver wholesale.
/// `apply_delta` — called for delta syncs: applies a list of ops to the live resolver.
///   Returns `Ok(())` on success; on `Err` the version counter is NOT advanced so the
///   server will ship a fresh full export on the next tick.
pub(crate) fn run_background<F, FD>(state: Arc<ConnectState>, apply_pull: F, apply_delta: FD)
where
    F: Fn(&str, Resolver, u64) + Send + 'static,
    FD: Fn(&str, &[crate::oplog::Op], u64) -> Result<(), crate::Error> + Send + 'static,
{
    let tick = Duration::from_secs(state.server.tick_interval_secs.max(1));
    loop {
        std::thread::sleep(tick);
        match batch_sync(&state) {
            Ok(resp) => {
                // Collect namespaces that need a full snapshot.
                let mut needs_snapshot: Vec<String> = Vec::new();
                for (app_id, ns_result) in resp.namespaces {
                    if ns_result.up_to_date {
                        continue;
                    }
                    if ns_result.cold_start_required {
                        needs_snapshot.push(app_id);
                    } else if let Some(ops) = ns_result.ops {
                        // Delta path: apply ops to the live resolver.
                        // Version counter is only advanced on success; a failure leaves
                        // the counter unchanged so the server will signal cold_start_required
                        // on the next tick.
                        match apply_delta(&app_id, &ops, ns_result.version) {
                            Ok(()) => {
                                state
                                    .versions
                                    .write()
                                    .unwrap()
                                    .insert(app_id.clone(), ns_result.version);
                                eprintln!(
                                    "[microresolve-connect] delta {} → v{} ({} ops applied)",
                                    app_id,
                                    ns_result.version,
                                    ops.len()
                                );
                            }
                            Err(e) => {
                                eprintln!(
                                    "[microresolve-connect] delta apply error {} (will retry snapshot): {}",
                                    app_id, e
                                );
                                // Do NOT update version — forces snapshot on next tick.
                            }
                        }
                    }
                }
                // Fetch a single snapshot for all namespaces that need one.
                if !needs_snapshot.is_empty() {
                    match state.fetch_snapshot(&needs_snapshot) {
                        Ok(snaps) => {
                            for (id, (resolver, version)) in snaps {
                                apply_pull(&id, resolver, version);
                                eprintln!(
                                    "[microresolve-connect] snapshot reloaded {} → v{}",
                                    id, version
                                );
                            }
                        }
                        Err(e) => eprintln!("[microresolve-connect] snapshot fetch error: {}", e),
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
        // Tell the server this client can receive and apply delta ops.
        "supports_delta": true,
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
