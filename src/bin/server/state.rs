//! Server state, shared types, and helper functions.

use crate::log_store::{LogRecord, LogStore};
use axum::http::HeaderMap;
use microresolve::{MicroResolve, MicroResolveConfig};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, Notify};

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct UiSettings {
    #[serde(default = "default_namespace")]
    pub selected_namespace_id: String,
    #[serde(default)]
    pub selected_domain: String,
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default = "default_languages")]
    pub languages: Vec<String>,
    /// L2 score above which Turn 1 LLM judge is skipped (0 = always judge, 1 = never judge).
    /// When the top detected intent scores above this, routing is trusted without LLM review.
    #[serde(default = "default_review_skip_threshold")]
    pub review_skip_threshold: f32,
    /// Instance-wide registry of named models. Intent `target.model` fields
    /// reference these labels. Per-application configuration; not per-namespace.
    #[serde(default)]
    pub models: Vec<microresolve::NamespaceModel>,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            selected_namespace_id: "default".to_string(),
            selected_domain: String::new(),
            threshold: 0.3,
            languages: vec!["en".to_string()],
            review_skip_threshold: 0.0,
            models: Vec::new(),
        }
    }
}

fn default_namespace() -> String {
    "default".to_string()
}
fn default_threshold() -> f32 {
    0.3
}
fn default_languages() -> Vec<String> {
    vec!["en".to_string()]
}
fn default_review_skip_threshold() -> f32 {
    0.0
}

/// Events broadcast to SSE subscribers (Studio page live feed).
#[derive(Clone, serde::Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StudioEvent {
    ItemQueued {
        id: u64,
        query: String,
        app_id: String,
    },
    LlmStarted {
        id: u64,
        query: String,
    },
    LlmDone {
        id: u64,
        correct: Vec<String>,
        wrong: Vec<String>,
        phrases_added: usize,
        summary: String,
    },
    FixApplied {
        id: u64,
        phrases_added: usize,
        phrases_replaced: usize,
        version_before: u64,
        version_after: u64,
    },
    Escalated {
        id: u64,
        reason: String,
    },
}

/// Per-client tracking: one entry per authenticated API key currently
/// hitting `/api/sync`. Keyed by the key's `name` (extracted from the
/// `mr_<name>_<hex>` format). Open-mode (no auth keys) clients are NOT
/// tracked — by design, "who is connected" only makes sense when each
/// client is identifiable.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ConnectedClient {
    /// API key name — what the operator labelled this client when
    /// generating the key (e.g. "alex-laptop", "ci-bot", "prod-app-1").
    pub name: String,
    /// Namespaces this client is currently subscribed to (last seen in
    /// the sync request body's `local_versions` map).
    pub namespaces: Vec<String>,
    /// The library's tick interval, sent in the sync body. The server uses
    /// `2 × tick_interval_secs` as the freshness window.
    pub tick_interval_secs: u32,
    /// Library version string (e.g. "microresolve-py/0.1.6") if the
    /// client supplied one. Useful for "who's still on the old client?"
    pub library_version: Option<String>,
    /// `now_ms()` at the most recent sync request from this client.
    pub last_seen_ms: u64,
}

pub struct ServerState {
    pub engine: MicroResolve,
    pub data_dir: Option<String>,
    /// When set, every auto-commit on `data_dir` is followed by a
    /// background `git push origin HEAD` so training data syncs to a
    /// real remote. Auth is whatever git is already configured with.
    /// Wrapped in RwLock so the /api/settings/git PUT can update it live.
    pub git_remote: RwLock<Option<String>>,
    pub log_store: Mutex<LogStore>,
    pub http: reqwest::Client,
    pub llm_key: Option<String>,
    /// Per-namespace review mode: "manual" | "auto". Defaults to "manual".
    pub review_mode: RwLock<HashMap<String, String>>,
    pub ui_settings: RwLock<UiSettings>,
    /// Broadcast channel for Studio real-time feed (SSE).
    pub event_tx: broadcast::Sender<StudioEvent>,
    /// Wakes the background auto-learn worker when new items are queued.
    pub worker_notify: Arc<Notify>,
    /// API key store for connected-mode endpoints. Empty = open mode (dev/local).
    /// Persisted to ~/.config/microresolve/keys.json (NOT in data dir, NOT in git).
    pub key_store: std::sync::RwLock<crate::key_store::KeyStore>,
    /// In-memory roster of connected library clients, keyed by API key name.
    /// Updated on each `/api/sync` POST; lazy-GC'd on read in
    /// `/api/connected_clients`. Volatile across server restarts.
    pub connected_clients: RwLock<HashMap<String, ConnectedClient>>,
}

pub type AppState = Arc<ServerState>;

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Extract the active namespace ID from the `X-Namespace-ID` request header.
/// Defaults to `"default"` when the header is absent.
pub fn app_id_from_headers(headers: &HeaderMap) -> String {
    headers
        .get("X-Namespace-ID")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("default")
        .to_string()
}

/// Ensure a namespace exists in the engine, creating it lazily if absent.
pub fn ensure_app(state: &AppState, app_id: &str) {
    let _ = state.engine.namespace(app_id);
}

pub fn load_ui_settings(data_dir: &str) -> UiSettings {
    let path = format!("{}/_settings.json", data_dir);
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_ui_settings(state: &ServerState) {
    let Some(ref dir) = state.data_dir else {
        return;
    };
    let settings = state.ui_settings.read().unwrap().clone();
    if let Ok(json) = serde_json::to_string_pretty(&settings) {
        let _ = std::fs::write(format!("{}/_settings.json", dir), json);
    }
}

/// Flush the namespace via MicroResolve and git-commit. Replaces the old
/// `maybe_persist(state, app_id, &router)` pattern.
pub fn maybe_commit(state: &ServerState, app_id: &str) {
    if let Some(h) = state.engine.try_namespace(app_id) {
        if let Err(e) = h.flush() {
            eprintln!("flush error for {}: {}", app_id, e);
        }
    }
    if let Some(ref dir) = state.data_dir {
        git_commit(
            dir,
            &format!("update {}", app_id),
            state.git_remote.read().unwrap().is_some(),
        );
    }
}

/// Fire-and-forget git commit. Only runs if data_dir is already a git repo.
/// When `push` is true, follows up with `git push origin HEAD` after the
/// commit lands. Both run on the tokio pool so the caller never blocks.
pub fn git_commit(data_dir: &str, message: &str, push: bool) {
    if !std::path::Path::new(&format!("{}/.git", data_dir)).exists() {
        return;
    }
    let dir = data_dir.to_string();
    let msg = message.to_string();
    tokio::spawn(async move {
        let _ = tokio::process::Command::new("git")
            .args(["add", "-A"])
            .current_dir(&dir)
            .output()
            .await;
        let commit_out = tokio::process::Command::new("git")
            .args(["commit", "--quiet", "-m", &msg])
            .current_dir(&dir)
            .output()
            .await;
        // Skip push if the commit itself failed (e.g., nothing to commit).
        if push
            && commit_out
                .as_ref()
                .map(|o| o.status.success())
                .unwrap_or(false)
        {
            let push_out = tokio::process::Command::new("git")
                .args(["push", "--quiet", "--set-upstream", "origin", "HEAD"])
                .current_dir(&dir)
                .output()
                .await;
            if let Ok(o) = push_out {
                if !o.status.success() {
                    eprintln!(
                        "[data_git] push failed: {}",
                        String::from_utf8_lossy(&o.stderr).trim()
                    );
                }
            }
        }
    });
}

/// Build a MicroResolve instance from a data directory path (loads all existing namespaces).
/// Ensures the "default" namespace exists.
pub fn build_engine(data_dir: Option<&str>) -> MicroResolve {
    let config = MicroResolveConfig {
        data_dir: data_dir.map(std::path::PathBuf::from),
        ..Default::default()
    };
    let engine = MicroResolve::new(config).expect("failed to initialise engine");
    // Ensure default namespace exists
    let _ = engine.namespace("default");
    engine
}

/// Append a query record to the log store. Returns the assigned id.
pub fn log_query(state: &ServerState, record: LogRecord) -> u64 {
    state
        .log_store
        .lock()
        .map(|mut s| s.append(record))
        .unwrap_or(0)
}

/// Get review mode for a namespace. Returns "manual" if not set.
pub fn get_ns_mode(state: &ServerState, app_id: &str) -> String {
    state
        .review_mode
        .read()
        .unwrap()
        .get(app_id)
        .cloned()
        .unwrap_or_else(|| "manual".to_string())
}

pub fn default_lang() -> String {
    "en".to_string()
}
