//! Server state, shared types, and helper functions.

use microresolve::Router;
use axum::http::HeaderMap;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, Notify};
use crate::log_store::{LogStore, LogRecord};

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
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            selected_namespace_id: "default".to_string(),
            selected_domain: String::new(),
            threshold: 0.3,
            languages: vec!["en".to_string()],
            review_skip_threshold: 0.0,
        }
    }
}

fn default_namespace() -> String { "default".to_string() }
fn default_threshold() -> f32 { 0.3 }
fn default_languages() -> Vec<String> { vec!["en".to_string()] }
fn default_review_skip_threshold() -> f32 { 0.0 }

/// Events broadcast to SSE subscribers (Studio page live feed).
#[derive(Clone, serde::Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StudioEvent {
    ItemQueued   { id: u64, query: String, app_id: String, flag: Option<String> },
    LlmStarted   { id: u64, query: String },
    LlmDone      { id: u64, correct: Vec<String>, wrong: Vec<String>, phrases_added: usize, summary: String },
    FixApplied   { id: u64, phrases_added: usize, phrases_replaced: usize, version_before: u64, version_after: u64 },
    Escalated    { id: u64, reason: String },
}

pub struct ServerState {
    pub routers: RwLock<HashMap<String, Router>>,
    pub data_dir: Option<String>,
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
    /// Global L1 base graph loaded from data/l1_base.json (WordNet + ConceptNet).
    /// Merged into every namespace's Router.l1 — new namespaces get it on first access.
    pub l1_base: Option<microresolve::scoring::LexicalGraph>,
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

pub fn ensure_app(state: &AppState, app_id: &str) {
    let exists = state.routers.read().unwrap().contains_key(app_id);
    if !exists {
        state.routers.write().unwrap()
            .entry(app_id.to_string())
            .or_insert_with(Router::new);
    }
}

pub fn load_ui_settings(data_dir: &str) -> UiSettings {
    let path = format!("{}/_settings.json", data_dir);
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_ui_settings(state: &ServerState) {
    let Some(ref dir) = state.data_dir else { return };
    let settings = state.ui_settings.read().unwrap().clone();
    if let Ok(json) = serde_json::to_string_pretty(&settings) {
        let _ = std::fs::write(format!("{}/_settings.json", dir), json);
    }
}

/// Persist a router to its namespace directory, then git-commit if in a git repo.
pub fn maybe_persist(state: &ServerState, app_id: &str, router: &Router) {
    let Some(ref dir) = state.data_dir else { return };
    let ns_dir = Path::new(dir).join(app_id);
    if let Err(e) = router.save_to_dir(&ns_dir) {
        eprintln!("persist error for {}: {}", app_id, e);
    }
    git_commit(dir, &format!("update {}", app_id));
}

/// Fire-and-forget git commit. Only runs if data_dir is already a git repo.
fn git_commit(data_dir: &str, message: &str) {
    if !std::path::Path::new(&format!("{}/.git", data_dir)).exists() { return; }
    let dir = data_dir.to_string();
    let msg = message.to_string();
    tokio::spawn(async move {
        let _ = tokio::process::Command::new("git")
            .args(["add", "-A"])
            .current_dir(&dir)
            .output().await;
        let _ = tokio::process::Command::new("git")
            .args(["commit", "--quiet", "-m", &msg])
            .current_dir(&dir)
            .output().await;
    });
}

/// Append a query record to the log store. Returns the assigned id.
pub fn log_query(state: &ServerState, record: LogRecord) -> u64 {
    state.log_store.lock().map(|mut s| s.append(record)).unwrap_or(0)
}

/// Get review mode for a namespace. Returns "manual" if not set.
pub fn get_ns_mode(state: &ServerState, app_id: &str) -> String {
    state.review_mode.read().unwrap()
        .get(app_id)
        .cloned()
        .unwrap_or_else(|| "manual".to_string())
}

pub fn default_lang() -> String { "en".to_string() }
