//! Server state, shared types, and helper functions.

use asv_router::Router;
use axum::http::HeaderMap;
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub const LOG_FILE: &str = "asv_queries.jsonl";

/// A flagged query pending review.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReviewItem {
    pub id: u64,
    pub query: String,
    pub detected: Vec<String>,
    pub flag: String,
    pub suggested_intent: Option<String>,
    pub suggested_seed: Option<String>,
    pub app_id: String,
    pub timestamp: u64,
    pub session_id: Option<String>,
}

pub struct ServerState {
    pub routers: RwLock<HashMap<String, Router>>,
    pub data_dir: Option<String>,
    pub log: Mutex<std::fs::File>,
    pub http: reqwest::Client,
    pub llm_key: Option<String>,
    pub review_queue: RwLock<Vec<ReviewItem>>,
    pub review_counter: std::sync::atomic::AtomicU64,
    pub review_mode: RwLock<String>,
}

pub type AppState = Arc<ServerState>;

pub fn open_log() -> std::fs::File {
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)
        .expect("Failed to open query log")
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn app_id_from_headers(headers: &HeaderMap) -> String {
    headers
        .get("X-App-ID")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("default")
        .to_string()
}

pub fn ensure_app(state: &AppState, app_id: &str) {
    let exists = state.routers.read().unwrap().contains_key(app_id);
    if !exists {
        state
            .routers
            .write()
            .unwrap()
            .entry(app_id.to_string())
            .or_insert_with(Router::new);
    }
}

pub fn maybe_persist(state: &ServerState, app_id: &str, router: &Router) {
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}.json", dir, app_id);
        let json = router.export_json();
        let _ = std::fs::write(&path, json);
    }
}

pub fn log_query(state: &ServerState, entry: &serde_json::Value) {
    if let Ok(mut file) = state.log.lock() {
        let _ = writeln!(file, "{}", entry);
        let _ = file.flush();
    }
}

pub fn default_lang() -> String { "en".to_string() }

