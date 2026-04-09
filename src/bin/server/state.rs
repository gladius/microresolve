//! Server state, shared types, and helper functions.

use asv_router::Router;
use axum::http::HeaderMap;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::log_store::{LogStore, LogRecord};

pub struct ServerState {
    pub routers: RwLock<HashMap<String, Router>>,
    pub data_dir: Option<String>,
    pub log_store: Mutex<LogStore>,
    pub http: reqwest::Client,
    pub llm_key: Option<String>,
    pub review_mode: RwLock<String>,
}

pub type AppState = Arc<ServerState>;

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
        state.routers.write().unwrap()
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

/// Append a query record to the log store.
pub fn log_query(state: &ServerState, record: LogRecord) {
    if let Ok(mut store) = state.log_store.lock() {
        store.append(record);
    }
}

pub fn default_lang() -> String { "en".to_string() }
