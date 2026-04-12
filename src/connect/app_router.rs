//! AppRouter: multi-namespace router for connected mode.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::Router;
use super::types::{ConnectConfig, LogEntry, SyncResponse};
use super::sync::build_client;

pub struct AppRouter {
    apps: Arc<RwLock<HashMap<String, Arc<Router>>>>,
    versions: Arc<RwLock<HashMap<String, u64>>>,
    config: ConnectConfig,
    log_buf: Arc<Mutex<Vec<LogEntry>>>,
}

impl AppRouter {
    pub fn new(config: ConnectConfig) -> Self {
        Self {
            apps: Arc::new(RwLock::new(HashMap::new())),
            versions: Arc::new(RwLock::new(HashMap::new())),
            config,
            log_buf: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Pull initial state from server for all configured app_ids.
    pub fn init(&self) -> Result<(), String> {
        let client = build_client()?;
        for app_id in &self.config.app_ids {
            let url = format!("{}/api/sync?version=0", self.config.server_url);
            let mut req = client.get(&url).header("X-App-ID", app_id.as_str());
            if let Some(ref key) = self.config.api_key { req = req.header("X-Api-Key", key); }
            let resp = req.send().map_err(|e| format!("connect failed for {}: {}", app_id, e))?;
            if !resp.status().is_success() {
                return Err(format!("server {} for app {}", resp.status(), app_id));
            }
            let sync: SyncResponse = resp.json().map_err(|e| e.to_string())?;
            if let Some(json) = sync.export {
                let router = Router::import_json(&json)?;
                self.apps.write().unwrap().insert(app_id.clone(), Arc::new(router));
                self.versions.write().unwrap().insert(app_id.clone(), sync.version);
            } else {
                self.apps.write().unwrap().insert(app_id.clone(), Arc::new(Router::new()));
                self.versions.write().unwrap().insert(app_id.clone(), 0);
            }
        }
        Ok(())
    }

    /// Start background sync + log flush thread.
    pub fn start_background(&self) -> std::thread::JoinHandle<()> {
        let apps     = Arc::clone(&self.apps);
        let versions = Arc::clone(&self.versions);
        let config   = self.config.clone();
        let log_buf  = Arc::clone(&self.log_buf);
        std::thread::Builder::new()
            .name("asv-connect-sync".into())
            .spawn(move || super::sync::run_background(apps, versions, config, log_buf))
            .expect("failed to spawn asv-connect thread")
    }

    /// Push a log entry (non-blocking, dropped if buffer full).
    pub fn push_log(&self, entry: LogEntry) {
        let mut buf = self.log_buf.lock().unwrap();
        if buf.len() < self.config.log_buffer_max {
            buf.push(entry);
        }
    }

    /// Convenience: build and push log entry after routing.
    pub fn log_route(
        &self,
        app_id: &str,
        query: &str,
        session_id: Option<String>,
        detected_intents: Vec<String>,
        confidence: &str,
        flag: Option<String>,
    ) {
        self.push_log(LogEntry {
            query: query.to_string(),
            app_id: app_id.to_string(),
            session_id,
            detected_intents,
            confidence: confidence.to_string(),
            flag,
            timestamp_ms: now_ms(),
            router_version: self.version(app_id),
        });
    }

    pub fn version(&self, app_id: &str) -> u64 {
        self.versions.read().unwrap().get(app_id).copied().unwrap_or(0)
    }

    pub fn intent_count(&self, app_id: &str) -> usize {
        self.apps.read().unwrap().get(app_id).map(|r| r.intent_count()).unwrap_or(0)
    }

    pub fn app_ids(&self) -> Vec<String> {
        self.apps.read().unwrap().keys().cloned().collect()
    }

    fn get_router(&self, app_id: &str) -> Option<Arc<Router>> {
        self.apps.read().unwrap().get(app_id).map(Arc::clone)
    }
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}
