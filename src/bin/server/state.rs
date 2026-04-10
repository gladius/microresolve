//! Server state, shared types, and helper functions.

use asv_router::Router;
use axum::http::HeaderMap;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::log_store::{LogStore, LogRecord};

#[derive(serde::Serialize, serde::Deserialize, Default)]
pub struct Metadata {
    #[serde(default)]
    pub namespace_descriptions: HashMap<String, String>,
    #[serde(default)]
    pub domain_descriptions: HashMap<String, HashMap<String, String>>,
}

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
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            selected_namespace_id: "default".to_string(),
            selected_domain: String::new(),
            threshold: 0.3,
            languages: vec!["en".to_string()],
        }
    }
}

fn default_namespace() -> String { "default".to_string() }
fn default_threshold() -> f32 { 0.3 }
fn default_languages() -> Vec<String> { vec!["en".to_string()] }

pub struct ServerState {
    pub routers: RwLock<HashMap<String, Router>>,
    pub data_dir: Option<String>,
    pub log_store: Mutex<LogStore>,
    pub http: reqwest::Client,
    pub llm_key: Option<String>,
    pub review_mode: RwLock<String>,
    pub namespace_descriptions: RwLock<HashMap<String, String>>,
    pub domain_descriptions: RwLock<HashMap<String, HashMap<String, String>>>,
    pub ui_settings: RwLock<UiSettings>,
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

pub fn load_metadata(data_dir: &str) -> Metadata {
    let mut ns_descs: HashMap<String, String> = HashMap::new();
    let mut domain_descs: HashMap<String, HashMap<String, String>> = HashMap::new();

    if let Ok(entries) = std::fs::read_dir(data_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() { continue; }
            let ns_id = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) if !n.starts_with('_') => n.to_string(),
                _ => continue,
            };
            // Read namespace description
            if let Ok(json) = std::fs::read_to_string(path.join("_ns.json")) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json) {
                    if let Some(desc) = val.get("description").and_then(|d| d.as_str()) {
                        if !desc.is_empty() {
                            ns_descs.insert(ns_id.clone(), desc.to_string());
                        }
                    }
                }
            }
            // Read domain descriptions from subdirectories
            if let Ok(sub_entries) = std::fs::read_dir(&path) {
                for sub in sub_entries.flatten() {
                    let sub_path = sub.path();
                    if !sub_path.is_dir() { continue; }
                    let domain = match sub_path.file_name().and_then(|n| n.to_str()) {
                        Some(n) if !n.starts_with('_') => n.to_string(),
                        _ => continue,
                    };
                    if let Ok(json) = std::fs::read_to_string(sub_path.join("_domain.json")) {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json) {
                            if let Some(desc) = val.get("description").and_then(|d| d.as_str()) {
                                domain_descs.entry(ns_id.clone()).or_default()
                                    .insert(domain, desc.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    // Migration: fall back to old _meta.json if nothing found
    if ns_descs.is_empty() && domain_descs.is_empty() {
        let old_path = format!("{}/_meta.json", data_dir);
        if let Ok(s) = std::fs::read_to_string(&old_path) {
            if let Ok(meta) = serde_json::from_str::<Metadata>(&s) {
                return meta;
            }
        }
    }

    Metadata { namespace_descriptions: ns_descs, domain_descriptions: domain_descs }
}

pub fn save_metadata(state: &ServerState) {
    let Some(ref dir) = state.data_dir else { return };
    let ns_descs = state.namespace_descriptions.read().unwrap();
    let domain_descs = state.domain_descriptions.read().unwrap();

    for (ns_id, desc) in ns_descs.iter() {
        let ns_dir = format!("{}/{}", dir, ns_id);
        std::fs::create_dir_all(&ns_dir).ok();
        let json = serde_json::json!({"description": desc});
        let _ = std::fs::write(
            format!("{}/_ns.json", ns_dir),
            serde_json::to_string_pretty(&json).unwrap_or_default(),
        );
    }

    for (ns_id, domains) in domain_descs.iter() {
        for (domain, desc) in domains.iter() {
            let domain_dir = format!("{}/{}/{}", dir, ns_id, domain);
            std::fs::create_dir_all(&domain_dir).ok();
            let json = serde_json::json!({"description": desc});
            let _ = std::fs::write(
                format!("{}/_domain.json", domain_dir),
                serde_json::to_string_pretty(&json).unwrap_or_default(),
            );
        }
    }
}

/// Write the fast-load monolithic snapshot + human-readable per-intent files.
pub fn maybe_persist(state: &ServerState, app_id: &str, router: &Router) {
    let Some(ref dir) = state.data_dir else { return };
    let ns_dir = format!("{}/{}", dir, app_id);
    std::fs::create_dir_all(&ns_dir).ok();

    // Fast-load snapshot (loaded at startup)
    let _ = std::fs::write(format!("{}/_router.json", ns_dir), router.export_json());

    // Human-readable per-intent files (git-friendly)
    write_intent_files(&ns_dir, router);

    // Git auto-commit if data_dir is a git repo
    git_commit(dir, &format!("update {}", app_id));
}

/// Write one JSON file per intent into the namespace directory.
/// Intents with domain prefix → subdirectory. Stale files are removed.
fn write_intent_files(ns_dir: &str, router: &Router) {
    let mut written: HashSet<PathBuf> = HashSet::new();

    for intent_id in router.intent_ids() {
        let (domain_opt, name) = if let Some(pos) = intent_id.find(':') {
            (Some(&intent_id[..pos]), &intent_id[pos + 1..])
        } else {
            (None, intent_id.as_str())
        };

        let file_path = if let Some(domain) = domain_opt {
            let domain_dir = format!("{}/{}", ns_dir, domain);
            std::fs::create_dir_all(&domain_dir).ok();
            PathBuf::from(format!("{}/{}.json", domain_dir, name))
        } else {
            PathBuf::from(format!("{}/{}.json", ns_dir, name))
        };

        let intent_json = serde_json::json!({
            "description": router.get_description(&intent_id),
            "type": router.get_intent_type(&intent_id),
            "phrases": router.get_training_by_lang(&intent_id).cloned().unwrap_or_default(),
            "metadata": router.get_metadata(&intent_id).cloned().unwrap_or_default(),
            "vector": router.get_vector(&intent_id),
            "situation_patterns": router.get_situation_patterns(&intent_id).cloned().unwrap_or_default(),
        });

        if let Ok(json) = serde_json::to_string_pretty(&intent_json) {
            let _ = std::fs::write(&file_path, json);
        }
        written.insert(file_path);
    }

    // Remove stale intent files from deleted intents
    cleanup_stale_intent_files(ns_dir, &written);
}

fn cleanup_stale_intent_files(ns_dir: &str, written: &HashSet<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(ns_dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_file() && name.ends_with(".json") && !name.starts_with('_') {
            if !written.contains(&path) {
                let _ = std::fs::remove_file(&path);
            }
        } else if path.is_dir() && !name.starts_with('_') {
            let Ok(sub) = std::fs::read_dir(&path) else { continue };
            for sub_entry in sub.flatten() {
                let sub_path = sub_entry.path();
                let sub_name = sub_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if sub_path.is_file() && sub_name.ends_with(".json") && !sub_name.starts_with('_') {
                    if !written.contains(&sub_path) {
                        let _ = std::fs::remove_file(&sub_path);
                    }
                }
            }
        }
    }
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

/// Append a query record to the log store.
pub fn log_query(state: &ServerState, record: LogRecord) {
    if let Ok(mut store) = state.log_store.lock() {
        store.append(record);
    }
}

pub fn default_lang() -> String { "en".to_string() }
