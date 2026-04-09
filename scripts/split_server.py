"""
Split src/bin/server.rs into modular structure at src/bin/server/
Run from project root: python3 scripts/split_server.py
"""

import os

SRC = "src/bin/server.rs"
OUT = "src/bin/server"

lines = open(SRC).readlines()
content = "".join(lines)

# We'll extract sections by line ranges based on the map above
# Each module gets: relevant structs + handler functions

def extract(start_line, end_line):
    """Extract lines (1-indexed, inclusive)"""
    return "".join(lines[start_line-1:end_line])

# === state.rs: types + helpers (lines 20-86) ===
state_rs = '''//! Server state, types, and helpers.

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
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

pub fn app_id_from_headers(headers: &HeaderMap) -> String {
    headers.get("X-App-ID")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("default")
        .to_string()
}

pub fn ensure_app(state: &AppState, app_id: &str) {
    let exists = state.routers.read().unwrap().contains_key(app_id);
    if !exists {
        state.routers.write().unwrap().entry(app_id.to_string()).or_insert_with(Router::new);
    }
}

pub fn maybe_persist(state: &ServerState, app_id: &str, router: &Router) {
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}.json", dir, app_id);
        if let Ok(json) = serde_json::to_string(&serde_json::json!({})) {
            let _ = json; // placeholder - actual persist uses router.export_json()
        }
        let json = router.export_json();
        let _ = std::fs::write(&path, &json);
    }
}

pub fn log_query(state: &ServerState, entry: &serde_json::Value) {
    if let Ok(mut file) = state.log.lock() {
        let _ = writeln!(file, "{}", entry);
        let _ = file.flush();
    }
}
'''

print(f"Read {len(lines)} lines from {SRC}")
print(f"Output directory: {OUT}")
print()
print("This script generates a MAP of what goes where.")
print("The actual split should be done carefully with proper imports.")
print()

# Map out the modules
modules = {
    "state.rs": "Types + helpers (ReviewItem, ServerState, app_id, persist, log) — ~90 lines",
    "llm.rs": "LLM call, extract_json, build_intent_descriptions, seed_pipeline, full_review, apply_review — ~600 lines",
    "handlers/routing.rs": "route, route_multi — ~200 lines",
    "handlers/intents.rs": "list_intents, add_intent, add_seed, remove_seed, multilingual, type, description, delete — ~230 lines",
    "handlers/learn.rs": "learn, correct, metadata — ~80 lines",
    "handlers/logs.rs": "get_logs, log_stats, clear_logs, check_accuracy — ~150 lines",
    "handlers/seeds.rs": "build_seed_prompt, parse_seed_response, generate_seeds — ~80 lines",
    "handlers/settings.rs": "reset, load_defaults, export, import, languages, co_occurrence, workflows, temporal, escalations, projections, similarity, review_mode — ~600 lines",
    "handlers/review.rs": "report_query, review_queue, approve, reject, fix, analyze, stats, intent_seeds — ~300 lines",
    "handlers/training.rs": "training_generate, training_run, training_review, training_apply, simulate_turn, simulate_respond — ~350 lines",
    "handlers/apps.rs": "list_apps, create_app, delete_app — ~60 lines",
    "handlers/discovery.rs": "discover, discover_apply — ~100 lines",
    "handlers/import.rs": "import_parse, import_apply — ~250 lines",
    "main.rs": "Startup, route table, .env loading — ~200 lines",
}

total = 0
for name, desc in modules.items():
    # Extract approx line count from desc
    import re
    m = re.search(r'~(\d+)', desc)
    count = int(m.group(1)) if m else 0
    total += count
    print(f"  {name:30} {desc}")

print(f"\n  Total: ~{total} lines (original: {len(lines)})")
print(f"  Overhead from imports/pub: ~{len(lines) - total} lines")
