//! ASV Router HTTP API server.
//!
//! Run with: cargo run --bin server --features server --release
//!
//! Default: http://localhost:3001

mod state;
mod llm;
mod routes_core;
mod routes_intents;
mod routes_learn;
mod routes_logs;
mod routes_seeds;
mod routes_settings;
mod routes_review_prompt;
mod routes_review_llm;
mod routes_training;
mod routes_apps;
mod routes_discovery;
mod log_store;
mod routes_review;
mod routes_import;
mod routes_connect;
mod routes_situation;

use state::*;
use log_store::LogStore;
use asv_router::Router;
use axum::{
    extract::State,
    http::{StatusCode, HeaderMap},
    routing::get,
    Json,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use tower_http::cors::CorsLayer;

#[tokio::main]
async fn main() {
    // Parse --data <dir> CLI argument
    let mut data_dir: Option<String> = None;
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--data" {
            if let Some(dir) = args.get(i + 1) {
                data_dir = Some(dir.clone());
            }
        }
    }

    let port = std::env::var("PORT").unwrap_or_else(|_| "3001".to_string());
    let addr = format!("0.0.0.0:{}", port);

    // Load .env if present
    if let Ok(env_content) = std::fs::read_to_string(".env") {
        for line in env_content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if let Some((key, val)) = line.split_once('=') {
                let key = key.trim();
                let val = val.trim().trim_matches('"').trim_matches('\'');
                if std::env::var(key).is_err() {
                    std::env::set_var(key, val);
                }
            }
        }
    }

    let llm_key = std::env::var("LLM_API_KEY").ok();
    if llm_key.is_some() {
        println!("LLM API key: loaded");
    } else {
        println!("LLM API key: not set (LLM features disabled)");
    }

    // Initialize routers
    let mut routers = HashMap::new();

    // Auto-load from data directory
    if let Some(ref dir) = data_dir {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        if let Ok(json) = std::fs::read_to_string(&path) {
                            match Router::import_json(&json) {
                                Ok(r) => {
                                    println!("Loaded app: {}", stem);
                                    routers.insert(stem.to_string(), r);
                                }
                                Err(e) => eprintln!("Failed to load {}: {}", stem, e),
                            }
                        }
                    }
                }
            }
        }
    }

    // Ensure default app exists
    routers.entry("default".to_string()).or_insert_with(Router::new);

    let log_store = LogStore::new(data_dir.as_deref());

    let state: AppState = Arc::new(ServerState {
        routers: RwLock::new(routers),
        data_dir,
        log_store: Mutex::new(log_store),
        http: reqwest::Client::new(),
        llm_key,
        review_mode: RwLock::new("manual".to_string()),
    });

    let app = axum::Router::new()
        .route("/api/health", get(health))
        .route("/api/llm/status", get(llm_status))
        .route("/api/version", get(get_version))
        .merge(routes_core::routes())
        .merge(routes_intents::routes())
        .merge(routes_learn::routes())
        .merge(routes_logs::routes())
        .merge(routes_seeds::routes())
        .merge(routes_settings::routes())
        .merge(routes_review_prompt::routes())
        .merge(routes_review_llm::routes())
        .merge(routes_training::routes())
        .merge(routes_apps::routes())
        .merge(routes_discovery::routes())
        .merge(routes_review::routes())
        .merge(routes_import::routes())
        .merge(routes_connect::routes())
        .merge(routes_situation::routes())
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    // Serve static UI files if ui/dist exists
    let app = if std::path::Path::new("ui/dist").exists() {
        app.fallback_service(tower_http::services::ServeDir::new("ui/dist")
            .fallback(tower_http::services::ServeFile::new("ui/dist/index.html")))
    } else {
        app
    };

    println!("ASV Router server listening on {}", addr);
    if let Some(ref dir) = state.data_dir {
        println!("Data directory: {}", dir);
    }

    let listener = tokio::net::TcpListener::bind(&addr).await
        .expect(&format!("Failed to bind to {} — is the port already in use?", addr));
    axum::serve(listener, app).await
        .expect("Server error");
}

async fn health() -> &'static str {
    "ok"
}

async fn llm_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let configured = state.llm_key.is_some();
    let provider = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "anthropic".to_string());
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());
    let url = std::env::var("LLM_API_URL").unwrap_or_else(|_| {
        if provider == "anthropic" { "https://api.anthropic.com/v1/messages".to_string() }
        else { "https://api.openai.com/v1/chat/completions".to_string() }
    });
    Json(serde_json::json!({
        "configured": configured,
        "provider": provider,
        "model": model,
        "url": url,
    }))
}

async fn get_version(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let version = routers.get(&app_id)
        .map(|r| r.version())
        .unwrap_or(0);
    Json(serde_json::json!({"version": version, "app_id": app_id}))
}
