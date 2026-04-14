//! ASV Router HTTP API server.
//!
//! Run with: cargo run --bin server --features server --release
//!
//! Default: http://localhost:3001

mod state;
mod pipeline;
mod routes_core;
mod routes_intents;
mod routes_learn;
mod routes_logs;
mod routes_phrases;
mod routes_settings;
mod routes_training;
mod routes_projects;
mod routes_discovery;
mod log_store;
mod routes_review;
mod routes_import;
mod routes_connect;
mod routes_ui_settings;
mod routes_events;
mod routes_assembly;
mod routes_hebbian;
mod worker;

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
use tokio::sync::{broadcast, Notify};
use tower_http::cors::CorsLayer;

#[tokio::main]
async fn main() {
    // Load .env first so ASV_DATA_DIR can be read from it
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

    // Data directory: --data flag > ASV_DATA_DIR env > ~/.local/share/asv (default)
    let mut data_dir: Option<String> = None;
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--data" {
            if let Some(dir) = args.get(i + 1) {
                data_dir = Some(dir.clone());
            }
        }
    }
    if data_dir.is_none() {
        let dir = std::env::var("ASV_DATA_DIR").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            format!("{}/.local/share/asv", home)
        });
        std::fs::create_dir_all(&dir).ok();
        data_dir = Some(dir);
    }

    let port = std::env::var("PORT").unwrap_or_else(|_| "3001".to_string());
    let addr = format!("0.0.0.0:{}", port);

    let llm_key = std::env::var("LLM_API_KEY").ok();
    if llm_key.is_some() {
        println!("LLM API key: loaded");
    } else {
        println!("LLM API key: not set (LLM features disabled)");
    }

    // Initialize routers from namespace directories
    let mut routers = HashMap::new();
    let mut hebbian_map = HashMap::new();
    let mut intent_graph_map = HashMap::new();
    let mut ngram_map = HashMap::new();

    if let Some(ref dir) = data_dir {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with('_') { continue; }

                if path.is_dir() {
                    match Router::load_from_dir(&path) {
                        Ok(r) => {
                            println!("Loaded namespace: {}", name);
                            routers.insert(name.to_string(), r);
                        }
                        Err(e) => eprintln!("Failed to load namespace {}: {}", name, e),
                    }
                    if let Some(graph) = routes_hebbian::load_hebbian(dir, name) {
                        println!("Loaded hebbian graph: {}", name);
                        hebbian_map.insert(name.to_string(), graph);
                    }
                    if let Some(ig) = routes_hebbian::load_intent_graph(dir, name) {
                        println!("Loaded intent graph: {}", name);
                        // Build L0 n-gram index from L1 + L2 vocabulary
                        let ng = asv_router::ngram::build_for_namespace(
                            hebbian_map.get(name),
                            Some(&ig),
                        );
                        println!("Built ngram index: {} ({} terms)", name, ng.len());
                        ngram_map.insert(name.to_string(), ng);
                        intent_graph_map.insert(name.to_string(), ig);
                    }
                }
            }
        }
    }

    // Ensure default namespace exists
    routers.entry("default".to_string()).or_insert_with(Router::new);

    let log_store = LogStore::new(data_dir.as_deref());
    let ui_settings = data_dir.as_deref().map(load_ui_settings).unwrap_or_default();

    let (event_tx, _) = broadcast::channel::<state::StudioEvent>(256);
    let worker_notify = Arc::new(Notify::new());

    let state: AppState = Arc::new(ServerState {
        routers: RwLock::new(routers),
        data_dir,
        log_store: Mutex::new(log_store),
        http: reqwest::Client::new(),
        llm_key,
        review_mode: RwLock::new(HashMap::new()),
        ui_settings: RwLock::new(ui_settings),
        event_tx,
        worker_notify: worker_notify.clone(),
        hebbian: RwLock::new(hebbian_map),
        intent_graph: RwLock::new(intent_graph_map),
        ngram: RwLock::new(ngram_map),
    });

    // Spawn the background auto-learn worker
    tokio::spawn(worker::run_worker(state.clone(), worker_notify));

    let app = axum::Router::new()
        .route("/api/health", get(health))
        .route("/api/llm/status", get(llm_status))
        .route("/api/version", get(get_version))
        .merge(routes_core::routes())
        .merge(routes_intents::routes())
        .merge(routes_learn::routes())
        .merge(routes_logs::routes())
        .merge(routes_phrases::routes())
        .merge(routes_settings::routes())
        .merge(routes_training::routes())
        .merge(routes_projects::routes())
        .merge(routes_discovery::routes())
        .merge(routes_review::routes())
        .merge(routes_import::routes())
        .merge(routes_connect::routes())
        .merge(routes_ui_settings::routes())
        .merge(routes_events::routes())
        .merge(routes_assembly::routes())
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    // Serve static UI files if ui/dist exists.
    // /assets/* are content-hashed by Vite — served by ServeDir (browser can cache them).
    // All other paths are SPA routes: serve index.html with no-cache so the browser
    // always fetches the latest hash references after a rebuild.
    let app = if std::path::Path::new("ui/dist").exists() {
        use axum::response::IntoResponse;
        use axum::http::header;

        async fn spa_index() -> impl IntoResponse {
            let html = std::fs::read_to_string("ui/dist/index.html")
                .unwrap_or_else(|_| "<html><body>UI not found</body></html>".to_string());
            (
                [(header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")],
                axum::response::Html(html),
            )
        }

        app
            .nest_service("/assets", tower_http::services::ServeDir::new("ui/dist/assets"))
            .fallback(spa_index)
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
    Json(serde_json::json!({"version": version, "project_id": app_id}))
}
