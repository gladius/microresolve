//! MicroResolve HTTP API server.
//!
//! Run with: cargo run --bin server --features server --release
//!
//! Default: http://localhost:3001

mod state;
mod key_store;
mod routes_auth;
mod pipeline;
mod routes_core;
mod routes_intents;
mod routes_logs;
mod routes_phrases;
mod routes_settings;
mod routes_training;
mod routes_projects;
mod log_store;
mod routes_review;
mod routes_import;
mod routes_connect;
mod routes_ui_settings;
mod routes_events;
mod routes_hebbian;
mod routes_stopwords;
mod routes_git;
mod worker;
mod cli;
mod data_git;

use state::*;
use log_store::LogStore;
use axum::{
    extract::State,
    http::HeaderMap,
    routing::get,
    Json,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use tokio::sync::{broadcast, Notify};
use tower_http::cors::CorsLayer;

#[tokio::main]
async fn main() {
    // ─── dev-vs-distributed detection ──────────────────────────────────────
    //
    // Single signal: is `ui/dist/` sitting next to our executable?
    //
    //   • Distributed install (tarball / brew / docker / cargo-dist artifact)
    //       → ui/dist IS next to the binary (we packaged it that way)
    //       → serve the UI, auto-open a browser, IGNORE any stray .env
    //
    //   • Cargo build (`cargo run`, `cargo run --release`, `cargo install`, …)
    //       → ui/dist is NOT next to `target/…/server` or `~/.cargo/bin/`
    //       → API-only (Vite's `npm run dev` on :3000 owns the UI),
    //         no browser auto-open, AUTO-LOAD ./.env so dev env vars just work
    //
    // This is zero-config, profile-independent, and matches the actual
    // packaging convention in the release pipeline. No flag to remember.
    let ui_dist: Option<std::path::PathBuf> = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("ui/dist")))
        .filter(|p| p.exists());
    let is_distributed = ui_dist.is_some();

    // `.env` is a developer convenience. Skip it on distributed installs so
    // a stray .env in the user's CWD can't silently override their config.
    if !is_distributed {
        if let Ok(env_content) = std::fs::read_to_string(".env") {
            let mut loaded = 0;
            for line in env_content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') { continue; }
                if let Some((key, val)) = line.split_once('=') {
                    let key = key.trim();
                    let val = val.trim().trim_matches('"').trim_matches('\'');
                    if std::env::var(key).is_err() {
                        std::env::set_var(key, val);
                        loaded += 1;
                    }
                }
            }
            if loaded > 0 {
                eprintln!("(dev) loaded {} variable(s) from ./.env", loaded);
            }
        }
    }

    // Parse CLI: handles --help, --version, subcommands, flag validation.
    let parsed = <cli::Cli as clap::Parser>::parse();

    // Subcommand: interactive config setup, then exit.
    if let Some(cli::Command::Config) = parsed.command {
        if let Err(e) = cli::run_config_subcommand() {
            eprintln!("Failed to write config: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // Merge CLI flags + env vars + config file into one resolved config.
    let cfg = cli::resolve(&parsed);

    if parsed.print_config {
        cli::print_resolved(&cfg);
        return;
    }

    // Ensure data dir exists, and turn it into a git repo so namespace
    // mutations get auto-committed (history + rollback via the API).
    // The `origin` remote is configured at runtime via the UI / PUT
    // /api/settings/git, not at boot time.
    std::fs::create_dir_all(&cfg.data_dir).ok();
    data_git::ensure_repo(&cfg.data_dir);
    let data_dir: Option<String> = Some(cfg.data_dir.display().to_string());

    // Pick up an existing `origin` so the in-memory state matches the repo
    // after a restart. Best-effort: empty/no-remote stays `None`.
    let git_remote: Option<String> = std::process::Command::new("git")
        .args(["-C", &cfg.data_dir.display().to_string(), "remote", "get-url", "origin"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());

    // Propagate resolved config to env so downstream modules (pipeline.rs,
    // routes_review.rs) that read LLM_* directly continue to work.
    std::env::set_var("LLM_PROVIDER", &cfg.llm_provider);
    std::env::set_var("LLM_MODEL", &cfg.llm_model);
    if let Some(k) = cfg.llm_api_key.as_ref() {
        std::env::set_var("LLM_API_KEY", k);
    }

    let addr = format!("{}:{}", cfg.host, cfg.port);

    let llm_key = cfg.llm_api_key.clone();
    if llm_key.is_some() {
        println!("LLM API key: loaded");
    } else {
        println!("LLM API key: not set (run `microresolve config` or set LLM_API_KEY to enable training features)");
    }

    // Build Engine — loads all namespace subdirectories from data_dir.
    let engine = build_engine(data_dir.as_deref());
    for id in engine.namespaces() {
        let count = engine.namespace(&id).with_resolver(|r| r.l2().word_intent.len());
        let l0 = engine.namespace(&id).with_resolver(|r| r.l0().len());
        println!("Loaded namespace: {} (L2 words: {}, L0 terms: {})", id, count, l0);
    }

    let log_store = LogStore::new(data_dir.as_deref());
    let ui_settings = data_dir.as_deref().map(load_ui_settings).unwrap_or_default();

    // API keys for connected-mode endpoints. Empty = open mode.
    // Managed via UI (Manage → Auth Keys) and stored at
    // ~/.config/microresolve/keys.json (separate from data dir; never git-tracked).
    let key_store = key_store::KeyStore::load();
    if key_store.is_enabled() {
        println!("Connected-mode endpoints require X-Api-Key");
    } else {
        println!("Connected-mode endpoints are OPEN (no keys configured)");
    }

    let (event_tx, _) = broadcast::channel::<state::StudioEvent>(256);
    let worker_notify = Arc::new(Notify::new());

    let state: AppState = Arc::new(ServerState {
        engine,
        data_dir,
        git_remote: RwLock::new(git_remote),
        log_store: Mutex::new(log_store),
        http: reqwest::Client::new(),
        llm_key,
        review_mode: RwLock::new(HashMap::new()),
        ui_settings: RwLock::new(ui_settings),
        event_tx,
        worker_notify: worker_notify.clone(),
        key_store: std::sync::RwLock::new(key_store),
    });

    // Spawn the background auto-learn worker
    tokio::spawn(worker::run_worker(state.clone(), worker_notify));

    let app = axum::Router::new()
        .route("/api/health", get(health))
        .route("/api/llm/status", get(llm_status))
        .route("/api/version", get(get_version))
        .merge(routes_core::routes())
        .merge(routes_intents::routes())
        .merge(routes_logs::routes())
        .merge(routes_phrases::routes())
        .merge(routes_settings::routes())
        .merge(routes_training::routes())
        .merge(routes_projects::routes())
        .merge(routes_review::routes())
        .merge(routes_import::routes())
        .merge(routes_connect::routes())
        .merge(routes_auth::routes())
        .merge(routes_ui_settings::routes())
        .merge(routes_events::routes())
        .merge(routes_hebbian::routes())
        .merge(routes_stopwords::routes())
        .merge(routes_git::routes())
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    // `ui_dist` was resolved at the top of main() via binary adjacency.
    // `is_distributed` is true iff we found it. That single boolean controls
    // UI serving AND browser auto-open below — no cfg(debug_assertions), no
    // CWD fallback. The packaging convention is the signal.
    let app = if let Some(dist) = ui_dist.as_ref() {
        use axum::response::IntoResponse;
        use axum::http::header;

        // Store the index.html path in a OnceLock so the fallback handler
        // (which must be a plain fn for axum to accept) can read it.
        static UI_INDEX_PATH: std::sync::OnceLock<std::path::PathBuf> =
            std::sync::OnceLock::new();
        let _ = UI_INDEX_PATH.set(dist.join("index.html"));

        async fn spa_index() -> impl IntoResponse {
            let html = UI_INDEX_PATH
                .get()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .unwrap_or_else(|| "<html><body>UI not found</body></html>".to_string());
            (
                [(header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")],
                axum::response::Html(html),
            )
        }

        println!("UI served from: {}", dist.display());
        app
            .nest_service(
                "/assets",
                tower_http::services::ServeDir::new(dist.join("assets")),
            )
            .fallback(spa_index)
    } else {
        println!("(dev) API-only — no ui/dist next to the binary. For the UI, run `cd ui && npm run dev` (http://localhost:3000).");
        app
    };

    println!("MicroResolve server listening on {}", addr);
    if let Some(ref dir) = state.data_dir {
        println!("Data directory: {}", dir);
    }

    let listener = tokio::net::TcpListener::bind(&addr).await
        .expect(&format!("Failed to bind to {} — is the port already in use?", addr));

    // Auto-open the browser — only for distributed installs. In dev builds
    // the Vite dev server on :3000 already owns the browser tab.
    if is_distributed {
        if !cfg.no_open && !cli::looks_headless() {
            let url = format!("http://localhost:{}/", cfg.port);
            if let Err(e) = open::that_detached(&url) {
                eprintln!("(could not auto-open browser: {}. Visit {} manually.)", e, url);
            } else {
                println!("Opening browser at {}", url);
            }
        } else if cfg.no_open {
            println!("Browser auto-open disabled (--no-open).");
        } else {
            println!("Headless environment detected — not opening browser.");
        }
    }

    axum::serve(listener, app).await
        .expect("Server error");
}

async fn health() -> &'static str {
    "ok"
}

async fn llm_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let configured = state.llm_key.is_some();
    let provider = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "anthropic".to_string());
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| match provider.as_str() {
        "gemini" => "gemini-2.5-flash".to_string(),
        _ => "claude-haiku-4-5-20251001".to_string(),
    });
    let url = match provider.as_str() {
        "gemini" => format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent", model),
        "anthropic" => std::env::var("LLM_API_URL").unwrap_or_else(|_| "https://api.anthropic.com/v1/messages".to_string()),
        _ => std::env::var("LLM_API_URL").unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string()),
    };
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
    let version = state.engine.try_namespace(&app_id)
        .map(|h| h.with_resolver(|r| r.version()))
        .unwrap_or(0);
    Json(serde_json::json!({"version": version, "project_id": app_id}))
}
