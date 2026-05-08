//! MicroResolve HTTP API server.
//!
//! Run with: cargo run --bin microresolve-studio --features server --release
//!
//! Default: http://localhost:4000

mod audit_export;
mod audit_log;
mod audit_verify;
mod cli;
mod data_git;
mod key_store;
mod log_store;
mod packs;
mod pipeline;
mod routes_audit;
mod routes_auth;
mod routes_connect;
mod routes_core;
mod routes_events;
mod routes_git;
mod routes_import;
mod routes_intents;
mod routes_logs;
mod routes_phrases;
mod routes_projects;
mod routes_review;
mod routes_settings;
mod routes_state;
mod routes_stopwords;
mod routes_training;
mod routes_ui_settings;
mod state;
mod worker;

use axum::{extract::State, http::HeaderMap, routing::get, Json};
use log_store::LogStore;
use state::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use tokio::sync::{broadcast, Notify};
use tower_http::cors::CorsLayer;

#[tokio::main]
async fn main() {
    // ─── dev-vs-distributed detection ──────────────────────────────────────
    //
    // bundled-ui: assets are compiled into the binary — always distributed.
    // Default: single signal — is `ui/dist/` sitting next to our executable?
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
    #[cfg(not(feature = "bundled-ui"))]
    let ui_dist: Option<std::path::PathBuf> = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("ui/dist")))
        .filter(|p| p.exists());
    #[cfg(not(feature = "bundled-ui"))]
    let is_distributed = ui_dist.is_some();
    #[cfg(feature = "bundled-ui")]
    // Bundled mode: ui/dist is compiled in — this binary is always distributed.
    let is_distributed = true;

    // `.env` is a developer convenience. Skip it on distributed installs so
    // a stray .env in the user's CWD can't silently override their config.
    #[cfg(not(feature = "bundled-ui"))]
    if !is_distributed {
        if let Ok(env_content) = std::fs::read_to_string(".env") {
            let mut loaded = 0;
            for line in env_content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
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

    // Subcommands that exit before starting the server.
    match &parsed.command {
        Some(cli::Command::Config) => {
            if let Err(e) = cli::run_config_subcommand(&parsed) {
                eprintln!("Failed to write config: {}", e);
                std::process::exit(1);
            }
            return;
        }
        Some(cli::Command::Install { pack }) => {
            let cfg = cli::resolve(&parsed);
            if let Err(e) = packs::install(pack, &cfg.data_dir) {
                eprintln!("Install failed: {}", e);
                std::process::exit(1);
            }
            return;
        }
        Some(cli::Command::ListPacks) => {
            let cfg = cli::resolve(&parsed);
            packs::list(&cfg.data_dir);
            return;
        }
        Some(cli::Command::ExportLog {
            key,
            since,
            event_prefix,
            namespace,
        }) => {
            let cfg = cli::resolve(&parsed);
            let data_dir = cfg.data_dir.display().to_string();
            let since_ms = match since.as_deref() {
                Some(s) => match audit_export::parse_duration_ms(s) {
                    Ok(ms) => {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis() as u64)
                            .unwrap_or(0);
                        Some(now.saturating_sub(ms))
                    }
                    Err(e) => {
                        eprintln!("--since: {}", e);
                        std::process::exit(2);
                    }
                },
                None => None,
            };
            let filter = audit_export::ExportFilter {
                key: key.clone(),
                since_ms,
                event_prefix: event_prefix.clone(),
                namespace: namespace.clone(),
            };
            match audit_export::export_chains(&data_dir, &filter) {
                Ok((written, chains)) => {
                    eprintln!(
                        "exported {} entries from {} chain(s) in {}/_audit/",
                        written, chains, data_dir
                    );
                }
                Err(e) => {
                    eprintln!("export failed: {}", e);
                    std::process::exit(1);
                }
            }
            return;
        }
        Some(cli::Command::VerifyLog { key }) => {
            let cfg = cli::resolve(&parsed);
            let data_dir = cfg.data_dir.display().to_string();
            let report = audit_verify::verify_all(Some(&data_dir));
            // Pretty print, optionally filter to a single key.
            let chains: Vec<_> = report
                .chains
                .iter()
                .filter(|c| key.as_deref().map(|k| c.kid == k).unwrap_or(true))
                .collect();
            println!("Verifying audit chains in {}/_audit/", data_dir);
            for c in &chains {
                if c.ok {
                    println!(
                        "  ✓ kid={:<32} {:>8} entries  head={}",
                        c.kid,
                        c.entries,
                        if c.head_hash.is_empty() {
                            "(empty)".to_string()
                        } else {
                            format!("{}…", &c.head_hash[..c.head_hash.len().min(12)])
                        }
                    );
                } else {
                    println!(
                        "  ✗ kid={:<32} {:>8} entries  ERROR: {}",
                        c.kid,
                        c.entries,
                        c.error.as_deref().unwrap_or("unknown")
                    );
                }
            }
            let any_err = chains.iter().any(|c| !c.ok);
            let total: u64 = chains.iter().map(|c| c.entries).sum();
            if any_err {
                println!(
                    "\nFAIL: {} entries across {} chains, {} with breaks.",
                    total,
                    chains.len(),
                    chains.iter().filter(|c| !c.ok).count()
                );
                std::process::exit(1);
            } else {
                println!(
                    "\nOK: {} entries across {} chains verified.",
                    total,
                    chains.len()
                );
            }
            return;
        }
        None => {}
    }

    // Merge CLI flags + env vars + config file into one resolved config.
    let cfg = cli::resolve(&parsed);

    if parsed.print_config {
        cli::print_resolved(&cfg, &parsed);
        return;
    }

    // Ensure data dir exists, and turn it into a git repo so namespace
    // mutations get auto-committed (history + rollback via the API).
    // The `origin` remote is configured at runtime via the UI / PUT
    // /api/settings/git, not at boot time.
    std::fs::create_dir_all(&cfg.data_dir).ok();
    data_git::ensure_repo(&cfg.data_dir);
    // For data dirs created before the .gitignore protection landed: write
    // it now and untrack any _logs/* that already entered the index. Idempotent.
    data_git::migrate_existing_repo(&cfg.data_dir);
    let data_dir: Option<String> = Some(cfg.data_dir.display().to_string());

    // Pick up an existing `origin` so the in-memory state matches the repo
    // after a restart. Best-effort: empty/no-remote stays `None`.
    let git_remote: Option<String> = std::process::Command::new("git")
        .args([
            "-C",
            &cfg.data_dir.display().to_string(),
            "remote",
            "get-url",
            "origin",
        ])
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

    // Build MicroResolve — loads all namespace subdirectories from data_dir.
    let engine = build_engine(data_dir.as_deref());
    for id in engine.namespaces() {
        let count = engine.namespace(&id).vocab_size();
        println!("Loaded namespace: {} (vocab: {})", id, count);
    }

    let log_store = LogStore::new(data_dir.as_deref());
    // Audit log — server-wide default mode comes from cascade
    // (env MICRORESOLVE_AUDIT_MODE > config.toml [audit].mode > "default").
    // Per-namespace overrides cascade on top at request time.
    let audit_log = audit_log::AuditLog::new(data_dir.as_deref(), cfg.audit_mode);
    println!(
        "Audit log: server default mode = {:?} (override via [audit].mode in config.toml or MICRORESOLVE_AUDIT_MODE env)",
        cfg.audit_mode
    );
    let ui_settings = data_dir
        .as_deref()
        .map(load_ui_settings)
        .unwrap_or_default();

    // API keys cover EVERY /api/* call (UI fetches included). On a fresh
    // install with an empty keystore the server auto-mints
    // `studio-admin` (Admin scope) and prints it once — operator pastes
    // it into the Studio paste-screen, browser stores it in localStorage,
    // every subsequent fetch carries `X-Api-Key`. The /api/auth/keys POST
    // requires the same auth like everything else; there is no
    // unauthenticated bootstrap route.
    //
    // If the operator loses stdout (Docker -d, redirected logs, etc.) the
    // key is also persisted to `<config>/admin-key.txt` mode 0600 — they
    // can `cat` it any time.
    let mut key_store = match cfg.keys_file.clone() {
        Some(p) => key_store::KeyStore::load_from(Some(p)),
        None => key_store::KeyStore::load(),
    };
    let admin_key_path = key_store.admin_key_path();
    match key_store.bootstrap_if_empty() {
        Ok(Some(key)) => {
            println!();
            println!("──────────────────────────────────────────────────────────────");
            println!(" Studio admin key (paste this into the browser on first visit):");
            println!();
            println!("   {}", key);
            println!();
            if let Some(p) = admin_key_path.as_ref() {
                println!(" Also saved to: {}", p.display());
            }
            println!("──────────────────────────────────────────────────────────────");
            println!();
        }
        Ok(None) => {
            println!(
                "Loaded {} API key(s) from keys.json — Studio expects X-Api-Key",
                key_store.list_redacted().len()
            );
            if let Some(p) = admin_key_path.as_ref() {
                if std::path::Path::new(p).exists() {
                    println!("  admin-key.txt: {}", p.display());
                }
            }
        }
        Err(e) => {
            eprintln!("[key_store] bootstrap failed: {}", e);
            eprintln!("  Studio will reject all /api/* requests until a key exists.");
        }
    }

    let (event_tx, _) = broadcast::channel::<state::StudioEvent>(256);
    let worker_notify = Arc::new(Notify::new());

    let state: AppState = Arc::new(ServerState {
        engine,
        data_dir,
        git_remote: RwLock::new(git_remote),
        log_store: Mutex::new(log_store),
        audit_log,
        http: reqwest::Client::new(),
        llm_key,
        review_mode: RwLock::new(HashMap::new()),
        ui_settings: RwLock::new(ui_settings),
        event_tx,
        worker_notify: worker_notify.clone(),
        key_store: std::sync::RwLock::new(key_store),
        connected_clients: RwLock::new(HashMap::new()),
    });

    // Spawn the background auto-learn worker
    tokio::spawn(worker::run_worker(state.clone(), worker_notify));

    // Public probes — no auth, by design. Bound on /api/health, /api/version,
    // /api/llm/status. Operators / orchestrators (k8s, healthcheck.io, etc.)
    // hit these without credentials.
    let public_api = axum::Router::new()
        .route("/api/health", get(health))
        .route("/api/llm/status", get(llm_status))
        .route("/api/version", get(get_version));

    // Every other /api/* route — UI fetches included — requires X-Api-Key.
    // The middleware below is a single chokepoint; individual route files no
    // longer need their own auth logic. Keep route_connect's check_auth for
    // attribution (it returns the key NAME so connected-clients tracking can
    // associate the sync with the right library), but the gate itself runs
    // here.
    let protected_api = axum::Router::new()
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
        .merge(routes_stopwords::routes())
        .merge(routes_git::routes())
        .merge(routes_state::routes())
        .merge(routes_audit::routes())
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ));

    let app = public_api
        .merge(protected_api)
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    // Bundled mode: assets are compiled into the binary via rust-embed.
    #[cfg(feature = "bundled-ui")]
    let app = {
        use axum::{
            body::Body,
            http::{header, Response, StatusCode},
            response::IntoResponse,
        };
        use rust_embed::RustEmbed;

        #[derive(RustEmbed)]
        #[folder = "ui/dist/"]
        struct UiAssets;

        async fn embedded_ui(uri: axum::http::Uri) -> impl IntoResponse {
            let path = uri.path().trim_start_matches('/');

            // 1. Try to serve the file at the requested path directly. Covers
            //    /assets/* (hashed JS/CSS — immutable cache) and root-level
            //    static files (favicon.svg, robots.txt, manifest.json, …).
            if !path.is_empty() {
                if let Some(content) = UiAssets::get(path) {
                    let mime = mime_guess::from_path(path)
                        .first_or_octet_stream()
                        .to_string();
                    let cache_header = if path.starts_with("assets/") {
                        "public, max-age=31536000, immutable"
                    } else {
                        "no-cache, no-store, must-revalidate"
                    };
                    return Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, mime)
                        .header(header::CACHE_CONTROL, cache_header)
                        .body(Body::from(content.data.into_owned()))
                        .unwrap();
                }
            }

            // 2. SPA fallback — serve index.html for client-side routes
            //    (/connected, /history, /resolve, …) and the root path.
            match UiAssets::get("index.html") {
                Some(content) => Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/html")
                    .header(header::CACHE_CONTROL, "no-cache, no-store, must-revalidate")
                    .body(Body::from(content.data.into_owned()))
                    .unwrap(),
                None => Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(Body::from("UI not found"))
                    .unwrap(),
            }
        }

        println!("UI served from: embedded (bundled-ui)");
        app.fallback(embedded_ui)
    };

    // Default (disk-based) mode: serve ui/dist/ from next to the binary if present.
    #[cfg(not(feature = "bundled-ui"))]
    let app = if let Some(dist) = ui_dist.as_ref() {
        use axum::http::header;
        use axum::response::IntoResponse;

        // Store the index.html path in a OnceLock so the fallback handler
        // (which must be a plain fn for axum to accept) can read it.
        static UI_INDEX_PATH: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
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
        app.nest_service(
            "/assets",
            tower_http::services::ServeDir::new(dist.join("assets")),
        )
        .fallback(spa_index)
    } else {
        println!("(dev) API-only — no ui/dist next to the binary. For the UI, run `cd ui && npm run dev` (http://localhost:3000).");
        app
    };

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!(
                "error: cannot bind to {} — is the port already in use?\n  {}",
                addr, e
            );
            std::process::exit(1);
        }
    };

    println!("MicroResolve Studio listening on {}", addr);
    if let Some(ref dir) = state.data_dir {
        println!("Data directory: {}", dir);
    }

    // Auto-open the browser — only for distributed installs. In dev builds
    // the Vite dev server on :3000 already owns the browser tab.
    if is_distributed {
        if !cfg.no_browser && !cli::looks_headless() {
            let url = format!("http://localhost:{}/", cfg.port);
            if let Err(e) = open::that_detached(&url) {
                eprintln!(
                    "(could not auto-open browser: {}. Visit {} manually.)",
                    e, url
                );
            } else {
                println!("Opening browser at {}", url);
            }
        } else if cfg.no_browser {
            println!("Browser auto-open disabled (--no-browser).");
        } else {
            println!("Headless environment detected — not opening browser.");
        }
    }

    axum::serve(listener, app).await.expect("Server error");
}

async fn health() -> &'static str {
    "ok"
}

/// Universal API key middleware. Runs in front of every protected route
/// (everything under `/api/*` except `/api/health`, `/api/version`,
/// `/api/llm/status`). Reads `X-Api-Key`, validates against the keystore,
/// rejects with 401 when missing or invalid; with 403 when the key's
/// scope doesn't permit the requested route.
///
/// # Scope rules (v0.2.2)
///
/// `KeyScope::App` permits:
///   - All GET requests (read-only state)
///   - POST /api/resolve              (the core routing call)
///   - POST /api/audit/event          (write app-defined audit events)
///   - POST /api/sync                 (delta sync for connected libs)
///   - POST /api/snapshot             (full-state bootstrap)
///   - POST /api/report               (queue an entry for review)
///
/// `KeyScope::Admin` permits everything.
///
/// Anything else from an App-scope key returns 403. This makes the
/// audit attribution claim coherent: an App-scope key cannot mutate
/// intents, namespaces, or other keys — its writes are bounded to its
/// own audit chain (via /api/audit/event) and routing requests.
async fn require_api_key(
    State(state): State<AppState>,
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, axum::http::StatusCode> {
    use crate::key_store::KeyScope;
    let provided = req
        .headers()
        .get("X-Api-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if provided.is_empty() {
        return Err(axum::http::StatusCode::UNAUTHORIZED);
    }
    let Some((name, scope)) = state.key_store.read().unwrap().validate(provided) else {
        return Err(axum::http::StatusCode::UNAUTHORIZED);
    };

    // Scope check. Admin = everything; App = read-only + named writes.
    if scope == KeyScope::App {
        let method = req.method().clone();
        let path = req.uri().path();
        let allowed = matches!(method, axum::http::Method::GET)
            || (matches!(method, axum::http::Method::POST)
                && matches!(
                    path,
                    "/api/resolve"
                        | "/api/audit/event"
                        | "/api/sync"
                        | "/api/snapshot"
                        | "/api/report"
                ));
        if !allowed {
            return Err(axum::http::StatusCode::FORBIDDEN);
        }
    }

    // Make the authenticated key's name available to handlers via
    // `Extension<KeyName>`. Used by the audit log (`kid` attribution).
    req.extensions_mut().insert(state::KeyName(name));
    Ok(next.run(req).await)
}

async fn llm_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let configured = state.llm_key.is_some();
    let provider = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "anthropic".to_string());
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| match provider.as_str() {
        "gemini" => "gemini-2.5-flash".to_string(),
        _ => "claude-haiku-4-5-20251001".to_string(),
    });
    let url = match provider.as_str() {
        "gemini" => format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            model
        ),
        "anthropic" => std::env::var("LLM_API_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com/v1/messages".to_string()),
        _ => std::env::var("LLM_API_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string()),
    };
    Json(serde_json::json!({
        "configured": configured,
        "provider": provider,
        "model": model,
        "url": url,
    }))
}

async fn get_version(State(state): State<AppState>, headers: HeaderMap) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let version = state
        .engine
        .try_namespace(&app_id)
        .map(|h| h.version())
        .unwrap_or(0);
    Json(serde_json::json!({
        "version": version,
        "project_id": app_id,
        "app_version": env!("CARGO_PKG_VERSION"),
    }))
}
