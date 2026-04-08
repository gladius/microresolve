//! ASV Router HTTP API server.
//!
//! Run with: cargo run --bin server --features server --release
//!
//! Default: http://localhost:3001

use asv_router::{Router, IntentType};
use axum::{
    extract::{State, Query},
    http::{StatusCode, HeaderMap},
    routing::{get, post, delete},
    Json,
};
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tower_http::cors::CorsLayer;

const LOG_FILE: &str = "asv_queries.jsonl";

struct ServerState {
    routers: RwLock<HashMap<String, Router>>,
    data_dir: Option<String>,
    log: Mutex<std::fs::File>,
    http: reqwest::Client,
    anthropic_key: Option<String>,
}

type AppState = Arc<ServerState>;

fn open_log() -> std::fs::File {
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)
        .expect("failed to open log file")
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

/// Extract app ID from X-App-ID header, defaulting to "default".
fn app_id_from_headers(headers: &HeaderMap) -> String {
    headers.get("X-App-ID")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("default")
        .to_string()
}

/// Persist a router's state to the data directory if configured.
fn maybe_persist(state: &ServerState, app_id: &str, router: &Router) {
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}.json", dir, app_id);
        let json = router.export_json();
        let _ = std::fs::write(&path, json);
    }
}

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

    // Load .env if present (for ANTHROPIC_API_KEY)
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

    let anthropic_key = std::env::var("ANTHROPIC_API_KEY").ok();
    if anthropic_key.is_some() {
        println!("Anthropic API key: loaded");
    } else {
        println!("Anthropic API key: not set (LLM features disabled)");
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

    let state: AppState = Arc::new(ServerState {
        routers: RwLock::new(routers),
        data_dir,
        log: Mutex::new(open_log()),
        http: reqwest::Client::new(),
        anthropic_key,
    });

    let app = axum::Router::new()
        .route("/api/health", get(health))
        .route("/api/version", get(get_version))
        .route("/api/route", post(route_query))
        .route("/api/route_multi", post(route_multi))
        .route("/api/intents", get(list_intents))
        .route("/api/intents", post(add_intent))
        .route("/api/intents/delete", post(delete_intent))
        .route("/api/intents/add_seed", post(add_seed))
        .route("/api/intents/remove_seed", post(remove_seed))
        .route("/api/intents/multilingual", post(add_intent_multilingual))
        .route("/api/intents/type", post(set_intent_type))
        .route("/api/intents/load_defaults", post(load_defaults))
        .route("/api/learn", post(learn))
        .route("/api/correct", post(correct))
        .route("/api/metadata", post(set_metadata))
        .route("/api/metadata/get", post(get_metadata))
        .route("/api/seed/prompt", post(build_seed_prompt))
        .route("/api/seed/parse", post(parse_seed_response))
        .route("/api/reset", post(reset))
        .route("/api/export", get(export_state))
        .route("/api/import", post(import_state))
        .route("/api/languages", get(get_languages))
        .route("/api/logs", get(get_logs))
        .route("/api/logs", delete(clear_logs))
        .route("/api/logs/stats", get(log_stats))
        .route("/api/co_occurrence", get(get_co_occurrence))
        .route("/api/projections", get(get_projections))
        .route("/api/review/prompt", post(build_review_prompt))
        .route("/api/review", post(review))
        .route("/api/seed/generate", post(generate_seeds))
        .route("/api/simulate/turn", post(simulate_turn))
        .route("/api/simulate/respond", post(simulate_respond))
        .route("/api/training/generate", post(training_generate))
        .route("/api/training/run", post(training_run))
        .route("/api/training/review", post(training_review))
        .route("/api/training/apply", post(training_apply))
        .route("/api/workflows", get(get_workflows))
        .route("/api/temporal_order", get(get_temporal_order))
        .route("/api/escalation_patterns", get(get_escalation_patterns))
        .route("/api/apps", get(list_apps))
        .route("/api/apps", post(create_app))
        .route("/api/apps", delete(delete_app))
        .route("/api/discover", post(discover))
        .route("/api/discover/apply", post(discover_apply))
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    println!("ASV Router server listening on {}", addr);
    println!("Query log: {}", LOG_FILE);
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

// --- Logging helper ---

fn log_query(state: &ServerState, entry: &serde_json::Value) {
    if let Ok(mut file) = state.log.lock() {
        let _ = writeln!(file, "{}", entry);
        let _ = file.flush();
    }
}

// --- Route ---

#[derive(serde::Deserialize)]
struct RouteRequest {
    query: String,
}

async fn route_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
    };
    let results = router.route(&req.query);
    let out: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "id": r.id,
                "score": (r.score * 100.0).round() / 100.0
            })
        })
        .collect();
    Json(serde_json::json!(out))
}

#[derive(serde::Deserialize)]
struct RouteMultiRequest {
    query: String,
    #[serde(default = "default_threshold")]
    threshold: f32,
}

fn default_threshold() -> f32 {
    0.3
}

async fn route_multi(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteMultiRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let t0 = std::time::Instant::now();
    let output = {
        let routers = state.routers.read().unwrap();
        let router = match routers.get(&app_id) {
            Some(r) => r,
            None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
        };
        router.route_multi(&req.query, req.threshold)
    };
    let latency_us = t0.elapsed().as_micros() as u64;

    // Record intent sequence (co-occurrence + temporal order + full sequence)
    if output.intents.len() > 1 {
        let ids: Vec<&str> = output.intents.iter().map(|i| i.id.as_str()).collect();
        if let Ok(mut routers) = state.routers.write() {
            if let Some(router) = routers.get_mut(&app_id) {
                router.record_intent_sequence(&ids);
                maybe_persist(&state, &app_id, router);
            }
        }
    }

    let intents: Vec<serde_json::Value> = output
        .intents
        .iter()
        .map(|i| {
            serde_json::json!({
                "id": i.id,
                "score": (i.score * 100.0).round() / 100.0,
                "position": i.position,
                "span": [i.span.0, i.span.1],
                "intent_type": i.intent_type,
                "confidence": i.confidence,
                "source": i.source,
                "negated": i.negated
            })
        })
        .collect();
    let relations: Vec<serde_json::Value> = output
        .relations
        .iter()
        .map(|r| {
            use asv_router::IntentRelation;
            match r {
                IntentRelation::Parallel => serde_json::json!({"type": "Parallel"}),
                IntentRelation::Sequential { first, then } => {
                    serde_json::json!({"type": "Sequential", "first": first, "then": then})
                }
                IntentRelation::Conditional { primary, fallback } => {
                    serde_json::json!({"type": "Conditional", "primary": primary, "fallback": fallback})
                }
                IntentRelation::Reverse {
                    stated_first,
                    execute_first,
                } => {
                    serde_json::json!({"type": "Reverse", "stated_first": stated_first, "execute_first": execute_first})
                }
                IntentRelation::Negation { do_this, not_this } => {
                    serde_json::json!({"type": "Negation", "do_this": do_this, "not_this": not_this})
                }
            }
        })
        .collect();

    // Compute projected_context from co-occurrence
    let projected_context = {
        let routers = state.routers.read().unwrap();
        let router = match routers.get(&app_id) {
            Some(r) => r,
            None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
        };
        let co_pairs = router.get_co_occurrence();
        let matched_ids: std::collections::HashSet<&str> = output.intents.iter().map(|i| i.id.as_str()).collect();

        // For each matched action intent, find context intents that co-occur but aren't already in results
        let mut context_scores: HashMap<String, (u32, u32)> = HashMap::new(); // id -> (co_count, total_action_count)

        for intent in &output.intents {
            if intent.intent_type != asv_router::IntentType::Action {
                continue;
            }
            // Count total co-occurrences for this action (denominator for strength)
            let mut action_total: u32 = 0;
            for &(a, b, count) in &co_pairs {
                if a == intent.id || b == intent.id {
                    action_total += count;
                }
            }
            if action_total == 0 {
                continue;
            }
            // Find context partners not already in results
            for &(a, b, count) in &co_pairs {
                let partner = if a == intent.id { b } else if b == intent.id { a } else { continue };
                if matched_ids.contains(partner) {
                    continue; // already in results, don't project
                }
                if router.get_intent_type(partner) != asv_router::IntentType::Context {
                    continue; // only project context intents
                }
                let entry = context_scores.entry(partner.to_string()).or_insert((0, 0));
                entry.0 += count;
                entry.1 += action_total;
            }
        }

        let mut projected: Vec<serde_json::Value> = context_scores
            .into_iter()
            .map(|(id, (co_count, total))| {
                let strength = co_count as f64 / total as f64;
                serde_json::json!({
                    "id": id,
                    "co_occurrence": co_count,
                    "strength": (strength * 100.0).round() / 100.0
                })
            })
            .filter(|v| v["strength"].as_f64().unwrap_or(0.0) >= 0.1) // min 10% strength
            .collect();
        projected.sort_by(|a, b| {
            b["strength"].as_f64().unwrap_or(0.0)
                .partial_cmp(&a["strength"].as_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        projected
    };

    // Split into confirmed (high + medium confidence) and candidates (low confidence)
    let confirmed: Vec<&serde_json::Value> = intents.iter()
        .filter(|i| i["confidence"].as_str() != Some("low"))
        .collect();
    let candidates: Vec<&serde_json::Value> = intents.iter()
        .filter(|i| i["confidence"].as_str() == Some("low"))
        .collect();

    let suggestions: Vec<serde_json::Value> = output.suggestions.iter()
        .map(|s| serde_json::json!({
            "id": s.id,
            "probability": (s.probability * 100.0).round() / 100.0,
            "observations": s.observations,
            "because_of": s.because_of
        }))
        .collect();

    let result = serde_json::json!({
        "confirmed": confirmed,
        "candidates": candidates,
        "relations": relations,
        "metadata": output.metadata,
        "projected_context": projected_context,
        "suggestions": suggestions
    });

    // Log this query
    log_query(&state, &serde_json::json!({
        "ts": now_ms(),
        "query": req.query,
        "threshold": req.threshold,
        "latency_us": latency_us,
        "results": intents,
    }));

    Json(result)
}

// --- Intents ---

async fn list_intents(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
    };
    let mut ids = router.intent_ids();
    ids.sort();
    let intents: Vec<serde_json::Value> = ids
        .iter()
        .map(|id| {
            let seeds = router.get_training(id).unwrap_or_default();
            let by_lang = router.get_training_by_lang(id).cloned().unwrap_or_default();
            let learned = router
                .get_vector(id)
                .map(|v| v.learned_term_count())
                .unwrap_or(0);
            let intent_type = router.get_intent_type(id);
            let metadata = router.get_metadata(id).cloned().unwrap_or_default();
            serde_json::json!({
                "id": id,
                "seeds": seeds,
                "seeds_by_lang": by_lang,
                "learned_count": learned,
                "intent_type": intent_type,
                "metadata": metadata
            })
        })
        .collect();
    Json(serde_json::json!(intents))
}

#[derive(serde::Deserialize)]
struct AddIntentRequest {
    id: String,
    seeds: Vec<String>,
    #[serde(default)]
    intent_type: Option<IntentType>,
    #[serde(default)]
    metadata: Option<HashMap<String, Vec<String>>>,
}

async fn add_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddIntentRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    let seed_refs: Vec<&str> = req.seeds.iter().map(|s| s.as_str()).collect();
    router.add_intent(&req.id, &seed_refs);
    if let Some(t) = req.intent_type {
        router.set_intent_type(&req.id, t);
    }
    if let Some(meta) = req.metadata {
        for (key, values) in meta {
            router.set_metadata(&req.id, &key, values);
        }
    }
    maybe_persist(&state, &app_id, router);
    StatusCode::CREATED
}

#[derive(serde::Deserialize)]
struct AddSeedRequest {
    intent_id: String,
    seed: String,
}

async fn add_seed(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddSeedRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
    let phrases = router
        .get_training(&req.intent_id)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("intent '{}' not found", req.intent_id),
            )
        })?;
    let mut all = phrases;
    all.push(req.seed);
    let refs: Vec<&str> = all.iter().map(|s| s.as_str()).collect();
    router.add_intent(&req.intent_id, &refs);
    maybe_persist(&state, &app_id, router);
    Ok(StatusCode::OK)
}

#[derive(serde::Deserialize)]
struct RemoveSeedRequest {
    intent_id: String,
    seed: String,
}

async fn remove_seed(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RemoveSeedRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
    if router.remove_seed(&req.intent_id, &req.seed) {
        maybe_persist(&state, &app_id, router);
        Ok(StatusCode::OK)
    } else {
        Err((StatusCode::NOT_FOUND, "seed not found".to_string()))
    }
}

#[derive(serde::Deserialize)]
struct AddIntentMultilingualRequest {
    id: String,
    seeds_by_lang: HashMap<String, Vec<String>>,
    #[serde(default)]
    intent_type: Option<IntentType>,
    #[serde(default)]
    metadata: Option<HashMap<String, Vec<String>>>,
}

async fn add_intent_multilingual(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddIntentMultilingualRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.add_intent_multilingual(&req.id, req.seeds_by_lang);
    if let Some(t) = req.intent_type {
        router.set_intent_type(&req.id, t);
    }
    if let Some(meta) = req.metadata {
        for (key, values) in meta {
            router.set_metadata(&req.id, &key, values);
        }
    }
    maybe_persist(&state, &app_id, router);
    StatusCode::CREATED
}

#[derive(serde::Deserialize)]
struct SetIntentTypeRequest {
    intent_id: String,
    intent_type: IntentType,
}

async fn set_intent_type(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetIntentTypeRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.set_intent_type(&req.intent_id, req.intent_type);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
struct DeleteIntentRequest {
    id: String,
}

async fn delete_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DeleteIntentRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.remove_intent(&req.id);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

// --- Learn / Correct ---

#[derive(serde::Deserialize)]
struct LearnRequest {
    query: String,
    intent_id: String,
}

async fn learn(State(state): State<AppState>, headers: HeaderMap, Json(req): Json<LearnRequest>) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.learn(&req.query, &req.intent_id);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
struct CorrectRequest {
    query: String,
    wrong_intent: String,
    correct_intent: String,
}

async fn correct(State(state): State<AppState>, headers: HeaderMap, Json(req): Json<CorrectRequest>) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.correct(&req.query, &req.wrong_intent, &req.correct_intent);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

// --- Metadata ---

#[derive(serde::Deserialize)]
struct SetMetadataRequest {
    intent_id: String,
    key: String,
    values: Vec<String>,
}

async fn set_metadata(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetMetadataRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.set_metadata(&req.intent_id, &req.key, req.values);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
struct GetMetadataRequest {
    intent_id: String,
}

async fn get_metadata(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<GetMetadataRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
    };
    let meta = router.get_metadata(&req.intent_id).cloned().unwrap_or_default();
    Json(serde_json::json!(meta))
}

// --- Query Log ---

#[derive(serde::Deserialize)]
struct LogQuery {
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

fn default_limit() -> usize { 100 }

async fn get_logs(Query(params): Query<LogQuery>) -> Json<serde_json::Value> {
    let content = std::fs::read_to_string(LOG_FILE).unwrap_or_default();
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();

    // Most recent first
    let entries: Vec<serde_json::Value> = lines.iter().rev()
        .skip(params.offset)
        .take(params.limit)
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    Json(serde_json::json!({
        "total": total,
        "offset": params.offset,
        "limit": params.limit,
        "entries": entries,
    }))
}

async fn log_stats() -> Json<serde_json::Value> {
    let count = std::fs::read_to_string(LOG_FILE)
        .map(|c| c.lines().count())
        .unwrap_or(0);
    let size = std::fs::metadata(LOG_FILE)
        .map(|m| m.len())
        .unwrap_or(0);

    Json(serde_json::json!({
        "count": count,
        "size_bytes": size,
        "file": LOG_FILE,
    }))
}

async fn clear_logs(State(state): State<AppState>) -> StatusCode {
    // Truncate the log file
    if let Ok(mut file) = state.log.lock() {
        if let Ok(f) = std::fs::File::create(LOG_FILE) {
            *file = f;
        }
    }
    StatusCode::OK
}

// --- Seed generation ---

#[derive(serde::Deserialize)]
struct BuildPromptRequest {
    intent_id: String,
    description: String,
    languages: Vec<String>,
}

async fn build_seed_prompt(Json(req): Json<BuildPromptRequest>) -> Json<serde_json::Value> {
    let prompt = asv_router::seed::build_prompt(&req.intent_id, &req.description, &req.languages);
    Json(serde_json::json!({ "prompt": prompt }))
}

#[derive(serde::Deserialize)]
struct ParseResponseRequest {
    response_text: String,
    languages: Vec<String>,
}

async fn parse_seed_response(
    Json(req): Json<ParseResponseRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let result = asv_router::seed::parse_response(&req.response_text, &req.languages)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let val: serde_json::Value =
        serde_json::from_str(&result).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(val))
}

// --- Reset ---

async fn reset(State(state): State<AppState>, headers: HeaderMap) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = Router::new();
    maybe_persist(&state, &app_id, &router);
    routers.insert(app_id, router);
    StatusCode::OK
}

// --- Load defaults ---

async fn load_defaults(State(state): State<AppState>, headers: HeaderMap) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };

    let actions: &[(&str, &[&str])] = &[
        // --- Original 6 ---
        ("cancel_order", &[
            "cancel my order",
            "I need to cancel an order I just placed",
            "please stop my order from shipping",
            "I changed my mind and want to cancel the purchase",
            "how do I cancel something I ordered yesterday",
            "cancel order number",
            "I accidentally ordered the wrong thing, cancel it",
            "withdraw my order before it ships",
        ]),
        ("refund", &[
            "I want a refund",
            "get my money back",
            "I received a damaged item and need a refund",
            "the product was nothing like the description, refund please",
            "how long does it take to process a return",
            "I returned it two weeks ago and still no refund",
            "I want to return this for a full refund",
            "money back",
        ]),
        ("contact_human", &[
            "talk to a human",
            "I need to speak with a real person not a bot",
            "connect me to customer service please",
            "this bot is useless, get me an agent",
            "transfer me to a representative",
            "I want to talk to someone who can actually help",
            "live agent please",
            "escalate this to a manager",
        ]),
        ("reset_password", &[
            "reset my password",
            "I forgot my password and can't log in",
            "my account is locked out",
            "how do I change my password",
            "the password reset email never arrived",
            "I keep getting invalid password error",
            "locked out of my account need help getting back in",
            "send me a password reset link",
        ]),
        ("update_address", &[
            "change my address",
            "I moved and need to update my shipping address",
            "update my delivery address before it ships",
            "my address is wrong on the order",
            "change the shipping destination",
            "I need to correct my mailing address",
            "ship it to a different address instead",
            "new address for future orders",
        ]),
        ("billing_issue", &[
            "wrong charge on my account",
            "I was charged twice for the same order",
            "there's a billing error on my statement",
            "I see an unauthorized charge",
            "you overcharged me by twenty dollars",
            "my credit card was charged the wrong amount",
            "dispute a charge",
            "the amount on my bill doesn't match what I ordered",
        ]),
        // --- New actions ---
        ("change_plan", &[
            "upgrade my plan",
            "I want to switch to the premium subscription",
            "downgrade my account to the basic tier",
            "change my subscription plan",
            "what plans are available for upgrade",
            "I want a cheaper plan",
            "switch me to the annual billing",
        ]),
        ("close_account", &[
            "delete my account",
            "I want to close my account permanently",
            "how do I deactivate my profile",
            "remove all my data and close the account",
            "I no longer want to use this service",
            "cancel my membership entirely",
            "please terminate my account",
        ]),
        ("report_fraud", &[
            "someone used my card without permission",
            "I think my account was hacked",
            "there are transactions I did not make",
            "report unauthorized access to my account",
            "fraudulent activity on my card",
            "someone stole my identity and made purchases",
            "I need to report suspicious charges",
        ]),
        ("apply_coupon", &[
            "I have a discount code",
            "apply my coupon to the order",
            "where do I enter a promo code",
            "this coupon isn't working",
            "I forgot to apply my discount before checkout",
            "can I use two coupons on one order",
            "my promotional code was rejected",
        ]),
        ("schedule_callback", &[
            "can someone call me back",
            "I'd like to schedule a phone call",
            "have an agent call me at this number",
            "request a callback for tomorrow morning",
            "I prefer a phone call over chat",
            "when can I expect a call back",
            "set up a time for support to call me",
        ]),
        ("file_complaint", &[
            "I want to file a formal complaint",
            "this is unacceptable, I'm filing a complaint",
            "how do I report poor service",
            "I want to submit a grievance",
            "your service has been terrible and I want it documented",
            "escalate my complaint to upper management",
            "I need to make an official complaint",
        ]),
        ("request_invoice", &[
            "send me an invoice for my purchase",
            "I need a receipt for tax purposes",
            "can I get a PDF of my invoice",
            "email me the billing statement",
            "I need documentation of this transaction",
            "where can I download my invoice",
            "generate an invoice for order number",
        ]),
        ("pause_subscription", &[
            "pause my subscription for a month",
            "I want to temporarily stop my membership",
            "can I freeze my account without canceling",
            "put my plan on hold",
            "suspend my subscription until next quarter",
            "I'm traveling and want to pause billing",
            "temporarily deactivate my subscription",
        ]),
        ("transfer_funds", &[
            "transfer money to another account",
            "send funds to my savings account",
            "move money between my accounts",
            "I want to wire money to someone",
            "initiate a bank transfer",
            "how do I send money to another person",
            "transfer fifty dollars to my checking",
        ]),
        ("add_payment_method", &[
            "add a new credit card to my account",
            "I want to register a different payment method",
            "update my card information",
            "save a new debit card for payments",
            "link my bank account for direct payment",
            "replace my expired card on file",
            "add PayPal as a payment option",
        ]),
        ("remove_item", &[
            "remove an item from my order",
            "take this product out of my cart",
            "I don't want one of the items in my order anymore",
            "delete the second item from my purchase",
            "can I remove something before it ships",
            "take off the extra item I added by mistake",
            "drop one item from my order",
        ]),
        ("reorder", &[
            "reorder my last purchase",
            "I want to buy the same thing again",
            "repeat my previous order",
            "order the same items as last time",
            "can I quickly reorder what I got before",
            "place the same order again",
            "buy this product again",
        ]),
        ("upgrade_shipping", &[
            "upgrade to express shipping",
            "I need this delivered faster",
            "can I switch to overnight delivery",
            "expedite my shipment",
            "change my shipping to two-day delivery",
            "I'll pay extra for faster shipping",
            "rush delivery please",
        ]),
        ("gift_card_redeem", &[
            "redeem my gift card",
            "I have a gift card code to apply",
            "how do I use a gift certificate",
            "enter my gift card balance",
            "apply a gift card to my purchase",
            "my gift card isn't being accepted",
            "check the balance on my gift card",
        ]),
    ];

    let contexts: &[(&str, &[&str])] = &[
        // --- Original 2 ---
        ("track_order", &[
            "where is my package",
            "track my order",
            "my order still hasn't arrived and it's been a week",
            "I need a shipping update on my recent purchase",
            "when will my delivery arrive",
            "package tracking number",
            "it says delivered but I never got it",
            "how long until my order gets here",
        ]),
        ("check_balance", &[
            "check my balance",
            "how much money is in my account",
            "what's my current account balance",
            "show me my available funds",
            "I need to know how much I have left",
            "account summary",
            "remaining balance on my card",
            "what do I owe right now",
        ]),
        // --- New context ---
        ("account_status", &[
            "is my account in good standing",
            "check my account status",
            "am I verified",
            "what is the state of my account",
            "is my account active or suspended",
            "show me my account details",
            "my account status page",
        ]),
        ("order_history", &[
            "show me my past orders",
            "what did I order last month",
            "view my order history",
            "list all my previous purchases",
            "I need to see what I bought before",
            "pull up my recent orders",
            "my purchase history",
        ]),
        ("payment_history", &[
            "show me my payment history",
            "list all charges to my account",
            "what payments have I made",
            "view my transaction log",
            "when was my last payment",
            "how much have I spent this month",
            "pull up my billing history",
        ]),
        ("shipping_options", &[
            "what shipping methods are available",
            "how much does express shipping cost",
            "what are my delivery options",
            "do you offer free shipping",
            "compare shipping speeds and prices",
            "international shipping rates",
            "same day delivery available",
        ]),
        ("return_policy", &[
            "what is your return policy",
            "how many days do I have to return something",
            "can I return a used product",
            "do you accept returns without receipt",
            "what items are not returnable",
            "is there a restocking fee for returns",
            "return and exchange policy",
        ]),
        ("product_availability", &[
            "is this item in stock",
            "when will this product be available again",
            "check if you have this in my size",
            "is this item available for delivery",
            "out of stock notification",
            "do you carry this brand",
            "product availability in my area",
        ]),
        ("warranty_info", &[
            "what does the warranty cover",
            "how long is the warranty period",
            "is my product still under warranty",
            "warranty claim process",
            "does this come with a manufacturer warranty",
            "extended warranty options",
            "what voids the warranty",
        ]),
        ("loyalty_points", &[
            "how many reward points do I have",
            "check my loyalty balance",
            "when do my points expire",
            "how can I redeem my reward points",
            "how many points do I earn per dollar",
            "my rewards program status",
            "transfer loyalty points",
        ]),
        ("subscription_status", &[
            "what plan am I on",
            "when does my subscription renew",
            "show me my current plan details",
            "how much am I paying monthly",
            "when is my next billing date",
            "what features are included in my plan",
            "subscription renewal date",
        ]),
        ("delivery_estimate", &[
            "when will my order arrive",
            "estimated delivery date",
            "how long does shipping take",
            "expected arrival for my package",
            "delivery timeframe for my area",
            "how many business days until delivery",
            "will it arrive before the weekend",
        ]),
        ("price_check", &[
            "how much does this cost",
            "what is the price of this item",
            "is this on sale right now",
            "price match guarantee",
            "compare prices for this product",
            "total cost including shipping",
            "any discounts on this item",
        ]),
        ("account_limits", &[
            "what is my spending limit",
            "daily transfer limit on my account",
            "maximum withdrawal amount",
            "how much can I send per transaction",
            "increase my account limits",
            "what are the restrictions on my account",
            "transaction limits for my plan",
        ]),
        ("transaction_details", &[
            "show me details of my last transaction",
            "what was that charge for",
            "transaction reference number lookup",
            "I need details about a specific payment",
            "when exactly was this charge made",
            "who was the merchant for this transaction",
            "breakdown of charges on my statement",
        ]),
        ("eligibility_check", &[
            "am I eligible for an upgrade",
            "do I qualify for a discount",
            "can I apply for this program",
            "check my eligibility for the promotion",
            "what are the requirements to qualify",
            "am I eligible for a credit increase",
            "do I meet the criteria for this offer",
        ]),
    ];

    router.begin_batch();
    for (id, seeds) in actions {
        router.add_intent(id, seeds);
        router.set_intent_type(id, IntentType::Action);
    }
    for (id, seeds) in contexts {
        router.add_intent(id, seeds);
        router.set_intent_type(id, IntentType::Context);
    }
    router.end_batch();

    // Paraphrase index starts empty at cold start.
    // Populated only through learn()/add_seed calls (training arena, learn mode, API).
    // Load previously learned paraphrases if available.
    if let Ok(data) = std::fs::read_to_string("tests/data/paraphrases.json") {
        if let Ok(paraphrases) = serde_json::from_str::<std::collections::HashMap<String, Vec<String>>>(&data) {
            router.begin_batch();
            router.add_paraphrases_bulk(&paraphrases);
            router.end_batch();
            eprintln!("Loaded {} learned paraphrase phrases", router.paraphrase_count());
        }
    }

    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

// --- Export / Import ---

async fn export_state(State(state): State<AppState>, headers: HeaderMap) -> String {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    match routers.get(&app_id) {
        Some(router) => router.export_json(),
        None => format!("{{\"error\": \"app '{}' not found\"}}", app_id),
    }
}

async fn import_state(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Result<StatusCode, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let new_router =
        Router::import_json(&body).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    maybe_persist(&state, &app_id, &new_router);
    let mut routers = state.routers.write().unwrap();
    routers.insert(app_id, new_router);
    Ok(StatusCode::OK)
}

// --- Languages ---

async fn get_languages() -> Json<serde_json::Value> {
    let json_str = asv_router::seed::supported_languages_json();
    let val: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_default();
    Json(val)
}

// --- Co-occurrence ---

async fn get_co_occurrence(State(state): State<AppState>, headers: HeaderMap) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
    };
    let pairs = router.get_co_occurrence();
    let out: Vec<serde_json::Value> = pairs.iter().map(|(a, b, count)| {
        serde_json::json!({"a": a, "b": b, "count": count})
    }).collect();
    Json(serde_json::json!(out))
}

// --- Workflows: emergent cluster discovery ---

#[derive(serde::Deserialize)]
struct WorkflowQuery {
    #[serde(default = "default_min_obs")]
    min_observations: u32,
}
fn default_min_obs() -> u32 { 3 }

async fn get_workflows(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<WorkflowQuery>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
    };
    let workflows = router.discover_workflows(params.min_observations);
    let out: Vec<serde_json::Value> = workflows.iter().map(|cluster| {
        let intents: Vec<serde_json::Value> = cluster.iter().map(|wi| {
            serde_json::json!({
                "id": wi.id,
                "connections": wi.connections,
                "neighbors": wi.neighbors,
            })
        }).collect();
        serde_json::json!({"intents": intents, "size": cluster.len()})
    }).collect();
    Json(serde_json::json!({"workflows": out, "count": out.len()}))
}

// --- Temporal ordering ---

async fn get_temporal_order(State(state): State<AppState>, headers: HeaderMap) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
    };
    let order = router.get_temporal_order();
    let out: Vec<serde_json::Value> = order.iter().map(|(first, second, prob, count)| {
        serde_json::json!({
            "first": first,
            "second": second,
            "probability": (prob * 100.0).round() / 100.0,
            "count": count,
        })
    }).collect();
    Json(serde_json::json!(out))
}

// --- Escalation patterns ---

#[derive(serde::Deserialize)]
struct EscalationQuery {
    #[serde(default = "default_min_occ")]
    min_occurrences: u32,
}
fn default_min_occ() -> u32 { 2 }

async fn get_escalation_patterns(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<EscalationQuery>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
    };
    let patterns = router.detect_escalation_patterns(params.min_occurrences);
    let out: Vec<serde_json::Value> = patterns.iter().map(|p| {
        serde_json::json!({
            "sequence": p.sequence,
            "occurrences": p.occurrences,
            "frequency": (p.frequency * 1000.0).round() / 1000.0,
        })
    }).collect();
    Json(serde_json::json!({"patterns": out, "count": out.len()}))
}

// --- Projections: full action → context map ---

async fn get_projections(State(state): State<AppState>, headers: HeaderMap) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
    };
    let co_pairs = router.get_co_occurrence();
    let ids = router.intent_ids();

    // Build adjacency and totals
    let mut adj: HashMap<String, HashMap<String, u32>> = HashMap::new();
    let mut totals: HashMap<String, u32> = HashMap::new();
    for &(a, b, count) in &co_pairs {
        adj.entry(a.to_string()).or_default().insert(b.to_string(), count);
        adj.entry(b.to_string()).or_default().insert(a.to_string(), count);
        *totals.entry(a.to_string()).or_default() += count;
        *totals.entry(b.to_string()).or_default() += count;
    }

    let mut projections: Vec<serde_json::Value> = Vec::new();
    for id in &ids {
        if router.get_intent_type(id) != asv_router::IntentType::Action {
            continue;
        }
        let total = totals.get(id.as_str()).copied().unwrap_or(0);
        if total == 0 {
            continue;
        }
        let neighbors = match adj.get(id.as_str()) {
            Some(n) => n,
            None => continue,
        };
        let mut context: Vec<serde_json::Value> = neighbors.iter()
            .filter(|(nid, _)| router.get_intent_type(nid) == asv_router::IntentType::Context)
            .map(|(nid, &count)| {
                let strength = count as f64 / total as f64;
                serde_json::json!({
                    "id": nid,
                    "count": count,
                    "strength": (strength * 100.0).round() / 100.0
                })
            })
            .filter(|v| v["strength"].as_f64().unwrap_or(0.0) >= 0.1)
            .collect();
        context.sort_by(|a, b| {
            b["strength"].as_f64().unwrap_or(0.0)
                .partial_cmp(&a["strength"].as_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if !context.is_empty() {
            projections.push(serde_json::json!({
                "action": id,
                "total_co_occurrences": total,
                "projected_context": context,
            }));
        }
    }
    projections.sort_by(|a, b| {
        b["total_co_occurrences"].as_u64().unwrap_or(0)
            .cmp(&a["total_co_occurrences"].as_u64().unwrap_or(0))
    });

    Json(serde_json::json!(projections))
}

// --- Learn mode: review prompt ---

#[derive(serde::Deserialize)]
struct ReviewPromptRequest {
    query: String,
    results: Vec<serde_json::Value>,
    threshold: f32,
}

async fn build_review_prompt(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReviewPromptRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": format!("app '{}' not found", app_id)})),
    };

    // Build intent definitions for the prompt
    let mut intent_defs = Vec::new();
    let mut ids = router.intent_ids();
    ids.sort();
    for id in &ids {
        let seeds = router.get_training(id).unwrap_or_default();
        let intent_type = router.get_intent_type(id);
        intent_defs.push(format!(
            "- {} (type: {:?}): seeds: {:?}",
            id, intent_type,
            seeds.iter().take(5).cloned().collect::<Vec<_>>()
        ));
    }

    let results_json = serde_json::to_string_pretty(&req.results).unwrap_or_default();

    let prompt = format!(
r#"You are reviewing intent routing results from ASV Router, a model-free intent classification system.

## Current intents and their seed phrases:
{}

## Query:
"{}"

## ASV routing result (threshold: {}):
{}

## Your task:
Analyze whether ASV's routing is correct. Return a JSON object with this exact structure:
{{
  "correct": ["intent_id", ...],
  "false_positives": [
    {{"id": "intent_id", "reason": "why this is wrong"}}
  ],
  "missed": [
    {{"id": "intent_id", "reason": "why this should have matched"}}
  ],
  "suggestions": [
    {{
      "action": "learn" | "correct" | "add_seed",
      "query": "the query text",
      "intent_id": "target intent",
      "wrong_intent": "only for correct action",
      "seed": "only for add_seed action",
      "reason": "why this helps"
    }}
  ],
  "confidence": "high" | "medium" | "low",
  "summary": "one sentence summary"
}}

Rules:
- A score below 30% of the best score is likely a false positive
- Context-type intents with low scores may be valid context suggestions, not false positives
- If the routing is perfect, return empty arrays for false_positives, missed, and suggestions
- Be conservative with suggestions — only suggest changes you're confident about
- Return ONLY the JSON object, no other text"#,
        intent_defs.join("\n"),
        req.query,
        req.threshold,
        results_json,
    );

    Json(serde_json::json!({ "prompt": prompt }))
}

// --- LLM call helper ---

async fn call_anthropic(
    state: &ServerState,
    prompt: &str,
    max_tokens: u32,
) -> Result<String, (StatusCode, String)> {
    let key = state.anthropic_key.as_ref().ok_or_else(|| {
        (StatusCode::SERVICE_UNAVAILABLE, "ANTHROPIC_API_KEY not set. Add it to .env file.".to_string())
    })?;

    let body = serde_json::json!({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    });

    let resp = state.http
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Anthropic request failed: {}", e)))?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return Err((StatusCode::BAD_GATEWAY, format!("Anthropic API {}: {}", status, text)));
    }

    let data: serde_json::Value = resp.json().await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Bad response: {}", e)))?;

    data["content"][0]["text"]
        .as_str()
        .map(|s| s.trim().to_string())
        .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No text in response".to_string()))
}

// --- Review: server-side LLM call ---

async fn review(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReviewPromptRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let prompt = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
        let mut intent_defs = Vec::new();
        let mut ids = router.intent_ids();
        ids.sort();
        for id in &ids {
            let seeds = router.get_training(id).unwrap_or_default();
            let intent_type = router.get_intent_type(id);
            intent_defs.push(format!(
                "- {} (type: {:?}): seeds: {:?}",
                id, intent_type,
                seeds.iter().take(5).cloned().collect::<Vec<_>>()
            ));
        }
        let results_json = serde_json::to_string_pretty(&req.results).unwrap_or_default();

        format!(
r#"You are reviewing intent routing results from ASV Router, a model-free intent classification system.

## Current intents and their seed phrases:
{}

## Query:
"{}"

## ASV routing result (threshold: {}):

The router returns two tiers:
- **Confirmed** (high confidence, dual-source verified): the orchestrator will act on these directly
- **Candidates** (low confidence, routing-only): detected but not yet verified

Results:
{}

## Your task:
Analyze whether ASV's routing is correct. Consider:
- Confirmed intents are high-confidence — only flag as false positive if clearly wrong
- Candidates are low-confidence — they're correctly detected but need promotion via training
- If a candidate is correct for the query, suggest an add_seed to promote it

Return a JSON object with this exact structure:
{{
  "correct": ["intent_id", ...],
  "false_positives": [
    {{"id": "intent_id", "reason": "why this is wrong"}}
  ],
  "missed": [
    {{"id": "intent_id", "reason": "why this should have matched"}}
  ],
  "suggestions": [
    {{
      "action": "add_seed",
      "intent_id": "target intent",
      "seed": "short focused phrase (3-8 words) from the query relevant to this intent only",
      "reason": "why this helps"
    }}
  ],
  "confidence": "high" | "medium" | "low",
  "summary": "one sentence summary"
}}

Rules:
- ONLY use action "add_seed". No learn or correct actions.
- Each seed phrase must be SHORT (3-8 words) containing ONLY words relevant to that one intent
- Never use the full query as a seed — extract just the relevant fragment
- Do NOT suggest seeds for intents already confirmed — they don't need it
- For candidates that are correct: suggest a short seed phrase to promote them
- For true misses (not in confirmed or candidates): suggest a seed phrase to teach the router
- Ignore false positives in candidates (low confidence, no action needed)
- Be conservative — only suggest changes you're confident about
- Return ONLY the JSON object, no other text"#,
            intent_defs.join("\n"),
            req.query,
            req.threshold,
            results_json,
        )
    };

    let text = call_anthropic(&state, &prompt, 1024).await?;

    // Extract JSON from response
    let json_str = text.find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No JSON in LLM response".to_string()))?;

    let review_val: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Invalid JSON from LLM: {}", e)))?;

    Ok(Json(review_val))
}

// --- Seed generation: server-side LLM call ---

#[derive(serde::Deserialize)]
struct GenerateSeedsRequest {
    intent_id: String,
    description: String,
    languages: Vec<String>,
}

async fn generate_seeds(
    State(state): State<AppState>,
    Json(req): Json<GenerateSeedsRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let prompt = asv_router::seed::build_prompt(&req.intent_id, &req.description, &req.languages);
    let text = call_anthropic(&state, &prompt, 2048).await?;
    let result = asv_router::seed::parse_response(&text, &req.languages)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Failed to parse seeds: {}", e)))?;
    let val: serde_json::Value = serde_json::from_str(&result)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(val))
}

// --- Simulation: LLM-driven scenario testing ---

#[derive(serde::Deserialize)]
struct SimulateTurnRequest {
    personality: String,     // e.g. "frustrated", "polite", "terse"
    sophistication: String,  // e.g. "low", "medium", "high"
    verbosity: String,       // e.g. "short", "medium", "long"
    history: Vec<serde_json::Value>, // previous turns [{role, message}]
    intents: Vec<String>,    // available intent IDs
    mode: String,            // "normal" or "adversarial"
}

async fn simulate_turn(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SimulateTurnRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let intent_defs = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
        let mut defs = Vec::new();
        // Use client-provided intent list to scope simulation
        let filter: std::collections::HashSet<&str> = req.intents.iter().map(|s| s.as_str()).collect();
        let mut ids = router.intent_ids();
        ids.sort();
        for id in &ids {
            if !filter.is_empty() && !filter.contains(id.as_str()) { continue; }
            let seeds = router.get_training(id).unwrap_or_default();
            defs.push(format!(
                "- {}: {}",
                id,
                seeds.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
            ));
        }
        defs.join("\n")
    };

    let history_text = if req.history.is_empty() {
        "This is the first message in the conversation.".to_string()
    } else {
        let turns: Vec<String> = req.history.iter().map(|t| {
            format!("{}: {}", t["role"].as_str().unwrap_or("?"), t["message"].as_str().unwrap_or(""))
        }).collect();
        turns.join("\n")
    };

    let adversarial_instructions = if req.mode == "adversarial" {
        r#"
ADVERSARIAL MODE: Deliberately try to break the routing system:
- Use unusual synonyms and slang the router may not know
- Be vague and describe things indirectly instead of using exact terms
- Mix multiple intents in confusing ways
- Use negations ambiguously ("I don't NOT want a refund")
- Switch topics mid-sentence
- Use typos or informal spelling"#
    } else {
        ""
    };

    let prompt = format!(
r#"You are simulating a customer interacting with a support system. Generate the next customer message.

## Your persona:
- Personality: {personality}
- Sophistication: {sophistication} (how technical/precise your language is)
- Verbosity: {verbosity}
{adversarial}

## Available intents in the system:
{intents}

## Conversation so far:
{history}

## Instructions:
Generate a realistic customer message. You must also specify which intent(s) you are trying to express as ground truth.

{turn_guidance}

Return ONLY a JSON object:
{{
  "message": "the customer message text",
  "ground_truth": ["intent_id_1", "intent_id_2"],
  "intent_description": "brief note on what the customer wants"
}}

Rules:
- ground_truth must use exact intent IDs from the list above
- Use 1-3 intents per message (multi-intent is encouraged)
- Stay in character for your persona throughout
- If this is a follow-up turn, react naturally to the agent's previous response
- Return ONLY the JSON object"#,
        personality = req.personality,
        sophistication = req.sophistication,
        verbosity = req.verbosity,
        adversarial = adversarial_instructions,
        intents = intent_defs,
        history = history_text,
        turn_guidance = if req.history.is_empty() {
            "This is the opening message. Start a new conversation topic."
        } else {
            "Continue the conversation naturally. You may stick with the same topic, follow up, or pivot to a new request."
        },
    );

    let text = call_anthropic(&state, &prompt, 512).await?;

    let json_str = text.find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No JSON in LLM response".to_string()))?;

    let val: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Invalid JSON from LLM: {}", e)))?;

    Ok(Json(val))
}

#[derive(serde::Deserialize)]
struct SimulateRespondRequest {
    query: String,
    routed_intents: Vec<serde_json::Value>,
    history: Vec<serde_json::Value>,
}

async fn simulate_respond(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SimulateRespondRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let intent_defs = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
        let mut defs = Vec::new();
        for intent in &req.routed_intents {
            let id = intent["id"].as_str().unwrap_or("");
            let intent_type = router.get_intent_type(id);
            defs.push(format!("- {} ({:?}, score: {})", id, intent_type,
                intent["score"].as_f64().unwrap_or(0.0)));
        }
        defs.join("\n")
    };

    let history_text = if req.history.is_empty() {
        String::new()
    } else {
        let turns: Vec<String> = req.history.iter().map(|t| {
            format!("{}: {}", t["role"].as_str().unwrap_or("?"), t["message"].as_str().unwrap_or(""))
        }).collect();
        format!("\n## Previous conversation:\n{}", turns.join("\n"))
    };

    let prompt = format!(
r#"You are a helpful customer support agent. Respond to the customer's message based on the routing results.

## Customer message:
"{query}"

## Routing detected these intents:
{intents}
{history}

## Instructions:
- Respond naturally and helpfully to ALL detected intents
- Keep your response concise (2-4 sentences)
- If multiple intents were detected, address each one
- Use a professional but friendly tone

Return ONLY a JSON object:
{{
  "message": "your response to the customer"
}}"#,
        query = req.query,
        intents = intent_defs,
        history = history_text,
    );

    let text = call_anthropic(&state, &prompt, 512).await?;

    let json_str = text.find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No JSON in respond response".to_string()))?;

    let respond_val: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Invalid JSON from respond: {}", e)))?;

    Ok(Json(respond_val))
}

// =============================================================================
// Training Arena endpoints
// =============================================================================

#[derive(serde::Deserialize)]
struct TrainingGenerateRequest {
    personality: String,
    sophistication: String,
    verbosity: String,
    turns: usize,
    scenario: Option<String>,
}

async fn training_generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainingGenerateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let intent_defs = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
        let mut defs = Vec::new();
        let mut ids = router.intent_ids();
        ids.sort();
        for id in &ids {
            let seeds = router.get_training(id).unwrap_or_default();
            let intent_type = router.get_intent_type(id);
            defs.push(format!(
                "- {} ({:?}): {}",
                id, intent_type,
                seeds.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
            ));
        }
        defs.join("\n")
    };

    let scenario_section = if let Some(ref scenario) = req.scenario {
        format!("\n## Scenario description:\n{}\nGenerate a conversation that follows this scenario. The customer's messages should naturally express the intents described in the scenario.\n", scenario)
    } else {
        "\nGenerate a random customer support conversation. Pick different intents across turns to test variety.\n".to_string()
    };

    let prompt = format!(
r#"You are generating a simulated customer support conversation for testing an intent routing system.

## Customer persona:
- Personality: {personality}
- Sophistication: {sophistication} (how technical/precise their language is)
- Verbosity: {verbosity}
{scenario}
## Available intents in the system:
{intents}

## Instructions:
Generate a {turns}-turn conversation. For each turn, provide the customer message, what intents they are expressing (ground truth), and a brief agent response.

Return ONLY a JSON object:
{{
  "turns": [
    {{
      "customer_message": "the customer's message",
      "ground_truth": ["intent_id_1", "intent_id_2"],
      "intent_description": "brief note on what the customer wants",
      "agent_response": "the agent's helpful response (2-3 sentences)"
    }}
  ]
}}

Rules:
- ground_truth must use exact intent IDs from the list above
- Use 1-3 intents per turn (multi-intent is encouraged)
- Stay in character for the persona throughout ALL turns
- Each turn should build on or react to the previous agent response
- Make conversations realistic — customers don't always state things clearly
- Return ONLY the JSON object, no other text"#,
        personality = req.personality,
        sophistication = req.sophistication,
        verbosity = req.verbosity,
        scenario = scenario_section,
        intents = intent_defs,
        turns = req.turns,
    );

    let text = call_anthropic(&state, &prompt, 2048).await?;

    let json_str = text.find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No JSON in generate response".to_string()))?;

    let gen_val: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Invalid JSON from generate: {}", e)))?;

    Ok(Json(gen_val))
}

#[derive(serde::Deserialize)]
struct TrainingRunRequest {
    turns: Vec<TrainingTurn>,
}

#[derive(serde::Deserialize, Clone)]
struct TrainingTurn {
    message: String,
    ground_truth: Vec<String>,
}

async fn training_run(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainingRunRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let router = routers.get(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
    let mut results = Vec::new();

    for turn in &req.turns {
        let output = router.route_multi(&turn.message, 0.3);

        // Confirmed = high + medium confidence (paraphrase-confirmed)
        // Candidates = low confidence (routing-only, needs LLM verification)
        let confirmed: Vec<String> = output.intents.iter()
            .filter(|i| i.confidence != "low")
            .map(|i| i.id.clone())
            .collect();
        let candidates: Vec<String> = output.intents.iter()
            .filter(|i| i.confidence == "low")
            .map(|i| i.id.clone())
            .collect();

        let ground_set: std::collections::HashSet<&str> = turn.ground_truth.iter().map(|s| s.as_str()).collect();
        let confirmed_set: std::collections::HashSet<&str> = confirmed.iter().map(|s| s.as_str()).collect();
        let candidate_set: std::collections::HashSet<&str> = candidates.iter().map(|s| s.as_str()).collect();

        // Pass/fail: confirmed matches ground truth
        let matched: Vec<&str> = turn.ground_truth.iter().map(|s| s.as_str()).filter(|s| confirmed_set.contains(s)).collect();
        // Candidates that match GT — auto-promotable, not true misses
        let promotable: Vec<&str> = turn.ground_truth.iter().map(|s| s.as_str()).filter(|s| !confirmed_set.contains(s) && candidate_set.contains(s)).collect();
        // True misses — not in confirmed OR candidates
        let missed: Vec<&str> = turn.ground_truth.iter().map(|s| s.as_str()).filter(|s| !confirmed_set.contains(s) && !candidate_set.contains(s)).collect();
        let extra: Vec<&str> = confirmed.iter().map(|s| s.as_str()).filter(|s| !ground_set.contains(s)).collect();

        // Pass = all GT in confirmed, no extras. Promotable candidates don't count as misses.
        let status = if missed.is_empty() && promotable.is_empty() && extra.is_empty() {
            "pass"
        } else if missed.is_empty() && extra.is_empty() {
            // All GT found (confirmed + candidates), just needs promotion
            "promotable"
        } else if !matched.is_empty() {
            "partial"
        } else {
            "fail"
        };

        results.push(serde_json::json!({
            "message": turn.message,
            "ground_truth": turn.ground_truth,
            "confirmed": confirmed,
            "candidates": candidates,
            "matched": matched,
            "promotable": promotable,
            "missed": missed,
            "extra": extra,
            "status": status,
            "details": output.intents.iter().map(|i| serde_json::json!({
                "id": i.id,
                "score": (i.score * 100.0).round() / 100.0,
                "confidence": i.confidence,
                "source": i.source,
                "negated": i.negated,
            })).collect::<Vec<_>>(),
        }));
    }

    let pass_count = results.iter().filter(|r| r["status"] == "pass").count();
    let promotable_count = results.iter().filter(|r| r["status"] == "promotable").count();
    let detected_count = pass_count + promotable_count; // router found the right intents
    let total = results.len();
    Ok(Json(serde_json::json!({
        "results": results,
        "pass_count": detected_count,
        "confirmed_count": pass_count,
        "promotable_count": promotable_count,
        "total": total,
        "accuracy": if total == 0 { 0.0 } else { detected_count as f64 / total as f64 },
        "confirmed_rate": if total == 0 { 0.0 } else { pass_count as f64 / total as f64 },
    })))
}

#[derive(serde::Deserialize)]
struct TrainingReviewRequest {
    message: String,
    detected: Vec<serde_json::Value>,
    ground_truth: Vec<String>,
}

async fn training_review(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainingReviewRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let intent_seeds = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
        let mut relevant_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for gt in &req.ground_truth {
            relevant_ids.insert(gt.clone());
        }
        for d in &req.detected {
            if let Some(id) = d["id"].as_str() {
                relevant_ids.insert(id.to_string());
            }
        }
        let mut defs = Vec::new();
        for id in &relevant_ids {
            let seeds = router.get_training(id).unwrap_or_default();
            defs.push(format!(
                "- {}: {}",
                id,
                seeds.iter().take(5).cloned().collect::<Vec<_>>().join(" | ")
            ));
        }
        defs.join("\n")
    };

    let detected_str: Vec<String> = req.detected.iter().map(|d| {
        format!("{} (score: {})", d["id"].as_str().unwrap_or("?"), d["score"].as_f64().unwrap_or(0.0))
    }).collect();

    let prompt = format!(
r#"You are reviewing a failed intent routing result from ASV Router, a keyword-based intent classifier.

## Customer message:
"{message}"

## Ground truth intents (what the customer actually wants):
{ground_truth}

## What the router detected:
{detected}

## Relevant intent seeds (existing phrases the router knows):
{seeds}

## Your task:
For each MISSED intent (in ground truth but not detected), extract a short focused phrase (3-8 words) from the customer message that captures ONLY that intent. This phrase will be added as a new seed to teach the router.

CRITICAL RULES:
- ONLY use action "add_seed". No other action types.
- Each phrase must be SHORT (3-8 words) and contain ONLY words relevant to that one intent.
- Never use the full customer message — extract just the relevant fragment.
- Do NOT suggest corrections for false positives (extra detected intents). Ignore them.
- Only suggest seeds for MISSED intents, not for intents already detected.
- Use exact intent IDs from the lists above.

Example: If the message is "I got the wrong item and I want my money back and someone needs to call me"
and missed intents are [refund, contact_human]:
- add_seed "want my money back" → refund
- add_seed "someone needs to call me" → contact_human

Return ONLY a JSON object:
{{
  "analysis": "brief explanation of what was missed and why",
  "corrections": [
    {{
      "action": "add_seed",
      "phrase": "short focused phrase from the message",
      "intent": "missed_intent_id"
    }}
  ]
}}"#,
        message = req.message,
        ground_truth = req.ground_truth.join(", "),
        detected = if detected_str.is_empty() { "nothing detected".to_string() } else { detected_str.join(", ") },
        seeds = intent_seeds,
    );

    let text = call_anthropic(&state, &prompt, 1024).await?;

    let json_str = text.find('{')
        .and_then(|start| text.rfind('}').map(|end| &text[start..=end]))
        .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No JSON in review response".to_string()))?;

    let review_result: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Invalid JSON from review: {}", e)))?;

    Ok(Json(review_result))
}

#[derive(serde::Deserialize)]
struct TrainingApplyRequest {
    corrections: Vec<serde_json::Value>,
}

async fn training_apply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TrainingApplyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
    router.begin_batch();

    let mut applied = 0;
    let mut errors = Vec::new();

    for correction in &req.corrections {
        let action = correction["action"].as_str().unwrap_or("");
        match action {
            "add_seed" => {
                let phrase = correction["phrase"].as_str().unwrap_or("");
                let intent = correction["intent"].as_str().unwrap_or("");
                if !phrase.is_empty() && !intent.is_empty() {
                    router.learn(phrase, intent);
                    applied += 1;
                } else {
                    errors.push("add_seed: missing phrase or intent".to_string());
                }
            }
            _ => {
                errors.push(format!("ignored action: {} (only add_seed allowed)", action));
            }
        }
    }

    router.end_batch();
    maybe_persist(&state, &app_id, router);

    Ok(Json(serde_json::json!({
        "applied": applied,
        "errors": errors,
    })))
}

// =============================================================================
// Multi-app management endpoints
// =============================================================================

async fn list_apps(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let routers = state.routers.read().unwrap();
    let apps: Vec<&String> = routers.keys().collect();
    Json(serde_json::json!(apps))
}

#[derive(serde::Deserialize)]
struct CreateAppRequest {
    app_id: String,
}

async fn create_app(
    State(state): State<AppState>,
    Json(req): Json<CreateAppRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut routers = state.routers.write().unwrap();
    if routers.contains_key(&req.app_id) {
        return Err((StatusCode::CONFLICT, format!("app '{}' already exists", req.app_id)));
    }
    let router = Router::new();
    maybe_persist(&state, &req.app_id, &router);
    routers.insert(req.app_id.clone(), router);
    Ok(Json(serde_json::json!({"created": req.app_id})))
}

#[derive(serde::Deserialize)]
struct DeleteAppRequest {
    app_id: String,
}

async fn delete_app(
    State(state): State<AppState>,
    Json(req): Json<DeleteAppRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if req.app_id == "default" {
        return Err((StatusCode::BAD_REQUEST, "cannot delete default app".to_string()));
    }
    let mut routers = state.routers.write().unwrap();
    if routers.remove(&req.app_id).is_none() {
        return Err((StatusCode::NOT_FOUND, format!("app '{}' not found", req.app_id)));
    }
    // Remove persisted file
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}.json", dir, req.app_id);
        let _ = std::fs::remove_file(&path);
    }
    Ok(Json(serde_json::json!({"deleted": req.app_id})))
}

// =============================================================================
// Discovery endpoints
// =============================================================================

#[derive(serde::Deserialize)]
struct DiscoverRequest {
    queries: Vec<String>,
    #[serde(default)]
    expected_intents: usize,
}

async fn discover(
    State(state): State<AppState>,
    Json(req): Json<DiscoverRequest>,
) -> Json<serde_json::Value> {
    let config = asv_router::discovery::DiscoveryConfig {
        expected_intents: req.expected_intents,
        ..Default::default()
    };
    let clusters = asv_router::discovery::discover_intents(&req.queries, &config);

    // LLM naming: send representative queries to Claude for each cluster
    let mut clusters_json: Vec<serde_json::Value> = Vec::new();
    for c in &clusters {
        let samples: Vec<&str> = c.representative_queries.iter().take(5).map(|s| s.as_str()).collect();
        let mut name = c.suggested_name.clone();
        let mut description = String::new();

        if state.anthropic_key.is_some() {
            let prompt = format!(
                "These user queries all belong to the same intent category:\n{}\n\n\
                 Respond with ONLY a JSON object (no markdown, no explanation):\n\
                 {{\"name\": \"snake_case_intent_name\", \"description\": \"one sentence description of what the user wants\"}}",
                samples.iter().enumerate()
                    .map(|(i, q)| format!("{}. {}", i + 1, q))
                    .collect::<Vec<_>>().join("\n")
            );
            if let Ok(response) = call_anthropic(&state, &prompt, 100).await {
                // Parse the JSON response
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response) {
                    if let Some(n) = parsed["name"].as_str() {
                        name = n.to_string();
                    }
                    if let Some(d) = parsed["description"].as_str() {
                        description = d.to_string();
                    }
                }
            }
        }

        clusters_json.push(serde_json::json!({
            "suggested_name": name,
            "description": description,
            "top_terms": c.top_terms,
            "representative_queries": c.representative_queries,
            "size": c.size,
            "confidence": (c.confidence * 100.0).round() / 100.0,
        }));
    }

    let total: usize = clusters.iter().map(|c| c.size).sum();
    Json(serde_json::json!({
        "clusters": clusters_json,
        "total_clusters": clusters.len(),
        "total_assigned": total,
        "total_queries": req.queries.len(),
    }))
}

#[derive(serde::Deserialize)]
struct DiscoverApplyRequest {
    clusters: Vec<DiscoverApplyCluster>,
}

#[derive(serde::Deserialize)]
struct DiscoverApplyCluster {
    name: String,
    representative_queries: Vec<String>,
}

async fn discover_apply(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DiscoverApplyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or((StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;

    let mut created = Vec::new();
    router.begin_batch();
    for cluster in &req.clusters {
        let seeds: Vec<&str> = cluster.representative_queries.iter().map(|s| s.as_str()).collect();
        router.add_intent(&cluster.name, &seeds);
        created.push(cluster.name.clone());
    }
    router.end_batch();

    maybe_persist(&state, &app_id, router);

    Ok(Json(serde_json::json!({
        "created": created,
        "count": created.len(),
    })))
}
