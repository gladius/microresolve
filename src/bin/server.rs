//! ASV Router HTTP API server.
//!
//! Run with: cargo run --bin server --features server --release
//!
//! Default: http://localhost:3001

use asv_router::{Router, IntentType};
use axum::{
    extract::{State, Query},
    http::StatusCode,
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
    router: RwLock<Router>,
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

#[tokio::main]
async fn main() {
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

    let state: AppState = Arc::new(ServerState {
        router: RwLock::new(Router::new()),
        log: Mutex::new(open_log()),
        http: reqwest::Client::new(),
        anthropic_key,
    });

    let app = axum::Router::new()
        .route("/api/health", get(health))
        .route("/api/route", post(route_query))
        .route("/api/route_multi", post(route_multi))
        .route("/api/intents", get(list_intents))
        .route("/api/intents", post(add_intent))
        .route("/api/intents/delete", post(delete_intent))
        .route("/api/intents/add_seed", post(add_seed))
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
        .layer(CorsLayer::permissive())
        .with_state(state);

    println!("ASV Router server listening on {}", addr);
    println!("Query log: {}", LOG_FILE);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str {
    "ok"
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
    Json(req): Json<RouteRequest>,
) -> Json<serde_json::Value> {
    let router = state.router.read().unwrap();
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
    Json(req): Json<RouteMultiRequest>,
) -> Json<serde_json::Value> {
    let t0 = std::time::Instant::now();
    let output = {
        let router = state.router.read().unwrap();
        router.route_multi(&req.query, req.threshold)
    };
    let latency_us = t0.elapsed().as_micros() as u64;

    // Record co-occurrence — only strong intents (>= 35% of top score)
    // to prevent noise from incidental term overlap polluting projections
    let strong = output.strong_intents(0.35);
    if strong.len() > 1 {
        let ids: Vec<&str> = strong.iter().map(|i| i.id.as_str()).collect();
        if let Ok(mut router) = state.router.write() {
            router.record_co_occurrence(&ids);
        }
    }

    let intents: Vec<serde_json::Value> = strong
        .iter()
        .map(|i| {
            serde_json::json!({
                "id": i.id,
                "score": (i.score * 100.0).round() / 100.0,
                "position": i.position,
                "span": [i.span.0, i.span.1],
                "intent_type": i.intent_type
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
        let router = state.router.read().unwrap();
        let co_pairs = router.get_co_occurrence();
        let strong_refs = output.strong_intents(0.35);
        let matched_ids: std::collections::HashSet<&str> = strong_refs.iter().map(|i| i.id.as_str()).collect();

        // For each matched action intent, find context intents that co-occur but aren't already in results
        let mut context_scores: HashMap<String, (u32, u32)> = HashMap::new(); // id -> (co_count, total_action_count)

        for intent in &strong_refs {
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

    let result = serde_json::json!({
        "intents": intents,
        "relations": relations,
        "metadata": output.metadata,
        "projected_context": projected_context
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

async fn list_intents(State(state): State<AppState>) -> Json<serde_json::Value> {
    let router = state.router.read().unwrap();
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
    Json(req): Json<AddIntentRequest>,
) -> StatusCode {
    let mut router = state.router.write().unwrap();
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
    StatusCode::CREATED
}

#[derive(serde::Deserialize)]
struct AddSeedRequest {
    intent_id: String,
    seed: String,
}

async fn add_seed(
    State(state): State<AppState>,
    Json(req): Json<AddSeedRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut router = state.router.write().unwrap();
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
    Ok(StatusCode::OK)
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
    Json(req): Json<AddIntentMultilingualRequest>,
) -> StatusCode {
    let mut router = state.router.write().unwrap();
    router.add_intent_multilingual(&req.id, req.seeds_by_lang);
    if let Some(t) = req.intent_type {
        router.set_intent_type(&req.id, t);
    }
    if let Some(meta) = req.metadata {
        for (key, values) in meta {
            router.set_metadata(&req.id, &key, values);
        }
    }
    StatusCode::CREATED
}

#[derive(serde::Deserialize)]
struct SetIntentTypeRequest {
    intent_id: String,
    intent_type: IntentType,
}

async fn set_intent_type(
    State(state): State<AppState>,
    Json(req): Json<SetIntentTypeRequest>,
) -> StatusCode {
    let mut router = state.router.write().unwrap();
    router.set_intent_type(&req.intent_id, req.intent_type);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
struct DeleteIntentRequest {
    id: String,
}

async fn delete_intent(
    State(state): State<AppState>,
    Json(req): Json<DeleteIntentRequest>,
) -> StatusCode {
    let mut router = state.router.write().unwrap();
    router.remove_intent(&req.id);
    StatusCode::OK
}

// --- Learn / Correct ---

#[derive(serde::Deserialize)]
struct LearnRequest {
    query: String,
    intent_id: String,
}

async fn learn(State(state): State<AppState>, Json(req): Json<LearnRequest>) -> StatusCode {
    let mut router = state.router.write().unwrap();
    router.learn(&req.query, &req.intent_id);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
struct CorrectRequest {
    query: String,
    wrong_intent: String,
    correct_intent: String,
}

async fn correct(State(state): State<AppState>, Json(req): Json<CorrectRequest>) -> StatusCode {
    let mut router = state.router.write().unwrap();
    router.correct(&req.query, &req.wrong_intent, &req.correct_intent);
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
    Json(req): Json<SetMetadataRequest>,
) -> StatusCode {
    let mut router = state.router.write().unwrap();
    router.set_metadata(&req.intent_id, &req.key, req.values);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
struct GetMetadataRequest {
    intent_id: String,
}

async fn get_metadata(
    State(state): State<AppState>,
    Json(req): Json<GetMetadataRequest>,
) -> Json<serde_json::Value> {
    let router = state.router.read().unwrap();
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

async fn reset(State(state): State<AppState>) -> StatusCode {
    let mut router = state.router.write().unwrap();
    *router = Router::new();
    StatusCode::OK
}

// --- Load defaults ---

async fn load_defaults(State(state): State<AppState>) -> StatusCode {
    let mut router = state.router.write().unwrap();

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

    for (id, seeds) in actions {
        router.add_intent(id, seeds);
        router.set_intent_type(id, IntentType::Action);
    }
    for (id, seeds) in contexts {
        router.add_intent(id, seeds);
        router.set_intent_type(id, IntentType::Context);
    }

    // No hardcoded metadata — let co-occurrence emerge naturally

    StatusCode::OK
}

// --- Export / Import ---

async fn export_state(State(state): State<AppState>) -> String {
    let router = state.router.read().unwrap();
    router.export_json()
}

async fn import_state(
    State(state): State<AppState>,
    body: String,
) -> Result<StatusCode, (StatusCode, String)> {
    let new_router =
        Router::import_json(&body).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let mut router = state.router.write().unwrap();
    *router = new_router;
    Ok(StatusCode::OK)
}

// --- Languages ---

async fn get_languages() -> Json<serde_json::Value> {
    let json_str = asv_router::seed::supported_languages_json();
    let val: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_default();
    Json(val)
}

// --- Co-occurrence ---

async fn get_co_occurrence(State(state): State<AppState>) -> Json<serde_json::Value> {
    let router = state.router.read().unwrap();
    let pairs = router.get_co_occurrence();
    let out: Vec<serde_json::Value> = pairs.iter().map(|(a, b, count)| {
        serde_json::json!({"a": a, "b": b, "count": count})
    }).collect();
    Json(serde_json::json!(out))
}

// --- Projections: full action → context map ---

async fn get_projections(State(state): State<AppState>) -> Json<serde_json::Value> {
    let router = state.router.read().unwrap();
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
    Json(req): Json<ReviewPromptRequest>,
) -> Json<serde_json::Value> {
    let router = state.router.read().unwrap();

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
    Json(req): Json<ReviewPromptRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let prompt = {
        let router = state.router.read().unwrap();
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
