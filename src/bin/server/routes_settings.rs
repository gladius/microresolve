//! Settings: reset, defaults, export/import, languages, analytics data.

use axum::{
    extract::{State, Query},
    http::{StatusCode, HeaderMap},
    routing::{get, post, delete},
    Json,
};
use std::collections::HashMap;
use asv_router::{Router, IntentType};
use crate::state::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/reset", post(reset))
        .route("/api/intents/load_defaults", post(load_defaults))
        .route("/api/export", get(export_state))
        .route("/api/import", post(import_state))
        .route("/api/languages", get(get_languages))
        .route("/api/co_occurrence", get(get_co_occurrence))
        .route("/api/projections", get(get_projections))
        .route("/api/workflows", get(get_workflows))
        .route("/api/temporal_order", get(get_temporal_order))
        .route("/api/escalation_patterns", get(get_escalation_patterns))
}

pub async fn reset(State(state): State<AppState>, headers: HeaderMap) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = Router::new();
    maybe_persist(&state, &app_id, &router);
    routers.insert(app_id, router);
    StatusCode::OK
}

// --- Load defaults ---

pub async fn load_defaults(State(state): State<AppState>, headers: HeaderMap) -> StatusCode {
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

pub async fn export_state(State(state): State<AppState>, headers: HeaderMap) -> String {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    match routers.get(&app_id) {
        Some(router) => router.export_json(),
        None => format!("{{\"error\": \"app '{}' not found\"}}", app_id),
    }
}

pub async fn import_state(
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

pub async fn get_languages() -> Json<serde_json::Value> {
    let json_str = asv_router::seed::supported_languages_json();
    let val: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_default();
    Json(val)
}

// --- Co-occurrence ---

pub async fn get_co_occurrence(State(state): State<AppState>, headers: HeaderMap) -> Json<serde_json::Value> {
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
pub struct WorkflowQuery {
    #[serde(default = "default_min_obs")]
    min_observations: u32,
}
pub fn default_min_obs() -> u32 { 3 }

pub async fn get_workflows(
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

pub async fn get_temporal_order(State(state): State<AppState>, headers: HeaderMap) -> Json<serde_json::Value> {
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
pub struct EscalationQuery {
    #[serde(default = "default_min_occ")]
    min_occurrences: u32,
}
pub fn default_min_occ() -> u32 { 2 }

pub async fn get_escalation_patterns(
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

pub async fn get_projections(State(state): State<AppState>, headers: HeaderMap) -> Json<serde_json::Value> {
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

