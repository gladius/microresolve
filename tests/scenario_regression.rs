//! Scenario regression tests: 30 multi-turn customer conversations (138 turns).
//!
//! These scenarios test the full routing pipeline against realistic queries
//! spanning frustrated customers, multilingual input, multi-intent, and edge cases.
//! Extracted from the scenario_test experiment binary.

use asv_router::{IntentType, Router};
use std::collections::HashSet;

#[derive(serde::Deserialize)]
struct Scenario {
    id: String,
    turns: Vec<Turn>,
}

#[derive(serde::Deserialize)]
struct Turn {
    message: String,
    ground_truth: Vec<String>,
}

fn build_router() -> Router {
    let mut r = Router::new();
    r.begin_batch();

    // Actions
    r.add_intent("cancel_order", &["cancel my order", "I want to cancel", "stop my order", "cancel purchase"]);
    r.add_intent("track_order", &["where is my package", "track my order", "shipping status", "order tracking"]);
    r.add_intent("refund", &["I want a refund", "get my money back", "refund my purchase", "return and refund"]);
    r.add_intent("billing_issue", &["charged twice", "wrong amount", "billing problem", "double charged"]);
    r.add_intent("change_address", &["update my address", "change shipping address", "new delivery address"]);
    r.add_intent("change_order", &["modify my order", "change my order", "update order items"]);
    r.add_intent("contact_human", &["talk to a person", "speak to agent", "human representative", "connect me to someone"]);
    r.add_intent("schedule_callback", &["call me back", "schedule a callback", "request callback"]);
    r.add_intent("password_reset", &["reset my password", "forgot password", "can't log in", "change password"]);
    r.add_intent("product_inquiry", &["tell me about this product", "product information", "what does this do"]);
    r.add_intent("feedback", &["I want to give feedback", "complaint", "suggestion", "rate service"]);
    r.add_intent("upgrade_plan", &["upgrade my plan", "better plan", "premium subscription"]);
    r.add_intent("downgrade_plan", &["downgrade my plan", "cheaper plan", "reduce subscription"]);
    r.add_intent("loyalty_program", &["loyalty points", "rewards program", "member benefits"]);
    r.add_intent("gift_card", &["buy a gift card", "gift card balance", "redeem gift card"]);
    r.add_intent("payment_method", &["add credit card", "change payment", "update payment method"]);
    r.add_intent("set_language", &["change language", "switch to spanish", "set language preference"]);
    r.add_intent("newsletter", &["subscribe to newsletter", "email updates", "unsubscribe from emails"]);
    r.add_intent("store_hours", &["what are your hours", "when do you open", "store hours"]);
    r.add_intent("warranty", &["warranty information", "warranty claim", "is this under warranty"]);

    // Context
    r.add_intent("check_balance", &["check my balance", "account balance", "how much do I owe"]);
    r.add_intent("order_history", &["my past orders", "order history", "previous purchases"]);
    r.add_intent("account_info", &["my account details", "account information", "profile info"]);
    r.add_intent("shipping_options", &["shipping methods", "delivery options", "how to ship"]);
    r.add_intent("return_policy", &["return policy", "can I return this", "return window"]);
    r.add_intent("promotions", &["current deals", "any promotions", "discount codes"]);

    r.end_batch();

    // Set context types
    for id in &["check_balance", "order_history", "account_info", "shipping_options", "return_policy", "promotions"] {
        r.set_intent_type(id, IntentType::Context);
    }

    r
}

#[test]
fn scenario_regression_baseline() {
    let path = "tests/scenarios/scenarios.json";
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => { eprintln!("Skipping: {} not found", path); return; }
    };
    let scenarios: Vec<Scenario> = serde_json::from_str(&data).unwrap();

    let router = build_router();
    let mut total_turns = 0;
    let mut pass = 0;
    let mut partial = 0;

    for scenario in &scenarios {
        for turn in &scenario.turns {
            total_turns += 1;
            let output = router.route_multi(&turn.message, 0.3);
            let detected: HashSet<String> = output.intents.iter()
                .map(|i| i.id.clone()).collect();
            let gt: HashSet<String> = turn.ground_truth.iter().cloned().collect();

            let matched = gt.intersection(&detected).count();
            if matched == gt.len() {
                pass += 1;
            } else if matched > 0 {
                partial += 1;
            }
        }
    }

    let accuracy = pass as f64 / total_turns as f64 * 100.0;
    let partial_rate = (pass + partial) as f64 / total_turns as f64 * 100.0;

    println!("\n  Scenario regression: {}/{} pass ({:.1}%), {}/{} partial+ ({:.1}%)",
        pass, total_turns, accuracy, pass + partial, total_turns, partial_rate);

    // Baseline: 23.2% exact, 37.0% partial (April 2026, with basic seeds only)
    // These thresholds should only go UP as we improve the router
    assert!(accuracy >= 20.0,
        "Regression: accuracy {:.1}% below 20% baseline", accuracy);
    assert!(partial_rate >= 30.0,
        "Regression: partial rate {:.1}% below 30% baseline", partial_rate);
}
