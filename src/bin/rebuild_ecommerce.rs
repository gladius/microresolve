//! Rebuild ecommerce-demo.json from diverse seeds.
//! Run: cargo run --release --bin rebuild_ecommerce

use asv_router::{Router, IntentType};
use std::collections::HashMap;

fn main() {
    let seeds: HashMap<String, Vec<String>> = serde_json::from_str(
        &std::fs::read_to_string("data/ecommerce-demo-seeds.json").unwrap()
    ).unwrap();

    let mut router = Router::new();

    // Action intents
    let actions = [
        "cancel_order", "refund", "contact_human", "billing_issue",
        "return_item", "change_order", "payment_method", "subscription",
    ];
    // Context intents
    let contexts = [
        "track_order", "shipping_complaint", "product_inquiry",
        "order_status", "account_issue", "feedback",
    ];

    for (id, phrases) in &seeds {
        let refs: Vec<&str> = phrases.iter().map(|s| s.as_str()).collect();
        router.add_intent(id, &refs);

        if actions.contains(&id.as_str()) {
            router.set_intent_type(id, IntentType::Action);
        } else if contexts.contains(&id.as_str()) {
            router.set_intent_type(id, IntentType::Context);
        }
    }

    router.save("data/ecommerce-demo.json").unwrap();
    println!("Rebuilt data/ecommerce-demo.json with {} intents", seeds.len());

    // Quick verification
    let loaded = Router::load("data/ecommerce-demo.json").unwrap();
    let ids = loaded.intent_ids();
    println!("Verified: {} intents loaded back", ids.len());
    for id in &ids {
        let count = loaded.get_training(id).unwrap_or_default().len();
        println!("  {} ({} seeds)", id, count);
    }
}
