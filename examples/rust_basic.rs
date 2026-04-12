//! Basic Rust example — Router is now a storage registry only.
//! Routing is handled by Hebbian L1+L3 in the server (POST /api/route_multi).
//!
//! Run: cargo run --example rust_basic

use asv_router::{IntentType, Router};

fn main() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &["cancel my order", "stop my subscription"]);
    router.add_intent("track_order", &["where is my order", "track my package"]);
    router.set_intent_type("cancel_order", IntentType::Action);
    router.set_intent_type("track_order", IntentType::Action);

    println!("Intents: {:?}", router.intent_ids());
    println!("cancel_order phrases: {:?}", router.get_training("cancel_order"));
    println!("\nRouting is handled by Hebbian L3 — start the server and POST to /api/route_multi");
}
