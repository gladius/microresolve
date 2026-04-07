//! Basic Rust usage: create router, add intents, route, learn.
//!
//! Run: cargo run --example rust_basic

use asv_router::{IntentType, Router};

fn main() {
    let mut router = Router::new();

    // Batch mode for faster setup
    router.begin_batch();

    // Action intents
    router.add_intent("cancel_order", &[
        "cancel my order",
        "I want to cancel",
        "stop my order from shipping",
    ]);
    router.add_intent("track_order", &[
        "where is my package",
        "track my order",
        "shipping status update",
    ]);
    router.add_intent("refund", &[
        "I want a refund",
        "get my money back",
        "return and refund",
    ]);

    // Context intent
    router.add_intent("order_history", &[
        "my past orders",
        "order history",
        "previous purchases",
    ]);
    router.set_intent_type("order_history", IntentType::Context);

    router.end_batch();

    // --- Single routing ---
    println!("=== Single routing ===");
    let results = router.route("I need to cancel something");
    for r in &results {
        println!("  {} (score: {:.2})", r.id, r.score);
    }

    // --- Multi-intent routing ---
    println!("\n=== Multi-intent ===");
    let multi = router.route_multi("cancel my order and show me my past orders", 0.3);
    for intent in &multi.intents {
        println!("  {} (score: {:.2}, type: {:?}, confidence: {:?})",
            intent.id, intent.score, intent.intent_type, intent.confidence);
    }
    for rel in &multi.relations {
        println!("  relation: {:?}", rel);
    }

    // --- Learning ---
    println!("\n=== Learning ===");
    let before = router.route("stop charging me");
    println!("  before learn: {:?}", before.first().map(|r| (&r.id, r.score)));

    router.learn("stop charging me", "cancel_order");

    let after = router.route("stop charging me");
    println!("  after learn:  {:?}", after.first().map(|r| (&r.id, r.score)));

    // --- Export / Import ---
    println!("\n=== Export/Import ===");
    let json = router.export_json();
    println!("  exported: {} bytes", json.len());

    let imported = Router::import_json(&json).unwrap();
    let result = imported.route("cancel this");
    println!("  imported route: {:?}", result.first().map(|r| &r.id));

    // --- Discovery ---
    println!("\n=== Discovery ===");
    let queries: Vec<String> = vec![
        "cancel my order", "I want to cancel", "stop my order",
        "cancel the purchase", "cancel it please", "undo my order",
        "where is my package", "track order", "shipping update",
        "track my delivery", "order tracking", "delivery status",
    ].into_iter().cycle().take(200).map(String::from).collect();

    let config = asv_router::discovery::DiscoveryConfig::default();
    let clusters = asv_router::discovery::discover_intents(&queries, &config);
    println!("  discovered {} clusters from {} queries", clusters.len(), queries.len());
    for c in &clusters {
        println!("    {} (size: {}, terms: {:?})", c.suggested_name, c.size, &c.top_terms[..c.top_terms.len().min(3)]);
    }
}
