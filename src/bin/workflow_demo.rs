//! Simulate realistic customer traffic and show emergent workflow discovery.
//! Run: cargo run --bin workflow_demo

use asv_router::Router;

fn main() {
    let mut router = Router::new();

    // Set up intents (subset of realistic customer support)
    router.add_intent("cancel_order", &["cancel my order", "cancel order", "stop my order", "withdraw my order"]);
    router.add_intent("refund", &["I want a refund", "get my money back", "return for refund", "money back"]);
    router.add_intent("track_order", &["track my order", "where is my package", "shipping status", "delivery status"]);
    router.add_intent("contact_human", &["talk to a human", "speak to agent", "live agent", "escalate to manager"]);
    router.add_intent("file_complaint", &["file a complaint", "formal complaint", "report poor service", "terrible service"]);
    router.add_intent("billing_issue", &["wrong charge", "billing error", "overcharged", "charged twice"]);
    router.add_intent("update_address", &["change my address", "update shipping address", "wrong address"]);
    router.add_intent("reset_password", &["reset my password", "forgot password", "locked out", "can't log in"]);
    router.add_intent("change_plan", &["upgrade my plan", "downgrade plan", "switch subscription"]);
    router.add_intent("close_account", &["delete my account", "close account", "deactivate account"]);
    router.add_intent("apply_coupon", &["apply discount code", "use coupon", "promo code"]);
    router.add_intent("report_fraud", &["unauthorized charge", "account hacked", "fraudulent activity"]);
    router.add_intent("upgrade_shipping", &["express shipping", "expedite shipment", "faster delivery"]);
    router.add_intent("check_balance", &["check my balance", "account balance", "how much credit"]);
    router.add_intent("reorder", &["reorder last purchase", "buy same thing again", "repeat order"]);

    // === Simulate realistic customer traffic patterns ===
    // These represent real multi-intent queries that would come in

    println!("=== Simulating Customer Traffic ===\n");

    // Pattern 1: Cancel → Refund (very common — unhappy purchase)
    let cancel_refund_queries = [
        "cancel my order and get a refund",
        "I want to cancel and get my money back",
        "cancel the order and refund me please",
        "stop my order I want my money back",
        "cancel order and I want a refund",
        "please cancel and process a refund",
        "I need to cancel this and get a full refund",
        "withdraw my order and refund the payment",
        "cancel everything and give me money back",
        "I want to cancel my purchase and get refund",
        "cancel my order immediately and refund me",
        "stop the order and return my money",
    ];

    // Pattern 2: Track → File complaint (package missing → frustrated)
    let track_complaint_queries = [
        "where is my package this is terrible service",
        "track my order I want to file a complaint",
        "shipping status is showing nothing, terrible service",
        "I can't find my delivery, this is unacceptable",
        "where is my package I want to report poor service",
        "track order and file a formal complaint",
        "my package is lost and your service is terrible",
    ];

    // Pattern 3: Track → Contact human (can't find package → need help)
    let track_human_queries = [
        "where is my package I need to speak to agent",
        "track my order and connect me to someone",
        "shipping status unknown, talk to a human please",
        "I can't find my delivery let me speak to agent",
        "where is my package I want to talk to someone",
        "track order and escalate to manager",
    ];

    // Pattern 4: Complaint → Contact human (angry → escalate)
    let complaint_human_queries = [
        "I want to file a complaint and speak to manager",
        "terrible service escalate to manager",
        "formal complaint and connect me to agent",
        "this is unacceptable talk to a human",
        "report poor service and escalate",
        "file complaint and speak to real person",
        "your service is terrible get me an agent",
        "I want to complain and talk to someone in charge",
    ];

    // Pattern 5: Billing → Refund (charged wrong → want money back)
    let billing_refund_queries = [
        "wrong charge on my account and I want refund",
        "you overcharged me give me my money back",
        "billing error I need a refund",
        "charged twice I want my money back",
        "there's a billing error refund the extra amount",
    ];

    // Pattern 6: Cancel → Close account (leaving entirely)
    let cancel_close_queries = [
        "cancel my order and delete my account",
        "cancel everything and close my account",
        "stop my order and deactivate account",
        "I want to cancel and close my account permanently",
    ];

    // Pattern 7: Update address → Upgrade shipping (moving, need fast delivery)
    let address_shipping_queries = [
        "change my address and upgrade to express shipping",
        "update shipping address and expedite shipment",
        "wrong address please fix and faster delivery",
        "new address and express shipping please",
    ];

    // Pattern 8: Report fraud → Reset password → Contact human (security breach escalation)
    let fraud_reset_human_queries = [
        "my account was hacked I need to reset password and talk to someone",
        "unauthorized charge reset my password and connect me to agent",
        "fraudulent activity I forgot my password speak to agent",
    ];

    // Pattern 9: Check balance → Apply coupon → Reorder (returning customer)
    let balance_coupon_reorder = [
        "check my balance apply discount code and reorder last purchase",
        "how much credit do I have use coupon and buy same thing again",
        "account balance and apply promo code and repeat order",
    ];

    // Pattern 10: Billing → File complaint (dispute → angry)
    let billing_complaint_queries = [
        "wrong charge and I want to file a complaint",
        "billing error this is terrible service",
        "overcharged and I want to report poor service",
        "charged twice this is unacceptable",
    ];

    // Route all queries and record sequences
    let all_traffic: Vec<(&str, &[&str])> = vec![
        ("cancel→refund", &cancel_refund_queries[..]),
        ("track→complaint", &track_complaint_queries[..]),
        ("track→human", &track_human_queries[..]),
        ("complaint→human", &complaint_human_queries[..]),
        ("billing→refund", &billing_refund_queries[..]),
        ("cancel→close", &cancel_close_queries[..]),
        ("address→shipping", &address_shipping_queries[..]),
        ("fraud→reset→human", &fraud_reset_human_queries[..]),
        ("balance→coupon→reorder", &balance_coupon_reorder[..]),
        ("billing→complaint", &billing_complaint_queries[..]),
    ];

    let mut total_queries = 0;
    let mut multi_detections = 0;

    for (label, queries) in &all_traffic {
        for query in *queries {
            let output = router.route_multi(query, 0.3);
            let ids: Vec<&str> = output.intents.iter().map(|i| i.id.as_str()).collect();

            if ids.len() >= 2 {
                router.record_intent_sequence(&ids);
                multi_detections += 1;
            }
            total_queries += 1;
        }
        println!("  Traffic batch '{}': {} queries", label, queries.len());
    }

    println!("\nTotal queries: {}, Multi-intent detections: {}\n", total_queries, multi_detections);

    // === Show Results ===

    println!("========================================");
    println!("  EMERGENT WORKFLOW DISCOVERY RESULTS");
    println!("========================================\n");

    // 1. Temporal Ordering
    println!("--- Temporal Ordering (which intent comes FIRST) ---\n");
    let order = router.get_temporal_order();
    for (first, second, prob, count) in &order {
        if *count >= 2 {
            println!("  {} → {}  (P={:.0}%, observed {} times)",
                first, second, prob * 100.0, count);
        }
    }

    // 2. Workflow Clusters
    println!("\n--- Discovered Workflow Clusters ---\n");
    let workflows = router.discover_workflows(2);
    for (i, cluster) in workflows.iter().enumerate() {
        println!("  Workflow {}: {} intents", i + 1, cluster.len());
        for wi in cluster {
            println!("    {} (connections: {}, neighbors: {:?})",
                wi.id, wi.connections, wi.neighbors);
        }
        println!();
    }

    // 3. Escalation Patterns
    println!("--- Escalation Patterns (recurring sequences) ---\n");
    let patterns = router.detect_escalation_patterns(2);
    let mut shown = 0;
    for p in &patterns {
        if shown >= 20 { break; }
        let arrow = p.sequence.join(" → ");
        println!("  {} ({}x, {:.0}% of traffic)",
            arrow, p.occurrences, p.frequency * 100.0);
        shown += 1;
    }

    // 4. Co-occurrence suggestions
    println!("\n--- Proactive Suggestions (\"customers who X also Y\") ---\n");
    let test_queries = [
        "cancel my order",
        "where is my package",
        "wrong charge on my account",
        "I want to file a complaint",
        "my account was hacked",
    ];
    for query in &test_queries {
        let output = router.route_multi(query, 0.3);
        if !output.suggestions.is_empty() {
            let detected: Vec<&str> = output.intents.iter().map(|i| i.id.as_str()).collect();
            println!("  Query: \"{}\"", query);
            println!("  Detected: {:?}", detected);
            for s in &output.suggestions {
                println!("    → Suggest: {} (P={:.0}%, {} obs, because: {})",
                    s.id, s.probability * 100.0, s.observations, s.because_of);
            }
            println!();
        }
    }

    println!("========================================");
    println!("  Done. All patterns discovered from {} queries.", total_queries);
    println!("========================================");
}
