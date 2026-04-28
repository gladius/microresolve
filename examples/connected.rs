//! End-to-end demo of MicroResolve connected mode.
//!
//! Shows the live-learning loop:
//!   1. Library connects to a running server and pulls a namespace
//!   2. Routes a query — gets a confident-but-wrong answer
//!   3. Pushes an explicit correction (local + server) via Engine.correct()
//!   4. Same query now routes correctly — and any other connected
//!      library subscribed to this namespace will pick up the change
//!      on its next sync tick.
//!
//! Prerequisites:
//!   $ cargo build --release --features server
//!   $ ./target/release/server --port 3001 --no-open &
//!
//! Run:
//!   $ cargo run --release --example connected --features connect
//!
//! With auth (after generating a key in the UI):
//!   $ MICRORESOLVE_API_KEY=mr_xxx... cargo run --release --example connected --features connect

use std::time::Duration;
use microresolve::{Engine, EngineConfig, ServerConfig};

const NS: &str = "demo-connected";
const SERVER: &str = "http://localhost:3001";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("MICRORESOLVE_API_KEY").ok();

    println!("─── 1. Setup namespace + intents on server ────────────────────");
    setup_namespace(api_key.as_deref())?;

    println!("\n─── 2. Connect Engine to server ───────────────────────────────");
    let engine = Engine::new(EngineConfig {
        server: Some(ServerConfig {
            url: SERVER.to_string(),
            api_key: api_key.clone(),
            subscribe: vec![NS.to_string()],
            tick_interval_secs: 5,   // poll every 5s for snappier demo
            log_buffer_max: 500,
        }),
        ..Default::default()
    })?;
    let ns = engine.namespace(NS);
    println!("  connected. version = {}", ns.version());
    println!("  intents in local cache: {}", ns.intent_count());

    let query = "drop my subscription right now";

    println!("\n─── 3. Route a query (initially) ──────────────────────────────");
    let matches = ns.resolve(query);
    let initial_intent = matches.first().map(|m| m.id.as_str()).unwrap_or("(none)");
    let initial_score = matches.first().map(|m| m.score).unwrap_or(0.0);
    println!("  query:        \"{}\"", query);
    println!("  routed to:    {} (score: {:.2})", initial_intent, initial_score);

    println!("\n─── 4. Apply correction ───────────────────────────────────────");
    println!("  Push: this query should map to 'cancel_subscription'");
    let wrong = if initial_intent == "(none)" { "list_subscriptions" } else { initial_intent };
    ns.correct(query, wrong, "cancel_subscription")?;
    println!("  ✓ applied locally + pushed to server");

    println!("\n─── 5. Re-route immediately (local already updated) ──────────");
    let matches = ns.resolve(query);
    let local_intent = matches.first().map(|m| m.id.as_str()).unwrap_or("(none)");
    let local_score = matches.first().map(|m| m.score).unwrap_or(0.0);
    println!("  query:        \"{}\"", query);
    println!("  routed to:    {} (score: {:.2})", local_intent, local_score);
    if local_intent == "cancel_subscription" {
        println!("  ✓ Local state instantly reflects correction (no network round-trip).");
    }

    println!("\n─── 6. Wait for next sync tick (server confirms version bump) ─");
    let v_before = ns.version();
    println!("  local version before tick: {}", v_before);
    println!("  waiting for sync tick (≤ 6s)...");
    for _ in 0..6 {
        std::thread::sleep(Duration::from_secs(1));
        let v = ns.version();
        if v > v_before {
            println!("  ✓ pulled v{} from server (was v{})", v, v_before);
            break;
        }
    }

    println!("\n─── Result ────────────────────────────────────────────────────");
    if local_intent == "cancel_subscription" {
        println!("  ✓ Library learned. The same query now routes correctly.");
        println!("  ✓ Any OTHER connected library subscribed to '{}' will see this correction", NS);
        println!("    on its next poll — that's the cross-instance propagation story.");
    } else {
        println!("  ✗ Local state didn't update. Check server logs.");
    }

    cleanup_namespace(api_key.as_deref())?;
    Ok(())
}

// ── Helpers using the HTTP API directly (out-of-band setup) ───────────────────

fn http() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap()
}

fn setup_namespace(api_key: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let _ = delete_namespace(api_key); // best-effort cleanup of prior runs

    let client = http();
    let mut req = client.post(format!("{}/api/namespaces", SERVER))
        .json(&serde_json::json!({"namespace_id": NS, "description": "connected demo"}));
    if let Some(k) = api_key { req = req.header("X-Api-Key", k); }
    req.send()?.error_for_status()?;
    println!("  ✓ namespace '{}' created", NS);

    let intents = [
        ("list_subscriptions", vec!["list my subscriptions", "show all subscriptions", "what subscriptions do I have"]),
        ("cancel_subscription", vec!["cancel subscription", "stop my subscription", "end my plan"]),
        ("greeting", vec!["hello", "hi there", "good morning"]),
    ];

    for (id, phrases) in intents {
        let mut req = client.post(format!("{}/api/intents", SERVER))
            .header("X-Namespace-ID", NS)
            .json(&serde_json::json!({"id": id, "phrases": phrases}));
        if let Some(k) = api_key { req = req.header("X-Api-Key", k); }
        req.send()?.error_for_status()?;
        println!("  ✓ intent '{}' added ({} phrases)", id, 3);
    }
    Ok(())
}

fn delete_namespace(api_key: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let client = http();
    let mut req = client.delete(format!("{}/api/namespaces", SERVER))
        .json(&serde_json::json!({"namespace_id": NS}));
    if let Some(k) = api_key { req = req.header("X-Api-Key", k); }
    let _ = req.send();
    Ok(())
}

fn cleanup_namespace(api_key: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    delete_namespace(api_key)
}
