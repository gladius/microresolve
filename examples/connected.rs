//! End-to-end demo of MicroResolve connected mode.
//!
//! Shows the live-learning loop:
//!   1. Library connects to a running server and pulls a namespace
//!   2. Routes a query — gets a confident-but-wrong answer
//!   3. Pushes an explicit correction (local + server) via MicroResolve.correct()
//!   4. Same query now routes correctly — and any other connected
//!      library subscribed to this namespace will pick up the change
//!      on its next sync tick.
//!
//! Prerequisites:
//!   $ cargo build --release --features server
//!   $ ./target/release/microresolve-studio --port 3001 --no-browser &
//!
//! Run:
//!   $ cargo run --release --example connected --features connect
//!
//! With auth (after generating a key in the UI):
//!   $ MICRORESOLVE_API_KEY=mr_xxx... cargo run --release --example connected --features connect

use microresolve::{MicroResolve, MicroResolveConfig, ServerConfig};
use std::time::Duration;

const NS: &str = "demo-connected";
const SERVER: &str = "http://localhost:3001";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("MICRORESOLVE_API_KEY").ok();

    println!("─── 1. Setup namespace + intents on server ────────────────────");
    setup_namespace(api_key.as_deref())?;

    println!("\n─── 2. Connect MicroResolve to server ─────────────────────────");
    let engine = MicroResolve::new(MicroResolveConfig {
        server: Some(ServerConfig {
            url: SERVER.to_string(),
            api_key: api_key.clone(),
            subscribe: vec![NS.to_string()],
            tick_interval_secs: 5, // poll every 5s for snappier demo
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
    println!(
        "  routed to:    {} (score: {:.2})",
        initial_intent, initial_score
    );

    println!("\n─── 4. Strict mode: library mutations refused ────────────────");
    println!("  Connected libraries are READ-ONLY caches. The server is the");
    println!("  only authoritative writer. Calling ns.correct(...) returns:");
    let wrong = if initial_intent == "(none)" {
        "list_subscriptions"
    } else {
        initial_intent
    };
    match ns.correct(query, wrong, "cancel_subscription") {
        Err(microresolve::Error::ConnectMode) => {
            println!("    Err(ConnectMode) — refused, as designed.");
        }
        other => println!("    unexpected: {:?}", other),
    }

    println!("\n─── 5. Apply correction via the server's HTTP API ────────────");
    println!("  POST /api/correct with the correction. Server applies it,");
    println!("  the library catches up on the next sync tick.");
    let client = reqwest::blocking::Client::new();
    let mut req = client
        .post(format!("{}/api/correct", SERVER))
        .header("X-Namespace-ID", NS)
        .json(&serde_json::json!({
            "query": query,
            "wrong_intent": wrong,
            "right_intent": "cancel_subscription",
        }));
    if let Some(ref key) = api_key {
        req = req.header("X-Api-Key", key);
    }
    let resp = req.send()?;
    println!("  ✓ POST /api/correct → HTTP {}", resp.status());

    println!("\n─── 6. Wait for sync tick to pull the change ─────────────────");
    let v_before = ns.version();
    println!("  local version before tick: {}", v_before);
    println!("  waiting for sync tick (≤ 6s)...");
    let mut local_intent = String::from("(unknown)");
    for _ in 0..6 {
        std::thread::sleep(Duration::from_secs(1));
        let v = ns.version();
        if v > v_before {
            println!("  ✓ pulled v{} from server (was v{})", v, v_before);
            let matches = ns.resolve(query);
            local_intent = matches
                .first()
                .map(|m| m.id.clone())
                .unwrap_or_else(|| "(none)".to_string());
            break;
        }
    }

    println!("\n─── Result ────────────────────────────────────────────────────");
    if local_intent == "cancel_subscription" {
        println!("  ✓ Library caught up. The same query now routes correctly.");
        println!(
            "  ✓ Any OTHER connected library subscribed to '{}' will also see this",
            NS
        );
        println!("    on its next poll — cross-instance propagation via the server.");
    } else {
        println!(
            "  Note: routes_to {} after sync. Check server-side correction handling.",
            local_intent
        );
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
    let mut req = client
        .post(format!("{}/api/namespaces", SERVER))
        .json(&serde_json::json!({"namespace_id": NS, "description": "connected demo"}));
    if let Some(k) = api_key {
        req = req.header("X-Api-Key", k);
    }
    req.send()?.error_for_status()?;
    println!("  ✓ namespace '{}' created", NS);

    let intents = [
        (
            "list_subscriptions",
            vec![
                "list my subscriptions",
                "show all subscriptions",
                "what subscriptions do I have",
            ],
        ),
        (
            "cancel_subscription",
            vec!["cancel subscription", "stop my subscription", "end my plan"],
        ),
        ("greeting", vec!["hello", "hi there", "good morning"]),
    ];

    for (id, phrases) in intents {
        let mut req = client
            .post(format!("{}/api/intents", SERVER))
            .header("X-Namespace-ID", NS)
            .json(&serde_json::json!({"id": id, "phrases": phrases}));
        if let Some(k) = api_key {
            req = req.header("X-Api-Key", k);
        }
        req.send()?.error_for_status()?;
        println!("  ✓ intent '{}' added ({} phrases)", id, 3);
    }
    Ok(())
}

fn delete_namespace(api_key: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let client = http();
    let mut req = client
        .delete(format!("{}/api/namespaces", SERVER))
        .json(&serde_json::json!({"namespace_id": NS}));
    if let Some(k) = api_key {
        req = req.header("X-Api-Key", k);
    }
    let _ = req.send();
    Ok(())
}

fn cleanup_namespace(api_key: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    delete_namespace(api_key)
}
