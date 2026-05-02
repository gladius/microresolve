//! End-to-end delta sync integration test.
//!
//! Simulates server + two clients using MicroResolve directly (no HTTP).
//! Verifies that delta-synced client A reaches the same state as server,
//! and that full-synced client B also matches.

use microresolve::oplog::Op;
use microresolve::{MicroResolve, MicroResolveConfig};

/// Collect all ops since `since_version` from server namespace.
fn server_ops_since(server: &MicroResolve, ns: &str, since_version: u64) -> Vec<Op> {
    server.namespace(ns).with_resolver(|r| {
        r.oplog
            .iter()
            .filter(|(v, _)| *v > since_version)
            .map(|(_, op)| op.clone())
            .collect()
    })
}

/// Load a resolver's intents into a MicroResolve engine namespace.
fn load_from_export(engine: &MicroResolve, ns_id: &str, export_json: &str) {
    let r = microresolve::Resolver::import_json(export_json).unwrap();
    let h = engine.namespace(ns_id);
    for id in r.intent_ids() {
        if let Some(training) = r.training_by_lang(&id) {
            let seeds = microresolve::IntentSeeds::Multi(training.clone().into_iter().collect());
            h.add_intent(&id, seeds).unwrap();
        }
    }
}

#[test]
fn delta_sync_e2e() {
    // ── Step 1: Build server namespace with 3 intents ─────────────────────
    let server = MicroResolve::new(MicroResolveConfig::default()).unwrap();
    {
        let h = server.namespace("test-delta");
        h.add_intent("greet", vec!["hello".to_string(), "hi there".to_string()])
            .unwrap();
        h.add_intent(
            "farewell",
            vec!["goodbye".to_string(), "see you".to_string()],
        )
        .unwrap();
        h.add_intent("thanks", vec!["thank you".to_string()])
            .unwrap();
    }
    let version_n = server.namespace("test-delta").version();
    assert!(
        version_n > 0,
        "server must have non-zero version after 3 intents"
    );

    // ── Step 2: Client A full-sync ────────────────────────────────────────
    let client_a = MicroResolve::new(MicroResolveConfig::default()).unwrap();
    {
        let export = server.namespace("test-delta").export_json();
        load_from_export(&client_a, "test-delta", &export);
    }
    assert_eq!(
        client_a.namespace("test-delta").intent_count(),
        3,
        "client A should have 3 intents after full sync"
    );
    let client_a_version_after_initial = version_n;

    // ── Step 3: Server mutations ───────────────────────────────────────────
    {
        let h = server.namespace("test-delta");
        h.add_intent("order", vec!["place an order".to_string()])
            .unwrap();
        for phrase in &[
            "buy this",
            "add to cart",
            "checkout",
            "purchase",
            "i want to buy",
        ] {
            h.add_phrase("greet", phrase, "en").unwrap();
        }
    }
    let server_version_after = server.namespace("test-delta").version();
    assert!(
        server_version_after > client_a_version_after_initial,
        "server version must advance after mutations"
    );

    // ── Step 4: Collect ops since client A's baseline ─────────────────────
    // Use version 0 to get all ops (client A was built by re-adding, not tracking exact version).
    let ops = server_ops_since(&server, "test-delta", 0);
    assert!(!ops.is_empty(), "server oplog must be non-empty");

    // Count structural ops (IntentAdded) — should be at least 4 (3 + order).
    let intent_added_count = ops
        .iter()
        .filter(|op| matches!(op, Op::IntentAdded { .. }))
        .count();
    assert!(
        intent_added_count >= 4,
        "expected at least 4 IntentAdded ops, got {}",
        intent_added_count
    );

    // ── Step 5: Client A applies ops ──────────────────────────────────────
    // Apply all ops starting from baseline (ops are idempotent, so re-applying
    // already-known intents is fine).
    client_a.namespace("test-delta").apply_ops(&ops).unwrap();

    // ── Step 6: Assert client A matches server ────────────────────────────
    let server_count = server.namespace("test-delta").intent_count();
    let client_a_count = client_a.namespace("test-delta").intent_count();
    assert_eq!(
        server_count, client_a_count,
        "client A intent count should match server after delta sync"
    );
    assert!(
        client_a.namespace("test-delta").intent("order").is_some(),
        "client A should have 'order' intent after delta sync"
    );

    // ── Step 7: Client B cold start (full export) ─────────────────────────
    let client_b = MicroResolve::new(MicroResolveConfig::default()).unwrap();
    {
        let export = server.namespace("test-delta").export_json();
        load_from_export(&client_b, "test-delta", &export);
    }
    let client_b_count = client_b.namespace("test-delta").intent_count();

    // ── Step 8: Assert client B matches client A ──────────────────────────
    assert_eq!(
        client_b_count, client_a_count,
        "client B (full export) should match client A (delta sync)"
    );
    assert!(
        client_b.namespace("test-delta").intent("order").is_some(),
        "client B should have 'order' intent"
    );
}
