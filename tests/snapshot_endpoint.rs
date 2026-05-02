//! Integration test: `/api/snapshot` endpoint.
//!
//! Covers:
//!   1. Empty `namespace_ids` → returns all namespaces.
//!   2. Explicit `namespace_ids` → returns exactly those namespaces.
//!   3. Response shape: each entry has `version` (u64) and `export` (non-empty string).
//!   4. After `/api/sync` signals `cold_start_required`, a follow-up `/api/snapshot`
//!      delivers the full state and the client can reconstruct the resolver.

mod common;
#[allow(unused_imports)]
use common::*;
use serde_json::json;

const NS_A: &str = "snapshot-test-ns-a";
const NS_B: &str = "snapshot-test-ns-b";

#[test]
fn snapshot_returns_full_state_for_all_namespaces() {
    let server = TestServer::spawn();
    let c = server.client();
    let b = format!("{}/api", server.url);

    // Create two namespaces.
    for ns in &[NS_A, NS_B] {
        let (s, body) = post_json(
            &c,
            &format!("{}/namespaces", b),
            &[],
            &json!({"namespace_id": ns}),
        );
        assert!((200..300).contains(&s), "create {}: {}", ns, body);
        let h = vec![("X-Namespace-ID", *ns)];
        post_json(
            &c,
            &format!("{}/intents", b),
            &h,
            &json!({"id": "alpha", "phrases": ["foo bar", "baz"]}),
        );
    }

    // Call /api/snapshot with no namespace_ids — should return both.
    let (s, body) = post_json(&c, &format!("{}/snapshot", b), &[], &json!({}));
    assert_eq!(s, 200, "snapshot HTTP: {}", body);
    let val: serde_json::Value = serde_json::from_str(&body).expect("parse snapshot response");
    let namespaces = val["namespaces"].as_object().expect("namespaces object");
    assert!(
        namespaces.contains_key(NS_A),
        "snapshot must include {}: keys={:?}",
        NS_A,
        namespaces.keys().collect::<Vec<_>>()
    );
    assert!(
        namespaces.contains_key(NS_B),
        "snapshot must include {}: keys={:?}",
        NS_B,
        namespaces.keys().collect::<Vec<_>>()
    );
    for (ns_id, entry) in namespaces {
        let version = entry["version"].as_u64();
        assert!(
            version.is_some(),
            "namespace {} missing version field",
            ns_id
        );
        let export = entry["export"].as_str().unwrap_or("");
        assert!(
            !export.is_empty(),
            "namespace {} has empty export field",
            ns_id
        );
        // The export must be valid JSON that can be round-tripped.
        let parsed: serde_json::Value =
            serde_json::from_str(export).expect("export must be valid JSON");
        assert!(
            parsed.is_object(),
            "export must be a JSON object for namespace {}",
            ns_id
        );
    }

    // Cleanup.
    for ns in &[NS_A, NS_B] {
        delete_json(
            &c,
            &format!("{}/namespaces", b),
            &[],
            &json!({"namespace_id": ns}),
        );
    }
}

#[test]
fn snapshot_with_explicit_ids_returns_only_requested() {
    let server = TestServer::spawn();
    let c = server.client();
    let b = format!("{}/api", server.url);

    for ns in &[NS_A, NS_B] {
        let (s, body) = post_json(
            &c,
            &format!("{}/namespaces", b),
            &[],
            &json!({"namespace_id": ns}),
        );
        assert!((200..300).contains(&s), "create {}: {}", ns, body);
        post_json(
            &c,
            &format!("{}/intents", b),
            &[("X-Namespace-ID", *ns)],
            &json!({"id": "alpha", "phrases": ["hello world"]}),
        );
    }

    // Only request NS_A.
    let (s, body) = post_json(
        &c,
        &format!("{}/snapshot", b),
        &[],
        &json!({"namespace_ids": [NS_A]}),
    );
    assert_eq!(s, 200, "snapshot HTTP: {}", body);
    let val: serde_json::Value = serde_json::from_str(&body).expect("parse");
    let namespaces = val["namespaces"].as_object().expect("namespaces object");
    assert!(
        namespaces.contains_key(NS_A),
        "response must contain requested namespace"
    );
    assert!(
        !namespaces.contains_key(NS_B),
        "response must NOT contain un-requested namespace"
    );

    for ns in &[NS_A, NS_B] {
        delete_json(
            &c,
            &format!("{}/namespaces", b),
            &[],
            &json!({"namespace_id": ns}),
        );
    }
}

#[test]
fn sync_signals_cold_start_required_for_old_client() {
    let server = TestServer::spawn();
    let c = server.client();
    let b = format!("{}/api", server.url);
    const NS: &str = "snapshot-cold-start-test";

    // Create namespace and add intents.
    let (s, body) = post_json(
        &c,
        &format!("{}/namespaces", b),
        &[],
        &json!({"namespace_id": NS}),
    );
    assert!((200..300).contains(&s), "create ns: {}", body);
    post_json(
        &c,
        &format!("{}/intents", b),
        &[("X-Namespace-ID", NS)],
        &json!({"id": "greet", "phrases": ["hello", "hi"]}),
    );

    // Sync with version 0 and supports_delta=true — client is behind but
    // oplog may cover version 0. First sync may or may not signal cold_start.
    // What we want to test: a client that does NOT send supports_delta gets
    // cold_start_required (not export).
    let (s, body) = post_json(
        &c,
        &format!("{}/sync", b),
        &[],
        &json!({
            "local_versions": { NS: 0u64 },
            "logs": [],
            "corrections": [],
            // No supports_delta — triggers the "baseline client" path → cold_start_required.
        }),
    );
    assert_eq!(s, 200, "sync HTTP: {}", body);
    let val: serde_json::Value = serde_json::from_str(&body).expect("parse sync response");
    let ns_entry = &val["namespaces"][NS];
    // Must NOT contain an "export" field.
    assert!(
        ns_entry.get("export").is_none(),
        "/api/sync must never return export field; got: {}",
        ns_entry
    );
    // Must signal cold_start_required.
    assert_eq!(
        ns_entry["cold_start_required"], true,
        "/api/sync should signal cold_start_required for non-delta client; got: {}",
        ns_entry
    );

    // Follow up with /api/snapshot — should deliver full state.
    let (s2, body2) = post_json(
        &c,
        &format!("{}/snapshot", b),
        &[],
        &json!({"namespace_ids": [NS]}),
    );
    assert_eq!(s2, 200, "snapshot HTTP: {}", body2);
    let snap: serde_json::Value = serde_json::from_str(&body2).expect("parse snapshot");
    let snap_ns = &snap["namespaces"][NS];
    assert!(
        snap_ns["version"].as_u64().unwrap_or(0) > 0,
        "snapshot version must be > 0"
    );
    let export_str = snap_ns["export"].as_str().unwrap_or("");
    assert!(!export_str.is_empty(), "snapshot export must be non-empty");

    // The export must be importable via the library.
    let resolver = microresolve::Resolver::import_json(export_str)
        .expect("snapshot export must be importable");
    assert_eq!(
        resolver.intent_count(),
        1,
        "imported resolver must have 1 intent"
    );

    delete_json(
        &c,
        &format!("{}/namespaces", b),
        &[],
        &json!({"namespace_id": NS}),
    );
}
