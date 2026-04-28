//! HTTP smoke tests — wire-format coverage on the most-used endpoints.
//!
//! Replaces `tests/integration/smoke.sh`. Run with `cargo test --test http_smoke`.
//! Requires a release build of the server binary:
//!   `cargo build --release --features server`

mod common;
use common::*;
use serde_json::json;

const NS: &str = "smoke-ns";

fn ns_headers() -> Vec<(&'static str, &'static str)> {
    vec![("X-Namespace-ID", NS)]
}

#[test]
fn full_smoke() {
    let server = TestServer::spawn();
    let c = server.client();
    let b = format!("{}/api", server.url);

    // 1. namespace CRUD
    let (s, body) = post_json(&c, &format!("{}/namespaces", b), &[],
        &json!({"namespace_id": NS, "description": "smoke"}));
    assert!(body.contains("created"), "create namespace: {}", body);
    assert!(s == 200, "create namespace status: {}", s);

    let (s, body) = get(&c, &format!("{}/namespaces", b), &[]);
    assert!(body.contains(NS), "list shows namespace");
    assert_eq!(s, 200);

    // 2. intent CRUD (mono)
    let s = post_json(&c, &format!("{}/intents", b), &ns_headers(),
        &json!({"id":"hello","phrases":["hi","hey","hello there"]})).0;
    assert_eq!(s, 201, "create mono intent");

    // 3. multilingual via /api/intents (was /multilingual, removed)
    let s = post_json(&c, &format!("{}/intents", b), &ns_headers(),
        &json!({"id":"bye","phrases_by_lang":{"en":["bye","goodbye"],"fr":["au revoir"]},"description":"farewell"})).0;
    assert_eq!(s, 201, "create multilingual intent");

    let (_, body) = get(&c, &format!("{}/intents", b), &ns_headers());
    assert!(body.contains("\"bye\""), "multilingual intent persisted");
    assert!(body.contains("farewell"), "description persisted");
    assert!(body.contains("\"fr\""), "fr language bucket persisted");

    // 4. patch via update_intent
    let s = patch_json(&c, &format!("{}/intents/hello", b), &ns_headers(),
        &json!({"description":"greeting","persona":"friendly","intent_type":"action"}));
    assert_eq!(s, 204, "patch intent");

    let (_, body) = get(&c, &format!("{}/intents", b), &ns_headers());
    assert!(body.contains("greeting"), "patched description");
    assert!(body.contains("friendly"), "patched persona");

    // 5. routing
    let (_, body) = post_json(&c, &format!("{}/route_multi", b), &ns_headers(),
        &json!({"query":"hi there"}));
    assert!(body.contains("\"hello\""), "EN routes to hello: {}", body);

    let (_, body) = post_json(&c, &format!("{}/route_multi", b), &ns_headers(),
        &json!({"query":"au revoir"}));
    assert!(body.contains("\"bye\""), "FR routes to bye: {}", body);

    // 6. add phrase
    let (s, _) = post_json(&c, &format!("{}/intents/hello/phrases", b), &ns_headers(),
        &json!({"phrase":"howdy","lang":"en"}));
    assert_eq!(s, 200, "add phrase");

    // 7. update namespace metadata
    let s = patch_json(&c, &format!("{}/namespaces", b), &[],
        &json!({"namespace_id": NS, "name":"Smoke", "default_threshold": 0.4}));
    assert_eq!(s, 200, "patch namespace");

    let (_, body) = get(&c, &format!("{}/namespaces", b), &[]);
    assert!(body.contains("Smoke"), "namespace name persisted");

    // 8. train_negative (audit log auto-fires)
    let (s, _) = post_json(&c, &format!("{}/namespaces/train_negative", b), &[],
        &json!({"namespace_id": NS, "queries":["unrelated"], "alpha": 0.1}));
    assert_eq!(s, 200, "train_negative");

    // 9. rebuild (clears audit log)
    let (s, _) = post_json(&c, &format!("{}/namespaces/rebuild", b), &[],
        &json!({"namespace_id": NS}));
    assert_eq!(s, 200, "rebuild");

    // 10. layer info
    let (_, body) = get(&c, &format!("{}/layers/info", b), &ns_headers());
    assert!(body.contains("terms"), "layer info has terms field");

    // 11. delete intent
    let s = delete_json(&c, &format!("{}/intents/hello", b), &ns_headers(), &json!({}));
    assert_eq!(s, 204, "delete intent");

    // cleanup
    delete_json(&c, &format!("{}/namespaces", b), &[],
        &json!({"namespace_id": NS}));
}
