//! Full E2E: manual creation + MCP import + auto-learn deterministic path.
//!
//! Replaces `tests/integration/full_e2e.sh`.
//! Run with `cargo test --test http_full_e2e`.

mod common;
use common::*;
use serde_json::json;

const NS: &str = "e2e-full";

fn ns_h() -> Vec<(&'static str, &'static str)> {
    vec![("X-Namespace-ID", NS)]
}

#[test]
fn manual_creation() {
    let server = TestServer::spawn();
    let c = server.client();
    let b = format!("{}/api", server.url);

    // Create namespace
    let (s, body) = post_json(
        &c,
        &format!("{}/namespaces", b),
        &[],
        &json!({"namespace_id": NS, "description": "e2e test"}),
    );
    assert!(body.contains("created"), "{}", body);
    assert_eq!(s, 200);

    // Multilingual intent + metadata patch
    let s = post_json(&c, &format!("{}/intents", b), &ns_h(),
        &json!({
            "id": "billing:cancel_subscription",
            "phrases_by_lang": {
                "en": ["cancel my subscription", "stop my subscription", "end my plan", "I want to cancel"],
                "fr": ["annuler mon abonnement", "arrêter mon abonnement"]
            },
            "description": "Cancel a recurring subscription"
        })).0;
    assert_eq!(s, 201);

    let s = patch_json(
        &c,
        &format!("{}/intents/billing:cancel_subscription", b),
        &ns_h(),
        &json!({
            "persona": "warm and apologetic",
            "guardrails": ["confirm with user before canceling", "offer pause as alternative"]
        }),
    );
    assert_eq!(s, 204);

    let (_, body) = get(&c, &format!("{}/intents", b), &ns_h());
    assert!(body.contains("warm and apologetic"), "persona persisted");
    assert!(
        body.contains("confirm with user before canceling"),
        "guardrail persisted"
    );
    assert!(
        body.contains("annuler mon abonnement"),
        "French phrase persisted"
    );

    // Add a decoy intent so IDF has signal
    post_json(
        &c,
        &format!("{}/intents", b),
        &ns_h(),
        &json!({"id":"weather:check","phrases":["what is the weather","is it raining"]}),
    );

    // Routing in both languages
    let (_, body) = post_json(
        &c,
        &format!("{}/resolve", b),
        &ns_h(),
        &json!({"query":"please cancel my plan"}),
    );
    assert!(
        body.contains("billing:cancel_subscription"),
        "EN routes correctly: {}",
        body
    );

    let (_, body) = post_json(
        &c,
        &format!("{}/resolve", b),
        &ns_h(),
        &json!({"query":"je veux annuler"}),
    );
    assert!(
        body.contains("billing:cancel_subscription"),
        "FR routes correctly: {}",
        body
    );

    delete_json(
        &c,
        &format!("{}/namespaces", b),
        &[],
        &json!({"namespace_id": NS}),
    );
}

#[test]
fn mcp_import_three_tools() {
    let server = TestServer::spawn();
    let c = server.client();
    let b = format!("{}/api", server.url);
    const NS_MCP: &str = "e2e-mcp";

    post_json(
        &c,
        &format!("{}/namespaces", b),
        &[],
        &json!({"namespace_id": NS_MCP}),
    );

    let h: Vec<(&str, &str)> = vec![("X-Namespace-ID", NS_MCP)];

    let tools_json = serde_json::to_string(&json!({
        "tools": [
            {
                "name": "search_orders",
                "description": "Search through customer orders by date range, status, or customer ID",
                "inputSchema": {"type":"object","properties":{"customer_id":{"type":"string"}},"required":["customer_id"]},
                "annotations": {"readOnlyHint": true}
            },
            {
                "name": "create_refund",
                "description": "Issue a refund for a completed order. Requires approval for amounts over $100.",
                "inputSchema": {"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}
            },
            {
                "name": "send_notification",
                "description": "Send an email or SMS notification to a customer about their order status",
                "inputSchema": {"type":"object","properties":{"customer_id":{"type":"string"}},"required":["customer_id"]}
            }
        ]
    })).unwrap();

    // Parse step
    let (_, body) = post_json(
        &c,
        &format!("{}/import/mcp/parse", b),
        &h,
        &json!({"tools_json": tools_json}),
    );
    assert!(
        body.contains("\"total_tools\":3"),
        "parse 3 tools: {}",
        body
    );
    assert!(body.contains("search_orders"));
    assert!(body.contains("create_refund"));
    assert!(body.contains("send_notification"));

    // Apply step
    let (s, _) = post_json(
        &c,
        &format!("{}/import/mcp/apply", b),
        &h,
        &json!({
            "tools_json": tools_json,
            "selected": ["search_orders","create_refund","send_notification"],
            "domain": "shop"
        }),
    );
    assert!((200..300).contains(&s), "MCP apply 2xx: {}", s);

    // Verify intents created with prefix + correct types
    let (_, body) = get(&c, &format!("{}/intents", b), &h);
    assert!(body.contains("shop:search_orders"));
    assert!(body.contains("shop:create_refund"));
    assert!(body.contains("shop:send_notification"));

    // Note: IntentType (Action/Context auto-classification) was removed in
    // v0.2.1. The field used to be derived from MCP `readOnlyHint`; nothing
    // consumed it at runtime, so it was dropped. See CHANGELOG [0.2.1].

    // Schema preserved
    assert!(body.contains("customer_id"));
    // Target = mcp_server
    assert!(body.contains("mcp_server"));

    // MCP intents resolve
    let (_, body) = post_json(
        &c,
        &format!("{}/resolve", b),
        &h,
        &json!({"query":"refund this order"}),
    );
    assert!(
        body.contains("shop:create_refund"),
        "refund routes: {}",
        body
    );

    let (_, body) = post_json(
        &c,
        &format!("{}/resolve", b),
        &h,
        &json!({"query":"send email to customer"}),
    );
    assert!(
        body.contains("shop:send_notification"),
        "notification routes: {}",
        body
    );

    delete_json(
        &c,
        &format!("{}/namespaces", b),
        &[],
        &json!({"namespace_id": NS_MCP}),
    );
}

#[test]
fn auto_learn_deterministic_path() {
    let server = TestServer::spawn();
    let c = server.client();
    let b = format!("{}/api", server.url);
    const NS_AL: &str = "e2e-autolearn";

    post_json(
        &c,
        &format!("{}/namespaces", b),
        &[],
        &json!({"namespace_id": NS_AL}),
    );
    let h: Vec<(&str, &str)> = vec![("X-Namespace-ID", NS_AL)];

    // Seed a couple intents
    post_json(
        &c,
        &format!("{}/intents", b),
        &h,
        &json!({"id":"refund","phrases":["refund please","I want a refund"]}),
    );
    post_json(
        &c,
        &format!("{}/intents", b),
        &h,
        &json!({"id":"cancel","phrases":["cancel order","stop order"]}),
    );

    // decay_for_intents (audit log auto-fires)
    let (s, _) = post_json(
        &c,
        &format!("{}/namespaces/decay", b),
        &[],
        &json!({"namespace_id": NS_AL, "queries":["weather is nice today"], "alpha": 0.1}),
    );
    assert_eq!(s, 200);

    // rebuild_index (clears audit log)
    let (s, _) = post_json(
        &c,
        &format!("{}/namespaces/rebuild", b),
        &[],
        &json!({"namespace_id": NS_AL}),
    );
    assert_eq!(s, 200);

    // Routing still works after rebuild
    let (_, body) = post_json(
        &c,
        &format!("{}/resolve", b),
        &h,
        &json!({"query":"refund please"}),
    );
    assert!(
        body.contains("refund"),
        "routing works post-rebuild: {}",
        body
    );

    delete_json(
        &c,
        &format!("{}/namespaces", b),
        &[],
        &json!({"namespace_id": NS_AL}),
    );
}

#[test]
fn auth_keys_endpoint() {
    let server = TestServer::spawn();
    let c = server.client();
    let b = format!("{}/api", server.url);

    // Server bootstrapped a `studio-admin` key on first boot. Tests
    // pick up `server.api_key` automatically via the thread-local in
    // helpers; verify it's there.
    assert!(server.api_key.starts_with("mr_studio-admin_"));

    // GET /auth/keys (auto-authed by the helper) — should list at least
    // the bootstrap key and report enabled=true.
    let (_, body) = get(&c, &format!("{}/auth/keys", b), &[]);
    assert!(body.contains("\"enabled\":true"));
    assert!(body.contains("studio-admin"));

    // Create a Library-scoped key for a hypothetical client.
    let (s, body) = post_json(
        &c,
        &format!("{}/auth/keys", b),
        &[],
        &json!({"name": "test-key"}),
    );
    assert_eq!(s, 200);
    assert!(body.contains("mr_test-key_"), "key has expected prefix");
    assert!(body.contains("This key is shown once"));
    assert!(body.contains("\"scope\":\"library\""));
    let key: serde_json::Value = serde_json::from_str(&body).unwrap();
    let full_key = key["key"].as_str().unwrap();

    // Sync body (unified POST since v0.1.5).
    let sync_body = json!({
        "local_versions": { "default": 0 },
        "logs": [],
        "corrections": [],
    });

    // ── Negative case: no key at all → 401 ────────────────────────────────
    // Bypass the helper's auto-inject by crafting the request directly.
    let raw_resp = c
        .post(format!("{}/sync", b))
        .header("X-Namespace-ID", "default")
        .json(&sync_body)
        .send()
        .expect("request");
    assert_eq!(
        raw_resp.status().as_u16(),
        401,
        "request without X-Api-Key returns 401"
    );

    // ── Wrong key → 401 ──────────────────────────────────────────────────
    let (s, _) = post_json(
        &c,
        &format!("{}/sync", b),
        &[
            ("X-Namespace-ID", "default"),
            (
                "X-Api-Key",
                "mr_wrong_0000000000000000000000000000000000000000000000000000000000000000",
            ),
        ],
        &sync_body,
    );
    assert_eq!(s, 401);

    // ── New library key works on /sync ───────────────────────────────────
    let (s, _) = post_json(
        &c,
        &format!("{}/sync", b),
        &[("X-Namespace-ID", "default"), ("X-Api-Key", full_key)],
        &sync_body,
    );
    assert_eq!(s, 200, "library key works on /sync");

    // List + redaction.
    let (_, body) = get(&c, &format!("{}/auth/keys", b), &[]);
    assert!(body.contains("test-key"));
    assert!(!body.contains(full_key), "full key not in list");

    // Revoke.
    let s = delete_json(&c, &format!("{}/auth/keys/test-key", b), &[], &json!({}));
    assert_eq!(s, 204);
}

#[test]
fn unauthorized_endpoints_return_401_without_key() {
    // Smoke that EVERY protected endpoint enforces the gate, not just
    // /sync. Picks one read (list namespaces) and one write (create
    // namespace) — middleware applies uniformly so this is a chokepoint
    // test, not an endpoint enumeration.
    let server = TestServer::spawn();
    let c = server.client();
    let b = format!("{}/api", server.url);

    let r = c.get(format!("{}/namespaces", b)).send().expect("req");
    assert_eq!(
        r.status().as_u16(),
        401,
        "GET /namespaces without auth → 401"
    );

    let r = c
        .post(format!("{}/namespaces", b))
        .json(&json!({"namespace_id":"x"}))
        .send()
        .expect("req");
    assert_eq!(
        r.status().as_u16(),
        401,
        "POST /namespaces without auth → 401"
    );

    // /version is public — used by health checks.
    let r = c.get(format!("{}/version", b)).send().expect("req");
    assert_eq!(r.status().as_u16(), 200, "GET /version is public");
}
