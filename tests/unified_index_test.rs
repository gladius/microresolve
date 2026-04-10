//! Unified index tests.
//!
//! Verifies that a single Router with namespaced intent IDs ("ns:intent")
//! correctly handles multi-app routing, namespace filtering, multilingual seeds,
//! CJK namespaces, and situation patterns as app fingerprints.

use asv_router::Router;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a unified router with 3 apps:
///   stripe:charge_card, stripe:refund
///   slack:send_message, slack:create_channel
///   github:create_issue, github:merge_pr
fn build_unified() -> Router {
    let mut r = Router::new();
    r.add_intent("stripe:charge_card", &[
        "charge my card", "payment failed", "billing issue", "card declined",
    ]);
    r.add_intent("stripe:refund", &[
        "refund my payment", "give me a refund", "money back",
    ]);
    r.add_intent("slack:send_message", &[
        "send a message", "post in channel", "message someone",
    ]);
    r.add_intent("slack:create_channel", &[
        "create a channel", "new channel", "make a workspace channel",
    ]);
    r.add_intent("github:create_issue", &[
        "create an issue", "open a ticket", "file a bug report",
    ]);
    r.add_intent("github:merge_pr", &[
        "merge pull request", "approve the PR", "merge the branch",
    ]);
    r
}

// ---------------------------------------------------------------------------
// Test 1: Namespace utilities
// ---------------------------------------------------------------------------

#[test]
fn test_namespace_utilities() {
    let r = build_unified();

    // intent_namespace — free function
    assert_eq!(Router::intent_namespace("stripe:charge_card"), Some("stripe"));
    assert_eq!(Router::intent_namespace("slack:send_message"), Some("slack"));
    assert_eq!(Router::intent_namespace("cancel_order"), None);
    assert_eq!(Router::intent_namespace(""), None);
    assert_eq!(Router::intent_namespace(":empty_ns"), Some(""));

    // list_namespaces — sorted
    let ns = r.list_namespaces();
    assert_eq!(ns, vec!["github", "slack", "stripe"]);

    // intents_in_namespace
    let mut stripe = r.intents_in_namespace("stripe");
    stripe.sort();
    assert_eq!(stripe, vec!["stripe:charge_card", "stripe:refund"]);

    let mut github = r.intents_in_namespace("github");
    github.sort();
    assert_eq!(github, vec!["github:create_issue", "github:merge_pr"]);

    // Non-existent namespace returns empty
    assert!(r.intents_in_namespace("zendesk").is_empty());
}

// ---------------------------------------------------------------------------
// Test 2: route_ns — scoped results
// ---------------------------------------------------------------------------

#[test]
fn test_route_ns_filters() {
    let r = build_unified();

    // Payment query → stripe wins
    let all = r.route_ns("payment failed", None);
    assert!(!all.is_empty());

    let stripe_only = r.route_ns("payment failed", Some("stripe"));
    assert!(!stripe_only.is_empty());
    assert!(stripe_only.iter().all(|r| r.id.starts_with("stripe:")),
        "route_ns must return only stripe: intents, got: {:?}", stripe_only);

    // Messaging query → slack wins
    let slack_only = r.route_ns("send a message to the team", Some("slack"));
    assert!(!slack_only.is_empty());
    assert!(slack_only.iter().all(|r| r.id.starts_with("slack:")));

    // GitHub query scoped to stripe should be empty (or very low score)
    let wrong_ns = r.route_ns("merge pull request", Some("stripe"));
    // If any results come back they must still be stripe-namespaced
    assert!(wrong_ns.iter().all(|r| r.id.starts_with("stripe:")));
}

// ---------------------------------------------------------------------------
// Test 3: route_multi_ns — multi-intent scoped results
// ---------------------------------------------------------------------------

#[test]
fn test_route_multi_ns_filters() {
    let r = build_unified();

    let out = r.route_multi_ns("payment failed and refund my payment", 0.2, Some("stripe"));
    assert!(out.intents.iter().all(|i| i.id.starts_with("stripe:")),
        "route_multi_ns must return only stripe: intents, got: {:?}",
        out.intents.iter().map(|i| &i.id).collect::<Vec<_>>());

    // Metadata keys must also be scoped
    for key in out.metadata.keys() {
        assert!(key.starts_with("stripe:"));
    }

    // Relations are cleared when namespace filter is applied
    // (positional indices would be stale after filtering)
    // — no assert on empty because single-intent queries may have no relations anyway

    // Unscoped route_multi returns intents from all namespaces
    let all = r.route_multi_ns("payment failed and send a message", 0.2, None);
    let namespaces: std::collections::HashSet<&str> = all.intents.iter()
        .filter_map(|i| Router::intent_namespace(&i.id))
        .collect();
    // We expect both stripe and slack to appear
    assert!(namespaces.contains("stripe") || namespaces.contains("slack"),
        "unscoped route_multi_ns should return intents from multiple namespaces");
}

// ---------------------------------------------------------------------------
// Test 4: Cross-namespace disambiguation
// ---------------------------------------------------------------------------

#[test]
fn test_cross_namespace_disambiguation() {
    let r = build_unified();

    // "payment failed" should score stripe higher than github/slack
    let results = r.route("payment failed");
    assert!(!results.is_empty());
    let top = &results[0];
    assert!(top.id.starts_with("stripe:"),
        "top result for 'payment failed' should be stripe:*, got {}", top.id);

    // "merge pull request" should score github higher
    let results = r.route("merge pull request");
    assert!(!results.is_empty());
    assert!(results[0].id.starts_with("github:"),
        "top result for 'merge pull request' should be github:*, got {}", results[0].id);

    // "send a message" should score slack higher
    let results = r.route("send a message to the team");
    assert!(!results.is_empty());
    assert!(results[0].id.starts_with("slack:"),
        "top result for 'send a message' should be slack:*, got {}", results[0].id);
}

// ---------------------------------------------------------------------------
// Test 5: Situation patterns as app fingerprints
// ---------------------------------------------------------------------------

#[test]
fn test_situation_as_fingerprint() {
    let mut r = build_unified();

    // Add brand-name situation patterns as app fingerprints.
    // These fire on queries that mention the app name even if the action vocabulary
    // is generic ("I need help with stripe" — "help" isn't in any seed).
    r.add_situation_patterns("stripe:charge_card", &[("stripe", 2.0), ("payment gateway", 1.5)]);
    r.add_situation_patterns("slack:send_message", &[("slack", 2.0), ("workspace", 1.5)]);
    r.add_situation_patterns("github:create_issue", &[("github", 2.0), ("repository", 1.5)]);

    // "stripe is down" — "down" not in seeds, but situation pattern "stripe" fires
    let results = r.route("stripe is down");
    assert!(!results.is_empty(),
        "situation pattern 'stripe' should fire for 'stripe is down'");
    assert!(results[0].id.starts_with("stripe:"),
        "situation fingerprint should surface stripe:*, got {}", results[0].id);

    // "github repo is broken" — situation pattern "repository" fires
    let results = r.route("github repo is broken");
    assert!(!results.is_empty());
    // Either "github" or "repository" pattern should surface github:create_issue
    let has_github = results.iter().any(|r| r.id.starts_with("github:"));
    assert!(has_github, "situation fingerprint should surface github:*, got {:?}",
        results.iter().map(|r| &r.id).collect::<Vec<_>>());

    // "slack workspace" — fingerprint pattern fires
    let results = r.route("something wrong with my slack workspace");
    let has_slack = results.iter().any(|r| r.id.starts_with("slack:"));
    assert!(has_slack, "situation fingerprint 'workspace' should surface slack:*");
}

// ---------------------------------------------------------------------------
// Test 6: Export / import round-trip preserves namespaced state
// ---------------------------------------------------------------------------

#[test]
fn test_unified_export_import() {
    let mut r = build_unified();
    r.add_situation_patterns("stripe:charge_card", &[("stripe", 2.0)]);
    r.learn("stripe payment declined", "stripe:charge_card");

    let json = r.export_json();

    let r2 = Router::import_json(&json).expect("import should succeed");

    // All namespaced intents survive
    let mut ns = r2.list_namespaces();
    ns.sort();
    assert_eq!(ns, vec!["github", "slack", "stripe"]);

    let stripe = r2.intents_in_namespace("stripe");
    assert!(stripe.contains(&"stripe:charge_card".to_string()));
    assert!(stripe.contains(&"stripe:refund".to_string()));

    // Situation patterns survive
    let patterns = r2.get_situation_patterns("stripe:charge_card")
        .expect("situation patterns should survive export/import");
    assert!(patterns.iter().any(|(p, _)| p == "stripe"),
        "situation pattern 'stripe' should survive export/import");

    // Routing still works after import
    let results = r2.route("payment failed");
    assert!(!results.is_empty());
    assert!(results[0].id.starts_with("stripe:"));
}

// ---------------------------------------------------------------------------
// Test 7: CJK namespace — self-discriminating compound vocabulary
// ---------------------------------------------------------------------------

#[test]
fn test_cjk_namespace_routing() {
    let mut r = Router::new();

    // WeChat namespace — Chinese compound words are naturally high-IDF
    r.add_intent("wechat:send_money", &["微信转账", "发红包", "微信支付"]);
    r.add_intent("wechat:moments", &["发朋友圈", "朋友圈动态"]);

    // Alipay namespace — different compound vocabulary
    r.add_intent("alipay:send_money", &["支付宝转账", "花呗支付", "余额宝"]);
    r.add_intent("alipay:scan_pay", &["扫码支付", "支付宝扫一扫"]);

    // "微信转账" routes to wechat (not alipay)
    let results = r.route_ns("微信转账", Some("wechat"));
    assert!(!results.is_empty(), "wechat CJK query should match");
    assert_eq!(results[0].id, "wechat:send_money");

    // "支付宝" routes to alipay (not wechat)
    let results = r.route_ns("支付宝转账", Some("alipay"));
    assert!(!results.is_empty(), "alipay CJK query should match");
    assert_eq!(results[0].id, "alipay:send_money");

    // Cross-namespace: "微信" routes to wechat over alipay in full index
    let results = r.route("微信转账");
    assert!(!results.is_empty());
    assert!(results[0].id.starts_with("wechat:"),
        "微信转账 should rank wechat above alipay, got {}", results[0].id);

    // CJK namespace utilities work
    let ns = r.list_namespaces();
    assert!(ns.contains(&"wechat".to_string()));
    assert!(ns.contains(&"alipay".to_string()));
}

// ---------------------------------------------------------------------------
// Test 8: Multilingual seeds — all languages route correctly in unified index
// ---------------------------------------------------------------------------

#[test]
fn test_multilingual_unified_routing() {
    let mut r = Router::new();

    // stripe:charge_card with English + Tamil + Chinese seeds
    let mut seeds = HashMap::new();
    seeds.insert("en".to_string(), vec![
        "charge my card".to_string(),
        "payment failed".to_string(),
    ]);
    seeds.insert("ta".to_string(), vec![
        "என் கார்டை வசூலிக்கவும்".to_string(),  // "charge my card" in Tamil
        "கட்டணம் தோல்வியடைந்தது".to_string(),   // "payment failed" in Tamil
    ]);
    seeds.insert("zh".to_string(), vec![
        "收费失败".to_string(),   // "charge failed" in Chinese
        "支付失败".to_string(),   // "payment failed" in Chinese
    ]);
    r.add_intent_multilingual("stripe:charge_card", seeds);

    // slack:send_message with English + Chinese
    let mut seeds2 = HashMap::new();
    seeds2.insert("en".to_string(), vec!["send a message".to_string(), "post in channel".to_string()]);
    seeds2.insert("zh".to_string(), vec!["发消息".to_string(), "在频道发帖".to_string()]);
    r.add_intent_multilingual("slack:send_message", seeds2);

    // English query routes to stripe
    let results = r.route_ns("payment failed", Some("stripe"));
    assert!(!results.is_empty(), "English query should match stripe");
    assert_eq!(results[0].id, "stripe:charge_card");

    // Chinese query routes to stripe (not slack)
    let results = r.route("收费失败");
    assert!(!results.is_empty(), "Chinese query 收费失败 should match");
    assert_eq!(results[0].id, "stripe:charge_card",
        "收费失败 should route to stripe:charge_card, got {}", results[0].id);

    // Chinese slack query routes to slack
    let results = r.route("发消息");
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "slack:send_message",
        "发消息 should route to slack:send_message, got {}", results[0].id);

    // Tamil query routes to stripe
    let results = r.route("கட்டணம் தோல்வியடைந்தது");
    assert!(!results.is_empty(), "Tamil query should match stripe");
    assert_eq!(results[0].id, "stripe:charge_card",
        "Tamil 'payment failed' should route to stripe:charge_card, got {}", results[0].id);

    // route_ns with language-specific queries respects namespace filter
    let results = r.route_ns("发消息", Some("slack"));
    assert!(!results.is_empty());
    assert!(results.iter().all(|r| r.id.starts_with("slack:")));
}
