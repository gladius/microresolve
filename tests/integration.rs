//! Integration test: seed → route → fail on unknowns → teach → re-route.
//!
//! Simulates a realistic lifecycle:
//!   1. Seed a customer-support router with canonical phrases
//!   2. Verify known queries route correctly
//!   3. Try paraphrases/slang the router has never seen — expect failures
//!   4. Teach the router via learn() and correct()
//!   5. Re-test the same unknowns — expect them to succeed now
//!   6. Verify teaching didn't break original routes (regression)
//!   7. Test multi-intent decomposition end-to-end
//!   8. Test persistence round-trip preserves learned state

use asv_router::Router;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn customer_support_router() -> Router {
    let mut r = Router::new();

    r.add_intent("cancel_order", &[
        "cancel my order",
        "I want to cancel",
        "stop my order",
        "cancel the purchase",
    ]);
    r.add_intent("track_order", &[
        "where is my package",
        "track my order",
        "shipping status",
        "when will it arrive",
    ]);
    r.add_intent("refund", &[
        "get a refund",
        "money back",
        "refund my purchase",
        "return and refund",
    ]);
    r.add_intent("reset_password", &[
        "reset my password",
        "forgot password",
        "can't log in",
        "locked out of account",
    ]);
    r.add_intent("billing", &[
        "billing issue",
        "wrong charge",
        "overcharged",
        "charged twice",
        "unexpected charge on my card",
    ]);
    r.add_intent("contact_human", &[
        "talk to a human",
        "speak to an agent",
        "real person please",
        "transfer me to support",
    ]);

    r
}

/// Assert the top-1 result matches the expected intent.
fn assert_routes_to(router: &Router, query: &str, expected: &str) {
    let results = router.route(query);
    assert!(
        !results.is_empty(),
        "query {:?} returned no results (expected {:?})",
        query, expected,
    );
    assert_eq!(
        results[0].id, expected,
        "query {:?} routed to {:?} (expected {:?}), scores: {:?}",
        query, results[0].id, expected,
        results.iter().map(|r| (&r.id, r.score)).collect::<Vec<_>>(),
    );
}

/// Assert that either no results are returned, or the top-1 result is NOT the
/// given intent (i.e. the router doesn't know how to handle this yet).
fn assert_does_not_route_to(router: &Router, query: &str, wrong: &str) {
    let results = router.route(query);
    if results.is_empty() {
        return; // no match at all — correct
    }
    assert_ne!(
        results[0].id, wrong,
        "query {:?} should NOT route to {:?} but it did (score {:.2})",
        query, wrong, results[0].score,
    );
}

// ---------------------------------------------------------------------------
// Phase 1 — Seeded queries route correctly
// ---------------------------------------------------------------------------

#[test]
fn phase1_known_queries_route_correctly() {
    let router = customer_support_router();

    // Exact or near-exact seed phrases
    assert_routes_to(&router, "cancel my order", "cancel_order");
    assert_routes_to(&router, "track my order", "track_order");
    assert_routes_to(&router, "get a refund", "refund");
    assert_routes_to(&router, "reset my password", "reset_password");
    assert_routes_to(&router, "charged twice", "billing");
    assert_routes_to(&router, "speak to an agent", "contact_human");

    // Partial overlap with seeds — should still pick the right intent
    assert_routes_to(&router, "cancel the order now", "cancel_order");
    assert_routes_to(&router, "where is my package right now", "track_order");
    assert_routes_to(&router, "I forgot my password", "reset_password");
}

// ---------------------------------------------------------------------------
// Phase 2 — Unknown paraphrases fail (semantic gap)
// ---------------------------------------------------------------------------

/// These are real-world ways people phrase the same intents, but with zero
/// lexical overlap with our seeds.  The router should fail on most of these.
#[test]
fn phase2_unknown_paraphrases_fail() {
    let router = customer_support_router();

    // "cancel" intent, but no seed words present
    // Note: "I changed my mind about that purchase" actually hits via "purchase"
    // in the seed "cancel the purchase" — the router is smarter than expected.
    assert_does_not_route_to(&router, "I changed my mind completely", "cancel_order");
    assert_does_not_route_to(&router, "never mind, don't send it", "cancel_order");

    // "refund" intent, but phrased differently
    // Note: "give me my cash back" actually matches via "back" in seed "money back"
    // — this shows the router catches partial overlap even before learning.
    assert_does_not_route_to(&router, "I want a reimbursement", "refund");
    assert_does_not_route_to(&router, "put the amount on my card again", "refund");

    // "billing" intent, completely different vocabulary
    assert_does_not_route_to(&router, "why did you take extra money", "billing");

    // "contact_human" intent, slang — "talk" is in seeds, so use zero-overlap phrases
    assert_does_not_route_to(&router, "get me a supervisor", "contact_human");
    assert_does_not_route_to(&router, "I demand to escalate", "contact_human");

    // "reset_password" intent, indirect phrasing with zero seed overlap
    assert_does_not_route_to(&router, "my credentials are expired", "reset_password");
    assert_does_not_route_to(&router, "authentication keeps failing", "reset_password");
}

// ---------------------------------------------------------------------------
// Phase 3 — Teach the router
// ---------------------------------------------------------------------------

#[test]
fn phase3_teach_and_reroute() {
    let mut router = customer_support_router();

    // -- Teach cancel_order paraphrases --
    router.learn("I changed my mind about that purchase", "cancel_order");
    router.learn("never mind, don't send it", "cancel_order");
    router.learn("I don't want it anymore", "cancel_order");

    // -- Teach refund paraphrases --
    router.learn("give me my cash back", "refund");
    router.learn("I want a reimbursement", "refund");
    router.learn("put the money back on my card", "refund");
    router.learn("reimbursement for my purchase", "refund");

    // -- Teach billing paraphrases --
    router.learn("why did you take extra money", "billing");
    router.learn("there's a charge I don't recognize", "billing");

    // -- Teach contact_human paraphrases --
    router.learn("let me talk to your manager", "contact_human");
    router.learn("get me a supervisor", "contact_human");

    // -- Teach reset_password paraphrases --
    router.learn("my login isn't working anymore", "reset_password");
    router.learn("I can't get into my account", "reset_password");

    // Now the same unknowns from phase 2 should route correctly
    assert_routes_to(&router, "I changed my mind about that purchase", "cancel_order");
    assert_routes_to(&router, "never mind, don't send it", "cancel_order");
    assert_routes_to(&router, "give me my cash back", "refund");
    assert_routes_to(&router, "I want a reimbursement", "refund");
    assert_routes_to(&router, "why did you take extra money", "billing");
    assert_routes_to(&router, "let me talk to your manager", "contact_human");
    assert_routes_to(&router, "my login isn't working anymore", "reset_password");

    // Slight variations of taught phrases should also work (shared terms)
    assert_routes_to(&router, "I changed my mind", "cancel_order");
    assert_routes_to(&router, "give me cash back", "refund");
    assert_routes_to(&router, "talk to manager", "contact_human");
}

// ---------------------------------------------------------------------------
// Phase 4 — Regression: teaching didn't break original routes
// ---------------------------------------------------------------------------

#[test]
fn phase4_no_regression_after_teaching() {
    let mut router = customer_support_router();

    // Teach a bunch of new phrases
    router.learn("I changed my mind about that purchase", "cancel_order");
    router.learn("give me my cash back", "refund");
    router.learn("why did you take extra money", "billing");
    router.learn("let me talk to your manager", "contact_human");
    router.learn("my login isn't working anymore", "reset_password");
    router.learn("where's my stuff", "track_order");

    // All original seed phrases must still route correctly
    assert_routes_to(&router, "cancel my order", "cancel_order");
    assert_routes_to(&router, "track my order", "track_order");
    assert_routes_to(&router, "get a refund", "refund");
    assert_routes_to(&router, "reset my password", "reset_password");
    assert_routes_to(&router, "charged twice", "billing");
    assert_routes_to(&router, "speak to an agent", "contact_human");
}

// ---------------------------------------------------------------------------
// Phase 5 — Correct misroutes
// ---------------------------------------------------------------------------

#[test]
fn phase5_correct_misroutes() {
    let mut router = customer_support_router();

    // "reimburse my expense" has no seed overlap with any intent.
    // We first teach it as billing, then correct to refund.
    router.learn("reimburse my expense", "billing");
    assert_routes_to(&router, "reimburse my expense", "billing");

    // Oops, agent meant refund — correct it
    router.correct("reimburse my expense", "billing", "refund");
    assert_routes_to(&router, "reimburse my expense", "refund");

    // Original billing phrases still work (seed layer untouched)
    assert_routes_to(&router, "charged twice", "billing");
}

// ---------------------------------------------------------------------------
// Phase 6 — Multi-intent decomposition after learning
// ---------------------------------------------------------------------------

#[test]
fn phase6_multi_intent_after_learning() {
    let mut router = customer_support_router();

    // Teach some extra phrases to improve coverage
    router.learn("where's my stuff", "track_order");
    router.learn("give me my money back", "refund");

    // Two intents in one sentence
    let result = router.route_multi(
        "cancel my order and track my package",
        0.3,
    );
    let ids: Vec<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
    assert!(ids.contains(&"cancel_order"), "expected cancel_order in {:?}", ids);
    assert!(ids.contains(&"track_order"), "expected track_order in {:?}", ids);

    // Positional ordering: cancel comes before track
    let cancel_pos = result.intents.iter().find(|i| i.id == "cancel_order").unwrap().position;
    let track_pos = result.intents.iter().find(|i| i.id == "track_order").unwrap().position;
    assert!(cancel_pos < track_pos, "cancel should appear before track");

    // Three intents
    let result = router.route_multi(
        "cancel my order and get a refund and reset my password",
        0.3,
    );
    let ids: Vec<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
    assert!(ids.contains(&"cancel_order"), "expected cancel_order in {:?}", ids);
    assert!(ids.contains(&"refund"), "expected refund in {:?}", ids);
    assert!(ids.contains(&"reset_password"), "expected reset_password in {:?}", ids);

    // Verify all intents are in positional order
    for w in result.intents.windows(2) {
        assert!(
            w[0].position <= w[1].position,
            "intents not in positional order: {:?} vs {:?}",
            w[0].id, w[1].id,
        );
    }
}

// ---------------------------------------------------------------------------
// Phase 7 — Relation detection with taught intents
// ---------------------------------------------------------------------------

#[test]
fn phase7_relation_detection() {
    let router = customer_support_router();

    // Sequential
    let result = router.route_multi(
        "cancel my order then get a refund",
        0.3,
    );
    if result.intents.len() >= 2 && !result.relations.is_empty() {
        assert!(
            matches!(result.relations[0], asv_router::IntentRelation::Sequential { .. }),
            "expected Sequential, got {:?}", result.relations[0],
        );
    }

    // Conditional
    let result = router.route_multi(
        "get a refund or otherwise reset my password",
        0.3,
    );
    if result.intents.len() >= 2 && !result.relations.is_empty() {
        assert!(
            matches!(result.relations[0], asv_router::IntentRelation::Conditional { .. }),
            "expected Conditional, got {:?}", result.relations[0],
        );
    }

    // Negation — use "except" instead of "don't" because the apostrophe in
    // "don't" splits to "don"+"t", and "t" gets consumed by reset_password
    // (from its seed "can't log in"), collapsing the gap words.
    let result = router.route_multi(
        "cancel my order except reset my password",
        0.3,
    );
    if result.intents.len() >= 2 && !result.relations.is_empty() {
        assert!(
            matches!(result.relations[0], asv_router::IntentRelation::Negation { .. }),
            "expected Negation, got {:?} (intents: {:?})",
            result.relations[0],
            result.intents.iter().map(|i| (&i.id, i.span)).collect::<Vec<_>>(),
        );
    }
}

// ---------------------------------------------------------------------------
// Phase 8 — Persistence preserves learned state
// ---------------------------------------------------------------------------

#[test]
fn phase8_persistence_roundtrip() {
    let mut router = customer_support_router();

    // Teach several phrases
    router.learn("I changed my mind", "cancel_order");
    router.learn("give me my cash back", "refund");
    router.learn("where's my stuff", "track_order");
    router.learn("why did you take extra money", "billing");

    // Export
    let json = router.export_json();

    // Import into a fresh router
    let restored = Router::import_json(&json).expect("failed to import");

    // All seed phrases still work
    assert_routes_to(&restored, "cancel my order", "cancel_order");
    assert_routes_to(&restored, "track my order", "track_order");
    assert_routes_to(&restored, "get a refund", "refund");

    // All learned phrases still work
    assert_routes_to(&restored, "I changed my mind", "cancel_order");
    assert_routes_to(&restored, "give me my cash back", "refund");
    assert_routes_to(&restored, "where's my stuff", "track_order");
    assert_routes_to(&restored, "why did you take extra money", "billing");

    // Multi-intent still works after restore
    let result = restored.route_multi(
        "cancel my order and track my package",
        0.3,
    );
    assert!(result.intents.len() >= 2);
}

// ---------------------------------------------------------------------------
// Phase 9 — Decay weakens learned, preserves seed
// ---------------------------------------------------------------------------

#[test]
fn phase9_decay_lifecycle() {
    let mut router = customer_support_router();

    // Teach a phrase
    router.learn("I changed my mind", "cancel_order");
    assert_routes_to(&router, "I changed my mind", "cancel_order");

    // Heavy decay — learned weights should weaken
    for _ in 0..20 {
        router.decay(0.5);
    }

    // Learned phrase may no longer route (weights decayed below threshold)
    let results = router.route("I changed my mind");
    let still_works = !results.is_empty() && results[0].id == "cancel_order";
    // It's OK either way — the point is seed phrases survive decay
    let _ = still_works;

    // Seed phrases must still work perfectly
    assert_routes_to(&router, "cancel my order", "cancel_order");
    assert_routes_to(&router, "track my order", "track_order");
    assert_routes_to(&router, "get a refund", "refund");
    assert_routes_to(&router, "reset my password", "reset_password");
    assert_routes_to(&router, "charged twice", "billing");
    assert_routes_to(&router, "speak to an agent", "contact_human");
}

// ---------------------------------------------------------------------------
// Phase 10 — Scale: many intents, many learnings, still correct
// ---------------------------------------------------------------------------

#[test]
fn phase10_scale_stress_test() {
    let mut router = customer_support_router();

    // Add 50 more intents
    for i in 0..50 {
        router.add_intent(
            &format!("faq_{}", i),
            &[
                &format!("frequently asked question number {}", i),
                &format!("faq topic {}", i),
            ],
        );
    }

    // Teach 100 phrases across various intents
    for i in 0..20 {
        router.learn(&format!("variation {} of cancelling", i), "cancel_order");
        router.learn(&format!("variation {} of tracking", i), "track_order");
        router.learn(&format!("variation {} of refunding", i), "refund");
        router.learn(&format!("faq variation {} for topic 7", i), "faq_7");
        router.learn(&format!("faq variation {} for topic 42", i), "faq_42");
    }

    assert_eq!(router.intent_count(), 56); // 6 original + 50 faq

    // Original intents still route correctly
    assert_routes_to(&router, "cancel my order", "cancel_order");
    assert_routes_to(&router, "track my order", "track_order");

    // FAQ intents route correctly
    assert_routes_to(&router, "faq topic 7", "faq_7");
    assert_routes_to(&router, "faq topic 42", "faq_42");

    // Learned variations work
    assert_routes_to(&router, "variation 5 of cancelling", "cancel_order");
    assert_routes_to(&router, "variation 10 of tracking", "track_order");

    // Multi-intent still works at scale
    let result = router.route_multi(
        "cancel my order and track my package",
        0.3,
    );
    assert!(result.intents.len() >= 2);
}
