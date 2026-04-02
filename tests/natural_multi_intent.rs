//! Natural conversation multi-intent tests.
//!
//! These test realistic user messages — not synthetic "X and also Y" concatenations.
//! Covers: casual requests, frustrated customers, multi-sentence, negation,
//! single-intent (no false multi-detection), edge cases, and 3-intent queries.

use asv_router::Router;
use std::collections::HashSet;

fn build_router() -> Router {
    let mut r = Router::new();

    r.add_intent("cancel_order", &[
        "cancel my order", "I want to cancel", "stop my order",
        "cancel the purchase", "revoke my order", "withdraw my order",
        "abort the order", "call off my order", "cancel that order",
        "I need to cancel an order",
    ]);
    r.add_intent("track_order", &[
        "where is my package", "track my order", "shipping status",
        "when will it arrive", "order tracking", "delivery status",
        "where is my delivery", "track my package", "package tracking",
        "when does my order arrive",
    ]);
    r.add_intent("refund", &[
        "get a refund", "money back", "refund my purchase",
        "return and refund", "I want my money back", "refund please",
        "process a refund", "give me a refund", "refund the amount",
        "reimburse me",
    ]);
    r.add_intent("reset_password", &[
        "reset my password", "forgot password", "change my password",
        "password reset", "new password", "update my password",
        "I forgot my login password", "help me reset password",
        "password change request", "lost my password",
    ]);
    r.add_intent("billing", &[
        "billing issue", "wrong charge", "overcharged",
        "charged twice", "unexpected charge", "billing error",
        "incorrect billing", "double charged", "billing problem",
        "charge dispute",
    ]);
    r.add_intent("check_balance", &[
        "check my balance", "account balance", "how much money",
        "what is my balance", "show my balance", "remaining balance",
        "current balance", "funds available", "balance inquiry",
        "balance check",
    ]);
    r.add_intent("transfer_money", &[
        "transfer money", "send money", "wire transfer",
        "move money", "transfer funds", "send funds",
        "bank transfer", "money transfer", "wire funds",
        "transfer to account",
    ]);
    r.add_intent("close_account", &[
        "close my account", "delete my account", "deactivate account",
        "shut down account", "remove my account", "terminate account",
        "cancel my account", "end my account", "close account please",
        "I want to close my account",
    ]);
    r.add_intent("contact_support", &[
        "talk to agent", "speak to human", "customer support",
        "contact support", "real person", "live agent",
        "help from agent", "human assistance", "speak to representative",
        "transfer to support",
    ]);
    r.add_intent("upgrade_plan", &[
        "upgrade my plan", "better plan", "premium plan",
        "change my plan", "upgrade subscription", "plan upgrade",
        "switch to premium", "upgrade account", "higher tier",
        "upgrade membership",
    ]);

    r
}

// ─── Short casual multi-intent ─────────────────────────────────────────

#[test]
fn natural_short_casual() {
    let router = build_router();

    let cases: Vec<(&str, Vec<&str>)> = vec![
        ("hey cancel my order and give me a refund", vec!["cancel_order", "refund"]),
        ("check my balance and send money to mom", vec!["check_balance", "transfer_money"]),
        ("track my order, also I was overcharged", vec!["track_order", "billing"]),
        ("refund and cancel", vec!["refund", "cancel_order"]),
    ];

    for (query, expected) in &cases {
        let result = router.route_multi(query, 0.3);
        let detected: HashSet<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        for exp in expected {
            assert!(
                detected.contains(exp),
                "query {:?}: missing intent '{}', detected: {:?}",
                query, exp, detected
            );
        }
    }
}

// ─── Frustrated customer run-on messages ──────────────────────────────

#[test]
fn natural_frustrated_customer() {
    let router = build_router();

    let cases: Vec<(&str, Vec<&str>)> = vec![
        ("where is my package and why was I charged twice", vec!["track_order", "billing"]),
        ("I want a refund and I need to talk to someone right now", vec!["refund", "contact_support"]),
        (
            "this is the third time I've asked to cancel my order just cancel it and give me my money back",
            vec!["cancel_order", "refund"],
        ),
    ];

    for (query, expected) in &cases {
        let result = router.route_multi(query, 0.3);
        let detected: HashSet<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        for exp in expected {
            assert!(
                detected.contains(exp),
                "query {:?}: missing intent '{}', detected: {:?}",
                query, exp, detected
            );
        }
    }
}

// ─── Polite multi-sentence ────────────────────────────────────────────

#[test]
fn natural_polite_multi_sentence() {
    let router = build_router();

    let cases: Vec<(&str, Vec<&str>)> = vec![
        ("Hi, I'd like to reset my password. Also, could you check my balance?", vec!["reset_password", "check_balance"]),
        ("I need to close my account. But first, can you transfer my funds?", vec!["close_account", "transfer_money"]),
        ("Could you please upgrade my plan? And I'd also like to check my current balance.", vec!["upgrade_plan", "check_balance"]),
    ];

    for (query, expected) in &cases {
        let result = router.route_multi(query, 0.3);
        let detected: HashSet<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        for exp in expected {
            assert!(
                detected.contains(exp),
                "query {:?}: missing intent '{}', detected: {:?}",
                query, exp, detected
            );
        }
    }
}

// ─── Negation — natural phrasing ──────────────────────────────────────

#[test]
fn natural_negation_detection() {
    let router = build_router();

    // Both intents should be DETECTED (negated one included)
    let cases: Vec<(&str, Vec<&str>, &str)> = vec![
        ("cancel my order but don't refund me", vec!["cancel_order", "refund"], "refund"),
        ("close my account but don't transfer money", vec!["close_account", "transfer_money"], "transfer_money"),
        ("I want a refund but don't close my account", vec!["refund", "close_account"], "close_account"),
        ("track my package without cancel the order", vec!["track_order", "cancel_order"], "cancel_order"),
    ];

    for (query, expected, negated_intent) in &cases {
        let result = router.route_multi(query, 0.3);
        let detected: HashSet<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();

        // Check all intents detected (including negated one)
        for exp in expected {
            assert!(
                detected.contains(exp),
                "query {:?}: missing intent '{}', detected: {:?}",
                query, exp, detected
            );
        }

        // Check negated flag is set on the right intent
        let neg_intent = result.intents.iter().find(|i| i.id == *negated_intent);
        if let Some(ni) = neg_intent {
            assert!(
                ni.negated,
                "query {:?}: intent '{}' should be flagged as negated but negated={}",
                query, negated_intent, ni.negated
            );
        }
    }
}

// ─── Single intent — no false multi-detection ─────────────────────────

#[test]
fn natural_single_intent() {
    let router = build_router();

    let cases: Vec<(&str, &str)> = vec![
        ("where is my package", "track_order"),
        ("I need a refund please", "refund"),
        ("can I speak to a real person", "contact_support"),
        ("I forgot my password and I can't log in", "reset_password"),
        ("how much money do I have", "check_balance"),
        ("I want to cancel my order", "cancel_order"),
        ("there's a wrong charge on my account", "billing"),
    ];

    for (query, expected_top) in &cases {
        let result = router.route_multi(query, 0.3);
        assert!(
            !result.intents.is_empty(),
            "query {:?}: no intents detected",
            query
        );
        // Check that the expected intent is detected (may not be positionally first
        // if route_multi sorts by position and a weaker intent matched earlier text)
        let highest = result.intents.iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap();
        assert_eq!(
            highest.id, *expected_top,
            "query {:?}: highest-scoring intent should be '{}' but got '{}' (score={:.2})",
            query, expected_top, highest.id, highest.score
        );
    }
}

// ─── Three-intent natural ─────────────────────────────────────────────

#[test]
fn natural_three_intents() {
    let router = build_router();

    let cases: Vec<(&str, Vec<&str>)> = vec![
        (
            "check my balance, transfer money to savings, and close the account",
            vec!["check_balance", "transfer_money", "close_account"],
        ),
        (
            "cancel my order and get a refund and then close my account",
            vec!["cancel_order", "refund", "close_account"],
        ),
    ];

    for (query, expected) in &cases {
        let result = router.route_multi(query, 0.3);
        let detected: HashSet<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        let found = expected.iter().filter(|e| detected.contains(*e)).count();
        assert!(
            found >= 2,
            "query {:?}: only {}/3 expected intents detected, got {:?}",
            query, found, detected
        );
    }
}

// ─── Edge cases ───────────────────────────────────────────────────────

#[test]
fn natural_edge_cases() {
    let router = build_router();

    // Empty
    assert!(router.route_multi("", 0.3).intents.is_empty());

    // Gibberish
    assert!(router.route_multi("asdfghjkl qwerty zxcvbn", 0.3).intents.is_empty());

    // All stop words
    assert!(router.route_multi("the a an in on at to of for by", 0.3).intents.is_empty());

    // Very long query with one intent buried at the end
    let result = router.route_multi(
        "so basically what happened was I ordered something last week and it was supposed to arrive on Monday but now it's Wednesday and I still haven't gotten it so I want to track my order",
        0.3,
    );
    assert!(
        result.intents.iter().any(|i| i.id == "track_order"),
        "track_order should be found in long narrative, got: {:?}",
        result.intents.iter().map(|i| &i.id).collect::<Vec<_>>()
    );
}

// ─── Positional ordering ──────────────────────────────────────────────

#[test]
fn natural_positional_ordering() {
    let router = build_router();

    let result = router.route_multi("cancel my order and track my package", 0.3);
    if result.intents.len() >= 2 {
        let cancel_pos = result.intents.iter().position(|i| i.id == "cancel_order");
        let track_pos = result.intents.iter().position(|i| i.id == "track_order");
        if let (Some(cp), Some(tp)) = (cancel_pos, track_pos) {
            assert!(cp < tp, "cancel_order should come before track_order");
        }
    }

    let result = router.route_multi("track my package and then cancel my order", 0.3);
    if result.intents.len() >= 2 {
        let track_pos = result.intents.iter().position(|i| i.id == "track_order");
        let cancel_pos = result.intents.iter().position(|i| i.id == "cancel_order");
        if let (Some(tp), Some(cp)) = (track_pos, cancel_pos) {
            assert!(tp < cp, "track_order should come before cancel_order");
        }
    }
}

// ─── Relation detection ──────────────────────────────────────────────

#[test]
fn natural_relation_detection() {
    let router = build_router();

    // Sequential: use intents with non-overlapping vocabulary
    let result = router.route_multi("reset my password then track my package", 0.3);
    assert!(result.intents.len() >= 2, "should detect 2 intents, got {}", result.intents.len());
    if !result.relations.is_empty() {
        assert!(
            matches!(result.relations[0], asv_router::IntentRelation::Sequential { .. }),
            "expected Sequential for 'then', got {:?}",
            result.relations[0]
        );
    }

    // Negation relation
    let result = router.route_multi("cancel my order but don't refund me", 0.3);
    assert!(result.intents.len() >= 2, "should detect 2 intents for negation");
    // Verify the negated intent has the flag
    let refund = result.intents.iter().find(|i| i.id == "refund");
    if let Some(r) = refund {
        assert!(r.negated, "refund should be flagged as negated");
    }
}

// ─── After learning — should improve detection ────────────────────────

#[test]
fn natural_after_learning() {
    let mut router = build_router();

    // Teach slang
    router.learn("where's my stuff", "track_order");
    router.learn("hit me with that refund", "refund");

    // After learning: should now detect both
    let result = router.route_multi("yo where's my stuff and hit me with that refund", 0.3);
    let detected: HashSet<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();

    assert!(
        detected.contains("track_order"),
        "should detect track_order after learning slang, got: {:?}", detected
    );
    assert!(
        detected.contains("refund"),
        "should detect refund after learning slang, got: {:?}", detected
    );
}

// ─── Messy paragraph dumps — real chat behavior ─────────────────────

#[test]
fn natural_messy_paragraph() {
    let router = build_router();

    let cases: Vec<(&str, Vec<&str>)> = vec![
        // Run-on no punctuation, multiple intents buried in story
        (
            "ok so I just checked my account and I think I got charged twice for something and also where is my package from last week it still hasnt arrived can someone please look into this",
            vec!["billing", "track_order"],
        ),
        // Venting with requests scattered throughout
        (
            "honestly I am so frustrated with this service I have been waiting for my package for two weeks now and nobody is helping me I just want to talk to a real person and get my money back",
            vec!["contact_support", "refund"],
        ),
        // Casual mid-sentence topic switch
        (
            "I need to reset my password oh and while youre at it can you check my balance too",
            vec!["reset_password", "check_balance"],
        ),
        // Story wrapping around intent
        (
            "so my friend told me I should check my balance because apparently there was a breach and people were losing money so I want to check my balance and also transfer everything to my savings just to be safe",
            vec!["check_balance", "transfer_money"],
        ),
        // Multiple sentences, filler between intents
        (
            "hi there so I placed an order last week and I need to cancel it because I found it cheaper elsewhere also I noticed you charged me twice for my previous order which is not cool",
            vec!["cancel_order", "billing"],
        ),
    ];

    for (query, expected) in &cases {
        let result = router.route_multi(query, 0.3);
        let detected: HashSet<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        for exp in expected {
            assert!(
                detected.contains(exp),
                "messy paragraph {:?}: missing intent '{}', detected: {:?}",
                query, exp, detected
            );
        }
    }
}

// ─── Complaint sandwich — requests buried in frustration ────────────

#[test]
fn natural_complaint_sandwich() {
    let router = build_router();

    let cases: Vec<(&str, Vec<&str>)> = vec![
        // 3 intents buried in 5-year-customer rant
        (
            "I have been a customer for 5 years and honestly the service lately has been terrible I need to cancel my order because its been 3 weeks and I want a full refund if things dont improve I might just close my account entirely",
            vec!["cancel_order", "refund"],
        ),
        // Anger then multiple requests
        (
            "this is ridiculous you guys messed up my billing again I see a double charge on my statement and on top of that my package never arrived I want to speak to someone in charge",
            vec!["billing", "contact_support"],
        ),
        // Escalation: context then two asks
        (
            "look I already called twice about this nobody seems to care my order is wrong and I want my money back please just give me a refund and cancel the whole thing",
            vec!["refund", "cancel_order"],
        ),
    ];

    for (query, expected) in &cases {
        let result = router.route_multi(query, 0.3);
        let detected: HashSet<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        for exp in expected {
            assert!(
                detected.contains(exp),
                "complaint sandwich {:?}: missing intent '{}', detected: {:?}",
                query, exp, detected
            );
        }
    }
}

// ─── Questions as intents — real people ask, not command ─────────────

#[test]
fn natural_questions_as_intents() {
    let router = build_router();

    let cases: Vec<(&str, Vec<&str>)> = vec![
        // Multiple questions in one message
        (
            "whats my current balance and can I transfer some to my other account",
            vec!["check_balance", "transfer_money"],
        ),
        // Question + request
        (
            "where is my package I ordered last tuesday and also can you reset my password I cant log in",
            vec!["track_order", "reset_password"],
        ),
        // Polite question chain
        (
            "could you tell me my balance and also help me upgrade my plan to premium",
            vec!["check_balance", "upgrade_plan"],
        ),
    ];

    for (query, expected) in &cases {
        let result = router.route_multi(query, 0.3);
        let detected: HashSet<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        for exp in expected {
            assert!(
                detected.contains(exp),
                "question intent {:?}: missing intent '{}', detected: {:?}",
                query, exp, detected
            );
        }
    }
}

// ─── Multi-paragraph single message — like real chat ────────────────

#[test]
fn natural_multi_paragraph_single_message() {
    let router = build_router();

    let cases: Vec<(&str, Vec<&str>)> = vec![
        // Two separate thoughts, no conjunction
        (
            "I forgot my password. Also my last order never showed up where is it",
            vec!["reset_password", "track_order"],
        ),
        // Stream of consciousness with pivot
        (
            "I was trying to log in but forgot my password so I need that fixed. while Im here can you also check if I have enough balance to transfer money next week",
            vec!["reset_password", "check_balance"],
        ),
        // Separate asks, different tones
        (
            "please cancel my order I changed my mind. oh by the way can I upgrade to the premium plan",
            vec!["cancel_order", "upgrade_plan"],
        ),
    ];

    for (query, expected) in &cases {
        let result = router.route_multi(query, 0.3);
        let detected: HashSet<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        for exp in expected {
            assert!(
                detected.contains(exp),
                "multi-paragraph {:?}: missing intent '{}', detected: {:?}",
                query, exp, detected
            );
        }
    }
}

// ─── Intent buried deep in long ramble ──────────────────────────────

#[test]
fn natural_intent_buried_in_ramble() {
    let router = build_router();

    let cases: Vec<(&str, Vec<&str>)> = vec![
        // 2 intents at the very end of a long story
        (
            "so basically I signed up for this service about three months ago and everything was fine until last week when I noticed something weird on my statement it looks like I was charged twice and I also cant find my package anywhere so yeah I need help with the billing and also please track my order",
            vec!["billing", "track_order"],
        ),
        // Intent scattered: one at start, one buried at end
        (
            "cancel my order please I dont want it anymore I was just browsing and accidentally clicked buy anyway while you handle that can you also check how much balance I have left in my account",
            vec!["cancel_order", "check_balance"],
        ),
    ];

    for (query, expected) in &cases {
        let result = router.route_multi(query, 0.3);
        let detected: HashSet<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        for exp in expected {
            assert!(
                detected.contains(exp),
                "buried intent {:?}: missing intent '{}', detected: {:?}",
                query, exp, detected
            );
        }
    }
}

// ─── Three intents in messy chat — real multi-ask ───────────────────

#[test]
fn natural_three_intents_messy() {
    let router = build_router();

    let cases: Vec<(&str, Vec<&str>)> = vec![
        // 3 intents, no structure
        (
            "ok so I need to cancel my order and get a refund for it and while youre at it can someone check my balance because I think the refund from last time never went through",
            vec!["cancel_order", "refund", "check_balance"],
        ),
        // 3 intents in frustrated ramble
        (
            "I want to close my account I am done with this but first transfer my remaining balance to my bank and also reset my password because I think someone else has been using it",
            vec!["close_account", "transfer_money", "reset_password"],
        ),
    ];

    for (query, expected) in &cases {
        let result = router.route_multi(query, 0.3);
        let detected: HashSet<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        let found = expected.iter().filter(|e| detected.contains(*e)).count();
        assert!(
            found >= 2,
            "messy 3-intent {:?}: only {}/3 expected intents detected, got {:?}",
            query, found, detected
        );
    }
}
