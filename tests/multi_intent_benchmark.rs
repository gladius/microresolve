//! Multi-intent benchmark — synthesize compound queries and test decomposition.
//!
//! Since no standard multi-intent dataset exists, we construct one:
//!   1. Take real single-intent queries from CLINC150/BANKING77
//!   2. Combine 2-3 of them with connectors ("and", "then", "but don't", etc.)
//!   3. Test whether route_multi correctly decomposes them
//!   4. Measure: intent detection accuracy, positional ordering, relation detection

use asv_router::{IntentRelation, Router};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

fn build_router() -> Router {
    let mut r = Router::new();

    // 12 intents with rich seed phrases — enough to stress test decomposition
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
    r.add_intent("update_address", &[
        "update my address", "change my address", "new address",
        "address change", "move to new address", "update shipping address",
        "change delivery address", "modify my address", "edit address",
        "address update",
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
    r.add_intent("apply_coupon", &[
        "apply coupon", "discount code", "promo code",
        "use coupon", "redeem voucher", "apply discount",
        "coupon code", "promotional offer", "enter promo",
        "apply voucher",
    ]);

    r
}

// ---------------------------------------------------------------------------
// Compound query builder
// ---------------------------------------------------------------------------

struct CompoundQuery {
    text: String,
    expected_intents: Vec<String>,
    expected_relation: Option<ExpectedRelation>,
}

#[derive(Debug, Clone, PartialEq)]
enum ExpectedRelation {
    Parallel,
    Sequential,
    Conditional,
    Negation,
}

fn parallel(texts: &[(&str, &str)]) -> Vec<CompoundQuery> {
    let connectors = ["and also", "and", "plus", "as well as"];
    let mut queries = Vec::new();

    for (i, &(t1, t2)) in texts.iter().enumerate() {
        let conn = connectors[i % connectors.len()];
        let (i1, q1) = parse_intent_query(t1);
        let (i2, q2) = parse_intent_query(t2);
        queries.push(CompoundQuery {
            text: format!("{} {} {}", q1, conn, q2),
            expected_intents: vec![i1, i2],
            expected_relation: Some(ExpectedRelation::Parallel),
        });
    }
    queries
}

fn sequential(texts: &[(&str, &str)]) -> Vec<CompoundQuery> {
    let connectors = ["then", "and then", "after that", "next"];
    let mut queries = Vec::new();

    for (i, &(t1, t2)) in texts.iter().enumerate() {
        let conn = connectors[i % connectors.len()];
        let (i1, q1) = parse_intent_query(t1);
        let (i2, q2) = parse_intent_query(t2);
        queries.push(CompoundQuery {
            text: format!("{} {} {}", q1, conn, q2),
            expected_intents: vec![i1, i2],
            expected_relation: Some(ExpectedRelation::Sequential),
        });
    }
    queries
}

fn conditional(texts: &[(&str, &str)]) -> Vec<CompoundQuery> {
    let connectors = ["or otherwise", "or", "otherwise", "if not"];
    let mut queries = Vec::new();

    for (i, &(t1, t2)) in texts.iter().enumerate() {
        let conn = connectors[i % connectors.len()];
        let (i1, q1) = parse_intent_query(t1);
        let (i2, q2) = parse_intent_query(t2);
        queries.push(CompoundQuery {
            text: format!("{} {} {}", q1, conn, q2),
            expected_intents: vec![i1, i2],
            expected_relation: Some(ExpectedRelation::Conditional),
        });
    }
    queries
}

fn negation(texts: &[(&str, &str)]) -> Vec<CompoundQuery> {
    let connectors = ["except", "without", "but without"];
    let mut queries = Vec::new();

    for (i, &(t1, t2)) in texts.iter().enumerate() {
        let conn = connectors[i % connectors.len()];
        let (i1, q1) = parse_intent_query(t1);
        let (i2, q2) = parse_intent_query(t2);
        queries.push(CompoundQuery {
            text: format!("{} {} {}", q1, conn, q2),
            expected_intents: vec![i1, i2],
            expected_relation: Some(ExpectedRelation::Negation),
        });
    }
    queries
}

fn triple(texts: &[(&str, &str, &str)]) -> Vec<CompoundQuery> {
    let mut queries = Vec::new();
    for &(t1, t2, t3) in texts {
        let (i1, q1) = parse_intent_query(t1);
        let (i2, q2) = parse_intent_query(t2);
        let (i3, q3) = parse_intent_query(t3);
        queries.push(CompoundQuery {
            text: format!("{} and {} and {}", q1, q2, q3),
            expected_intents: vec![i1, i2, i3],
            expected_relation: None, // mixed relations
        });
    }
    queries
}

/// Parse "intent:query text" format
fn parse_intent_query(s: &str) -> (String, String) {
    let parts: Vec<&str> = s.splitn(2, ':').collect();
    assert_eq!(parts.len(), 2, "expected 'intent:query', got {:?}", s);
    (parts[0].to_string(), parts[1].to_string())
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

struct MultiIntentResults {
    total: usize,
    all_intents_detected: usize,
    correct_order: usize,
    correct_relation: usize,
    partial_detection: usize, // at least one intent found
    // Per-category
    failures: Vec<String>,
}

fn evaluate_multi(router: &Router, queries: &[CompoundQuery], threshold: f32) -> MultiIntentResults {
    let mut results = MultiIntentResults {
        total: queries.len(),
        all_intents_detected: 0,
        correct_order: 0,
        correct_relation: 0,
        partial_detection: 0,
        failures: Vec::new(),
    };

    for q in queries {
        let output = router.route_multi(&q.text, threshold);
        let detected_ids: Vec<&str> = output.intents.iter().map(|i| i.id.as_str()).collect();
        let detected_set: HashSet<&str> = detected_ids.iter().copied().collect();
        let expected_set: HashSet<&str> = q.expected_intents.iter().map(|s| s.as_str()).collect();

        // Check: all expected intents detected?
        let all_found = expected_set.iter().all(|e| detected_set.contains(e));

        if all_found {
            results.all_intents_detected += 1;

            // Check positional ordering: expected intents should appear in order
            let expected_positions: Vec<usize> = q
                .expected_intents
                .iter()
                .filter_map(|e| detected_ids.iter().position(|d| d == e))
                .collect();
            let ordered = expected_positions.windows(2).all(|w| w[0] < w[1]);
            if ordered {
                results.correct_order += 1;
            }

            // Check relation detection
            if let Some(ref expected_rel) = q.expected_relation {
                if output.intents.len() >= 2 && !output.relations.is_empty() {
                    let actual_matches = match (&output.relations[0], expected_rel) {
                        (IntentRelation::Parallel, ExpectedRelation::Parallel) => true,
                        (IntentRelation::Sequential { .. }, ExpectedRelation::Sequential) => true,
                        (IntentRelation::Conditional { .. }, ExpectedRelation::Conditional) => true,
                        (IntentRelation::Negation { .. }, ExpectedRelation::Negation) => true,
                        _ => false,
                    };
                    if actual_matches {
                        results.correct_relation += 1;
                    }
                }
            }
        } else {
            // Partial?
            let any_found = expected_set.iter().any(|e| detected_set.contains(e));
            if any_found {
                results.partial_detection += 1;
            }

            let missing: Vec<&&str> = expected_set.difference(&detected_set).collect();
            let extra: Vec<&&str> = detected_set.difference(&expected_set).collect();
            results.failures.push(format!(
                "  query: {:?}\n    expected: {:?}\n    detected: {:?}\n    missing: {:?}, extra: {:?}",
                q.text, q.expected_intents, detected_ids, missing, extra
            ));
        }
    }

    results
}

fn print_results(label: &str, r: &MultiIntentResults, show_failures: bool) {
    let pct = |n: usize| -> f32 { n as f32 / r.total.max(1) as f32 * 100.0 };
    println!("  {}", label);
    println!(
        "    All intents detected: {}/{} ({:.1}%)",
        r.all_intents_detected, r.total, pct(r.all_intents_detected)
    );
    println!(
        "    Correct order:        {}/{} ({:.1}%)",
        r.correct_order, r.total, pct(r.correct_order)
    );
    println!(
        "    Correct relation:     {}/{} ({:.1}%)",
        r.correct_relation, r.total, pct(r.correct_relation)
    );
    println!(
        "    Partial (>=1 found):  {}/{} ({:.1}%)",
        r.partial_detection + r.all_intents_detected,
        r.total,
        pct(r.partial_detection + r.all_intents_detected)
    );
    if show_failures && !r.failures.is_empty() {
        println!("    Failures (first 10):");
        for f in r.failures.iter().take(10) {
            println!("{}", f);
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// Test datasets
// ---------------------------------------------------------------------------

fn two_intent_parallel_queries() -> Vec<CompoundQuery> {
    parallel(&[
        ("cancel_order:cancel my order", "track_order:track my package"),
        ("refund:get a refund", "check_balance:check my balance"),
        ("reset_password:reset my password", "update_address:update my address"),
        ("billing:dispute a charge", "contact_support:talk to agent"),
        ("transfer_money:transfer money", "check_balance:check balance"),
        ("upgrade_plan:upgrade my plan", "apply_coupon:apply coupon"),
        ("cancel_order:cancel the purchase", "refund:refund my purchase"),
        ("track_order:shipping status", "update_address:change my address"),
        ("close_account:close my account", "transfer_money:transfer funds"),
        ("billing:wrong charge", "refund:money back"),
        ("reset_password:forgot password", "contact_support:speak to human"),
        ("check_balance:account balance", "transfer_money:send money"),
        ("apply_coupon:discount code", "upgrade_plan:premium plan"),
        ("cancel_order:stop my order", "close_account:deactivate account"),
        ("track_order:when will it arrive", "billing:overcharged"),
        ("refund:reimburse me", "cancel_order:revoke my order"),
    ])
}

fn two_intent_sequential_queries() -> Vec<CompoundQuery> {
    sequential(&[
        ("transfer_money:transfer my money", "close_account:close the account"),
        ("check_balance:check my balance", "transfer_money:send the funds"),
        ("cancel_order:cancel the order", "refund:get a refund"),
        ("reset_password:reset my password", "update_address:change my address"),
        ("apply_coupon:apply the coupon", "upgrade_plan:upgrade my plan"),
        ("track_order:track my order", "contact_support:talk to agent"),
        ("billing:dispute the charge", "refund:process a refund"),
        ("check_balance:check balance", "close_account:close my account"),
    ])
}

fn two_intent_conditional_queries() -> Vec<CompoundQuery> {
    conditional(&[
        ("refund:get a refund", "contact_support:talk to support"),
        ("reset_password:reset my password", "contact_support:speak to agent"),
        ("cancel_order:cancel my order", "track_order:track the order"),
        ("transfer_money:wire transfer", "check_balance:check balance"),
        ("upgrade_plan:upgrade plan", "apply_coupon:use coupon"),
        ("billing:billing issue", "contact_support:customer support"),
    ])
}

fn two_intent_negation_queries() -> Vec<CompoundQuery> {
    negation(&[
        ("cancel_order:cancel my order", "refund:get a refund"),
        ("close_account:close my account", "transfer_money:transfer money"),
        ("reset_password:reset password", "close_account:close the account"),
        ("upgrade_plan:upgrade my plan", "close_account:shut down account"),
        ("track_order:track the package", "cancel_order:cancel the order"),
        ("billing:fix billing issue", "close_account:deactivate account"),
    ])
}

fn three_intent_queries() -> Vec<CompoundQuery> {
    triple(&[
        (
            "cancel_order:cancel my order",
            "refund:get a refund",
            "close_account:close my account",
        ),
        (
            "check_balance:check my balance",
            "transfer_money:transfer funds",
            "close_account:close the account",
        ),
        (
            "reset_password:reset my password",
            "update_address:update my address",
            "contact_support:talk to support",
        ),
        (
            "track_order:track my order",
            "cancel_order:cancel the purchase",
            "refund:process a refund",
        ),
        (
            "billing:billing issue",
            "refund:money back",
            "contact_support:speak to agent",
        ),
        (
            "apply_coupon:apply coupon",
            "upgrade_plan:upgrade my plan",
            "check_balance:check balance",
        ),
        (
            "transfer_money:send money",
            "check_balance:remaining balance",
            "close_account:terminate account",
        ),
        (
            "cancel_order:cancel order",
            "track_order:track package",
            "billing:wrong charge",
        ),
    ])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn multi_intent_benchmark_full() {
    let router = build_router();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        Multi-Intent Decomposition Benchmark                 ║");
    println!("║  12 intents, 10 seeds each, synthesized compound queries    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let threshold = 0.3;

    // Two-intent parallel
    let queries = two_intent_parallel_queries();
    let r = evaluate_multi(&router, &queries, threshold);
    print_results("2-Intent Parallel (and/also)", &r, true);

    // Two-intent sequential
    let queries = two_intent_sequential_queries();
    let r = evaluate_multi(&router, &queries, threshold);
    print_results("2-Intent Sequential (then/after)", &r, true);

    // Two-intent conditional
    let queries = two_intent_conditional_queries();
    let r = evaluate_multi(&router, &queries, threshold);
    print_results("2-Intent Conditional (or/otherwise)", &r, true);

    // Two-intent negation
    let queries = two_intent_negation_queries();
    let r = evaluate_multi(&router, &queries, threshold);
    print_results("2-Intent Negation (except/without)", &r, true);

    // Three-intent
    let queries = three_intent_queries();
    let r = evaluate_multi(&router, &queries, threshold);
    print_results("3-Intent Parallel (and...and...)", &r, true);

    // Aggregate
    let mut all_queries = Vec::new();
    all_queries.extend(two_intent_parallel_queries());
    all_queries.extend(two_intent_sequential_queries());
    all_queries.extend(two_intent_conditional_queries());
    all_queries.extend(two_intent_negation_queries());
    all_queries.extend(three_intent_queries());

    let total = evaluate_multi(&router, &all_queries, threshold);
    println!("  ════════════════════════════════════════════════");
    print_results(
        &format!("TOTAL ({} queries)", all_queries.len()),
        &total,
        false,
    );

    // Assert minimum quality bar
    let detection_rate =
        total.all_intents_detected as f32 / total.total as f32;
    assert!(
        detection_rate >= 0.50,
        "Multi-intent detection rate {:.1}% is below 50% minimum",
        detection_rate * 100.0
    );
}

/// Stress test: every pair of 12 intents combined
#[test]
fn multi_intent_exhaustive_pairs() {
    let router = build_router();
    let intent_queries: Vec<(&str, &str)> = vec![
        ("cancel_order", "cancel my order"),
        ("track_order", "track my package"),
        ("refund", "get a refund"),
        ("reset_password", "reset my password"),
        ("billing", "billing issue"),
        ("check_balance", "check my balance"),
        ("transfer_money", "transfer money"),
        ("close_account", "close my account"),
        ("update_address", "update my address"),
        ("contact_support", "talk to agent"),
        ("upgrade_plan", "upgrade my plan"),
        ("apply_coupon", "apply coupon"),
    ];

    let mut total = 0;
    let mut detected = 0;
    let mut ordered = 0;
    let mut failures = Vec::new();

    // Test every pair (i, j) where i != j
    for (i, &(id1, q1)) in intent_queries.iter().enumerate() {
        for (j, &(id2, q2)) in intent_queries.iter().enumerate() {
            if i == j {
                continue;
            }
            total += 1;
            let combined = format!("{} and also {}", q1, q2);
            let result = router.route_multi(&combined, 0.3);
            let detected_ids: HashSet<&str> =
                result.intents.iter().map(|r| r.id.as_str()).collect();

            let found_both = detected_ids.contains(id1) && detected_ids.contains(id2);
            if found_both {
                detected += 1;
                // Check order
                let pos1 = result.intents.iter().position(|r| r.id == id1);
                let pos2 = result.intents.iter().position(|r| r.id == id2);
                if let (Some(p1), Some(p2)) = (pos1, pos2) {
                    if p1 < p2 {
                        ordered += 1;
                    }
                }
            } else {
                let detected_list: Vec<&str> = result.intents.iter().map(|r| r.id.as_str()).collect();
                failures.push(format!(
                    "  {} + {} → {:?} (missing: {:?})",
                    id1,
                    id2,
                    detected_list,
                    if !detected_ids.contains(id1) { id1 } else { id2 }
                ));
            }
        }
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Exhaustive Pair Test: 12 intents × 12 = 132 pairs         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!(
        "  Both detected: {}/{} ({:.1}%)",
        detected,
        total,
        detected as f32 / total as f32 * 100.0
    );
    println!(
        "  Correct order: {}/{} ({:.1}%)",
        ordered,
        total,
        ordered as f32 / total as f32 * 100.0
    );
    if !failures.is_empty() {
        println!("  Failures ({}):", failures.len());
        for f in failures.iter().take(20) {
            println!("{}", f);
        }
        if failures.len() > 20 {
            println!("  ... and {} more", failures.len() - 20);
        }
    }
    println!();

    assert!(
        detected as f32 / total as f32 >= 0.60,
        "Exhaustive pair detection {:.1}% is below 60%",
        detected as f32 / total as f32 * 100.0
    );
}
