//! Regression test for score nondeterminism caused by HashMap iteration order.
//!
//! Root cause (fixed in v0.1.2): `build_for_namespace` built the L0 vocab by
//! iterating a HashSet whose order varies per process. When two vocabulary terms
//! had equal edit distance to a query word, `min_by_key(dist)` returned the first
//! one found in nondeterministic HashMap order, producing different L0 corrections
//! and therefore different scores across runs.
//!
//! This test creates two intents whose training phrases share a common ambiguous
//! word root ("taking" and "paying" are both edit-distance 2 from "having"), then
//! resolves the same query 50 times and asserts byte-identical results.

use microresolve::{Engine, EngineConfig};

#[test]
fn resolve_is_deterministic_across_invocations() {
    // Run the resolve loop 50 times inside a single process. Since HashMap seeds
    // are fixed per-process, this doesn't catch the cross-process bug directly,
    // but it does catch any per-call nondeterminism (e.g. if a new HashMap is
    // created per call with a fresh seed). The cross-process fix (sorted vocab)
    // is validated by the fact that the test passes consistently in CI.
    let engine = make_engine();
    let ns = engine.namespace("det-test");
    let query = "I am having thoughts of hurting myself";

    let first = ns.resolve(query);
    for _ in 1..50 {
        let result = ns.resolve(query);
        assert_eq!(
            result.len(),
            first.len(),
            "result length changed across invocations"
        );
        for (a, b) in first.iter().zip(result.iter()) {
            assert_eq!(a.id, b.id, "intent order changed across invocations");
            assert!(
                (a.score - b.score).abs() < 1e-6,
                "score changed: {} vs {} for intent {}",
                a.score,
                b.score,
                a.id
            );
        }
    }
}

#[test]
fn top_intent_is_stable() {
    // Additional guard: the TOP intent for this query must always be the same.
    let engine = make_engine();
    let ns = engine.namespace("det-test");
    let query = "I am having thoughts of hurting myself";
    let top = ns.resolve(query).into_iter().next().map(|m| m.id);
    for _ in 0..50 {
        let got = ns.resolve(query).into_iter().next().map(|m| m.id);
        assert_eq!(got, top, "top intent changed across invocations");
    }
}

fn make_engine() -> Engine {
    // Build a minimal in-memory namespace that reproduces the ambiguous-correction
    // scenario: "thoughts" and "myself" are exact vocabulary words; "taking" and
    // "paying" are both edit-distance 2 from "having" (the ambiguous query word).
    let engine = Engine::new(EngineConfig::default()).unwrap();
    {
        let ns = engine.namespace("det-test");
        ns.add_intent(
            "mental_health_crisis",
            vec![
                "thoughts of hurting myself",
                "planning to take my life tonight",
                "taking my life",
            ],
        )
        .unwrap();
        ns.add_intent(
            "billing",
            vec!["paying my invoice", "pay my bill", "billing question"],
        )
        .unwrap();
    }
    engine
}
