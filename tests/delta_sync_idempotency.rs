//! Idempotency tests for every Op variant.
//!
//! Invariant: applying an op twice produces the same state as applying it once.

use microresolve::{MicroResolve, MicroResolveConfig};

fn engine() -> MicroResolve {
    MicroResolve::new(MicroResolveConfig::default()).unwrap()
}

/// Apply a WeightUpdates op via apply_weight_updates (the typed method).
fn apply_weight_updates(e: &MicroResolve, ns: &str, changes: &[(String, String, f32)]) {
    e.namespace(ns).apply_weight_updates(changes);
}

#[test]
fn idempotent_intent_added() {
    let e = engine();
    let h = e.namespace("ns");

    // First application
    h.add_intent("greet", vec!["hello world".to_string()])
        .unwrap();
    let count_after_1 = h.intent_count();
    let training_after_1 = h.training("greet").unwrap();

    // Second application (same intent, same phrase — should deduplicate)
    h.add_intent("greet", vec!["hello world".to_string()])
        .unwrap();
    let count_after_2 = h.intent_count();
    let training_after_2 = h.training("greet").unwrap();

    assert_eq!(
        count_after_1, count_after_2,
        "intent count should not change on re-add"
    );
    assert_eq!(
        training_after_1.len(),
        training_after_2.len(),
        "phrase count should not grow"
    );
}

#[test]
fn idempotent_intent_removed() {
    let e = engine();
    let h = e.namespace("ns");
    h.add_intent("bye", vec!["goodbye".to_string()]).unwrap();
    assert_eq!(h.intent_count(), 1);

    h.remove_intent("bye");
    assert_eq!(h.intent_count(), 0);

    // Second remove — no panic, still 0.
    h.remove_intent("bye");
    assert_eq!(h.intent_count(), 0);
}

#[test]
fn idempotent_phrase_added() {
    let e = engine();
    let h = e.namespace("ns");
    h.add_intent("greet", vec!["hi".to_string()]).unwrap();

    h.add_phrase("greet", "hello there", "en");
    let phrases_1 = h.training("greet").unwrap();

    h.add_phrase("greet", "hello there", "en");
    let phrases_2 = h.training("greet").unwrap();

    assert_eq!(
        phrases_1.len(),
        phrases_2.len(),
        "duplicate phrase must not be added twice"
    );
}

#[test]
fn idempotent_phrase_removed() {
    let e = engine();
    let h = e.namespace("ns");
    h.add_intent("greet", vec!["hi".to_string(), "hello".to_string()])
        .unwrap();

    let removed_1 = h.remove_phrase("greet", "hi");
    assert!(removed_1);

    let removed_2 = h.remove_phrase("greet", "hi");
    assert!(
        !removed_2,
        "second remove should return false (already gone)"
    );

    let phrases = h.training("greet").unwrap();
    assert_eq!(phrases.len(), 1, "only 'hello' remains");
}

#[test]
fn idempotent_weight_updates() {
    let e = engine();
    let h = e.namespace("ns");
    h.add_intent("greet", vec!["hello world".to_string()])
        .unwrap();

    let changes = vec![("hello".to_string(), "greet".to_string(), 0.8f32)];

    apply_weight_updates(&e, "ns", &changes);
    let w1 = e
        .namespace("ns")
        .with_resolver(|r| r.index().get_weight("hello", "greet"));

    apply_weight_updates(&e, "ns", &changes);
    let w2 = e
        .namespace("ns")
        .with_resolver(|r| r.index().get_weight("hello", "greet"));

    assert_eq!(w1, w2, "WeightUpdates applied twice must yield same weight");
    assert_eq!(w1, Some(0.8f32));
}

#[test]
fn idempotent_intent_metadata_updated() {
    let e = engine();
    let h = e.namespace("ns");
    h.add_intent("greet", vec!["hi".to_string()]).unwrap();

    let edit = microresolve::IntentEdit {
        description: Some("A greeting intent".to_string()),
        ..Default::default()
    };
    h.update_intent("greet", edit.clone()).unwrap();
    let info1 = h.intent("greet").unwrap();

    h.update_intent("greet", edit).unwrap();
    let info2 = h.intent("greet").unwrap();

    assert_eq!(info1.description, info2.description);
    assert_eq!(info2.description, "A greeting intent");
}

#[test]
fn idempotent_namespace_metadata_updated() {
    let e = engine();
    let h = e.namespace("ns");

    let edit = microresolve::NamespaceEdit {
        name: Some("My Namespace".to_string()),
        ..Default::default()
    };
    h.update_namespace(edit.clone()).unwrap();
    let info1 = h.namespace_info();

    h.update_namespace(edit).unwrap();
    let info2 = h.namespace_info();

    assert_eq!(info1.name, info2.name);
    assert_eq!(info2.name, "My Namespace");
}

#[test]
fn idempotent_domain_description_set() {
    let e = engine();
    let h = e.namespace("ns");

    h.set_domain_description("billing", "Billing domain");
    let d1 = h.domain_description("billing");

    h.set_domain_description("billing", "Billing domain");
    let d2 = h.domain_description("billing");

    assert_eq!(d1, d2);
    assert_eq!(d2, Some("Billing domain".to_string()));
}

#[test]
fn idempotent_domain_description_removed() {
    let e = engine();
    let h = e.namespace("ns");

    h.set_domain_description("billing", "Billing domain");
    h.remove_domain_description("billing");
    let d1 = h.domain_description("billing");

    // Remove again — should be no-op.
    h.remove_domain_description("billing");
    let d2 = h.domain_description("billing");

    assert_eq!(d1, None);
    assert_eq!(d2, None);
}
