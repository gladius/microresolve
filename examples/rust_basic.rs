//! Basic embedded usage of MicroResolve.
//!
//! Run: cargo run --example rust_basic

use microresolve::{IntentEdit, IntentType, MicroResolve, MicroResolveConfig};

fn main() {
    let engine = MicroResolve::new(MicroResolveConfig::default()).expect("engine init");
    let support = engine.namespace("support");

    support
        .add_intent(
            "cancel_order",
            vec![
                "cancel my order".to_string(),
                "stop my subscription".to_string(),
            ],
        )
        .unwrap();
    support
        .add_intent(
            "track_order",
            vec![
                "where is my order".to_string(),
                "track my package".to_string(),
            ],
        )
        .unwrap();
    support
        .update_intent(
            "cancel_order",
            IntentEdit {
                intent_type: Some(IntentType::Action),
                ..Default::default()
            },
        )
        .ok();
    support
        .update_intent(
            "track_order",
            IntentEdit {
                intent_type: Some(IntentType::Action),
                ..Default::default()
            },
        )
        .ok();

    println!("Intents: {:?}", support.intent_ids());
    let result = support.resolve("please cancel my order");
    println!("Top match: {:?}", result.intents.first());
    println!("Disposition: {:?}", result.disposition);
}
