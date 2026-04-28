//! Basic embedded usage of the MicroResolve Engine.
//!
//! Run: cargo run --example rust_basic

use microresolve::{Engine, EngineConfig, IntentEdit, IntentType};

fn main() {
    let engine = Engine::new(EngineConfig::default()).expect("engine init");
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
    let matches = support.resolve("please cancel my order");
    println!("Top match: {:?}", matches.first());
}
