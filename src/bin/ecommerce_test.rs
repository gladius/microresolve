//! Test ecommerce-demo seed quality: old (persisted) vs new (diverse) seeds.
//!
//! Run: cargo run --release --bin ecommerce_test

use asv_router::Router;
use std::collections::{HashMap, HashSet};

#[derive(serde::Deserialize)]
struct Session { turns: Vec<Turn> }

#[derive(serde::Deserialize)]
struct Turn { message: String, intents: Vec<String> }

fn main() {
    let sessions: Vec<Session> = serde_json::from_str(
        &std::fs::read_to_string("tests/data/simulation_sessions.json").unwrap()
    ).unwrap();
    let turns: Vec<&Turn> = sessions.iter().flat_map(|s| s.turns.iter()).collect();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     Ecommerce Demo: Old vs New Seeds (50 turns)         ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // OLD: load from persisted ecommerce-demo.json
    println!("━━━ OLD seeds (ecommerce-demo.json) ━━━\n");
    let old_router = Router::load("data/ecommerce-demo.json").unwrap();
    evaluate(&old_router, &turns);

    // NEW: load from diverse seeds file
    println!("\n━━━ NEW seeds (vocabulary-diverse) ━━━\n");
    let new_seeds: HashMap<String, Vec<String>> = serde_json::from_str(
        &std::fs::read_to_string("data/ecommerce-demo-seeds.json").unwrap()
    ).unwrap();
    let mut new_router = Router::new();
    for (id, phrases) in &new_seeds {
        let refs: Vec<&str> = phrases.iter().map(|s| s.as_str()).collect();
        new_router.add_intent(&id, &refs);
    }
    evaluate(&new_router, &turns);
}

fn evaluate(router: &Router, turns: &[&Turn]) {
    let mut total_recall = 0.0f64;
    let mut total_precision = 0.0f64;
    let mut exact = 0;
    let mut per_hit: HashMap<String, usize> = HashMap::new();
    let mut per_total: HashMap<String, usize> = HashMap::new();

    for turn in turns {
        let output = router.route_multi(&turn.message, 0.3);
        let detected: HashSet<String> = output.intents.iter().map(|i| i.id.clone()).collect();
        let expected: HashSet<String> = turn.intents.iter().cloned().collect();
        let overlap = detected.intersection(&expected).count();

        total_recall += if expected.is_empty() { 1.0 } else { overlap as f64 / expected.len() as f64 };
        total_precision += if detected.is_empty() { 0.0 } else { overlap as f64 / detected.len() as f64 };
        if detected == expected { exact += 1; }

        for intent in &turn.intents {
            *per_total.entry(intent.clone()).or_insert(0) += 1;
            if detected.contains(intent) { *per_hit.entry(intent.clone()).or_insert(0) += 1; }
        }
    }

    let n = turns.len() as f64;
    let recall = total_recall / n * 100.0;
    let precision = total_precision / n * 100.0;
    let f1 = if recall + precision > 0.0 { 2.0 * recall * precision / (recall + precision) } else { 0.0 };

    println!("  Exact: {}/{} = {:.0}%  Recall: {:.1}%  Precision: {:.1}%  F1: {:.1}%",
        exact, turns.len(), exact as f64 / n * 100.0, recall, precision, f1);

    let mut stats: Vec<_> = per_total.iter().collect();
    stats.sort_by(|a, b| b.1.cmp(a.1));
    for (intent, total) in &stats {
        let hit = per_hit.get(intent.as_str()).copied().unwrap_or(0);
        let pct = hit as f32 / **total as f32 * 100.0;
        println!("    {:<25} {}/{} = {:.0}%", intent, hit, total, pct);
    }
}
