//! Test distributional similarity impact on routing accuracy.
//!
//! Compares Bitext benchmark accuracy with and without similarity expansion
//! built from the Bitext corpus (26,872 customer support queries).
//!
//! Run: cargo run --release --bin similarity_test

use asv_router::Router;
use std::collections::HashMap;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct Example {
    text: String,
    intents: Vec<String>,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   Distributional Similarity Impact Test (Bitext 27K)    ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let base = "tests/data/benchmarks";

    // Load seeds and test data
    let seeds: HashMap<String, Vec<String>> = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/bitext_seeds.json", base)).unwrap()
    ).unwrap();

    let all_examples: Vec<Example> = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/bitext_all.json", base)).unwrap()
    ).unwrap();

    // Use every 10th example as test set (same as dataset_benchmark)
    let test_examples: Vec<&Example> = all_examples.iter().step_by(10).collect();
    let corpus: Vec<String> = all_examples.iter().map(|e| e.text.clone()).collect();

    println!("  Intents: {} | Corpus: {} queries | Test: {} queries\n",
        seeds.len(), all_examples.len(), test_examples.len());

    // === BASELINE: No similarity ===
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  BASELINE: No similarity expansion");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut router_base = Router::new();
    for (intent_id, phrases) in &seeds {
        let refs: Vec<&str> = phrases.iter().map(|s| s.as_str()).collect();
        router_base.add_intent(intent_id, &refs);
    }

    let (base_top1, base_top3, base_per_intent) = evaluate(&router_base, &test_examples);

    // === SWEEP: Try different expansion discount factors ===
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PARAMETER SWEEP: expansion discount factor");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Build similarity once, reuse
    let mut router_sim = Router::new();
    for (intent_id, phrases) in &seeds {
        let refs: Vec<&str> = phrases.iter().map(|s| s.as_str()).collect();
        router_sim.add_intent(intent_id, &refs);
    }
    let t0 = Instant::now();
    router_sim.build_similarity(&corpus);
    let build_ms = t0.elapsed().as_millis();
    println!("  Similarity index built in {}ms\n", build_ms);

    // Print some sample expansions
    println!("  Sample term expansions:");
    for term in &["cancel", "refund", "track", "delivery", "account", "invoice", "wrong", "help", "address", "payment"] {
        let similar = router_sim.similar_terms(term);
        if !similar.is_empty() {
            let top3: Vec<String> = similar.iter().take(5).map(|(t, s)| format!("{}({:.2})", t, s)).collect();
            println!("    {} → {}", term, top3.join(", "));
        } else {
            println!("    {} → (no similar terms)", term);
        }
    }
    println!();

    // Test different discount factors
    for discount in &[0.10, 0.15, 0.20, 0.25, 0.30, 0.40] {
        router_sim.set_expansion_discount(*discount);
        let (top1, top3, _) = evaluate_quiet(&router_sim, &test_examples);
        let d1 = top1 - base_top1;
        let d3 = top3 - base_top3;
        println!("  discount={:.2}: Top-1 {:.1}% ({:+.1}%), Top-3 {:.1}% ({:+.1}%)",
            discount, top1, d1, top3, d3);
    }

    // Use best discount for detailed comparison
    router_sim.set_expansion_discount(0.3); // default
    let (sim_top1, sim_top3, sim_per_intent) = evaluate(&router_sim, &test_examples);

    // === COMPARISON ===
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  COMPARISON (discount=0.30)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let top1_delta = sim_top1 - base_top1;
    let top3_delta = sim_top3 - base_top3;
    println!("  Top-1: {:.1}% → {:.1}% ({:+.1}%)", base_top1, sim_top1, top1_delta);
    println!("  Top-3: {:.1}% → {:.1}% ({:+.1}%)", base_top3, sim_top3, top3_delta);

    // Per-intent deltas
    println!("\n  Per-intent changes:");
    let mut deltas: Vec<(String, f32, f32, f32)> = Vec::new();
    for (intent, &base_acc) in &base_per_intent {
        let sim_acc = sim_per_intent.get(intent).copied().unwrap_or(0.0);
        let delta = sim_acc - base_acc;
        if delta.abs() > 0.1 {
            deltas.push((intent.clone(), base_acc, sim_acc, delta));
        }
    }
    deltas.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    if deltas.is_empty() {
        println!("    (no significant changes)");
    } else {
        for (intent, base, sim, delta) in &deltas {
            let arrow = if *delta > 0.0 { "▲" } else { "▼" };
            println!("    {:<35} {:.0}% → {:.0}% {} {:+.0}%", intent, base, sim, arrow, delta);
        }
    }

    // Investigate: show specific examples where similarity helped or hurt
    println!("\n  Examples where similarity CHANGED the top-1 result:");
    let mut helped = 0;
    let mut hurt = 0;
    let mut helped_examples: Vec<(String, String, String, String)> = Vec::new();
    let mut hurt_examples: Vec<(String, String, String, String)> = Vec::new();

    for ex in &test_examples {
        let base_result = router_base.route(&ex.text);
        let sim_result = router_sim.route(&ex.text);

        let base_top = base_result.first().map(|r| r.id.as_str()).unwrap_or("(none)");
        let sim_top = sim_result.first().map(|r| r.id.as_str()).unwrap_or("(none)");
        let expected = &ex.intents[0];

        if base_top != sim_top {
            if sim_top == expected && base_top != expected {
                helped += 1;
                if helped_examples.len() < 5 {
                    helped_examples.push((
                        ex.text.clone(),
                        expected.clone(),
                        base_top.to_string(),
                        sim_top.to_string(),
                    ));
                }
            } else if base_top == expected && sim_top != expected {
                hurt += 1;
                if hurt_examples.len() < 5 {
                    hurt_examples.push((
                        ex.text.clone(),
                        expected.clone(),
                        base_top.to_string(),
                        sim_top.to_string(),
                    ));
                }
            }
        }
    }

    println!("    Helped (wrong→right): {}", helped);
    for (query, expected, base, sim) in &helped_examples {
        println!("      \"{}\"", &query[..query.len().min(60)]);
        println!("        expected={}, was={}, now={}", expected, base, sim);
    }

    println!("    Hurt (right→wrong): {}", hurt);
    for (query, expected, base, sim) in &hurt_examples {
        println!("      \"{}\"", &query[..query.len().min(60)]);
        println!("        expected={}, was={}, now={}", expected, base, sim);
    }
}

fn evaluate_quiet(router: &Router, examples: &[&Example]) -> (f32, f32, HashMap<String, f32>) {
    let mut correct = 0;
    let mut top3_correct = 0;
    let total = examples.len();
    let mut intent_stats: HashMap<String, (usize, usize)> = HashMap::new();

    for ex in examples {
        let result = router.route(&ex.text);
        if !result.is_empty() && result[0].id == ex.intents[0] { correct += 1; }
        if result.iter().take(3).any(|r| r.id == ex.intents[0]) { top3_correct += 1; }
        let entry = intent_stats.entry(ex.intents[0].clone()).or_insert((0, 0));
        entry.1 += 1;
        if !result.is_empty() && result[0].id == ex.intents[0] { entry.0 += 1; }
    }

    let top1_acc = correct as f32 / total as f32 * 100.0;
    let top3_acc = top3_correct as f32 / total as f32 * 100.0;
    let per_intent: HashMap<String, f32> = intent_stats.into_iter()
        .map(|(k, (c, t))| (k, if t > 0 { c as f32 / t as f32 * 100.0 } else { 0.0 }))
        .collect();
    (top1_acc, top3_acc, per_intent)
}

fn evaluate(router: &Router, examples: &[&Example]) -> (f32, f32, HashMap<String, f32>) {
    let t0 = Instant::now();
    let mut correct = 0;
    let mut top3_correct = 0;
    let total = examples.len();

    let mut intent_stats: HashMap<String, (usize, usize)> = HashMap::new();

    for ex in examples {
        let result = router.route(&ex.text);

        if !result.is_empty() && result[0].id == ex.intents[0] {
            correct += 1;
        }
        if result.iter().take(3).any(|r| r.id == ex.intents[0]) {
            top3_correct += 1;
        }

        let entry = intent_stats.entry(ex.intents[0].clone()).or_insert((0, 0));
        entry.1 += 1;
        if !result.is_empty() && result[0].id == ex.intents[0] {
            entry.0 += 1;
        }
    }

    let elapsed_us = t0.elapsed().as_micros();
    let top1_acc = correct as f32 / total as f32 * 100.0;
    let top3_acc = top3_correct as f32 / total as f32 * 100.0;

    println!("  Top-1 accuracy:  {}/{} = {:.1}%", correct, total, top1_acc);
    println!("  Top-3 accuracy:  {}/{} = {:.1}%", top3_correct, total, top3_acc);
    println!("  Avg latency:     {:.1} µs/query", elapsed_us as f64 / total as f64);

    // Per-intent
    println!("\n  Per-intent accuracy:");
    let mut stats: Vec<_> = intent_stats.iter().collect();
    stats.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    for (intent, (correct, total)) in &stats {
        let pct = if *total > 0 { *correct as f32 / *total as f32 * 100.0 } else { 0.0 };
        println!("    {:<35} {}/{} = {:.0}%", intent, correct, total, pct);
    }

    let per_intent: HashMap<String, f32> = intent_stats.into_iter()
        .map(|(k, (c, t))| (k, if t > 0 { c as f32 / t as f32 * 100.0 } else { 0.0 }))
        .collect();

    (top1_acc, top3_acc, per_intent)
}
