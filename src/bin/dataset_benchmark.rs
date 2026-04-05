//! Benchmark ASV Router against real NLU datasets.
//!
//! Datasets:
//! 1. MixSNIPS — 7 intents, 2199 test (1749 multi-intent), voice assistant domain
//! 2. MixATIS — 18 intents, 828 test (685 multi-intent), airline travel domain
//! 3. Bitext — 27 intents, 26872 examples, customer support domain
//! 4. SGD — 34 intents, 4201 dialogues, multi-domain workflow sequences
//!
//! Run: cargo run --release --bin dataset_benchmark

use asv_router::Router;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(serde::Deserialize)]
struct Example {
    text: String,
    intents: Vec<String>,
}

#[derive(serde::Deserialize)]
struct Dialogue {
    intent_sequence: Vec<String>,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     ASV Router — Multi-Dataset Benchmark Suite          ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let base = "tests/data/benchmarks";

    // === 1. MixSNIPS ===
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  BENCHMARK 1: MixSNIPS (Voice Assistant, 7 intents)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    run_multi_intent_benchmark(
        &format!("{}/mixsnips_seeds.json", base),
        &format!("{}/mixsnips_test.json", base),
        "MixSNIPS",
        0.2,
    );

    // === 2. MixATIS ===
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  BENCHMARK 2: MixATIS (Airline Travel, 18 intents)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    run_multi_intent_benchmark(
        &format!("{}/mixatis_seeds.json", base),
        &format!("{}/mixatis_test.json", base),
        "MixATIS",
        0.2,
    );

    // === 3. Bitext ===
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  BENCHMARK 3: Bitext (Customer Support, 27 intents)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    run_single_intent_benchmark(
        &format!("{}/bitext_seeds.json", base),
        &format!("{}/bitext_all.json", base),
        "Bitext",
    );

    // === 4. SGD Workflow Discovery ===
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  BENCHMARK 4: SGD (Workflow Discovery, 34 intents)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    run_workflow_benchmark(
        &format!("{}/sgd_dialogues.json", base),
        "SGD",
    );
}

fn load_seeds(path: &str) -> HashMap<String, Vec<String>> {
    let data = std::fs::read_to_string(path).expect("failed to read seeds");
    serde_json::from_str(&data).expect("failed to parse seeds")
}

fn load_examples(path: &str) -> Vec<Example> {
    let data = std::fs::read_to_string(path).expect("failed to read examples");
    serde_json::from_str(&data).expect("failed to parse examples")
}

fn load_dialogues(path: &str) -> Vec<Dialogue> {
    let data = std::fs::read_to_string(path).expect("failed to read dialogues");
    serde_json::from_str(&data).expect("failed to parse dialogues")
}

fn build_router(seeds: &HashMap<String, Vec<String>>) -> Router {
    let mut router = Router::new();
    for (intent_id, phrases) in seeds {
        let refs: Vec<&str> = phrases.iter().map(|s| s.as_str()).collect();
        router.add_intent(intent_id, &refs);
    }
    router
}

fn run_multi_intent_benchmark(seeds_path: &str, test_path: &str, _name: &str, threshold: f32) {
    let seeds = load_seeds(seeds_path);
    let examples = load_examples(test_path);
    let router = build_router(&seeds);

    println!("  Intents: {} | Test examples: {} | Threshold: {}", seeds.len(), examples.len(), threshold);

    let multi_examples: Vec<&Example> = examples.iter().filter(|e| e.intents.len() > 1).collect();
    let single_examples: Vec<&Example> = examples.iter().filter(|e| e.intents.len() == 1).collect();

    println!("  Single-intent: {} | Multi-intent: {}\n", single_examples.len(), multi_examples.len());

    // --- Single-intent accuracy (top-1) ---
    let t0 = Instant::now();
    let mut single_correct = 0;
    let mut single_total = 0;
    for ex in &single_examples {
        let result = router.route(&ex.text);
        single_total += 1;
        if !result.is_empty() && result[0].id == ex.intents[0] {
            single_correct += 1;
        }
    }
    let single_us = t0.elapsed().as_micros();
    let single_acc = if single_total > 0 { single_correct as f32 / single_total as f32 * 100.0 } else { 0.0 };
    println!("  Single-intent accuracy (top-1): {}/{} = {:.1}%", single_correct, single_total, single_acc);
    if single_total > 0 {
        println!("  Avg latency: {:.1} µs/query", single_us as f64 / single_total as f64);
    }

    // --- Multi-intent decomposition ---
    let t0 = Instant::now();
    let mut exact_match = 0;   // all intents detected exactly
    let mut partial_match = 0; // at least one correct intent detected
    let mut intent_recall_sum = 0.0f64; // per-example recall
    let mut intent_precision_sum = 0.0f64;
    let mut total_multi = 0;

    for ex in &multi_examples {
        let output = router.route_multi(&ex.text, threshold);
        let detected: HashSet<String> = output.intents.iter().map(|i| i.id.clone()).collect();
        let expected: HashSet<String> = ex.intents.iter().cloned().collect();

        total_multi += 1;

        // Exact match: detected set == expected set
        if detected == expected {
            exact_match += 1;
        }

        // Partial match: any overlap
        let overlap = detected.intersection(&expected).count();
        if overlap > 0 {
            partial_match += 1;
        }

        // Per-example recall and precision
        let recall = if expected.is_empty() { 0.0 } else { overlap as f64 / expected.len() as f64 };
        let precision = if detected.is_empty() { 0.0 } else { overlap as f64 / detected.len() as f64 };
        intent_recall_sum += recall;
        intent_precision_sum += precision;
    }
    let multi_us = t0.elapsed().as_micros();

    let exact_pct = if total_multi > 0 { exact_match as f32 / total_multi as f32 * 100.0 } else { 0.0 };
    let partial_pct = if total_multi > 0 { partial_match as f32 / total_multi as f32 * 100.0 } else { 0.0 };
    let avg_recall = if total_multi > 0 { intent_recall_sum / total_multi as f64 * 100.0 } else { 0.0 };
    let avg_precision = if total_multi > 0 { intent_precision_sum / total_multi as f64 * 100.0 } else { 0.0 };
    let f1 = if avg_recall + avg_precision > 0.0 { 2.0 * avg_recall * avg_precision / (avg_recall + avg_precision) } else { 0.0 };

    println!("\n  Multi-intent decomposition ({} examples):", total_multi);
    println!("    Exact match:     {}/{} = {:.1}%", exact_match, total_multi, exact_pct);
    println!("    Partial match:   {}/{} = {:.1}%", partial_match, total_multi, partial_pct);
    println!("    Avg recall:      {:.1}%", avg_recall);
    println!("    Avg precision:   {:.1}%", avg_precision);
    println!("    F1:              {:.1}%", f1);
    if total_multi > 0 {
        println!("    Avg latency:     {:.1} µs/query", multi_us as f64 / total_multi as f64);
    }

    // --- Per-intent breakdown for multi ---
    println!("\n  Per-intent recall (multi-intent):");
    let mut intent_stats: HashMap<String, (usize, usize)> = HashMap::new(); // (correct, total)
    for ex in &multi_examples {
        let output = router.route_multi(&ex.text, threshold);
        let detected: HashSet<String> = output.intents.iter().map(|i| i.id.clone()).collect();
        for expected_intent in &ex.intents {
            let entry = intent_stats.entry(expected_intent.clone()).or_insert((0, 0));
            entry.1 += 1;
            if detected.contains(expected_intent) {
                entry.0 += 1;
            }
        }
    }
    let mut stats: Vec<_> = intent_stats.into_iter().collect();
    stats.sort_by(|a, b| b.1.1.cmp(&a.1.1)); // by total desc
    for (intent, (correct, total)) in &stats {
        let pct = if *total > 0 { *correct as f32 / *total as f32 * 100.0 } else { 0.0 };
        println!("    {:<30} {}/{} = {:.0}%", intent, correct, total, pct);
    }
}

fn run_single_intent_benchmark(seeds_path: &str, test_path: &str, _name: &str) {
    let seeds = load_seeds(seeds_path);
    let all_examples = load_examples(test_path);
    let router = build_router(&seeds);

    println!("  Intents: {} | Total examples: {}", seeds.len(), all_examples.len());

    // Use a sample for testing (every 10th example, to keep it fast)
    let test_examples: Vec<&Example> = all_examples.iter().step_by(10).collect();
    println!("  Test sample: {} (every 10th)\n", test_examples.len());

    let t0 = Instant::now();
    let mut correct = 0;
    let mut top3_correct = 0;
    let mut total = 0;

    for ex in &test_examples {
        let result = router.route(&ex.text);
        total += 1;

        if !result.is_empty() && result[0].id == ex.intents[0] {
            correct += 1;
        }
        if result.iter().take(3).any(|r| r.id == ex.intents[0]) {
            top3_correct += 1;
        }
    }
    let elapsed_us = t0.elapsed().as_micros();

    let top1_acc = correct as f32 / total as f32 * 100.0;
    let top3_acc = top3_correct as f32 / total as f32 * 100.0;

    println!("  Top-1 accuracy:  {}/{} = {:.1}%", correct, total, top1_acc);
    println!("  Top-3 accuracy:  {}/{} = {:.1}%", top3_correct, total, top3_acc);
    println!("  Avg latency:     {:.1} µs/query", elapsed_us as f64 / total as f64);

    // Per-intent breakdown
    println!("\n  Per-intent accuracy (top-1, sample):");
    let mut intent_stats: HashMap<String, (usize, usize)> = HashMap::new();
    for ex in &test_examples {
        let result = router.route(&ex.text);
        let entry = intent_stats.entry(ex.intents[0].clone()).or_insert((0, 0));
        entry.1 += 1;
        if !result.is_empty() && result[0].id == ex.intents[0] {
            entry.0 += 1;
        }
    }
    let mut stats: Vec<_> = intent_stats.into_iter().collect();
    stats.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    for (intent, (correct, total)) in &stats {
        let pct = if *total > 0 { *correct as f32 / *total as f32 * 100.0 } else { 0.0 };
        println!("    {:<35} {}/{} = {:.0}%", intent, correct, total, pct);
    }
}

fn run_workflow_benchmark(dialogues_path: &str, _name: &str) {
    let dialogues = load_dialogues(dialogues_path);

    // Collect all intents from dialogue sequences
    let mut all_intents: HashSet<String> = HashSet::new();
    for d in &dialogues {
        for intent in &d.intent_sequence {
            all_intents.insert(intent.clone());
        }
    }

    println!("  Dialogues: {} | Unique intents: {}", dialogues.len(), all_intents.len());
    let multi = dialogues.iter().filter(|d| d.intent_sequence.len() >= 2).count();
    println!("  Multi-intent dialogues: {} ({:.0}%)\n", multi, multi as f32 / dialogues.len() as f32 * 100.0);

    // Build router and feed sequences
    let mut router = Router::new();

    // Create intents with placeholder phrases (we only need the sequence tracking)
    for intent in &all_intents {
        router.add_intent(intent, &[intent.as_str()]);
    }

    // Feed all dialogue sequences
    let t0 = Instant::now();
    for d in &dialogues {
        if d.intent_sequence.len() >= 2 {
            let refs: Vec<&str> = d.intent_sequence.iter().map(|s| s.as_str()).collect();
            router.record_intent_sequence(&refs);
        }
    }
    let feed_us = t0.elapsed().as_micros();
    println!("  Sequence recording: {} µs ({:.1} µs/dialogue)\n", feed_us, feed_us as f64 / dialogues.len() as f64);

    // --- Ground truth: count actual transitions ---
    let mut ground_truth: HashMap<(String, String), u32> = HashMap::new();
    for d in &dialogues {
        for i in 0..d.intent_sequence.len().saturating_sub(1) {
            let key = (d.intent_sequence[i].clone(), d.intent_sequence[i + 1].clone());
            *ground_truth.entry(key).or_insert(0) += 1;
        }
    }

    // --- Temporal ordering results ---
    println!("  --- Temporal Ordering (top 15) ---\n");
    let order = router.get_temporal_order();
    let mut tp = 0;  // true positives: ASV found a transition that exists in ground truth
    let mut total_found = 0;
    for (first, second, prob, count) in &order {
        if *count >= 5 {
            let gt_count = ground_truth.get(&(first.to_string(), second.to_string())).copied().unwrap_or(0);
            let match_str = if gt_count > 0 { format!("GT={}", gt_count) } else { "NOT IN GT".to_string() };
            if total_found < 15 {
                println!("    {} → {}  (P={:.0}%, n={}, {})", first, second, prob * 100.0, count, match_str);
            }
            total_found += 1;
            if gt_count > 0 { tp += 1; }
        }
    }
    let temporal_precision = if total_found > 0 { tp as f32 / total_found as f32 * 100.0 } else { 0.0 };
    println!("\n    Temporal ordering precision: {}/{} = {:.1}% (matches ground truth transitions)", tp, total_found, temporal_precision);

    // --- Workflow clusters ---
    println!("\n  --- Discovered Workflow Clusters (min 10 co-occurrences) ---\n");
    let workflows = router.discover_workflows(10);
    println!("    Found {} clusters", workflows.len());
    for (i, cluster) in workflows.iter().enumerate().take(5) {
        let ids: Vec<&str> = cluster.iter().map(|w| w.id.as_str()).collect();
        println!("    Cluster {} ({} intents): {:?}", i + 1, cluster.len(), ids);
    }

    // --- Escalation patterns ---
    println!("\n  --- Top Escalation Patterns (min 20 occurrences) ---\n");
    let patterns = router.detect_escalation_patterns(20);
    let mut shown = 0;
    for p in &patterns {
        if p.sequence.len() >= 2 && shown < 15 {
            let gt_match = if p.sequence.len() == 2 {
                ground_truth.get(&(p.sequence[0].clone(), p.sequence[1].clone()))
                    .map(|c| format!("GT={}", c))
                    .unwrap_or("no direct GT".to_string())
            } else {
                "compound".to_string()
            };
            println!("    {} ({}x, {:.1}% of traffic, {})",
                p.sequence.join(" → "), p.occurrences, p.frequency * 100.0, gt_match);
            shown += 1;
        }
    }

    // --- Proactive suggestions validation ---
    println!("\n  --- Suggestion Quality ---\n");
    // For each intent, check if suggestions match actual co-occurrence from ground truth
    let mut suggestion_hits = 0;
    let mut suggestion_total = 0;
    for intent in all_intents.iter().take(20) {
        let suggestions = router.suggest_intents(&[intent.as_str()], 5, 0.1);
        for s in &suggestions {
            suggestion_total += 1;
            // Check if this suggestion reflects a real ground truth transition
            let has_gt = ground_truth.contains_key(&(intent.clone(), s.id.clone()))
                || ground_truth.contains_key(&(s.id.clone(), intent.clone()));
            if has_gt {
                suggestion_hits += 1;
            }
        }
    }
    let suggestion_precision = if suggestion_total > 0 { suggestion_hits as f32 / suggestion_total as f32 * 100.0 } else { 0.0 };
    println!("    Suggestions that match ground truth transitions: {}/{} = {:.1}%",
        suggestion_hits, suggestion_total, suggestion_precision);
}
