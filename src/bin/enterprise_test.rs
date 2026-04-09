//! Enterprise-grade end-to-end seed guard test.
//!
//! Simulates a real deployment lifecycle:
//! 1. Create 27 intents from Bitext seeds (real customer support domain)
//! 2. Route real queries, collect failures
//! 3. Attempt fixes through seed guard — some accepted, some blocked
//! 4. Verify: accuracy improves, no existing intent degrades
//! 5. Deliberately try dangerous seeds — verify guard blocks them
//!
//! Run: cargo run --release --bin enterprise_test

use asv_router::Router;
use std::collections::{HashMap, HashSet};

#[derive(serde::Deserialize)]
struct Example {
    text: String,
    intents: Vec<String>,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Enterprise E2E Test: 27 Intents, Seed Guard Lifecycle  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/benchmarks");
    let seeds: HashMap<String, Vec<String>> = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/bitext_seeds.json", base)).unwrap()
    ).unwrap();
    let all_examples: Vec<Example> = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/bitext_all.json", base)).unwrap()
    ).unwrap();

    // Use every 10th as test set (same as benchmarks)
    let test_examples: Vec<&Example> = all_examples.iter().step_by(10).collect();

    println!("  Intents: {} | Seeds/intent: ~{} | Test queries: {}\n",
        seeds.len(),
        seeds.values().map(|v| v.len()).sum::<usize>() / seeds.len(),
        test_examples.len());

    // === Phase 1: Create intents ===
    println!("━━━ Phase 1: Create all intents ━━━\n");
    let mut router = Router::new();
    for (intent_id, phrases) in &seeds {
        let refs: Vec<&str> = phrases.iter().map(|s| s.as_str()).collect();
        router.add_intent(intent_id, &refs);
    }
    let (baseline_top1, baseline_per_intent) = evaluate(&router, &test_examples);
    println!("  Baseline top-1: {:.1}%\n", baseline_top1);

    // === Phase 2: Identify failures ===
    println!("━━━ Phase 2: Identify failures ━━━\n");
    let mut failures: Vec<(&Example, String)> = Vec::new(); // (example, expected_intent)
    for ex in &test_examples {
        let result = router.route(&ex.text);
        if result.is_empty() || result[0].id != ex.intents[0] {
            failures.push((ex, ex.intents[0].clone()));
        }
    }
    println!("  Failures: {} out of {} ({:.1}% fail rate)\n",
        failures.len(), test_examples.len(),
        failures.len() as f64 / test_examples.len() as f64 * 100.0);

    // Show some failures
    println!("  Sample failures:");
    for (ex, expected) in failures.iter().take(5) {
        let result = router.route(&ex.text);
        let got = result.first().map(|r| r.id.as_str()).unwrap_or("(none)");
        println!("    \"{}\"", &ex.text[..ex.text.len().min(60)]);
        println!("      expected={}, got={}", expected, got);
    }
    println!();

    // === Phase 3: Attempt fixes through seed guard ===
    println!("━━━ Phase 3: Apply fixes through seed guard ━━━\n");

    // Generate fix seeds from failure patterns (simulating what an LLM or human would suggest)
    let fix_seeds = generate_fix_seeds(&failures);
    let mut added = 0;
    let mut blocked_collision = 0;
    let mut blocked_redundant = 0;
    let mut blocked_empty = 0;

    for (intent_id, seed) in &fix_seeds {
        let result = router.add_seed_checked(intent_id, seed, "en");
        if result.added {
            added += 1;
        } else if !result.conflicts.is_empty() {
            blocked_collision += 1;
            if blocked_collision <= 5 {
                let conflict = &result.conflicts[0];
                println!("  BLOCKED (collision): \"{}\" → {}", seed, intent_id);
                println!("    '{}' conflicts with {} (severity {:.0}%)",
                    conflict.term, conflict.competing_intent, conflict.severity * 100.0);
            }
        } else if result.redundant {
            blocked_redundant += 1;
        } else {
            blocked_empty += 1;
        }
    }

    println!("\n  Fix results:");
    println!("    Added: {}", added);
    println!("    Blocked (collision): {}", blocked_collision);
    println!("    Blocked (redundant): {}", blocked_redundant);
    println!("    Blocked (empty): {}", blocked_empty);

    let (after_fix_top1, after_fix_per_intent) = evaluate(&router, &test_examples);
    println!("\n  After fixes top-1: {:.1}% (was {:.1}%, delta {:+.1}%)\n",
        after_fix_top1, baseline_top1, after_fix_top1 - baseline_top1);

    // === Phase 4: Check no intent degraded ===
    println!("━━━ Phase 4: Regression check ━━━\n");
    let mut degraded = 0;
    let mut improved = 0;
    let mut unchanged = 0;

    for (intent, &before) in &baseline_per_intent {
        let after = after_fix_per_intent.get(intent).copied().unwrap_or(0.0);
        let delta = after - before;
        if delta < -5.0 {
            degraded += 1;
            println!("  DEGRADED: {:<35} {:.0}% → {:.0}% ({:+.0}%)", intent, before, after, delta);
        } else if delta > 5.0 {
            improved += 1;
            println!("  IMPROVED: {:<35} {:.0}% → {:.0}% ({:+.0}%)", intent, before, after, delta);
        } else {
            unchanged += 1;
        }
    }
    println!("\n  Improved: {}, Degraded: {}, Unchanged: {}", improved, degraded, unchanged);

    // === Phase 5: Deliberately try dangerous seeds ===
    println!("\n━━━ Phase 5: Adversarial seed test ━━━\n");

    let dangerous_seeds: Vec<(&str, &str, &str)> = vec![
        // (intent, seed, expected_block_reason)
        ("cancel_order", "check my invoice", "invoice exclusive to check_invoice/get_invoice"),
        ("get_refund", "track my delivery", "delivery exclusive to delivery_period/delivery_options"),
        ("cancel_order", "delete my account", "delete/account exclusive to delete_account"),
        ("check_invoice", "refund policy details", "refund exclusive to get_refund/check_refund_policy"),
        ("track_order", "newsletter subscription", "newsletter exclusive to newsletter_subscription"),
        ("payment_issue", "recover my password", "password/recover exclusive to recover_password"),
        ("place_order", "switch my account", "switch exclusive to switch_account"),
        ("edit_account", "cancel my order", "cancel exclusive to cancel_order"),
    ];

    let mut correctly_blocked = 0;
    let mut incorrectly_allowed = 0;

    for (intent, seed, reason) in &dangerous_seeds {
        let result = router.add_seed_checked(intent, seed, "en");
        if !result.added {
            correctly_blocked += 1;
            println!("  BLOCKED (correct): \"{}\" → {}",  seed, intent);
            if let Some(ref w) = result.warning {
                println!("    Reason: {}", w);
            }
        } else {
            incorrectly_allowed += 1;
            println!("  ALLOWED (wrong!): \"{}\" → {} — should have blocked: {}",
                seed, intent, reason);
            // Remove it to not pollute further tests
            router.remove_seed(intent, seed);
        }
    }

    println!("\n  Dangerous seeds: {}/{} correctly blocked",
        correctly_blocked, dangerous_seeds.len());
    if incorrectly_allowed > 0 {
        println!("  WARNING: {} dangerous seeds were incorrectly allowed!", incorrectly_allowed);
    }

    // === Phase 6: Verify final state ===
    println!("\n━━━ Phase 6: Final accuracy ━━━\n");
    let (final_top1, _) = evaluate(&router, &test_examples);
    println!("  Baseline:    {:.1}%", baseline_top1);
    println!("  After fixes: {:.1}%", after_fix_top1);
    println!("  Final:       {:.1}%", final_top1);
    println!("  Total delta: {:+.1}%", final_top1 - baseline_top1);

    if degraded == 0 && final_top1 >= baseline_top1 {
        println!("\n  PASS: No regressions, accuracy maintained or improved.");
    } else if degraded > 0 {
        println!("\n  WARN: {} intents degraded. Seed guard may need tuning.", degraded);
    }
}

/// Generate fix seeds from failure patterns.
/// Simulates what an LLM or human reviewer would suggest.
/// Importantly: these are generated WITHOUT looking at test query text,
/// only at the intent name and what vocabulary patterns typically match.
fn generate_fix_seeds(failures: &[(&Example, String)]) -> Vec<(String, String)> {
    let mut fixes: Vec<(String, String)> = Vec::new();
    let mut seen_intents: HashSet<String> = HashSet::new();

    for (_, intent) in failures {
        if seen_intents.contains(intent) { continue; }
        seen_intents.insert(intent.clone());

        // Generate 2-3 fix seeds per failing intent based on common patterns
        // These are NOT derived from the test queries — they're generic patterns
        match intent.as_str() {
            "edit_account" => {
                fixes.push((intent.clone(), "modify my profile settings".to_string()));
                fixes.push((intent.clone(), "change my account details".to_string()));
            }
            "get_refund" => {
                fixes.push((intent.clone(), "reimburse my purchase".to_string()));
                fixes.push((intent.clone(), "how do I get compensated".to_string()));
            }
            "change_order" => {
                fixes.push((intent.clone(), "modify items in my purchase".to_string()));
                fixes.push((intent.clone(), "swap something in my order".to_string()));
            }
            "get_invoice" => {
                fixes.push((intent.clone(), "send me the receipt".to_string()));
                fixes.push((intent.clone(), "download my billing document".to_string()));
            }
            "check_invoice" => {
                fixes.push((intent.clone(), "view my billing statement".to_string()));
                fixes.push((intent.clone(), "look at recent charges".to_string()));
            }
            "switch_account" => {
                fixes.push((intent.clone(), "change to different profile".to_string()));
                fixes.push((intent.clone(), "swap between my accounts".to_string()));
            }
            "delete_account" => {
                fixes.push((intent.clone(), "remove my profile permanently".to_string()));
                fixes.push((intent.clone(), "close and erase my data".to_string()));
            }
            "payment_issue" => {
                fixes.push((intent.clone(), "having trouble with payment".to_string()));
                fixes.push((intent.clone(), "transaction declined or failed".to_string()));
            }
            "change_shipping_address" => {
                fixes.push((intent.clone(), "update where to deliver".to_string()));
                fixes.push((intent.clone(), "correct my mailing location".to_string()));
            }
            "check_refund_policy" => {
                fixes.push((intent.clone(), "what are the return rules".to_string()));
                fixes.push((intent.clone(), "conditions for getting money back".to_string()));
            }
            "contact_customer_service" => {
                fixes.push((intent.clone(), "reach your support team".to_string()));
                fixes.push((intent.clone(), "how do I get help from an agent".to_string()));
            }
            "contact_human_agent" => {
                fixes.push((intent.clone(), "speak with a real person".to_string()));
                fixes.push((intent.clone(), "transfer me to a live agent".to_string()));
            }
            "create_account" => {
                fixes.push((intent.clone(), "sign up for a new profile".to_string()));
                fixes.push((intent.clone(), "register as a new user".to_string()));
            }
            "check_cancellation_fee" => {
                fixes.push((intent.clone(), "how much to cancel".to_string()));
                fixes.push((intent.clone(), "is there a penalty for cancelling".to_string()));
            }
            _ => {
                // Generic: use intent name words as a seed
                let words: Vec<&str> = intent.split('_').collect();
                fixes.push((intent.clone(), format!("help with {} issue", words.join(" "))));
            }
        }
    }

    // Also add some deliberately dangerous seeds to test the guard
    fixes.push(("cancel_order".to_string(), "refund my money back".to_string())); // "refund" collides with get_refund
    fixes.push(("get_refund".to_string(), "cancel and return everything".to_string())); // "cancel" collides with cancel_order
    fixes.push(("track_order".to_string(), "check my invoice status".to_string())); // "invoice" collides with check_invoice

    fixes
}

fn evaluate(router: &Router, examples: &[&Example]) -> (f32, HashMap<String, f32>) {
    let mut correct = 0;
    let total = examples.len();
    let mut intent_stats: HashMap<String, (usize, usize)> = HashMap::new();

    for ex in examples {
        let result = router.route(&ex.text);
        let entry = intent_stats.entry(ex.intents[0].clone()).or_insert((0, 0));
        entry.1 += 1;
        if !result.is_empty() && result[0].id == ex.intents[0] {
            correct += 1;
            entry.0 += 1;
        }
    }

    let top1 = correct as f32 / total as f32 * 100.0;
    let per_intent: HashMap<String, f32> = intent_stats.into_iter()
        .map(|(k, (c, t))| (k, if t > 0 { c as f32 / t as f32 * 100.0 } else { 0.0 }))
        .collect();

    (top1, per_intent)
}
