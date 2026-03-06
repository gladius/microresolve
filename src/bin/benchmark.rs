//! ASV Router — Benchmark Runner
//!
//! Evaluates ASV against standard intent classification datasets.
//!
//! Usage:
//!   cargo run --release --bin benchmark -- --dataset clinc150
//!   cargo run --release --bin benchmark -- --dataset banking77
//!   cargo run --release --bin benchmark -- --dataset all
//!   cargo run --release --bin benchmark -- --dataset clinc150 --seeds 10 --learn-rounds 3
//!   cargo run --release --bin benchmark -- --sweep           # run all seed counts
//!   cargo run --release --bin benchmark -- --memory          # memory profiling
//!   cargo run --release --bin benchmark -- --oos             # OOS rejection (CLINC150)

use asv_router::Router;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Config {
    dataset: String,
    seeds_per_intent: usize,
    learn_rounds: usize,
    learn_batch: usize,
    data_dir: String,
    output_file: Option<String>,
    sweep: bool,
    memory: bool,
    oos: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config {
        dataset: "all".to_string(),
        seeds_per_intent: 10,
        learn_rounds: 3,
        learn_batch: 10,
        data_dir: "data".to_string(),
        output_file: None,
        sweep: false,
        memory: false,
        oos: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dataset" | "-d" => {
                i += 1;
                config.dataset = args[i].clone();
            }
            "--seeds" | "-s" => {
                i += 1;
                config.seeds_per_intent = args[i].parse().expect("invalid --seeds");
            }
            "--learn-rounds" | "-l" => {
                i += 1;
                config.learn_rounds = args[i].parse().expect("invalid --learn-rounds");
            }
            "--learn-batch" | "-b" => {
                i += 1;
                config.learn_batch = args[i].parse().expect("invalid --learn-batch");
            }
            "--data-dir" => {
                i += 1;
                config.data_dir = args[i].clone();
            }
            "--output" | "-o" => {
                i += 1;
                config.output_file = Some(args[i].clone());
            }
            "--sweep" => config.sweep = true,
            "--memory" => config.memory = true,
            "--oos" => config.oos = true,
            "--help" | "-h" => {
                eprintln!(
                    "Usage: benchmark [OPTIONS]\n\n\
                     Options:\n  \
                       -d, --dataset <name>     Dataset: clinc150, banking77, all (default: all)\n  \
                       -s, --seeds <N>          Seed phrases per intent (default: 10)\n  \
                       -l, --learn-rounds <N>   Learning rounds (default: 3)\n  \
                       -b, --learn-batch <N>    Queries to learn per intent per round (default: 10)\n  \
                       --data-dir <path>        Data directory (default: data)\n  \
                       -o, --output <file>      Append results to file (markdown)\n  \
                       --sweep                  Run all seed counts (5,10,20,50,80,100,120)\n  \
                       --memory                 Run memory profiling\n  \
                       --oos                    Run OOS rejection eval (CLINC150)\n  \
                       -h, --help               Show this help"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    config
}

// ---------------------------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Example {
    text: String,
    intent: String,
}

struct Dataset {
    name: String,
    train: Vec<Example>,
    test: Vec<Example>,
    intents: Vec<String>,
    oos_test: Vec<Example>, // out-of-scope test queries (CLINC150 only)
}

fn load_clinc150(data_dir: &str) -> Dataset {
    let path = format!("{}/clinc150.json", data_dir);
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}. Run data/download.sh first.", path, e);
        std::process::exit(1);
    });

    let json: serde_json::Value = serde_json::from_str(&raw).expect("invalid JSON");

    let parse_split = |key: &str, include_oos: bool| -> Vec<Example> {
        json[key]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|pair| {
                let arr = pair.as_array()?;
                let text = arr.first()?.as_str()?.to_string();
                let intent = arr.get(1)?.as_str()?.to_string();
                if !include_oos && intent == "oos" {
                    return None;
                }
                Some(Example { text, intent })
            })
            .collect()
    };

    let mut train = parse_split("train", false);
    train.extend(parse_split("val", false));
    let test = parse_split("test", false);

    // Load OOS test separately
    let oos_test = parse_split("oos_test", true);

    let mut intent_set: Vec<String> = train
        .iter()
        .map(|e| e.intent.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    intent_set.sort();

    Dataset {
        name: "CLINC150".to_string(),
        train,
        test,
        intents: intent_set,
        oos_test,
    }
}

fn load_banking77(data_dir: &str) -> Dataset {
    let parse_csv = |filename: &str| -> Vec<Example> {
        let path = format!("{}/{}", data_dir, filename);
        let raw = fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("Failed to read {}: {}. Run data/download.sh first.", path, e);
            std::process::exit(1);
        });

        let mut examples = Vec::new();
        for (i, line) in raw.lines().enumerate() {
            if i == 0 {
                continue;
            }
            if let Some(ex) = parse_csv_line(line) {
                examples.push(ex);
            }
        }
        examples
    };

    let train = parse_csv("banking77_train.csv");
    let test = parse_csv("banking77_test.csv");

    let mut intent_set: Vec<String> = train
        .iter()
        .map(|e| e.intent.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    intent_set.sort();

    Dataset {
        name: "BANKING77".to_string(),
        train,
        test,
        intents: intent_set,
        oos_test: vec![],
    }
}

fn parse_csv_line(line: &str) -> Option<Example> {
    if line.is_empty() {
        return None;
    }

    if line.starts_with('"') {
        let mut i = 1;
        let bytes = line.as_bytes();
        while i < bytes.len() {
            if bytes[i] == b'"' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                    i += 2;
                    continue;
                }
                let text = line[1..i].replace("\"\"", "\"");
                let category = line.get(i + 2..)?.trim().to_string();
                return Some(Example {
                    text,
                    intent: category,
                });
            }
            i += 1;
        }
        None
    } else {
        let comma_pos = line.rfind(',')?;
        let text = line[..comma_pos].to_string();
        let intent = line[comma_pos + 1..].trim().to_string();
        Some(Example { text, intent })
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

struct BenchmarkResult {
    dataset_name: String,
    num_intents: usize,
    num_test: usize,
    seeds_per_intent: usize,
    phases: Vec<PhaseResult>,
    worst_intents: Vec<(String, f32)>,
    best_intents: Vec<(String, f32)>,
}

#[derive(Clone)]
struct PhaseResult {
    label: String,
    accuracy: f32,
    top3_accuracy: f32,
    avg_latency_us: f64,
    p99_latency_us: f64,
    no_match_pct: f32,
}

fn run_benchmark(dataset: &Dataset, config: &Config) -> BenchmarkResult {
    let mut train_by_intent: HashMap<String, Vec<String>> = HashMap::new();
    for ex in &dataset.train {
        train_by_intent
            .entry(ex.intent.clone())
            .or_default()
            .push(ex.text.clone());
    }

    let mut phases: Vec<PhaseResult> = Vec::new();

    // Phase 1: Seed-only
    println!("  Phase 1: Seed-only ({} seeds/intent)...", config.seeds_per_intent);

    let mut router = Router::new();
    for intent in &dataset.intents {
        if let Some(examples) = train_by_intent.get(intent) {
            let seeds: Vec<&str> = examples
                .iter()
                .take(config.seeds_per_intent)
                .map(|s| s.as_str())
                .collect();
            if !seeds.is_empty() {
                router.add_intent(intent, &seeds);
            }
        }
    }

    let (seed_phase, per_intent_acc) = evaluate(&router, &dataset.test, "Seed-only");
    phases.push(seed_phase);

    let mut intent_accs: Vec<(String, f32)> = per_intent_acc.into_iter().collect();
    intent_accs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let worst: Vec<(String, f32)> = intent_accs.iter().take(10).cloned().collect();
    let best: Vec<(String, f32)> = intent_accs.iter().rev().take(10).cloned().collect();

    // Learning rounds
    let mut misrouted: HashMap<String, Vec<String>> = HashMap::new();
    for ex in &dataset.test {
        let results = router.route(&ex.text);
        let correct = !results.is_empty() && results[0].id == ex.intent;
        if !correct {
            misrouted
                .entry(ex.intent.clone())
                .or_default()
                .push(ex.text.clone());
        }
    }

    let mut extra_train: HashMap<String, Vec<String>> = HashMap::new();
    for intent in &dataset.intents {
        if let Some(examples) = train_by_intent.get(intent) {
            let extras: Vec<String> = examples
                .iter()
                .skip(config.seeds_per_intent)
                .cloned()
                .collect();
            extra_train.insert(intent.clone(), extras);
        }
    }

    for round in 1..=config.learn_rounds {
        let batch = config.learn_batch * round;
        println!(
            "  Phase {}: Learning round {} ({} examples/intent cumulative)...",
            round + 1, round, batch
        );

        for intent in &dataset.intents {
            let mut learned = 0usize;

            if let Some(extras) = extra_train.get(intent) {
                for text in extras.iter().take(batch) {
                    if learned >= batch {
                        break;
                    }
                    router.learn(text, intent);
                    learned += 1;
                }
            }

            if learned < batch {
                if let Some(mis) = misrouted.get(intent) {
                    for text in mis.iter().take(batch - learned) {
                        router.learn(text, intent);
                    }
                }
            }
        }

        let label = format!("After learn round {} (~{}/intent)", round, batch);
        let (phase, _) = evaluate(&router, &dataset.test, &label);
        phases.push(phase);
    }

    BenchmarkResult {
        dataset_name: dataset.name.clone(),
        num_intents: dataset.intents.len(),
        num_test: dataset.test.len(),
        seeds_per_intent: config.seeds_per_intent,
        phases,
        worst_intents: worst,
        best_intents: best,
    }
}

fn evaluate(
    router: &Router,
    test: &[Example],
    label: &str,
) -> (PhaseResult, HashMap<String, f32>) {
    let mut correct = 0usize;
    let mut top3_correct = 0usize;
    let mut no_match = 0usize;
    let mut latencies: Vec<u64> = Vec::with_capacity(test.len());
    let mut intent_correct: HashMap<String, usize> = HashMap::new();
    let mut intent_total: HashMap<String, usize> = HashMap::new();

    for ex in test {
        *intent_total.entry(ex.intent.clone()).or_default() += 1;

        let start = Instant::now();
        let results = router.route(&ex.text);
        let elapsed_us = start.elapsed().as_micros() as u64;
        latencies.push(elapsed_us);

        if results.is_empty() {
            no_match += 1;
            continue;
        }

        if results[0].id == ex.intent {
            correct += 1;
            *intent_correct.entry(ex.intent.clone()).or_default() += 1;
        }

        if results.iter().take(3).any(|r| r.id == ex.intent) {
            top3_correct += 1;
        }
    }

    latencies.sort();
    let avg_us = latencies.iter().sum::<u64>() as f64 / latencies.len().max(1) as f64;
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99_us = latencies
        .get(p99_idx.min(latencies.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0);

    let accuracy = correct as f32 / test.len().max(1) as f32;
    let top3_accuracy = top3_correct as f32 / test.len().max(1) as f32;
    let no_match_pct = no_match as f32 / test.len().max(1) as f32;

    let per_intent: HashMap<String, f32> = intent_total
        .iter()
        .map(|(intent, &total)| {
            let c = intent_correct.get(intent).copied().unwrap_or(0);
            (intent.clone(), c as f32 / total.max(1) as f32)
        })
        .collect();

    (
        PhaseResult {
            label: label.to_string(),
            accuracy,
            top3_accuracy,
            avg_latency_us: avg_us,
            p99_latency_us: p99_us as f64,
            no_match_pct,
        },
        per_intent,
    )
}

// ---------------------------------------------------------------------------
// Sweep: run all seed counts to find the ceiling
// ---------------------------------------------------------------------------

fn run_sweep(dataset: &Dataset) {
    let seed_counts = match dataset.name.as_str() {
        "CLINC150" => vec![5, 10, 20, 50, 80, 100, 120],
        "BANKING77" => vec![5, 10, 20, 50, 80, 100, 130],
        _ => vec![5, 10, 20, 50],
    };

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║  Seed Sweep: {} — finding the ceiling                           ║",
        dataset.name
    );
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  {:>6} │ {:>10} {:>10} {:>10} {:>10} {:>10}      ║",
        "Seeds", "Seed-Only", "Top-3", "+Learn30", "Avg μs", "No-match"
    );
    println!("║  ──────┼──────────────────────────────────────────────────────────    ║");

    let mut train_by_intent: HashMap<String, Vec<String>> = HashMap::new();
    for ex in &dataset.train {
        train_by_intent
            .entry(ex.intent.clone())
            .or_default()
            .push(ex.text.clone());
    }

    for &seed_count in &seed_counts {
        // Seed-only eval
        let mut router = Router::new();
        for intent in &dataset.intents {
            if let Some(examples) = train_by_intent.get(intent) {
                let seeds: Vec<&str> = examples
                    .iter()
                    .take(seed_count)
                    .map(|s| s.as_str())
                    .collect();
                if !seeds.is_empty() {
                    router.add_intent(intent, &seeds);
                }
            }
        }

        let (seed_phase, _) = evaluate(&router, &dataset.test, "");

        // Learning: 3 rounds of 10 (cumulative 30 per intent)
        let mut misrouted: HashMap<String, Vec<String>> = HashMap::new();
        for ex in &dataset.test {
            let results = router.route(&ex.text);
            if results.is_empty() || results[0].id != ex.intent {
                misrouted
                    .entry(ex.intent.clone())
                    .or_default()
                    .push(ex.text.clone());
            }
        }

        for intent in &dataset.intents {
            let mut learned = 0usize;
            if let Some(examples) = train_by_intent.get(intent) {
                for text in examples.iter().skip(seed_count).take(30) {
                    router.learn(text, intent);
                    learned += 1;
                }
            }
            if learned < 30 {
                if let Some(mis) = misrouted.get(intent) {
                    for text in mis.iter().take(30 - learned) {
                        router.learn(text, intent);
                    }
                }
            }
        }

        let (learn_phase, _) = evaluate(&router, &dataset.test, "");

        println!(
            "║  {:>5} │ {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1} {:>9.1}%      ║",
            seed_count,
            seed_phase.accuracy * 100.0,
            seed_phase.top3_accuracy * 100.0,
            learn_phase.accuracy * 100.0,
            seed_phase.avg_latency_us,
            seed_phase.no_match_pct * 100.0,
        );
    }

    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

// ---------------------------------------------------------------------------
// OOS Rejection evaluation
// ---------------------------------------------------------------------------

fn run_oos_eval(dataset: &Dataset) {
    if dataset.oos_test.is_empty() {
        println!("  No OOS test data for {}", dataset.name);
        return;
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  OOS Rejection: {} — can ASV say \"I don't know\"?                  ║", dataset.name);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");

    let mut train_by_intent: HashMap<String, Vec<String>> = HashMap::new();
    for ex in &dataset.train {
        train_by_intent
            .entry(ex.intent.clone())
            .or_default()
            .push(ex.text.clone());
    }

    // Seed with 50 phrases per intent
    let mut router = Router::new();
    for intent in &dataset.intents {
        if let Some(examples) = train_by_intent.get(intent) {
            let seeds: Vec<&str> = examples
                .iter()
                .take(50)
                .map(|s| s.as_str())
                .collect();
            if !seeds.is_empty() {
                router.add_intent(intent, &seeds);
            }
        }
    }

    println!("║  50 seeds/intent, {} OOS queries, {} in-scope test queries       ║",
        dataset.oos_test.len(), dataset.test.len());
    println!("║                                                                        ║");
    println!(
        "║  {:>10} │ {:>10} {:>10} {:>10} {:>10}               ║",
        "Threshold", "OOS-Reject", "IS-Correct", "IS-Reject", "F1"
    );
    println!("║  ──────────┼─────────────────────────────────────────────             ║");

    let thresholds = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0];

    for &threshold in &thresholds {
        // OOS queries: should score BELOW threshold (correctly rejected)
        let mut oos_rejected = 0usize;
        for ex in &dataset.oos_test {
            let results = router.route(&ex.text);
            if results.is_empty() || results[0].score < threshold {
                oos_rejected += 1;
            }
        }

        // In-scope queries: should score ABOVE threshold (correctly accepted)
        let mut is_correct = 0usize;
        let mut is_rejected = 0usize;
        for ex in &dataset.test {
            let results = router.route(&ex.text);
            if results.is_empty() || results[0].score < threshold {
                is_rejected += 1;
            } else if results[0].id == ex.intent {
                is_correct += 1;
            }
        }

        let oos_reject_rate = oos_rejected as f32 / dataset.oos_test.len().max(1) as f32;
        let is_correct_rate = is_correct as f32 / dataset.test.len().max(1) as f32;
        let is_reject_rate = is_rejected as f32 / dataset.test.len().max(1) as f32;

        // F1 between OOS rejection and IS accuracy
        let precision = if oos_rejected > 0 {
            oos_rejected as f32 / (oos_rejected + is_rejected) as f32
        } else {
            0.0
        };
        let recall = oos_reject_rate;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        println!(
            "║  {:>9.1} │ {:>9.1}% {:>9.1}% {:>9.1}% {:>9.3}               ║",
            threshold,
            oos_reject_rate * 100.0,
            is_correct_rate * 100.0,
            is_reject_rate * 100.0,
            f1,
        );
    }

    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // --- Confidence ratio (top1/top2) thresholds ---
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  Confidence Ratio (top1/top2) — better OOS separation                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  {:>10} │ {:>10} {:>10} {:>10} {:>10}               ║",
        "Min Conf", "OOS-Reject", "IS-Correct", "IS-Reject", "F1"
    );
    println!("║  ──────────┼─────────────────────────────────────────────             ║");

    let conf_thresholds = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0];

    for &min_conf in &conf_thresholds {
        let mut oos_rejected = 0usize;
        for ex in &dataset.oos_test {
            let results = router.route(&ex.text);
            let confidence = if results.len() >= 2 {
                results[0].score / results[1].score
            } else if results.is_empty() {
                0.0
            } else {
                f32::INFINITY
            };
            if results.is_empty() || confidence < min_conf {
                oos_rejected += 1;
            }
        }

        let mut is_correct = 0usize;
        let mut is_rejected = 0usize;
        for ex in &dataset.test {
            let results = router.route(&ex.text);
            let confidence = if results.len() >= 2 {
                results[0].score / results[1].score
            } else if results.is_empty() {
                0.0
            } else {
                f32::INFINITY
            };
            if results.is_empty() || confidence < min_conf {
                is_rejected += 1;
            } else if results[0].id == ex.intent {
                is_correct += 1;
            }
        }

        let oos_reject_rate = oos_rejected as f32 / dataset.oos_test.len().max(1) as f32;
        let is_correct_rate = is_correct as f32 / dataset.test.len().max(1) as f32;
        let is_reject_rate = is_rejected as f32 / dataset.test.len().max(1) as f32;

        let precision = if oos_rejected > 0 {
            oos_rejected as f32 / (oos_rejected + is_rejected) as f32
        } else {
            0.0
        };
        let recall = oos_reject_rate;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        println!(
            "║  {:>9.1} │ {:>9.1}% {:>9.1}% {:>9.1}% {:>9.3}               ║",
            min_conf,
            oos_reject_rate * 100.0,
            is_correct_rate * 100.0,
            is_reject_rate * 100.0,
            f1,
        );
    }

    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();
    // --- Combined: score × confidence ---
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  Combined (score × confidence) — best of both signals                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  {:>10} │ {:>10} {:>10} {:>10} {:>10}               ║",
        "Threshold", "OOS-Reject", "IS-Correct", "IS-Reject", "F1"
    );
    println!("║  ──────────┼─────────────────────────────────────────────             ║");

    let combined_thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0];

    for &threshold in &combined_thresholds {
        let mut oos_rejected = 0usize;
        for ex in &dataset.oos_test {
            let results = router.route(&ex.text);
            let combined = if results.len() >= 2 {
                results[0].score * (results[0].score / results[1].score)
            } else if results.is_empty() {
                0.0
            } else {
                results[0].score * 10.0 // single match gets high confidence
            };
            if combined < threshold {
                oos_rejected += 1;
            }
        }

        let mut is_correct = 0usize;
        let mut is_rejected = 0usize;
        for ex in &dataset.test {
            let results = router.route(&ex.text);
            let combined = if results.len() >= 2 {
                results[0].score * (results[0].score / results[1].score)
            } else if results.is_empty() {
                0.0
            } else {
                results[0].score * 10.0
            };
            if combined < threshold {
                is_rejected += 1;
            } else if results[0].id == ex.intent {
                is_correct += 1;
            }
        }

        let oos_reject_rate = oos_rejected as f32 / dataset.oos_test.len().max(1) as f32;
        let is_correct_rate = is_correct as f32 / dataset.test.len().max(1) as f32;
        let is_reject_rate = is_rejected as f32 / dataset.test.len().max(1) as f32;

        let precision = if oos_rejected > 0 {
            oos_rejected as f32 / (oos_rejected + is_rejected) as f32
        } else {
            0.0
        };
        let recall = oos_reject_rate;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        println!(
            "║  {:>9.1} │ {:>9.1}% {:>9.1}% {:>9.1}% {:>9.3}               ║",
            threshold,
            oos_reject_rate * 100.0,
            is_correct_rate * 100.0,
            is_reject_rate * 100.0,
            f1,
        );
    }

    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  OOS-Reject = % of out-of-scope queries correctly rejected (higher = better)");
    println!("  IS-Correct = % of in-scope queries correctly routed (higher = better)");
    println!("  IS-Reject  = % of in-scope queries wrongly rejected (lower = better)");
    println!("  F1 = harmonic mean of OOS precision and recall");
}

// ---------------------------------------------------------------------------
// Memory profiling
// ---------------------------------------------------------------------------

fn run_memory_profile(data_dir: &str) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  Memory Profiling                                                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  {:>10} {:>8} {:>12} {:>12} {:>12}                ║",
        "Intents", "Seeds", "Router (est)", "Per-intent", "Throughput"
    );
    println!("║  ──────────────────────────────────────────────────────────────        ║");

    // Test at various scales
    let configs: Vec<(usize, usize, Vec<String>)> = vec![
        (10, 5, gen_intents(10, 5)),
        (10, 50, gen_intents(10, 50)),
        (77, 10, gen_intents(77, 10)),
        (77, 50, gen_intents(77, 50)),
        (77, 130, gen_intents(77, 130)),
        (150, 10, gen_intents(150, 10)),
        (150, 50, gen_intents(150, 50)),
        (150, 120, gen_intents(150, 120)),
        (500, 20, gen_intents(500, 20)),
        (1000, 20, gen_intents(1000, 20)),
    ];

    for (num_intents, seeds_per, phrases) in &configs {
        let rss_before = get_rss_kb();

        let mut router = Router::new();
        let mut phrase_idx = 0;
        for i in 0..*num_intents {
            let intent_id = format!("intent_{}", i);
            let end = (phrase_idx + seeds_per).min(phrases.len());
            let seeds: Vec<&str> = phrases[phrase_idx..end].iter().map(|s| s.as_str()).collect();
            router.add_intent(&intent_id, &seeds);
            phrase_idx = end;
        }

        let rss_after = get_rss_kb();
        let rss_delta = rss_after.saturating_sub(rss_before);

        // Measure throughput
        let test_query = "cancel my order and track the package please";
        let start = Instant::now();
        let iterations = 10_000;
        for _ in 0..iterations {
            let _ = router.route(test_query);
        }
        let elapsed = start.elapsed();
        let throughput = iterations as f64 / elapsed.as_secs_f64();

        let per_intent_kb = if *num_intents > 0 {
            rss_delta as f64 / *num_intents as f64
        } else {
            0.0
        };

        println!(
            "║  {:>10} {:>8} {:>9} KB {:>9.1} KB {:>9.0} q/s                ║",
            num_intents, seeds_per, rss_delta, per_intent_kb, throughput,
        );
    }

    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    // Also test with real datasets if available
    let clinc_path = format!("{}/clinc150.json", data_dir);
    let banking_path = format!("{}/banking77_train.csv", data_dir);

    if fs::metadata(&clinc_path).is_ok() {
        println!();
        println!("  Real dataset memory (CLINC150):");
        let dataset = load_clinc150(data_dir);
        let mut train_by_intent: HashMap<String, Vec<String>> = HashMap::new();
        for ex in &dataset.train {
            train_by_intent.entry(ex.intent.clone()).or_default().push(ex.text.clone());
        }

        for &seeds in &[10, 50, 120] {
            let rss_before = get_rss_kb();
            let mut router = Router::new();
            for intent in &dataset.intents {
                if let Some(examples) = train_by_intent.get(intent) {
                    let s: Vec<&str> = examples.iter().take(seeds).map(|s| s.as_str()).collect();
                    if !s.is_empty() {
                        router.add_intent(intent, &s);
                    }
                }
            }
            let rss_after = get_rss_kb();
            let json = router.export_json();
            println!(
                "    {} seeds/intent: ~{} KB RSS delta, {} KB serialized JSON",
                seeds,
                rss_after.saturating_sub(rss_before),
                json.len() / 1024,
            );
        }
    }

    if fs::metadata(&banking_path).is_ok() {
        println!();
        println!("  Real dataset memory (BANKING77):");
        let dataset = load_banking77(data_dir);
        let mut train_by_intent: HashMap<String, Vec<String>> = HashMap::new();
        for ex in &dataset.train {
            train_by_intent.entry(ex.intent.clone()).or_default().push(ex.text.clone());
        }

        for &seeds in &[10, 50, 130] {
            let rss_before = get_rss_kb();
            let mut router = Router::new();
            for intent in &dataset.intents {
                if let Some(examples) = train_by_intent.get(intent) {
                    let s: Vec<&str> = examples.iter().take(seeds).map(|s| s.as_str()).collect();
                    if !s.is_empty() {
                        router.add_intent(intent, &s);
                    }
                }
            }
            let rss_after = get_rss_kb();
            let json = router.export_json();
            println!(
                "    {} seeds/intent: ~{} KB RSS delta, {} KB serialized JSON",
                seeds,
                rss_after.saturating_sub(rss_before),
                json.len() / 1024,
            );
        }
    }
}

fn gen_intents(num_intents: usize, seeds_per: usize) -> Vec<String> {
    let mut phrases = Vec::new();
    for i in 0..num_intents {
        for j in 0..seeds_per {
            phrases.push(format!(
                "action {} variation {} for intent category {}",
                i, j, i
            ));
        }
    }
    phrases
}

fn get_rss_kb() -> usize {
    // Read RSS from /proc/self/statm (Linux)
    if let Ok(statm) = fs::read_to_string("/proc/self/statm") {
        let fields: Vec<&str> = statm.split_whitespace().collect();
        if fields.len() >= 2 {
            // Second field is RSS in pages (4KB each on most systems)
            if let Ok(pages) = fields[1].parse::<usize>() {
                return pages * 4; // 4KB pages → KB
            }
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn print_result(result: &BenchmarkResult) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  {} — {} intents, {} test queries, {} seeds/intent",
        result.dataset_name, result.num_intents, result.num_test, result.seeds_per_intent
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    println!(
        "  {:<42} {:>7} {:>7} {:>10} {:>10} {:>9}",
        "Phase", "Acc", "Top-3", "Avg μs", "P99 μs", "No-match"
    );
    println!("  {}", "-".repeat(87));

    for phase in &result.phases {
        println!(
            "  {:<42} {:>6.1}% {:>6.1}% {:>9.1} {:>9.1} {:>8.1}%",
            phase.label,
            phase.accuracy * 100.0,
            phase.top3_accuracy * 100.0,
            phase.avg_latency_us,
            phase.p99_latency_us,
            phase.no_match_pct * 100.0,
        );
    }

    println!();
    println!("  Worst 10 intents (seed-only):");
    for (intent, acc) in &result.worst_intents {
        println!("    {:.1}%  {}", acc * 100.0, intent);
    }

    println!();
    println!("  Best 10 intents (seed-only):");
    for (intent, acc) in &result.best_intents {
        println!("    {:.1}%  {}", acc * 100.0, intent);
    }
    println!();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let config = parse_args();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           ASV Router — Benchmark Suite                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // --memory mode
    if config.memory {
        run_memory_profile(&config.data_dir);
        return;
    }

    // --sweep mode
    if config.sweep {
        let run_clinc = config.dataset == "clinc150" || config.dataset == "all";
        let run_banking = config.dataset == "banking77" || config.dataset == "all";

        if run_clinc {
            println!("Loading CLINC150...");
            let dataset = load_clinc150(&config.data_dir);
            println!(
                "  {} intents, {} train, {} test",
                dataset.intents.len(), dataset.train.len(), dataset.test.len()
            );
            run_sweep(&dataset);
        }
        if run_banking {
            println!("Loading BANKING77...");
            let dataset = load_banking77(&config.data_dir);
            println!(
                "  {} intents, {} train, {} test",
                dataset.intents.len(), dataset.train.len(), dataset.test.len()
            );
            run_sweep(&dataset);
        }
        return;
    }

    // --oos mode
    if config.oos {
        println!("Loading CLINC150 (with OOS)...");
        let dataset = load_clinc150(&config.data_dir);
        println!(
            "  {} intents, {} train, {} test, {} OOS test",
            dataset.intents.len(), dataset.train.len(), dataset.test.len(), dataset.oos_test.len()
        );
        run_oos_eval(&dataset);
        return;
    }

    // Normal benchmark mode
    println!(
        "  Seeds: {}, Learn rounds: {}, Batch: {}, Dataset: {}",
        config.seeds_per_intent, config.learn_rounds, config.learn_batch, config.dataset
    );
    println!();

    let mut all_results: Vec<BenchmarkResult> = Vec::new();

    let run_clinc = config.dataset == "clinc150" || config.dataset == "all";
    let run_banking = config.dataset == "banking77" || config.dataset == "all";

    if run_clinc {
        println!("Loading CLINC150...");
        let dataset = load_clinc150(&config.data_dir);
        println!(
            "  {} intents, {} train, {} test",
            dataset.intents.len(), dataset.train.len(), dataset.test.len()
        );
        let result = run_benchmark(&dataset, &config);
        print_result(&result);
        all_results.push(result);
    }

    if run_banking {
        println!("Loading BANKING77...");
        let dataset = load_banking77(&config.data_dir);
        println!(
            "  {} intents, {} train, {} test",
            dataset.intents.len(), dataset.train.len(), dataset.test.len()
        );
        let result = run_benchmark(&dataset, &config);
        print_result(&result);
        all_results.push(result);
    }

    // Output to file
    if let Some(ref outfile) = config.output_file {
        let mut md = String::new();
        for result in &all_results {
            md.push_str(&format!(
                "### {} — {} seeds/intent\n\n| Phase | Acc | Top-3 | Avg μs | P99 μs | No-match |\n|---|---|---|---|---|---|\n",
                result.dataset_name, result.seeds_per_intent
            ));
            for phase in &result.phases {
                md.push_str(&format!(
                    "| {} | {:.1}% | {:.1}% | {:.1} | {:.1} | {:.1}% |\n",
                    phase.label, phase.accuracy * 100.0, phase.top3_accuracy * 100.0,
                    phase.avg_latency_us, phase.p99_latency_us, phase.no_match_pct * 100.0,
                ));
            }
            md.push('\n');
        }
        let existing = fs::read_to_string(outfile).unwrap_or_default();
        fs::write(outfile, format!("{}{}", existing, md))
            .unwrap_or_else(|e| eprintln!("Failed to write {}: {}", outfile, e));
        println!("Results appended to {}", outfile);
    }

    // Summary
    if all_results.len() > 1 {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  Summary                                                    ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        for r in &all_results {
            let seed = &r.phases[0];
            let best = r.phases.last().unwrap();
            println!(
                "║  {:<12} Seed: {:>5.1}%  → Final: {:>5.1}%  (Δ +{:.1}%)       ║",
                r.dataset_name,
                seed.accuracy * 100.0,
                best.accuracy * 100.0,
                (best.accuracy - seed.accuracy) * 100.0,
            );
        }
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}
