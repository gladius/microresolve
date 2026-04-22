//! Presidio benchmark: run our hybrid entity detector against Microsoft
//! Presidio's synth_dataset_v2.json (1500 examples, 17 entity types).
//!
//! Source dataset: github.com/microsoft/presidio-research/data/synth_dataset_v2.json
//!
//! Output: precision, recall, F1 per entity type. The honest interpretation
//! is "for the entity types our hybrid detector has patterns for, here's
//! how we compare." Types we don't cover (PERSON, ORGANIZATION, etc.)
//! show 0% recall — that's a coverage gap, not a quality issue, and
//! highlights where LLM-distilled pattern generation needs to fill in next.
//!
//! Run:
//!   cargo run --release --bin presidio_bench -- /tmp/presidio-research/data/synth_dataset_v2.json

use microresolve::entity::EntityLayer;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Deserialize)]
struct Example {
    full_text: String,
    spans: Vec<Span>,
}

#[derive(Deserialize)]
struct Span {
    entity_type: String,
}

/// Map Presidio's entity_type names to our hybrid detector's labels.
/// Anything unmapped is a coverage gap (we have no detector for it yet).
fn map_presidio_label(presidio: &str) -> Option<&'static str> {
    match presidio {
        "CREDIT_CARD"   => Some("CC"),
        "US_SSN"        => Some("SSN"),
        "EMAIL_ADDRESS" => Some("EMAIL"),
        "PHONE_NUMBER"  => Some("PHONE"),
        "IP_ADDRESS"    => Some("IPV4"),
        // Out-of-scope for current built-in patterns:
        //   PERSON, ORGANIZATION, GPE, STREET_ADDRESS, DATE_TIME, TITLE,
        //   AGE, NRP, ZIP_CODE, DOMAIN_NAME, IBAN_CODE, US_DRIVER_LICENSE
        // These will count as "not covered" in the report.
        _ => None,
    }
}

#[derive(Default)]
struct PerLabelStats {
    tp: usize,
    fp: usize,
    fn_: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/presidio-research/data/synth_dataset_v2.json".to_string());

    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {}", path, e));
    let examples: Vec<Example> = serde_json::from_str(&json)
        .expect("parse synth_dataset_v2.json");

    println!("\n=== Presidio Scale Test (synth_dataset_v2.json) ===");
    println!("Source: github.com/microsoft/presidio-research");
    println!("Examples: {}\n", examples.len());

    let entity = EntityLayer::default_pii();

    // Per-label stats and overall.
    let mut per_label: HashMap<&'static str, PerLabelStats> = HashMap::new();
    let mut covered_types_seen: HashMap<&'static str, usize> = HashMap::new();
    let mut uncovered_types_seen: HashMap<String, usize> = HashMap::new();
    let mut covered_examples: usize = 0;       // examples that have at least one entity we cover
    let mut uncovered_only_examples: usize = 0; // examples where all entities are out-of-scope
    let mut empty_examples: usize = 0;          // examples with no entities at all

    let mut total_micros: u128 = 0;
    let mut latencies: Vec<u128> = Vec::with_capacity(examples.len());

    for ex in &examples {
        // Build the set of entity types our detector should find for this example.
        let expected: HashSet<&'static str> = ex.spans.iter()
            .filter_map(|s| {
                if let Some(label) = map_presidio_label(&s.entity_type) {
                    *covered_types_seen.entry(label).or_insert(0) += 1;
                    Some(label)
                } else {
                    *uncovered_types_seen.entry(s.entity_type.clone()).or_insert(0) += 1;
                    None
                }
            })
            .collect();

        if ex.spans.is_empty() { empty_examples += 1; }
        else if expected.is_empty() { uncovered_only_examples += 1; }
        else { covered_examples += 1; }

        // Run detector and time it.
        let t0 = Instant::now();
        let detected: HashSet<&'static str> = entity.detect(&ex.full_text)
            .into_iter()
            .collect();
        let elapsed = t0.elapsed().as_micros();
        total_micros += elapsed;
        latencies.push(elapsed);

        // Per-label confusion (only over labels we have detectors for).
        for label in &expected {
            let s = per_label.entry(*label).or_default();
            if detected.contains(label) { s.tp += 1; } else { s.fn_ += 1; }
        }
        for label in &detected {
            if !expected.contains(label) {
                let s = per_label.entry(*label).or_default();
                s.fp += 1;
            }
        }
    }

    // ── Report ───────────────────────────────────────────────────────────────
    println!("Coverage breakdown:");
    println!("  examples with at least one in-scope entity : {}", covered_examples);
    println!("  examples with only out-of-scope entities   : {}", uncovered_only_examples);
    println!("  examples with no entities                  : {}", empty_examples);
    println!();

    println!("Per-label results (only labels our detector covers):");
    println!("{:<10} {:>8} {:>5} {:>5} {:>5} {:>10} {:>9} {:>8}",
        "label", "expected", "TP", "FP", "FN", "precision", "recall", "F1");
    println!("{}", "─".repeat(70));

    let mut labels: Vec<&&str> = per_label.keys().collect();
    labels.sort();

    let mut total_tp = 0usize;
    let mut total_fp = 0usize;
    let mut total_fn = 0usize;

    for label in labels {
        let s = &per_label[*label];
        let expected_count = covered_types_seen.get(*label).copied().unwrap_or(0);
        let p = if s.tp + s.fp == 0 { 0.0 } else { s.tp as f32 / (s.tp + s.fp) as f32 };
        let r = if s.tp + s.fn_ == 0 { 0.0 } else { s.tp as f32 / (s.tp + s.fn_) as f32 };
        let f1 = if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) };
        println!("{:<10} {:>8} {:>5} {:>5} {:>5} {:>9.2} {:>9.2} {:>8.2}",
            label, expected_count, s.tp, s.fp, s.fn_, p, r, f1);
        total_tp += s.tp;
        total_fp += s.fp;
        total_fn += s.fn_;
    }

    println!("{}", "─".repeat(70));
    let p_overall = if total_tp + total_fp == 0 { 0.0 } else { total_tp as f32 / (total_tp + total_fp) as f32 };
    let r_overall = if total_tp + total_fn == 0 { 0.0 } else { total_tp as f32 / (total_tp + total_fn) as f32 };
    let f1_overall = if p_overall + r_overall == 0.0 { 0.0 } else { 2.0 * p_overall * r_overall / (p_overall + r_overall) };
    println!("{:<10} {:>8} {:>5} {:>5} {:>5} {:>9.2} {:>9.2} {:>8.2}",
        "OVERALL", total_tp + total_fn, total_tp, total_fp, total_fn,
        p_overall, r_overall, f1_overall);

    println!("\nLatency:");
    latencies.sort();
    let median = latencies[latencies.len() / 2];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
    let max = *latencies.last().unwrap();
    println!("  median : {} µs", median);
    println!("  p99    : {} µs", p99);
    println!("  max    : {} µs", max);
    println!("  mean   : {} µs", total_micros / examples.len() as u128);
    println!("  total  : {:.2} ms across {} examples", total_micros as f64 / 1000.0, examples.len());

    println!("\nOut-of-scope entity types (we have no patterns for these):");
    let mut uncov: Vec<(String, usize)> = uncovered_types_seen.into_iter().collect();
    uncov.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (t, c) in &uncov {
        println!("  {:<20} : {} occurrences", t, c);
    }

    println!("\nNotes:");
    println!("  - This measures only entity TYPES we built detectors for");
    println!("    (CC, SSN, EMAIL, PHONE, IPV4). Out-of-scope types listed above");
    println!("    represent coverage gaps to close via LLM-distilled patterns.");
    println!("  - Spans with values that span multiple lines or use unusual");
    println!("    formats may be missed by the standard regex patterns.");
    println!("  - Precision penalizes detecting entities Presidio didn't tag.");
    println!("    Some FPs may be valid detections in unannotated text.");
}
