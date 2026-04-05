//! Auto-discovery experiment: can ASV discover intents from unlabeled data?
//!
//! Methodology:
//!   1. Take labeled dataset (Bitext: 27 intents, 26872 examples)
//!   2. Strip all labels — treat as raw unlabeled queries
//!   3. Feed queries one-by-one to a blank router
//!   4. Router auto-clusters using route() + learn()
//!   5. Compare discovered clusters to ground truth labels
//!
//! The router IS the clustering engine:
//!   - route(query) → finds nearest existing cluster
//!   - score > threshold → learn() into that cluster
//!   - score < threshold → create new cluster
//!
//! Run: cargo run --release --bin auto_discover

use asv_router::Router;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(serde::Deserialize)]
struct Example {
    text: String,
    intents: Vec<String>,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   ASV Auto-Discovery: Unsupervised Intent Clustering   ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let base = "tests/data/benchmarks";

    // Load Bitext dataset (has 27 ground truth intents)
    let data = std::fs::read_to_string(format!("{}/bitext_all.json", base))
        .expect("Run download_datasets.py first");
    let examples: Vec<Example> = serde_json::from_str(&data).unwrap();

    println!("Dataset: Bitext Customer Support");
    println!("Total queries: {}", examples.len());
    println!("Ground truth intents: 27\n");

    // Count ground truth distribution
    let mut gt_counts: HashMap<String, usize> = HashMap::new();
    for ex in &examples {
        *gt_counts.entry(ex.intents[0].clone()).or_insert(0) += 1;
    }

    // Shuffle deterministically (use index-based shuffle to avoid rand dependency)
    let mut indices: Vec<usize> = (0..examples.len()).collect();
    // Simple deterministic shuffle using modular arithmetic
    let n = indices.len();
    for i in 0..n {
        let j = (i * 7919 + 104729) % n; // prime-based shuffle
        indices.swap(i, j);
    }

    // Use a sample of 5000 for threshold sweep (fast), full for best threshold
    let sample_size = 5000.min(indices.len());
    let sample_indices: Vec<usize> = indices[..sample_size].to_vec();

    println!("--- Threshold sweep (sample of {} queries) ---\n", sample_size);

    let thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
    let mut best_threshold = 1.0f32;
    let mut best_nmi = 0.0f64;

    for &threshold in &thresholds {
        let nmi = run_experiment(&examples, &sample_indices, &gt_counts, threshold);
        if nmi > best_nmi {
            best_nmi = nmi;
            best_threshold = threshold;
        }
    }

    println!("\n  Best threshold: {:.1} (NMI={:.3})", best_threshold, best_nmi);

    // Run best threshold with detailed analysis on full dataset
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  DETAILED ANALYSIS (threshold = {:.1}, full {} queries)", best_threshold, examples.len());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    run_detailed(&examples, &indices, &gt_counts, best_threshold);
}

fn run_experiment(
    examples: &[Example],
    indices: &[usize],
    gt_counts: &HashMap<String, usize>,
    threshold: f32,
) -> f64 {
    let t0 = Instant::now();

    let mut router = Router::new();
    router.begin_batch(); // Defer automaton rebuilds for speed
    let mut cluster_count = 0;
    // Map: cluster_id → list of ground truth labels assigned to it
    let mut cluster_labels: HashMap<String, Vec<String>> = HashMap::new();

    for (step, &idx) in indices.iter().enumerate() {
        let ex = &examples[idx];
        let query = &ex.text;
        let gt_label = &ex.intents[0];

        // Periodically flush batch to keep routing accurate
        if step > 0 && step % 500 == 0 {
            router.end_batch();
            router.begin_batch();
        }

        let results = router.route(query);

        if results.is_empty() || results[0].score < threshold {
            // New cluster
            let cluster_id = format!("cluster_{}", cluster_count);
            cluster_count += 1;
            router.add_intent(&cluster_id, &[query.as_str()]);
            cluster_labels.entry(cluster_id).or_default().push(gt_label.clone());
        } else {
            // Assign to existing cluster
            let cluster_id = results[0].id.clone();
            router.learn(query, &cluster_id);
            cluster_labels.entry(cluster_id).or_default().push(gt_label.clone());
        }
    }
    router.end_batch();

    let elapsed = t0.elapsed();

    // === Evaluation metrics ===

    // 1. Cluster purity: for each cluster, what fraction is the dominant label?
    let mut total_pure = 0usize;
    let mut total_assigned = 0usize;
    for (_cluster_id, labels) in &cluster_labels {
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        for l in labels {
            *label_counts.entry(l.as_str()).or_insert(0) += 1;
        }
        let max_count = label_counts.values().max().copied().unwrap_or(0);
        total_pure += max_count;
        total_assigned += labels.len();
    }
    let purity = total_pure as f32 / total_assigned as f32 * 100.0;

    // 2. Coverage: how many ground truth intents are represented as a majority in at least one cluster?
    let mut gt_covered: HashSet<String> = HashSet::new();
    for (_cluster_id, labels) in &cluster_labels {
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        for l in labels {
            *label_counts.entry(l.as_str()).or_insert(0) += 1;
        }
        if let Some((&dominant, _)) = label_counts.iter().max_by_key(|(_, &c)| c) {
            gt_covered.insert(dominant.to_string());
        }
    }
    let coverage = gt_covered.len() as f32 / gt_counts.len() as f32 * 100.0;

    // 3. Normalized Mutual Information (simplified)
    let nmi = compute_nmi(&cluster_labels, gt_counts, examples.len());

    println!(
        "  Threshold {:.1}: {} clusters | Purity {:.1}% | Coverage {}/{} ({:.0}%) | NMI {:.3} | {:.1}s",
        threshold,
        cluster_count,
        purity,
        gt_covered.len(),
        gt_counts.len(),
        coverage,
        nmi,
        elapsed.as_secs_f64(),
    );
    nmi
}

fn run_detailed(
    examples: &[Example],
    indices: &[usize],
    gt_counts: &HashMap<String, usize>,
    threshold: f32,
) {
    let mut router = Router::new();
    router.begin_batch();
    let mut cluster_count = 0;
    let mut cluster_labels: HashMap<String, Vec<String>> = HashMap::new();

    for (step, &idx) in indices.iter().enumerate() {
        let ex = &examples[idx];
        let query = &ex.text;
        let gt_label = &ex.intents[0];

        if step > 0 && step % 500 == 0 {
            router.end_batch();
            router.begin_batch();
            eprint!("\r  Processing: {}/{}", step, indices.len());
        }

        let results = router.route(query);

        if results.is_empty() || results[0].score < threshold {
            let cluster_id = format!("cluster_{}", cluster_count);
            cluster_count += 1;
            router.add_intent(&cluster_id, &[query.as_str()]);
            cluster_labels.entry(cluster_id).or_default().push(gt_label.clone());
        } else {
            let cluster_id = results[0].id.clone();
            router.learn(query, &cluster_id);
            cluster_labels.entry(cluster_id).or_default().push(gt_label.clone());
        }
    }
    router.end_batch();
    eprintln!("\r  Processing: done                ");

    // Show top clusters and their dominant labels
    let mut clusters: Vec<_> = cluster_labels.iter().collect();
    clusters.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    println!("  Top 30 clusters:\n");
    println!("  {:>10} {:>6} {:>7}  {}", "Cluster", "Size", "Purity", "Dominant Label (count) | Other labels");
    println!("  {}", "-".repeat(80));

    for &(cluster_id, labels) in clusters.iter().take(30) {
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        for l in labels.iter() {
            *label_counts.entry(l.as_str()).or_insert(0) += 1;
        }
        let mut sorted_labels: Vec<_> = label_counts.iter().collect();
        sorted_labels.sort_by(|a, b| b.1.cmp(a.1));

        let dominant = sorted_labels[0];
        let purity = *dominant.1 as f32 / labels.len() as f32 * 100.0;

        let others: Vec<String> = sorted_labels.iter().skip(1).take(3)
            .map(|(l, c)| format!("{}({})", l, c))
            .collect();
        let others_str = if others.is_empty() { "pure".to_string() } else { others.join(", ") };

        println!("  {:>10} {:>6} {:>6.0}%  {}({}) | {}",
            cluster_id, labels.len(), purity, dominant.0, dominant.1, others_str);
    }

    // Ground truth intent → which clusters captured it
    println!("\n  Ground truth intent → discovered clusters:\n");
    let mut gt_to_clusters: HashMap<String, Vec<(String, usize, usize)>> = HashMap::new(); // gt → [(cluster, count_in_cluster, cluster_size)]
    for (cluster_id, labels) in &cluster_labels {
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        for l in labels {
            *label_counts.entry(l.as_str()).or_insert(0) += 1;
        }
        for (&label, &count) in &label_counts {
            gt_to_clusters.entry(label.to_string()).or_default()
                .push((cluster_id.clone(), count, labels.len()));
        }
    }

    let mut gt_list: Vec<_> = gt_to_clusters.iter().collect();
    gt_list.sort_by(|a, b| a.0.cmp(b.0));

    for (gt_intent, clusters) in &gt_list {
        let gt_total = gt_counts.get(gt_intent.as_str()).copied().unwrap_or(0);
        let mut sorted_clusters: Vec<_> = clusters.iter().cloned().collect();
        sorted_clusters.sort_by(|a, b| b.1.cmp(&a.1));

        let primary = &sorted_clusters[0];
        let primary_pct = primary.1 as f32 / gt_total as f32 * 100.0;
        let fragmented = sorted_clusters.len();

        println!("  {:<35} → {} ({:.0}% captured in {}, {} fragments total)",
            gt_intent, primary.0, primary_pct, primary.0, fragmented);
    }

    // Summary stats
    let mut single_cluster_intents = 0;
    let mut majority_captured = 0;
    for (gt_intent, clusters) in &gt_list {
        let gt_total = gt_counts.get(gt_intent.as_str()).copied().unwrap_or(0);
        if clusters.len() == 1 {
            single_cluster_intents += 1;
        }
        let max_in_one = clusters.iter().map(|c| c.1).max().unwrap_or(0);
        if max_in_one as f32 / gt_total as f32 > 0.5 {
            majority_captured += 1;
        }
    }
    println!("\n  Intents captured in single cluster: {}/{}",
        single_cluster_intents, gt_counts.len());
    println!("  Intents with >50% in one cluster:  {}/{}",
        majority_captured, gt_counts.len());
}

/// Compute Normalized Mutual Information between cluster assignments and ground truth.
fn compute_nmi(
    cluster_labels: &HashMap<String, Vec<String>>,
    gt_counts: &HashMap<String, usize>,
    n: usize,
) -> f64 {
    let n_f = n as f64;

    // Mutual information
    let mut mi = 0.0f64;
    for (_cluster_id, labels) in cluster_labels {
        let cluster_size = labels.len() as f64;
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        for l in labels {
            *label_counts.entry(l.as_str()).or_insert(0) += 1;
        }
        for (&label, &count) in &label_counts {
            let gt_total = *gt_counts.get(label).unwrap_or(&1) as f64;
            let count_f = count as f64;
            if count_f > 0.0 {
                mi += (count_f / n_f) * (n_f * count_f / (cluster_size * gt_total)).ln();
            }
        }
    }

    // Entropy of clusters
    let mut h_clusters = 0.0f64;
    for (_cluster_id, labels) in cluster_labels {
        let p = labels.len() as f64 / n_f;
        if p > 0.0 {
            h_clusters -= p * p.ln();
        }
    }

    // Entropy of ground truth
    let mut h_gt = 0.0f64;
    for (_, &count) in gt_counts {
        let p = count as f64 / n_f;
        if p > 0.0 {
            h_gt -= p * p.ln();
        }
    }

    // NMI = 2 * MI / (H_clusters + H_gt)
    let denom = h_clusters + h_gt;
    if denom > 0.0 {
        2.0 * mi / denom
    } else {
        0.0
    }
}
