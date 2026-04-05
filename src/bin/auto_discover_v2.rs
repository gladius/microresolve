//! Auto-discovery v2: PMI-based term clustering → intent discovery
//!
//! Key insight: cluster TERMS first, then group queries by their term cluster.
//!
//! Algorithm:
//! 1. Tokenize all queries → term frequency + co-occurrence matrix
//! 2. Compute PMI between all term pairs
//! 3. Cluster terms into groups (high PMI = same group)
//! 4. Each query gets assigned to the term-cluster that best represents it
//! 5. Term clusters ≈ discovered intents
//!
//! Additional signal: IDF-weighted terms. Common terms ("my", "order") are
//! ignored for clustering. Only discriminating terms drive assignments.
//!
//! Run: cargo run --release --bin auto_discover_v2

use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(serde::Deserialize)]
struct Example {
    text: String,
    intents: Vec<String>,
}

/// Simple tokenizer matching ASV's approach
fn tokenize_simple(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| w.len() >= 2)
        .filter(|w| !is_stop(w))
        .map(|w| w.to_string())
        .collect()
}

fn is_stop(word: &str) -> bool {
    matches!(word, "i" | "me" | "my" | "we" | "our" | "you" | "your" | "he" | "she" |
        "it" | "its" | "they" | "them" | "their" | "what" | "which" | "who" |
        "this" | "that" | "these" | "those" | "is" | "am" | "are" | "was" |
        "were" | "be" | "been" | "being" | "have" | "has" | "had" | "do" |
        "does" | "did" | "will" | "would" | "could" | "should" | "may" |
        "might" | "shall" | "can" | "need" | "to" | "of" | "in" | "for" |
        "on" | "with" | "at" | "by" | "from" | "as" | "into" | "about" |
        "the" | "a" | "an" | "and" | "but" | "or" | "if" | "so" | "not" |
        "no" | "nor" | "too" | "very" | "just" | "don't" | "didn't" |
        "isn't" | "aren't" | "wasn't" | "weren't" | "hasn't" | "haven't" |
        "won't" | "wouldn't" | "couldn't" | "shouldn't" | "can't" |
        "there" | "here" | "how" | "when" | "where" | "why" | "all" |
        "each" | "every" | "both" | "some" | "any" | "such" | "only" |
        "also" | "than" | "like" | "get" | "got" | "make" | "know" |
        "want" | "help" | "please" | "thanks" | "thank" | "hi" | "hello")
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Auto-Discovery v2: PMI Term Clustering → Intents      ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let data = std::fs::read_to_string("tests/data/benchmarks/bitext_all.json")
        .expect("Run download_datasets.py first");
    let examples: Vec<Example> = serde_json::from_str(&data).unwrap();

    println!("Dataset: Bitext (26872 queries, 27 ground truth intents)\n");

    let t0 = Instant::now();

    // === Step 1: Build term statistics ===
    println!("Step 1: Building term statistics...");

    let n_docs = examples.len() as f64;
    let mut df: HashMap<String, usize> = HashMap::new(); // document frequency
    let mut cooc: HashMap<(String, String), usize> = HashMap::new(); // co-occurrence
    let mut query_terms: Vec<Vec<String>> = Vec::new();

    for ex in &examples {
        let terms: Vec<String> = tokenize_simple(&ex.text);
        let unique: HashSet<String> = terms.iter().cloned().collect();

        for t in &unique {
            *df.entry(t.clone()).or_insert(0) += 1;
        }

        // Co-occurrence: pairs of terms in same query
        let unique_vec: Vec<&String> = unique.iter().collect();
        for i in 0..unique_vec.len() {
            for j in (i+1)..unique_vec.len() {
                let (a, b) = if unique_vec[i] < unique_vec[j] {
                    (unique_vec[i].clone(), unique_vec[j].clone())
                } else {
                    (unique_vec[j].clone(), unique_vec[i].clone())
                };
                *cooc.entry((a, b)).or_insert(0) += 1;
            }
        }
        query_terms.push(terms);
    }

    // Filter to discriminating terms: not too common, not too rare
    // IDF-based: keep terms with df between 0.1% and 20% of docs
    let min_df = (n_docs * 0.001).max(5.0) as usize;
    let max_df = (n_docs * 0.20) as usize;
    let discriminating: HashSet<String> = df.iter()
        .filter(|(_, &d)| d >= min_df && d <= max_df)
        .map(|(t, _)| t.clone())
        .collect();

    println!("  Total unique terms: {}", df.len());
    println!("  Discriminating terms (df {}-{}): {}", min_df, max_df, discriminating.len());

    // === Step 2: Compute PMI between discriminating terms ===
    println!("\nStep 2: Computing PMI...");

    let mut pmi_edges: Vec<(String, String, f64)> = Vec::new();

    for ((a, b), &count) in &cooc {
        if !discriminating.contains(a) || !discriminating.contains(b) {
            continue;
        }
        let df_a = *df.get(a).unwrap() as f64;
        let df_b = *df.get(b).unwrap() as f64;
        let p_ab = count as f64 / n_docs;
        let p_a = df_a / n_docs;
        let p_b = df_b / n_docs;

        let pmi = (p_ab / (p_a * p_b)).ln();

        // Only keep strong positive associations
        // Also require minimum co-occurrence count for reliability
        if pmi > 1.0 && count >= 10 {
            pmi_edges.push((a.clone(), b.clone(), pmi));
        }
    }

    pmi_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    println!("  Strong PMI edges (PMI > 1.0, count >= 10): {}", pmi_edges.len());

    // Show top PMI pairs
    println!("\n  Top 20 PMI pairs:");
    for (a, b, pmi) in pmi_edges.iter().take(20) {
        let count = cooc.get(&(a.clone(), b.clone()))
            .or_else(|| cooc.get(&(b.clone(), a.clone())))
            .unwrap_or(&0);
        println!("    {:<20} + {:<20} PMI={:.2} (n={})", a, b, pmi, count);
    }

    // === Step 3: Anchor-based term clustering ===
    // Strategy: find "anchor terms" that are high-IDF and mutually exclusive (low/negative PMI).
    // Then assign all other terms to their nearest anchor by PMI affinity.
    println!("\nStep 3: Anchor-based term clustering...");

    // Build PMI lookup
    let mut pmi_map: HashMap<(String, String), f64> = HashMap::new();
    for (a, b, pmi) in &pmi_edges {
        pmi_map.insert((a.clone(), b.clone()), *pmi);
        pmi_map.insert((b.clone(), a.clone()), *pmi);
    }

    // Rank discriminating terms by IDF (most distinctive first)
    let mut term_idf: Vec<(String, f64)> = discriminating.iter()
        .map(|t| {
            let d = *df.get(t).unwrap() as f64;
            let idf = (n_docs / d).ln();
            (t.clone(), idf)
        })
        .collect();
    term_idf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Greedy anchor selection: pick top-IDF terms that are NOT strongly co-occurring with existing anchors
    let mut anchors: Vec<String> = Vec::new();
    let max_anchors = 40; // generous upper bound; some will merge later

    for (term, _idf) in &term_idf {
        if anchors.len() >= max_anchors {
            break;
        }

        // Check: is this term strongly associated with any existing anchor?
        let mut too_close = false;
        for anchor in &anchors {
            let pmi_val = pmi_map.get(&(term.clone(), anchor.clone())).copied().unwrap_or(-10.0);
            if pmi_val > 2.0 {
                too_close = true;
                break;
            }
        }

        if !too_close {
            anchors.push(term.clone());
        }
    }

    println!("  Anchors selected: {}", anchors.len());
    println!("  Anchor terms: {:?}", &anchors.iter().take(20).map(|s| s.as_str()).collect::<Vec<_>>());

    // Assign every discriminating term to its nearest anchor by PMI
    let mut term_clusters: Vec<Vec<String>> = vec![vec![]; anchors.len()];
    let mut term_to_cluster_map: HashMap<String, usize> = HashMap::new();

    // Each anchor belongs to its own cluster
    for (i, anchor) in anchors.iter().enumerate() {
        term_clusters[i].push(anchor.clone());
        term_to_cluster_map.insert(anchor.clone(), i);
    }

    // Assign remaining terms
    for term in &discriminating {
        if term_to_cluster_map.contains_key(term) {
            continue;  // already an anchor
        }
        let mut best_anchor = 0usize;
        let mut best_pmi = f64::NEG_INFINITY;
        for (i, anchor) in anchors.iter().enumerate() {
            let pmi_val = pmi_map.get(&(term.clone(), anchor.clone())).copied().unwrap_or(-10.0);
            if pmi_val > best_pmi {
                best_pmi = pmi_val;
                best_anchor = i;
            }
        }
        if best_pmi > 0.5 {  // Only assign if there's meaningful affinity
            term_clusters[best_anchor].push(term.clone());
            term_to_cluster_map.insert(term.clone(), best_anchor);
        }
    }

    // Remove empty clusters
    let term_clusters: Vec<Vec<String>> = term_clusters.into_iter()
        .filter(|c| c.len() >= 2)
        .collect();

    println!("  Non-trivial term clusters: {}", term_clusters.len());
    println!("\n  Term clusters (top 30):");
    for (i, cluster) in term_clusters.iter().enumerate().take(30) {
        let display: Vec<&str> = cluster.iter().take(10).map(|s| s.as_str()).collect();
        let suffix = if cluster.len() > 10 { format!(" +{} more", cluster.len() - 10) } else { String::new() };
        println!("    Cluster {:>2} ({:>3} terms): {:?}{}", i, cluster.len(), display, suffix);
    }

    // === Step 4: Rebuild term → cluster_id mapping (after empty removal) ===
    let mut term_to_cluster: HashMap<String, usize> = HashMap::new();
    for (cluster_id, terms) in term_clusters.iter().enumerate() {
        for term in terms {
            term_to_cluster.insert(term.clone(), cluster_id);
        }
    }

    // === Step 5: Assign queries to clusters by voting ===
    println!("\nStep 4: Assigning queries to term clusters...");

    let mut query_assignments: Vec<Option<usize>> = Vec::new();
    let mut cluster_query_labels: HashMap<usize, Vec<String>> = HashMap::new();
    let mut unassigned = 0;

    for (qi, ex) in examples.iter().enumerate() {
        let terms = &query_terms[qi];

        // Vote: which term cluster has the most discriminating terms in this query?
        let mut votes: HashMap<usize, f64> = HashMap::new();
        for term in terms {
            if let Some(&cluster_id) = term_to_cluster.get(term) {
                let idf = (n_docs / *df.get(term).unwrap_or(&1) as f64).ln();
                *votes.entry(cluster_id).or_insert(0.0) += idf;
            }
        }

        if let Some((&best_cluster, _)) = votes.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            query_assignments.push(Some(best_cluster));
            cluster_query_labels.entry(best_cluster).or_default().push(ex.intents[0].clone());
        } else {
            query_assignments.push(None);
            unassigned += 1;
        }
    }

    println!("  Assigned: {} | Unassigned: {}", examples.len() - unassigned, unassigned);

    // === Step 6: Evaluate against ground truth ===
    println!("\nStep 5: Evaluation\n");

    let elapsed = t0.elapsed();

    // Purity
    let mut total_pure = 0usize;
    let mut total_assigned = 0usize;
    for (_cluster_id, labels) in &cluster_query_labels {
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        for l in labels {
            *label_counts.entry(l.as_str()).or_insert(0) += 1;
        }
        let max_count = label_counts.values().max().copied().unwrap_or(0);
        total_pure += max_count;
        total_assigned += labels.len();
    }
    let purity = if total_assigned > 0 { total_pure as f32 / total_assigned as f32 * 100.0 } else { 0.0 };

    // Coverage
    let mut gt_counts: HashMap<String, usize> = HashMap::new();
    for ex in &examples {
        *gt_counts.entry(ex.intents[0].clone()).or_insert(0) += 1;
    }

    let mut gt_covered: HashSet<String> = HashSet::new();
    for (_cluster_id, labels) in &cluster_query_labels {
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        for l in labels {
            *label_counts.entry(l.as_str()).or_insert(0) += 1;
        }
        if let Some((&dominant, &count)) = label_counts.iter().max_by_key(|(_, &c)| c) {
            if count as f32 / labels.len() as f32 > 0.3 { // At least 30% pure
                gt_covered.insert(dominant.to_string());
            }
        }
    }

    // NMI
    let nmi = compute_nmi(&cluster_query_labels, &gt_counts, examples.len());

    println!("  Clusters discovered:  {}", cluster_query_labels.len());
    println!("  Ground truth intents: {}", gt_counts.len());
    println!("  Purity:               {:.1}%", purity);
    println!("  Coverage:             {}/{} ({:.0}%)", gt_covered.len(), gt_counts.len(),
        gt_covered.len() as f32 / gt_counts.len() as f32 * 100.0);
    println!("  NMI:                  {:.3}", nmi);
    println!("  Time:                 {:.2}s", elapsed.as_secs_f64());

    // Per-cluster breakdown
    println!("\n  Top clusters:");
    println!("  {:>5} {:>6} {:>7}  {}", "ID", "Size", "Purity", "Dominant label | Terms");
    println!("  {}", "-".repeat(90));

    let mut sorted_clusters: Vec<_> = cluster_query_labels.iter().collect();
    sorted_clusters.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for &(cluster_id, labels) in sorted_clusters.iter().take(30) {
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        for l in labels.iter() {
            *label_counts.entry(l.as_str()).or_insert(0) += 1;
        }
        let mut sorted_labels: Vec<_> = label_counts.iter().collect();
        sorted_labels.sort_by(|a, b| b.1.cmp(a.1));
        let dominant = sorted_labels[0];
        let purity_pct = *dominant.1 as f32 / labels.len() as f32 * 100.0;

        let terms: Vec<&str> = term_clusters[*cluster_id].iter().take(6).map(|s: &String| s.as_str()).collect();

        println!("  {:>5} {:>6} {:>6.0}%  {}({}) | {:?}",
            cluster_id, labels.len(), purity_pct, dominant.0, dominant.1, terms);
    }

    // Per ground-truth intent breakdown
    println!("\n  Ground truth → best cluster:");
    let mut gt_to_best: Vec<(String, usize, f32, usize)> = Vec::new(); // (intent, best_cluster, capture_pct, fragments)
    for (gt_intent, &gt_total) in &gt_counts {
        let mut best_cluster = 0;
        let mut best_count = 0;
        let mut fragments = 0;
        for (&cluster_id, labels) in &cluster_query_labels {
            let count = labels.iter().filter(|l| l.as_str() == gt_intent.as_str()).count();
            if count > 0 {
                fragments += 1;
                if count > best_count {
                    best_count = count;
                    best_cluster = cluster_id;
                }
            }
        }
        let capture = best_count as f32 / gt_total as f32 * 100.0;
        gt_to_best.push((gt_intent.clone(), best_cluster, capture, fragments));
    }
    gt_to_best.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    let mut gt_majority = 0;
    for (intent, cluster, capture, fragments) in &gt_to_best {
        if *capture > 50.0 { gt_majority += 1; }
        println!("    {:<35} → cluster {:>3} ({:.0}% captured, {} fragments)", intent, cluster, capture, fragments);
    }
    println!("\n  Intents with >50% in one cluster: {}/{}", gt_majority, gt_counts.len());
}

fn compute_nmi(
    cluster_labels: &HashMap<usize, Vec<String>>,
    gt_counts: &HashMap<String, usize>,
    n: usize,
) -> f64 {
    let n_f = n as f64;
    let mut mi = 0.0f64;
    for (_cluster_id, labels) in cluster_labels.iter() {
        let cluster_size = labels.len() as f64;
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        for l in labels.iter() {
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
    let mut h_clusters = 0.0f64;
    for (_cluster_id, labels) in cluster_labels {
        let p = labels.len() as f64 / n_f;
        if p > 0.0 { h_clusters -= p * p.ln(); }
    }
    let mut h_gt = 0.0f64;
    for (_, &count) in gt_counts {
        let p = count as f64 / n_f;
        if p > 0.0 { h_gt -= p * p.ln(); }
    }
    let denom = h_clusters + h_gt;
    if denom > 0.0 { 2.0 * mi / denom } else { 0.0 }
}
