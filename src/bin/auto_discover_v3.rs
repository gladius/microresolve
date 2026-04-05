//! Auto-discovery v3: PMI Clustering + Post-Clustering Split
//!
//! Extends v2 (PMI anchor-based term clustering) with impure cluster detection
//! and splitting based on within-cluster mutual exclusivity.
//!
//! Key insight: v2 correctly merges "cancel order" and "track order" because they
//! share "order". But WITHIN that merged cluster, "cancel" and "track" never
//! co-occur. This mutual exclusivity is the signal to split.
//!
//! Algorithm:
//!   1-4. Same as v2: term stats, PMI, anchor clustering, query assignment
//!   5. NEW: Detect impure clusters (large clusters with mutually exclusive terms)
//!   6. NEW: Split impure clusters using within-cluster term exclusivity
//!   7. Evaluate against ground truth
//!
//! Run: cargo run --release --bin auto_discover_v3

use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(serde::Deserialize)]
struct Example {
    text: String,
    intents: Vec<String>,
}

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

/// Detect which clusters are impure based on within-cluster mutual exclusivity.
///
/// A cluster is impure if it contains discriminating terms that appear frequently
/// within the cluster but never co-occur with each other. These mutually exclusive
/// terms represent different intents merged into one cluster.
fn detect_impure_clusters(
    cluster_queries: &HashMap<usize, Vec<usize>>,
    query_terms: &[Vec<String>],
    term_to_cluster: &HashMap<String, usize>,
    discriminating: &HashSet<String>,
    min_cluster_size: usize,
) -> HashMap<usize, Vec<String>> {
    // Returns: cluster_id → list of "splitter" terms (mutually exclusive within cluster)
    let mut impure: HashMap<usize, Vec<String>> = HashMap::new();

    for (&cluster_id, query_indices) in cluster_queries {
        if query_indices.len() < min_cluster_size {
            continue;
        }

        // Find discriminating terms that appear in this cluster's queries
        // but are NOT the cluster's own anchor terms (i.e., unclustered or from other clusters)
        let mut term_freq: HashMap<String, usize> = HashMap::new();
        for &qi in query_indices {
            let unique: HashSet<&String> = query_terms[qi].iter().collect();
            for term in unique {
                if discriminating.contains(term) {
                    // Count terms that are either unclustered or belong to THIS cluster
                    // (terms assigned to other clusters are less interesting)
                    let assigned_cluster = term_to_cluster.get(term);
                    if assigned_cluster.is_none() || *assigned_cluster.unwrap() == cluster_id {
                        *term_freq.entry(term.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        // Keep terms that appear in at least 5% of the cluster's queries
        let min_freq = (query_indices.len() as f64 * 0.03).max(10.0) as usize;
        let mut candidate_terms: Vec<(String, usize)> = term_freq.into_iter()
            .filter(|(_, freq)| *freq >= min_freq)
            .collect();
        candidate_terms.sort_by(|a, b| b.1.cmp(&a.1));

        // Take top 20 candidates
        let candidates: Vec<String> = candidate_terms.iter()
            .take(20)
            .map(|(t, _)| t.clone())
            .collect();

        if candidates.len() < 3 {
            continue;
        }

        // Compute pairwise co-occurrence WITHIN this cluster
        let mut local_cooc: HashMap<(String, String), usize> = HashMap::new();
        let cand_set: HashSet<&String> = candidates.iter().collect();

        for &qi in query_indices {
            let unique_cands: HashSet<String> = query_terms[qi].iter()
                .filter(|t| cand_set.contains(t))
                .cloned()
                .collect();
            let unique_vec: Vec<&String> = unique_cands.iter().collect();
            for i in 0..unique_vec.len() {
                for j in (i + 1)..unique_vec.len() {
                    let (a, b) = if unique_vec[i] < unique_vec[j] {
                        (unique_vec[i].clone(), unique_vec[j].clone())
                    } else {
                        (unique_vec[j].clone(), unique_vec[i].clone())
                    };
                    *local_cooc.entry((a, b)).or_insert(0) += 1;
                }
            }
        }

        // Count mutually exclusive pairs (zero co-occurrence)
        let mut zero_pairs = 0;
        let mut total_pairs = 0;
        for i in 0..candidates.len() {
            for j in (i + 1)..candidates.len() {
                let (a, b) = if candidates[i] < candidates[j] {
                    (candidates[i].clone(), candidates[j].clone())
                } else {
                    (candidates[j].clone(), candidates[i].clone())
                };
                total_pairs += 1;
                let cooc = local_cooc.get(&(a, b)).copied().unwrap_or(0);
                if cooc == 0 {
                    zero_pairs += 1;
                }
            }
        }

        let exclusivity = if total_pairs > 0 { zero_pairs as f64 / total_pairs as f64 } else { 0.0 };

        if exclusivity > 0.3 {
            impure.insert(cluster_id, candidates);
        }
    }

    impure
}

/// Split an impure cluster using sub-anchor selection (recursive v2 within cluster).
///
/// Within the cluster, find candidate terms that are mutually exclusive
/// (rarely co-occur). These become "sub-anchors" — local cluster seeds.
/// Assign each query to the sub-anchor it matches best.
fn split_cluster(
    query_indices: &[usize],
    query_terms: &[Vec<String>],
    candidates: &[String],
    cluster_terms: &[String], // the original term cluster members (to exclude)
    df: &HashMap<String, usize>,
    n_docs: f64,
) -> Vec<Vec<usize>> {
    let cand_set: HashSet<&String> = candidates.iter().collect();
    let defining_terms: HashSet<&String> = cluster_terms.iter().collect();

    // Compute local frequency and pairwise co-occurrence within this cluster
    let mut local_freq: HashMap<String, usize> = HashMap::new();
    let mut local_cooc: HashMap<(String, String), usize> = HashMap::new();

    for &qi in query_indices {
        let query_cands: HashSet<String> = query_terms[qi].iter()
            .filter(|t| cand_set.contains(t))
            .cloned()
            .collect();
        for t in &query_cands {
            *local_freq.entry(t.clone()).or_insert(0) += 1;
        }
        let cands_vec: Vec<&String> = query_cands.iter().collect();
        for i in 0..cands_vec.len() {
            for j in (i + 1)..cands_vec.len() {
                let (a, b) = if cands_vec[i] < cands_vec[j] {
                    (cands_vec[i].clone(), cands_vec[j].clone())
                } else {
                    (cands_vec[j].clone(), cands_vec[i].clone())
                };
                *local_cooc.entry((a, b)).or_insert(0) += 1;
            }
        }
    }

    // Select sub-anchors: most frequent DIFFERENTIATING candidates that are mutually exclusive.
    // EXCLUDE defining terms (the cluster's original terms like "order", "account")
    // because they appear in ALL queries and can't differentiate.
    let exclusion_ratio = 0.05;
    let min_anchor_freq = (query_indices.len() as f64 * 0.05).max(15.0) as usize;

    // Sort candidates by local frequency (descending), excluding defining terms
    let mut ranked_candidates: Vec<(String, usize)> = candidates.iter()
        .filter(|t| !defining_terms.contains(t)) // EXCLUDE cluster's own terms
        .filter_map(|t| {
            let freq = local_freq.get(t).copied().unwrap_or(0);
            if freq >= min_anchor_freq { Some((t.clone(), freq)) } else { None }
        })
        .collect();
    ranked_candidates.sort_by(|a, b| b.1.cmp(&a.1));

    // Greedy sub-anchor selection: pick top-frequency terms that don't co-occur
    let mut sub_anchors: Vec<(String, usize)> = Vec::new();
    let max_sub_anchors = 8;

    for (term, freq) in &ranked_candidates {
        if sub_anchors.len() >= max_sub_anchors {
            break;
        }

        let mut exclusive_of_all = true;
        for (anchor, anchor_freq) in &sub_anchors {
            let (a, b) = if term < anchor {
                (term.clone(), anchor.clone())
            } else {
                (anchor.clone(), term.clone())
            };
            let cooc = local_cooc.get(&(a, b)).copied().unwrap_or(0);
            let min_f = (*freq).min(*anchor_freq);
            let threshold = (min_f as f64 * exclusion_ratio) as usize;
            if cooc > threshold {
                exclusive_of_all = false;
                break;
            }
        }

        if exclusive_of_all {
            sub_anchors.push((term.clone(), *freq));
        }
    }

    if sub_anchors.len() < 2 {
        return vec![query_indices.to_vec()]; // can't split
    }

    // For each sub-anchor, gather associated terms (terms that co-occur with this
    // anchor but not with other anchors)
    let mut anchor_terms: Vec<HashSet<String>> = vec![HashSet::new(); sub_anchors.len()];

    for (term, _) in &ranked_candidates {
        // Which sub-anchor does this term most co-occur with?
        let mut best_anchor = None;
        let mut best_cooc = 0usize;
        let mut second_best = 0usize;

        for (ai, (anchor, _)) in sub_anchors.iter().enumerate() {
            if term == anchor {
                best_anchor = Some(ai);
                best_cooc = usize::MAX;
                break;
            }
            let (a, b) = if term < anchor {
                (term.clone(), anchor.clone())
            } else {
                (anchor.clone(), term.clone())
            };
            let cooc = local_cooc.get(&(a, b)).copied().unwrap_or(0);
            if cooc > best_cooc {
                second_best = best_cooc;
                best_cooc = cooc;
                best_anchor = Some(ai);
            } else if cooc > second_best {
                second_best = cooc;
            }
        }

        // Only assign if clearly associated with one anchor (2x more than second best)
        if let Some(ai) = best_anchor {
            if best_cooc > second_best * 2 || best_cooc == usize::MAX {
                anchor_terms[ai].insert(term.clone());
            }
        }
    }

    // Assign queries to sub-anchors by IDF-weighted match
    let mut sub_clusters: Vec<Vec<usize>> = vec![Vec::new(); sub_anchors.len()];
    let mut residual: Vec<usize> = Vec::new();

    for &qi in query_indices {
        let terms: HashSet<&String> = query_terms[qi].iter().collect();

        let mut best_anchor = None;
        let mut best_score = 0.0f64;
        let mut second_score = 0.0f64;

        for (ai, a_terms) in anchor_terms.iter().enumerate() {
            let mut score = 0.0;
            for term in a_terms {
                if terms.contains(term) {
                    let idf = (n_docs / *df.get(term).unwrap_or(&1) as f64).ln();
                    score += idf;
                }
            }
            if score > best_score {
                second_score = best_score;
                best_score = score;
                best_anchor = Some(ai);
            } else if score > second_score {
                second_score = score;
            }
        }

        if let Some(ai) = best_anchor {
            if best_score > 0.0 {
                sub_clusters[ai].push(qi);
            } else {
                residual.push(qi);
            }
        } else {
            residual.push(qi);
        }
    }

    // Assign residual to largest sub-cluster
    if !residual.is_empty() {
        let largest = sub_clusters.iter().enumerate()
            .max_by_key(|(_, sc)| sc.len())
            .map(|(i, _)| i)
            .unwrap_or(0);
        sub_clusters[largest].extend(residual);
    }

    // Filter tiny sub-clusters — require at least 15% of parent to avoid fragmentation
    let min_size = (query_indices.len() as f64 * 0.15).max(30.0) as usize;
    let mut final_clusters: Vec<Vec<usize>> = Vec::new();
    let mut tiny: Vec<usize> = Vec::new();

    for sc in sub_clusters {
        if sc.len() >= min_size {
            final_clusters.push(sc);
        } else {
            tiny.extend(sc);
        }
    }

    if !tiny.is_empty() {
        if final_clusters.is_empty() {
            final_clusters.push(tiny);
        } else {
            final_clusters[0].extend(tiny);
        }
    }

    if final_clusters.len() < 2 {
        return vec![query_indices.to_vec()];
    }

    final_clusters
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Auto-Discovery v3: PMI + Post-Clustering Split        ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let data = std::fs::read_to_string("tests/data/benchmarks/bitext_all.json")
        .expect("Run download_datasets.py first");
    let examples: Vec<Example> = serde_json::from_str(&data).unwrap();

    println!("Dataset: Bitext (26872 queries, 27 ground truth intents)\n");

    let t0 = Instant::now();

    // === Steps 1-2: Same as v2 — term stats + PMI ===
    println!("Step 1: Building term statistics...");

    let n_docs = examples.len() as f64;
    let mut df: HashMap<String, usize> = HashMap::new();
    let mut cooc: HashMap<(String, String), usize> = HashMap::new();
    let mut query_terms: Vec<Vec<String>> = Vec::new();

    for ex in &examples {
        let terms: Vec<String> = tokenize_simple(&ex.text);
        let unique: HashSet<String> = terms.iter().cloned().collect();

        for t in &unique {
            *df.entry(t.clone()).or_insert(0) += 1;
        }

        let unique_vec: Vec<&String> = unique.iter().collect();
        for i in 0..unique_vec.len() {
            for j in (i + 1)..unique_vec.len() {
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

    let min_df = (n_docs * 0.001).max(5.0) as usize;
    let max_df = (n_docs * 0.20) as usize;
    let discriminating: HashSet<String> = df.iter()
        .filter(|(_, &d)| d >= min_df && d <= max_df)
        .map(|(t, _)| t.clone())
        .collect();

    println!("  Total unique terms: {}", df.len());
    println!("  Discriminating terms (df {}-{}): {}", min_df, max_df, discriminating.len());

    // === Step 2: PMI ===
    println!("\nStep 2: Computing PMI...");

    let mut pmi_map: HashMap<(String, String), f64> = HashMap::new();

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

        if pmi > 1.0 && count >= 10 {
            pmi_map.insert((a.clone(), b.clone()), pmi);
            pmi_map.insert((b.clone(), a.clone()), pmi);
        }
    }

    println!("  Strong PMI edges: {}", pmi_map.len() / 2);

    // === Step 3: Anchor-based clustering (same as v2) ===
    println!("\nStep 3: Anchor-based term clustering...");

    let mut term_idf: Vec<(String, f64)> = discriminating.iter()
        .map(|t| {
            let d = *df.get(t).unwrap() as f64;
            let idf = (n_docs / d).ln();
            (t.clone(), idf)
        })
        .collect();
    term_idf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut anchors: Vec<String> = Vec::new();
    let max_anchors = 40;

    for (term, _idf) in &term_idf {
        if anchors.len() >= max_anchors {
            break;
        }
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

    let mut term_clusters: Vec<Vec<String>> = vec![vec![]; anchors.len()];
    let mut term_to_cluster: HashMap<String, usize> = HashMap::new();

    for (i, anchor) in anchors.iter().enumerate() {
        term_clusters[i].push(anchor.clone());
        term_to_cluster.insert(anchor.clone(), i);
    }

    for term in &discriminating {
        if term_to_cluster.contains_key(term) {
            continue;
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
        if best_pmi > 0.5 {
            term_clusters[best_anchor].push(term.clone());
            term_to_cluster.insert(term.clone(), best_anchor);
        }
    }

    let term_clusters: Vec<Vec<String>> = term_clusters.into_iter()
        .filter(|c| c.len() >= 2)
        .collect();

    let mut term_to_cluster: HashMap<String, usize> = HashMap::new();
    for (cluster_id, terms) in term_clusters.iter().enumerate() {
        for term in terms {
            term_to_cluster.insert(term.clone(), cluster_id);
        }
    }

    println!("  Non-trivial term clusters: {}", term_clusters.len());

    // === Step 4: Assign queries to clusters (same as v2) ===
    println!("\nStep 4: Assigning queries to term clusters...");

    let mut cluster_queries: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut unassigned_queries: Vec<usize> = Vec::new();

    for (qi, _ex) in examples.iter().enumerate() {
        let terms = &query_terms[qi];
        let mut votes: HashMap<usize, f64> = HashMap::new();
        for term in terms {
            if let Some(&cluster_id) = term_to_cluster.get(term) {
                let idf = (n_docs / *df.get(term).unwrap_or(&1) as f64).ln();
                *votes.entry(cluster_id).or_insert(0.0) += idf;
            }
        }

        if let Some((&best_cluster, _)) = votes.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            cluster_queries.entry(best_cluster).or_default().push(qi);
        } else {
            unassigned_queries.push(qi);
        }
    }

    let assigned_count: usize = cluster_queries.values().map(|v| v.len()).sum();
    println!("  Assigned: {} | Unassigned: {}", assigned_count, unassigned_queries.len());

    // === Step 5: Detect impure clusters (NEW) ===
    println!("\nStep 5: Detecting impure clusters...");

    let impure = detect_impure_clusters(
        &cluster_queries,
        &query_terms,
        &term_to_cluster,
        &discriminating,
        500, // only split large clusters — small ones fragment too much
    );

    println!("  Impure clusters detected: {}", impure.len());
    for (&cluster_id, candidates) in &impure {
        let size = cluster_queries.get(&cluster_id).map(|v| v.len()).unwrap_or(0);
        let terms: Vec<&str> = term_clusters.get(cluster_id)
            .map(|c| c.iter().take(4).map(|s| s.as_str()).collect())
            .unwrap_or_default();
        let splitters: Vec<&str> = candidates.iter().take(8).map(|s| s.as_str()).collect();
        println!("    Cluster {} ({} queries, terms {:?})", cluster_id, size, terms);
        println!("      Splitter candidates: {:?}", splitters);
    }

    // === Step 6: Split impure clusters (NEW) ===
    println!("\nStep 6: Splitting impure clusters...");

    let mut final_cluster_queries: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut next_id = term_clusters.len(); // start new IDs after existing ones

    for (&cluster_id, query_indices) in &cluster_queries {
        if let Some(candidates) = impure.get(&cluster_id) {
            let empty_terms = Vec::new();
            let cluster_own_terms = if cluster_id < term_clusters.len() {
                &term_clusters[cluster_id]
            } else {
                &empty_terms
            };
            let sub_clusters = split_cluster(
                query_indices,
                &query_terms,
                candidates,
                cluster_own_terms,
                &df,
                n_docs,
            );

            if sub_clusters.len() > 1 {
                println!("    Cluster {} → split into {} sub-clusters (sizes: {:?})",
                    cluster_id,
                    sub_clusters.len(),
                    sub_clusters.iter().map(|sc| sc.len()).collect::<Vec<_>>());

                // First sub-cluster keeps original ID
                final_cluster_queries.insert(cluster_id, sub_clusters[0].clone());
                // Additional sub-clusters get new IDs
                for sc in sub_clusters.iter().skip(1) {
                    final_cluster_queries.insert(next_id, sc.clone());
                    next_id += 1;
                }
            } else {
                final_cluster_queries.insert(cluster_id, query_indices.clone());
            }
        } else {
            final_cluster_queries.insert(cluster_id, query_indices.clone());
        }
    }

    // Add unassigned to a special cluster if non-empty
    if !unassigned_queries.is_empty() {
        final_cluster_queries.insert(next_id, unassigned_queries);
    }

    println!("  Total clusters after split: {}", final_cluster_queries.len());

    // === Step 7: Evaluate ===
    println!("\nStep 7: Evaluation\n");

    let elapsed = t0.elapsed();

    // Convert to label-based format for evaluation
    let mut cluster_labels: HashMap<usize, Vec<String>> = HashMap::new();
    for (&cluster_id, query_indices) in &final_cluster_queries {
        for &qi in query_indices {
            cluster_labels.entry(cluster_id).or_default().push(examples[qi].intents[0].clone());
        }
    }

    // Ground truth
    let mut gt_counts: HashMap<String, usize> = HashMap::new();
    for ex in &examples {
        *gt_counts.entry(ex.intents[0].clone()).or_insert(0) += 1;
    }

    // Purity
    let mut total_pure = 0usize;
    let mut total_assigned = 0usize;
    for labels in cluster_labels.values() {
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
    let mut gt_covered: HashSet<String> = HashSet::new();
    for labels in cluster_labels.values() {
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        for l in labels {
            *label_counts.entry(l.as_str()).or_insert(0) += 1;
        }
        if let Some((&dominant, &count)) = label_counts.iter().max_by_key(|(_, &c)| c) {
            if count as f32 / labels.len() as f32 > 0.3 {
                gt_covered.insert(dominant.to_string());
            }
        }
    }

    // NMI
    let nmi = compute_nmi(&cluster_labels, &gt_counts, examples.len());

    println!("  ┌─────────────────────────────────────────────────────┐");
    println!("  │ v3 Results (PMI + Post-Clustering Split)           │");
    println!("  ├─────────────────────────────────────────────────────┤");
    println!("  │ Clusters discovered:  {:>4}                         │", cluster_labels.len());
    println!("  │ Ground truth intents: {:>4}                         │", gt_counts.len());
    println!("  │ Purity:               {:>5.1}%                       │", purity);
    println!("  │ Coverage:             {:>2}/{:>2} ({:>3.0}%)                  │",
        gt_covered.len(), gt_counts.len(),
        gt_covered.len() as f32 / gt_counts.len() as f32 * 100.0);
    println!("  │ NMI:                  {:.3}                        │", nmi);
    println!("  │ Time:                 {:.2}s                        │", elapsed.as_secs_f64());
    println!("  └─────────────────────────────────────────────────────┘");

    println!("\n  Comparison:");
    println!("  ┌──────────────┬──────────┬──────────┐");
    println!("  │ Metric       │  v2      │  v3      │");
    println!("  ├──────────────┼──────────┼──────────┤");
    println!("  │ NMI          │  0.589   │  {:.3}   │", nmi);
    println!("  │ Purity       │  48.4%   │  {:>5.1}%  │", purity);
    println!("  │ Clusters     │  27      │  {:>4}    │", cluster_labels.len());
    println!("  └──────────────┴──────────┴──────────┘");

    // Per-cluster breakdown
    println!("\n  Top clusters:");
    println!("  {:>5} {:>6} {:>7}  {}", "ID", "Size", "Purity", "Dominant label (count) | Others");
    println!("  {}", "-".repeat(90));

    let mut sorted_clusters: Vec<_> = cluster_labels.iter().collect();
    sorted_clusters.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for &(cluster_id, labels) in sorted_clusters.iter().take(35) {
        let mut label_counts: HashMap<&str, usize> = HashMap::new();
        for l in labels.iter() {
            *label_counts.entry(l.as_str()).or_insert(0) += 1;
        }
        let mut sorted_labels: Vec<_> = label_counts.iter().collect();
        sorted_labels.sort_by(|a, b| b.1.cmp(a.1));
        let dominant = sorted_labels[0];
        let purity_pct = *dominant.1 as f32 / labels.len() as f32 * 100.0;

        let others: Vec<String> = sorted_labels.iter().skip(1).take(3)
            .map(|(l, c)| format!("{}({})", l, c))
            .collect();
        let others_str = if others.is_empty() { "pure".to_string() } else { others.join(", ") };

        println!("  {:>5} {:>6} {:>6.0}%  {}({}) | {}",
            cluster_id, labels.len(), purity_pct, dominant.0, dominant.1, others_str);
    }

    // Per ground-truth intent
    println!("\n  Ground truth → best cluster:");
    let mut gt_to_best: Vec<(String, usize, f32, usize)> = Vec::new();
    for (gt_intent, &gt_total) in &gt_counts {
        let mut best_cluster = 0;
        let mut best_count = 0;
        let mut fragments = 0;
        for (&cluster_id, labels) in &cluster_labels {
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
    println!("\n  Intents with >50% in one cluster: {}/{} (v2: 16/27)", gt_majority, gt_counts.len());
}

fn compute_nmi(
    cluster_labels: &HashMap<usize, Vec<String>>,
    gt_counts: &HashMap<String, usize>,
    n: usize,
) -> f64 {
    let n_f = n as f64;
    let mut mi = 0.0f64;
    for labels in cluster_labels.values() {
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
    for labels in cluster_labels.values() {
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
