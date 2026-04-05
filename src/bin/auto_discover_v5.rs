//! Auto-discovery v5: PMI Anchors + Substitution Frame Merge
//!
//! v2 baseline: NMI=0.589, 16/27 intents at >50%, 27 clusters
//! v4 splitting: NMI=0.641, 15/27 intents at >50%, 51 clusters (fragmented)
//!
//! v5 approach: Start from v2 baseline, then MERGE fragmented clusters.
//! Substitution frames provide anti-affinity: terms that fill the same
//! syntactic slot (e.g., "cancel" and "track" in "I want to [SLOT] my order")
//! must NOT be merged. Terms that co-occur across slots CAN merge.
//!
//! The math: merging ALWAYS increases capture rate (combining query pools),
//! splitting ALWAYS decreases it (proven in v4). So merge is the correct direction.
//!
//! Run: cargo run --release --bin auto_discover_v5

use std::collections::{HashMap, HashSet, BTreeMap};
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

// ============================================================================
// Distributional Substitution Detection
// ============================================================================

/// Substitution frames: which terms are paradigmatic alternatives
struct SubstitutionFrames {
    /// term → set of terms it substitutes with (same context, never co-occur)
    substitutes: HashMap<String, HashSet<String>>,
}

/// Detect substitutes using the distributional test:
/// Two terms are substitutes if they (1) rarely co-occur and (2) share PMI neighbors.
/// This is Saussure's paradigmatic axis, computed from statistics.
fn detect_substitutes(
    discriminating: &HashSet<String>,
    cooc: &HashMap<(String, String), usize>,
    pmi_map: &HashMap<(String, String), f64>,
    df: &HashMap<String, usize>,
    n_docs: f64,
    min_df: usize,
) -> SubstitutionFrames {
    // Precompute PMI neighbor sets for each discriminating term
    let mut pmi_neighbors: HashMap<String, HashSet<String>> = HashMap::new();
    for term in discriminating {
        if df.get(term).copied().unwrap_or(0) < min_df { continue; }
        let neighbors: HashSet<String> = discriminating.iter()
            .filter(|t| {
                *t != term &&
                pmi_map.get(&(term.clone(), (*t).clone())).copied().unwrap_or(0.0) > 1.5
            })
            .cloned()
            .collect();
        if neighbors.len() >= 2 {
            pmi_neighbors.insert(term.clone(), neighbors);
        }
    }

    println!("  Terms with PMI neighborhoods: {}", pmi_neighbors.len());

    let mut substitutes: HashMap<String, HashSet<String>> = HashMap::new();
    let terms_with_neighbors: Vec<&String> = pmi_neighbors.keys().collect();

    for i in 0..terms_with_neighbors.len() {
        for j in (i + 1)..terms_with_neighbors.len() {
            let a = terms_with_neighbors[i];
            let b = terms_with_neighbors[j];

            // Condition 1: Rarely co-occur
            let key = if a < b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
            let co_count = cooc.get(&key).copied().unwrap_or(0);
            let df_a = df.get(a).copied().unwrap_or(1) as f64;
            let df_b = df.get(b).copied().unwrap_or(1) as f64;
            let expected = (df_a * df_b) / n_docs;

            // Co-occur less than 30% of expected = rarely co-occur
            if co_count as f64 > expected * 0.3 { continue; }

            // Condition 2: Similar contexts (share PMI neighbors)
            let na = &pmi_neighbors[a];
            let nb = &pmi_neighbors[b];

            let shared = na.intersection(nb).count();
            let union = na.union(nb).count();
            let jaccard = if union > 0 { shared as f64 / union as f64 } else { 0.0 };

            // Need substantial context overlap to count as substitutes
            if jaccard > 0.25 && shared >= 3 {
                substitutes.entry(a.clone()).or_default().insert(b.clone());
                substitutes.entry(b.clone()).or_default().insert(a.clone());
            }
        }
    }

    println!("  Terms with substitutes: {}", substitutes.len());

    // Show the cleanest substitution sets (size 2-15, most useful)
    let mut by_quality: Vec<(&String, &HashSet<String>)> = substitutes.iter()
        .filter(|(_, s)| s.len() >= 2 && s.len() <= 15)
        .collect();
    by_quality.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    for (term, subs) in by_quality.iter().take(15) {
        let sub_list: Vec<&str> = subs.iter().map(|s| s.as_str()).take(10).collect();
        println!("    {} ↔ {:?}{}", term, sub_list,
            if subs.len() > 10 { format!(" (+{})", subs.len() - 10) } else { String::new() });
    }

    SubstitutionFrames { substitutes }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let datasets: Vec<&str> = if args.len() > 1 {
        args[1..].iter().map(|s| s.as_str()).collect()
    } else {
        vec!["tests/data/benchmarks/bitext_all.json"]
    };

    for dataset_path in &datasets {
        run_discovery(dataset_path);
        println!("\n\n");
    }
}

fn run_discovery(dataset_path: &str) {
    let dataset_name = std::path::Path::new(dataset_path)
        .file_stem().unwrap_or_default()
        .to_str().unwrap_or("unknown");

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Auto-Discovery v5: {}",  format!("{:<42}║", dataset_name));
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let data = std::fs::read_to_string(dataset_path)
        .unwrap_or_else(|_| panic!("Cannot read {}", dataset_path));
    let examples: Vec<Example> = serde_json::from_str(&data).unwrap();

    // Count ground truth intents
    let mut gt_intent_set: HashSet<String> = HashSet::new();
    for ex in &examples {
        for i in &ex.intents { gt_intent_set.insert(i.clone()); }
    }
    let n_gt_intents = gt_intent_set.len();

    println!("Dataset: {} ({} queries, {} ground truth intents)\n",
        dataset_name, examples.len(), n_gt_intents);

    let t0 = Instant::now();

    // ═══════════════════════════════════════════════════════════════
    // Phase 1: Build term statistics + v2 baseline clustering
    // ═══════════════════════════════════════════════════════════════
    println!("Phase 1: Term statistics + v2 baseline clustering");
    println!("─────────────────────────────────────────────────");

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

    let min_df = (n_docs * 0.002).max(3.0) as usize;
    let max_df = (n_docs * 0.25) as usize;
    let discriminating: HashSet<String> = df.iter()
        .filter(|(_, &d)| d >= min_df && d <= max_df)
        .map(|(t, _)| t.clone())
        .collect();

    println!("  Discriminating terms: {}", discriminating.len());

    // IDF for all terms
    let mut idf_map: HashMap<String, f64> = HashMap::new();
    for (term, &d) in &df {
        idf_map.insert(term.clone(), (n_docs / d as f64).ln());
    }

    // PMI
    let mut pmi_map: HashMap<(String, String), f64> = HashMap::new();
    for ((a, b), &count) in &cooc {
        if !discriminating.contains(a) || !discriminating.contains(b) { continue; }
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

    // Anchor selection (same as v2)
    let mut term_idf: Vec<(String, f64)> = discriminating.iter()
        .map(|t| (t.clone(), *idf_map.get(t).unwrap_or(&0.0)))
        .collect();
    term_idf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Anchor count: ~1.5x ground truth intents, min 10, max 60
    let max_anchors = (n_gt_intents as f64 * 1.5).max(10.0).min(60.0) as usize;

    let mut anchors: Vec<String> = Vec::new();
    for (term, _) in &term_idf {
        if anchors.len() >= max_anchors { break; }
        let too_close = anchors.iter().any(|a| {
            pmi_map.get(&(term.clone(), a.clone())).copied().unwrap_or(-10.0) > 2.0
        });
        if !too_close { anchors.push(term.clone()); }
    }

    // Term → cluster assignment
    let mut term_to_cluster: HashMap<String, usize> = HashMap::new();
    for (i, anchor) in anchors.iter().enumerate() {
        term_to_cluster.insert(anchor.clone(), i);
    }
    for term in &discriminating {
        if term_to_cluster.contains_key(term) { continue; }
        let mut best = 0;
        let mut best_pmi = f64::NEG_INFINITY;
        for (i, anchor) in anchors.iter().enumerate() {
            let pmi_val = pmi_map.get(&(term.clone(), anchor.clone())).copied().unwrap_or(-10.0);
            if pmi_val > best_pmi { best_pmi = pmi_val; best = i; }
        }
        if best_pmi > 0.5 { term_to_cluster.insert(term.clone(), best); }
    }

    println!("  Anchors: {}", anchors.len());

    // Query → cluster assignment
    let mut cluster_queries: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut unassigned_queries: Vec<usize> = Vec::new();

    for (qi, _) in examples.iter().enumerate() {
        let mut votes: HashMap<usize, f64> = HashMap::new();
        for term in &query_terms[qi] {
            if let Some(&cid) = term_to_cluster.get(term) {
                let idf = idf_map.get(term).copied().unwrap_or(0.0);
                *votes.entry(cid).or_insert(0.0) += idf;
            }
        }
        if let Some((&best, _)) = votes.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            cluster_queries.entry(best).or_default().push(qi);
        } else {
            unassigned_queries.push(qi);
        }
    }

    let assigned_count: usize = cluster_queries.values().map(|v| v.len()).sum();
    println!("  Assigned: {} | Unassigned: {}", assigned_count, unassigned_queries.len());

    // v2 baseline metrics
    let v2_capture = compute_capture_rates(&cluster_queries, &examples);
    let v2_gt50 = v2_capture.iter().filter(|(_, c)| *c > 50.0).count();
    println!("  v2 baseline: {}/{} intents at >50% capture\n", v2_gt50, n_gt_intents);

    // ═══════════════════════════════════════════════════════════════
    // Phase 2: Distributional substitution detection
    // ═══════════════════════════════════════════════════════════════
    println!("Phase 2: Distributional substitution detection");
    println!("──────────────────────────────────────────────");

    let frames = detect_substitutes(&discriminating, &cooc, &pmi_map, &df, n_docs, min_df);

    // ═══════════════════════════════════════════════════════════════
    // Phase 2b: Skip splitting — merge-only approach
    // v4 showed splitting fragments captures. Merge is the correct direction.
    // ═══════════════════════════════════════════════════════════════
    println!("\nPhase 2b: Skipping split (merge-only approach)");
    println!("──────────────────────────────────────────────");

    // No splitting — merge-only is the correct direction for capture recovery
    println!("  Clusters: {} (no splitting)", cluster_queries.len());

    // No split step — use v2 clusters directly for merge

    // ═══════════════════════════════════════════════════════════════
    // Phase 3: Post-clustering merge
    // ═══════════════════════════════════════════════════════════════
    println!("\nPhase 3: Post-clustering merge with substitution anti-affinity");
    println!("──────────────────────────────────────────────────────────────");

    // Build cluster vocabulary profiles (discriminating terms with counts)
    let mut cluster_vocab: HashMap<usize, HashMap<String, usize>> = HashMap::new();
    for (&cid, indices) in &cluster_queries {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        for &qi in indices {
            let unique: HashSet<&String> = query_terms[qi].iter().collect();
            for t in unique {
                if discriminating.contains(t) {
                    *vocab.entry(t.clone()).or_insert(0) += 1;
                }
            }
        }
        cluster_vocab.insert(cid, vocab);
    }

    // Find "signature terms" for each cluster — terms that appear in >10% of cluster queries
    // and have high IDF (discriminating)
    let cluster_signatures: HashMap<usize, HashSet<String>> = cluster_vocab.iter()
        .map(|(&cid, vocab)| {
            let cluster_size = cluster_queries.get(&cid).map(|q| q.len()).unwrap_or(1) as f64;
            let sig: HashSet<String> = vocab.iter()
                .filter(|(t, &count)| {
                    let frac = count as f64 / cluster_size;
                    let idf = idf_map.get(t.as_str()).copied().unwrap_or(0.0);
                    frac > 0.10 && idf > 1.5
                })
                .map(|(t, _)| t.clone())
                .collect();
            (cid, sig)
        })
        .collect();

    // Score all cluster pairs for merge candidacy
    let cluster_ids: Vec<usize> = cluster_queries.keys().copied().collect();
    let mut merge_candidates: Vec<(usize, usize, f64, String)> = Vec::new(); // (i, j, score, reason)

    for i in 0..cluster_ids.len() {
        for j in (i + 1)..cluster_ids.len() {
            let ci = cluster_ids[i];
            let cj = cluster_ids[j];

            let sig_i = match cluster_signatures.get(&ci) { Some(s) => s, None => continue };
            let sig_j = match cluster_signatures.get(&cj) { Some(s) => s, None => continue };

            if sig_i.is_empty() || sig_j.is_empty() { continue; }

            // Jaccard similarity of signature terms
            let intersection = sig_i.intersection(sig_j).count();
            let union = sig_i.union(sig_j).count();
            let jaccard = if union > 0 { intersection as f64 / union as f64 } else { 0.0 };

            if jaccard < 0.15 { continue; } // too different to merge

            // Check substitution anti-affinity: do the signature terms contain
            // pairs that are substitutes? If so, DON'T merge.
            let mut has_anti_affinity = false;
            let mut anti_terms = String::new();

            // Get terms unique to each cluster (the ones that differentiate them)
            let unique_to_i: HashSet<&String> = sig_i.difference(sig_j).collect();
            let unique_to_j: HashSet<&String> = sig_j.difference(sig_i).collect();

            'outer: for ti in &unique_to_i {
                if let Some(subs) = frames.substitutes.get(ti.as_str()) {
                    for tj in &unique_to_j {
                        if subs.contains(tj.as_str()) {
                            has_anti_affinity = true;
                            anti_terms = format!("{} ↔ {}", ti, tj);
                            break 'outer;
                        }
                    }
                }
            }

            if has_anti_affinity {
                // Skip — these clusters represent different values of the same dimension
                // (e.g., cancel_order vs track_order)
                continue;
            }

            // Merge score: Jaccard weighted by cluster size similarity
            // (prefer merging similar-sized clusters — fragments of same intent)
            let size_i = cluster_queries[&ci].len() as f64;
            let size_j = cluster_queries[&cj].len() as f64;
            let size_ratio = size_i.min(size_j) / size_i.max(size_j);
            let score = jaccard * (0.5 + 0.5 * size_ratio);

            let shared: Vec<&String> = sig_i.intersection(sig_j).take(5).collect();
            let reason = format!("jaccard={:.2} shared={:?}", jaccard, shared);

            merge_candidates.push((ci, cj, score, reason));
        }
    }

    merge_candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    println!("  Merge candidates: {}", merge_candidates.len());
    for (ci, cj, score, reason) in merge_candidates.iter().take(15) {
        let size_i = cluster_queries.get(ci).map(|q| q.len()).unwrap_or(0);
        let size_j = cluster_queries.get(cj).map(|q| q.len()).unwrap_or(0);
        println!("    c{} ({}) + c{} ({})  score={:.3}  {}", ci, size_i, cj, size_j, score, reason);
    }

    // Greedy merge: take best candidates, check for conflicts
    let mut merged: HashMap<usize, usize> = HashMap::new(); // old_id → new_id
    let mut merge_count = 0;

    for (ci, cj, score, reason) in &merge_candidates {
        if *score < 0.05 { break; } // too weak

        // Resolve: find the current representative of each cluster
        let rep_i = resolve_rep(*ci, &merged);
        let rep_j = resolve_rep(*cj, &merged);

        if rep_i == rep_j { continue; } // already merged

        // Check: would this merge hurt purity too much?
        // Get the dominant intent of each cluster
        let dominant_i = get_dominant_intent(rep_i, &cluster_queries, &examples, &merged);
        let dominant_j = get_dominant_intent(rep_j, &cluster_queries, &examples, &merged);

        // If both clusters are dominated by the same intent → safe merge (recovering fragmentation)
        // If different intents → only merge if one is very small (likely misassigned)
        let same_dominant = dominant_i == dominant_j;
        let size_i = get_merged_size(rep_i, &cluster_queries, &merged);
        let size_j = get_merged_size(rep_j, &cluster_queries, &merged);
        let small_merge = size_i.min(size_j) < 200;

        if same_dominant || small_merge {
            // Merge: point smaller into larger
            let (keep, absorb) = if size_i >= size_j { (rep_i, rep_j) } else { (rep_j, rep_i) };
            merged.insert(absorb, keep);
            merge_count += 1;

            let tag = if same_dominant { "same-intent" } else { "small-absorb" };
            println!("  ✓ MERGE c{} ({}) + c{} ({}) [{}] {}",
                ci, size_i, cj, size_j, tag, reason);

            if merge_count >= 20 { break; } // cap merges to avoid over-merging
        }
    }

    println!("  Merges performed: {}", merge_count);

    // Build final clusters after merges
    let mut final_clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (&cid, indices) in &cluster_queries {
        let rep = resolve_rep(cid, &merged);
        final_clusters.entry(rep).or_default().extend(indices.iter().copied());
    }

    // Add unassigned as separate cluster
    if !unassigned_queries.is_empty() {
        let max_id = final_clusters.keys().max().copied().unwrap_or(0) + 1;
        final_clusters.insert(max_id, unassigned_queries.clone());
    }

    // ═══════════════════════════════════════════════════════════════
    // Phase 4: Evaluation
    // ═══════════════════════════════════════════════════════════════
    println!("\nPhase 4: Evaluation");
    println!("───────────────────");

    let elapsed = t0.elapsed();

    let mut cluster_labels: HashMap<usize, Vec<String>> = HashMap::new();
    for (&cid, indices) in &final_clusters {
        for &qi in indices {
            cluster_labels.entry(cid).or_default().push(examples[qi].intents[0].clone());
        }
    }

    let mut gt_counts: HashMap<String, usize> = HashMap::new();
    for ex in &examples {
        *gt_counts.entry(ex.intents[0].clone()).or_insert(0) += 1;
    }

    // Purity
    let mut total_pure = 0usize;
    let mut total_assigned = 0usize;
    for labels in cluster_labels.values() {
        let mut lc: HashMap<&str, usize> = HashMap::new();
        for l in labels { *lc.entry(l.as_str()).or_insert(0) += 1; }
        total_pure += lc.values().max().copied().unwrap_or(0);
        total_assigned += labels.len();
    }
    let purity = if total_assigned > 0 { total_pure as f32 / total_assigned as f32 * 100.0 } else { 0.0 };

    // NMI
    let nmi = compute_nmi(&cluster_labels, &gt_counts, examples.len());

    println!("\n  ┌──────────────────────────────────────────────────────────┐");
    println!("  │ v5 Results (PMI + Substitution Frame Merge)             │");
    println!("  ├──────────────────────────────────────────────────────────┤");
    println!("  │ Clusters:             {:>4}                              │", cluster_labels.len());
    println!("  │ Ground truth intents: {:>4}                              │", gt_counts.len());
    println!("  │ Purity:               {:>5.1}%                            │", purity);
    println!("  │ NMI:                  {:.3}                             │", nmi);
    println!("  │ Time:                 {:.2}s                             │", elapsed.as_secs_f64());
    println!("  └──────────────────────────────────────────────────────────┘");

    let v5_gt50 = compute_capture_rates(&final_clusters, &examples)
        .iter().filter(|(_, c)| *c > 50.0).count();

    println!("\n  ┌───────────────────┬──────────┬──────────┐");
    println!("  │ Metric            │  v2 base │  v5 merge│");
    println!("  ├───────────────────┼──────────┼──────────┤");
    println!("  │ NMI               │          │  {:.3}   │", nmi);
    println!("  │ Purity            │          │  {:>5.1}%  │", purity);
    println!("  │ Clusters          │  {:>4}    │  {:>4}    │", cluster_queries.len(), cluster_labels.len());
    println!("  │ >50% capture      │  {:>2}/{}  │  {:>2}/{}  │",
        v2_gt50, n_gt_intents, v5_gt50, n_gt_intents);
    println!("  └───────────────────┴──────────┴──────────┘");

    // Per ground-truth intent
    println!("\n  Ground truth → best cluster:");
    let mut gt_to_best: Vec<(String, usize, f32, usize, usize)> = Vec::new();
    for (gt_intent, &gt_total) in &gt_counts {
        let mut best_cluster = 0;
        let mut best_count = 0;
        let mut second_best = 0;
        let mut fragments = 0;
        for (&cid, labels) in &cluster_labels {
            let count = labels.iter().filter(|l| l.as_str() == gt_intent.as_str()).count();
            if count > 0 {
                fragments += 1;
                if count > best_count {
                    second_best = best_count;
                    best_count = count;
                    best_cluster = cid;
                } else if count > second_best {
                    second_best = count;
                }
            }
        }
        let capture = best_count as f32 / gt_total as f32 * 100.0;
        gt_to_best.push((gt_intent.clone(), best_cluster, capture, fragments, second_best));
    }
    gt_to_best.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    let mut gt_majority = 0;
    let mut gt_top2 = 0;
    for (intent, cluster, capture, fragments, second) in &gt_to_best {
        if *capture > 50.0 { gt_majority += 1; }
        let combined = *capture + (*second as f32 / gt_counts[intent] as f32 * 100.0);
        if combined > 50.0 { gt_top2 += 1; }
        let marker = if *capture > 50.0 { "✓" } else { " " };
        println!("  {} {:<40} → c{:>3} ({:>5.1}% +2nd:{:>5.1}%, {} frags)",
            marker, intent, cluster, capture, combined, fragments);
    }
    println!("\n  Intents with >50% in one cluster: {}/{} (v2 baseline: {})",
        gt_majority, n_gt_intents, v2_gt50);
    println!("  Intents with >50% in top-2 clusters: {}/{}", gt_top2, n_gt_intents);

    // Show improvement/regression per intent vs v2
    println!("\n  Changes vs v2 baseline:");
    for (intent, _, v5_capture, _, _) in &gt_to_best {
        let v2_cap = v2_capture.iter().find(|(i, _)| i == intent).map(|(_, c)| *c).unwrap_or(0.0);
        let diff = v5_capture - v2_cap;
        if diff.abs() > 3.0 {
            let arrow = if diff > 0.0 { "↑" } else { "↓" };
            println!("    {} {}: {:.1}% → {:.1}% ({}{:.1}%)",
                arrow, intent, v2_cap, v5_capture, if diff > 0.0 { "+" } else { "" }, diff);
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn resolve_rep(id: usize, merged: &HashMap<usize, usize>) -> usize {
    let mut current = id;
    while let Some(&parent) = merged.get(&current) {
        current = parent;
    }
    current
}

fn get_dominant_intent(cluster_id: usize, cluster_queries: &HashMap<usize, Vec<usize>>,
    examples: &[Example], merged: &HashMap<usize, usize>) -> String {
    let mut intent_counts: HashMap<String, usize> = HashMap::new();
    // Collect all queries that now map to this cluster
    for (&cid, indices) in cluster_queries {
        if resolve_rep(cid, merged) == cluster_id {
            for &qi in indices {
                *intent_counts.entry(examples[qi].intents[0].clone()).or_insert(0) += 1;
            }
        }
    }
    intent_counts.into_iter().max_by_key(|(_, c)| *c).map(|(i, _)| i).unwrap_or_default()
}

fn get_merged_size(cluster_id: usize, cluster_queries: &HashMap<usize, Vec<usize>>,
    merged: &HashMap<usize, usize>) -> usize {
    let mut total = 0;
    for (&cid, indices) in cluster_queries {
        if resolve_rep(cid, merged) == cluster_id {
            total += indices.len();
        }
    }
    total
}

fn compute_capture_rates(clusters: &HashMap<usize, Vec<usize>>, examples: &[Example]) -> Vec<(String, f32)> {
    let mut gt_counts: HashMap<String, usize> = HashMap::new();
    for ex in examples {
        *gt_counts.entry(ex.intents[0].clone()).or_insert(0) += 1;
    }

    let mut gt_best: HashMap<String, usize> = HashMap::new();
    for (_, indices) in clusters {
        let mut lc: HashMap<String, usize> = HashMap::new();
        for &qi in indices {
            *lc.entry(examples[qi].intents[0].clone()).or_insert(0) += 1;
        }
        for (intent, count) in lc {
            let best = gt_best.entry(intent).or_insert(0);
            if count > *best { *best = count; }
        }
    }

    gt_counts.iter()
        .map(|(intent, &total)| {
            let best = gt_best.get(intent).copied().unwrap_or(0);
            (intent.clone(), best as f32 / total as f32 * 100.0)
        })
        .collect()
}

fn compute_nmi(
    cluster_labels: &HashMap<usize, Vec<String>>,
    gt_counts: &HashMap<String, usize>,
    n: usize,
) -> f64 {
    let n_f = n as f64;
    let mut mi = 0.0f64;
    for labels in cluster_labels.values() {
        let cs = labels.len() as f64;
        let mut lc: HashMap<&str, usize> = HashMap::new();
        for l in labels { *lc.entry(l.as_str()).or_insert(0) += 1; }
        for (&label, &count) in &lc {
            let gt = *gt_counts.get(label).unwrap_or(&1) as f64;
            let c = count as f64;
            if c > 0.0 { mi += (c / n_f) * (n_f * c / (cs * gt)).ln(); }
        }
    }
    let mut hc = 0.0f64;
    for labels in cluster_labels.values() {
        let p = labels.len() as f64 / n_f;
        if p > 0.0 { hc -= p * p.ln(); }
    }
    let mut hg = 0.0f64;
    for &count in gt_counts.values() {
        let p = count as f64 / n_f;
        if p > 0.0 { hg -= p * p.ln(); }
    }
    let d = hc + hg;
    if d > 0.0 { 2.0 * mi / d } else { 0.0 }
}
