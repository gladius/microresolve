//! Auto-discovery v4: PMI Anchors + Targeted Impurity Splitting
//!
//! v2 baseline: 16/27 intents at >50% capture. NMI=0.589.
//!
//! v4 approach: Keep v2's initial clustering (it works well for pure intents),
//! then detect and fix impure clusters using two signals:
//!
//! 1. CROSS-CUTTING TERMS: After v2 clustering, identify terms that have strong
//!    PMI connections to 3+ different clusters. These are typically action verbs
//!    (cancel, track, change) that cross object boundaries.
//!
//! 2. WITHIN-CLUSTER MUTUAL EXCLUSIVITY: For each cluster, check if any cross-cutting
//!    terms have mutually exclusive query sets (<15% overlap). If "cancel" and "track"
//!    never appear in the same query within the order cluster, they define a clean split.
//!
//! Why this preserves v2's strengths:
//! - Pure clusters (newsletter_subscription) won't have mutually exclusive cross-cutting
//!   terms → they stay intact
//! - Only impure clusters get split, and only along the right dimension (verbs)
//!
//! Run: cargo run --release --bin auto_discover_v4

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

/// Group morphological variants by shared prefix.
/// Returns a map: canonical_form → set of variant forms.
/// Only groups terms that share a 5+ char prefix AND appear in similar contexts
/// (to avoid false merges like "order" and "ordeal").
fn find_morphological_groups(
    terms: &HashSet<String>,
    df: &HashMap<String, usize>,
    cooc: &HashMap<(String, String), usize>,
    n_docs: f64,
) -> HashMap<String, Vec<String>> {
    let mut prefix_groups: HashMap<String, Vec<String>> = HashMap::new();

    // Group by 5-char prefix
    for term in terms {
        if term.len() >= 5 {
            let prefix = term[..5].to_string();
            prefix_groups.entry(prefix).or_default().push(term.clone());
        }
    }

    // For each prefix group, verify they're actually related
    // (share high conditional co-occurrence with the same other terms)
    let mut canonical_map: HashMap<String, Vec<String>> = HashMap::new();

    for (_, group) in &prefix_groups {
        if group.len() < 2 { continue; }

        // Canonical = the shortest form (or most common)
        let canonical = group.iter()
            .min_by_key(|t| {
                // Prefer shorter, then more common
                (t.len(), usize::MAX - df.get(t.as_str()).copied().unwrap_or(0))
            })
            .unwrap()
            .clone();

        // Check each variant: does it have high PMI with the same terms as the canonical?
        let mut verified_group = vec![canonical.clone()];
        for variant in group {
            if *variant == canonical { continue; }

            // Check: do canonical and variant have overlapping PMI neighbors?
            // If they share 50%+ of strong PMI neighbors, they're likely variants
            let can_neighbors: HashSet<&String> = terms.iter()
                .filter(|t| {
                    let key = if *t < &canonical { ((*t).clone(), canonical.clone()) } else { (canonical.clone(), (*t).clone()) };
                    cooc.get(&key).copied().unwrap_or(0) as f64 / n_docs > 0.0005
                })
                .collect();
            let var_neighbors: HashSet<&String> = terms.iter()
                .filter(|t| {
                    let key = if *t < variant { ((*t).clone(), variant.clone()) } else { (variant.clone(), (*t).clone()) };
                    cooc.get(&key).copied().unwrap_or(0) as f64 / n_docs > 0.0005
                })
                .collect();

            if can_neighbors.is_empty() || var_neighbors.is_empty() { continue; }

            let shared = can_neighbors.intersection(&var_neighbors).count();
            let union = can_neighbors.union(&var_neighbors).count();
            let jaccard = if union > 0 { shared as f64 / union as f64 } else { 0.0 };

            if jaccard > 0.3 {
                verified_group.push(variant.clone());
            }
        }

        if verified_group.len() >= 2 {
            canonical_map.insert(canonical, verified_group);
        }
    }

    canonical_map
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Auto-Discovery v4: PMI + Targeted Impurity Splitting          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let data = std::fs::read_to_string("tests/data/benchmarks/bitext_all.json")
        .expect("Run download_datasets.py first");
    let examples: Vec<Example> = serde_json::from_str(&data).unwrap();

    println!("Dataset: Bitext ({} queries, 27 ground truth intents)\n", examples.len());

    let t0 = Instant::now();

    // === Step 1: Build term statistics (no stemming) ===
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
    println!("  Discriminating terms: {}", discriminating.len());

    // === Step 1b: Find morphological groups ===
    let morph_groups = find_morphological_groups(&discriminating, &df, &cooc, n_docs);
    println!("  Morphological groups: {}", morph_groups.len());
    for (canonical, group) in morph_groups.iter().take(10) {
        println!("    {} → {:?}", canonical, group);
    }

    // Build variant → canonical mapping
    let mut variant_to_canonical: HashMap<String, String> = HashMap::new();
    for (canonical, group) in &morph_groups {
        for variant in group {
            variant_to_canonical.insert(variant.clone(), canonical.clone());
        }
    }

    // === Step 2: PMI ===
    println!("\nStep 2: Computing PMI...");

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

    // === Step 3: Anchor-based clustering (same as v2) ===
    println!("\nStep 3: Anchor-based term clustering (v2 baseline)...");

    let mut term_idf: Vec<(String, f64)> = discriminating.iter()
        .map(|t| {
            let d = *df.get(t).unwrap() as f64;
            (t.clone(), (n_docs / d).ln())
        })
        .collect();
    term_idf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut anchors: Vec<String> = Vec::new();
    for (term, _) in &term_idf {
        if anchors.len() >= 40 { break; }
        let too_close = anchors.iter().any(|a| {
            pmi_map.get(&(term.clone(), a.clone())).copied().unwrap_or(-10.0) > 2.0
        });
        if !too_close { anchors.push(term.clone()); }
    }

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

    println!("  Anchors selected: {}", anchors.len());

    // === Step 4: Assign queries to clusters ===
    println!("\nStep 4: Assigning queries to term clusters...");

    let mut cluster_queries: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut unassigned: Vec<usize> = Vec::new();

    for (qi, _) in examples.iter().enumerate() {
        let mut votes: HashMap<usize, f64> = HashMap::new();
        for term in &query_terms[qi] {
            if let Some(&cid) = term_to_cluster.get(term) {
                let idf = (n_docs / *df.get(term).unwrap_or(&1) as f64).ln();
                *votes.entry(cid).or_insert(0.0) += idf;
            }
        }
        if let Some((&best, _)) = votes.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            cluster_queries.entry(best).or_default().push(qi);
        } else {
            unassigned.push(qi);
        }
    }

    let assigned_count: usize = cluster_queries.values().map(|v| v.len()).sum();
    println!("  Assigned: {} | Unassigned: {}", assigned_count, unassigned.len());

    // Show per-cluster intent breakdown (for diagnostics)
    println!("\n  v2 baseline cluster breakdown:");
    let mut cluster_info: Vec<(usize, usize, Vec<(String, usize)>)> = Vec::new();
    for (&cid, indices) in &cluster_queries {
        let mut intent_counts: HashMap<String, usize> = HashMap::new();
        for &qi in indices {
            *intent_counts.entry(examples[qi].intents[0].clone()).or_insert(0) += 1;
        }
        let mut ic_sorted: Vec<_> = intent_counts.into_iter().collect();
        ic_sorted.sort_by(|a, b| b.1.cmp(&a.1));
        cluster_info.push((cid, indices.len(), ic_sorted));
    }
    cluster_info.sort_by(|a, b| b.1.cmp(&a.1));

    // Count v2 baseline score
    let mut v2_gt_majority = 0;
    {
        let mut gt_counts_temp: HashMap<String, usize> = HashMap::new();
        for ex in &examples {
            *gt_counts_temp.entry(ex.intents[0].clone()).or_insert(0) += 1;
        }
        let mut gt_best_count: HashMap<String, usize> = HashMap::new();
        for (_, indices) in &cluster_queries {
            let mut lc: HashMap<String, usize> = HashMap::new();
            for &qi in indices {
                *lc.entry(examples[qi].intents[0].clone()).or_insert(0) += 1;
            }
            for (intent, count) in lc {
                let best = gt_best_count.entry(intent).or_insert(0);
                if count > *best { *best = count; }
            }
        }
        for (intent, &best_count) in &gt_best_count {
            let total = gt_counts_temp.get(intent).copied().unwrap_or(1);
            if best_count as f32 / total as f32 > 0.50 {
                v2_gt_majority += 1;
            }
        }
    }
    println!("  v2 baseline: {}/27 intents at >50%\n", v2_gt_majority);

    for (cid, size, intents) in &cluster_info {
        if *size >= 500 {
            let top: Vec<String> = intents.iter().take(4)
                .map(|(i, c)| format!("{}:{}", i, c)).collect();
            let purity = intents[0].1 as f32 / *size as f32;
            let marker = if purity > 0.65 { "✓" } else { "✗" };
            println!("  {} c{:>2} ({:>4})  purity={:.0}%  [{}]",
                marker, cid, size, purity * 100.0, top.join(", "));
        }
    }

    // === Step 5: Detect impure clusters and split ===
    println!("\nStep 5: Targeted impurity splitting...");

    let mut final_clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut next_id = 0usize;
    let mut splits_made = 0;

    for (&cid, query_indices) in &cluster_queries {
        let n_local = query_indices.len();

        // ONLY split clusters > 1500 — these are clearly impure (2+ intents × ~1000)
        // Smaller clusters are likely pure or close enough; splitting them fragments good clusters
        if n_local >= 1500 {
            // Build term → query set within this cluster
            let mut raw_term_sets: HashMap<String, HashSet<usize>> = HashMap::new();
            for &qi in query_indices {
                for term in &query_terms[qi] {
                    raw_term_sets.entry(term.clone()).or_default().insert(qi);
                }
            }

            // Merge morphological variants by 5-char prefix
            // "cancel"/"canceling"/"cancelling" → prefix "cance" → merged query set
            let mut prefix_groups: HashMap<String, Vec<String>> = HashMap::new();
            for term in raw_term_sets.keys() {
                if term.len() >= 5 {
                    prefix_groups.entry(term[..5].to_string()).or_default().push(term.clone());
                } else {
                    prefix_groups.entry(term.clone()).or_default().push(term.clone());
                }
            }

            let mut term_query_sets: HashMap<String, HashSet<usize>> = HashMap::new();
            for (_, group) in &prefix_groups {
                // Canonical = shortest form in group
                let canonical = group.iter().min_by_key(|t| t.len()).unwrap().clone();
                let mut merged_set = HashSet::new();
                for variant in group {
                    if let Some(s) = raw_term_sets.get(variant) {
                        merged_set.extend(s.iter().copied());
                    }
                }
                // Only merge if the variants actually represent the same concept
                // (if merged set is much larger than any individual, variants are complementary → merge)
                let max_individual = group.iter()
                    .map(|v| raw_term_sets.get(v).map(|s| s.len()).unwrap_or(0))
                    .max().unwrap_or(0);
                if merged_set.len() > max_individual || group.len() == 1 {
                    term_query_sets.insert(canonical, merged_set);
                } else {
                    // Variants overlap too much — keep largest only
                    let best = group.iter()
                        .max_by_key(|v| raw_term_sets.get(*v).map(|s| s.len()).unwrap_or(0))
                        .unwrap();
                    if let Some(s) = raw_term_sets.get(best) {
                        term_query_sets.insert(best.clone(), s.clone());
                    }
                }
            }

            // Identify cluster-common terms (>50% of cluster) — these define the cluster, not intents
            let cluster_common: HashSet<String> = term_query_sets.iter()
                .filter(|(_, s)| s.len() as f64 / n_local as f64 > 0.50)
                .map(|(t, _)| t.clone())
                .collect();

            // Filter: terms appearing in 5-40% of cluster, excluding cluster-common terms
            let min_local_df = (n_local as f64 * 0.05).max(20.0) as usize;
            let max_local_df = (n_local as f64 * 0.40) as usize;
            let mut candidates: Vec<(String, HashSet<usize>)> = term_query_sets.into_iter()
                .filter(|(t, s)| s.len() >= min_local_df && s.len() <= max_local_df && !cluster_common.contains(t))
                .collect();

            // Sort by DF descending (most common = major split candidates)
            candidates.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

            // Greedy select mutually exclusive partition points
            let mut partition_points: Vec<(String, HashSet<usize>)> = Vec::new();
            for (term, query_set) in candidates {
                let mut exclusive = true;
                for (_, pp_set) in &partition_points {
                    let overlap = query_set.intersection(pp_set).count();
                    let min_size = query_set.len().min(pp_set.len());
                    if min_size > 0 && overlap as f64 / min_size as f64 > 0.15 {
                        exclusive = false;
                        break;
                    }
                }
                if exclusive {
                    partition_points.push((term, query_set));
                }
                if partition_points.len() >= 8 { break; }
            }

            // Only split if 2+ exclusive partition points covering > 35% of cluster
            if partition_points.len() >= 2 {
                let covered: HashSet<usize> = partition_points.iter()
                    .flat_map(|(_, s)| s.iter().copied())
                    .collect();
                let coverage = covered.len() as f64 / n_local as f64;

                if coverage > 0.35 {
                    // Assign queries to partition points
                    let mut sub_clusters: Vec<Vec<usize>> = vec![Vec::new(); partition_points.len()];
                    let mut residual: Vec<usize> = Vec::new();

                    for &qi in query_indices {
                        let mut best_pp = None;
                        let mut best_specificity = f64::NEG_INFINITY;
                        for (i, (pp_term, pp_set)) in partition_points.iter().enumerate() {
                            if pp_set.contains(&qi) {
                                let gdf = *df.get(pp_term).unwrap_or(&1) as f64;
                                let specificity = (n_docs / gdf).ln();
                                if specificity > best_specificity {
                                    best_pp = Some(i);
                                    best_specificity = specificity;
                                }
                            }
                        }
                        match best_pp {
                            Some(i) => sub_clusters[i].push(qi),
                            None => residual.push(qi),
                        }
                    }

                    // Keep residual as separate sub-cluster
                    if !residual.is_empty() {
                        sub_clusters.push(residual);
                    }

                    // === QUALITY CHECK: Vocabulary distinctness ===
                    // Sub-clusters must have distinct vocabulary (beyond cluster-common terms).
                    // Bad splits (by modifier instead of verb) produce sub-clusters with
                    // identical differentiating vocabulary → reject.
                    let sig_terms: Vec<HashSet<String>> = sub_clusters.iter().map(|sc| {
                        let mut tf: HashMap<String, usize> = HashMap::new();
                        for &qi in sc {
                            let unique: HashSet<&String> = query_terms[qi].iter().collect();
                            for t in unique {
                                if !cluster_common.contains(t) {
                                    *tf.entry(t.clone()).or_insert(0) += 1;
                                }
                            }
                        }
                        let min_c = (sc.len() as f64 * 0.10).max(5.0) as usize;
                        let max_c = (sc.len() as f64 * 0.60) as usize;
                        tf.into_iter()
                            .filter(|(_, c)| *c >= min_c && *c <= max_c)
                            .map(|(t, _)| t)
                            .collect()
                    }).collect();

                    // Check pairwise distinctness of non-residual sub-clusters
                    let n_pp = partition_points.len(); // non-residual count
                    let mut total_jaccard = 0.0;
                    let mut n_pairs = 0;
                    for i in 0..n_pp {
                        for j in (i+1)..n_pp {
                            if sig_terms[i].is_empty() || sig_terms[j].is_empty() { continue; }
                            let shared = sig_terms[i].intersection(&sig_terms[j]).count();
                            let union = sig_terms[i].union(&sig_terms[j]).count();
                            let jaccard = if union > 0 { shared as f64 / union as f64 } else { 0.0 };
                            total_jaccard += jaccard;
                            n_pairs += 1;
                        }
                    }
                    let avg_jaccard = if n_pairs > 0 { total_jaccard / n_pairs as f64 } else { 0.0 };
                    let distinct_enough = avg_jaccard < 0.40; // <40% overlap = truly distinct vocabulary

                    // Filter tiny sub-clusters
                    let min_size = (n_local as f64 * 0.05).max(30.0) as usize;
                    let mut final_scs: Vec<Vec<usize>> = Vec::new();
                    let mut tiny: Vec<usize> = Vec::new();
                    for sc in sub_clusters {
                        if sc.len() >= min_size {
                            final_scs.push(sc);
                        } else {
                            tiny.extend(sc);
                        }
                    }
                    if !tiny.is_empty() {
                        if final_scs.is_empty() {
                            final_scs.push(tiny);
                        } else {
                            let largest_idx = final_scs.iter().enumerate()
                                .max_by_key(|(_, s)| s.len())
                                .map(|(i, _)| i).unwrap_or(0);
                            final_scs[largest_idx].extend(tiny);
                        }
                    }

                    if final_scs.len() >= 2 && distinct_enough {
                        let pp_names: Vec<String> = partition_points.iter()
                            .map(|(t, s)| format!("{}({})", t, s.len())).collect();
                        println!("    c{} ({}) → {} sub-clusters via [{}]  jaccard={:.2}",
                            cid, n_local, final_scs.len(), pp_names.join(", "), avg_jaccard);

                        for (i, sc) in final_scs.iter().enumerate() {
                            let mut sc_intents: HashMap<String, usize> = HashMap::new();
                            for &qi in sc {
                                *sc_intents.entry(examples[qi].intents[0].clone()).or_insert(0) += 1;
                            }
                            let mut si_sorted: Vec<_> = sc_intents.iter().collect();
                            si_sorted.sort_by(|a, b| b.1.cmp(a.1));
                            let top: Vec<String> = si_sorted.iter().take(3)
                                .map(|(i, c)| format!("{}:{}", i, c)).collect();
                            let purity = si_sorted[0].1;
                            println!("      sc{}: {:>4} queries  purity={:.0}%  [{}]",
                                i, sc.len(), *purity as f32 / sc.len() as f32 * 100.0, top.join(", "));
                        }

                        for sc in final_scs {
                            final_clusters.insert(next_id, sc);
                            next_id += 1;
                        }
                        splits_made += 1;
                        continue;
                    } else if !distinct_enough {
                        let pp_names: Vec<String> = partition_points.iter()
                            .map(|(t, s)| format!("{}({})", t, s.len())).collect();
                        println!("    c{} ({}) → REJECTED (jaccard={:.2}, sub-clusters too similar) via [{}]",
                            cid, n_local, avg_jaccard, pp_names.join(", "));
                    }
                }
            }
        }

        // Keep cluster as-is
        final_clusters.insert(next_id, query_indices.clone());
        next_id += 1;
    }

    // Unassigned as separate cluster
    if !unassigned.is_empty() {
        final_clusters.insert(next_id, unassigned);
        next_id += 1;
    }

    println!("  Splits made: {}", splits_made);
    println!("  Total clusters: {}", final_clusters.len());

    // === Step 6: Evaluate ===
    println!("\nStep 6: Evaluation\n");

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

    println!("  ┌──────────────────────────────────────────────────────────┐");
    println!("  │ v4 Results (PMI + Targeted Impurity Splitting)          │");
    println!("  ├──────────────────────────────────────────────────────────┤");
    println!("  │ Clusters discovered:  {:>4}                              │", cluster_labels.len());
    println!("  │ Ground truth intents: {:>4}                              │", gt_counts.len());
    println!("  │ Purity:               {:>5.1}%                            │", purity);
    println!("  │ NMI:                  {:.3}                             │", nmi);
    println!("  │ Time:                 {:.2}s                             │", elapsed.as_secs_f64());
    println!("  └──────────────────────────────────────────────────────────┘");

    println!("\n  Comparison:");
    println!("  ┌──────────────┬──────────┬──────────┬──────────┐");
    println!("  │ Metric       │  v2      │  v3      │  v4      │");
    println!("  ├──────────────┼──────────┼──────────┼──────────┤");
    println!("  │ NMI          │  0.589   │  0.613   │  {:.3}   │", nmi);
    println!("  │ Purity       │  48.4%   │  47.7%   │  {:>5.1}%  │", purity);
    println!("  │ Clusters     │  27      │  34      │  {:>4}    │", cluster_labels.len());
    println!("  └──────────────┴──────────┴──────────┴──────────┘");

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
    for (intent, cluster, capture, fragments, second) in &gt_to_best {
        if *capture > 50.0 { gt_majority += 1; }
        let marker = if *capture > 50.0 { "✓" } else { " " };
        let combined = *capture + (*second as f32 / gt_counts[intent] as f32 * 100.0);
        println!("  {} {:<40} → c{:>3} ({:>5.1}% +2nd:{:>5.1}%, {} frags)",
            marker, intent, cluster, capture, combined, fragments);
    }
    println!("\n  Intents with >50% in one cluster: {}/{} (v2: {}, v3: 16)",
        gt_majority, gt_counts.len(), v2_gt_majority);

    // Show how many intents reach >50% if we merge best + 2nd best clusters
    let mut gt_top2 = 0;
    for (intent, _, capture, _, second) in &gt_to_best {
        let top2 = *capture + (*second as f32 / gt_counts[intent] as f32 * 100.0);
        if top2 > 50.0 { gt_top2 += 1; }
    }
    println!("  Intents with >50% in top-2 clusters: {}/27", gt_top2);
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
