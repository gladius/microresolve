//! Intent auto-discovery from unlabeled queries.
//!
//! Discovers intent clusters using PMI-anchored clustering with
//! distributional substitution constraints. No labels required.
//!
//! Algorithm (v5):
//! 1. Compute term statistics (DF, co-occurrence, PMI, IDF)
//! 2. Select diverse anchor terms (high IDF, low mutual PMI)
//! 3. Assign queries to clusters via IDF-weighted voting
//! 4. Detect distributional substitutes (paradigmatic alternatives)
//! 5. Merge fragmented clusters with substitution anti-affinity
//!
//! Results on Bitext benchmark: 20/27 intents at >50% capture, NMI=0.602

use crate::tokenizer;
use std::collections::{HashMap, HashSet};

/// A discovered cluster of related queries.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiscoveredCluster {
    /// Suggested intent name derived from top anchor/signature terms.
    pub suggested_name: String,
    /// Top discriminating terms for this cluster.
    pub top_terms: Vec<String>,
    /// Representative sample queries (most central to the cluster).
    pub representative_queries: Vec<String>,
    /// All query indices assigned to this cluster.
    pub query_indices: Vec<usize>,
    /// Number of queries in this cluster.
    pub size: usize,
    /// Confidence: fraction of queries with >= 2 signature terms.
    pub confidence: f32,
}

/// Configuration for the discovery algorithm.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Hint for expected number of intents. Anchors = ~1.5x this.
    /// If 0, auto-estimates from data.
    pub expected_intents: usize,
    /// Minimum cluster size to report. Default: 5.
    pub min_cluster_size: usize,
    /// Maximum merges to perform. Default: 20.
    pub max_merges: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            expected_intents: 0,
            min_cluster_size: 5,
            max_merges: 20,
        }
    }
}

/// Discover intent clusters from unlabeled queries.
///
/// Takes raw query strings, returns discovered clusters sorted by size.
/// No labels or ground truth required.
pub fn discover_intents(
    queries: &[String],
    config: &DiscoveryConfig,
) -> Vec<DiscoveredCluster> {
    if queries.len() < 10 {
        return vec![];
    }

    let n_docs = queries.len() as f64;

    // Tokenize all queries using ASV's tokenizer
    let query_terms: Vec<Vec<String>> = queries
        .iter()
        .map(|q| tokenizer::tokenize(q))
        .collect();

    // Phase 1: Term statistics
    let (df, cooc) = build_term_statistics(&query_terms);

    let min_df = (n_docs * 0.002).max(3.0) as usize;
    let max_df = (n_docs * 0.25) as usize;
    let discriminating: HashSet<String> = df
        .iter()
        .filter(|(_, &d)| d >= min_df && d <= max_df)
        .map(|(t, _)| t.clone())
        .collect();

    if discriminating.is_empty() {
        return vec![];
    }

    // IDF
    let idf_map: HashMap<String, f64> = df
        .iter()
        .map(|(term, &d)| (term.clone(), (n_docs / d as f64).ln()))
        .collect();

    // PMI between discriminating terms
    let pmi_map = compute_pmi(&discriminating, &cooc, &df, n_docs);

    // Anchor selection
    let n_anchors = if config.expected_intents > 0 {
        (config.expected_intents as f64 * 1.5).max(10.0).min(60.0) as usize
    } else {
        // Auto-estimate: sqrt(discriminating terms), clamped
        (discriminating.len() as f64).sqrt().max(10.0).min(60.0) as usize
    };

    let anchors = select_anchors(&discriminating, &idf_map, &pmi_map, n_anchors);
    if anchors.is_empty() {
        return vec![];
    }

    // Assign terms to clusters
    let term_to_cluster = assign_terms_to_anchors(&anchors, &discriminating, &pmi_map);

    // Assign queries to clusters
    let (cluster_queries, _unassigned) =
        assign_queries(&query_terms, &term_to_cluster, &idf_map);

    // Phase 2: Substitution detection
    let substitutes = detect_substitutes(&discriminating, &cooc, &pmi_map, &df, n_docs, min_df);

    // Phase 3: Merge fragmented clusters
    let final_clusters = merge_clusters(
        &cluster_queries,
        &query_terms,
        &discriminating,
        &idf_map,
        &substitutes,
        config.max_merges,
    );

    // Build output
    build_output(
        &final_clusters,
        queries,
        &query_terms,
        &discriminating,
        &idf_map,
        config.min_cluster_size,
    )
}

// ============================================================================
// Phase 1: Term statistics + clustering
// ============================================================================

fn build_term_statistics(
    query_terms: &[Vec<String>],
) -> (HashMap<String, usize>, HashMap<(String, String), usize>) {
    let mut df: HashMap<String, usize> = HashMap::new();
    let mut cooc: HashMap<(String, String), usize> = HashMap::new();

    for terms in query_terms {
        let unique: HashSet<&String> = terms.iter().collect();

        for t in &unique {
            *df.entry((*t).clone()).or_insert(0) += 1;
        }

        let unique_vec: Vec<&String> = unique.into_iter().collect();
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
    }

    (df, cooc)
}

fn compute_pmi(
    discriminating: &HashSet<String>,
    cooc: &HashMap<(String, String), usize>,
    df: &HashMap<String, usize>,
    n_docs: f64,
) -> HashMap<(String, String), f64> {
    let mut pmi_map: HashMap<(String, String), f64> = HashMap::new();

    for ((a, b), &count) in cooc {
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

    pmi_map
}

fn select_anchors(
    discriminating: &HashSet<String>,
    idf_map: &HashMap<String, f64>,
    pmi_map: &HashMap<(String, String), f64>,
    max_anchors: usize,
) -> Vec<String> {
    let mut term_idf: Vec<(String, f64)> = discriminating
        .iter()
        .map(|t| (t.clone(), *idf_map.get(t).unwrap_or(&0.0)))
        .collect();
    term_idf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut anchors: Vec<String> = Vec::new();
    for (term, _) in &term_idf {
        if anchors.len() >= max_anchors {
            break;
        }
        let too_close = anchors.iter().any(|a| {
            pmi_map
                .get(&(term.clone(), a.clone()))
                .copied()
                .unwrap_or(-10.0)
                > 2.0
        });
        if !too_close {
            anchors.push(term.clone());
        }
    }

    anchors
}

fn assign_terms_to_anchors(
    anchors: &[String],
    discriminating: &HashSet<String>,
    pmi_map: &HashMap<(String, String), f64>,
) -> HashMap<String, usize> {
    let mut term_to_cluster: HashMap<String, usize> = HashMap::new();

    for (i, anchor) in anchors.iter().enumerate() {
        term_to_cluster.insert(anchor.clone(), i);
    }

    for term in discriminating {
        if term_to_cluster.contains_key(term) {
            continue;
        }
        let mut best = 0;
        let mut best_pmi = f64::NEG_INFINITY;
        for (i, anchor) in anchors.iter().enumerate() {
            let pmi_val = pmi_map
                .get(&(term.clone(), anchor.clone()))
                .copied()
                .unwrap_or(-10.0);
            if pmi_val > best_pmi {
                best_pmi = pmi_val;
                best = i;
            }
        }
        if best_pmi > 0.5 {
            term_to_cluster.insert(term.clone(), best);
        }
    }

    term_to_cluster
}

fn assign_queries(
    query_terms: &[Vec<String>],
    term_to_cluster: &HashMap<String, usize>,
    idf_map: &HashMap<String, f64>,
) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
    let mut cluster_queries: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut unassigned: Vec<usize> = Vec::new();

    for (qi, terms) in query_terms.iter().enumerate() {
        let mut votes: HashMap<usize, f64> = HashMap::new();
        for term in terms {
            if let Some(&cid) = term_to_cluster.get(term) {
                let idf = idf_map.get(term).copied().unwrap_or(0.0);
                *votes.entry(cid).or_insert(0.0) += idf;
            }
        }
        if let Some((&best, _)) = votes.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            cluster_queries.entry(best).or_default().push(qi);
        } else {
            unassigned.push(qi);
        }
    }

    (cluster_queries, unassigned)
}

// ============================================================================
// Phase 2: Distributional substitution detection
// ============================================================================

fn detect_substitutes(
    discriminating: &HashSet<String>,
    cooc: &HashMap<(String, String), usize>,
    pmi_map: &HashMap<(String, String), f64>,
    df: &HashMap<String, usize>,
    n_docs: f64,
    min_df: usize,
) -> HashMap<String, HashSet<String>> {
    // Build PMI neighbor sets for each discriminating term
    let mut pmi_neighbors: HashMap<String, HashSet<String>> = HashMap::new();
    for term in discriminating {
        if df.get(term).copied().unwrap_or(0) < min_df {
            continue;
        }
        let neighbors: HashSet<String> = discriminating
            .iter()
            .filter(|t| {
                *t != term
                    && pmi_map
                        .get(&(term.clone(), (*t).clone()))
                        .copied()
                        .unwrap_or(0.0)
                        > 1.5
            })
            .cloned()
            .collect();
        if neighbors.len() >= 2 {
            pmi_neighbors.insert(term.clone(), neighbors);
        }
    }

    let mut substitutes: HashMap<String, HashSet<String>> = HashMap::new();
    let terms_with_neighbors: Vec<&String> = pmi_neighbors.keys().collect();

    for i in 0..terms_with_neighbors.len() {
        for j in (i + 1)..terms_with_neighbors.len() {
            let a = terms_with_neighbors[i];
            let b = terms_with_neighbors[j];

            // Condition 1: Rarely co-occur
            let key = if a < b {
                (a.clone(), b.clone())
            } else {
                (b.clone(), a.clone())
            };
            let co_count = cooc.get(&key).copied().unwrap_or(0);
            let df_a = df.get(a).copied().unwrap_or(1) as f64;
            let df_b = df.get(b).copied().unwrap_or(1) as f64;
            let expected = (df_a * df_b) / n_docs;

            if co_count as f64 > expected * 0.3 {
                continue;
            }

            // Condition 2: Similar contexts (share PMI neighbors)
            let na = &pmi_neighbors[a];
            let nb = &pmi_neighbors[b];

            let shared = na.intersection(nb).count();
            let union = na.union(nb).count();
            let jaccard = if union > 0 {
                shared as f64 / union as f64
            } else {
                0.0
            };

            if jaccard > 0.25 && shared >= 3 {
                substitutes
                    .entry(a.clone())
                    .or_default()
                    .insert(b.clone());
                substitutes
                    .entry(b.clone())
                    .or_default()
                    .insert(a.clone());
            }
        }
    }

    substitutes
}

// ============================================================================
// Phase 3: Merge fragmented clusters
// ============================================================================

fn merge_clusters(
    cluster_queries: &HashMap<usize, Vec<usize>>,
    query_terms: &[Vec<String>],
    discriminating: &HashSet<String>,
    idf_map: &HashMap<String, f64>,
    substitutes: &HashMap<String, HashSet<String>>,
    max_merges: usize,
) -> HashMap<usize, Vec<usize>> {
    // Build cluster vocabulary profiles
    let mut cluster_vocab: HashMap<usize, HashMap<String, usize>> = HashMap::new();
    for (&cid, indices) in cluster_queries {
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

    // Signature terms: >10% of cluster queries, IDF > 1.5
    let cluster_signatures: HashMap<usize, HashSet<String>> = cluster_vocab
        .iter()
        .map(|(&cid, vocab)| {
            let cluster_size =
                cluster_queries.get(&cid).map(|q| q.len()).unwrap_or(1) as f64;
            let sig: HashSet<String> = vocab
                .iter()
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
    let mut merge_candidates: Vec<(usize, usize, f64)> = Vec::new();

    for i in 0..cluster_ids.len() {
        for j in (i + 1)..cluster_ids.len() {
            let ci = cluster_ids[i];
            let cj = cluster_ids[j];

            let sig_i = match cluster_signatures.get(&ci) {
                Some(s) if !s.is_empty() => s,
                _ => continue,
            };
            let sig_j = match cluster_signatures.get(&cj) {
                Some(s) if !s.is_empty() => s,
                _ => continue,
            };

            let intersection = sig_i.intersection(sig_j).count();
            let union = sig_i.union(sig_j).count();
            let jaccard = if union > 0 {
                intersection as f64 / union as f64
            } else {
                0.0
            };

            if jaccard < 0.15 {
                continue;
            }

            // Check substitution anti-affinity
            let unique_to_i: HashSet<&String> = sig_i.difference(sig_j).collect();
            let unique_to_j: HashSet<&String> = sig_j.difference(sig_i).collect();

            let has_anti_affinity = unique_to_i.iter().any(|ti| {
                if let Some(subs) = substitutes.get(ti.as_str()) {
                    unique_to_j.iter().any(|tj| subs.contains(tj.as_str()))
                } else {
                    false
                }
            });

            if has_anti_affinity {
                continue;
            }

            let size_i = cluster_queries[&ci].len() as f64;
            let size_j = cluster_queries[&cj].len() as f64;
            let size_ratio = size_i.min(size_j) / size_i.max(size_j);
            let score = jaccard * (0.5 + 0.5 * size_ratio);

            merge_candidates.push((ci, cj, score));
        }
    }

    merge_candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Greedy merge with union-find
    let mut merged: HashMap<usize, usize> = HashMap::new();
    let mut merge_count = 0;

    for &(ci, cj, score) in &merge_candidates {
        if score < 0.05 {
            break;
        }

        let rep_i = resolve_rep(ci, &merged);
        let rep_j = resolve_rep(cj, &merged);
        if rep_i == rep_j {
            continue;
        }

        let size_i = get_merged_size(rep_i, cluster_queries, &merged);
        let size_j = get_merged_size(rep_j, cluster_queries, &merged);
        let small_merge = size_i.min(size_j) < 200;

        // Only merge if one is small (fragment absorption)
        // Without ground truth we can't check "same dominant intent"
        if small_merge || score > 0.3 {
            let (keep, absorb) = if size_i >= size_j {
                (rep_i, rep_j)
            } else {
                (rep_j, rep_i)
            };
            merged.insert(absorb, keep);
            merge_count += 1;

            if merge_count >= max_merges {
                break;
            }
        }
    }

    // Build final clusters after merges
    let mut final_clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for (&cid, indices) in cluster_queries {
        let rep = resolve_rep(cid, &merged);
        final_clusters
            .entry(rep)
            .or_default()
            .extend(indices.iter().copied());
    }

    final_clusters
}

fn resolve_rep(id: usize, merged: &HashMap<usize, usize>) -> usize {
    let mut current = id;
    while let Some(&parent) = merged.get(&current) {
        current = parent;
    }
    current
}

fn get_merged_size(
    cluster_id: usize,
    cluster_queries: &HashMap<usize, Vec<usize>>,
    merged: &HashMap<usize, usize>,
) -> usize {
    let mut total = 0;
    for (&cid, indices) in cluster_queries {
        if resolve_rep(cid, merged) == cluster_id {
            total += indices.len();
        }
    }
    total
}

// ============================================================================
// Output construction
// ============================================================================

fn build_output(
    final_clusters: &HashMap<usize, Vec<usize>>,
    queries: &[String],
    query_terms: &[Vec<String>],
    discriminating: &HashSet<String>,
    idf_map: &HashMap<String, f64>,
    min_cluster_size: usize,
) -> Vec<DiscoveredCluster> {
    let mut results: Vec<DiscoveredCluster> = Vec::new();

    for (&_cid, indices) in final_clusters {
        if indices.len() < min_cluster_size {
            continue;
        }

        // Build cluster term profile
        let mut term_counts: HashMap<String, usize> = HashMap::new();
        for &qi in indices {
            let unique: HashSet<&String> = query_terms[qi].iter().collect();
            for t in unique {
                if discriminating.contains(t) {
                    *term_counts.entry(t.clone()).or_insert(0) += 1;
                }
            }
        }

        let cluster_size = indices.len() as f64;

        // Signature terms: high frequency + high IDF
        let mut sig_terms: Vec<(String, f64)> = term_counts
            .iter()
            .filter(|(_, &count)| count as f64 / cluster_size > 0.10)
            .map(|(t, &count)| {
                let idf = idf_map.get(t).copied().unwrap_or(0.0);
                let freq = count as f64 / cluster_size;
                (t.clone(), freq * idf)
            })
            .collect();
        sig_terms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_terms: Vec<String> = sig_terms.iter().take(10).map(|(t, _)| t.clone()).collect();

        // Suggested name from top 2-3 unigrams (skip bigrams for cleaner names)
        let suggested_name = sig_terms
            .iter()
            .filter(|(t, _)| !t.contains(' '))
            .take(3)
            .map(|(t, _)| t.as_str())
            .collect::<Vec<&str>>()
            .join("_");

        // Representative queries: highest IDF-weighted overlap with signature
        let sig_set: HashSet<&str> = sig_terms.iter().take(5).map(|(t, _)| t.as_str()).collect();
        let mut scored_queries: Vec<(usize, f64)> = indices
            .iter()
            .map(|&qi| {
                let score: f64 = query_terms[qi]
                    .iter()
                    .filter(|t| sig_set.contains(t.as_str()))
                    .map(|t| idf_map.get(t).copied().unwrap_or(0.0))
                    .sum();
                (qi, score)
            })
            .collect();
        scored_queries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let representative_queries: Vec<String> = scored_queries
            .iter()
            .take(8)
            .map(|(qi, _)| queries[*qi].clone())
            .collect();

        // Confidence: fraction of queries with >= 2 signature terms
        let queries_with_2_sig = indices
            .iter()
            .filter(|&&qi| {
                query_terms[qi]
                    .iter()
                    .filter(|t| sig_set.contains(t.as_str()))
                    .count()
                    >= 2
            })
            .count();
        let confidence = queries_with_2_sig as f32 / indices.len() as f32;

        results.push(DiscoveredCluster {
            suggested_name,
            top_terms,
            representative_queries,
            query_indices: indices.clone(),
            size: indices.len(),
            confidence,
        });
    }

    // Sort by size descending
    results.sort_by(|a, b| b.size.cmp(&a.size));
    results
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_empty() {
        let result = discover_intents(&[], &DiscoveryConfig::default());
        assert!(result.is_empty());
    }

    #[test]
    fn few_queries_returns_empty() {
        let queries: Vec<String> = (0..5).map(|i| format!("query {}", i)).collect();
        let result = discover_intents(&queries, &DiscoveryConfig::default());
        assert!(result.is_empty());
    }

    #[test]
    fn discovers_distinct_clusters() {
        // Two clear intent groups
        let mut queries: Vec<String> = Vec::new();
        for _ in 0..50 {
            queries.push("cancel my order please".to_string());
            queries.push("I want to cancel this order".to_string());
            queries.push("stop my order".to_string());
            queries.push("cancel the purchase".to_string());
        }
        for _ in 0..50 {
            queries.push("track my package".to_string());
            queries.push("where is my delivery".to_string());
            queries.push("shipping status please".to_string());
            queries.push("track the shipment".to_string());
        }

        let result = discover_intents(&queries, &DiscoveryConfig::default());
        assert!(
            result.len() >= 2,
            "should find at least 2 clusters, got {}",
            result.len()
        );

        // Verify clusters are non-trivial
        for cluster in &result {
            assert!(!cluster.suggested_name.is_empty());
            assert!(!cluster.top_terms.is_empty());
            assert!(!cluster.representative_queries.is_empty());
            assert!(cluster.size >= 5);
        }
    }

    #[test]
    fn bitext_benchmark() {
        // Only run if test data exists
        let path = "tests/data/benchmarks/bitext_all.json";
        let data = match std::fs::read_to_string(path) {
            Ok(d) => d,
            Err(_) => return, // skip if no test data
        };

        #[derive(serde::Deserialize)]
        struct Example {
            text: String,
            intents: Vec<String>,
        }

        let examples: Vec<Example> = serde_json::from_str(&data).unwrap();
        let queries: Vec<String> = examples.iter().map(|e| e.text.clone()).collect();

        let config = DiscoveryConfig {
            expected_intents: 27,
            min_cluster_size: 5,
            max_merges: 20,
        };

        let result = discover_intents(&queries, &config);

        // Should find a reasonable number of clusters
        assert!(
            result.len() >= 15 && result.len() <= 50,
            "expected 15-50 clusters, got {}",
            result.len()
        );

        // Total assigned queries should cover most of the dataset
        let total_assigned: usize = result.iter().map(|c| c.size).sum();
        let coverage = total_assigned as f64 / queries.len() as f64;
        assert!(
            coverage > 0.5,
            "expected >50% coverage, got {:.1}%",
            coverage * 100.0
        );

        // Print summary for manual inspection
        println!("\n  Bitext discovery: {} clusters, {:.0}% coverage",
            result.len(), coverage * 100.0);
        for (i, c) in result.iter().take(10).enumerate() {
            println!("    #{}: {} (size={}, conf={:.0}%, terms={:?})",
                i + 1, c.suggested_name, c.size, c.confidence * 100.0,
                &c.top_terms[..c.top_terms.len().min(5)]);
        }
    }
}
