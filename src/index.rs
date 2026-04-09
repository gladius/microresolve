//! InvertedIndex — sub-millisecond intent search.
//!
//! Maps terms to intent IDs with weights. Only scores intents that share
//! at least one term with the query — O(matched postings), not O(all intents).

use crate::vector::LearnedVector;
use std::collections::HashMap;

/// A scored search result.
#[derive(Debug, Clone)]
pub struct ScoredIntent {
    /// The intent identifier.
    pub id: String,
    /// Accumulated score from matching terms.
    pub score: f32,
}

/// Inverted index: term -> [(intent_id, weight)].
pub struct InvertedIndex {
    postings: HashMap<String, Vec<(String, f32)>>,
    intent_count: usize,
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl InvertedIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            intent_count: 0,
        }
    }

    /// Build from all intent vectors.
    pub fn build(intent_vectors: &HashMap<String, LearnedVector>) -> Self {
        let mut postings: HashMap<String, Vec<(String, f32)>> = HashMap::new();

        for (intent_id, vector) in intent_vectors {
            for (term, weight) in vector.effective_terms() {
                postings
                    .entry(term)
                    .or_default()
                    .push((intent_id.clone(), weight));
            }
        }

        // Sort each posting list by weight descending
        for list in postings.values_mut() {
            list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        Self {
            intent_count: intent_vectors.len(),
            postings,
        }
    }

    /// Search: score intents matching any query term, return top-k.
    ///
    /// Applies IDF (inverse document frequency) weighting at search time:
    /// terms shared across many intents get discounted, terms unique to
    /// one intent get boosted. Formula: `weight * (1 + 0.5 * ln(N/df))`.
    pub fn search(&self, query_terms: &[String], top_k: usize) -> Vec<ScoredIntent> {
        let n = self.intent_count.max(1) as f32;
        let mut scores: HashMap<&str, f32> = HashMap::new();

        for term in query_terms {
            if let Some(postings) = self.postings.get(term) {
                let df = postings.len() as f32;
                let idf = 1.0 + 0.5 * (n / df).ln();
                for (intent_id, weight) in postings {
                    *scores.entry(intent_id.as_str()).or_insert(0.0) += weight * idf;
                }
            }
        }

        let mut results: Vec<ScoredIntent> = scores
            .into_iter()
            .map(|(id, score)| ScoredIntent { id: id.to_string(), score })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Search with weighted query terms. Each term has a weight that multiplies its contribution.
    /// Used for similarity-expanded queries where similar terms have discounted weight.
    pub fn search_weighted(&self, query_terms: &HashMap<String, f32>, top_k: usize) -> Vec<ScoredIntent> {
        let n = self.intent_count.max(1) as f32;
        let mut scores: HashMap<&str, f32> = HashMap::new();

        for (term, query_weight) in query_terms {
            if let Some(postings) = self.postings.get(term) {
                let df = postings.len() as f32;
                let idf = 1.0 + 0.5 * (n / df).ln();
                for (intent_id, weight) in postings {
                    *scores.entry(intent_id.as_str()).or_insert(0.0) += weight * idf * query_weight;
                }
            }
        }

        let mut results: Vec<ScoredIntent> = scores
            .into_iter()
            .map(|(id, score)| ScoredIntent { id: id.to_string(), score })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Get all indexed terms.
    pub fn all_terms(&self) -> Vec<String> {
        self.postings.keys().cloned().collect()
    }

    /// Update a single intent's entries without full rebuild.
    pub fn update_intent(&mut self, intent_id: &str, vector: &LearnedVector) {
        // Remove old entries
        for list in self.postings.values_mut() {
            list.retain(|(id, _)| id != intent_id);
        }
        self.postings.retain(|_, list| !list.is_empty());

        // Add new entries
        for (term, weight) in vector.effective_terms() {
            let list = self.postings.entry(term).or_default();
            list.push((intent_id.to_string(), weight));
            list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    /// Remove an intent from the index.
    pub fn remove_intent(&mut self, intent_id: &str) {
        for list in self.postings.values_mut() {
            list.retain(|(id, _)| id != intent_id);
        }
        self.postings.retain(|_, list| !list.is_empty());
        self.intent_count = self.intent_count.saturating_sub(1);
    }

    /// Number of unique terms in the index.
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }

    /// Number of intents in the index.
    pub fn intent_count(&self) -> usize {
        self.intent_count
    }

    /// Document frequency: number of intents containing this term.
    pub fn df(&self, term: &str) -> usize {
        self.postings.get(term).map_or(0, |v| v.len())
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }

    /// Iterate over all terms in the index.
    pub fn terms(&self) -> impl Iterator<Item = &String> {
        self.postings.keys()
    }

    /// Get postings for a term: Vec<(intent_id, weight)>.
    /// Returns empty slice if term is not in the index.
    pub fn postings(&self, term: &str) -> &[(String, f32)] {
        self.postings.get(term).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(pairs: &[(&str, f32)]) -> HashMap<String, f32> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn terms(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    fn intents(entries: &[(&str, &[(&str, f32)])]) -> HashMap<String, LearnedVector> {
        entries.iter().map(|(id, seeds)| {
            (id.to_string(), LearnedVector::from_seed(seed(seeds)))
        }).collect()
    }

    #[test]
    fn empty_index() {
        let index = InvertedIndex::new();
        assert!(index.is_empty());
        assert!(index.search(&terms(&["payment"]), 10).is_empty());
    }

    #[test]
    fn build_and_search() {
        let vecs = intents(&[
            ("cancel_order", &[("cancel", 1.0), ("order", 0.9)]),
            ("track_order", &[("track", 1.0), ("order", 0.8), ("shipping", 0.7)]),
            ("refund", &[("refund", 1.0), ("cancel", 0.3), ("money", 0.8)]),
        ]);

        let index = InvertedIndex::build(&vecs);
        assert_eq!(index.intent_count(), 3);

        // "cancel" matches cancel_order (1.0) and refund (0.3)
        let results = index.search(&terms(&["cancel"]), 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "cancel_order");

        // "track shipping" matches only track_order
        let results = index.search(&terms(&["track", "shipping"]), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "track_order");
    }

    #[test]
    fn multi_term_aggregation() {
        let vecs = intents(&[
            ("cancel_order", &[("cancel", 1.0), ("order", 0.9)]),
            ("track_order", &[("track", 1.0), ("order", 0.8)]),
        ]);

        let index = InvertedIndex::build(&vecs);

        // "cancel order" -> cancel_order wins. "cancel" is unique (IDF boost),
        // "order" is shared (IDF=1.0). Score > raw 1.9 due to IDF.
        let results = index.search(&terms(&["cancel", "order"]), 10);
        assert_eq!(results[0].id, "cancel_order");
        assert!(results[0].score > 1.9); // IDF boosts unique "cancel"
        // track_order only matches "order", so cancel_order should be significantly ahead
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn top_k_truncation() {
        let mut vecs = HashMap::new();
        for i in 0..100 {
            vecs.insert(
                format!("intent_{}", i),
                LearnedVector::from_seed(seed(&[("common", 0.5 + (i as f32) * 0.001)])),
            );
        }

        let index = InvertedIndex::build(&vecs);
        let results = index.search(&terms(&["common"]), 5);
        assert_eq!(results.len(), 5);

        for i in 0..4 {
            assert!(results[i].score >= results[i + 1].score);
        }
    }

    #[test]
    fn incremental_update() {
        let vecs = intents(&[
            ("a", &[("cancel", 1.0)]),
            ("b", &[("track", 1.0)]),
        ]);

        let mut index = InvertedIndex::build(&vecs);

        // Add "refund" to intent "a"
        let mut v = LearnedVector::from_seed(seed(&[("cancel", 1.0), ("refund", 0.8)]));
        v.learn_with_rate(&terms(&["return"]), 0.5);
        index.update_intent("a", &v);

        let results = index.search(&terms(&["refund"]), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn remove_intent() {
        let vecs = intents(&[
            ("a", &[("cancel", 1.0)]),
            ("b", &[("cancel", 0.5)]),
        ]);

        let mut index = InvertedIndex::build(&vecs);
        index.remove_intent("a");

        let results = index.search(&terms(&["cancel"]), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn no_match_returns_empty() {
        let vecs = intents(&[("a", &[("cancel", 1.0)])]);
        let index = InvertedIndex::build(&vecs);
        assert!(index.search(&terms(&["nonexistent"]), 10).is_empty());
    }
}
