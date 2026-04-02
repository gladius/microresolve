//! LearnedVector — dual-layer sparse vector with asymptotic learning.
//!
//! Each intent has two term-weight layers:
//! - **seed**: Generated at setup time (immutable during operation)
//! - **learned**: Accumulated from user corrections (grows asymptotically toward 1.0)
//!
//! Scoring takes `max(seed, learned)` per term — layers compete, not compound.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default learning rate for positive feedback.
const DEFAULT_LEARNING_RATE: f32 = 0.15;

/// Default unlearn rate for negative feedback.
const DEFAULT_UNLEARN_RATE: f32 = 0.1;

/// Terms below this weight are pruned during decay.
const MIN_WEIGHT: f32 = 0.01;

/// Adaptive sparse vector with seed/learned separation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedVector {
    seed_terms: HashMap<String, f32>,
    learned_terms: HashMap<String, f32>,
}

impl Default for LearnedVector {
    fn default() -> Self {
        Self::new()
    }
}

impl LearnedVector {
    /// Create an empty vector.
    pub fn new() -> Self {
        Self {
            seed_terms: HashMap::new(),
            learned_terms: HashMap::new(),
        }
    }

    /// Create from seed terms (at setup time).
    pub fn from_seed(seed_terms: HashMap<String, f32>) -> Self {
        Self {
            seed_terms,
            learned_terms: HashMap::new(),
        }
    }

    /// Create from both layers (loading from persistence).
    pub fn from_parts(seed_terms: HashMap<String, f32>, learned_terms: HashMap<String, f32>) -> Self {
        Self { seed_terms, learned_terms }
    }

    /// Replace seed terms (preserves learned).
    pub fn set_seed_terms(&mut self, terms: HashMap<String, f32>) {
        self.seed_terms = terms;
    }

    /// Replace learned terms (preserves seed).
    pub fn set_learned_terms(&mut self, terms: HashMap<String, f32>) {
        self.learned_terms = terms;
    }

    /// Get seed terms (for persistence).
    pub fn seed_terms(&self) -> &HashMap<String, f32> {
        &self.seed_terms
    }

    /// Get learned terms (for persistence).
    pub fn learned_terms(&self) -> &HashMap<String, f32> {
        &self.learned_terms
    }

    /// Score a query against this vector.
    ///
    /// For each query term, takes `max(seed_weight, learned_weight)` then sums.
    /// This means layers compete — the stronger signal wins per term.
    pub fn score(&self, query_terms: &[String]) -> f32 {
        query_terms
            .iter()
            .map(|t| {
                let seed = self.seed_terms.get(t).copied().unwrap_or(0.0);
                let learned = self.learned_terms.get(t).copied().unwrap_or(0.0);
                seed.max(learned)
            })
            .sum()
    }

    /// Positive feedback — reinforce terms with default learning rate.
    ///
    /// Uses asymptotic growth: `w' = w + 0.15 * (1 - w)`.
    /// Weight approaches 1.0 but never exceeds it.
    pub fn learn(&mut self, query_terms: &[String]) {
        self.learn_with_rate(query_terms, DEFAULT_LEARNING_RATE);
    }

    /// Positive feedback with custom learning rate.
    pub fn learn_with_rate(&mut self, query_terms: &[String], learning_rate: f32) {
        for term in query_terms {
            if term.is_empty() {
                continue;
            }
            let weight = self.learned_terms.entry(term.clone()).or_insert(0.0);
            *weight += learning_rate * (1.0 - *weight);
        }
    }

    /// Negative feedback — weaken learned terms. Never touches seed.
    pub fn unlearn(&mut self, query_terms: &[String]) {
        self.unlearn_with_rate(query_terms, DEFAULT_UNLEARN_RATE);
    }

    /// Negative feedback with custom rate.
    pub fn unlearn_with_rate(&mut self, query_terms: &[String], rate: f32) {
        for term in query_terms {
            if let Some(weight) = self.learned_terms.get_mut(term) {
                *weight = (*weight - rate).max(0.0);
            }
        }
    }

    /// Periodic decay to forget stale associations.
    ///
    /// Multiplies all learned weights by `factor` (e.g., 0.9).
    /// Prunes terms that fall below threshold.
    pub fn decay(&mut self, factor: f32) {
        self.learned_terms.retain(|_, w| {
            *w *= factor;
            *w > MIN_WEIGHT
        });
    }

    /// Effective terms: union of seed + learned, max weight per term.
    pub fn effective_terms(&self) -> HashMap<String, f32> {
        let mut merged = self.seed_terms.clone();
        for (term, &weight) in &self.learned_terms {
            let existing = merged.entry(term.clone()).or_insert(0.0);
            *existing = existing.max(weight);
        }
        merged
    }

    /// Total number of effective terms.
    pub fn term_count(&self) -> usize {
        self.effective_terms().len()
    }

    /// Check if this vector has any terms at all.
    pub fn is_empty(&self) -> bool {
        self.seed_terms.is_empty() && self.learned_terms.is_empty()
    }

    /// Number of learned terms (excluding seed).
    pub fn learned_term_count(&self) -> usize {
        self.learned_terms.len()
    }

    /// Check if seed layer has terms.
    pub fn is_seeded(&self) -> bool {
        !self.seed_terms.is_empty()
    }

    /// Check if any learning has occurred.
    pub fn has_learned(&self) -> bool {
        !self.learned_terms.is_empty()
    }

    /// Merge another vector's learned layer into this one using max() per term.
    ///
    /// This is a CRDT merge — commutative, associative, idempotent.
    /// Seed layer is never touched. Only learned weights are combined.
    /// For each term, the result is `max(self.learned, other.learned)`.
    pub fn merge_learned(&mut self, other: &LearnedVector) {
        for (term, &weight) in &other.learned_terms {
            let existing = self.learned_terms.entry(term.clone()).or_insert(0.0);
            *existing = existing.max(weight);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    fn seed(pairs: &[(&str, f32)]) -> HashMap<String, f32> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn empty_vector_scores_zero() {
        let v = LearnedVector::new();
        assert_eq!(v.score(&terms(&["charge", "payment"])), 0.0);
        assert!(v.is_empty());
        assert!(!v.is_seeded());
    }

    #[test]
    fn seed_only_scoring() {
        let v = LearnedVector::from_seed(seed(&[
            ("charge", 1.0), ("payment", 0.9), ("card", 0.8),
        ]));
        assert!(v.is_seeded());
        assert!(!v.has_learned());

        let score = v.score(&terms(&["charge", "card"]));
        assert!((score - 1.8).abs() < 0.001);

        assert_eq!(v.score(&terms(&["repository"])), 0.0);
    }

    #[test]
    fn learning_adds_terms() {
        let mut v = LearnedVector::from_seed(seed(&[("charge", 1.0)]));
        v.learn(&terms(&["bill", "customer"]));
        assert!(v.has_learned());
        let score = v.score(&terms(&["bill"]));
        assert!(score > 0.0);
        assert!(score < 1.0);
        assert_eq!(v.score(&terms(&["charge"])), 1.0);
    }

    #[test]
    fn asymptotic_growth_bounded() {
        let mut v = LearnedVector::new();
        for _ in 0..100 {
            v.learn(&terms(&["payment"]));
        }
        let score = v.score(&terms(&["payment"]));
        assert!(score > 0.9);
        assert!(score <= 1.0);
    }

    #[test]
    fn max_not_sum() {
        let mut v = LearnedVector::from_seed(seed(&[("charge", 0.5)]));
        for _ in 0..20 {
            v.learn(&terms(&["charge"]));
        }
        let score = v.score(&terms(&["charge"]));
        assert!(score > 0.5);
        assert!(score <= 1.0);
    }

    #[test]
    fn unlearn_reduces_weight() {
        let mut v = LearnedVector::new();
        for _ in 0..10 {
            v.learn(&terms(&["wrong"]));
        }
        let before = v.score(&terms(&["wrong"]));
        v.unlearn(&terms(&["wrong"]));
        assert!(v.score(&terms(&["wrong"])) < before);
    }

    #[test]
    fn unlearn_never_touches_seed() {
        let mut v = LearnedVector::from_seed(seed(&[("charge", 1.0)]));
        v.unlearn(&terms(&["charge"]));
        assert_eq!(v.score(&terms(&["charge"])), 1.0);
    }

    #[test]
    fn decay_shrinks_learned() {
        let mut v = LearnedVector::new();
        for _ in 0..10 {
            v.learn(&terms(&["payment"]));
        }
        let before = v.score(&terms(&["payment"]));
        v.decay(0.5);
        assert!(v.score(&terms(&["payment"])) < before);
    }

    #[test]
    fn decay_prunes_tiny_weights() {
        let mut v = LearnedVector::new();
        v.learn_with_rate(&terms(&["tiny"]), 0.02);
        v.decay(0.1);
        assert_eq!(v.learned_terms().len(), 0);
    }

    #[test]
    fn decay_preserves_seed() {
        let mut v = LearnedVector::from_seed(seed(&[("charge", 1.0)]));
        v.learn(&terms(&["payment"]));
        v.decay(0.1);
        assert_eq!(v.score(&terms(&["charge"])), 1.0);
    }

    #[test]
    fn effective_terms_merges_correctly() {
        let mut v = LearnedVector::from_seed(seed(&[
            ("charge", 0.5), ("payment", 0.9),
        ]));
        v.learn_with_rate(&terms(&["charge", "bill"]), 0.8);
        let effective = v.effective_terms();
        assert!(*effective.get("charge").unwrap() > 0.5);
        assert_eq!(*effective.get("payment").unwrap(), 0.9);
        assert!(effective.contains_key("bill"));
    }

    #[test]
    fn empty_terms_ignored() {
        let mut v = LearnedVector::new();
        v.learn(&terms(&["", "valid", ""]));
        assert_eq!(v.learned_terms().len(), 1);
    }

    #[test]
    fn merge_learned_takes_max() {
        let mut v1 = LearnedVector::from_seed(seed(&[("charge", 0.5)]));
        v1.learn_with_rate(&terms(&["bill"]), 0.3);
        v1.learn_with_rate(&terms(&["payment"]), 0.8);

        let mut v2 = LearnedVector::from_seed(seed(&[("charge", 0.5)]));
        v2.learn_with_rate(&terms(&["bill"]), 0.7);
        v2.learn_with_rate(&terms(&["refund"]), 0.4);

        v1.merge_learned(&v2);

        // bill: max(0.3, 0.7) = 0.7
        assert!((*v1.learned_terms().get("bill").unwrap() - 0.7).abs() < 0.001);
        // payment: only in v1, stays 0.8
        assert!((*v1.learned_terms().get("payment").unwrap() - 0.8).abs() < 0.001);
        // refund: only in v2, gets added as 0.4
        assert!((*v1.learned_terms().get("refund").unwrap() - 0.4).abs() < 0.001);
        // seed is untouched
        assert_eq!(*v1.seed_terms().get("charge").unwrap(), 0.5);
    }

    #[test]
    fn merge_learned_is_commutative() {
        let mut v1a = LearnedVector::new();
        v1a.learn_with_rate(&terms(&["a"]), 0.3);
        v1a.learn_with_rate(&terms(&["b"]), 0.8);

        let mut v2 = LearnedVector::new();
        v2.learn_with_rate(&terms(&["a"]), 0.7);
        v2.learn_with_rate(&terms(&["c"]), 0.5);

        // merge(v1, v2)
        let mut forward = v1a.clone();
        forward.merge_learned(&v2);

        // merge(v2, v1)
        let mut reverse = v2.clone();
        reverse.merge_learned(&v1a);

        // Should be identical
        for key in ["a", "b", "c"] {
            let fw = forward.learned_terms().get(key).copied().unwrap_or(0.0);
            let rv = reverse.learned_terms().get(key).copied().unwrap_or(0.0);
            assert!((fw - rv).abs() < 0.001, "key '{}': forward={}, reverse={}", key, fw, rv);
        }
    }
}
