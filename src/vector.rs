//! LearnedVector — dual-layer sparse vector with asymptotic learning.
//!
//! Each intent has two term-weight layers:
//! - **phrase**: Generated at setup time from training phrases (immutable during operation)
//! - **learned**: Accumulated from user corrections (grows asymptotically toward 1.0)
//!
//! Scoring takes `max(phrase, learned)` per term — layers compete, not compound.
//!
//! ## Storage
//! Both layers are stored as `Vec<(String, f32)>` sorted by term for cache efficiency.
//! A contiguous sorted Vec (~20 entries × 28 bytes ≈ 560 bytes) reloads from cache
//! as a single sequential read. HashMap for the same data uses 3–4× more memory with
//! scattered pointer indirections — expensive when cache is shared with the host server.
//! Binary search on 20 entries (≈ 4 comparisons) beats HashMap's hash + cache miss.
//!
//! Serialization format is unchanged: JSON objects `{"term": weight, ...}`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default learning rate for positive feedback.
const DEFAULT_LEARNING_RATE: f32 = 0.15;

/// Default unlearn rate for negative feedback.
const DEFAULT_UNLEARN_RATE: f32 = 0.1;

/// Terms below this weight are pruned during decay.
const MIN_WEIGHT: f32 = 0.01;

/// Serialize/deserialize `Vec<(String, f32)>` as a JSON object `{"term": weight}`.
/// Keeps the on-disk format identical to the old HashMap representation.
mod sorted_vec_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S>(vec: &Vec<(String, f32)>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Emit as JSON object — same format as HashMap serialization
        let map: HashMap<&str, f32> = vec.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        map.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<(String, f32)>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let map = HashMap::<String, f32>::deserialize(deserializer)?;
        let mut vec: Vec<(String, f32)> = map.into_iter().collect();
        vec.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(vec)
    }
}

/// Convert a HashMap into a sorted Vec.
fn sorted_from_map(map: HashMap<String, f32>) -> Vec<(String, f32)> {
    let mut vec: Vec<(String, f32)> = map.into_iter().collect();
    vec.sort_by(|a, b| a.0.cmp(&b.0));
    vec
}

/// Adaptive sparse vector with phrase/learned separation.
///
/// Both layers stored as sorted Vec for cache-friendly hot-path access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedVector {
    #[serde(alias = "seed_terms", with = "sorted_vec_serde")]
    phrase_terms: Vec<(String, f32)>,
    #[serde(with = "sorted_vec_serde")]
    learned_terms: Vec<(String, f32)>,
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
            phrase_terms: Vec::new(),
            learned_terms: Vec::new(),
        }
    }

    /// Create from phrase terms (at setup time from training phrases).
    pub fn from_phrases(phrase_terms: HashMap<String, f32>) -> Self {
        Self {
            phrase_terms: sorted_from_map(phrase_terms),
            learned_terms: Vec::new(),
        }
    }

    /// Create from both layers (loading from persistence).
    pub fn from_parts(phrase_terms: HashMap<String, f32>, learned_terms: HashMap<String, f32>) -> Self {
        Self {
            phrase_terms: sorted_from_map(phrase_terms),
            learned_terms: sorted_from_map(learned_terms),
        }
    }

    /// Replace phrase terms (preserves learned).
    pub fn set_phrase_terms(&mut self, terms: HashMap<String, f32>) {
        self.phrase_terms = sorted_from_map(terms);
    }

    /// Replace learned terms (preserves phrase).
    pub fn set_learned_terms(&mut self, terms: HashMap<String, f32>) {
        self.learned_terms = sorted_from_map(terms);
    }

    /// Get phrase terms as a HashMap (for persistence/export — not hot path).
    pub fn phrase_terms(&self) -> HashMap<String, f32> {
        self.phrase_terms.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }

    /// Get learned terms as a HashMap (for persistence/export — not hot path).
    pub fn learned_terms(&self) -> HashMap<String, f32> {
        self.learned_terms.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }

    /// Score a query against this vector.
    ///
    /// For each query term, takes `max(phrase_weight, learned_weight)` then sums.
    /// Binary search on sorted Vec — no HashMap overhead, cache-friendly.
    pub fn score(&self, query_terms: &[String]) -> f32 {
        query_terms
            .iter()
            .map(|t| {
                let phrase = self.phrase_terms
                    .binary_search_by(|(k, _)| k.as_str().cmp(t.as_str()))
                    .map(|i| self.phrase_terms[i].1)
                    .unwrap_or(0.0);
                let learned = self.learned_terms
                    .binary_search_by(|(k, _)| k.as_str().cmp(t.as_str()))
                    .map(|i| self.learned_terms[i].1)
                    .unwrap_or(0.0);
                phrase.max(learned)
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
            match self.learned_terms.binary_search_by(|(k, _)| k.as_str().cmp(term.as_str())) {
                Ok(i) => {
                    let w = &mut self.learned_terms[i].1;
                    *w += learning_rate * (1.0 - *w);
                }
                Err(i) => {
                    // Insert in sorted position; initial weight = learning_rate * (1 - 0)
                    self.learned_terms.insert(i, (term.clone(), learning_rate));
                }
            }
        }
    }

    /// Negative feedback — weaken learned terms. Never touches phrase layer.
    pub fn unlearn(&mut self, query_terms: &[String]) {
        self.unlearn_with_rate(query_terms, DEFAULT_UNLEARN_RATE);
    }

    /// Negative feedback with custom rate.
    pub fn unlearn_with_rate(&mut self, query_terms: &[String], rate: f32) {
        for term in query_terms {
            if let Ok(i) = self.learned_terms.binary_search_by(|(k, _)| k.as_str().cmp(term.as_str())) {
                self.learned_terms[i].1 = (self.learned_terms[i].1 - rate).max(0.0);
            }
        }
    }

    /// Periodic decay to forget stale associations.
    ///
    /// Multiplies all learned weights by `factor` (e.g., 0.9).
    /// Prunes terms that fall below threshold.
    pub fn decay(&mut self, factor: f32) {
        self.learned_terms.retain_mut(|(_, w)| {
            *w *= factor;
            *w > MIN_WEIGHT
        });
    }

    /// Check if a term exists in either layer. O(log n) binary search, no allocation.
    pub fn contains_term(&self, term: &str) -> bool {
        self.phrase_terms.binary_search_by(|(k, _)| k.as_str().cmp(term)).is_ok()
            || self.learned_terms.binary_search_by(|(k, _)| k.as_str().cmp(term)).is_ok()
    }

    /// Effective terms: union of phrase + learned, max weight per term.
    pub fn effective_terms(&self) -> HashMap<String, f32> {
        let mut merged: HashMap<String, f32> = self.phrase_terms.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        for (term, weight) in &self.learned_terms {
            let existing = merged.entry(term.clone()).or_insert(0.0);
            *existing = existing.max(*weight);
        }
        merged
    }

    /// Total number of effective terms.
    pub fn term_count(&self) -> usize {
        self.effective_terms().len()
    }

    /// Check if this vector has any terms at all.
    pub fn is_empty(&self) -> bool {
        self.phrase_terms.is_empty() && self.learned_terms.is_empty()
    }

    /// Number of learned terms (excluding phrase layer).
    pub fn learned_term_count(&self) -> usize {
        self.learned_terms.len()
    }

    /// Check if phrase layer has terms.
    pub fn has_phrases(&self) -> bool {
        !self.phrase_terms.is_empty()
    }

    /// Check if any learning has occurred.
    pub fn has_learned(&self) -> bool {
        !self.learned_terms.is_empty()
    }

    /// Merge another vector's learned layer into this one using max() per term.
    ///
    /// This is a CRDT merge — commutative, associative, idempotent.
    /// Phrase layer is never touched. Only learned weights are combined.
    /// For each term, the result is `max(self.learned, other.learned)`.
    pub fn merge_learned(&mut self, other: &LearnedVector) {
        for (term, weight) in &other.learned_terms {
            match self.learned_terms.binary_search_by(|(k, _)| k.as_str().cmp(term.as_str())) {
                Ok(i) => self.learned_terms[i].1 = self.learned_terms[i].1.max(*weight),
                Err(i) => self.learned_terms.insert(i, (term.clone(), *weight)),
            }
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
        assert!(!v.has_phrases());
    }

    #[test]
    fn seed_only_scoring() {
        let v = LearnedVector::from_phrases(seed(&[
            ("charge", 1.0), ("payment", 0.9), ("card", 0.8),
        ]));
        assert!(v.has_phrases());
        assert!(!v.has_learned());

        let score = v.score(&terms(&["charge", "card"]));
        assert!((score - 1.8).abs() < 0.001);

        assert_eq!(v.score(&terms(&["repository"])), 0.0);
    }

    #[test]
    fn learning_adds_terms() {
        let mut v = LearnedVector::from_phrases(seed(&[("charge", 1.0)]));
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
        let mut v = LearnedVector::from_phrases(seed(&[("charge", 0.5)]));
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
    fn unlearn_never_touches_phrase_layer() {
        let mut v = LearnedVector::from_phrases(seed(&[("charge", 1.0)]));
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
    fn decay_preserves_phrase_layer() {
        let mut v = LearnedVector::from_phrases(seed(&[("charge", 1.0)]));
        v.learn(&terms(&["payment"]));
        v.decay(0.1);
        assert_eq!(v.score(&terms(&["charge"])), 1.0);
    }

    #[test]
    fn effective_terms_merges_correctly() {
        let mut v = LearnedVector::from_phrases(seed(&[
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
        let mut v1 = LearnedVector::from_phrases(seed(&[("charge", 0.5)]));
        v1.learn_with_rate(&terms(&["bill"]), 0.3);
        v1.learn_with_rate(&terms(&["payment"]), 0.8);

        let mut v2 = LearnedVector::from_phrases(seed(&[("charge", 0.5)]));
        v2.learn_with_rate(&terms(&["bill"]), 0.7);
        v2.learn_with_rate(&terms(&["refund"]), 0.4);

        v1.merge_learned(&v2);

        let lt = v1.learned_terms();
        // bill: max(0.3, 0.7) = 0.7
        assert!((lt.get("bill").unwrap() - 0.7).abs() < 0.001);
        // payment: only in v1, stays 0.8
        assert!((lt.get("payment").unwrap() - 0.8).abs() < 0.001);
        // refund: only in v2, gets added as 0.4
        assert!((lt.get("refund").unwrap() - 0.4).abs() < 0.001);
        // seed is untouched
        assert_eq!(*v1.phrase_terms().get("charge").unwrap(), 0.5);
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

        let fw = forward.learned_terms();
        let rv = reverse.learned_terms();
        for key in ["a", "b", "c"] {
            let f = fw.get(key).copied().unwrap_or(0.0);
            let r = rv.get(key).copied().unwrap_or(0.0);
            assert!((f - r).abs() < 0.001, "key '{}': forward={}, reverse={}", key, f, r);
        }
    }
}
