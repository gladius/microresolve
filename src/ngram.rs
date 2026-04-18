//! Character n-gram index — Layer 0 typo correction.
//!
//! Before L1 normalization, each query token is checked against the known
//! vocabulary. Unknown tokens are corrected to the closest vocabulary term
//! by Jaccard similarity of character trigrams.
//!
//! - No training required — purely algorithmic
//! - Language-agnostic (character-level, any script)
//! - CJK tokens pass through unchanged (CJK has no space-separated misspellings)
//! - Only corrects tokens ≥ 4 chars and with Jaccard ≥ 0.5

use std::collections::{HashMap, HashSet};

const N: usize = 3;
const MIN_SIMILARITY: f32 = 0.35;
const MIN_TERM_LEN: usize = 4;

pub struct NgramIndex {
    /// trigram → vocab term indices that contain it
    index: HashMap<String, Vec<usize>>,
    /// all known vocabulary terms
    vocab: Vec<String>,
    /// O(1) membership check
    vocab_set: HashSet<String>,
}

impl Default for NgramIndex {
    fn default() -> Self { Self::new() }
}

impl NgramIndex {
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            vocab: Vec::new(),
            vocab_set: HashSet::new(),
        }
    }

    /// Build from an iterator of vocabulary terms.
    pub fn build(terms: impl IntoIterator<Item = String>) -> Self {
        let mut idx = Self::new();
        for term in terms {
            idx.add(term);
        }
        idx
    }

    fn add(&mut self, term: String) {
        if self.vocab_set.contains(&term) || term.len() < MIN_TERM_LEN { return; }
        let vi = self.vocab.len();
        for ng in char_ngrams(&term, N) {
            self.index.entry(ng).or_default().push(vi);
        }
        self.vocab_set.insert(term.clone());
        self.vocab.push(term);
    }

    pub fn len(&self) -> usize { self.vocab.len() }
    pub fn is_empty(&self) -> bool { self.vocab.is_empty() }

    /// Best-match for an unknown token. Returns None if token is already in
    /// vocabulary, too short, or no match above threshold.
    pub fn best_match(&self, token: &str) -> Option<String> {
        if self.vocab_set.contains(token) { return None; }
        if token.chars().count() < MIN_TERM_LEN { return None; }

        let query_ngs: HashSet<String> = char_ngrams(token, N).into_iter().collect();
        if query_ngs.is_empty() { return None; }

        let mut hits: HashMap<usize, usize> = HashMap::new();
        for ng in &query_ngs {
            if let Some(idxs) = self.index.get(ng) {
                for &vi in idxs { *hits.entry(vi).or_insert(0) += 1; }
            }
        }

        let query_chars: Vec<char> = token.chars().collect();
        let candidates: Vec<(usize, f32)> = hits.into_iter()
            .filter_map(|(vi, intersection)| {
                let term = &self.vocab[vi];
                let term_ng_count = term.chars().count().saturating_sub(N - 1).max(1);
                let union = query_ngs.len() + term_ng_count - intersection;
                let jaccard = intersection as f32 / union as f32;
                if jaccard >= MIN_SIMILARITY { Some((vi, jaccard)) } else { None }
            })
            .collect();

        // Re-rank by edit distance to avoid polysemy collisions (e.g. "creting"→"meeting" vs "creating")
        candidates.into_iter()
            .map(|(vi, _)| {
                let term = &self.vocab[vi];
                let term_chars: Vec<char> = term.chars().collect();
                let dist = edit_distance(&query_chars, &term_chars);
                (vi, dist)
            })
            .min_by_key(|(_, dist)| *dist)
            .map(|(vi, _)| self.vocab[vi].clone())
    }

    /// Correct all whitespace-split tokens in a query.
    /// CJK tokens pass through unchanged.
    pub fn correct_query(&self, query: &str) -> String {
        if self.is_empty() { return query.to_string(); }
        query.split_whitespace()
            .map(|tok| {
                if tok.chars().any(crate::tokenizer::is_cjk) {
                    tok.to_string()
                } else {
                    self.best_match(tok).unwrap_or_else(|| tok.to_string())
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn edit_distance(a: &[char], b: &[char]) -> usize {
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![0usize; n + 1];
    for j in 0..=n { dp[j] = j; }
    for i in 1..=m {
        let mut prev = dp[0];
        dp[0] = i;
        for j in 1..=n {
            let temp = dp[j];
            dp[j] = if a[i-1] == b[j-1] { prev } else { 1 + prev.min(dp[j]).min(dp[j-1]) };
            prev = temp;
        }
    }
    dp[n]
}

fn char_ngrams(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < n {
        return if chars.is_empty() { vec![] } else { vec![s.to_string()] };
    }
    chars.windows(n).map(|w| w.iter().collect()).collect()
}

/// Build a combined vocabulary from L1 (LexicalGraph) + L2 (IntentGraph) for a namespace.
pub fn build_for_namespace(
    lexical: Option<&crate::scoring::LexicalGraph>,
    intent_graph: Option<&crate::scoring::IntentGraph>,
) -> NgramIndex {
    let mut terms: HashSet<String> = HashSet::new();

    // Always include L2 words — these are the scoring vocabulary.
    let l2_words: HashSet<&str> = if let Some(ig) = intent_graph {
        let w: HashSet<&str> = ig.word_intent.keys().map(|s| s.as_str()).collect();
        terms.extend(ig.word_intent.keys().cloned());
        w
    } else {
        HashSet::new()
    };

    // Include L1 "from" words only when their target is in L2.
    // This keeps L0 focused: it only corrects to words that actually score,
    // preventing 177K generic WordNet terms from polluting the correction space.
    if let Some(g) = lexical {
        for (from, edges) in &g.edges {
            for e in edges {
                if l2_words.contains(e.target.as_str()) {
                    terms.insert(from.clone());
                    terms.insert(e.target.clone());
                }
            }
        }
    }

    NgramIndex::build(terms.into_iter())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrects_typo() {
        let idx = NgramIndex::build(["cancel".to_string(), "subscription".to_string(), "refund".to_string()]);
        // "cancell" → "cancel"
        assert_eq!(idx.best_match("cancell"), Some("cancel".to_string()));
    }

    #[test]
    fn passes_through_known_token() {
        let idx = NgramIndex::build(["cancel".to_string()]);
        assert_eq!(idx.best_match("cancel"), None);
    }

    #[test]
    fn short_tokens_skipped() {
        let idx = NgramIndex::build(["cat".to_string()]);
        assert_eq!(idx.best_match("cot"), None); // too short
    }

    #[test]
    fn correct_query_leaves_cjk_alone() {
        let idx = NgramIndex::build(["cancel".to_string()]);
        let q = "cancell 取消";
        let result = idx.correct_query(q);
        assert!(result.contains("取消"));
        assert!(result.contains("cancel"));
    }
}
