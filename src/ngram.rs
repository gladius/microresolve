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

use crate::{FxHashMap, FxHashSet};

const N: usize = 3;
const MIN_TERM_LEN: usize = 4;

// Two-pass Jaccard: strict first (fast path for well-spelled queries), loose
// fallback only when strict finds nothing (catches transpositions like "cahnge"
// which share only "nge" with "change" → Jaccard 0.14). This keeps the normal-case
// latency at the original ~10µs level; only typo queries pay the extra pass.
const JACCARD_STRICT: f32 = 0.35;
// Loose pass Jaccard is length-tiered: short words need stricter filter to avoid
// real-word errors (e.g. "loose"/"close" are edit-dist 1 but unrelated).
const JACCARD_LOOSE_SHORT: f32 = 0.30; // length 4-5
const JACCARD_LOOSE_LONG: f32 = 0.10; // length ≥ 6

// Max Damerau-Levenshtein edit distance between query and correction.
const MAX_EDIT_DIST_SHORT: usize = 1; // length == 4
const MAX_EDIT_DIST_LONG: usize = 2; // length ≥ 5

#[derive(Clone)]
pub struct NgramIndex {
    /// trigram → vocab term indices that contain it
    index: FxHashMap<String, Vec<usize>>,
    /// all known vocabulary terms
    vocab: Vec<String>,
    /// O(1) membership check
    vocab_set: FxHashSet<String>,
}

impl Default for NgramIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl NgramIndex {
    pub fn new() -> Self {
        Self {
            index: FxHashMap::default(),
            vocab: Vec::new(),
            vocab_set: FxHashSet::default(),
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
        if self.vocab_set.contains(&term) || term.len() < MIN_TERM_LEN {
            return;
        }
        let vi = self.vocab.len();
        for ng in char_ngrams(&term, N) {
            self.index.entry(ng).or_default().push(vi);
        }
        self.vocab_set.insert(term.clone());
        self.vocab.push(term);
    }

    pub fn len(&self) -> usize {
        self.vocab.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    /// Best-match for an unknown token. Returns None if token is already in
    /// vocabulary, too short, or no match above threshold.
    pub fn best_match(&self, token: &str) -> Option<String> {
        if self.vocab_set.contains(token) {
            return None;
        }
        if token.chars().count() < MIN_TERM_LEN {
            return None;
        }

        let query_ngs: FxHashSet<String> = char_ngrams(token, N).into_iter().collect();
        if query_ngs.is_empty() {
            return None;
        }

        let mut hits: FxHashMap<usize, usize> = FxHashMap::default();
        for ng in &query_ngs {
            if let Some(idxs) = self.index.get(ng) {
                for &vi in idxs {
                    *hits.entry(vi).or_insert(0) += 1;
                }
            }
        }

        let query_chars: Vec<char> = token.chars().collect();
        let query_len = query_chars.len();
        let max_dist = if query_len >= 5 {
            MAX_EDIT_DIST_LONG
        } else {
            MAX_EDIT_DIST_SHORT
        };
        let loose_jacc = if query_len >= 6 {
            JACCARD_LOOSE_LONG
        } else {
            JACCARD_LOOSE_SHORT
        };

        // Compute Jaccard for each hit once; reuse across passes.
        let scored: Vec<(usize, f32)> = hits
            .into_iter()
            .map(|(vi, intersection)| {
                let term = &self.vocab[vi];
                let term_ng_count = term.chars().count().saturating_sub(N - 1).max(1);
                let union = query_ngs.len() + term_ng_count - intersection;
                let jaccard = intersection as f32 / union as f32;
                (vi, jaccard)
            })
            .collect();

        // Fast path: strict Jaccard ≥ 0.35 catches well-spelled queries quickly.
        let strict: Vec<(usize, f32)> = scored
            .iter()
            .filter(|(_, j)| *j >= JACCARD_STRICT)
            .cloned()
            .collect();
        let candidates = if !strict.is_empty() {
            strict
        } else {
            // Slow path: loose Jaccard fallback. Only runs when no strict candidate
            // exists — typically means a transposition typo.
            scored
                .into_iter()
                .filter(|(_, j)| *j >= loose_jacc)
                .collect()
        };

        candidates
            .into_iter()
            .filter_map(|(vi, _)| {
                let term = &self.vocab[vi];
                let term_chars: Vec<char> = term.chars().collect();
                // Block corrections that shrink the word by 2+ chars — those are
                // substring collapses ("vacation"→"action", "oncall"→"call"), not typos.
                if query_chars.len() > term_chars.len() + 1 {
                    return None;
                }
                let dist = edit_distance(&query_chars, &term_chars);
                if dist <= max_dist {
                    Some((vi, dist))
                } else {
                    None
                }
            })
            .min_by(|(ai, ad), (bi, bd)| {
                ad.cmp(bd)
                    .then_with(|| self.vocab[*ai].cmp(&self.vocab[*bi]))
            })
            .map(|(vi, _)| self.vocab[vi].clone())
    }

    /// Correct all whitespace-split tokens in a query.
    /// CJK tokens pass through unchanged.
    pub fn correct_query(&self, query: &str) -> String {
        if self.is_empty() {
            return query.to_string();
        }
        query
            .split_whitespace()
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

/// Damerau-Levenshtein edit distance (Damerau, 1964).
/// Treats adjacent character transposition ("cahnge" ↔ "change") as a single edit.
/// This is the "Optimal String Alignment" variant — sufficient for typo correction
/// and O(mn) time. Language-agnostic; operates on Unicode codepoints.
fn edit_distance(a: &[char], b: &[char]) -> usize {
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Full 2D matrix required so we can peek at dp[i-2][j-2] for the transposition rule.
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    #[allow(clippy::needless_range_loop)]
    for i in 0..=m {
        dp[i][0] = i;
    }
    #[allow(clippy::needless_range_loop)]
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let mut best = dp[i - 1][j - 1] + cost; // substitute / match
            if dp[i - 1][j] + 1 < best {
                best = dp[i - 1][j] + 1;
            } // delete
            if dp[i][j - 1] + 1 < best {
                best = dp[i][j - 1] + 1;
            } // insert
              // Transposition: swapping a[i-1] with a[i-2] matches b[j-2..j]
            if i >= 2
                && j >= 2
                && a[i - 1] == b[j - 2]
                && a[i - 2] == b[j - 1]
                && dp[i - 2][j - 2] + 1 < best
            {
                best = dp[i - 2][j - 2] + 1;
            }
            dp[i][j] = best;
        }
    }
    dp[m][n]
}

fn char_ngrams(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < n {
        return if chars.is_empty() {
            vec![]
        } else {
            vec![s.to_string()]
        };
    }
    chars.windows(n).map(|w| w.iter().collect()).collect()
}

/// Build a combined vocabulary from the morphology graph + scoring index for a namespace.
pub fn build_for_namespace(
    lexical: Option<&crate::scoring::LexicalGraph>,
    intent_graph: Option<&crate::scoring::IntentIndex>,
) -> NgramIndex {
    let mut terms: FxHashSet<String> = FxHashSet::default();

    // Always include L2 words — these are the scoring vocabulary.
    let l2_words: FxHashSet<&str> = if let Some(ig) = intent_graph {
        let w: FxHashSet<&str> = ig.word_intent.keys().map(|s| s.as_str()).collect();
        terms.extend(ig.word_intent.keys().cloned());
        w
    } else {
        FxHashSet::default()
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

    // Sort before building so vocab indices are deterministic regardless of
    // HashMap/HashSet iteration order. Without this, same-edit-distance candidates
    // in best_match are chosen by HashMap order (varies per process run).
    let mut sorted: Vec<String> = terms.into_iter().collect();
    sorted.sort_unstable();
    NgramIndex::build(sorted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrects_typo() {
        let idx = NgramIndex::build([
            "cancel".to_string(),
            "subscription".to_string(),
            "refund".to_string(),
        ]);
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
