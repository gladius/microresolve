//! Per-namespace lexical normalization — morph + abbrev.
//!
//! A `LexicalIndex` maps token variants to a canonical form. Both index-time
//! tokenization (when seeds get added) and query-time tokenization (when a
//! user query comes in) apply the same normalization, so equivalent variants
//! are stored and looked up under the same key.
//!
//! Two kinds of equivalence are supported:
//!
//! - **Morph** — inflectional variants of the same lexeme (`child` ⇄
//!   `children`, `predict` ⇄ `predicts` ⇄ `predicting`). Lexical fact,
//!   context-independent.
//! - **Abbrev** — operator-defined references (`RBI` ⇄ `real-time biometric
//!   identification`). Per-namespace shorthand for domain phrases.
//!
//! We deliberately do NOT support synonyms (the `cancel` ≈ `abort` kind).
//! Synonyms are semantic claims that vary by context and were the part of
//! the historical L1 lexical graph that polluted the index across packs.
//! See `_internal/V0_3_LEXICAL_GROUPS_PLAN.md` for the full rationale.
//!
//! Groups live in `_ns.json` (loaded by `resolver_persist`); the LexicalIndex
//! rebuilds from them on every namespace load.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LexicalKind {
    Morph,
    Abbrev,
}

/// A single equivalence group: variants of one canonical form, scoped to a
/// language code. Stored as part of `_ns.json` and approved by the operator
/// (manually or via the LLM-suggest review queue).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LexicalGroup {
    pub kind: LexicalKind,
    /// Language code ("en", "fr", "de", "zh", ...). Used by the LLM
    /// suggester to scope morphology proposals; runtime normalization is
    /// language-blind (a token simply matches or doesn't).
    pub lang: String,
    /// Lowercase canonical form. All variants normalize to this.
    pub canonical: String,
    /// Lowercase variant tokens. Should include the canonical itself.
    pub variants: Vec<String>,
}

/// Built from a `Vec<LexicalGroup>` at namespace-load time. Provides O(1)
/// variant → canonical lookup.
#[derive(Clone, Debug, Default)]
pub struct LexicalIndex {
    /// variant_lowercase → canonical_lowercase
    by_variant: HashMap<String, String>,
}

impl LexicalIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a normalization index from a list of approved groups. Later
    /// groups that conflict with earlier ones (same variant, different
    /// canonical) are silently dropped — load-time validation is the
    /// caller's job.
    pub fn from_groups(groups: &[LexicalGroup]) -> Self {
        let mut by_variant: HashMap<String, String> = HashMap::new();
        for g in groups {
            let canonical = g.canonical.to_lowercase();
            for v in &g.variants {
                let variant = v.to_lowercase();
                if variant.is_empty() {
                    continue;
                }
                by_variant
                    .entry(variant)
                    .or_insert_with(|| canonical.clone());
            }
            // Ensure the canonical itself maps to itself.
            by_variant.entry(canonical.clone()).or_insert(canonical);
        }
        Self { by_variant }
    }

    /// Returns true if this index has any groups loaded. Callers can
    /// short-circuit normalization when empty (the common case for packs
    /// without authored groups).
    pub fn is_empty(&self) -> bool {
        self.by_variant.is_empty()
    }

    /// Map a single token to its canonical form. Returns the original token
    /// (as a borrowed slice when no rewrite happens, or an owned String
    /// when it does) — callers can collect into `Vec<String>` cheaply.
    pub fn normalize<'a>(&'a self, token: &'a str) -> &'a str {
        if self.by_variant.is_empty() {
            return token;
        }
        // Tokens may carry the `not_` negation prefix from the tokenizer.
        // Normalize the base form and re-prefix if needed.
        if let Some(base) = token.strip_prefix("not_") {
            self.by_variant
                .get(base)
                .map(|s| s.as_str())
                .unwrap_or(base)
        } else {
            self.by_variant
                .get(token)
                .map(|s| s.as_str())
                .unwrap_or(token)
        }
    }

    /// Rewrite each token in `tokens` to its canonical form. Preserves the
    /// `not_` negation prefix when a normalized form is found for the base.
    pub fn normalize_in_place(&self, tokens: &mut [String]) {
        if self.by_variant.is_empty() {
            return;
        }
        for tok in tokens.iter_mut() {
            if let Some(base) = tok.strip_prefix("not_") {
                if let Some(canonical) = self.by_variant.get(base) {
                    *tok = format!("not_{}", canonical);
                }
            } else if let Some(canonical) = self.by_variant.get(tok.as_str()) {
                *tok = canonical.clone();
            }
        }
    }

    /// Diagnostic helper: how many distinct variants are registered.
    pub fn variant_count(&self) -> usize {
        self.by_variant.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn morph_group(canonical: &str, variants: &[&str]) -> LexicalGroup {
        LexicalGroup {
            kind: LexicalKind::Morph,
            lang: "en".into(),
            canonical: canonical.into(),
            variants: variants.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn empty_index_is_identity() {
        let idx = LexicalIndex::new();
        assert!(idx.is_empty());
        let mut tokens = vec!["foo".to_string(), "bar".to_string()];
        idx.normalize_in_place(&mut tokens);
        assert_eq!(tokens, vec!["foo", "bar"]);
    }

    #[test]
    fn morph_normalizes_plural_to_singular() {
        let groups = vec![morph_group("child", &["child", "children", "child's"])];
        let idx = LexicalIndex::from_groups(&groups);
        assert_eq!(idx.normalize("children"), "child");
        assert_eq!(idx.normalize("child"), "child");
        assert_eq!(idx.normalize("child's"), "child");
        assert_eq!(idx.normalize("dog"), "dog");
    }

    #[test]
    fn normalize_in_place_rewrites() {
        let groups = vec![
            morph_group("child", &["child", "children"]),
            morph_group("warrant", &["warrant", "warrants"]),
        ];
        let idx = LexicalIndex::from_groups(&groups);
        let mut tokens = vec![
            "missing".to_string(),
            "children".to_string(),
            "outstanding".to_string(),
            "warrants".to_string(),
        ];
        idx.normalize_in_place(&mut tokens);
        assert_eq!(tokens, vec!["missing", "child", "outstanding", "warrant"]);
    }

    #[test]
    fn negation_prefix_survives_normalization() {
        let groups = vec![morph_group("child", &["child", "children"])];
        let idx = LexicalIndex::from_groups(&groups);
        let mut tokens = vec!["not_children".to_string(), "not_dog".to_string()];
        idx.normalize_in_place(&mut tokens);
        assert_eq!(tokens, vec!["not_child", "not_dog"]);
    }

    #[test]
    fn abbrev_kind_round_trips() {
        // Same machinery serves both kinds. Single-token abbreviations
        // (RBI → "real-time biometric identification") work; multi-token
        // abbreviations would need to expand at index/query time before
        // tokenization, which is a separate feature (out of scope for
        // Phase 1).
        let groups = vec![LexicalGroup {
            kind: LexicalKind::Abbrev,
            lang: "en".into(),
            canonical: "rbi".into(),
            variants: vec!["rbi".into()],
        }];
        let idx = LexicalIndex::from_groups(&groups);
        assert_eq!(idx.normalize("rbi"), "rbi");
    }

    #[test]
    fn first_canonical_wins_on_conflict() {
        // Author error: same variant in two groups. First registration
        // wins; second is dropped. Logged elsewhere as a warning.
        let groups = vec![
            morph_group("child", &["child", "children"]),
            morph_group("kid", &["kid", "children"]), // children already taken
        ];
        let idx = LexicalIndex::from_groups(&groups);
        assert_eq!(idx.normalize("children"), "child");
    }

    #[test]
    fn variant_count_reflects_loaded_groups() {
        let groups = vec![morph_group("child", &["child", "children", "child's"])];
        let idx = LexicalIndex::from_groups(&groups);
        // 3 variants + canonical (child is already in variants, so 3 total).
        assert_eq!(idx.variant_count(), 3);
    }

    #[test]
    fn case_insensitivity_at_load_time() {
        let groups = vec![LexicalGroup {
            kind: LexicalKind::Morph,
            lang: "en".into(),
            canonical: "CHILD".into(),
            variants: vec!["Child".into(), "CHILDREN".into()],
        }];
        let idx = LexicalIndex::from_groups(&groups);
        assert_eq!(idx.normalize("child"), "child");
        assert_eq!(idx.normalize("children"), "child");
    }
}
