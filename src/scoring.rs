//! # Lexical association graph (L1)
//!
//! A weighted word-association graph distilled from LLM knowledge,
//! plus a built-in English morphology base. Replaces the need for
//! pre-trained embeddings by encoding lexical relationships directly
//! as weighted edges.
//!
//! ## Relationship types (by weight tier)
//!
//! | Weight    | Kind          | Query action                                    |
//! |-----------|---------------|-------------------------------------------------|
//! | 0.97–1.0  | Morphological | Normalize (substitute in place)                 |
//! | 0.97–1.0  | Abbreviation  | Normalize (substitute in place)                 |
//! | 0.80–0.96 | Synonym       | Stored only — no query-time injection           |
//! | 0.60–0.79 | Semantic      | Stored only — no query-time injection           |
//!
//! Only Morphological + Abbreviation rewrite the query at runtime. Synonym
//! and Semantic edges are stored for the Studio inspection panel and for
//! offline phrase-generation flows; they are NOT injected into the query.
//! Query-time synonym expansion was removed in v0.0 because expanding
//! `cancel` → `cancel terminate end` inflated L2 scores ("token stuffing")
//! rather than behaving like canonical IR-style index-time expansion.
//! For semantic coverage, generate additional training phrases (via the
//! "Generate phrases" UI flow) instead.
//!
//! ## Example
//! ```no_run
//! use microresolve::scoring::{LexicalGraph, EdgeKind};
//! let mut g = LexicalGraph::new();
//! g.add("canceling", "cancel",       0.99, EdgeKind::Morphological);
//! g.add("sub",       "subscription", 0.99, EdgeKind::Abbreviation);
//!
//! let r = g.preprocess("canceling my sub");
//! assert_eq!(r.normalized, "cancel my subscription");
//! ```

use crate::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// Inflected or derived form of the same lexeme.
    /// "canceling", "cancelled", "cancellation" → "cancel"
    Morphological,
    /// Shortened form → full form.
    /// "pr" → "pull request", "repo" → "repository"
    Abbreviation,
    /// Different word, same meaning in this domain.
    /// "terminate" → "cancel", "ping" → "send"
    Synonym,
    /// Semantically related but context-dependent.
    /// Used to boost concept confidence, NOT for query expansion.
    Semantic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HebbianEdge {
    /// The canonical / target term (always lowercase).
    pub target: String,
    /// Association strength 0.0–1.0.
    pub weight: f32,
    pub kind: EdgeKind,
}

/// Weighted lexical association graph, per namespace.
/// Serializes to JSON alongside the namespace state.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LexicalGraph {
    /// source_term (lowercase) → outgoing edges
    pub edges: HashMap<String, Vec<HebbianEdge>>,
}

// ── Result type ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PreprocessResult {
    pub original: String,
    /// After morphology + abbreviation substitution (the only rewrites that
    /// happen at query time).
    pub normalized: String,
    /// Equal to `normalized` — kept as a separate field for backward-compat
    /// with code paths that read `expanded`. Synonym injection is OFF; this
    /// will never differ from `normalized`.
    pub expanded: String,
    /// Always empty in v0.1 (synonym injection removed). Field retained so
    /// the Studio Layers panel keeps rendering. Deprecated; will go in a
    /// later release.
    pub injected: Vec<String>,
    /// Semantic-weight edges that fired against the query (Studio inspection).
    pub semantic_hits: Vec<(String, String, f32)>, // (source, target, weight)
    pub was_modified: bool,
}

// ── Core impl ─────────────────────────────────────────────────────────────────

impl LexicalGraph {
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    /// Add a directed edge: `from` → `to` with given weight and kind.
    /// Both terms are stored lowercase.
    pub fn add(&mut self, from: &str, to: &str, weight: f32, kind: EdgeKind) {
        self.edges
            .entry(from.to_lowercase())
            .or_default()
            .push(HebbianEdge {
                target: to.to_lowercase(),
                weight,
                kind,
            });
    }

    /// Hebbian update: strengthen an edge from observed routing confirmation.
    /// Uses asymptotic update: Δw = delta × (1 - w), so weight approaches 1.0
    /// but never exceeds it regardless of how many reinforcements occur.
    pub fn reinforce(&mut self, from: &str, to: &str, delta: f32) {
        let from = from.to_lowercase();
        let to = to.to_lowercase();
        if let Some(edges) = self.edges.get_mut(&from) {
            for e in edges.iter_mut() {
                if e.target == to {
                    e.weight = (e.weight + delta * (1.0 - e.weight)).min(1.0);
                    return;
                }
            }
        }
        // Edge didn't exist — create it as a learned synonym
        self.add(&from, &to, 0.60 + delta, EdgeKind::Synonym);
    }

    // ── Query preprocessing ───────────────────────────────────────────────

    /// Token split for L1 substitution — pub(crate) so auto-learn can scan original query words.
    pub fn l1_tokens_pub(query: &str) -> Vec<String> {
        Self::l1_tokens(query)
    }

    /// Token split for Layer 1 substitution.
    /// Latin: whitespace split (preserves stop words so normalized phrases stay coherent).
    /// Sentence-ending punctuation ('.', '!', '?') is preserved as a standalone "." token
    /// so downstream tokenize() calls can split on sentence boundaries and scope negation correctly.
    /// CJK:   tokenizer bigrams (Chinese/Japanese/Korean have no whitespace between words).
    fn l1_tokens(query: &str) -> Vec<String> {
        let lower = query.to_lowercase();
        let has_cjk = lower.chars().any(crate::tokenizer::is_cjk);
        if !has_cjk {
            let mut out = Vec::new();
            for w in lower.split_whitespace() {
                let has_boundary = w.ends_with('.') || w.ends_with('!') || w.ends_with('?');
                let clean: String = w.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
                if !clean.is_empty() {
                    out.push(clean);
                }
                if has_boundary {
                    out.push(".".to_string()); // sentence boundary sentinel
                }
            }
            out
        } else {
            // For CJK, tokenize() gives us bigrams + individual content chars.
            // stop-word filtering is acceptable here — CJK stop chars rarely match edges.
            crate::tokenizer::tokenize(query)
        }
    }

    /// Phase 1: normalize query word-by-word.
    /// Substitutes morphological variants and abbreviations with canonical forms.
    /// Multi-word abbreviation targets are also handled ("pr" → "pull request").
    /// CJK: operates on bigrams from the tokenizer.
    pub fn normalize_query(&self, query: &str) -> String {
        self.normalize_query_with_kinds(query, true, true)
    }

    /// Variant of [`Self::normalize_query`] that lets the caller suppress
    /// morphological or abbreviation substitutions. Used by [`Resolver`]
    /// to honor per-namespace L1 toggles. When both flags are `false`
    /// this is a tokenize-only pass with no substitution.
    pub fn normalize_query_with_kinds(
        &self,
        query: &str,
        allow_morphology: bool,
        allow_abbreviation: bool,
    ) -> String {
        let words = Self::l1_tokens(query);
        let mut out: Vec<String> = Vec::with_capacity(words.len());

        for word in &words {
            let replacement = self.edges.get(word.as_str()).and_then(|edges| {
                edges.iter().find(|e| {
                    let kind_ok = match e.kind {
                        EdgeKind::Morphological => allow_morphology,
                        EdgeKind::Abbreviation => allow_abbreviation,
                        _ => false,
                    };
                    kind_ok && e.weight >= 0.97
                })
            });
            match replacement {
                Some(e) => out.push(e.target.clone()),
                None => out.push(word.clone()),
            }
        }
        out.join(" ")
    }

    // Note: a previous version had `expand_query` here that injected
    // synonym targets into the query string. It was removed because
    // expanding `cancel` → `cancel terminate end` polluted L2 scoring —
    // an "terminate" appearing in a different intent's training would
    // then match "cancel" queries. Synonym edges remain in the graph
    // for Studio inspection + offline phrase generation; they do not
    // mutate queries at runtime. See preprocess() / preprocess_grounded().

    /// Collect semantic-weight hits for concept confidence boosting.
    pub fn semantic_hits(&self, query: &str) -> Vec<(String, String, f32)> {
        let words = Self::l1_tokens(query);
        let mut hits = Vec::new();
        for word in &words {
            if let Some(edges) = self.edges.get(word.as_str()) {
                for edge in edges {
                    if matches!(edge.kind, EdgeKind::Semantic) {
                        hits.push((word.clone(), edge.target.clone(), edge.weight));
                    }
                }
            }
        }
        hits
    }

    /// Full pipeline: normalize → collect semantic signals.
    ///
    /// Synonym *edges* stay in the graph for inspection, semantic confidence boosts,
    /// and future phrase-generation flows. They are NOT used for query-time token
    /// injection — that path was removed because expanding "cancel" → "cancel
    /// terminate end" inflates L2 scores (token stuffing) rather than behaving like
    /// index-time expansion in standard IR systems.
    ///
    /// For semantic coverage, add training phrases (e.g. via "Generate phrases")
    /// instead of relying on synonym edges.
    pub fn preprocess(&self, query: &str) -> PreprocessResult {
        self.preprocess_with_kinds(query, true, true)
    }

    /// L1 preprocessing with per-edge-kind toggles. Synonym substitution
    /// lives in [`Self::preprocess_grounded`] (the OOV-only path) — this
    /// method only normalizes via Morphological / Abbreviation edges, so
    /// it has no synonym knob.
    pub fn preprocess_with_kinds(
        &self,
        query: &str,
        allow_morphology: bool,
        allow_abbreviation: bool,
    ) -> PreprocessResult {
        let normalized =
            self.normalize_query_with_kinds(query, allow_morphology, allow_abbreviation);
        let semantic_hits = self.semantic_hits(&normalized);
        let was_modified = normalized != query.to_lowercase();

        PreprocessResult {
            original: query.to_string(),
            normalized: normalized.clone(),
            expanded: normalized,
            injected: vec![],
            semantic_hits,
            was_modified,
        }
    }

    /// L2-vocabulary-aware preprocessing: applies the standard normalize
    /// (morphological + abbreviation), THEN substitutes Synonym edges
    /// **only when the source word is not in the L2 vocabulary**.
    ///
    /// This is "Option E" — targeted synonym replacement, the OOV-only
    /// variant. Keeps distinctive vocabulary intact (a word someone trained
    /// on stays as itself; its training signal isn't merged into a generic
    /// canonical word). Resorts to the synonym→canonical substitution only
    /// when the query word is entirely unknown to L2 — better than zero
    /// signal in that case.
    ///
    /// This is **substitution, not expansion**: the original token is
    /// replaced (one token → one token), so it cannot inflate L2 scores
    /// the way the old query-expansion path did.
    pub fn preprocess_grounded(
        &self,
        query: &str,
        known_words: &std::collections::HashSet<&str>,
    ) -> PreprocessResult {
        self.preprocess_grounded_with_kinds(query, known_words, true, true, true)
    }

    /// `preprocess_grounded` with per-edge-kind toggles. Used by
    /// [`Resolver`] to honor per-namespace L1 toggles.
    pub fn preprocess_grounded_with_kinds(
        &self,
        query: &str,
        known_words: &std::collections::HashSet<&str>,
        allow_morphology: bool,
        allow_abbreviation: bool,
        allow_synonym: bool,
    ) -> PreprocessResult {
        let words = Self::l1_tokens(query);
        let mut out: Vec<String> = Vec::with_capacity(words.len());
        let mut injected: Vec<String> = Vec::new();

        for word in &words {
            let edges = self.edges.get(word.as_str());
            let is_oov = !known_words.contains(word.as_str());

            // Phase 1: morphological / abbreviation substitution — OOV-gated.
            if is_oov {
                let canon = edges.and_then(|es| {
                    es.iter().find(|e| {
                        let kind_ok = match e.kind {
                            EdgeKind::Morphological => allow_morphology,
                            EdgeKind::Abbreviation => allow_abbreviation,
                            _ => false,
                        };
                        kind_ok && e.weight >= 0.97 && known_words.contains(e.target.as_str())
                    })
                });
                if let Some(e) = canon {
                    out.push(e.target.clone());
                    continue;
                }
            }

            // Phase 2: OOV synonym substitution (Option E) — same gate.
            if is_oov && allow_synonym {
                if let Some(syn) = edges.and_then(|es| {
                    es.iter().find(|e| {
                        matches!(e.kind, EdgeKind::Synonym)
                            && e.weight >= 0.90
                            && known_words.contains(e.target.as_str())
                    })
                }) {
                    injected.push(format!("{} → {}", word, syn.target));
                    out.push(syn.target.clone());
                    continue;
                }
            }

            // Phase 3: otherwise, leave the word as-is.
            out.push(word.clone());
        }

        let expanded = out.join(" ");
        let was_modified = expanded != query.to_lowercase();
        PreprocessResult {
            original: query.to_string(),
            normalized: expanded.clone(),
            expanded,
            injected,
            semantic_hits: vec![],
            was_modified,
        }
    }

    // ── Persistence ───────────────────────────────────────────────────────

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

/// Hand-crafted test graph for the 20-intent SaaS namespace (stripe/shopify/github/slack).
/// Used by unit tests and the standalone demo binary.
pub fn saas_test_graph() -> LexicalGraph {
    let mut g = LexicalGraph::new();

    // ── Morphological (0.99) ─────────────────────────────────────────────
    // cancel family
    for v in &["canceling", "cancelled", "cancellation", "cancels"] {
        g.add(v, "cancel", 0.99, EdgeKind::Morphological);
    }
    // refund family
    for v in &["refunding", "refunded", "refunds"] {
        g.add(v, "refund", 0.99, EdgeKind::Morphological);
    }
    // charge family
    for v in &["charging", "charged", "charges"] {
        g.add(v, "charge", 0.99, EdgeKind::Morphological);
    }
    // ship family
    for v in &["shipped", "shipment", "shipments"] {
        g.add(v, "ship", 0.99, EdgeKind::Morphological);
    }
    // merge family
    for v in &["merging", "merged", "merges"] {
        g.add(v, "merge", 0.99, EdgeKind::Morphological);
    }
    // list family
    for v in &["listing", "listed", "lists"] {
        g.add(v, "list", 0.99, EdgeKind::Morphological);
    }
    // create family
    for v in &["creating", "created", "creates", "creation"] {
        g.add(v, "create", 0.99, EdgeKind::Morphological);
    }
    // schedule family
    for v in &["scheduling", "scheduled", "schedules"] {
        g.add(v, "schedule", 0.99, EdgeKind::Morphological);
    }
    // invite family
    for v in &["inviting", "invited", "invites"] {
        g.add(v, "invite", 0.99, EdgeKind::Morphological);
    }
    // send family
    for v in &["sending", "sent", "sends"] {
        g.add(v, "send", 0.99, EdgeKind::Morphological);
    }
    // close family
    for v in &["closing", "closed", "closes"] {
        g.add(v, "close", 0.99, EdgeKind::Morphological);
    }

    // ── Abbreviations (0.99) ─────────────────────────────────────────────
    g.add("pr", "pull request", 0.99, EdgeKind::Abbreviation);
    g.add("prs", "pull requests", 0.99, EdgeKind::Abbreviation);
    g.add("repo", "repository", 0.99, EdgeKind::Abbreviation);
    g.add("repos", "repositories", 0.99, EdgeKind::Abbreviation);
    g.add("sub", "subscription", 0.99, EdgeKind::Abbreviation);
    g.add("subs", "subscriptions", 0.99, EdgeKind::Abbreviation);
    g.add("msg", "message", 0.99, EdgeKind::Abbreviation);
    g.add("msgs", "messages", 0.99, EdgeKind::Abbreviation);
    g.add("chan", "channel", 0.99, EdgeKind::Abbreviation);

    // ── Synonyms + their morph variants ──────────────────────────────────
    // cancel synonyms
    for (v, w) in &[
        ("terminate", 0.92f32),
        ("terminating", 0.92),
        ("terminated", 0.92),
    ] {
        g.add(v, "cancel", *w, EdgeKind::Synonym);
    }
    for (v, w) in &[("kill", 0.85f32), ("killing", 0.85), ("killed", 0.85)] {
        g.add(v, "cancel", *w, EdgeKind::Synonym);
    }
    for (v, w) in &[("axe", 0.83f32), ("axed", 0.83), ("axing", 0.83)] {
        g.add(v, "cancel", *w, EdgeKind::Synonym);
    }
    g.add("ditch", "cancel", 0.80, EdgeKind::Synonym);

    // send synonyms
    g.add("ping", "send", 0.92, EdgeKind::Synonym);
    g.add("dm", "send", 0.90, EdgeKind::Synonym);
    g.add("notify", "send", 0.85, EdgeKind::Synonym);
    g.add("blast", "send", 0.80, EdgeKind::Synonym);

    // create synonyms
    g.add("spin", "create", 0.82, EdgeKind::Synonym);
    g.add("make", "create", 0.85, EdgeKind::Synonym);
    g.add("build", "create", 0.82, EdgeKind::Synonym);
    // NOTE: "open" excluded — ambiguous ("open an issue" vs "open the settings")

    // refund synonyms
    g.add("reimburse", "refund", 0.90, EdgeKind::Synonym);
    g.add("reimbursement", "refund", 0.90, EdgeKind::Synonym);
    g.add("compensate", "refund", 0.80, EdgeKind::Synonym);

    // charge synonyms
    g.add("run", "charge", 0.82, EdgeKind::Synonym); // "run the card"
    g.add("bill", "charge", 0.85, EdgeKind::Synonym);

    // list synonyms
    g.add("show", "list", 0.85, EdgeKind::Synonym);
    g.add("fetch", "list", 0.82, EdgeKind::Synonym);
    // NOTE: "get", "pull", "open" excluded — too ambiguous as standalone words
    // ("pull request" contains "pull", "open an account" vs "open a file")

    // merge synonyms
    g.add("integrate", "merge", 0.82, EdgeKind::Synonym);
    g.add("squash", "merge", 0.80, EdgeKind::Synonym);

    // ── Semantic (0.60–0.79) — confidence boost only ──────────────────────
    g.add("stop", "cancel", 0.65, EdgeKind::Semantic);
    g.add("end", "cancel", 0.62, EdgeKind::Semantic);
    g.add("drop", "cancel", 0.68, EdgeKind::Semantic);
    g.add("fire", "send", 0.70, EdgeKind::Semantic); // "fire off a message"
    g.add("throw", "create", 0.65, EdgeKind::Semantic); // "throw up a repo"

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Morphological normalization ───────────────────────────────────────

    #[test]
    fn morph_canceling() {
        let g = saas_test_graph();
        assert_eq!(
            g.normalize_query("canceling my subscription"),
            "cancel my subscription"
        );
    }

    #[test]
    fn morph_cancelled() {
        let g = saas_test_graph();
        assert_eq!(
            g.normalize_query("the order was cancelled"),
            "the order was cancel"
        );
    }

    #[test]
    fn morph_multiple_in_one_query() {
        let g = saas_test_graph();
        assert_eq!(
            g.normalize_query("merged the pr and closed the issue"),
            "merge the pull request and close the issue"
        );
    }

    #[test]
    fn morph_shipped() {
        let g = saas_test_graph();
        assert_eq!(
            g.normalize_query("get all shipped orders"),
            "get all ship orders"
        );
    }

    // ── Abbreviation normalization ────────────────────────────────────────

    #[test]
    fn abbrev_sub() {
        let g = saas_test_graph();
        assert_eq!(g.normalize_query("cancel my sub"), "cancel my subscription");
    }

    #[test]
    fn abbrev_pr_and_repo() {
        let g = saas_test_graph();
        assert_eq!(
            g.normalize_query("merge the pr in that repo"),
            "merge the pull request in that repository"
        );
    }

    #[test]
    fn abbrev_msg_chan() {
        let g = saas_test_graph();
        assert_eq!(
            g.normalize_query("send a msg to the chan"),
            "send a message to the channel"
        );
    }

    // ── Morphology + abbreviation combined ────────────────────────────────

    #[test]
    fn combined_morph_and_abbrev() {
        let g = saas_test_graph();
        // "canceling" → "cancel", "sub" → "subscription"
        assert_eq!(
            g.normalize_query("canceling my sub"),
            "cancel my subscription"
        );
    }

    #[test]
    fn combined_morph_then_synonym() {
        let g = saas_test_graph();
        // "canceling" → normalize to "cancel" → no synonym needed (already canonical)
        let r = g.preprocess("canceling my sub");
        assert_eq!(r.normalized, "cancel my subscription");
        // "cancel" is already canonical so nothing extra injected
        assert!(
            r.injected.is_empty(),
            "should not inject anything when already canonical"
        );
    }

    // ── Semantic hits (no expansion) ──────────────────────────────────────

    #[test]
    fn semantic_stop_does_not_expand() {
        let g = saas_test_graph();
        let r = g.preprocess("stop sending me emails");
        // "stop" is Semantic weight 0.65; semantic edges never inject at query time.
        assert!(
            !r.expanded.contains("cancel"),
            "semantic word should not expand query"
        );
        // But it should appear in semantic_hits
        let hit = r
            .semantic_hits
            .iter()
            .any(|(src, tgt, _)| src == "stop" && tgt == "cancel");
        assert!(hit, "stop → cancel should appear as semantic hit");
    }

    #[test]
    fn semantic_end_does_not_expand() {
        let g = saas_test_graph();
        let r = g.preprocess("at the end of the month");
        assert!(!r.expanded.contains("cancel"));
    }

    // ── No modification for clean queries ────────────────────────────────

    #[test]
    fn no_modification_clean_query() {
        let g = saas_test_graph();
        let r = g.preprocess("cancel my subscription");
        assert!(!r.was_modified);
        assert_eq!(r.expanded, "cancel my subscription");
    }

    // ── Hebbian reinforcement ─────────────────────────────────────────────

    #[test]
    fn reinforce_strengthens_existing_edge() {
        let mut g = saas_test_graph();
        // terminate → cancel starts at 0.92
        let before = g.edges["terminate"][0].weight;
        g.reinforce("terminate", "cancel", 0.05);
        let after = g.edges["terminate"][0].weight;
        assert!(after > before, "reinforcement should increase weight");
        assert!(after <= 1.0, "should not exceed 1.0");
    }

    #[test]
    fn reinforce_creates_new_edge() {
        let mut g = saas_test_graph();
        g.reinforce("nuke", "cancel", 0.05);
        let has_edge = g
            .edges
            .get("nuke")
            .map(|es| es.iter().any(|e| e.target == "cancel"))
            .unwrap_or(false);
        assert!(has_edge, "new word should get a learned edge");
    }

    // ── Full pipeline demo ────────────────────────────────────────────────

    #[test]
    fn pipeline_merged_the_pr() {
        let g = saas_test_graph();
        let r = g.preprocess("merged the pr");
        assert_eq!(r.normalized, "merge the pull request");
        assert!(!r.was_modified || r.injected.is_empty()); // merge is canonical, no synonym needed
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Layer 2 — Intent Graph (spreading activation router)
// ════════════════════════════════════════════════════════════════════════════

/// A conjunction rule fires when ALL listed words appear in the normalized query.
/// Adds a bonus activation to the target intent on top of individual word weights.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct ConjunctionRule {
    /// All of these (canonical) words must appear in the normalized query.
    pub words: Vec<String>,
    pub intent: String,
    /// Bonus activation added to the intent score when rule fires.
    pub bonus: f32,
}

/// L2 — word-to-intent spreading activation graph.
///
/// Works with L1 (LexicalGraph): L1 normalizes the query first
/// (morphology, abbreviations, synonyms), then L2 activates intent nodes
/// from the canonical words. Conjunction bonuses are computed in the same pass.
/// L3 inhibition (anti-Hebbian suppression) is applied last.
///
/// Scoring: IDF-weighted activation. `score += weight * ln(N / df)`.
/// Words shared across many intents get low IDF; rare words get high IDF.
/// One new phrase = immediate weight update — no accumulation needed.
/// Full routing result with disposition and ranked candidates.
#[derive(Debug, Clone)]
pub struct RouteResult {
    /// Confirmed intents from token consumption (best guess).
    pub confirmed: Vec<(String, f32)>,
    /// Raw IDF ranked list (top_n, before token consumption).
    pub ranked: Vec<(String, f32)>,
    /// "confident" | "low_confidence" | "escalate" | "no_match"
    pub disposition: String,
    /// True if query contained negation tokens.
    pub has_negation: bool,
}

/// One round of multi-intent resolution captured for debug/inspection.
#[derive(serde::Serialize, Clone, Debug)]
pub struct RoundTrace {
    pub tokens_in: Vec<String>,
    pub scored: Vec<(String, f32)>,
    pub confirmed: Vec<String>,
    pub consumed: Vec<String>,
}

/// Full multi-intent resolution trace returned by `score_multi_normalized_traced`.
#[derive(serde::Serialize, Clone, Debug)]
pub struct MultiIntentTrace {
    pub rounds: Vec<RoundTrace>,
    pub stop_reason: String,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default)]
pub struct IntentIndex {
    /// word → [(intent_id, weight 0.0–1.0)]
    #[serde(default)]
    pub word_intent: HashMap<String, Vec<(String, f32)>>,

    /// Conjunction bonuses — word pairs that together strongly indicate an intent.
    #[serde(default)]
    pub conjunctions: Vec<ConjunctionRule>,

    /// Char-ngram tiebreaker index: intent_id → set of char 4-grams from seed phrases.
    #[serde(default)]
    pub char_ngrams: HashMap<String, std::collections::HashSet<String>>,

    /// Cached count of distinct intents in the index.
    /// Maintained incrementally by index_phrase/learn_word/remove_intent.
    /// Serialized so it survives save/load without a recount.
    #[serde(default)]
    pub intent_count: usize,

    /// Per-word IDF cache: ln(intent_count / posting_list_len).
    /// Recomputed when a word's posting list changes length (add/remove intent).
    /// NOT serialized — rebuilt in one O(words) pass on load via rebuild_idf().
    #[serde(skip)]
    idf_cache: FxHashMap<String, f32>,

    /// Distinct intent IDs seen across all posting lists.
    /// Maintained incrementally in learn_word so intent_count is correct for
    /// namespaces built in memory (not loaded from disk) without needing rebuild_idf().
    /// NOT serialized — rebuilt from posting lists in rebuild_idf().
    #[serde(skip)]
    known_intents: FxHashSet<String>,
}

impl IntentIndex {
    pub fn new() -> Self {
        Self::default()
    }

    const PHRASE_RATE: f32 = 0.4;
    const LEARN_RATE: f32 = 0.3;

    /// Recompute intent_count and idf_cache from word_intent in one pass.
    /// Called after deserialization (load from disk) since idf_cache is not serialized.
    pub fn rebuild_idf(&mut self) {
        self.known_intents.clear();
        for entries in self.word_intent.values() {
            for (id, _) in entries {
                self.known_intents.insert(id.clone());
            }
        }
        let n = self.known_intents.len();
        self.intent_count = n;
        let n_f = n.max(1) as f32;
        self.idf_cache.clear();
        for (word, entries) in &self.word_intent {
            let idf = (n_f / entries.len() as f32).ln().max(0.0);
            self.idf_cache.insert(word.clone(), idf);
        }
    }

    /// Update idf_cache for a single word after its posting list changed length.
    fn refresh_idf_for(&mut self, word: &str) {
        if let Some(entries) = self.word_intent.get(word) {
            let n_f = self.intent_count.max(1) as f32;
            let idf = (n_f / entries.len() as f32).ln().max(0.0);
            self.idf_cache.insert(word.to_string(), idf);
        } else {
            self.idf_cache.remove(word);
        }
    }

    /// Look up cached IDF for a word, or compute on the fly if missing.
    #[inline]
    fn idf(&self, word: &str) -> f32 {
        self.idf_cache.get(word).copied().unwrap_or_else(|| {
            self.word_intent
                .get(word)
                .map(|e| {
                    (self.intent_count.max(1) as f32 / e.len() as f32)
                        .ln()
                        .max(0.0)
                })
                .unwrap_or(0.0)
        })
    }

    /// Learn a single word → intent association with asymptotic Hebbian update.
    pub fn learn_word(&mut self, word: &str, intent: &str, rate: f32) {
        if word.is_empty() {
            return;
        }
        let entries = self.word_intent.entry(word.to_string()).or_default();
        if let Some(e) = entries.iter_mut().find(|(id, _)| id == intent) {
            // Weight update only — posting list length unchanged, IDF unchanged.
            e.1 = (e.1 + rate * (1.0 - e.1)).min(1.0);
        } else {
            // New (word, intent) pair — posting list grows.
            // Update intent_count if this is a brand-new intent in the index.
            let new_intent = self.known_intents.insert(intent.to_string());
            if new_intent {
                self.intent_count += 1;
            }
            entries.push((intent.to_string(), rate));
            if new_intent {
                // A new intent invalidates the IDF of every previously-indexed
                // word (their denominator was N, now it's N+1). Sub-millisecond
                // even for thousands of words, runs only when intents are
                // registered (rare event vs phrase-adds).
                self.rebuild_idf();
            } else {
                // Existing intent — only this word's IDF needs refreshing.
                self.refresh_idf_for(word);
            }
        }
    }

    /// Learn a phrase: tokenize into words, learn each word for the intent.
    /// Uses PHRASE_RATE (0.4) for seed phrases.
    pub fn learn_phrase(&mut self, words: &[&str], intent: &str) {
        for word in words {
            self.learn_word(word, intent, Self::PHRASE_RATE);
        }
    }

    /// Index char 4-grams from a phrase for the tiebreaker layer.
    /// Takes the ORIGINAL phrase (before tokenization) so char-level patterns
    /// include word boundaries implicitly via spaces.
    pub fn index_char_ngrams(&mut self, phrase: &str, intent: &str) {
        let normalized: String = phrase.to_lowercase();
        let s: String = format!(
            "  {}  ",
            normalized.split_whitespace().collect::<Vec<_>>().join(" ")
        );
        if s.chars().count() < 4 {
            return;
        }
        let chars: Vec<char> = s.chars().collect();
        let set = self.char_ngrams.entry(intent.to_string()).or_default();
        for window in chars.windows(4) {
            let ngram: String = window.iter().collect();
            set.insert(ngram);
        }
    }

    /// Char-ngram tiebreaker: when top-1/top-2 scores are close, re-rank
    /// top-K by combined score + alpha * jaccard(query_char_ngrams, intent_char_ngrams).
    ///
    /// - `ratio_threshold`: only fires when top1/(top1+top2) < this (default 0.65)
    /// - `alpha`: weight of jaccard in the combined score (default 0.5)
    ///
    /// Dormant when confident top-1 exists; activates on ambiguous cases.
    pub fn apply_char_ngram_tiebreaker(
        &self,
        query: &str,
        ranked: Vec<(String, f32)>,
        ratio_threshold: f32,
        alpha: f32,
    ) -> Vec<(String, f32)> {
        if ranked.len() < 2 {
            return ranked;
        }
        let top1 = ranked[0].1;
        let top2 = ranked[1].1;
        if top1 + top2 <= 0.0 {
            return ranked;
        }
        let ratio = top1 / (top1 + top2);
        if ratio >= ratio_threshold {
            return ranked;
        }

        // Extract char 4-grams from the query
        let normalized: String = query.to_lowercase();
        let s: String = format!(
            "  {}  ",
            normalized.split_whitespace().collect::<Vec<_>>().join(" ")
        );
        if s.chars().count() < 4 {
            return ranked;
        }
        let chars: Vec<char> = s.chars().collect();
        let mut q_ngrams: FxHashSet<String> = FxHashSet::default();
        for window in chars.windows(4) {
            let ngram: String = window.iter().collect();
            q_ngrams.insert(ngram);
        }
        if q_ngrams.is_empty() {
            return ranked;
        }

        // Re-rank top-5 (cap)
        let k = ranked.len().min(5);
        let (head, tail) = ranked.split_at(k);
        let mut rescored: Vec<(String, f32)> = head
            .iter()
            .map(|(id, score)| {
                let intent_set = self.char_ngrams.get(id);
                let jaccard = match intent_set {
                    Some(iset) if !iset.is_empty() => {
                        let inter = q_ngrams.iter().filter(|n| iset.contains(*n)).count();
                        let uni = q_ngrams.len() + iset.len() - inter;
                        if uni == 0 {
                            0.0
                        } else {
                            inter as f32 / uni as f32
                        }
                    }
                    _ => 0.0,
                };
                (id.clone(), score + alpha * jaccard)
            })
            .collect();
        rescored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rescored.extend_from_slice(tail);
        rescored
    }

    /// Learn specific intent-bearing words from a user query.
    /// Uses LEARN_RATE (0.3) — slightly lower than seed to preserve seed discrimination.
    /// `words` should be LLM-confirmed intent-bearing words only, NOT all query words.
    pub fn learn_query_words(&mut self, words: &[&str], intent: &str) {
        for word in words {
            self.learn_word(word, intent, Self::LEARN_RATE);
        }
    }

    /// Routing threshold — identical in production and simulation.
    pub fn default_threshold(&self) -> f32 {
        0.3
    }

    /// Multi-intent gap — identical in production and simulation.
    pub fn default_gap(&self) -> f32 {
        1.5
    }

    /// Hebbian reinforcement from a routing confirmation.
    /// `words` must already be Layer-1 normalized canonical terms.
    ///
    /// Positive delta (+0.05): asymptotic strengthening — Δw = delta × (1 - w).
    ///   Weight approaches 1.0, never exceeds it. 1000 reinforcements converge
    ///   the same as 10 — diminishing returns prevent runaway weight growth.
    ///
    /// Negative delta (-0.05): asymptotic suppression — w = w × (1 + delta).
    ///   Weight approaches 0, never goes negative. Slow to suppress a strong
    ///   edge — intentional, a word right 100 times shouldn't die from 3 wrong routings.
    ///
    /// New edges are only created for positive delta.
    pub fn reinforce(&mut self, words: &[&str], intent: &str, delta: f32) {
        for word in words {
            let entries = self.word_intent.entry(word.to_string()).or_default();
            if let Some(e) = entries.iter_mut().find(|(id, _)| id == intent) {
                if delta >= 0.0 {
                    e.1 = (e.1 + delta * (1.0 - e.1)).min(1.0);
                } else {
                    e.1 = (e.1 * (1.0 + delta)).max(0.0);
                }
                // Weight-only change — posting list length unchanged, IDF unchanged.
            } else if delta > 0.0 {
                // New (word, intent) pair — posting list grows, refresh IDF for this word.
                entries.push((intent.to_string(), delta.min(1.0)));
                self.refresh_idf_for(word);
            }
        }
    }

    /// Returns indices of conjunction rules that fire for the given canonical word set.
    /// Used by auto-learn to know which conjunction bonuses contributed to a routing.
    pub fn fired_conjunction_indices(&self, words: &[&str]) -> Vec<usize> {
        let word_set: FxHashSet<&str> = words.iter().copied().collect();
        self.conjunctions
            .iter()
            .enumerate()
            .filter(|(_, rule)| rule.words.iter().all(|w| word_set.contains(w.as_str())))
            .map(|(i, _)| i)
            .collect()
    }

    /// Asymptotic Hebbian update on a conjunction rule's bonus.
    /// Positive delta: bonus approaches 1.0 (strengthen useful conjunction).
    /// Negative delta: bonus decays toward 0.0 (weaken misleading conjunction).
    pub fn reinforce_conjunction(&mut self, idx: usize, delta: f32) {
        if let Some(rule) = self.conjunctions.get_mut(idx) {
            if delta >= 0.0 {
                rule.bonus = (rule.bonus + delta * (1.0 - rule.bonus)).min(1.0);
            } else {
                rule.bonus = (rule.bonus * (1.0 + delta)).max(0.0);
            }
        }
    }

    /// IDF-weighted 1-gram scoring.
    ///
    /// For each query token, accumulates weight × IDF across all matching intents.
    /// CJK: character bigrams from the tokenizer are treated as words.
    /// Negation: "not_X" tokens subtract from intent scores and set has_negation flag.
    pub fn score_normalized(&self, normalized: &str) -> (Vec<(String, f32)>, bool) {
        // CJK negation pre-pass
        const CJK_NEG: &[char] = &['不', '没', '别', '未'];
        let cjk_negated = normalized.chars().any(|c| CJK_NEG.contains(&c));
        let query_for_tokenize: std::borrow::Cow<str> = if cjk_negated {
            std::borrow::Cow::Owned(
                normalized
                    .chars()
                    .map(|c| if CJK_NEG.contains(&c) { ' ' } else { c })
                    .collect(),
            )
        } else {
            std::borrow::Cow::Borrowed(normalized)
        };

        let tokens = crate::tokenizer::tokenize(&query_for_tokenize);
        let mut scores: FxHashMap<String, f32> = FxHashMap::default();
        let mut has_negation = cjk_negated;

        let all_bases: FxHashSet<&str> = tokens
            .iter()
            .map(|t| t.strip_prefix("not_").unwrap_or(t.as_str()))
            .collect();

        for token in &tokens {
            let is_negated = token.starts_with("not_");
            let base = if is_negated {
                &token["not_".len()..]
            } else {
                token.as_str()
            };
            if is_negated {
                has_negation = true;
            }
            if let Some(activations) = self.word_intent.get(base) {
                let idf = self.idf(base);
                for (intent, weight) in activations {
                    let delta = weight * idf;
                    *scores.entry(intent.clone()).or_insert(0.0) +=
                        if is_negated { -delta } else { delta };
                }
            }
        }

        // Conjunction bonuses
        for rule in &self.conjunctions {
            if rule.words.iter().all(|w| all_bases.contains(w.as_str())) {
                *scores.entry(rule.intent.clone()).or_insert(0.0) += rule.bonus;
            }
        }

        let mut result: Vec<(String, f32)> = scores.into_iter().filter(|(_, s)| *s > 0.0).collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        (result, has_negation)
    }

    /// Convenience: score with L1 preprocessing included.
    pub fn score(&self, layer1: &LexicalGraph, query: &str) -> (Vec<(String, f32)>, bool) {
        let preprocessed = layer1.preprocess(query);
        self.score_normalized(&preprocessed.expanded)
    }

    /// Multi-intent score with L1 preprocessing.
    pub fn score_multi(
        &self,
        layer1: &LexicalGraph,
        query: &str,
        threshold: f32,
        gap: f32,
    ) -> (Vec<(String, f32)>, bool) {
        let preprocessed = layer1.preprocess(query);
        self.score_multi_normalized(&preprocessed.expanded, threshold, gap)
    }

    /// Multi-intent scoring with token consumption.
    ///
    /// Round 1: Score all tokens → confirm intents above ratio threshold.
    /// Round 2+: Remove tokens consumed by confirmed intents → re-score remaining.
    /// Stops when remaining score < gate_ratio of original top.
    ///
    /// Token consumption: after confirming an intent, remove tokens that
    /// are primarily associated with it. Remaining tokens indicate additional intents.
    pub fn score_multi_normalized(
        &self,
        normalized: &str,
        threshold: f32,
        gap: f32,
    ) -> (Vec<(String, f32)>, bool) {
        let (results, neg, _trace) =
            self.score_multi_normalized_traced(normalized, threshold, gap, false);
        (results, neg)
    }

    /// Same as `score_multi_normalized` but optionally captures per-round trace data
    /// for debugging/inspection. Used by the Layers probe endpoint.
    pub fn score_multi_normalized_traced(
        &self,
        normalized: &str,
        threshold: f32,
        _gap: f32,
        with_trace: bool,
    ) -> (Vec<(String, f32)>, bool, Option<MultiIntentTrace>) {
        const GATE_RATIO: f32 = 0.55;
        const MAX_ROUNDS: usize = 3;

        // CJK negation pre-pass
        const CJK_NEG: &[char] = &['不', '没', '别', '未'];
        let cjk_negated = normalized.chars().any(|c| CJK_NEG.contains(&c));
        let query_for_tokenize: std::borrow::Cow<str> = if cjk_negated {
            std::borrow::Cow::Owned(
                normalized
                    .chars()
                    .map(|c| if CJK_NEG.contains(&c) { ' ' } else { c })
                    .collect(),
            )
        } else {
            std::borrow::Cow::Borrowed(normalized)
        };

        let all_tokens: Vec<String> = crate::tokenizer::tokenize(&query_for_tokenize);
        let has_negation = cjk_negated || all_tokens.iter().any(|t| t.starts_with("not_"));

        let mut remaining: Vec<String> = all_tokens;
        let mut confirmed: Vec<(String, f32)> = Vec::new();
        let mut confirmed_ids: FxHashSet<String> = FxHashSet::default();
        let mut original_top: f32 = 0.0;
        let mut trace_rounds: Vec<RoundTrace> = Vec::new();
        let mut stop_reason: Option<String> = None;

        for round in 0..MAX_ROUNDS {
            let scored = self.score_tokens(&remaining, &confirmed_ids);
            if scored.is_empty() {
                if with_trace {
                    stop_reason = Some("no scores".into());
                }
                break;
            }

            let round_top = scored[0].1;
            if round == 0 {
                original_top = round_top;
            }
            if round_top < threshold {
                if with_trace {
                    stop_reason =
                        Some(format!("top {:.2} < threshold {:.2}", round_top, threshold));
                    trace_rounds.push(RoundTrace {
                        tokens_in: remaining.clone(),
                        scored: scored.iter().take(5).cloned().collect(),
                        confirmed: vec![],
                        consumed: vec![],
                    });
                }
                break;
            }
            if round > 0 && round_top < original_top * GATE_RATIO {
                if with_trace {
                    stop_reason = Some(format!(
                        "top {:.2} < gate {:.2}",
                        round_top,
                        original_top * GATE_RATIO
                    ));
                    trace_rounds.push(RoundTrace {
                        tokens_in: remaining.clone(),
                        scored: scored.iter().take(5).cloned().collect(),
                        confirmed: vec![],
                        consumed: vec![],
                    });
                }
                break;
            }

            // Confirm only intents very close to this round's top (within 10%).
            // Looser multi-intent detection comes from multiple rounds after token consumption.
            let mut round_confirmed: Vec<(String, f32)> = Vec::new();
            for (intent, score) in &scored {
                if *score >= round_top * 0.90 && *score >= threshold {
                    round_confirmed.push((intent.clone(), *score));
                    confirmed_ids.insert(intent.clone());
                }
            }

            if round_confirmed.is_empty() {
                if with_trace {
                    stop_reason = Some("no confirmed".into());
                }
                break;
            }
            confirmed.extend(round_confirmed.iter().cloned());

            let tokens_before: Vec<String> = remaining.clone();
            // Consume tokens: remove tokens primarily associated with confirmed intents
            remaining.retain(|token| {
                let base = token.strip_prefix("not_").unwrap_or(token.as_str());
                if let Some(activations) = self.word_intent.get(base) {
                    // Token's strongest intent
                    let best_intent = activations
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    match best_intent {
                        Some((id, _)) => !confirmed_ids.contains(id.as_str()),
                        None => true,
                    }
                } else {
                    true // unknown token, keep
                }
            });

            if with_trace {
                let remaining_set: FxHashSet<&String> = remaining.iter().collect();
                let consumed: Vec<String> = tokens_before
                    .iter()
                    .filter(|t| !remaining_set.contains(t))
                    .cloned()
                    .collect();
                trace_rounds.push(RoundTrace {
                    tokens_in: tokens_before,
                    scored: scored.iter().take(5).cloned().collect(),
                    confirmed: round_confirmed.iter().map(|(id, _)| id.clone()).collect(),
                    consumed,
                });
            }

            if remaining.is_empty() {
                if with_trace {
                    stop_reason = Some("all tokens consumed".into());
                }
                break;
            }
        }

        let trace = if with_trace {
            Some(MultiIntentTrace {
                rounds: trace_rounds,
                stop_reason: stop_reason.unwrap_or_else(|| "max rounds reached".into()),
            })
        } else {
            None
        };

        (confirmed, has_negation, trace)
    }

    /// Score a specific set of tokens (not a full query string).
    fn score_tokens(
        &self,
        tokens: &[String],
        exclude_intents: &FxHashSet<String>,
    ) -> Vec<(String, f32)> {
        let mut scores: FxHashMap<String, f32> = FxHashMap::default();

        for token in tokens {
            let is_negated = token.starts_with("not_");
            let base = if is_negated {
                &token["not_".len()..]
            } else {
                token.as_str()
            };
            if let Some(activations) = self.word_intent.get(base) {
                let idf = self.idf(base);
                for (intent, weight) in activations {
                    if exclude_intents.contains(intent) {
                        continue;
                    }
                    let delta = weight * idf;
                    *scores.entry(intent.clone()).or_insert(0.0) +=
                        if is_negated { -delta } else { delta };
                }
            }
        }

        // Conjunction bonuses
        let all_bases: FxHashSet<&str> = tokens
            .iter()
            .map(|t| t.strip_prefix("not_").unwrap_or(t.as_str()))
            .collect();
        for rule in &self.conjunctions {
            if !exclude_intents.contains(&rule.intent)
                && rule.words.iter().all(|w| all_bases.contains(w.as_str()))
            {
                *scores.entry(rule.intent.clone()).or_insert(0.0) += rule.bonus;
            }
        }

        let mut result: Vec<(String, f32)> = scores.into_iter().filter(|(_, s)| *s > 0.0).collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(std::io::Error::other)
    }

    // ── Library-level resolve API ───────────────────────────────────────────

    /// Resolve a query through the full pipeline: L1 preprocessing → IDF scoring →
    /// token consumption → cross-provider disambiguation → disposition.
    ///
    /// `top_n`: how many ranked candidates to return (default 5).
    pub fn resolve(
        &self,
        layer1: Option<&LexicalGraph>,
        query: &str,
        threshold: f32,
        top_n: usize,
    ) -> RouteResult {
        // L1 preprocessing
        let processed = match layer1 {
            Some(l1) => l1.preprocess(query).expanded,
            None => query.to_string(),
        };

        // Raw IDF scores (for ranked list)
        let (raw, has_negation) = self.score_normalized(&processed);
        let ranked: Vec<(String, f32)> = raw.into_iter().take(top_n).collect();

        // Token consumption (for confirmed list)
        let (mut confirmed, _) = self.score_multi_normalized(&processed, threshold, 0.0);

        if confirmed.is_empty() {
            return RouteResult {
                confirmed: vec![],
                ranked,
                disposition: "no_match".to_string(),
                has_negation,
            };
        }

        // Cross-provider disambiguation: when same action appears from multiple
        // providers, use query word exclusivity to pick the right one.
        if confirmed.len() > 1 {
            self.disambiguate_providers(&mut confirmed, &processed);
        }

        // Disposition
        let top = confirmed[0].1;
        let disposition = if confirmed.len() >= 3 && confirmed[2].1 / top >= 0.75 {
            "escalate"
        } else if top < threshold * 2.0 {
            "low_confidence"
        } else {
            "confident"
        };

        RouteResult {
            confirmed,
            ranked,
            disposition: disposition.to_string(),
            has_negation,
        }
    }

    /// Cross-provider disambiguation: when the same action suffix appears from
    /// multiple providers, pick the one with the most unique query word matches.
    fn disambiguate_providers(&self, confirmed: &mut Vec<(String, f32)>, query: &str) {
        if confirmed.len() < 2 {
            return;
        }

        // Group by action name (part after ':')
        let mut action_groups: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        for (i, (id, _)) in confirmed.iter().enumerate() {
            let action = id.split(':').nth(1).unwrap_or(id.as_str());
            action_groups.entry(action.to_string()).or_default().push(i);
        }

        let duplicate_groups: Vec<Vec<usize>> = action_groups
            .values()
            .filter(|g| g.len() > 1)
            .cloned()
            .collect();

        if duplicate_groups.is_empty() {
            return;
        }

        let tokens = crate::tokenizer::tokenize(query);
        let confirmed_ids: FxHashSet<&str> = confirmed.iter().map(|(id, _)| id.as_str()).collect();

        // Count unique words per intent
        let mut unique_count: FxHashMap<&str, usize> = FxHashMap::default();
        for token in &tokens {
            let base = token.strip_prefix("not_").unwrap_or(token.as_str());
            if let Some(activations) = self.word_intent.get(base) {
                let matching: Vec<&str> = activations
                    .iter()
                    .filter(|(id, _)| confirmed_ids.contains(id.as_str()))
                    .map(|(id, _)| id.as_str())
                    .collect();
                if matching.len() == 1 {
                    *unique_count.entry(matching[0]).or_insert(0) += 1;
                }
            }
        }

        let mut to_remove: FxHashSet<usize> = FxHashSet::default();
        for group in &duplicate_groups {
            let best = group.iter().max_by_key(|&&i| {
                unique_count
                    .get(confirmed[i].0.as_str())
                    .copied()
                    .unwrap_or(0)
            });
            if let Some(&best_idx) = best {
                if unique_count
                    .get(confirmed[best_idx].0.as_str())
                    .copied()
                    .unwrap_or(0)
                    > 0
                {
                    for &i in group {
                        if i != best_idx {
                            to_remove.insert(i);
                        }
                    }
                }
            }
        }

        if !to_remove.is_empty() {
            let mut i = 0;
            confirmed.retain(|_| {
                let keep = !to_remove.contains(&i);
                i += 1;
                keep
            });
        }
    }

    /// Stats: (unique_words, activation_edges, conjunctions).
    pub fn stats(&self) -> (usize, usize, usize) {
        let activation_edges: usize = self.word_intent.values().map(|v| v.len()).sum();
        (
            self.word_intent.len(),
            activation_edges,
            self.conjunctions.len(),
        )
    }
}

/// Global English morphological base — shared across all namespaces.
/// Contains only morphological (inflection) edges. No synonyms, no abbreviations.
/// Synonyms and abbreviations are domain-specific and belong per-namespace.
pub fn english_morphology_base() -> LexicalGraph {
    let mut g = LexicalGraph::new();
    let morph = EdgeKind::Morphological;

    // cancel family
    for v in &[
        "canceling",
        "cancelling",
        "cancelled",
        "canceled",
        "cancellation",
        "cancels",
    ] {
        g.add(v, "cancel", 0.99, morph.clone());
    }
    // refund family
    for v in &[
        "refunding",
        "refunded",
        "refunds",
        "reimbursing",
        "reimbursed",
    ] {
        g.add(v, "refund", 0.99, morph.clone());
    }
    // charge family
    for v in &["charging", "charged", "charges"] {
        g.add(v, "charge", 0.99, morph.clone());
    }
    // update family
    for v in &["updating", "updated", "updates"] {
        g.add(v, "update", 0.99, morph.clone());
    }
    // create family
    for v in &["creating", "created", "creates", "creation"] {
        g.add(v, "create", 0.99, morph.clone());
    }
    // delete family
    for v in &["deleting", "deleted", "deletes", "deletion"] {
        g.add(v, "delete", 0.99, morph.clone());
    }
    // send family
    for v in &["sending", "sent", "sends"] {
        g.add(v, "send", 0.99, morph.clone());
    }
    // receive family
    for v in &["receiving", "received", "receives"] {
        g.add(v, "receive", 0.99, morph.clone());
    }
    // reset family
    for v in &["resetting", "resetted", "resets"] {
        g.add(v, "reset", 0.99, morph.clone());
    }
    // change family
    for v in &["changing", "changed", "changes"] {
        g.add(v, "change", 0.99, morph.clone());
    }
    // upgrade family
    for v in &["upgrading", "upgraded", "upgrades"] {
        g.add(v, "upgrade", 0.99, morph.clone());
    }
    // downgrade family
    for v in &["downgrading", "downgraded", "downgrades"] {
        g.add(v, "downgrade", 0.99, morph.clone());
    }
    // connect family
    for v in &["connecting", "connected", "connects", "connection"] {
        g.add(v, "connect", 0.99, morph.clone());
    }
    // disconnect family
    for v in &["disconnecting", "disconnected", "disconnects"] {
        g.add(v, "disconnect", 0.99, morph.clone());
    }
    // install family
    for v in &["installing", "installed", "installs", "installation"] {
        g.add(v, "install", 0.99, morph.clone());
    }
    // remove family
    for v in &["removing", "removed", "removes", "removal"] {
        g.add(v, "remove", 0.99, morph.clone());
    }
    // enable family
    for v in &["enabling", "enabled", "enables"] {
        g.add(v, "enable", 0.99, morph.clone());
    }
    // disable family
    for v in &["disabling", "disabled", "disables"] {
        g.add(v, "disable", 0.99, morph.clone());
    }
    // block family
    for v in &["blocking", "blocked", "blocks"] {
        g.add(v, "block", 0.99, morph.clone());
    }
    // report family
    for v in &["reporting", "reported", "reports"] {
        g.add(v, "report", 0.99, morph.clone());
    }
    // transfer family
    for v in &["transferring", "transferred", "transfers"] {
        g.add(v, "transfer", 0.99, morph.clone());
    }
    // schedule family
    for v in &["scheduling", "scheduled", "schedules"] {
        g.add(v, "schedule", 0.99, morph.clone());
    }
    // merge family
    for v in &["merging", "merged", "merges"] {
        g.add(v, "merge", 0.99, morph.clone());
    }
    // ship family
    for v in &["shipping", "shipped", "shipment", "shipments"] {
        g.add(v, "ship", 0.99, morph.clone());
    }
    // pay family
    for v in &["paying", "paid", "pays", "payment", "payments"] {
        g.add(v, "pay", 0.99, morph.clone());
    }
    // subscribe family
    for v in &[
        "subscribing",
        "subscribed",
        "subscribes",
        "subscription",
        "subscriptions",
    ] {
        g.add(v, "subscribe", 0.99, morph.clone());
    }
    // list family
    for v in &["listing", "listed", "lists"] {
        g.add(v, "list", 0.99, morph.clone());
    }
    // invite family
    for v in &["inviting", "invited", "invites", "invitation"] {
        g.add(v, "invite", 0.99, morph.clone());
    }
    // verify family
    for v in &["verifying", "verified", "verifies", "verification"] {
        g.add(v, "verify", 0.99, morph.clone());
    }
    // access family
    for v in &["accessing", "accessed", "accesses"] {
        g.add(v, "access", 0.99, morph.clone());
    }
    // close family
    for v in &["closing", "closed", "closes", "closure"] {
        g.add(v, "close", 0.99, morph.clone());
    }
    // open family
    for v in &["opening", "opened", "opens"] {
        g.add(v, "open", 0.99, morph.clone());
    }
    // configure family
    for v in &["configuring", "configured", "configures", "configuration"] {
        g.add(v, "configure", 0.99, morph.clone());
    }
    // deploy family
    for v in &["deploying", "deployed", "deploys", "deployment"] {
        g.add(v, "deploy", 0.99, morph.clone());
    }
    // detect family
    for v in &["detecting", "detected", "detects", "detection"] {
        g.add(v, "detect", 0.99, morph.clone());
    }
    // fail family
    for v in &["failing", "failed", "fails", "failure"] {
        g.add(v, "fail", 0.99, morph.clone());
    }
    // expire family
    for v in &["expiring", "expired", "expires", "expiration"] {
        g.add(v, "expire", 0.99, morph.clone());
    }
    // renew family
    for v in &["renewing", "renewed", "renews", "renewal"] {
        g.add(v, "renew", 0.99, morph.clone());
    }
    // approve family
    for v in &["approving", "approved", "approves", "approval"] {
        g.add(v, "approve", 0.99, morph.clone());
    }
    // reject family
    for v in &["rejecting", "rejected", "rejects", "rejection"] {
        g.add(v, "reject", 0.99, morph.clone());
    }

    g
}

#[cfg(test)]
mod intent_graph_tests {
    use super::*;

    fn mini_intent_graph() -> (LexicalGraph, IntentIndex) {
        let layer1 = saas_test_graph();
        let mut ig = IntentIndex::new();

        // cancel_subscription: "cancel", "subscription"
        ig.learn_phrase(&["cancel", "subscription"], "cancel_subscription");
        ig.conjunctions.push(ConjunctionRule {
            words: vec!["cancel".into(), "subscription".into()],
            intent: "cancel_subscription".into(),
            bonus: 0.50,
        });

        // cancel_order: "cancel", "order" — IDF disambiguates via unique terms
        ig.learn_phrase(&["cancel", "order"], "cancel_order");

        // send_message: "send", "message"
        ig.learn_phrase(&["send", "message"], "send_message");

        (layer1, ig)
    }

    #[test]
    fn layer3_basic_activation() {
        let (l1, ig) = mini_intent_graph();
        let (scores, neg) = ig.score(&l1, "cancel my subscription");
        let top = &scores[0];
        assert_eq!(top.0, "cancel_subscription");
        assert!(top.1 > 0.0, "cancel_subscription should score positively");
        assert!(!neg, "no negation in this query");
    }

    #[test]
    fn layer3_oov_via_layer1() {
        let (l1, ig) = mini_intent_graph();
        let (scores, _) = ig.score(&l1, "terminate my sub");
        assert_eq!(scores[0].0, "cancel_subscription");
    }

    #[test]
    fn layer3_idf_disambiguates() {
        let (l1, ig) = mini_intent_graph();
        let (scores, _) = ig.score(&l1, "cancel order");
        assert_eq!(
            scores[0].0, "cancel_order",
            "IDF should push cancel_order above cancel_subscription (unique word 'order')"
        );
    }

    #[test]
    fn layer3_reinforcement() {
        let (l1, mut ig) = mini_intent_graph();
        let (before, _) = ig.score(&l1, "kill the subscription");
        let kill_sub_before = before
            .iter()
            .find(|(id, _)| id == "cancel_subscription")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);

        ig.reinforce(&["kill"], "cancel_subscription", 0.80);

        let (after, _) = ig.score(&l1, "kill the subscription");
        let kill_sub_after = after
            .iter()
            .find(|(id, _)| id == "cancel_subscription")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);

        assert!(
            kill_sub_after > kill_sub_before,
            "reinforcement should improve score"
        );
    }

    #[test]
    fn layer3_multi_intent() {
        let (l1, ig) = mini_intent_graph();
        let (results, _) = ig.score_multi(&l1, "cancel subscription and send message", 0.4, 2.0);
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            ids.contains(&"cancel_subscription"),
            "should detect cancel_subscription"
        );
        assert!(ids.contains(&"send_message"), "should detect send_message");
    }

    #[test]
    fn layer3_negation_flags_not_suppresses() {
        let (l1, ig) = mini_intent_graph();
        // Negation subtracts activations: "don't cancel my subscription" should NOT route.
        let (with_neg, neg_flag) = ig.score(&l1, "don't cancel my subscription");
        let (without_neg, _) = ig.score(&l1, "cancel my subscription");

        let neg_score = with_neg
            .iter()
            .find(|(id, _)| id == "cancel_subscription")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let pos_score = without_neg
            .iter()
            .find(|(id, _)| id == "cancel_subscription")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);

        assert!(
            neg_score <= 0.0,
            "cancel_subscription should be suppressed by negation (score={neg_score})"
        );
        assert!(
            pos_score > 0.0,
            "cancel_subscription should route without negation"
        );
        assert!(neg_flag, "has_negation flag should be true");
    }

    #[test]
    fn layer3_cjk_negation() {
        let (_, mut ig) = mini_intent_graph();
        ig.learn_phrase(&["取消", "订阅"], "cancel_subscription");

        // Positive: "取消订阅" → should route to cancel_subscription
        let (pos_scores, pos_neg) = ig.score_normalized("取消订阅");
        assert!(!pos_scores.is_empty(), "positive CJK should score");
        assert_eq!(pos_scores[0].0, "cancel_subscription");
        assert!(!pos_neg, "no negation in positive query");

        // Negative: "不取消订阅" → tokenizer produces not_取消订阅, not_取消, not_订阅
        // L3 still routes to cancel_subscription (intent IS cancellation) but flags negation
        let (neg_scores, neg_flag) = ig.score_normalized("不取消订阅");
        assert!(neg_flag, "CJK negation marker 不 should set has_negation");
        let found = neg_scores.iter().any(|(id, _)| id == "cancel_subscription");
        assert!(
            found,
            "cancel_subscription should still appear (intent is about cancellation)"
        );
    }

    /// Regression test for the in-memory IDF-staleness bug.
    ///
    /// When intents are added incrementally (as happens in any newly-created
    /// namespace built via API calls before any disk save/reload), the IDF
    /// cache for previously-indexed words must remain correct. Earlier the
    /// fix only refreshed the current word's IDF on each new intent, leaving
    /// the IDF of every prior word stuck at its initial value (denominator
    /// frozen at the intent_count when that word was first seen).
    ///
    /// Symptom: a fresh namespace would route nothing until the first server
    /// restart, because IDFs were ln(1/1)=0 across the board.
    #[test]
    fn idf_stays_correct_when_intents_are_added_incrementally() {
        let mut ig = IntentIndex::new();

        // Add intent A with one word. After this:
        //   intent_count = 1, IDF("foo") = ln(1/1) = 0
        ig.learn_phrase(&["foo"], "intent_a");

        // Add intent B with the same word and a unique word. After this:
        //   intent_count = 2
        //   IDF("foo") should be ln(2/2) = 0      (in both intents)
        //   IDF("bar") should be ln(2/1) = 0.69   (in only one intent)
        // BEFORE THE FIX, IDF("foo") stayed at 0 from the original computation
        // and IDF("bar") was correct. Now both should be correct.
        ig.learn_phrase(&["foo", "bar"], "intent_b");

        // Add a third intent so we have a non-trivial idf landscape.
        ig.learn_phrase(&["baz"], "intent_c");
        // intent_count = 3
        //   IDF("foo") = ln(3/2) ≈ 0.405
        //   IDF("bar") = ln(3/1) ≈ 1.099
        //   IDF("baz") = ln(3/1) ≈ 1.099

        let lex = LexicalGraph::default();

        // Querying any of these words should produce non-zero scores.
        let (scores_foo, _) = ig.score(&lex, "foo");
        assert!(
            !scores_foo.is_empty(),
            "fresh-namespace IDF bug: 'foo' should score against intent_a and intent_b"
        );
        assert!(
            scores_foo.iter().all(|(_, s)| *s > 0.0),
            "scores should be positive after the IDF rebuild on new intent"
        );

        let (scores_bar, _) = ig.score(&lex, "bar");
        assert!(
            scores_bar.iter().any(|(id, _)| id == "intent_b"),
            "'bar' should score against intent_b"
        );

        let (scores_baz, _) = ig.score(&lex, "baz");
        assert!(
            scores_baz.iter().any(|(id, _)| id == "intent_c"),
            "'baz' should score against intent_c"
        );
    }
}
