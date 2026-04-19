//! # Hebbian Association Graph
//!
//! A weighted term association graph distilled from LLM knowledge.
//! Replaces the need for pre-trained embeddings by encoding semantic
//! relationships directly as weighted edges — initialized from LLM,
//! optionally updated from routing confirmations.
//!
//! ## Relationship types (by weight tier)
//!
//! | Weight    | Kind          | Query action                        |
//! |-----------|---------------|-------------------------------------|
//! | 0.97–1.0  | Morphological | Normalize (substitute in place)     |
//! | 0.97–1.0  | Abbreviation  | Normalize (substitute in place)     |
//! | 0.80–0.96 | Synonym       | Expand (append canonical term)      |
//! | 0.60–0.79 | Semantic      | Confidence boost only — no expand   |
//!
//! ## Example
//! ```no_run
//! use microresolve::hebbian::{LexicalGraph, EdgeKind};
//! let mut g = LexicalGraph::new();
//! g.add("canceling", "cancel", 0.99, EdgeKind::Morphological);
//! g.add("terminate", "cancel", 0.92, EdgeKind::Synonym);
//! g.add("sub",       "subscription", 0.99, EdgeKind::Abbreviation);
//!
//! let r = g.preprocess("canceling my sub");
//! assert_eq!(r.normalized, "cancel my subscription");
//!
//! let r = g.preprocess("terminate my plan");
//! assert!(r.expanded.contains("cancel"));
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

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

/// Weighted term association graph, per namespace.
/// Serializes to JSON alongside the concept registry.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LexicalGraph {
    /// source_term (lowercase) → outgoing edges
    pub edges: HashMap<String, Vec<HebbianEdge>>,
    /// Minimum synonym weight to trigger query expansion (default 0.80).
    #[serde(default = "default_threshold")]
    pub synonym_threshold: f32,
}

fn default_threshold() -> f32 { 0.80 }

// ── Result type ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PreprocessResult {
    pub original:     String,
    /// After morphology + abbreviation substitution.
    pub normalized:   String,
    /// After synonym injection (final query for term-index).
    pub expanded:     String,
    /// Terms injected during expansion (for logging / debug endpoint).
    pub injected:     Vec<String>,
    /// Semantic-weight edges that fired (for concept confidence boost).
    pub semantic_hits: Vec<(String, String, f32)>,  // (source, target, weight)
    pub was_modified: bool,
}

// ── Core impl ─────────────────────────────────────────────────────────────────

impl LexicalGraph {
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            synonym_threshold: 0.80,
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
        let to   = to.to_lowercase();
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
        let words = Self::l1_tokens(query);
        let mut out: Vec<String> = Vec::with_capacity(words.len());

        for word in &words {
            let replacement = self.edges.get(word.as_str()).and_then(|edges| {
                edges.iter().find(|e| {
                    matches!(e.kind, EdgeKind::Morphological | EdgeKind::Abbreviation)
                        && e.weight >= 0.97
                })
            });
            match replacement {
                Some(e) => out.push(e.target.clone()),
                None    => out.push(word.clone()),
            }
        }
        out.join(" ")
    }

    /// Phase 2: expand query with synonym injections.
    /// For each word that has Synonym edges above `synonym_threshold`,
    /// append the canonical target — only if not already present.
    pub fn expand_query(&self, query: &str) -> (String, Vec<String>) {
        let lower = query.to_lowercase();
        let words = Self::l1_tokens(query);
        let mut injected: Vec<String> = Vec::new();

        for word in &words {
            if let Some(edges) = self.edges.get(word.as_str()) {
                for edge in edges {
                    if matches!(edge.kind, EdgeKind::Synonym)
                        && edge.weight >= self.synonym_threshold
                        && !lower.contains(edge.target.as_str())
                        && !injected.contains(&edge.target)
                    {
                        injected.push(edge.target.clone());
                    }
                }
            }
        }

        if injected.is_empty() {
            (lower, vec![])
        } else {
            (format!("{} {}", lower, injected.join(" ")), injected)
        }
    }

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

    /// Full pipeline: normalize → expand → collect semantic signals.
    pub fn preprocess(&self, query: &str) -> PreprocessResult {
        let normalized    = self.normalize_query(query);
        let (expanded, injected) = self.expand_query(&normalized);
        let semantic_hits = self.semantic_hits(&normalized);
        let was_modified  = expanded != query.to_lowercase();

        PreprocessResult {
            original:  query.to_string(),
            normalized,
            expanded,
            injected,
            semantic_hits,
            was_modified,
        }
    }

    /// Morphology-only pipeline: normalize only, no synonym expansion.
    /// Safe for all intent set sizes — no false positive risk.
    pub fn preprocess_morphonly(&self, query: &str) -> PreprocessResult {
        let normalized = self.normalize_query(query);
        let was_modified = normalized != query.to_lowercase();
        PreprocessResult {
            original: query.to_string(),
            normalized: normalized.clone(),
            expanded: normalized,
            injected: vec![],
            semantic_hits: vec![],
            was_modified,
        }
    }

    /// Vocabulary-grounded pipeline: normalize always, expand synonyms ONLY for
    /// tokens not already present in the L2 index (known_words).
    ///
    /// If a query token is already known to L2, it doesn't need synonym bridging —
    /// L2 already has direct entries for it. Only unknown tokens benefit from expansion.
    /// This eliminates false positives from over-expansion of common words.
    pub fn preprocess_grounded<'a>(
        &self,
        query: &str,
        known_words: &std::collections::HashSet<&'a str>,
    ) -> PreprocessResult {
        let normalized = self.normalize_query(query);
        let lower = normalized.to_lowercase();
        let words = Self::l1_tokens(&normalized);
        let mut injected: Vec<String> = Vec::new();

        for word in &words {
            // Skip synonym expansion for words already in L2 — they don't need bridging.
            if known_words.contains(word.as_str()) {
                continue;
            }
            if let Some(edges) = self.edges.get(word.as_str()) {
                for edge in edges {
                    if matches!(edge.kind, EdgeKind::Synonym)
                        && edge.weight >= self.synonym_threshold
                        && !lower.contains(edge.target.as_str())
                        && !injected.contains(&edge.target)
                    {
                        injected.push(edge.target.clone());
                    }
                }
            }
        }

        let expanded = if injected.is_empty() {
            lower.clone()
        } else {
            format!("{} {}", lower, injected.join(" "))
        };
        let was_modified = expanded != query.to_lowercase();

        PreprocessResult {
            original: query.to_string(),
            normalized: lower,
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
    for v in &["canceling","cancelled","cancellation","cancels"] {
        g.add(v, "cancel", 0.99, EdgeKind::Morphological);
    }
    // refund family
    for v in &["refunding","refunded","refunds"] {
        g.add(v, "refund", 0.99, EdgeKind::Morphological);
    }
    // charge family
    for v in &["charging","charged","charges"] {
        g.add(v, "charge", 0.99, EdgeKind::Morphological);
    }
    // ship family
    for v in &["shipped","shipment","shipments"] {
        g.add(v, "ship", 0.99, EdgeKind::Morphological);
    }
    // merge family
    for v in &["merging","merged","merges"] {
        g.add(v, "merge", 0.99, EdgeKind::Morphological);
    }
    // list family
    for v in &["listing","listed","lists"] {
        g.add(v, "list", 0.99, EdgeKind::Morphological);
    }
    // create family
    for v in &["creating","created","creates","creation"] {
        g.add(v, "create", 0.99, EdgeKind::Morphological);
    }
    // schedule family
    for v in &["scheduling","scheduled","schedules"] {
        g.add(v, "schedule", 0.99, EdgeKind::Morphological);
    }
    // invite family
    for v in &["inviting","invited","invites"] {
        g.add(v, "invite", 0.99, EdgeKind::Morphological);
    }
    // send family
    for v in &["sending","sent","sends"] {
        g.add(v, "send", 0.99, EdgeKind::Morphological);
    }
    // close family
    for v in &["closing","closed","closes"] {
        g.add(v, "close", 0.99, EdgeKind::Morphological);
    }

    // ── Abbreviations (0.99) ─────────────────────────────────────────────
    g.add("pr",    "pull request",   0.99, EdgeKind::Abbreviation);
    g.add("prs",   "pull requests",  0.99, EdgeKind::Abbreviation);
    g.add("repo",  "repository",     0.99, EdgeKind::Abbreviation);
    g.add("repos", "repositories",   0.99, EdgeKind::Abbreviation);
    g.add("sub",   "subscription",   0.99, EdgeKind::Abbreviation);
    g.add("subs",  "subscriptions",  0.99, EdgeKind::Abbreviation);
    g.add("msg",   "message",        0.99, EdgeKind::Abbreviation);
    g.add("msgs",  "messages",       0.99, EdgeKind::Abbreviation);
    g.add("chan",  "channel",        0.99, EdgeKind::Abbreviation);

    // ── Synonyms + their morph variants ──────────────────────────────────
    // cancel synonyms
    for (v, w) in &[("terminate",0.92f32),("terminating",0.92),("terminated",0.92)] {
        g.add(v, "cancel", *w, EdgeKind::Synonym);
    }
    for (v, w) in &[("kill",0.85f32),("killing",0.85),("killed",0.85)] {
        g.add(v, "cancel", *w, EdgeKind::Synonym);
    }
    for (v, w) in &[("axe",0.83f32),("axed",0.83),("axing",0.83)] {
        g.add(v, "cancel", *w, EdgeKind::Synonym);
    }
    g.add("ditch", "cancel", 0.80, EdgeKind::Synonym);

    // send synonyms
    g.add("ping",   "send", 0.92, EdgeKind::Synonym);
    g.add("dm",     "send", 0.90, EdgeKind::Synonym);
    g.add("notify", "send", 0.85, EdgeKind::Synonym);
    g.add("blast",  "send", 0.80, EdgeKind::Synonym);

    // create synonyms
    g.add("spin",  "create", 0.82, EdgeKind::Synonym);
    g.add("make",  "create", 0.85, EdgeKind::Synonym);
    g.add("build", "create", 0.82, EdgeKind::Synonym);
    // NOTE: "open" excluded — ambiguous ("open an issue" vs "open the settings")

    // refund synonyms
    g.add("reimburse",    "refund", 0.90, EdgeKind::Synonym);
    g.add("reimbursement","refund", 0.90, EdgeKind::Synonym);
    g.add("compensate",   "refund", 0.80, EdgeKind::Synonym);

    // charge synonyms
    g.add("run",  "charge", 0.82, EdgeKind::Synonym);  // "run the card"
    g.add("bill", "charge", 0.85, EdgeKind::Synonym);

    // list synonyms
    g.add("show",  "list", 0.85, EdgeKind::Synonym);
    g.add("fetch", "list", 0.82, EdgeKind::Synonym);
    // NOTE: "get", "pull", "open" excluded — too ambiguous as standalone words
    // ("pull request" contains "pull", "open an account" vs "open a file")

    // merge synonyms
    g.add("integrate", "merge", 0.82, EdgeKind::Synonym);
    g.add("squash",    "merge", 0.80, EdgeKind::Synonym);

    // ── Semantic (0.60–0.79) — confidence boost only ──────────────────────
    g.add("stop",  "cancel", 0.65, EdgeKind::Semantic);
    g.add("end",   "cancel", 0.62, EdgeKind::Semantic);
    g.add("drop",  "cancel", 0.68, EdgeKind::Semantic);
    g.add("fire",  "send",   0.70, EdgeKind::Semantic);  // "fire off a message"
    g.add("throw", "create", 0.65, EdgeKind::Semantic);  // "throw up a repo"

    g
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Morphological normalization ───────────────────────────────────────

    #[test]
    fn morph_canceling() {
        let g = saas_test_graph();
        assert_eq!(g.normalize_query("canceling my subscription"), "cancel my subscription");
    }

    #[test]
    fn morph_cancelled() {
        let g = saas_test_graph();
        assert_eq!(g.normalize_query("the order was cancelled"), "the order was cancel");
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
        assert_eq!(g.normalize_query("get all shipped orders"), "get all ship orders");
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
        assert_eq!(g.normalize_query("merge the pr in that repo"), "merge the pull request in that repository");
    }

    #[test]
    fn abbrev_msg_chan() {
        let g = saas_test_graph();
        assert_eq!(g.normalize_query("send a msg to the chan"), "send a message to the channel");
    }

    // ── Synonym expansion ─────────────────────────────────────────────────

    #[test]
    fn synonym_terminate_expands_to_cancel() {
        let g = saas_test_graph();
        let r = g.preprocess("terminate my plan");
        assert!(r.expanded.contains("cancel"), "expected 'cancel' injected, got: {}", r.expanded);
        assert!(r.injected.contains(&"cancel".to_string()));
    }

    #[test]
    fn synonym_kill_expands_to_cancel() {
        let g = saas_test_graph();
        let r = g.preprocess("kill the subscription");
        assert!(r.expanded.contains("cancel"));
    }

    #[test]
    fn synonym_ping_expands_to_send() {
        let g = saas_test_graph();
        let r = g.preprocess("ping the team");
        assert!(r.expanded.contains("send"), "expected 'send' injected, got: {}", r.expanded);
    }

    #[test]
    fn synonym_run_expands_to_charge() {
        let g = saas_test_graph();
        let r = g.preprocess("run their card");
        assert!(r.expanded.contains("charge"));
    }

    #[test]
    fn synonym_show_expands_to_list() {
        let g = saas_test_graph();
        let r = g.preprocess("show me all invoices");
        assert!(r.expanded.contains("list"));
    }

    // ── Morphology + abbreviation combined ────────────────────────────────

    #[test]
    fn combined_morph_and_abbrev() {
        let g = saas_test_graph();
        // "canceling" → "cancel", "sub" → "subscription"
        assert_eq!(g.normalize_query("canceling my sub"), "cancel my subscription");
    }

    #[test]
    fn combined_morph_then_synonym() {
        let g = saas_test_graph();
        // "canceling" → normalize to "cancel" → no synonym needed (already canonical)
        let r = g.preprocess("canceling my sub");
        assert_eq!(r.normalized, "cancel my subscription");
        // "cancel" is already canonical so nothing extra injected
        assert!(r.injected.is_empty(), "should not inject anything when already canonical");
    }

    // ── Semantic hits (no expansion) ──────────────────────────────────────

    #[test]
    fn semantic_stop_does_not_expand() {
        let g = saas_test_graph();
        let r = g.preprocess("stop sending me emails");
        // "stop" is Semantic weight 0.65, below synonym_threshold 0.80
        assert!(!r.expanded.contains("cancel"), "semantic word should not expand query");
        // But it should appear in semantic_hits
        let hit = r.semantic_hits.iter().any(|(src, tgt, _)| src == "stop" && tgt == "cancel");
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
        let has_edge = g.edges.get("nuke")
            .map(|es| es.iter().any(|e| e.target == "cancel"))
            .unwrap_or(false);
        assert!(has_edge, "new word should get a learned edge");
    }

    // ── Full pipeline demo ────────────────────────────────────────────────

    #[test]
    fn pipeline_terminate_my_sub() {
        let g = saas_test_graph();
        let r = g.preprocess("terminate my sub");
        // morph: nothing (terminate is base)
        // abbrev: sub → subscription
        assert!(r.normalized.contains("subscription"));
        // synonym: terminate → cancel injected
        assert!(r.expanded.contains("cancel"));
        assert!(r.was_modified);
    }

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

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default)]
pub struct IntentIndex {
    /// word → [(intent_id, weight 0.0–1.0)]
    ///
    /// Pure 1-gram IDF scoring. Each word maps to intents with Hebbian weights.
    /// IDF at query time: score += weight * ln(N/df).
    /// CJK: character bigrams from the tokenizer are treated as "words."
    ///
    /// Learning: asymptotic w' = w + rate*(1-w). Never exceeds 1.0.
    /// IDF stays healthy when learning QUERY WORDS directly (each word
    /// maps to 1-2 intents). Degrades when learning LLM paraphrases
    /// (generic words spread across many intents).
    #[serde(default)]
    pub word_intent: HashMap<String, Vec<(String, f32)>>,

    /// Conjunction bonuses — word pairs that together strongly indicate an intent.
    #[serde(default)]
    pub conjunctions: Vec<ConjunctionRule>,
    /// L3 Inhibition: correct_intent → false_positive_intent → suppression strength.
    /// Anti-Hebbian: when A fires and B is wrong, A learns to suppress B.
    #[serde(default)]
    pub inhibit: HashMap<String, HashMap<String, f32>>,

    /// Char-ngram tiebreaker index: intent_id → set of char 4-grams from seed phrases.
    /// Used when top-1/top-2 scores are close (ambiguous). Derived from phrases
    /// at seed/learn time — no separate training needed.
    #[serde(default)]
    pub char_ngrams: HashMap<String, std::collections::HashSet<String>>,
}

/// Backward compat alias: code still using IntentGraph compiles without change.
pub type IntentGraph = IntentIndex;

impl IntentIndex {
    pub fn new() -> Self {
        Self::default()
    }

    const PHRASE_RATE: f32 = 0.4;
    const LEARN_RATE: f32 = 0.3;

    /// Learn a single word → intent association with asymptotic Hebbian update.
    pub fn learn_word(&mut self, word: &str, intent: &str, rate: f32) {
        if word.is_empty() { return; }
        let entries = self.word_intent.entry(word.to_string()).or_default();
        if let Some(e) = entries.iter_mut().find(|(id, _)| id == intent) {
            e.1 = (e.1 + rate * (1.0 - e.1)).min(1.0);
        } else {
            entries.push((intent.to_string(), rate));
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
        let s: String = format!("  {}  ", normalized.split_whitespace().collect::<Vec<_>>().join(" "));
        if s.chars().count() < 4 { return; }
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
        if ranked.len() < 2 { return ranked; }
        let top1 = ranked[0].1;
        let top2 = ranked[1].1;
        if top1 + top2 <= 0.0 { return ranked; }
        let ratio = top1 / (top1 + top2);
        if ratio >= ratio_threshold { return ranked; }

        // Extract char 4-grams from the query
        let normalized: String = query.to_lowercase();
        let s: String = format!("  {}  ", normalized.split_whitespace().collect::<Vec<_>>().join(" "));
        if s.chars().count() < 4 { return ranked; }
        let chars: Vec<char> = s.chars().collect();
        let mut q_ngrams: std::collections::HashSet<String> = std::collections::HashSet::new();
        for window in chars.windows(4) {
            let ngram: String = window.iter().collect();
            q_ngrams.insert(ngram);
        }
        if q_ngrams.is_empty() { return ranked; }

        // Re-rank top-5 (cap)
        let k = ranked.len().min(5);
        let (head, tail) = ranked.split_at(k);
        let mut rescored: Vec<(String, f32)> = head.iter().map(|(id, score)| {
            let intent_set = self.char_ngrams.get(id);
            let jaccard = match intent_set {
                Some(iset) if !iset.is_empty() => {
                    let inter = q_ngrams.iter().filter(|n| iset.contains(*n)).count();
                    let uni = q_ngrams.len() + iset.len() - inter;
                    if uni == 0 { 0.0 } else { inter as f32 / uni as f32 }
                }
                _ => 0.0,
            };
            (id.clone(), score + alpha * jaccard)
        }).collect();
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
    pub fn default_threshold(&self) -> f32 { 0.3 }

    /// Multi-intent gap — identical in production and simulation.
    pub fn default_gap(&self) -> f32 { 1.5 }

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
                    // Asymptotic approach to 1.0
                    e.1 = (e.1 + delta * (1.0 - e.1)).min(1.0);
                } else {
                    // Asymptotic decay toward 0.0
                    e.1 = (e.1 * (1.0 + delta)).max(0.0);
                }
            } else if delta > 0.0 {
                // New word seen in context of this intent — create edge for positive learning only.
                // For suppression (delta < 0): no edge to suppress, skip.
                entries.push((intent.to_string(), delta.min(1.0)));
            }
        }
    }

    /// Returns indices of conjunction rules that fire for the given canonical word set.
    /// Used by auto-learn to know which conjunction bonuses contributed to a routing.
    pub fn fired_conjunction_indices(&self, words: &[&str]) -> Vec<usize> {
        let word_set: std::collections::HashSet<&str> = words.iter().copied().collect();
        self.conjunctions.iter().enumerate()
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
            std::borrow::Cow::Owned(normalized.chars()
                .map(|c| if CJK_NEG.contains(&c) { ' ' } else { c })
                .collect())
        } else {
            std::borrow::Cow::Borrowed(normalized)
        };

        // Total distinct intents for IDF
        let total_intents: f32 = {
            let mut all: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for entries in self.word_intent.values() {
                for (id, _) in entries { all.insert(id.as_str()); }
            }
            all.len().max(1) as f32
        };

        let tokens = crate::tokenizer::tokenize(&query_for_tokenize);
        let mut scores: HashMap<String, f32> = HashMap::new();
        let mut has_negation = cjk_negated;

        let all_bases: std::collections::HashSet<&str> = tokens.iter()
            .map(|t| t.strip_prefix("not_").unwrap_or(t.as_str()))
            .collect();

        for token in &tokens {
            let is_negated = token.starts_with("not_");
            let base = if is_negated { &token["not_".len()..] } else { token.as_str() };
            if is_negated { has_negation = true; }
            if let Some(activations) = self.word_intent.get(base) {
                let idf = (total_intents / activations.len() as f32).ln().max(0.0);
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

        let mut result: Vec<(String, f32)> = scores
            .into_iter()
            .filter(|(_, s)| *s > 0.0)
            .collect();
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
    pub fn score_multi_normalized(&self, normalized: &str, threshold: f32, _gap: f32) -> (Vec<(String, f32)>, bool) {
        const GATE_RATIO: f32 = 0.55;
        const MAX_ROUNDS: usize = 3;

        // CJK negation pre-pass
        const CJK_NEG: &[char] = &['不', '没', '别', '未'];
        let cjk_negated = normalized.chars().any(|c| CJK_NEG.contains(&c));
        let query_for_tokenize: std::borrow::Cow<str> = if cjk_negated {
            std::borrow::Cow::Owned(normalized.chars()
                .map(|c| if CJK_NEG.contains(&c) { ' ' } else { c })
                .collect())
        } else {
            std::borrow::Cow::Borrowed(normalized)
        };

        let all_tokens: Vec<String> = crate::tokenizer::tokenize(&query_for_tokenize);
        let has_negation = cjk_negated || all_tokens.iter().any(|t| t.starts_with("not_"));

        let total_intents: f32 = {
            let mut all: std::collections::HashSet<&str> = std::collections::HashSet::new();
            for entries in self.word_intent.values() {
                for (id, _) in entries { all.insert(id.as_str()); }
            }
            all.len().max(1) as f32
        };

        let mut remaining: Vec<String> = all_tokens;
        let mut confirmed: Vec<(String, f32)> = Vec::new();
        let mut confirmed_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut original_top: f32 = 0.0;

        for round in 0..MAX_ROUNDS {
            // Score using remaining tokens only
            let scored = self.score_tokens(&remaining, total_intents, &confirmed_ids);
            if scored.is_empty() { break; }

            let round_top = scored[0].1;
            if round == 0 { original_top = round_top; }
            if round_top < threshold { break; }
            if round > 0 && round_top < original_top * GATE_RATIO { break; }

            // Confirm only intents very close to this round's top (within 10%).
            // Looser multi-intent detection comes from multiple rounds after token consumption.
            let mut round_confirmed: Vec<(String, f32)> = Vec::new();
            for (intent, score) in &scored {
                if *score >= round_top * 0.90 && *score >= threshold {
                    round_confirmed.push((intent.clone(), *score));
                    confirmed_ids.insert(intent.clone());
                }
            }

            if round_confirmed.is_empty() { break; }
            confirmed.extend(round_confirmed.iter().cloned());

            // Consume tokens: remove tokens primarily associated with confirmed intents
            remaining.retain(|token| {
                let base = token.strip_prefix("not_").unwrap_or(token.as_str());
                if let Some(activations) = self.word_intent.get(base) {
                    // Token's strongest intent
                    let best_intent = activations.iter()
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    match best_intent {
                        Some((id, _)) => !confirmed_ids.contains(id.as_str()),
                        None => true,
                    }
                } else {
                    true // unknown token, keep
                }
            });

            if remaining.is_empty() { break; }
        }

        let inhibited = self.apply_inhibition(confirmed);
        (inhibited, has_negation)
    }

    /// Score a specific set of tokens (not a full query string).
    fn score_tokens(&self, tokens: &[String], total_intents: f32,
                    exclude_intents: &std::collections::HashSet<String>) -> Vec<(String, f32)> {
        let mut scores: HashMap<String, f32> = HashMap::new();

        for token in tokens {
            let is_negated = token.starts_with("not_");
            let base = if is_negated { &token["not_".len()..] } else { token.as_str() };
            if let Some(activations) = self.word_intent.get(base) {
                let idf = (total_intents / activations.len() as f32).ln().max(0.0);
                for (intent, weight) in activations {
                    if exclude_intents.contains(intent) { continue; }
                    let delta = weight * idf;
                    *scores.entry(intent.clone()).or_insert(0.0) +=
                        if is_negated { -delta } else { delta };
                }
            }
        }

        // Conjunction bonuses
        let all_bases: std::collections::HashSet<&str> = tokens.iter()
            .map(|t| t.strip_prefix("not_").unwrap_or(t.as_str()))
            .collect();
        for rule in &self.conjunctions {
            if !exclude_intents.contains(&rule.intent)
                && rule.words.iter().all(|w| all_bases.contains(w.as_str()))
            {
                *scores.entry(rule.intent.clone()).or_insert(0.0) += rule.bonus;
            }
        }

        let mut result: Vec<(String, f32)> = scores.into_iter()
            .filter(|(_, s)| *s > 0.0)
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    // ── Library-level route API ─────────────────────────────────────────────

    /// Route a query through the full pipeline: L1 preprocessing → IDF scoring →
    /// token consumption → cross-provider disambiguation → disposition.
    ///
    /// `top_n`: how many ranked candidates to return (default 5).
    pub fn route(
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
        use std::collections::HashSet;

        if confirmed.len() < 2 { return; }

        // Group by action name (part after ':')
        let mut action_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, (id, _)) in confirmed.iter().enumerate() {
            let action = id.split(':').nth(1).unwrap_or(id.as_str());
            action_groups.entry(action.to_string()).or_default().push(i);
        }

        let duplicate_groups: Vec<Vec<usize>> = action_groups.values()
            .filter(|g| g.len() > 1).cloned().collect();

        if duplicate_groups.is_empty() { return; }

        let tokens = crate::tokenizer::tokenize(query);
        let confirmed_ids: HashSet<&str> = confirmed.iter().map(|(id, _)| id.as_str()).collect();

        // Count unique words per intent
        let mut unique_count: HashMap<&str, usize> = HashMap::new();
        for token in &tokens {
            let base = token.strip_prefix("not_").unwrap_or(token.as_str());
            if let Some(activations) = self.word_intent.get(base) {
                let matching: Vec<&str> = activations.iter()
                    .filter(|(id, _)| confirmed_ids.contains(id.as_str()))
                    .map(|(id, _)| id.as_str()).collect();
                if matching.len() == 1 {
                    *unique_count.entry(matching[0]).or_insert(0) += 1;
                }
            }
        }

        let mut to_remove: HashSet<usize> = HashSet::new();
        for group in &duplicate_groups {
            let best = group.iter()
                .max_by_key(|&&i| unique_count.get(confirmed[i].0.as_str()).copied().unwrap_or(0));
            if let Some(&best_idx) = best {
                if unique_count.get(confirmed[best_idx].0.as_str()).copied().unwrap_or(0) > 0 {
                    for &i in group {
                        if i != best_idx { to_remove.insert(i); }
                    }
                }
            }
        }

        if !to_remove.is_empty() {
            let mut i = 0;
            confirmed.retain(|_| { let keep = !to_remove.contains(&i); i += 1; keep });
        }
    }

    /// Stats: (unique_words, activation_edges, conjunctions).
    pub fn stats(&self) -> (usize, usize, usize) {
        let activation_edges: usize = self.word_intent.values().map(|v| v.len()).sum();
        (self.word_intent.len(), activation_edges, self.conjunctions.len())
    }

    // ── L3 Inhibition layer ────────────────────────────────────────────────────

    /// Learn that `false_positive` should be suppressed when `correct` fires.
    ///
    /// Each correction increments suppression strength by DELTA (capped at 1.0).
    /// After ~3 corrections the suppression is strong enough to reliably fire.
    pub fn learn_inhibition(&mut self, correct: &str, false_positive: &str) {
        // DELTA=0.4: suppression fires after the first confirmed correction (0.4 ≥ 0.35 threshold).
        // Ground truth is authoritative — trust it immediately.
        const DELTA: f32 = 0.4;
        let v = self.inhibit
            .entry(correct.to_string())
            .or_default()
            .entry(false_positive.to_string())
            .or_insert(0.0);
        *v = (*v + DELTA).min(1.0);
        eprintln!("[inhibit] learn: when '{}' fires → suppress '{}' (strength={:.2})",
            correct, false_positive, *v);
    }

    /// Apply L3 lateral inhibition to a scored result set.
    ///
    /// For each pair (A, B) in `results` where inhibit[A][B] >= threshold AND
    /// A scores meaningfully higher than B: remove B.
    ///
    /// `score_ratio_min` — B is only suppressed if A's score is at least this
    /// fraction above B (prevents suppression when both fire equally strongly,
    /// which would indicate a genuine multi-intent query).
    pub fn apply_inhibition(&self, mut results: Vec<(String, f32)>) -> Vec<(String, f32)> {
        // Fires after one authoritative ground-truth correction (strength 0.4 ≥ 0.35).
        const INHIBIT_THRESHOLD: f32 = 0.35;

        if self.inhibit.is_empty() || results.len() < 2 {
            return results;
        }

        let mut to_remove: std::collections::HashSet<String> = std::collections::HashSet::new();

        // A suppresses B if inhibit[A][B] >= threshold.
        // No score-ratio check: ground truth is authoritative. If we've seen this pair
        // corrected at least once, suppress unconditionally when A is present.
        let snapshot = results.clone();
        for (a_id, _a_score) in &snapshot {
            if let Some(suppressed) = self.inhibit.get(a_id.as_str()) {
                for (b_id, _b_score) in &snapshot {
                    if a_id == b_id { continue; }
                    let strength = suppressed.get(b_id.as_str()).copied().unwrap_or(0.0);
                    if strength >= INHIBIT_THRESHOLD {
                        eprintln!("[inhibit] suppress '{}' (strength={:.2}) because '{}' fires",
                            b_id, strength, a_id);
                        to_remove.insert(b_id.clone());
                    }
                }
            }
        }

        results.retain(|(id, _)| !to_remove.contains(id));
        results
    }

    /// Inhibition layer stats: total pairs learned.
    pub fn inhibit_stats(&self) -> usize {
        self.inhibit.values().map(|m| m.len()).sum()
    }
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
        assert_eq!(scores[0].0, "cancel_order",
            "IDF should push cancel_order above cancel_subscription (unique word 'order')");
    }

    #[test]
    fn layer3_reinforcement() {
        let (l1, mut ig) = mini_intent_graph();
        let (before, _) = ig.score(&l1, "kill the subscription");
        let kill_sub_before = before.iter().find(|(id, _)| id == "cancel_subscription")
            .map(|(_, s)| *s).unwrap_or(0.0);

        ig.reinforce(&["kill"], "cancel_subscription", 0.80);

        let (after, _) = ig.score(&l1, "kill the subscription");
        let kill_sub_after = after.iter().find(|(id, _)| id == "cancel_subscription")
            .map(|(_, s)| *s).unwrap_or(0.0);

        assert!(kill_sub_after > kill_sub_before, "reinforcement should improve score");
    }

    #[test]
    fn layer3_multi_intent() {
        let (l1, ig) = mini_intent_graph();
        let (results, _) = ig.score_multi(&l1, "cancel subscription and send message", 0.4, 2.0);
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"cancel_subscription"), "should detect cancel_subscription");
        assert!(ids.contains(&"send_message"), "should detect send_message");
    }

    #[test]
    fn layer3_negation_flags_not_suppresses() {
        let (l1, ig) = mini_intent_graph();
        // Negation subtracts activations: "don't cancel my subscription" should NOT route.
        let (with_neg, neg_flag) = ig.score(&l1, "don't cancel my subscription");
        let (without_neg, _) = ig.score(&l1, "cancel my subscription");

        let neg_score = with_neg.iter()
            .find(|(id, _)| id == "cancel_subscription").map(|(_, s)| *s).unwrap_or(0.0);
        let pos_score = without_neg.iter()
            .find(|(id, _)| id == "cancel_subscription").map(|(_, s)| *s).unwrap_or(0.0);

        assert!(neg_score <= 0.0, "cancel_subscription should be suppressed by negation (score={neg_score})");
        assert!(pos_score > 0.0, "cancel_subscription should route without negation");
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
        assert!(found, "cancel_subscription should still appear (intent is about cancellation)");
    }
}
