//! # Concept-Signal Intent Detection
//!
//! A semantic intent layer that is entirely LLM-distilled. No training, no embeddings,
//! no gradient descent. The LLM defines the semantic structure; this module executes it.
//!
//! ## Architecture (Experiment 1)
//!
//! Three layers:
//!
//! ```text
//! Query text
//!   ↓
//! Layer 1 — Concept Activation
//!   Each concept has a list of signals (words/phrases).
//!   Scan query for signal matches → activation score per concept.
//!   ↓
//! Layer 2 — Intent Scoring
//!   Each intent has a profile: weights over concepts.
//!   Dot product: activations × profile weights → intent score.
//!   ↓
//! Layer 3 — Output
//!   Return intents above threshold. Multiple intents can score simultaneously.
//!   Multi-intent detection is natural — no special logic needed.
//! ```
//!
//! ## Learning
//!
//! When a query is misrouted:
//! - LLM identifies the correct intent and which signal was missing
//! - `add_signal(concept, new_signal)` — one call, live immediately
//! - No rebuild, no retraining
//!
//! ## Example
//!
//! ```rust
//! use asv_router::concept::ConceptRegistry;
//! use std::collections::HashMap;
//!
//! let mut reg = ConceptRegistry::new();
//!
//! reg.set_concept("wants_to_stop", vec!["cancel".into(), "terminate".into(), "quit".into()]);
//! reg.set_concept("wants_refund",  vec!["refund".into(), "money back".into()]);
//!
//! let mut profile = HashMap::new();
//! profile.insert("wants_to_stop".into(), 1.0f32);
//! profile.insert("wants_refund".into(),  0.1f32);
//! reg.set_intent_profile("billing:cancel", profile);
//!
//! let results = reg.score_query("I want to terminate my plan");
//! assert_eq!(results[0].0, "billing:cancel");
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ── Core types ────────────────────────────────────────────────────────────────

/// A named semantic unit. Signals are words or phrases (case-insensitive)
/// that indicate this concept is present in a query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    pub name: String,
    pub signals: Vec<String>,
}

/// Full concept + intent registry for one namespace.
/// Serializes to/from JSON for persistence.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConceptRegistry {
    /// concept_name → list of signals (lowercase words/phrases)
    pub concepts: HashMap<String, Vec<String>>,
    /// intent_id → { concept_name → weight (0.0–1.0) }
    pub intent_profiles: HashMap<String, HashMap<String, f32>>,
}

/// Result of explaining which signals fired for a query.
#[derive(Debug, Clone)]
pub struct ConceptActivation {
    pub concept: String,
    pub score: f32,
    pub matched_signals: Vec<String>,
}

// ── Registry ─────────────────────────────────────────────────────────────────

impl ConceptRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    // ── Setup ─────────────────────────────────────────────────────────────

    /// Set (or replace) a concept's signal list.
    /// Signals are stored lowercase. Called once per concept at bootstrap.
    pub fn set_concept(&mut self, name: &str, signals: Vec<String>) {
        let lowered: Vec<String> = signals.iter().map(|s| s.to_lowercase()).collect();
        self.concepts.insert(name.to_string(), lowered);
    }

    /// Set an intent's concept weight profile.
    /// profile: { concept_name → weight }, weights typically 0.0–1.0.
    pub fn set_intent_profile(&mut self, intent_id: &str, profile: HashMap<String, f32>) {
        self.intent_profiles.insert(intent_id.to_string(), profile);
    }

    // ── Continuous learning ───────────────────────────────────────────────

    /// Add a single signal to an existing concept.
    /// This is the primary learning step: called when LLM identifies a missing signal.
    /// Idempotent — adding a duplicate is a no-op.
    pub fn add_signal(&mut self, concept: &str, signal: &str) {
        let sig = signal.to_lowercase();
        let signals = self.concepts.entry(concept.to_string()).or_default();
        if !signals.contains(&sig) {
            signals.push(sig);
        }
    }

    /// Remove a signal from a concept (e.g., a false positive that was added by mistake).
    pub fn remove_signal(&mut self, concept: &str, signal: &str) {
        let sig = signal.to_lowercase();
        if let Some(signals) = self.concepts.get_mut(concept) {
            signals.retain(|s| s != &sig);
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────

    /// Activate concepts from a query.
    ///
    /// Matching rules:
    /// - Case-insensitive
    /// - Word-boundary aware: "cancel" won't match inside "cancellation"
    /// - Phrase signals (multi-word) score higher than single-word signals
    /// - Score = sqrt(word_count_of_signal) per match — longer = stronger
    /// - Negation: if a negation word appears within 3 words before a signal,
    ///   the signal activation is suppressed
    pub fn activate(&self, query: &str) -> Vec<ConceptActivation> {
        let lower = query.to_lowercase();
        // Expand contractions before stripping apostrophes
        let expanded = lower
            .replace("don't", "do not").replace("dont", "do not")
            .replace("doesn't", "does not").replace("doesnt", "does not")
            .replace("can't", "cannot").replace("cant", "cannot")
            .replace("won't", "will not").replace("wont", "will not")
            .replace("didn't", "did not").replace("didnt", "did not")
            .replace("isn't", "is not").replace("isnt", "is not")
            .replace("aren't", "are not").replace("arent", "are not")
            .replace("haven't", "have not").replace("havent", "have not")
            .replace("shouldn't", "should not").replace("shouldnt", "should not")
            .replace("wouldn't", "would not").replace("wouldnt", "would not");
        // Normalize: replace punctuation with spaces, collapse whitespace
        let normalized: String = expanded.chars()
            .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { ' ' })
            .collect();
        // Pad with spaces for clean word-boundary matching
        let padded = format!(" {} ", normalized.split_whitespace().collect::<Vec<_>>().join(" "));

        // Build negation positions: word indices that are negated
        let words: Vec<&str> = padded.split_whitespace().collect();
        let negation_words = ["not", "no", "never", "don't", "dont", "doesn't",
                              "doesnt", "can't", "cant", "won't", "wont", "without",
                              "stop", "didn't", "didnt"];
        let negated_positions: Vec<usize> = words.iter().enumerate()
            .flat_map(|(i, w)| {
                if negation_words.contains(w) {
                    // Mark the next 3 word positions as potentially negated
                    (i+1..=(i+3).min(words.len().saturating_sub(1))).collect::<Vec<_>>()
                } else {
                    vec![]
                }
            })
            .collect();

        let mut result = Vec::new();

        for (concept, signals) in &self.concepts {
            let mut score = 0.0f32;
            let mut matched = Vec::new();

            for signal in signals {
                let padded_signal = format!(" {} ", signal);
                if !padded.contains(padded_signal.as_str()) {
                    continue;
                }

                // Find word position of signal start to check negation
                let signal_words: Vec<&str> = signal.split_whitespace().collect();
                if let Some(first_signal_word) = signal_words.first() {
                    let negated = words.iter().enumerate().any(|(i, w)| {
                        w == first_signal_word && negated_positions.contains(&i)
                    });
                    if negated { continue; }
                }

                let word_count = signal.split_whitespace().count();
                score += (word_count as f32).sqrt();
                matched.push(signal.clone());
            }

            if score > 0.0 {
                result.push(ConceptActivation {
                    concept: concept.clone(),
                    score,
                    matched_signals: matched,
                });
            }
        }

        result.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Score all intents for a query. Returns (intent_id, score) sorted descending.
    pub fn score_query(&self, query: &str) -> Vec<(String, f32)> {
        let activations = self.activate(query);
        if activations.is_empty() { return Vec::new(); }

        // Build activation map for fast lookup
        let act_map: HashMap<&str, f32> = activations.iter()
            .map(|a| (a.concept.as_str(), a.score))
            .collect();

        let mut scores: Vec<(String, f32)> = self.intent_profiles.iter()
            .filter_map(|(intent, profile)| {
                let score: f32 = profile.iter()
                    .map(|(concept, weight)| act_map.get(concept.as_str()).unwrap_or(&0.0) * weight)
                    .sum();
                if score > 0.0 { Some((intent.clone(), score)) } else { None }
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Multi-intent scoring.
    ///
    /// Returns all intents that:
    /// 1. Score >= threshold (absolute floor)
    /// 2. Score within `gap` of the top-scoring intent
    ///
    /// A genuine second intent scores close to the first because it has its own
    /// independent concept cluster that fires. Noise or weak partial matches
    /// score much lower and are excluded by the gap filter.
    pub fn score_query_multi(
        &self,
        query: &str,
        threshold: f32,
        gap: f32,
    ) -> Vec<(String, f32)> {
        let scores = self.score_query(query);
        if scores.is_empty() { return Vec::new(); }

        let top = scores[0].1;
        scores.into_iter()
            .filter(|(_, s)| *s >= threshold && top - s <= gap)
            .collect()
    }

    /// Explain which concepts fired and which signals matched — for debugging and UI.
    pub fn explain(&self, query: &str) -> Vec<ConceptActivation> {
        self.activate(query)
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

// ── Test helpers ──────────────────────────────────────────────────────────────

#[cfg(test)]
fn billing_support_deploy_registry() -> ConceptRegistry {
    let mut reg = ConceptRegistry::new();

    // ── Concepts ──────────────────────────────────────────────────────────
    reg.set_concept("wants_to_stop_service", vec![
        "cancel".into(), "terminate".into(), "discontinue".into(), "quit".into(),
        "unsubscribe".into(), "close account".into(), "stop my".into(), "end my".into(),
        "no longer need".into(), "want to leave".into(), "get out".into(),
    ]);
    reg.set_concept("wants_money_back", vec![
        "refund".into(), "money back".into(), "reimburse".into(), "reimbursement".into(),
        "return payment".into(), "get my money".into(), "repay".into(),
        "credited back".into(),
    ]);
    reg.set_concept("financial_dispute", vec![
        "dispute".into(), "chargeback".into(), "unauthorized".into(),
        "overcharged".into(), "charged twice".into(), "wrong charge".into(),
        "incorrect charge".into(), "not authorized".into(), "fraudulent".into(),
        "double charged".into(), "debited".into(),
    ]);
    reg.set_concept("software_failure", vec![
        "bug".into(), "crash".into(), "crashing".into(), "broken".into(),
        "error".into(), "not working".into(), "failure".into(), "exception".into(),
        "glitch".into(), "freezing".into(), "stopped working".into(), "erroring".into(),
    ]);
    reg.set_concept("feature_request", vec![
        "feature".into(), "add".into(), "new functionality".into(), "enhancement".into(),
        "improvement".into(), "would like".into(), "suggestion".into(), "wish".into(),
        "could you add".into(),
    ]);
    reg.set_concept("ship_to_production", vec![
        "deploy".into(), "release".into(), "ship".into(), "push to production".into(),
        "go live".into(), "launch".into(), "new version".into(), "publish".into(),
        "send to production".into(),
    ]);
    reg.set_concept("undo_deployment", vec![
        "rollback".into(), "roll back".into(), "revert".into(), "undo deployment".into(),
        "restore previous".into(), "previous version".into(), "undo release".into(),
        "take back".into(),
    ]);

    // ── Intent profiles ───────────────────────────────────────────────────
    reg.set_intent_profile("billing:cancel", {
        let mut p = HashMap::new();
        p.insert("wants_to_stop_service".into(), 1.0);
        p.insert("wants_money_back".into(), 0.1);
        p
    });
    reg.set_intent_profile("billing:refund", {
        let mut p = HashMap::new();
        p.insert("wants_money_back".into(), 1.0);
        p.insert("financial_dispute".into(), 0.3);
        p.insert("wants_to_stop_service".into(), 0.1);
        p
    });
    reg.set_intent_profile("billing:dispute", {
        let mut p = HashMap::new();
        p.insert("financial_dispute".into(), 1.0);
        p.insert("wants_money_back".into(), 0.4);
        p
    });
    reg.set_intent_profile("support:bug", {
        let mut p = HashMap::new();
        p.insert("software_failure".into(), 1.0);
        p
    });
    reg.set_intent_profile("support:feature", {
        let mut p = HashMap::new();
        p.insert("feature_request".into(), 1.0);
        p
    });
    reg.set_intent_profile("deploy:release", {
        let mut p = HashMap::new();
        p.insert("ship_to_production".into(), 1.0);
        p.insert("undo_deployment".into(), 0.0);
        p
    });
    reg.set_intent_profile("deploy:rollback", {
        let mut p = HashMap::new();
        p.insert("undo_deployment".into(), 1.0);
        p.insert("software_failure".into(), 0.3);
        p.insert("ship_to_production".into(), 0.1);
        p
    });

    reg
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn top(reg: &ConceptRegistry, q: &str) -> String {
        reg.score_query(q).into_iter().next().map(|(id,_)| id).unwrap_or_default()
    }

    // ── IVQ: in-vocabulary queries (signals exactly as defined) ──────────

    #[test]
    fn test_ivq_basic_routing() {
        let reg = billing_support_deploy_registry();
        let cases = vec![
            ("cancel my subscription",           "billing:cancel"),
            ("I want to terminate my plan",      "billing:cancel"),
            ("quit the service",                 "billing:cancel"),
            ("I need a refund for this payment", "billing:refund"),
            ("money back please",                "billing:refund"),
            ("dispute this charge",              "billing:dispute"),
            ("unauthorized charge on my account","billing:dispute"),
            ("report a bug in the app",          "support:bug"),
            ("the app keeps crashing",           "support:bug"),
            ("request a new feature",            "support:feature"),
            ("deploy to production",             "deploy:release"),
            ("ship the new version",             "deploy:release"),
            ("rollback the deployment",          "deploy:rollback"),
            ("roll back to previous version",    "deploy:rollback"),
            ("revert the release",               "deploy:rollback"),
        ];
        println!("\n  IVQ routing:");
        let mut pass = 0;
        for (q, expected) in &cases {
            let got = top(&reg, q);
            let ok = got == *expected;
            if ok { pass += 1; }
            println!("  [{}] \"{}\" → {} (expected {})", if ok {"✓"} else {"✗"}, q, got, expected);
        }
        println!("  Score: {}/{}", pass, cases.len());
        assert!(pass >= cases.len() * 4 / 5,
            "IVQ accuracy too low: {}/{}", pass, cases.len());
    }

    // ── OOV: out-of-vocabulary for term index, but in concept signals ─────

    #[test]
    fn test_oov_vocabulary_variation() {
        let reg = billing_support_deploy_registry();
        // These words are NOT in the term index training phrases but ARE in signal lists
        let cases = vec![
            ("I want to discontinue my account",    "billing:cancel"),
            ("I need to unsubscribe from this",     "billing:cancel"),
            ("please reimburse me",                 "billing:refund"),
            ("I was double charged",                "billing:dispute"),
            ("the system keeps erroring",           "support:bug"),
            ("revert to stable please",             "deploy:rollback"),
            ("go live with the new release",        "deploy:release"),
        ];
        println!("\n  OOV routing (vocabulary variation):");
        let mut pass = 0;
        for (q, expected) in &cases {
            let got = top(&reg, q);
            let ok = got == *expected;
            if ok { pass += 1; }
            println!("  [{}] \"{}\" → {} (expected {})", if ok {"✓"} else {"✗"}, q, got, expected);
        }
        println!("  Score: {}/{}", pass, cases.len());
        assert!(pass >= cases.len() * 4 / 5,
            "OOV accuracy too low: {}/{}", pass, cases.len());
    }

    // ── Multi-intent: same domain ─────────────────────────────────────────

    #[test]
    fn test_multi_intent_same_domain() {
        let reg = billing_support_deploy_registry();
        let q = "I want to cancel my subscription and also get a refund for this month";
        let results = reg.score_query_multi(q, 0.5, 2.0);

        println!("\n  Multi-intent (same domain): \"{}\"", q);
        for (id, score) in &results {
            println!("    {} : {:.2}", id, score);
        }

        let has_cancel = results.iter().any(|(id,_)| id == "billing:cancel");
        let has_refund = results.iter().any(|(id,_)| id == "billing:refund");
        assert!(has_cancel, "should detect billing:cancel");
        assert!(has_refund, "should detect billing:refund");
    }

    // ── Multi-intent: cross domain ────────────────────────────────────────

    #[test]
    fn test_multi_intent_cross_domain() {
        let reg = billing_support_deploy_registry();
        let cases = vec![
            (
                "cancel my account and rollback the last deployment",
                "billing:cancel", "deploy:rollback",
            ),
            (
                "there is a bug crashing the app and I need a refund",
                "support:bug", "billing:refund",
            ),
            (
                "deploy the new release and rollback if it crashes",
                "deploy:release", "deploy:rollback",
            ),
        ];

        println!("\n  Multi-intent (cross domain):");
        let mut pass = 0;
        for (q, intent_a, intent_b) in &cases {
            let results = reg.score_query_multi(q, 0.3, 2.0);
            let has_a = results.iter().any(|(id,_)| id == intent_a);
            let has_b = results.iter().any(|(id,_)| id == intent_b);
            let ok = has_a && has_b;
            if ok { pass += 1; }
            println!("  [{}] \"{}\"", if ok {"✓"} else {"✗"}, &q[..q.len().min(60)]);
            println!("    expected: {} + {}", intent_a, intent_b);
            println!("    got: {}", results.iter().map(|(id,s)| format!("{id}:{s:.2}")).collect::<Vec<_>>().join(", "));
        }
        println!("  Score: {}/{}", pass, cases.len());
        assert!(pass >= 1, "must detect at least 1/3 cross-domain multi-intent queries");
    }

    // ── Single-intent precision ───────────────────────────────────────────

    #[test]
    fn test_single_intent_precision() {
        let reg = billing_support_deploy_registry();
        let cases = vec![
            ("cancel my subscription I no longer need it", "billing:cancel"),
            ("the app is crashing please fix this bug",    "support:bug"),
            ("deploy the new version to production",       "deploy:release"),
        ];

        println!("\n  Single-intent precision:");
        let mut pass = 0;
        for (q, expected) in &cases {
            let results = reg.score_query_multi(q, 0.5, 0.5);
            let top_correct = results.first().map(|(id,_)| id.as_str()) == Some(expected);
            let not_too_many = results.len() <= 2;
            let ok = top_correct && not_too_many;
            if ok { pass += 1; }
            println!("  [{}] \"{}\" → {:?}", if ok {"✓"} else {"✗"},
                &q[..q.len().min(55)],
                results.iter().map(|(id,s)| format!("{id}:{s:.2}")).collect::<Vec<_>>());
        }
        assert!(pass >= 2, "single-intent precision: top intent must be correct");
    }

    // ── Continuous learning ───────────────────────────────────────────────

    #[test]
    fn test_continuous_learning() {
        let mut reg = billing_support_deploy_registry();

        // Novel expression — not in any signal list
        let q = "I want to axe my membership";
        let before = top(&reg, q);
        println!("\n  Continuous learning:");
        println!("  Before: \"{}\" → {:?}", q, before);
        // Likely empty or wrong — "axe my" and "membership" are unknown

        // LLM verdict: billing:cancel. LLM says "axe my" signals wants_to_stop_service
        reg.add_signal("wants_to_stop_service", "axe my");

        let after = top(&reg, q);
        println!("  After adding signal 'axe my': \"{}\" → {}", q, after);
        assert_eq!(after, "billing:cancel",
            "after LLM teaches 'axe my' = wants_to_stop_service, should route to billing:cancel");

        // Second novel expression
        let q2 = "please nix this subscription";
        reg.add_signal("wants_to_stop_service", "nix");
        let result2 = top(&reg, q2);
        println!("  After adding signal 'nix': \"{}\" → {}", q2, result2);
        assert_eq!(result2, "billing:cancel");
    }

    // ── Negation handling ─────────────────────────────────────────────────

    #[test]
    fn test_negation_suppression() {
        let reg = billing_support_deploy_registry();

        // "don't cancel" — negation should suppress the cancel signal
        let q = "please don't cancel my account I still need it";
        let results = reg.score_query(q);
        println!("\n  Negation: \"{}\"", q);
        println!("  Results: {:?}", results.iter().map(|(id,s)| format!("{id}:{s:.2}")).collect::<Vec<_>>());
        // billing:cancel should NOT be top result (negated)
        // It may return empty or route to something else
        let top_is_cancel = results.first().map(|(id,_)| id.as_str()) == Some("billing:cancel");
        assert!(!top_is_cancel, "negated cancel should not route to billing:cancel");
    }

    // ── Explain / interpretability ────────────────────────────────────────

    #[test]
    fn test_explain() {
        let reg = billing_support_deploy_registry();
        let q = "I was double charged and want to cancel my subscription";
        let activations = reg.explain(q);

        println!("\n  Explain: \"{}\"", q);
        for act in &activations {
            println!("    concept: {}  score: {:.2}  signals: {:?}",
                act.concept, act.score, act.matched_signals);
        }

        let concepts_fired: Vec<&str> = activations.iter().map(|a| a.concept.as_str()).collect();
        assert!(concepts_fired.contains(&"financial_dispute"),
            "double charged should fire financial_dispute");
        assert!(concepts_fired.contains(&"wants_to_stop_service"),
            "cancel should fire wants_to_stop_service");
    }

    // ── Save / load ───────────────────────────────────────────────────────

    #[test]
    fn test_save_load() {
        let reg = billing_support_deploy_registry();
        let path = "/tmp/test_concept_registry.json";

        reg.save(path).expect("save failed");
        let loaded = ConceptRegistry::load(path).expect("load failed");

        // Verify routing works after load
        assert_eq!(
            loaded.score_query("cancel my subscription").first().map(|(id,_)| id.as_str()),
            Some("billing:cancel")
        );
        assert_eq!(
            loaded.score_query("rollback the deployment").first().map(|(id,_)| id.as_str()),
            Some("deploy:rollback")
        );

        std::fs::remove_file(path).ok();
    }

    // ── No signal — unknown query ─────────────────────────────────────────

    #[test]
    fn test_unknown_query_returns_empty() {
        let reg = billing_support_deploy_registry();
        // Completely unrelated query with no concept signals
        let results = reg.score_query("the weather is nice today");
        println!("\n  Unknown query results: {:?}", results);
        // Should return empty or very low scores — nothing confidently matches
        // Either empty or all scores below 0.1
        let confident = results.iter().any(|(_, s)| *s > 0.5);
        assert!(!confident, "unrelated query should not score confidently");
    }

    // ── Full benchmark: IVQ + OOV + MIXED ────────────────────────────────

    #[test]
    fn test_full_benchmark() {
        let reg = billing_support_deploy_registry();

        let benchmark: Vec<(&str, &str, &str)> = vec![
            // IVQ
            ("cancel my subscription",              "billing:cancel",  "IVQ"),
            ("I need a refund",                     "billing:refund",  "IVQ"),
            ("dispute this unauthorized charge",    "billing:dispute", "IVQ"),
            ("there is a bug in the app",           "support:bug",     "IVQ"),
            ("request a new feature",               "support:feature", "IVQ"),
            ("deploy to production",                "deploy:release",  "IVQ"),
            ("rollback the deployment",             "deploy:rollback", "IVQ"),
            // OOV — vocabulary variation (in signal list, not in typical training phrases)
            ("I want to terminate my plan",         "billing:cancel",  "OOV"),
            ("please reimburse me",                 "billing:refund",  "OOV"),
            ("I was double charged",                "billing:dispute", "OOV"),
            ("the app keeps erroring",              "support:bug",     "OOV"),
            ("revert to stable",                    "deploy:rollback", "OOV"),
            ("go live with the new version",        "deploy:release",  "OOV"),
            // MIXED — ambiguous signals, cross-domain vocabulary
            ("cancel my plan and get a refund",     "billing:cancel",  "MIXED"),
            ("rollback failed and app is broken",   "deploy:rollback", "MIXED"),
            ("deploy failed the app is crashing",   "support:bug",     "MIXED"),
            ("revert the release it broke things",  "deploy:rollback", "MIXED"),
            ("bug caused a chargeback dispute",     "billing:dispute", "MIXED"),
            ("the feature I requested has errors",  "support:feature", "MIXED"),
            ("ship the fix for the crashing bug",   "support:bug",     "MIXED"),
        ];

        let mut ivq = (0usize, 0usize);
        let mut oov = (0usize, 0usize);
        let mut mixed = (0usize, 0usize);

        println!("\n╔════════════════════════════════════════════════════════╗");
        println!(  "║  ConceptRegistry full benchmark (Experiment 1)         ║");
        println!(  "╠════════════════════════════════════════════════════════╣");

        for (q, expected, cat) in &benchmark {
            let got = top(&reg, q);
            let ok = got == *expected;
            match *cat {
                "IVQ"   => { ivq.1 += 1; if ok { ivq.0 += 1; } }
                "OOV"   => { oov.1 += 1; if ok { oov.0 += 1; } }
                "MIXED" => { mixed.1 += 1; if ok { mixed.0 += 1; } }
                _ => {}
            }
            println!("  [{}] [{cat}] \"{}\" → {} (expected {})",
                if ok {"✓"} else {"✗"}, &q[..q.len().min(45)], got, expected);
        }

        let total = ivq.0 + oov.0 + mixed.0;
        let n = benchmark.len();
        println!("╠════════════════════════════════════════════════════════╣");
        println!("║  IVQ:   {}/{}", ivq.0, ivq.1);
        println!("║  OOV:   {}/{}", oov.0, oov.1);
        println!("║  MIXED: {}/{}", mixed.0, mixed.1);
        println!("║  TOTAL: {}/{}", total, n);
        println!("╚════════════════════════════════════════════════════════╝");

        assert!(ivq.0 >= ivq.1 * 4 / 5,     "IVQ must be ≥80%");
        assert!(oov.0 >= oov.1 * 4 / 5,     "OOV must be ≥80%");
        assert!(total >= n * 3 / 4,          "Overall must be ≥75%");
    }

    // ── Real namespace: ip-poc-v2 (Stripe + Linear + Vercel) ─────────────
    //
    // Registry built from actual ip-poc-v2 intent data.
    // This is what the LLM bootstrap call would produce automatically.
    // Concepts derived from intent descriptions + training phrases.

    fn ip_poc_registry() -> ConceptRegistry {
        let mut reg = ConceptRegistry::new();

        // Concepts — what the LLM would define from the intent descriptions
        reg.set_concept("stop_subscription", vec![
            "cancel".into(), "unsubscribe".into(), "end my membership".into(),
            "stop recurring".into(), "turn off auto-renewal".into(),
            "remove my subscription".into(), "kill this subscription".into(),
            "cancel recurring billing".into(), "cancel subscription".into(),
            "stop my subscription".into(), "discontinue".into(), "terminate".into(),
            "stop the recurring payment".into(), "how do i unsubscribe".into(),
        ]);
        reg.set_concept("get_money_back", vec![
            "refund".into(), "reimburse".into(), "reimbursement".into(),
            "money back".into(), "reverse this charge".into(), "issue a refund".into(),
            "send money back".into(), "undo the payment".into(),
            "process a reimbursement".into(), "refund the transaction".into(),
            "get my money back".into(), "credit back".into(),
        ]);
        reg.set_concept("billing_dispute", vec![
            "dispute".into(), "chargeback".into(), "respond to a chargeback".into(),
            "submit evidence".into(), "mark this dispute".into(),
            "provide my response".into(), "contest".into(), "contesting".into(),
            "claim".into(), "respond to the claim".into(), "close out this dispute".into(),
            "i spoke with them about the charge".into(), "fraudulent".into(),
            "unauthorized".into(), "fight this charge".into(),
        ]);
        reg.set_concept("create_work_item", vec![
            "create issue".into(), "report a problem".into(), "create a task".into(),
            "create a work item".into(), "submit a ticket".into(),
            "open something for tracking".into(), "submit a work request".into(),
            "create a task for the queue".into(), "log a bug".into(),
            "open a ticket".into(), "file an issue".into(), "track this".into(),
            "add to backlog".into(),
        ]);
        reg.set_concept("deploy_code", vec![
            "deploy to vercel".into(), "push this live".into(), "ship it to vercel".into(),
            "i want to deploy".into(), "launch this project".into(),
            "get this code on vercel".into(), "go ahead and deploy".into(),
            "publish the latest changes".into(), "trigger a deployment".into(),
            "send the update live".into(), "deploy the current build".into(),
            "deploy".into(), "ship".into(), "push live".into(), "go live".into(),
            "launch".into(), "publish".into(),
        ]);

        // Intent profiles — how strongly each concept signals each intent
        reg.set_intent_profile("stripe:cancel_subscription", {
            let mut p = HashMap::new();
            p.insert("stop_subscription".into(), 1.0);
            p.insert("get_money_back".into(), 0.05);
            p
        });
        reg.set_intent_profile("stripe:create_refund", {
            let mut p = HashMap::new();
            p.insert("get_money_back".into(), 1.0);
            p.insert("billing_dispute".into(), 0.2);
            p
        });
        reg.set_intent_profile("stripe:update_dispute", {
            let mut p = HashMap::new();
            p.insert("billing_dispute".into(), 1.0);
            p.insert("get_money_back".into(), 0.3);
            p
        });
        reg.set_intent_profile("linear:create_issue", {
            let mut p = HashMap::new();
            p.insert("create_work_item".into(), 1.0);
            p
        });
        reg.set_intent_profile("vercel:deploy_to_vercel", {
            let mut p = HashMap::new();
            p.insert("deploy_code".into(), 1.0);
            p
        });

        reg
    }

    #[test]
    fn test_ip_poc_real_namespace() {
        let reg = ip_poc_registry();

        // IVQ: taken directly from the actual training phrases in the namespace
        let ivq: Vec<(&str, &str)> = vec![
            ("cancel subscription",                    "stripe:cancel_subscription"),
            ("stop my subscription",                   "stripe:cancel_subscription"),
            ("i want to unsubscribe",                  "stripe:cancel_subscription"),
            ("cancel this recurring charge",           "stripe:cancel_subscription"),
            ("end my membership",                      "stripe:cancel_subscription"),
            ("i need to refund this payment",          "stripe:create_refund"),
            ("process a refund",                       "stripe:create_refund"),
            ("reverse this charge",                    "stripe:create_refund"),
            ("send money back",                        "stripe:create_refund"),
            ("i need to respond to a chargeback",      "stripe:update_dispute"),
            ("submit evidence for this chargeback",    "stripe:update_dispute"),
            ("provide my response to the claim",       "stripe:update_dispute"),
            ("create issue",                           "linear:create_issue"),
            ("submit a ticket",                        "linear:create_issue"),
            ("create a task for the queue",            "linear:create_issue"),
            ("deploy to vercel",                       "vercel:deploy_to_vercel"),
            ("push this live",                         "vercel:deploy_to_vercel"),
            ("trigger a deployment",                   "vercel:deploy_to_vercel"),
            ("publish the latest changes",             "vercel:deploy_to_vercel"),
        ];

        // OOV: realistic user queries NOT in the training phrases
        // These test whether the concept signals generalise beyond exact training vocabulary
        let oov: Vec<(&str, &str)> = vec![
            ("I want to discontinue my plan",           "stripe:cancel_subscription"),
            ("I need to terminate this subscription",   "stripe:cancel_subscription"),
            ("please reimburse me for that charge",     "stripe:create_refund"),
            ("I want to get my money back",             "stripe:create_refund"),
            ("I need to contest this transaction",      "stripe:update_dispute"),
            ("I want to fight this fraudulent charge",  "stripe:update_dispute"),
            ("log a bug for the engineering team",      "linear:create_issue"),
            ("open a ticket for this problem",          "linear:create_issue"),
            ("ship it and go live",                     "vercel:deploy_to_vercel"),
            ("launch the project now",                  "vercel:deploy_to_vercel"),
        ];

        // Multi-intent: realistic compound requests
        let multi: Vec<(&str, &str, &str)> = vec![
            ("cancel my subscription and refund my last payment",
             "stripe:cancel_subscription", "stripe:create_refund"),
            ("there is a chargeback dispute and I also want a refund",
             "stripe:update_dispute", "stripe:create_refund"),
            ("create a ticket and deploy to vercel when done",
             "linear:create_issue", "vercel:deploy_to_vercel"),
        ];

        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!(  "║  ip-poc-v2 namespace: real Stripe + Linear + Vercel intents  ║");
        println!(  "╠══════════════════════════════════════════════════════════════╣");

        // IVQ
        println!("\n  [IVQ — exact training phrases]");
        let mut ivq_pass = 0;
        for (q, expected) in &ivq {
            let got = reg.score_query(q).into_iter().next().map(|(id,_)| id).unwrap_or_default();
            let ok = got == *expected;
            if ok { ivq_pass += 1; }
            println!("  [{}] \"{}\" → {}", if ok {"✓"} else {"✗"}, q, got);
        }

        // OOV
        println!("\n  [OOV — vocabulary variations not in training phrases]");
        let mut oov_pass = 0;
        for (q, expected) in &oov {
            let got = reg.score_query(q).into_iter().next().map(|(id,_)| id).unwrap_or_default();
            let ok = got == *expected;
            if ok { oov_pass += 1; }
            println!("  [{}] \"{}\" → {}", if ok {"✓"} else {"✗"}, q, got);
        }

        // Multi-intent
        println!("\n  [Multi-intent — compound requests]");
        let mut multi_pass = 0;
        for (q, a, b) in &multi {
            let results = reg.score_query_multi(q, 0.3, 2.0);
            let has_a = results.iter().any(|(id,_)| id == a);
            let has_b = results.iter().any(|(id,_)| id == b);
            let ok = has_a && has_b;
            if ok { multi_pass += 1; }
            println!("  [{}] \"{}\"", if ok {"✓"} else {"✗"}, &q[..q.len().min(60)]);
            println!("    expected: {} + {}", a, b);
            println!("    got: {}", results.iter().map(|(id,s)| format!("{id}:{s:.2}")).collect::<Vec<_>>().join(", "));
        }

        println!("\n╠══════════════════════════════════════════════════════════════╣");
        println!("║  IVQ:        {}/{}",   ivq_pass, ivq.len());
        println!("║  OOV:        {}/{}",   oov_pass, oov.len());
        println!("║  Multi-intent: {}/{}", multi_pass, multi.len());
        println!("║  TOTAL:      {}/{}", ivq_pass + oov_pass, ivq.len() + oov.len());
        println!("╚══════════════════════════════════════════════════════════════╝");

        assert!(ivq_pass >= ivq.len() * 4 / 5,
            "IVQ must be ≥80%: got {}/{}", ivq_pass, ivq.len());
        assert!(oov_pass >= oov.len() * 4 / 5,
            "OOV must be ≥80%: got {}/{}", oov_pass, oov.len());
        assert!(multi_pass >= 2,
            "Multi-intent must detect ≥2/3: got {}/3", multi_pass);
    }
}
