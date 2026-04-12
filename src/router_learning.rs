//! Router: learning from corrections (phrase storage only).
//!
//! `learn()` and `correct()` store phrase associations in the training map.
//! Routing-weight updates are handled by the Hebbian L3 auto-learn system.

use crate::*;
use crate::tokenizer::is_cjk;

impl Router {
    /// Store that `query` maps to `intent_id`.
    ///
    /// Adds the query as a training phrase. Language is auto-detected (CJK → "zh", else "en").
    /// This data feeds into Hebbian bootstrap when the graph is regenerated.
    pub fn learn(&mut self, query: &str, intent_id: &str) {
        self.require_local();
        if query.trim().is_empty() { return; }

        let lang = if query.chars().any(is_cjk) { "zh" } else { "en" };

        let lang_map = self.training.entry(intent_id.to_string()).or_default();
        let phrases = lang_map.entry(lang.to_string()).or_default();
        if !phrases.contains(&query.to_string()) {
            phrases.push(query.to_string());
        }

        // CJK: store as situation pattern for persistence
        if query.chars().any(is_cjk) {
            self.learn_situation(query, intent_id);
        }

        self.version += 1;
    }

    /// Correct a routing mistake: move `query` from `wrong_intent` to `correct_intent`.
    pub fn correct(&mut self, query: &str, wrong_intent: &str, correct_intent: &str) {
        self.require_local();
        if query.trim().is_empty() { return; }

        // Remove phrase from wrong intent
        if let Some(lang_map) = self.training.get_mut(wrong_intent) {
            for phrases in lang_map.values_mut() {
                phrases.retain(|p| p != query);
            }
        }

        // Add to correct intent
        self.learn(query, correct_intent);
    }

    /// Reinforce a correct detection. No-op in Hebbian system (reinforcement handled by L3).
    pub fn reinforce(&mut self, _query: &str, _intent_id: &str) {
        // Reinforcement is handled by Hebbian L3 auto-learn (llm.rs apply_review).
    }

    /// Decay is a no-op. Remove phrases explicitly to reduce influence.
    pub fn decay(&mut self, _factor: f32) {}
}
