//! Router: phrase storage and corrections.

use crate::*;
use crate::tokenizer::is_cjk;

impl Router {
    /// Add a phrase to an intent. Language is auto-detected (CJK → "zh", else "en").
    /// Indexes immediately so routing improves without a full rebuild.
    pub fn add_phrase_auto(&mut self, query: &str, intent_id: &str) {
        self.require_local();
        if query.trim().is_empty() { return; }

        let lang = if query.chars().any(is_cjk) { "zh" } else { "en" };

        let lang_map = self.training.entry(intent_id.to_string()).or_default();
        let phrases = lang_map.entry(lang.to_string()).or_default();
        if !phrases.contains(&query.to_string()) {
            phrases.push(query.to_string());
            self.index_phrase(intent_id, query);
        }

        self.version += 1;
    }

    /// Correct a routing mistake: move `query` from `wrong_intent` to `correct_intent`.
    pub fn correct(&mut self, query: &str, wrong_intent: &str, correct_intent: &str) {
        self.require_local();
        if query.trim().is_empty() { return; }

        if let Some(lang_map) = self.training.get_mut(wrong_intent) {
            for phrases in lang_map.values_mut() {
                phrases.retain(|p| p != query);
            }
        }

        self.add_phrase_auto(query, correct_intent);
    }
}
