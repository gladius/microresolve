//! Resolver: continuous learning — phrase moves between intents.

use crate::*;
use crate::tokenizer::is_cjk;

impl Resolver {
    /// Move `query` from `wrong_intent` to `correct_intent` and reinforce.
    ///
    /// Language is auto-detected (CJK → "zh", else "en"). The phrase is indexed
    /// into the correct intent so routing improves immediately.
    ///
    /// Returns `Err(Error::IntentNotFound)` if `correct_intent` does not exist —
    /// this guards against typos silently spawning phantom intents. To create a
    /// new intent, call `add_intent` first.
    ///
    /// Empty queries are accepted as a no-op and return `Ok(())`. Removing from
    /// `wrong_intent` is best-effort: if `wrong_intent` does not exist, no error
    /// is raised (the phrase simply isn't there to remove).
    pub fn correct(
        &mut self,
        query: &str,
        wrong_intent: &str,
        correct_intent: &str,
    ) -> Result<(), Error> {
        if query.trim().is_empty() {
            return Ok(());
        }

        // Guard: correct_intent must exist. Without this guard a typo would
        // silently create a phantom intent via entry().or_default().
        if !self.training.contains_key(correct_intent) {
            return Err(Error::IntentNotFound(correct_intent.to_string()));
        }

        // Drop the phrase from the wrong intent's training (if present).
        if let Some(lang_map) = self.training.get_mut(wrong_intent) {
            for phrases in lang_map.values_mut() {
                phrases.retain(|p| p != query);
            }
        }

        // Add to the correct intent.
        let lang = if query.chars().any(is_cjk) { "zh" } else { "en" };
        let lang_map = self.training.get_mut(correct_intent).expect("checked above");
        let phrases = lang_map.entry(lang.to_string()).or_default();
        if !phrases.contains(&query.to_string()) {
            phrases.push(query.to_string());
            self.index_phrase(correct_intent, query);
        }
        self.version += 1;
        Ok(())
    }
}
