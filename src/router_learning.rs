//! Router: learning from corrections.
//!
//! All training phrases — whether added as seeds or via corrections — are stored
//! in the same `training` map and term weights are recomputed from the full
//! phrase collection. There is no separate "learned_terms" layer.
//!
//! Weight formula (from training_to_terms): 0.3 + 0.65 * (count / max_count)
//! Every phrase contributes immediately. A unique correction term gets ≥0.30 weight.

use crate::*;
use crate::tokenizer::*;
use crate::vector::LearnedVector;
use std::collections::HashMap;

impl Router {
    /// Learn that `query` maps to `intent_id`.
    ///
    /// Adds the query as a training phrase and recomputes the intent's term weights
    /// from all phrases (seeds + corrections). Language is auto-detected.
    pub fn learn(&mut self, query: &str, intent_id: &str) {
        self.require_local();
        if query.trim().is_empty() { return; }

        // Auto-detect language: CJK → "zh", otherwise "en"
        let lang = if query.chars().any(is_cjk) { "zh" } else { "en" };

        // Add phrase (deduplicate)
        let lang_map = self.training.entry(intent_id.to_string()).or_default();
        let phrases = lang_map.entry(lang.to_string()).or_default();
        if !phrases.contains(&query.to_string()) {
            phrases.push(query.to_string());
        }

        // Recompute vector from all training phrases
        let all_phrases: Vec<String> = lang_map.values().flat_map(|v| v.clone()).collect();
        let terms = training_to_terms(&all_phrases);
        let vector = LearnedVector::from_phrases(terms);
        self.vectors.insert(intent_id.to_string(), vector.clone());
        self.index.update_intent(intent_id, &vector);

        // Learn paraphrases (n-gram → intent associations)
        self.learn_paraphrases(query, intent_id);

        // CJK: auto-learn situation patterns and rebuild automaton
        if query.chars().any(is_cjk) {
            self.learn_situation(query, intent_id);
            self.rebuild_cjk_automaton();
        }

        self.version += 1;
    }

    /// Correct a routing mistake: move `query` from `wrong_intent` to `correct_intent`.
    ///
    /// Removes the query phrase from `wrong_intent`'s training, recomputes its weights,
    /// then adds it to `correct_intent` via `learn`.
    pub fn correct(&mut self, query: &str, wrong_intent: &str, correct_intent: &str) {
        self.require_local();
        if query.trim().is_empty() { return; }

        // Remove phrase from wrong intent's training
        if let Some(lang_map) = self.training.get_mut(wrong_intent) {
            for phrases in lang_map.values_mut() {
                phrases.retain(|p| p != query);
            }
        }

        // Recompute wrong intent's vector from remaining phrases
        let remaining: Vec<String> = self.training.get(wrong_intent)
            .map(|m| m.values().flat_map(|v| v.clone()).collect())
            .unwrap_or_default();

        if remaining.is_empty() {
            // No phrases left — keep intent but clear its vector
            if let Some(v) = self.vectors.get_mut(wrong_intent) {
                *v = LearnedVector::new();
            }
            self.index.remove_intent(wrong_intent);
        } else {
            let terms = training_to_terms(&remaining);
            let vector = LearnedVector::from_phrases(terms);
            self.vectors.insert(wrong_intent.to_string(), vector.clone());
            self.index.update_intent(wrong_intent, &vector);
        }

        // Remove paraphrase entries pointing to wrong intent for this query
        let lower = query.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        for window_size in 3..=5 {
            if words.len() >= window_size {
                for start in 0..=(words.len() - window_size) {
                    let phrase = words[start..start + window_size].join(" ");
                    if self.paraphrase_phrases.get(&phrase).map(|(i, _)| i == wrong_intent).unwrap_or(false) {
                        self.paraphrase_phrases.remove(&phrase);
                    }
                }
            }
        }
        if self.paraphrase_phrases.get(&lower).map(|(i, _)| i == wrong_intent).unwrap_or(false) {
            self.paraphrase_phrases.remove(&lower);
        }

        // Add to correct intent
        self.learn(query, correct_intent);
    }

    /// Reinforce a correct detection: strengthen paraphrase index without changing routing weights.
    pub fn reinforce(&mut self, query: &str, intent_id: &str) {
        self.require_local();
        self.learn_paraphrases(query, intent_id);
    }

    /// Decay is a no-op in the simplified model.
    ///
    /// Phrase-based training is explicit: remove unwanted phrases rather than decaying weights.
    pub fn decay(&mut self, _factor: f32) {
        // No-op: all weights are derived from phrases. Remove phrases to reduce influence.
    }

    /// Route and return the best match with a confidence score.
    pub fn route_confident(&self, query: &str) -> Option<(RouteResult, f32)> {
        let results = self.route(query);
        if results.is_empty() { return None; }
        let confidence = if results.len() >= 2 {
            results[0].score / results[1].score
        } else {
            f32::INFINITY
        };
        Some((results[0].clone(), confidence))
    }
}
