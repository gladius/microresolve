//! Router: learning from corrections.

use crate::*;
use crate::tokenizer::*;
use crate::vector::LearnedVector;
use crate::index::InvertedIndex;
use std::collections::{HashMap, HashSet};

impl Router {
    pub fn learn(&mut self, query: &str, intent_id: &str) {
        self.require_local();
        let terms = self.extract_terms_for_learning(query);
        if terms.is_empty() {
            return;
        }

        // Store raw phrase under "_learned" language key
        let lang_map = self.training.entry(intent_id.to_string()).or_default();
        lang_map.entry("_learned".to_string()).or_default().push(query.to_string());

        // Update learned weights
        if let Some(vector) = self.vectors.get_mut(intent_id) {
            vector.learn(&terms);
            self.index.update_intent(intent_id, vector);
        } else {
            let mut vector = LearnedVector::new();
            vector.learn(&terms);
            self.vectors.insert(intent_id.to_string(), vector.clone());
            self.index.update_intent(intent_id, &vector);
        }

        // Learn paraphrases (n-gram extraction into paraphrase index)
        self.learn_paraphrases(query, intent_id);

        // Situation index auto-learning — CJK only.
        //
        // Why CJK and not Latin?
        //
        // CJK char bigrams/trigrams ("跑红", "OOM了", "挂了") are compound words —
        // 2-3 characters encode a complete domain-specific concept. They make high-
        // precision situation patterns because the same compound rarely appears in
        // unrelated intents. Auto-learning from corrections is safe and valuable.
        //
        // Latin char bigrams ("ca", "nc", "el") are sub-word noise — they appear in
        // every word across every intent. The guard blocks most of them, so learning
        // pays write cost for near-zero benefit. Latin situation patterns should be
        // configured manually (operator adds "chargeback", "LGTM", "402") or via LLM
        // generation — not learned automatically from corrections.
        let has_cjk = query.chars().any(is_cjk);
        if has_cjk {
            self.learn_situation(query, intent_id);
        }

        // Only rebuild CJK automaton if the query had CJK characters
        if query.chars().any(is_cjk) {
            self.rebuild_cjk_automaton();
        }
        self.version += 1;
    }

    /// Correct a routing mistake: move query from wrong intent to right intent.
    ///
    /// Unlearns the query from the wrong intent's routing index and paraphrase index,
    /// then learns it into the correct intent for both indexes.
    pub fn correct(&mut self, query: &str, wrong_intent: &str, correct_intent: &str) {
        self.require_local();
        let terms = self.extract_terms(query);
        if terms.is_empty() {
            return;
        }

        // Unlearn from wrong (use all extracted terms for thorough unlearning)
        if let Some(vector) = self.vectors.get_mut(wrong_intent) {
            vector.unlearn(&terms);
            self.index.update_intent(wrong_intent, vector);
        }
        if let Some(lang_map) = self.training.get_mut(wrong_intent) {
            for phrases in lang_map.values_mut() {
                phrases.retain(|p| p != query);
            }
        }

        // Remove paraphrase entries that point to the wrong intent for this query
        let lower = query.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        for window_size in 3..=5 {
            if words.len() >= window_size {
                for start in 0..=(words.len() - window_size) {
                    let phrase: String = words[start..start + window_size].join(" ");
                    if let Some((intent, _)) = self.paraphrase_phrases.get(&phrase) {
                        if intent == wrong_intent {
                            self.paraphrase_phrases.remove(&phrase);
                        }
                    }
                }
            }
        }
        // Remove full message paraphrase if it pointed to wrong intent
        if let Some((intent, _)) = self.paraphrase_phrases.get(&lower) {
            if intent == wrong_intent {
                self.paraphrase_phrases.remove(&lower);
            }
        }

        // Learn on correct (uses selective CJK extraction + paraphrase learning)
        self.learn(query, correct_intent);
    }

    /// Reinforce a correct detection: learn paraphrase n-grams without modifying
    /// routing weights. Call this when routing already detected the correct intent
    /// to strengthen the paraphrase index's association with this message.
    ///
    /// This mirrors the clean experiment's behavior where correct detections were
    /// reinforced through `paraphrase_index.learn_from_message()`.
    pub fn reinforce(&mut self, query: &str, intent_id: &str) {
        self.require_local();
        self.learn_paraphrases(query, intent_id);
    }

    /// Apply decay to all learned weights.
    ///
    /// Multiplies learned weights by `factor` (e.g., 0.9) and prunes
    /// terms below threshold. Call periodically to forget stale associations.
    pub fn decay(&mut self, factor: f32) {
        for (id, vector) in &mut self.vectors {
            vector.decay(factor);
            self.index.update_intent(id, vector);
        }
        self.rebuild_cjk_automaton();
    }

    /// Route and return the best match with a confidence score.
    ///
    /// Confidence = top1_score / top2_score. High confidence (>2.0) means
    /// the top intent stands out clearly. Low confidence (~1.0) means
    /// multiple intents scored similarly — likely ambiguous or out-of-scope.
    ///
    /// Returns `None` if no intent matches any query terms.
    pub fn route_confident(&self, query: &str) -> Option<(RouteResult, f32)> {
        let results = self.route(query);
        if results.is_empty() {
            return None;
        }
        let confidence = if results.len() >= 2 {
            results[0].score / results[1].score
        } else {
            f32::INFINITY
        };
        Some((results[0].clone(), confidence))
    }

    // Route a query that may contain multiple intents.
    //
    // Uses greedy term consumption to decompose the query into individual
    // intents, then re-sorts by position to match the user's original ordering.
    // Also detects relationships (sequential, conditional, negation) between
    // consecutive intents from gap words.
    //
    // When a paraphrase index is configured, each detected intent is tagged with:
    // - `source`: "dual" (both indexes), "paraphrase" (phrase only), "routing" (term only)
    // - `confidence`: "high" (dual), "medium" (paraphrase), "low" (routing)
    //
    // Supports both Latin and CJK scripts.
    //
    // ```
    // use asv_router::Router;
    //
    // let mut router = Router::new();
    // router.add_intent("cancel_order", &["cancel my order", "cancel order"]);
    // router.add_intent("track_order", &["track my order", "where is my package"]);
    //
    // let result = router.route_multi("cancel my order and track the package", 0.3);
    // assert!(result.intents.len() >= 2);
    // // Intents are in positional order (left to right)
    // assert_eq!(result.intents[0].id, "cancel_order");
    // ```

}
