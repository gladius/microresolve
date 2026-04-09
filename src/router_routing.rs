//! Router: query routing (route, route_multi, route_best).

use crate::*;
use crate::tokenizer::*;
use crate::vector::LearnedVector;
use crate::index::InvertedIndex;
use std::collections::{HashMap, HashSet};

impl Router {
    pub fn route(&self, query: &str) -> Vec<RouteResult> {
        let terms = self.extract_terms(query);
        if terms.is_empty() {
            return vec![];
        }

        if self.similarity.is_empty() {
            // No similarity index — use standard search
            self.index
                .search(&terms, self.top_k)
                .into_iter()
                .map(|s| RouteResult { id: s.id, score: s.score })
                .collect()
        } else {
            // Expand with similar terms (Space 4) and use weighted search
            let term_weights: HashMap<String, f32> = terms.iter().map(|t| (t.clone(), 1.0)).collect();
            let expanded = self.expand_terms(&term_weights);
            self.index
                .search_weighted(&expanded, self.top_k)
                .into_iter()
                .map(|s| RouteResult { id: s.id, score: s.score })
                .collect()
        }
    }

    /// Expand query terms with distributionally similar terms.
    /// Similar terms get discounted weight (0.3x) to avoid overriding exact matches.
    pub(crate) fn expand_terms(&self, terms: &HashMap<String, f32>) -> HashMap<String, f32> {
        if self.similarity.is_empty() {
            return terms.clone();
        }

        let mut expanded = terms.clone();
        let all_terms = self.index.all_terms();
        let index_terms: HashSet<&str> = all_terms.iter().map(|s| s.as_str()).collect();

        for (term, weight) in terms {
            if let Some(similar) = self.similarity.get(term.as_str()) {
                for (sim_term, sim_score) in similar {
                    // Only expand to terms that exist in the index (seed vocabulary)
                    if index_terms.contains(sim_term.as_str()) && !expanded.contains_key(sim_term) {
                        expanded.insert(sim_term.clone(), weight * sim_score * self.expansion_discount);
                    }
                }
            }
        }

        expanded
    }

    /// Route and return the best match if score exceeds threshold.
    ///
    /// Returns `None` if no intent scores above the threshold.
    pub fn route_best(&self, query: &str, min_score: f32) -> Option<RouteResult> {
        let results = self.route(query);
        results.into_iter().find(|r| r.score >= min_score)
    }


    /// Route a query that may contain multiple intents.
    ///
    /// Uses greedy term consumption to decompose the query into individual
    /// intents, then re-sorts by position to match the user's original ordering.
    /// Also detects relationships (sequential, conditional, negation) between
    /// consecutive intents from gap words.
    ///
    /// When a paraphrase index is configured, each detected intent is tagged with:
    /// - `source`: "dual" (both indexes), "paraphrase" (phrase only), "routing" (term only)
    /// - `confidence`: "high" (dual), "medium" (paraphrase), "low" (routing)
    ///
    /// Supports both Latin and CJK scripts.
    ///
    /// ```
    /// use asv_router::Router;
    ///
    /// let mut router = Router::new();
    /// router.add_intent("cancel_order", &["cancel my order", "cancel order"]);
    /// router.add_intent("track_order", &["track my order", "where is my package"]);
    ///
    /// let result = router.route_multi("cancel my order and track the package", 0.3);
    /// assert!(result.intents.len() >= 2);
    /// // Intents are in positional order (left to right)
    /// assert_eq!(result.intents[0].id, "cancel_order");
    /// ```
    pub fn route_multi(&self, query: &str, threshold: f32) -> MultiRouteOutput {
        let (positioned, query_chars) = self.extract_terms_positioned(query);
        let mut output = multi::route_multi(&self.index, &self.vectors, positioned, query_chars, threshold, self.max_intents);

        // Paraphrase index: scan original message for phrase matches
        let paraphrase_hits = self.paraphrase_scan(query);
        let paraphrase_intent_ids: HashSet<String> = paraphrase_hits.iter()
            .map(|(id, _, _)| id.clone()).collect();

        // Separate streams merge: routing and paraphrase-only detections
        // don't compete on score. This prevents high-scoring routing matches from
        // crowding out paraphrase-only detections via top-N truncation.

        let routing_intent_ids: HashSet<String> = output.intents.iter()
            .map(|i| i.id.clone()).collect();

        // Stream 1: Tag routing detections with confidence tiers
        for intent in &mut output.intents {
            if paraphrase_intent_ids.contains(&intent.id) {
                // Dual-source: both indexes detected this intent → high confidence
                intent.source = "dual".to_string();
                intent.confidence = "high".to_string();
                // Boost score with paraphrase weight
                if let Some((_, weight, _)) = paraphrase_hits.iter().find(|(id, _, _)| *id == intent.id) {
                    intent.score += weight * 3.0;
                }
            } else {
                // Routing-only: use score to determine confidence
                // Score >= 5.0 → high (90%+ precision from analysis)
                // Score >= 3.0 → medium
                // Score < 3.0 → low
                intent.confidence = if intent.score >= 5.0 {
                    "high".to_string()
                } else if intent.score >= 3.0 {
                    "medium".to_string()
                } else {
                    "low".to_string()
                };
            }
        }

        // Stream 2: Paraphrase-only detections — included if score meets threshold
        for (intent_id, weight, position) in &paraphrase_hits {
            if !routing_intent_ids.contains(intent_id) {
                let score = weight * 3.0;
                if score >= threshold {
                    output.intents.push(MultiRouteResult {
                        id: intent_id.clone(),
                        score,
                        position: *position,
                        span: (*position, *position),
                        intent_type: self.get_intent_type(intent_id),
                        confidence: "medium".to_string(),
                        source: "paraphrase".to_string(),
                        negated: false,
                    });
                }
            }
        }

        // Final sort by position for output ordering
        output.intents.sort_by_key(|i| i.position);

        // Enforce max_intents cap on total output (routing + paraphrase combined)
        if output.intents.len() > self.max_intents {
            // Keep highest scoring intents
            output.intents.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            output.intents.truncate(self.max_intents);
            output.intents.sort_by_key(|i| i.position);
        }

        // Attach intent types and metadata for each detected intent
        for intent in &mut output.intents {
            intent.intent_type = self.get_intent_type(&intent.id);
        }
        for intent in &output.intents {
            if let Some(meta) = self.metadata.get(&intent.id) {
                for (key, values) in meta {
                    output.metadata
                        .entry(intent.id.clone())
                        .or_default()
                        .insert(key.clone(), values.clone());
                }
            }
        }

        // Suggest intents based on co-occurrence patterns.
        // "You detected cancel_order. 73% of customers also want refund."
        let detected_ids: Vec<&str> = output.intents.iter().map(|i| i.id.as_str()).collect();
        output.suggestions = self.suggest_intents(&detected_ids, 3, 0.2);

        output
    }

}
