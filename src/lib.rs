//! # ASV Router
//!
//! Model-free intent routing with incremental learning.
//! Sub-millisecond, no embeddings, no GPU, no neural network.
//!
//! ## Quick Start
//!
//! ```
//! use asv_router::Router;
//!
//! let mut router = Router::new();
//!
//! // Add intents with seed phrases
//! router.add_intent("cancel_order", &[
//!     "cancel my order",
//!     "I want to cancel",
//!     "stop my order",
//! ]);
//! router.add_intent("track_order", &[
//!     "where is my package",
//!     "track my order",
//!     "shipping status",
//! ]);
//!
//! // Route a query
//! let result = router.route("I need to cancel something");
//! assert_eq!(result[0].id, "cancel_order");
//!
//! // Learn from user correction
//! router.learn("stop charging me", "cancel_order");
//!
//! // Now "stop charging me" routes correctly
//! let result = router.route("stop charging me");
//! assert_eq!(result[0].id, "cancel_order");
//! ```
//!
//! ## How It Works
//!
//! Each intent has a **dual-layer sparse vector**:
//! - **Seed layer**: Generated from example phrases at setup time (immutable)
//! - **Learned layer**: Grows from user corrections (asymptotic toward 1.0)
//!
//! Routing tokenizes the query into unigrams + bigrams, looks up matching
//! intents via an inverted index, and scores by summing `max(seed, learned)`
//! per term. The entire operation is a HashMap lookup — no matrix math,
//! no model inference.
//!
//! ## When to Use This
//!
//! - You have 10-1000 intents (support tickets, chatbot routing, command dispatch)
//! - You need sub-millisecond latency (edge, mobile, IoT)
//! - You want interpretable routing (see exactly why intent X was chosen)
//! - You want the system to learn from corrections without retraining
//! - You don't want to host an embedding model
//!
//! ## When NOT to Use This
//!
//! - You need semantic understanding ("stop charging me" won't match "cancel subscription" without training)
//! - You have 10K+ intents with heavy overlap
//! - You need deep semantic multilingual matching (tokenizer handles any Unicode but stop words are English-centric)

pub mod index;
pub mod multi;
pub mod tokenizer;
pub mod vector;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use multi::{IntentRelation, MultiRouteOutput, MultiRouteResult};

use index::InvertedIndex;
use std::collections::HashMap;
use tokenizer::{tokenize, training_to_terms};
use vector::LearnedVector;

/// Intent router with incremental learning.
///
/// The main entry point for the library. Manages intents, routing, and learning.
pub struct Router {
    vectors: HashMap<String, LearnedVector>,
    index: InvertedIndex,
    /// Raw training phrases per intent, grouped by language code.
    /// Structure: { intent_id: { lang_code: [phrases] } }
    training: HashMap<String, HashMap<String, Vec<String>>>,
    top_k: usize,
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

impl Router {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            index: InvertedIndex::new(),
            training: HashMap::new(),
            top_k: 10,
        }
    }

    /// Set the maximum number of results returned by `route()`.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Add an intent with seed phrases.
    ///
    /// Seed phrases are example queries that should route to this intent.
    /// They are tokenized into term weights and used for matching.
    ///
    /// ```
    /// use asv_router::Router;
    ///
    /// let mut router = Router::new();
    /// router.add_intent("greeting", &["hello", "hi there", "hey"]);
    /// ```
    pub fn add_intent(&mut self, id: &str, seed_phrases: &[&str]) {
        let phrases: Vec<String> = seed_phrases.iter().map(|s| s.to_string()).collect();
        let terms = training_to_terms(&phrases);
        let vector = LearnedVector::from_seed(terms);
        self.vectors.insert(id.to_string(), vector);
        let mut lang_map = HashMap::new();
        lang_map.insert("en".to_string(), phrases);
        self.training.insert(id.to_string(), lang_map);
        self.rebuild_index();
    }

    /// Add an intent with seed phrases grouped by language.
    ///
    /// All phrases across all languages are indexed together into one flat vector.
    /// Language grouping is preserved in the datastore for display/export.
    ///
    /// ```
    /// use asv_router::Router;
    /// use std::collections::HashMap;
    ///
    /// let mut router = Router::new();
    /// let mut seeds = HashMap::new();
    /// seeds.insert("en".to_string(), vec!["cancel my order".to_string()]);
    /// seeds.insert("es".to_string(), vec!["cancelar mi pedido".to_string()]);
    /// router.add_intent_multilingual("cancel_order", seeds);
    /// ```
    pub fn add_intent_multilingual(&mut self, id: &str, seeds_by_lang: HashMap<String, Vec<String>>) {
        let all_phrases: Vec<String> = seeds_by_lang.values().flat_map(|v| v.clone()).collect();
        let terms = training_to_terms(&all_phrases);
        let vector = LearnedVector::from_seed(terms);
        self.vectors.insert(id.to_string(), vector);
        self.training.insert(id.to_string(), seeds_by_lang);
        self.rebuild_index();
    }

    /// Add an intent with pre-computed term weights.
    ///
    /// Use this when you have term weights from an external source
    /// (e.g., LLM-generated, imported from another system).
    pub fn add_intent_with_weights(&mut self, id: &str, seed_terms: HashMap<String, f32>) {
        let vector = LearnedVector::from_seed(seed_terms);
        self.vectors.insert(id.to_string(), vector);
        self.rebuild_index();
    }

    /// Remove an intent.
    pub fn remove_intent(&mut self, id: &str) {
        self.vectors.remove(id);
        self.training.remove(id);
        self.index.remove_intent(id);
    }

    /// Route a query to matching intents, ranked by score.
    ///
    /// Returns up to `top_k` results (default 10), sorted by score descending.
    /// Empty results means no intent matched any query terms.
    pub fn route(&self, query: &str) -> Vec<RouteResult> {
        let terms = tokenize(query);
        if terms.is_empty() {
            return vec![];
        }

        self.index
            .search(&terms, self.top_k)
            .into_iter()
            .map(|s| RouteResult {
                id: s.id,
                score: s.score,
            })
            .collect()
    }

    /// Route and return the best match if score exceeds threshold.
    ///
    /// Returns `None` if no intent scores above the threshold.
    pub fn route_best(&self, query: &str, min_score: f32) -> Option<RouteResult> {
        let results = self.route(query);
        results.into_iter().find(|r| r.score >= min_score)
    }

    /// Learn from a user correction: this query should route to this intent.
    ///
    /// Appends the query to the intent's training phrases and reinforces
    /// learned term weights. Weights grow asymptotically toward 1.0.
    pub fn learn(&mut self, query: &str, intent_id: &str) {
        let terms = tokenize(query);
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
    }

    /// Correct a routing mistake: move query from wrong intent to right intent.
    pub fn correct(&mut self, query: &str, wrong_intent: &str, correct_intent: &str) {
        let terms = tokenize(query);
        if terms.is_empty() {
            return;
        }

        // Unlearn from wrong
        if let Some(vector) = self.vectors.get_mut(wrong_intent) {
            vector.unlearn(&terms);
            self.index.update_intent(wrong_intent, vector);
        }
        if let Some(lang_map) = self.training.get_mut(wrong_intent) {
            for phrases in lang_map.values_mut() {
                phrases.retain(|p| p != query);
            }
        }

        // Learn on correct
        self.learn(query, correct_intent);
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

    /// Route a query that may contain multiple intents.
    ///
    /// Uses greedy term consumption to decompose the query into individual
    /// intents, then re-sorts by position to match the user's original ordering.
    /// Also detects relationships (sequential, conditional, negation) between
    /// consecutive intents from gap words.
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
        multi::route_multi(&self.index, &self.vectors, query, threshold)
    }

    /// Export router state as JSON for persistence.
    pub fn export_json(&self) -> String {
        let state = RouterState {
            intents: self.vectors.clone(),
            training: self.training.clone(),
            top_k: self.top_k,
        };
        serde_json::to_string(&state).unwrap_or_default()
    }

    /// Import router state from JSON.
    pub fn import_json(json: &str) -> Result<Self, String> {
        let state: RouterState =
            serde_json::from_str(json).map_err(|e| format!("invalid JSON: {}", e))?;

        let index = InvertedIndex::build(&state.intents);

        Ok(Self {
            vectors: state.intents,
            index,
            training: state.training,
            top_k: state.top_k,
        })
    }

    /// Number of registered intents.
    pub fn intent_count(&self) -> usize {
        self.vectors.len()
    }

    /// Get the vector for an intent (for inspection/debugging).
    pub fn get_vector(&self, intent_id: &str) -> Option<&LearnedVector> {
        self.vectors.get(intent_id)
    }

    /// Get all intent IDs.
    pub fn intent_ids(&self) -> Vec<String> {
        self.vectors.keys().cloned().collect()
    }

    /// Get all training phrases for an intent (flat, all languages combined).
    pub fn get_training(&self, intent_id: &str) -> Option<Vec<String>> {
        self.training.get(intent_id).map(|lang_map| {
            lang_map.values().flat_map(|v| v.clone()).collect()
        })
    }

    /// Get training phrases grouped by language.
    pub fn get_training_by_lang(&self, intent_id: &str) -> Option<&HashMap<String, Vec<String>>> {
        self.training.get(intent_id)
    }

    fn rebuild_index(&mut self) {
        self.index = InvertedIndex::build(&self.vectors);
    }
}

/// A routing result.
#[derive(Debug, Clone)]
pub struct RouteResult {
    /// The intent identifier.
    pub id: String,
    /// Match score (higher = better match).
    pub score: f32,
}

/// Serializable router state for persistence.
#[derive(serde::Serialize, serde::Deserialize)]
struct RouterState {
    intents: HashMap<String, LearnedVector>,
    /// Training phrases grouped by language: { intent_id: { lang: [phrases] } }
    training: HashMap<String, HashMap<String, Vec<String>>>,
    top_k: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_routing() {
        let mut router = Router::new();
        router.add_intent("cancel_order", &[
            "cancel my order",
            "I want to cancel",
            "stop my order",
        ]);
        router.add_intent("track_order", &[
            "where is my package",
            "track my order",
            "shipping status",
        ]);

        let result = router.route("I need to cancel something");
        assert!(!result.is_empty());
        assert_eq!(result[0].id, "cancel_order");

        let result = router.route("where is my package");
        assert_eq!(result[0].id, "track_order");
    }

    #[test]
    fn learning_improves_routing() {
        let mut router = Router::new();
        router.add_intent("cancel_sub", &["cancel subscription"]);

        // Before learning: "stop charging me" has no term overlap
        let before = router.route("stop charging me");
        let cancel_before = before.iter().find(|r| r.id == "cancel_sub");

        // Learn the mapping
        router.learn("stop charging me", "cancel_sub");

        // After learning: should route correctly
        let after = router.route("stop charging me");
        assert!(!after.is_empty());
        assert_eq!(after[0].id, "cancel_sub");

        if let Some(cb) = cancel_before {
            assert!(after[0].score > cb.score);
        }
    }

    #[test]
    fn correction_moves_signal() {
        let mut router = Router::new();
        router.add_intent("cancel", &["cancel order"]);
        router.add_intent("refund", &["get refund"]);

        router.learn("I want my money back", "cancel");
        router.correct("I want my money back", "cancel", "refund");

        let result = router.route("I want my money back");
        assert_eq!(result[0].id, "refund");
    }

    #[test]
    fn route_best_with_threshold() {
        let mut router = Router::new();
        router.add_intent("greet", &["hello", "hi there"]);

        assert!(router.route_best("hello", 0.1).is_some());
        assert!(router.route_best("quantum physics", 0.1).is_none());
    }

    #[test]
    fn remove_intent() {
        let mut router = Router::new();
        router.add_intent("a", &["cancel order"]);
        router.add_intent("b", &["track order"]);

        router.remove_intent("a");
        assert_eq!(router.intent_count(), 1);

        let result = router.route("cancel");
        assert!(result.is_empty() || result[0].id != "a");
    }

    #[test]
    fn export_import_roundtrip() {
        let mut router = Router::new();
        router.add_intent("cancel", &["cancel my order", "stop order"]);
        router.learn("drop my order", "cancel");

        let json = router.export_json();
        let restored = Router::import_json(&json).unwrap();

        let result = restored.route("cancel my order");
        assert_eq!(result[0].id, "cancel");

        let result = restored.route("drop my order");
        assert!(!result.is_empty());
        assert_eq!(result[0].id, "cancel");
    }

    #[test]
    fn empty_router_returns_empty() {
        let router = Router::new();
        assert!(router.route("anything").is_empty());
    }

    #[test]
    fn all_stop_words_returns_empty() {
        let mut router = Router::new();
        router.add_intent("a", &["cancel"]);
        assert!(router.route("can you please do this").is_empty());
    }

    #[test]
    fn learn_creates_new_intent() {
        let mut router = Router::new();
        router.learn("reset password", "password_reset");
        assert_eq!(router.intent_count(), 1);

        let result = router.route("reset password");
        assert_eq!(result[0].id, "password_reset");
    }

    #[test]
    fn many_intents_still_fast() {
        let mut router = Router::new();
        for i in 0..500 {
            router.add_intent(
                &format!("intent_{}", i),
                &[&format!("action_{} thing_{}", i, i)],
            );
        }

        let result = router.route("action_42 thing_42");
        assert!(!result.is_empty());
        assert_eq!(result[0].id, "intent_42");
    }
}
