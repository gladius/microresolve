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
//! - You need deep semantic multilingual matching (CJK supported via Aho-Corasick, but coverage depends on seed quality)

pub mod index;
pub mod multi;
pub mod seed;
pub mod tokenizer;
pub mod vector;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use multi::{IntentRelation, MultiRouteOutput, MultiRouteResult};

/// The type of an intent — whether it represents a user action or supporting context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntentType {
    /// User explicitly wants this done (e.g. cancel_order, refund).
    Action,
    /// Supporting data for fulfillment (e.g. check_balance, get_user_profile).
    Context,
}

use aho_corasick::AhoCorasick;
use index::InvertedIndex;
use std::collections::{HashMap, HashSet};
use tokenizer::{
    is_cjk, tokenize, training_to_terms, split_script_runs, generate_cjk_residual_bigrams,
    find_cjk_negated_regions, is_learnable_cjk_bigram, PositionedTerm, ScriptType,
};
use vector::LearnedVector;

/// Intent router with incremental learning.
///
/// The main entry point for the library. Manages intents, routing, and learning.
/// Supports both Latin and CJK scripts via a dual-path tokenization architecture:
/// Latin text uses whitespace tokenization; CJK text uses Aho-Corasick automaton
/// matching with character bigram fallback for novel terms.
pub struct Router {
    vectors: HashMap<String, LearnedVector>,
    index: InvertedIndex,
    /// Raw training phrases per intent, grouped by language code.
    /// Structure: { intent_id: { lang_code: [phrases] } }
    training: HashMap<String, HashMap<String, Vec<String>>>,
    top_k: usize,
    /// Aho-Corasick automaton for CJK term matching. None if no CJK terms exist.
    cjk_automaton: Option<AhoCorasick>,
    /// Pattern strings for the automaton. cjk_patterns[pattern_id] = term string.
    cjk_patterns: Vec<String>,
    /// When true, defers automaton rebuilds until `end_batch()` is called.
    batch_mode: bool,
    /// Tracks whether the automaton needs rebuilding (dirty during batch mode).
    cjk_dirty: bool,
    /// Intent type per intent (Action or Context). Default: Action.
    intent_types: HashMap<String, IntentType>,
    /// Opaque metadata per intent. User-defined key-value pairs.
    /// ASV stores and returns this data but never interprets it.
    metadata: HashMap<String, HashMap<String, Vec<String>>>,
    /// Co-occurrence counts: how often intent pairs fire together in route_multi.
    /// Key: (intent_a, intent_b) where a < b lexicographically. Value: count.
    co_occurrence: HashMap<(String, String), u32>,
    /// Paraphrase index: phrase (lowercase) -> (intent_id, weight).
    /// Multi-word phrase matching via Aho-Corasick automaton for dual-source confidence.
    paraphrase_phrases: HashMap<String, (String, f32)>,
    /// Aho-Corasick automaton for paraphrase matching.
    paraphrase_automaton: Option<AhoCorasick>,
    /// Pattern strings for paraphrase automaton.
    paraphrase_patterns: Vec<String>,
    /// Tracks whether paraphrase automaton needs rebuild (dirty during batch mode).
    paraphrase_dirty: bool,
    /// Monotonic version counter. Incremented on every mutation (learn, correct, add_intent, merge).
    version: u64,
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
            cjk_automaton: None,
            cjk_patterns: Vec::new(),
            batch_mode: false,
            cjk_dirty: false,
            intent_types: HashMap::new(),
            metadata: HashMap::new(),
            co_occurrence: HashMap::new(),
            paraphrase_phrases: HashMap::new(),
            paraphrase_automaton: None,
            paraphrase_patterns: Vec::new(),
            paraphrase_dirty: false,
            version: 0,
        }
    }

    /// Set the maximum number of results returned by `route()`.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Begin batch mode: defers CJK automaton rebuilds until `end_batch()`.
    ///
    /// Use this when calling `learn()` or `correct()` many times in sequence.
    /// The inverted index is still updated incrementally per call, so routing
    /// remains functional. Only the CJK automaton rebuild is deferred.
    ///
    /// ```
    /// use asv_router::Router;
    ///
    /// let mut router = Router::new();
    /// router.add_intent("cancel", &["取消 订单"]);
    ///
    /// router.begin_batch();
    /// for i in 0..100 {
    ///     router.learn(&format!("query {}", i), "cancel");
    /// }
    /// router.end_batch(); // single automaton rebuild
    /// ```
    pub fn begin_batch(&mut self) {
        self.batch_mode = true;
        self.cjk_dirty = false;
        self.paraphrase_dirty = false;
    }

    /// End batch mode and rebuild automatons if needed.
    pub fn end_batch(&mut self) {
        self.batch_mode = false;
        if self.cjk_dirty {
            self.rebuild_cjk_automaton_now();
            self.cjk_dirty = false;
        }
        if self.paraphrase_dirty {
            self.rebuild_paraphrase_automaton_now();
            self.paraphrase_dirty = false;
        }
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
        self.version += 1;
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
        self.version += 1;
    }

    /// Add an intent with pre-computed term weights.
    ///
    /// Use this when you have term weights from an external source
    /// (e.g., LLM-generated, imported from another system).
    pub fn add_intent_with_weights(&mut self, id: &str, seed_terms: HashMap<String, f32>) {
        let vector = LearnedVector::from_seed(seed_terms);
        self.vectors.insert(id.to_string(), vector);
        self.rebuild_index();
        self.version += 1;
    }

    /// Remove an intent.
    pub fn remove_intent(&mut self, id: &str) {
        self.vectors.remove(id);
        self.training.remove(id);
        self.index.remove_intent(id);
        self.intent_types.remove(id);
        self.metadata.remove(id);

        // Remove paraphrase phrases pointing to this intent
        self.paraphrase_phrases.retain(|_, (intent, _)| intent != id);
        self.rebuild_paraphrase_automaton();
        self.version += 1;
    }

    /// Route a query to matching intents, ranked by score.
    ///
    /// Returns up to `top_k` results (default 10), sorted by score descending.
    /// Empty results means no intent matched any query terms.
    /// Supports both Latin and CJK scripts via dual-path extraction.
    pub fn route(&self, query: &str) -> Vec<RouteResult> {
        let terms = self.extract_terms(query);
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
    /// For CJK queries, only learns clean automaton matches and filtered residual bigrams.
    pub fn learn(&mut self, query: &str, intent_id: &str) {
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
        let mut output = multi::route_multi(&self.index, &self.vectors, positioned, query_chars, threshold);

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
            }
            // else: routing-only → stays "low" confidence
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
        output
    }

    /// Export router state as JSON for persistence.
    pub fn export_json(&self) -> String {
        let paraphrases: Vec<(String, String, f32)> = self.paraphrase_phrases.iter()
            .map(|(phrase, (intent_id, weight))| (phrase.clone(), intent_id.clone(), *weight))
            .collect();
        let co_occurrence: Vec<(String, String, u32)> = self.co_occurrence.iter()
            .map(|((a, b), &count)| (a.clone(), b.clone(), count))
            .collect();
        let state = RouterState {
            intents: self.vectors.clone(),
            training: self.training.clone(),
            top_k: self.top_k,
            intent_types: self.intent_types.clone(),
            metadata: self.metadata.clone(),
            paraphrases,
            co_occurrence,
            version: self.version,
        };
        serde_json::to_string(&state).unwrap_or_default()
    }

    /// Import router state from JSON.
    pub fn import_json(json: &str) -> Result<Self, String> {
        let state: RouterState =
            serde_json::from_str(json).map_err(|e| format!("invalid JSON: {}", e))?;

        let index = InvertedIndex::build(&state.intents);

        // Restore paraphrase phrases from serialized Vec<(phrase, intent_id, weight)>
        let mut paraphrase_phrases: HashMap<String, (String, f32)> = HashMap::new();
        for (phrase, intent_id, weight) in &state.paraphrases {
            paraphrase_phrases.insert(phrase.clone(), (intent_id.clone(), *weight));
        }

        // Restore co-occurrence from serialized Vec
        let mut co_occurrence_map: HashMap<(String, String), u32> = HashMap::new();
        for (a, b, count) in state.co_occurrence {
            co_occurrence_map.insert((a, b), count);
        }

        let mut router = Self {
            vectors: state.intents,
            index,
            training: state.training,
            top_k: state.top_k,
            cjk_automaton: None,
            cjk_patterns: Vec::new(),
            batch_mode: false,
            cjk_dirty: false,
            intent_types: state.intent_types,
            metadata: state.metadata,
            co_occurrence: co_occurrence_map,
            paraphrase_phrases,
            paraphrase_automaton: None,
            paraphrase_patterns: Vec::new(),
            paraphrase_dirty: false,
            version: state.version,
        };
        router.rebuild_cjk_automaton_now();
        router.rebuild_paraphrase_automaton_now();
        Ok(router)
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

    /// Set the type of an intent (Action or Context).
    ///
    /// ```
    /// use asv_router::{Router, IntentType};
    ///
    /// let mut router = Router::new();
    /// router.add_intent("check_balance", &["check my balance", "account balance"]);
    /// router.set_intent_type("check_balance", IntentType::Context);
    /// assert_eq!(router.get_intent_type("check_balance"), IntentType::Context);
    /// ```
    pub fn set_intent_type(&mut self, intent_id: &str, intent_type: IntentType) {
        self.intent_types.insert(intent_id.to_string(), intent_type);
    }

    /// Get the type of an intent. Defaults to Action if not set.
    pub fn get_intent_type(&self, intent_id: &str) -> IntentType {
        self.intent_types.get(intent_id).copied().unwrap_or(IntentType::Action)
    }

    /// Set opaque metadata for an intent.
    ///
    /// ASV stores and returns this data but never interprets it.
    /// The application layer decides what to do with it.
    ///
    /// ```
    /// use asv_router::Router;
    ///
    /// let mut router = Router::new();
    /// router.add_intent("cancel_order", &["cancel my order"]);
    /// router.set_metadata("cancel_order", "context_intents", vec!["check_balance".into(), "track_order".into()]);
    /// ```
    pub fn set_metadata(&mut self, intent_id: &str, key: &str, values: Vec<String>) {
        self.metadata
            .entry(intent_id.to_string())
            .or_default()
            .insert(key.to_string(), values);
    }

    /// Get all metadata for an intent.
    pub fn get_metadata(&self, intent_id: &str) -> Option<&HashMap<String, Vec<String>>> {
        self.metadata.get(intent_id)
    }

    /// Get a specific metadata key for an intent.
    pub fn get_metadata_key(&self, intent_id: &str, key: &str) -> Option<&Vec<String>> {
        self.metadata.get(intent_id)?.get(key)
    }

    /// Record co-occurrence for a set of intents detected together.
    /// Call after route_multi to track which intents fire together.
    pub fn record_co_occurrence(&mut self, intent_ids: &[&str]) {
        for i in 0..intent_ids.len() {
            for j in (i + 1)..intent_ids.len() {
                let (a, b) = if intent_ids[i] < intent_ids[j] {
                    (intent_ids[i].to_string(), intent_ids[j].to_string())
                } else {
                    (intent_ids[j].to_string(), intent_ids[i].to_string())
                };
                *self.co_occurrence.entry((a, b)).or_insert(0) += 1;
            }
        }
    }

    /// Get co-occurrence data as a list of (intent_a, intent_b, count) sorted by count desc.
    pub fn get_co_occurrence(&self) -> Vec<(&str, &str, u32)> {
        let mut pairs: Vec<(&str, &str, u32)> = self.co_occurrence
            .iter()
            .map(|((a, b), &count)| (a.as_str(), b.as_str(), count))
            .collect();
        pairs.sort_by(|a, b| b.2.cmp(&a.2));
        pairs
    }

    /// Clear co-occurrence data.
    pub fn clear_co_occurrence(&mut self) {
        self.co_occurrence.clear();
    }

    /// Get the current version number. Incremented on every mutation.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Merge learned weights from another router into this one.
    ///
    /// Uses max() per term per intent — this is a CRDT merge:
    /// commutative, associative, idempotent, conflict-free.
    ///
    /// - Seed weights are never modified (immutable layer preserved)
    /// - Only learned weights are combined
    /// - New intents in `other` that don't exist here are ignored
    ///   (they have no seed layer to anchor them)
    /// - Co-occurrence counts are summed
    /// - Paraphrase phrases are merged (other's phrases added if not present)
    ///
    /// After merge, the inverted index is rebuilt to reflect new weights.
    pub fn merge_learned(&mut self, other: &Router) {
        let mut changed = false;

        // Merge learned weights per intent
        for (intent_id, other_vector) in &other.vectors {
            if let Some(self_vector) = self.vectors.get_mut(intent_id) {
                if other_vector.has_learned() {
                    self_vector.merge_learned(other_vector);
                    changed = true;
                }
            }
            // Intents only in `other` are skipped — no seed layer here to anchor them
        }

        // Merge co-occurrence (additive)
        for ((a, b), &count) in &other.co_occurrence {
            *self.co_occurrence.entry((a.clone(), b.clone())).or_insert(0) += count;
        }

        // Merge paraphrase phrases (keep existing if conflict, add new)
        for (phrase, (intent_id, weight)) in &other.paraphrase_phrases {
            self.paraphrase_phrases
                .entry(phrase.clone())
                .or_insert_with(|| (intent_id.clone(), *weight));
        }

        // Merge training phrases (union)
        for (intent_id, other_lang_map) in &other.training {
            let self_lang_map = self.training.entry(intent_id.clone()).or_default();
            for (lang, other_phrases) in other_lang_map {
                let self_phrases = self_lang_map.entry(lang.clone()).or_default();
                let existing: HashSet<String> = self_phrases.iter().cloned().collect();
                for phrase in other_phrases {
                    if !existing.contains(phrase) {
                        self_phrases.push(phrase.clone());
                    }
                }
            }
        }

        if changed {
            self.rebuild_index();
            self.rebuild_paraphrase_automaton_now();
            self.version += 1;
        }
    }

    /// Export only the learned layer weights for lightweight sync.
    ///
    /// Returns a JSON object: { intent_id: { term: weight, ... }, ... }
    /// Only includes intents that have learned weights.
    /// Much smaller than full export — just the delta from seed state.
    pub fn export_learned_only(&self) -> String {
        let learned: HashMap<&str, &HashMap<String, f32>> = self.vectors.iter()
            .filter(|(_, v)| v.has_learned())
            .map(|(id, v)| (id.as_str(), v.learned_terms()))
            .collect();
        serde_json::to_string(&learned).unwrap_or_default()
    }

    /// Import and merge learned weights from a lightweight sync payload.
    ///
    /// Input format: { intent_id: { term: weight, ... }, ... }
    /// Uses max() merge — safe to call multiple times with same data (idempotent).
    pub fn import_learned_merge(&mut self, json: &str) -> Result<(), String> {
        let learned: HashMap<String, HashMap<String, f32>> =
            serde_json::from_str(json).map_err(|e| format!("invalid JSON: {}", e))?;

        let mut changed = false;
        for (intent_id, other_terms) in &learned {
            if let Some(vector) = self.vectors.get_mut(intent_id) {
                let other_vec = LearnedVector::from_parts(HashMap::new(), other_terms.clone());
                vector.merge_learned(&other_vec);
                changed = true;
            }
        }

        if changed {
            self.rebuild_index();
            self.version += 1;
        }
        Ok(())
    }

    fn rebuild_index(&mut self) {
        self.index = InvertedIndex::build(&self.vectors);
        // Full index rebuild always rebuilds automaton immediately (not deferred)
        self.rebuild_cjk_automaton_now();
    }

    /// Request a CJK automaton rebuild. Deferred if in batch mode.
    fn rebuild_cjk_automaton(&mut self) {
        if self.batch_mode {
            self.cjk_dirty = true;
        } else {
            self.rebuild_cjk_automaton_now();
        }
    }

    /// Unconditionally rebuild the Aho-Corasick automaton from CJK terms in the index.
    fn rebuild_cjk_automaton_now(&mut self) {
        let cjk_terms: Vec<String> = self.index.terms()
            .filter(|t| t.chars().any(is_cjk))
            .cloned()
            .collect();

        if cjk_terms.is_empty() {
            self.cjk_automaton = None;
            self.cjk_patterns = Vec::new();
            return;
        }

        self.cjk_automaton = Some(
            AhoCorasick::builder()
                .match_kind(aho_corasick::MatchKind::Standard)
                .build(&cjk_terms)
                .expect("failed to build CJK automaton")
        );
        self.cjk_patterns = cjk_terms;
    }

    // ===== Paraphrase Index =====

    /// Add paraphrase phrases for an intent.
    ///
    /// Paraphrases are multi-word expressions scanned via Aho-Corasick automaton.
    /// When both the routing index and paraphrase index detect the same intent,
    /// the detection is tagged as "dual-source" with "high" confidence.
    ///
    /// ```
    /// use asv_router::Router;
    ///
    /// let mut router = Router::new();
    /// router.add_intent("refund", &["I want a refund", "money back"]);
    /// router.add_paraphrases("refund", &[
    ///     "get my money back",
    ///     "return for a full refund",
    ///     "I need a refund please",
    /// ]);
    /// ```
    pub fn add_paraphrases(&mut self, intent_id: &str, phrases: &[&str]) {
        for phrase in phrases {
            let lower = phrase.to_lowercase();
            if lower.split_whitespace().count() >= 2 && lower.len() >= 5 {
                self.paraphrase_phrases.insert(lower, (intent_id.to_string(), 0.8));
            }
        }
        self.rebuild_paraphrase_automaton();
    }

    /// Add paraphrases from a map of intent_id -> phrases (for bulk loading).
    pub fn add_paraphrases_bulk(&mut self, data: &HashMap<String, Vec<String>>) {
        for (intent_id, phrases) in data {
            for phrase in phrases {
                let lower = phrase.to_lowercase();
                if lower.split_whitespace().count() >= 2 && lower.len() >= 5 {
                    self.paraphrase_phrases.insert(lower, (intent_id.clone(), 0.8));
                }
            }
        }
        self.rebuild_paraphrase_automaton();
    }

    /// Scan a message against the paraphrase automaton.
    /// Returns: Vec of (intent_id, weight, match_start_position).
    fn paraphrase_scan(&self, message: &str) -> Vec<(String, f32, usize)> {
        let lower = message.to_lowercase();
        let mut results: Vec<(String, f32, usize)> = Vec::new();
        let mut seen_intents: HashSet<String> = HashSet::new();

        if let Some(ref ac) = self.paraphrase_automaton {
            for mat in ac.find_iter(&lower) {
                let pattern = &self.paraphrase_patterns[mat.pattern().as_usize()];
                if let Some((intent_id, weight)) = self.paraphrase_phrases.get(pattern) {
                    if seen_intents.insert(intent_id.clone()) {
                        results.push((intent_id.clone(), *weight, mat.start()));
                    }
                }
            }
        }
        results
    }

    /// Learn paraphrase n-grams from a message for an intent.
    /// Extracts all overlapping 3-5 word windows as paraphrase phrases.
    /// Multi-word phrases are inherently discriminative so no filtering is needed.
    /// Matches clean experiment's extraction: min 2 words + 5 chars, overwrites allowed.
    fn learn_paraphrases(&mut self, message: &str, intent_id: &str) {
        let lower = message.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        for window_size in 3..=5 {
            if words.len() >= window_size {
                for start in 0..=(words.len() - window_size) {
                    let phrase: String = words[start..start + window_size].join(" ");
                    if phrase.split_whitespace().count() >= 2 && phrase.len() >= 5 {
                        self.paraphrase_phrases.insert(phrase, (intent_id.to_string(), 0.5));
                    }
                }
            }
        }

        // Also add the full message if it's short enough
        if words.len() >= 3 && words.len() <= 12 {
            self.paraphrase_phrases.insert(lower, (intent_id.to_string(), 0.6));
        }

        self.rebuild_paraphrase_automaton();
    }

    /// Request a paraphrase automaton rebuild. Deferred if in batch mode.
    fn rebuild_paraphrase_automaton(&mut self) {
        if self.batch_mode {
            self.paraphrase_dirty = true;
        } else {
            self.rebuild_paraphrase_automaton_now();
        }
    }

    /// Unconditionally rebuild the paraphrase Aho-Corasick automaton.
    fn rebuild_paraphrase_automaton_now(&mut self) {
        self.paraphrase_patterns = self.paraphrase_phrases.keys().cloned().collect();
        if self.paraphrase_patterns.is_empty() {
            self.paraphrase_automaton = None;
            return;
        }
        // Sort by length descending for leftmost-longest matching
        self.paraphrase_patterns.sort_by(|a, b| b.len().cmp(&a.len()));
        self.paraphrase_automaton = AhoCorasick::builder()
            .match_kind(aho_corasick::MatchKind::LeftmostLongest)
            .build(&self.paraphrase_patterns)
            .ok();
    }

    /// Get paraphrase count (for diagnostics).
    pub fn paraphrase_count(&self) -> usize {
        self.paraphrase_phrases.len()
    }

    /// Extract terms from a query using dual-path (Latin tokenizer + CJK automaton).
    fn extract_terms(&self, query: &str) -> Vec<String> {
        if !query.chars().any(is_cjk) {
            return tokenize(query);
        }

        let lower = query.to_lowercase();
        let runs = split_script_runs(&lower);
        let mut all_terms = Vec::new();
        let mut seen = HashSet::new();

        for run in &runs {
            match run.script {
                ScriptType::Latin => {
                    for term in tokenize(&run.text) {
                        if seen.insert(term.clone()) {
                            all_terms.push(term);
                        }
                    }
                }
                ScriptType::Cjk => {
                    let terms = self.extract_cjk_run_terms(&run.text, false);
                    for term in terms {
                        if seen.insert(term.clone()) {
                            all_terms.push(term);
                        }
                    }
                }
            }
        }

        all_terms
    }

    /// Extract terms for learning — more selective for CJK to prevent noise pollution.
    fn extract_terms_for_learning(&self, query: &str) -> Vec<String> {
        if !query.chars().any(is_cjk) {
            return tokenize(query);
        }

        let lower = query.to_lowercase();
        let runs = split_script_runs(&lower);
        let mut all_terms = Vec::new();
        let mut seen = HashSet::new();

        for run in &runs {
            match run.script {
                ScriptType::Latin => {
                    for term in tokenize(&run.text) {
                        if seen.insert(term.clone()) {
                            all_terms.push(term);
                        }
                    }
                }
                ScriptType::Cjk => {
                    let terms = self.extract_cjk_run_terms(&run.text, true);
                    for term in terms {
                        if seen.insert(term.clone()) {
                            all_terms.push(term);
                        }
                    }
                }
            }
        }

        all_terms
    }

    /// Extract terms from a CJK text run.
    ///
    /// 1. Detect negation marker positions
    /// 2. Scan automaton on original text (overlapping matches)
    /// 3. Find unmatched residual regions
    /// 4. Generate bigrams from cleaned residuals
    ///
    /// If `for_learning` is true, only include residual bigrams that pass the noise filter.
    fn extract_cjk_run_terms(&self, cjk_text: &str, for_learning: bool) -> Vec<String> {
        let negated_regions = find_cjk_negated_regions(cjk_text);
        let chars: Vec<char> = cjk_text.chars().collect();
        let mut matched_terms = Vec::new();
        let mut covered: HashSet<usize> = HashSet::new();

        // Step 1: Automaton scan (if available)
        if let Some(ref automaton) = self.cjk_automaton {
            for mat in automaton.find_overlapping_iter(cjk_text) {
                let pattern_idx = mat.pattern().as_usize();
                let term = &self.cjk_patterns[pattern_idx];

                // Convert byte offset to char offset
                let start_char = cjk_text[..mat.start()].chars().count();
                let end_char = cjk_text[..mat.end()].chars().count();

                // Check if this match falls in a negated region — prefix instead of skip
                let is_neg = negated_regions.iter().any(|(ns, ne)| start_char >= *ns && start_char < *ne);
                if is_neg {
                    matched_terms.push(format!("not_{}", term));
                } else {
                    matched_terms.push(term.clone());
                }
                for i in start_char..end_char {
                    covered.insert(i);
                }
            }
        }

        // Step 2: Find unmatched residual regions
        let mut residual_runs: Vec<String> = Vec::new();
        let mut current_run = String::new();

        for (i, &c) in chars.iter().enumerate() {
            if !covered.contains(&i) && is_cjk(c) {
                current_run.push(c);
            } else if !current_run.is_empty() {
                residual_runs.push(std::mem::take(&mut current_run));
            }
        }
        if !current_run.is_empty() {
            residual_runs.push(current_run);
        }

        // Step 3: Generate bigrams from residuals (with stop char filtering)
        for residual in &residual_runs {
            let bigrams = generate_cjk_residual_bigrams(residual);
            for bg in bigrams {
                // For learning, apply stricter filter
                if for_learning && !is_learnable_cjk_bigram(&bg) {
                    continue;
                }

                // Check negation for residual bigrams — prefix instead of skip
                let is_neg = if let Some(pos) = cjk_text.find(&bg) {
                    let char_pos = cjk_text[..pos].chars().count();
                    negated_regions.iter().any(|(ns, ne)| char_pos >= *ns && char_pos < *ne)
                } else {
                    false
                };

                if is_neg {
                    matched_terms.push(format!("not_{}", bg));
                } else {
                    matched_terms.push(bg);
                }
            }
        }

        matched_terms
    }

    /// Extract positioned terms for multi-intent decomposition.
    ///
    /// Returns positioned terms with character offsets and the processed query as chars.
    fn extract_terms_positioned(&self, query: &str) -> (Vec<PositionedTerm>, Vec<char>) {
        let lower = query.to_lowercase();

        if !lower.chars().any(is_cjk) {
            // Fast path: Latin only
            return tokenizer::tokenize_positioned(&lower);
        }

        // Dual path: expand contractions, split into script runs
        let expanded = tokenizer::expand_contractions_public(&lower);
        let full_chars: Vec<char> = expanded.chars().collect();
        let runs = split_script_runs(&expanded);

        let mut all_positioned = Vec::new();

        for run in &runs {
            match run.script {
                ScriptType::Latin => {
                    // Tokenize the Latin run and adjust offsets
                    let (terms, _) = tokenizer::tokenize_positioned(&run.text);
                    for mut pt in terms {
                        pt.offset += run.char_offset;
                        pt.end_offset += run.char_offset;
                        all_positioned.push(pt);
                    }
                }
                ScriptType::Cjk => {
                    let cjk_terms = self.extract_cjk_run_positioned(&run.text, run.char_offset);
                    all_positioned.extend(cjk_terms);
                }
            }
        }

        (all_positioned, full_chars)
    }

    /// Extract positioned CJK terms from a CJK text run using the automaton.
    fn extract_cjk_run_positioned(&self, cjk_text: &str, base_offset: usize) -> Vec<PositionedTerm> {
        let negated_regions = find_cjk_negated_regions(cjk_text);
        let chars: Vec<char> = cjk_text.chars().collect();

        let mut positioned = Vec::new();
        let mut covered: HashSet<usize> = HashSet::new();

        // Automaton scan
        if let Some(ref automaton) = self.cjk_automaton {
            for mat in automaton.find_overlapping_iter(cjk_text) {
                let pattern_idx = mat.pattern().as_usize();
                let term = &self.cjk_patterns[pattern_idx];

                let start_char = cjk_text[..mat.start()].chars().count();
                let end_char = cjk_text[..mat.end()].chars().count();

                let is_neg = negated_regions.iter().any(|(ns, ne)| start_char >= *ns && start_char < *ne);
                let final_term = if is_neg {
                    format!("not_{}", term)
                } else {
                    term.clone()
                };

                positioned.push(PositionedTerm {
                    term: final_term,
                    offset: base_offset + start_char,
                    end_offset: base_offset + end_char,
                    is_cjk: true,
                });

                for i in start_char..end_char {
                    covered.insert(i);
                }
            }
        }

        // Residual bigrams
        let mut current_run_start = None;
        let mut current_run = String::new();

        for (i, &c) in chars.iter().enumerate() {
            if !covered.contains(&i) && is_cjk(c) {
                if current_run_start.is_none() {
                    current_run_start = Some(i);
                }
                current_run.push(c);
            } else if !current_run.is_empty() {
                let run_start = current_run_start.take().unwrap();
                let bigrams = generate_cjk_residual_bigrams(&current_run);
                let mut bi = 0;
                for bg in bigrams {
                    positioned.push(PositionedTerm {
                        term: bg,
                        offset: base_offset + run_start + bi,
                        end_offset: base_offset + run_start + bi + 2,
                        is_cjk: true,
                    });
                    bi += 1;
                }
                current_run.clear();
            }
        }
        if !current_run.is_empty() {
            let run_start = current_run_start.unwrap();
            let bigrams = generate_cjk_residual_bigrams(&current_run);
            let mut bi = 0;
            for bg in bigrams {
                positioned.push(PositionedTerm {
                    term: bg,
                    offset: base_offset + run_start + bi,
                    end_offset: base_offset + run_start + bi + 2,
                    is_cjk: true,
                });
                bi += 1;
            }
        }

        positioned
    }

    // ===== Experimental methods for scenario testing =====

    /// Document frequency of a term across all intents.
    pub fn term_df(&self, term: &str) -> usize {
        self.index.df(term)
    }

    /// Analyze query terms: returns (term, idf, df) for each content term.
    pub fn analyze_query_terms(&self, query: &str) -> Vec<(String, f32, usize)> {
        let terms = tokenize(&query.to_lowercase());
        let n = self.index.intent_count().max(1) as f32;
        terms.into_iter().map(|t| {
            let df = self.index.df(&t);
            let idf = if df > 0 { 1.0 + 0.5 * (n / df as f32).ln() } else { 0.0 };
            (t, idf, df)
        }).collect()
    }

    /// Route multi-intent with noise gate: exclude terms appearing in > max_df intents.
    pub fn route_multi_noise_gated(&self, query: &str, threshold: f32, max_df: usize) -> MultiRouteOutput {
        let (positioned, query_chars) = self.extract_terms_positioned(query);
        let filtered: Vec<PositionedTerm> = positioned.into_iter()
            .filter(|pt| self.index.df(&pt.term) <= max_df)
            .collect();

        let mut output = multi::route_multi(&self.index, &self.vectors, filtered, query_chars, threshold);
        for intent in &mut output.intents {
            intent.intent_type = self.get_intent_type(&intent.id);
        }
        output
    }

    /// Route multi-intent with anchor-based scoring.
    /// Only detects intents that have an anchor term (high discrimination) in the query.
    /// Scores using a local window of terms around each anchor.
    pub fn route_multi_anchored(&self, query: &str, threshold: f32, window: usize) -> MultiRouteOutput {
        let (positioned, query_chars) = self.extract_terms_positioned(query);
        if positioned.is_empty() {
            return MultiRouteOutput { intents: vec![], relations: vec![], metadata: HashMap::new() };
        }

        let n = self.index.intent_count();
        let disc_max_df = (n / 15).max(3);

        // Build reverse map: term -> intents it can anchor
        let mut term_to_intents: HashMap<&str, Vec<&str>> = HashMap::new();
        for (intent_id, vector) in &self.vectors {
            for (term, weight) in vector.effective_terms() {
                if self.index.df(&term) <= disc_max_df && weight >= 0.5 {
                    term_to_intents.entry(
                        // Leak string to get &str with right lifetime — only for experiments
                        // In production this would use a proper data structure
                        Box::leak(term.into_boxed_str()) as &str
                    ).or_default().push(
                        Box::leak(intent_id.clone().into_boxed_str()) as &str
                    );
                }
            }
        }

        // Find anchor matches in query positioned terms
        let mut anchored: HashMap<String, Vec<usize>> = HashMap::new(); // intent -> [term indices]
        for (idx, pt) in positioned.iter().enumerate() {
            if let Some(intents) = term_to_intents.get(pt.term.as_str()) {
                for &intent in intents {
                    anchored.entry(intent.to_string()).or_default().push(idx);
                }
            }
        }

        // For each anchored intent, score in local window around anchor
        let mut results: Vec<MultiRouteResult> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for (intent_id, anchor_positions) in &anchored {
            if seen.contains(intent_id) { continue; }

            let mut best_score = 0.0f32;
            let mut best_anchor_idx = 0usize;

            for &anchor_idx in anchor_positions {
                let start = anchor_idx.saturating_sub(window);
                let end = (anchor_idx + window + 1).min(positioned.len());
                let window_terms: Vec<String> = positioned[start..end].iter()
                    .map(|pt| pt.term.clone())
                    .collect();

                let search_results = self.index.search(&window_terms, 10);
                if let Some(sr) = search_results.iter().find(|r| r.id == *intent_id) {
                    if sr.score > best_score {
                        best_score = sr.score;
                        best_anchor_idx = anchor_idx;
                    }
                }
            }

            if best_score >= threshold {
                seen.insert(intent_id.clone());
                let start = best_anchor_idx.saturating_sub(window);
                let end = (best_anchor_idx + window + 1).min(positioned.len());
                let min_off = positioned[start..end].iter().map(|p| p.offset).min().unwrap_or(0);
                let max_off = positioned[start..end].iter().map(|p| p.end_offset).max().unwrap_or(0);

                results.push(MultiRouteResult {
                    id: intent_id.clone(),
                    score: best_score,
                    position: positioned[best_anchor_idx].offset,
                    span: (min_off, max_off),
                    intent_type: self.get_intent_type(intent_id),
                    confidence: "low".to_string(),
                    source: "routing".to_string(),
                    negated: false,
                });
            }
        }

        results.sort_by_key(|r| r.position);
        let relations = multi::detect_relations_public(&results, &query_chars);

        MultiRouteOutput { intents: results, relations, metadata: HashMap::new() }
    }

    /// Query coverage: (known_terms, total_terms) — fraction of terms in the index.
    pub fn query_coverage(&self, query: &str) -> (usize, usize) {
        let terms = tokenize(&query.to_lowercase());
        let total = terms.len();
        let known = terms.iter().filter(|t| self.index.df(t) > 0).count();
        (known, total)
    }

    /// Search the index directly (for experimental scoring).
    pub fn search_terms(&self, terms: &[String], top_k: usize) -> Vec<RouteResult> {
        self.index.search(terms, top_k).iter().map(|si| RouteResult {
            id: si.id.clone(),
            score: si.score,
        }).collect()
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
    /// Intent types (Action or Context).
    #[serde(default)]
    intent_types: HashMap<String, IntentType>,
    /// Opaque metadata per intent.
    #[serde(default)]
    metadata: HashMap<String, HashMap<String, Vec<String>>>,
    /// Paraphrase phrases: Vec<(phrase, intent_id, weight)>.
    #[serde(default)]
    paraphrases: Vec<(String, String, f32)>,
    /// Co-occurrence counts: Vec<(intent_a, intent_b, count)>.
    #[serde(default)]
    co_occurrence: Vec<(String, String, u32)>,
    /// Monotonic version counter.
    #[serde(default)]
    version: u64,
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
        assert!(router.route("the a an in on at to").is_empty());
    }

    #[test]
    fn learn_creates_new_intent() {
        let mut router = Router::new();
        router.learn("reset password", "password_reset");
        assert_eq!(router.intent_count(), 1);

        let result = router.route("reset password");
        assert_eq!(result[0].id, "password_reset");
    }

    // --- CJK routing tests ---

    #[test]
    fn cjk_chinese_basic_routing() {
        let mut router = Router::new();
        // Space-separated seeds (as LLM would provide)
        router.add_intent("cancel_order", &[
            "取消 订单",
            "我 要 取消",
            "退订",
        ]);
        router.add_intent("track_order", &[
            "查看 订单",
            "物流 状态",
            "快递 到 哪里",
        ]);

        // Query: "我想取消我的订单" (I want to cancel my order)
        let result = router.route("我想取消我的订单");
        assert!(!result.is_empty(), "should match CJK query");
        assert_eq!(result[0].id, "cancel_order");
    }

    #[test]
    fn cjk_japanese_basic_routing() {
        let mut router = Router::new();
        router.add_intent("cancel", &[
            "キャンセル",
            "取り消し",
        ]);
        router.add_intent("track", &[
            "追跡",
            "配送 状況",
        ]);

        let result = router.route("キャンセルしたい");
        assert!(!result.is_empty(), "should match Japanese query");
        assert_eq!(result[0].id, "cancel");
    }

    #[test]
    fn cjk_four_char_idiom() {
        let mut router = Router::new();
        // Test 4-character compound term (automaton handles any length)
        router.add_intent("complaint", &[
            "莫名其妙",
            "投诉",
        ]);
        router.add_intent("praise", &[
            "非常满意",
            "好评",
        ]);

        let result = router.route("这个服务莫名其妙");
        assert!(!result.is_empty());
        assert_eq!(result[0].id, "complaint");
    }

    #[test]
    fn cjk_mixed_language_query() {
        let mut router = Router::new();
        router.add_intent("cancel_order", &[
            "cancel my order",
            "取消 订单",
        ]);

        // Mixed: "I want to 取消订单"
        let result = router.route("I want to 取消订单");
        assert!(!result.is_empty());
        assert_eq!(result[0].id, "cancel_order");
    }

    #[test]
    fn cjk_learning() {
        let mut router = Router::new();
        router.add_intent("refund", &[
            "退款",
            "退钱",
        ]);

        // Before learning: "要回我的钱" has no seed match
        let before = router.route("要回我的钱");
        let _had_refund = before.iter().any(|r| r.id == "refund");

        // Learn the phrase
        router.learn("要回我的钱", "refund");

        // After learning: should route to refund
        let after = router.route("要回我的钱");
        assert!(!after.is_empty());
        assert_eq!(after[0].id, "refund");
    }

    #[test]
    fn cjk_negation_routing() {
        let mut router = Router::new();
        router.add_intent("cancel", &["取消", "退订"]);
        router.add_intent("track", &["查看", "追踪"]);

        // "不取消" — negation should suppress 取消
        let result = router.route("不取消");
        // 取消 is negated, so cancel intent should not be top
        let cancel_score = result.iter().find(|r| r.id == "cancel").map(|r| r.score).unwrap_or(0.0);
        // Without the negated term, cancel shouldn't score
        assert_eq!(cancel_score, 0.0, "negated term should not score");
    }

    #[test]
    fn cjk_multi_intent() {
        let mut router = Router::new();
        router.add_intent("cancel_order", &["取消 订单", "退订"]);
        router.add_intent("check_balance", &["查看 余额", "账户 余额"]);

        let result = router.route_multi("取消订单然后查看余额", 0.3);
        assert!(result.intents.len() >= 2, "should detect 2 intents, got {}", result.intents.len());

        let ids: Vec<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        assert!(ids.contains(&"cancel_order"), "missing cancel_order in {:?}", ids);
        assert!(ids.contains(&"check_balance"), "missing check_balance in {:?}", ids);

        // Should detect sequential relation from 然后
        if !result.relations.is_empty() {
            assert!(
                matches!(result.relations[0], IntentRelation::Sequential { .. }),
                "expected Sequential from 然后, got {:?}", result.relations[0]
            );
        }
    }

    #[test]
    fn cjk_unsegmented_seeds() {
        // LLM might generate seeds without spaces — tokenizer must still produce
        // character bigrams so the automaton can find substrings in queries
        let mut router = Router::new();
        router.add_intent("save_recipe", &[
            "保存食谱",         // unsegmented: "save recipe"
            "保存我的食谱",     // unsegmented: "save my recipe"
        ]);

        // Query with those characters embedded in longer text
        let result = router.route("你能帮我保存一下食谱吗");
        assert!(!result.is_empty(), "should match unsegmented CJK seeds");
        assert_eq!(result[0].id, "save_recipe");
    }

    #[test]
    fn cjk_export_import_roundtrip() {
        let mut router = Router::new();
        router.add_intent("cancel", &["取消 订单"]);
        router.learn("退订服务", "cancel");

        let json = router.export_json();
        let restored = Router::import_json(&json).unwrap();

        let result = restored.route("取消订单");
        assert!(!result.is_empty());
        assert_eq!(result[0].id, "cancel");
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

    // --- Prerequisite tests ---

    #[test]
    fn intent_type_default_is_action() {
        let mut router = Router::new();
        router.add_intent("cancel_order", &["cancel my order"]);
        assert_eq!(router.get_intent_type("cancel_order"), IntentType::Action);
    }

    #[test]
    fn intent_type_set_and_get() {
        let mut router = Router::new();
        router.add_intent("check_balance", &["check my balance"]);
        router.set_intent_type("check_balance", IntentType::Context);
        assert_eq!(router.get_intent_type("check_balance"), IntentType::Context);
    }

    #[test]
    fn intent_type_in_route_multi_output() {
        let mut router = Router::new();
        router.add_intent("cancel_order", &["cancel my order", "I want to cancel"]);
        router.add_intent("check_balance", &["check my balance", "account balance"]);
        router.set_intent_type("check_balance", IntentType::Context);

        let result = router.route_multi("cancel my order and check my balance", 0.3);
        assert!(result.intents.len() >= 2);
        let cancel = result.intents.iter().find(|i| i.id == "cancel_order").unwrap();
        let balance = result.intents.iter().find(|i| i.id == "check_balance").unwrap();
        assert_eq!(cancel.intent_type, IntentType::Action);
        assert_eq!(balance.intent_type, IntentType::Context);
    }

    #[test]
    fn metadata_set_and_get() {
        let mut router = Router::new();
        router.add_intent("cancel_order", &["cancel my order"]);
        router.set_metadata("cancel_order", "context_intents", vec!["check_balance".into(), "track_order".into()]);
        router.set_metadata("cancel_order", "action_intents", vec!["refund".into()]);

        let meta = router.get_metadata("cancel_order").unwrap();
        assert_eq!(meta.get("context_intents").unwrap(), &vec!["check_balance".to_string(), "track_order".to_string()]);
        assert_eq!(meta.get("action_intents").unwrap(), &vec!["refund".to_string()]);
    }

    #[test]
    fn metadata_key_lookup() {
        let mut router = Router::new();
        router.add_intent("cancel_order", &["cancel my order"]);
        router.set_metadata("cancel_order", "context_intents", vec!["check_balance".into()]);

        assert_eq!(router.get_metadata_key("cancel_order", "context_intents").unwrap(), &vec!["check_balance".to_string()]);
        assert!(router.get_metadata_key("cancel_order", "nonexistent").is_none());
        assert!(router.get_metadata_key("nonexistent", "context_intents").is_none());
    }

    #[test]
    fn metadata_in_route_multi_output() {
        let mut router = Router::new();
        router.add_intent("cancel_order", &["cancel my order", "I want to cancel"]);
        router.add_intent("track_order", &["where is my package", "track my order"]);
        router.set_metadata("cancel_order", "context_intents", vec!["check_balance".into()]);
        router.set_metadata("track_order", "context_intents", vec!["get_shipping_info".into()]);

        let result = router.route_multi("cancel my order and track my package", 0.3);
        assert!(result.intents.len() >= 2);
        let cancel_meta = result.metadata.get("cancel_order").unwrap();
        assert_eq!(cancel_meta.get("context_intents").unwrap(), &vec!["check_balance".to_string()]);
    }

    #[test]
    fn intent_type_and_metadata_persist_through_export_import() {
        let mut router = Router::new();
        router.add_intent("refund", &["refund my order"]);
        router.set_intent_type("refund", IntentType::Context);
        router.set_metadata("refund", "context_intents", vec!["check_balance".into()]);
        router.set_metadata("refund", "team", vec!["billing".into()]);

        let json = router.export_json();
        let restored = Router::import_json(&json).unwrap();
        assert_eq!(restored.get_intent_type("refund"), IntentType::Context);
        assert_eq!(restored.get_metadata_key("refund", "context_intents").unwrap(), &vec!["check_balance".to_string()]);
        assert_eq!(restored.get_metadata_key("refund", "team").unwrap(), &vec!["billing".to_string()]);
    }

    #[test]
    fn remove_intent_cleans_type_and_metadata() {
        let mut router = Router::new();
        router.add_intent("cancel_order", &["cancel my order"]);
        router.set_intent_type("cancel_order", IntentType::Context);
        router.set_metadata("cancel_order", "team", vec!["ops".into()]);

        router.remove_intent("cancel_order");
        assert_eq!(router.get_intent_type("cancel_order"), IntentType::Action); // default
        assert!(router.get_metadata("cancel_order").is_none());
    }

    #[test]
    fn co_occurrence_tracking() {
        let mut router = Router::new();
        router.record_co_occurrence(&["cancel_order", "refund"]);
        router.record_co_occurrence(&["cancel_order", "refund"]);
        router.record_co_occurrence(&["cancel_order", "track_order"]);

        let pairs = router.get_co_occurrence();
        assert_eq!(pairs[0], ("cancel_order", "refund", 2));
        assert_eq!(pairs[1], ("cancel_order", "track_order", 1));
    }

    #[test]
    fn version_increments_on_mutation() {
        let mut router = Router::new();
        assert_eq!(router.version(), 0);

        router.add_intent("cancel", &["cancel my order"]);
        assert_eq!(router.version(), 1);

        router.learn("stop that", "cancel");
        assert_eq!(router.version(), 2);

        router.add_intent("track", &["track my order"]);
        assert_eq!(router.version(), 3);

        router.remove_intent("track");
        assert_eq!(router.version(), 4);
    }

    #[test]
    fn merge_learned_combines_weights() {
        let mut router_a = Router::new();
        router_a.add_intent("cancel", &["cancel my order"]);
        router_a.add_intent("track", &["track my order"]);
        router_a.learn("stop that purchase", "cancel");

        let mut router_b = Router::new();
        router_b.add_intent("cancel", &["cancel my order"]);
        router_b.add_intent("track", &["track my order"]);
        router_b.learn("where is my stuff", "track");

        // Before merge: router_a doesn't know "stuff", router_b doesn't know "stop"
        let before_a = router_a.route("where is my stuff");
        let before_track_score = before_a.iter().find(|r| r.id == "track").map(|r| r.score).unwrap_or(0.0);

        router_a.merge_learned(&router_b);

        // After merge: router_a should know "stuff" from router_b's learning
        let after_a = router_a.route("where is my stuff");
        let after_track_score = after_a.iter().find(|r| r.id == "track").map(|r| r.score).unwrap_or(0.0);
        assert!(after_track_score > before_track_score,
            "merge should improve track score for 'stuff': before={}, after={}", before_track_score, after_track_score);
    }

    #[test]
    fn merge_is_idempotent() {
        let mut router_a = Router::new();
        router_a.add_intent("cancel", &["cancel my order"]);
        router_a.learn("stop it", "cancel");

        let mut router_b = Router::new();
        router_b.add_intent("cancel", &["cancel my order"]);
        router_b.learn("halt order", "cancel");

        router_a.merge_learned(&router_b);
        let score_after_first = router_a.route("halt order")[0].score;

        router_a.merge_learned(&router_b); // same merge again
        let score_after_second = router_a.route("halt order")[0].score;

        assert!((score_after_first - score_after_second).abs() < 0.001,
            "merge should be idempotent: first={}, second={}", score_after_first, score_after_second);
    }

    #[test]
    fn export_learned_only_is_lightweight() {
        let mut router = Router::new();
        router.add_intent("cancel", &["cancel my order"]);
        router.add_intent("track", &["track my order"]);
        router.learn("stop it", "cancel");

        let learned_json = router.export_learned_only();
        let parsed: HashMap<String, HashMap<String, f32>> =
            serde_json::from_str(&learned_json).unwrap();

        // Only "cancel" should appear (it has learned terms), not "track"
        assert!(parsed.contains_key("cancel"));
        assert!(!parsed.contains_key("track"));
    }

    #[test]
    fn import_learned_merge_roundtrip() {
        let mut router_a = Router::new();
        router_a.add_intent("cancel", &["cancel my order"]);
        router_a.learn("stop it", "cancel");

        let learned_json = router_a.export_learned_only();

        let mut router_b = Router::new();
        router_b.add_intent("cancel", &["cancel my order"]);

        // Before import: router_b doesn't know "stop"
        let before = router_b.route("stop it");
        let before_score = before.iter().find(|r| r.id == "cancel").map(|r| r.score).unwrap_or(0.0);

        router_b.import_learned_merge(&learned_json).unwrap();

        // After import: router_b should know "stop" from router_a
        let after = router_b.route("stop it");
        let after_score = after.iter().find(|r| r.id == "cancel").map(|r| r.score).unwrap_or(0.0);
        assert!(after_score > before_score,
            "import_learned_merge should improve score: before={}, after={}", before_score, after_score);
    }

    #[test]
    fn co_occurrence_survives_export_import() {
        let mut router = Router::new();
        router.record_co_occurrence(&["cancel_order", "refund"]);
        router.record_co_occurrence(&["cancel_order", "refund"]);

        let json = router.export_json();
        let restored = Router::import_json(&json).unwrap();

        let pairs = restored.get_co_occurrence();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], ("cancel_order", "refund", 2));
    }
}
