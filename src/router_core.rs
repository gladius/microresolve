//! Router: constructor, configuration, persistence, accessors.

use crate::*;
use crate::tokenizer::*;
use crate::vector::LearnedVector;
use crate::index::InvertedIndex;
use std::collections::{HashMap, HashSet};

impl Router {
    /// Create a new empty router in local mode.
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
            descriptions: HashMap::new(),
            metadata: HashMap::new(),
            co_occurrence: HashMap::new(),
            temporal_order: HashMap::new(),
            intent_sequences: Vec::new(),
            paraphrase_phrases: HashMap::new(),
            paraphrase_automaton: None,
            paraphrase_patterns: Vec::new(),
            paraphrase_dirty: false,
            version: 0,
            max_intents: 5,
            connected: false,
            similarity: HashMap::new(),
            expansion_discount: 0.3,
        }
    }

    /// Create a router with configuration.
    ///
    /// ```
    /// use asv_router::{Router, RouterConfig};
    /// let r = Router::with_config(RouterConfig {
    ///     top_k: 5,
    ///     max_intents: 10,
    ///     ..Default::default()
    /// });
    /// ```
    pub fn with_config(config: RouterConfig) -> Self {
        let mut r = Self::new();
        r.top_k = config.top_k;
        r.max_intents = config.max_intents;
        if config.server.is_some() {
            r.connected = true;
        }
        r
    }

    /// Load router state from a JSON file. Returns error if file not found or invalid.
    pub fn load(path: &str) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path, e))?;
        Self::import_json(&json)
    }

    /// Save router state to a JSON file.
    pub fn save(&self, path: &str) -> Result<(), String> {
        if self.connected {
            return Err("Cannot save in connected mode — server manages state".to_string());
        }
        let json = self.export_json();
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write {}: {}", path, e))
    }

    /// Returns true if this router is in connected (read-only) mode.
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Set the maximum number of results returned by `route()`.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the maximum number of intents detected by `route_multi()`. Default: 5.
    pub fn set_max_intents(&mut self, max: usize) {
        self.max_intents = max;
    }

    /// Get the current max intents setting.
    pub fn max_intents(&self) -> usize {
        self.max_intents
    }

    /// Guard: panics if router is in connected (read-only) mode.
    pub(crate) fn require_local(&self) {
        if self.connected {
            panic!("Cannot modify router in connected mode — server manages state. \
                    Use the server UI or API to make changes.");
        }
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

    /// Get seed counts per language for an intent.
    pub fn seed_counts_by_lang(&self, intent_id: &str) -> HashMap<String, usize> {
        self.training.get(intent_id)
            .map(|lang_map| lang_map.iter().map(|(lang, seeds)| (lang.clone(), seeds.len())).collect())
            .unwrap_or_default()
    }

    /// Check a seed phrase for collisions and redundancy before adding.

    pub fn export_json(&self) -> String {
        let paraphrases: Vec<(String, String, f32)> = self.paraphrase_phrases.iter()
            .map(|(phrase, (intent_id, weight))| (phrase.clone(), intent_id.clone(), *weight))
            .collect();
        let co_occurrence: Vec<(String, String, u32)> = self.co_occurrence.iter()
            .map(|((a, b), &count)| (a.clone(), b.clone(), count))
            .collect();
        let temporal_order: Vec<(String, String, u32)> = self.temporal_order.iter()
            .map(|((a, b), &count)| (a.clone(), b.clone(), count))
            .collect();
        let state = RouterState {
            intents: self.vectors.clone(),
            training: self.training.clone(),
            top_k: self.top_k,
            intent_types: self.intent_types.clone(),
            descriptions: self.descriptions.clone(),
            metadata: self.metadata.clone(),
            paraphrases,
            co_occurrence,
            temporal_order,
            intent_sequences: self.intent_sequences.clone(),
            version: self.version,
            max_intents: self.max_intents,
            similarity: self.similarity.clone(),
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

        // Restore temporal ordering from serialized Vec
        let mut temporal_order_map: HashMap<(String, String), u32> = HashMap::new();
        for (a, b, count) in state.temporal_order {
            temporal_order_map.insert((a, b), count);
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
            descriptions: state.descriptions,
            metadata: state.metadata,
            co_occurrence: co_occurrence_map,
            temporal_order: temporal_order_map,
            intent_sequences: state.intent_sequences,
            paraphrase_phrases,
            paraphrase_automaton: None,
            paraphrase_patterns: Vec::new(),
            paraphrase_dirty: false,
            version: state.version,
            max_intents: state.max_intents,
            connected: false,
            similarity: state.similarity,
            expansion_discount: 0.3,
        };
        router.rebuild_cjk_automaton_now();
        router.rebuild_paraphrase_automaton_now();
        Ok(router)
    }

    // Number of registered intents.


}

// Serializable router state for persistence.
#[derive(serde::Serialize, serde::Deserialize)]
struct RouterState {
    intents: HashMap<String, LearnedVector>,
    training: HashMap<String, HashMap<String, Vec<String>>>,
    top_k: usize,
    #[serde(default)]
    intent_types: HashMap<String, IntentType>,
    #[serde(default)]
    descriptions: HashMap<String, String>,
    #[serde(default)]
    metadata: HashMap<String, HashMap<String, Vec<String>>>,
    #[serde(default)]
    paraphrases: Vec<(String, String, f32)>,
    #[serde(default)]
    co_occurrence: Vec<(String, String, u32)>,
    #[serde(default)]
    temporal_order: Vec<(String, String, u32)>,
    #[serde(default)]
    intent_sequences: Vec<Vec<String>>,
    #[serde(default)]
    version: u64,
    #[serde(default = "default_max_intents")]
    max_intents: usize,
    #[serde(default)]
    similarity: HashMap<String, Vec<(String, f32)>>,
}

fn default_max_intents() -> usize { 5 }
