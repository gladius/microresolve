//! Router: constructor, configuration, persistence, accessors.

use crate::*;
use std::collections::HashMap;

impl Router {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self {
            l0: crate::ngram::NgramIndex::default(),
            l1: crate::scoring::english_morphology_base(),
            l2: crate::scoring::IntentIndex::new(),
            training: HashMap::new(),
            intent_types: HashMap::new(),
            descriptions: HashMap::new(),
            instructions: HashMap::new(),
            persona: HashMap::new(),
            sources: HashMap::new(),
            targets: HashMap::new(),
            schemas: HashMap::new(),
            guardrails: HashMap::new(),
            version: 0,
            connected: false,
            similarity: HashMap::new(),
            namespace_name: String::new(),
            namespace_description: String::new(),
            namespace_models: Vec::new(),
            namespace_default_threshold: None,
            namespace_entity_config: None,
            cached_entity_layer: None,
            domain_descriptions: HashMap::new(),
            top_k: 10,
            max_intents: 5,
            batch_mode: false,
        }
    }

    /// Create a router with configuration.
    pub fn with_config(config: RouterConfig) -> Self {
        let mut r = Self::new();
        r.top_k = config.top_k;
        r.max_intents = config.max_intents;
        if config.server.is_some() {
            r.connected = true;
        }
        r
    }

    /// Load router state from a JSON file.
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

    /// Set the maximum number of results returned by route(). Legacy config.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the maximum number of intents for multi-intent routing. Legacy config.
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
            panic!("Cannot modify router in connected mode — server manages state.");
        }
    }

    /// Begin batch mode. No-op since automata have been removed; kept for API compat.
    pub fn begin_batch(&mut self) {
        self.batch_mode = true;
    }

    /// End batch mode. No-op since automata have been removed; kept for API compat.
    pub fn end_batch(&mut self) {
        self.batch_mode = false;
    }

    /// Number of registered intents.
    pub fn intent_count(&self) -> usize {
        self.training.len()
    }

    /// Get all intent IDs. Canonical source is the training map.
    pub fn intent_ids(&self) -> Vec<String> {
        // Union of training keys and intent_types keys to include intents
        // that have a type/description set but no training phrases yet.
        let mut ids: std::collections::HashSet<String> = self.training.keys().cloned().collect();
        ids.extend(self.intent_types.keys().cloned());
        ids.extend(self.descriptions.keys().cloned());
        let mut v: Vec<String> = ids.into_iter().collect();
        v.sort();
        v
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
            .map(|m| m.iter().map(|(lang, seeds)| (lang.clone(), seeds.len())).collect())
            .unwrap_or_default()
    }

    /// Get the current version number. Incremented on every mutation.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Export router state as JSON.
    pub fn export_json(&self) -> String {
        let state = RouterState {
            training: self.training.clone(),
            intent_types: self.intent_types.clone(),
            descriptions: self.descriptions.clone(),
            instructions: self.instructions.clone(),
            persona: self.persona.clone(),
            sources: self.sources.clone(),
            targets: self.targets.clone(),
            schemas: self.schemas.clone(),
            guardrails: self.guardrails.clone(),
            version: self.version,
            top_k: self.top_k,
            max_intents: self.max_intents,
            similarity: self.similarity.clone(),
            metadata: serde_json::Value::Null,
            intents: serde_json::Value::Null,
            paraphrases: serde_json::Value::Null,
        };
        serde_json::to_string(&state).unwrap_or_default()
    }

    pub fn import_json(json: &str) -> Result<Self, String> {
        let state: RouterState =
            serde_json::from_str(json).map_err(|e| format!("invalid JSON: {}", e))?;

        Ok(Self {
            l0: crate::ngram::NgramIndex::default(),
            l1: crate::scoring::LexicalGraph::new(),
            l2: crate::scoring::IntentIndex::new(),
            training: state.training,
            intent_types: state.intent_types,
            descriptions: state.descriptions,
            instructions: state.instructions,
            persona: state.persona,
            sources: state.sources,
            targets: state.targets,
            schemas: state.schemas,
            guardrails: state.guardrails,
            version: state.version,
            connected: false,
            similarity: state.similarity,
            namespace_name: String::new(),
            namespace_description: String::new(),
            namespace_models: Vec::new(),
            namespace_default_threshold: None,
            namespace_entity_config: None,
            cached_entity_layer: None,
            domain_descriptions: HashMap::new(),
            top_k: state.top_k,
            max_intents: state.max_intents,
            batch_mode: false,
        })
    }

    // ── Scoring layer accessors ───────────────────────────────────────────────

    /// Read access to the morphology graph (normalizes inflections, abbreviations, synonyms).
    pub fn morphology(&self) -> &crate::scoring::LexicalGraph { &self.l1 }
    /// Mutable access to the morphology graph.
    pub fn morphology_mut(&mut self) -> &mut crate::scoring::LexicalGraph { &mut self.l1 }

    /// Read access to the scoring index (word→intent weights).
    pub fn scoring(&self) -> &crate::scoring::IntentIndex { &self.l2 }
    /// Mutable access to the scoring index.
    pub fn scoring_mut(&mut self) -> &mut crate::scoring::IntentIndex { &mut self.l2 }

    /// Read access to the typo corrector (character n-gram index).
    pub fn typo_index(&self) -> &crate::ngram::NgramIndex { &self.l0 }

    // Undocumented layer accessors kept for server binary compatibility.
    // Prefer morphology()/scoring()/typo_index() in new code.
    #[doc(hidden)] pub fn l1(&self) -> &crate::scoring::LexicalGraph { &self.l1 }
    #[doc(hidden)] pub fn l1_mut(&mut self) -> &mut crate::scoring::LexicalGraph { &mut self.l1 }
    #[doc(hidden)] pub fn l2(&self) -> &crate::scoring::IntentIndex { &self.l2 }
    #[doc(hidden)] pub fn l2_mut(&mut self) -> &mut crate::scoring::IntentIndex { &mut self.l2 }
    #[doc(hidden)] pub fn l0(&self) -> &crate::ngram::NgramIndex { &self.l0 }

    /// Merge base L1 edges into this router's L1 graph.
    /// Base edges are only added where the target term has no existing entry
    /// (namespace-specific edges take priority).
    /// Used by the server to inject global WordNet/ConceptNet at namespace creation.
    pub fn merge_l1_base(&mut self, base: &crate::scoring::LexicalGraph) {
        use crate::scoring::EdgeKind;
        for (term, edges) in &base.edges {
            let existing = self.l1.edges.entry(term.clone()).or_default();
            for edge in edges {
                // Only merge morphological and abbreviation edges from the global base.
                // Synonym edges from WordNet/ConceptNet are context-free — they inject
                // all word senses without knowing the domain, causing polysemy pollution
                // in L2. Domain synonyms come from LLM import only (namespace-specific).
                if matches!(edge.kind, EdgeKind::Synonym) { continue; }
                if !existing.iter().any(|e| e.target == edge.target) {
                    existing.push(edge.clone());
                }
            }
        }
    }

    /// Rebuild L0 from the combined vocabulary of L1 + L2.
    pub fn rebuild_l0(&mut self) {
        self.l0 = crate::ngram::build_for_namespace(Some(&self.l1), Some(&self.l2));
    }

    /// Preprocess a phrase through L1, learn it into L2, and rebuild L0.
    /// Called internally whenever a new phrase is added to this router.
    pub fn index_phrase(&mut self, intent_id: &str, phrase: &str) {
        self.index_phrase_no_rebuild(intent_id, phrase);
        self.rebuild_l0();
    }

    /// Index a phrase into L2 without rebuilding L0.
    /// Call `rebuild_l0()` once after bulk indexing.
    pub(crate) fn index_phrase_no_rebuild(&mut self, intent_id: &str, phrase: &str) {
        let preprocessed = self.l1.preprocess(phrase);
        let words = crate::tokenizer::tokenize(&preprocessed.expanded);
        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        if !word_refs.is_empty() {
            self.l2.learn_phrase(&word_refs, intent_id);
        }
        self.l2.index_char_ngrams(phrase, intent_id);
    }

    /// Rebuild L2 from scratch using all training phrases currently in this router.
    /// Clears the existing L2 index and re-indexes every stored phrase.
    /// Used after a phrase or intent is removed so stale word→intent edges are cleared.
    pub fn rebuild_l2(&mut self) {
        self.l2 = crate::scoring::IntentIndex::new();
        let all: Vec<(String, String)> = self.training.iter()
            .flat_map(|(intent_id, lang_map)| {
                lang_map.values()
                    .flat_map(|phrases| phrases.iter().map(|p| (intent_id.clone(), p.clone())))
            })
            .collect();
        for (intent_id, phrase) in &all {
            self.index_phrase_no_rebuild(intent_id, phrase);
        }
        self.l2.rebuild_idf();
        self.rebuild_l0();
    }

    /// Resolve a natural language query to matching intents.
    /// Returns sorted (intent_id, score) pairs above `threshold`.
    pub fn resolve(&self, query: &str, threshold: f32, gap: f32) -> Vec<(String, f32)> {
        // L0: typo correction
        let q0 = self.l0.correct_query(query);
        // L1: normalize + expand
        let preprocessed = self.l1.preprocess(&q0);
        // L2: score
        let (scored, _negation) = self.l2.score_multi_normalized(&preprocessed.expanded, threshold, gap);
        scored
    }
}

/// Serializable router state for persistence.
#[derive(serde::Serialize, serde::Deserialize)]
struct RouterState {
    training: HashMap<String, HashMap<String, Vec<String>>>,
    #[serde(default)]
    intent_types: HashMap<String, IntentType>,
    #[serde(default)]
    descriptions: HashMap<String, String>,
    #[serde(default)]
    instructions: HashMap<String, String>,
    #[serde(default)]
    persona: HashMap<String, String>,
    #[serde(default)]
    sources: HashMap<String, IntentSource>,
    #[serde(default)]
    targets: HashMap<String, IntentTarget>,
    #[serde(default)]
    schemas: HashMap<String, serde_json::Value>,
    #[serde(default)]
    guardrails: HashMap<String, Vec<String>>,
    #[serde(default)]
    version: u64,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_max_intents")]
    max_intents: usize,
    #[serde(default)]
    similarity: HashMap<String, Vec<(String, f32)>>,
    // Old fields present in saved JSON — ignored during load.
    #[serde(default, skip_serializing)]
    #[allow(dead_code)]
    metadata: serde_json::Value,
    #[serde(default, skip_serializing)]
    #[allow(dead_code)]
    intents: serde_json::Value,
    #[serde(default, skip_serializing)]
    #[allow(dead_code)]
    paraphrases: serde_json::Value,
}

fn default_top_k() -> usize { 10 }
fn default_max_intents() -> usize { 5 }
