//! Router: constructor, configuration, persistence, accessors.

use crate::*;
use std::collections::HashMap;

impl Router {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self {
            training: HashMap::new(),
            intent_types: HashMap::new(),
            descriptions: HashMap::new(),
            metadata: HashMap::new(),
            version: 0,
            connected: false,
            similarity: HashMap::new(),
            namespace_name: String::new(),
            namespace_description: String::new(),
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

    /// Export router state as JSON (training phrases, types, descriptions, metadata).
    pub fn export_json(&self) -> String {
        let state = RouterState {
            training: self.training.clone(),
            intent_types: self.intent_types.clone(),
            descriptions: self.descriptions.clone(),
            metadata: self.metadata.clone(),
            version: self.version,
            top_k: self.top_k,
            max_intents: self.max_intents,
            similarity: self.similarity.clone(),
            intents: serde_json::Value::Null,
            paraphrases: serde_json::Value::Null,
        };
        serde_json::to_string(&state).unwrap_or_default()
    }

    /// Import router state from JSON.
    pub fn import_json(json: &str) -> Result<Self, String> {
        let state: RouterState =
            serde_json::from_str(json).map_err(|e| format!("invalid JSON: {}", e))?;

        Ok(Self {
            training: state.training,
            intent_types: state.intent_types,
            descriptions: state.descriptions,
            metadata: state.metadata,
            version: state.version,
            connected: false,
            similarity: state.similarity,
            namespace_name: String::new(),
            namespace_description: String::new(),
            domain_descriptions: HashMap::new(),
            top_k: state.top_k,
            max_intents: state.max_intents,
            batch_mode: false,
        })
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
    metadata: HashMap<String, HashMap<String, Vec<String>>>,
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
    intents: serde_json::Value,
    #[serde(default, skip_serializing)]
    #[allow(dead_code)]
    paraphrases: serde_json::Value,
}

fn default_top_k() -> usize { 10 }
fn default_max_intents() -> usize { 5 }
