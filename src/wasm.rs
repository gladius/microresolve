//! WebAssembly bindings for ASV Router.

use wasm_bindgen::prelude::*;
use crate::{Router, IntentRelation, IntentType, seed};

#[wasm_bindgen]
pub struct WasmRouter {
    inner: Router,
}

#[wasm_bindgen]
impl WasmRouter {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { inner: Router::new() }
    }

    /// Add an intent. seeds_json is a JSON array: ["phrase1", "phrase2"]
    pub fn add_intent(&mut self, id: &str, seeds_json: &str) {
        let seeds: Vec<String> = serde_json::from_str(seeds_json).unwrap_or_default();
        let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();
        self.inner.add_intent(id, &seed_refs);
    }

    /// Add an intent with seeds grouped by language.
    /// seeds_json is: {"en": ["phrase1"], "es": ["frase1"]}
    pub fn add_intent_multilingual(&mut self, id: &str, seeds_json: &str) {
        let seeds: std::collections::HashMap<String, Vec<String>> =
            serde_json::from_str(seeds_json).unwrap_or_default();
        self.inner.add_intent_multilingual(id, seeds);
    }

    /// Add a single seed phrase with collision guard. Returns JSON with result.
    pub fn add_seed(&mut self, intent_id: &str, seed: &str) -> String {
        let result = self.inner.add_seed_checked(intent_id, seed, "en");
        serde_json::json!({
            "added": result.added,
            "new_terms": result.new_terms,
            "redundant": result.redundant,
            "conflicts": result.conflicts.iter().map(|c| format!("'{}' conflicts with {}", c.term, c.competing_intent)).collect::<Vec<_>>(),
            "reason": result.warning,
        }).to_string()
    }

    pub fn remove_intent(&mut self, id: &str) {
        self.inner.remove_intent(id);
    }

    /// Set intent type: "action" or "context".
    pub fn set_intent_type(&mut self, intent_id: &str, intent_type: &str) {
        let t = match intent_type {
            "context" => IntentType::Context,
            _ => IntentType::Action,
        };
        self.inner.set_intent_type(intent_id, t);
    }

    /// Get intent type. Returns "action" or "context".
    pub fn get_intent_type(&self, intent_id: &str) -> String {
        match self.inner.get_intent_type(intent_id) {
            IntentType::Action => "action".to_string(),
            IntentType::Context => "context".to_string(),
        }
    }

    /// Set metadata for an intent. values_json is a JSON array: ["val1", "val2"]
    pub fn set_metadata(&mut self, intent_id: &str, key: &str, values_json: &str) {
        let values: Vec<String> = serde_json::from_str(values_json).unwrap_or_default();
        self.inner.set_metadata(intent_id, key, values);
    }

    /// Get all metadata for an intent. Returns JSON object or "null".
    pub fn get_metadata(&self, intent_id: &str) -> String {
        match self.inner.get_metadata(intent_id) {
            Some(meta) => serde_json::to_string(meta).unwrap_or_default(),
            None => "null".to_string(),
        }
    }

    /// Route a query. Returns JSON array of {id, score}.
    pub fn route(&self, query: &str) -> String {
        let results = self.inner.route(query);
        let out: Vec<serde_json::Value> = results.iter().map(|r| {
            serde_json::json!({"id": r.id, "score": (r.score * 100.0).round() / 100.0})
        }).collect();
        serde_json::to_string(&out).unwrap_or_default()
    }

    /// Route multi-intent. Returns JSON with intents, relations, and metadata.
    pub fn route_multi(&self, query: &str, threshold: f32) -> String {
        let output = self.inner.route_multi(query, threshold);
        let intents: Vec<serde_json::Value> = output.intents.iter().map(|i| {
            serde_json::json!({
                "id": i.id,
                "score": (i.score * 100.0).round() / 100.0,
                "position": i.position,
                "span": [i.span.0, i.span.1],
                "intent_type": i.intent_type
            })
        }).collect();
        let relations: Vec<serde_json::Value> = output.relations.iter().map(|r| {
            match r {
                IntentRelation::Parallel => serde_json::json!({"type": "Parallel"}),
                IntentRelation::Sequential { first, then } =>
                    serde_json::json!({"type": "Sequential", "first": first, "then": then}),
                IntentRelation::Conditional { primary, fallback } =>
                    serde_json::json!({"type": "Conditional", "primary": primary, "fallback": fallback}),
                IntentRelation::Reverse { stated_first, execute_first } =>
                    serde_json::json!({"type": "Reverse", "stated_first": stated_first, "execute_first": execute_first}),
                IntentRelation::Negation { do_this, not_this } =>
                    serde_json::json!({"type": "Negation", "do_this": do_this, "not_this": not_this}),
            }
        }).collect();
        serde_json::to_string(&serde_json::json!({
            "intents": intents,
            "relations": relations,
            "metadata": output.metadata
        })).unwrap_or_default()
    }

    pub fn learn(&mut self, query: &str, intent_id: &str) {
        self.inner.learn(query, intent_id);
    }

    pub fn correct(&mut self, query: &str, wrong: &str, right: &str) {
        self.inner.correct(query, wrong, right);
    }

    pub fn intent_count(&self) -> usize {
        self.inner.intent_count()
    }

    pub fn begin_batch(&mut self) {
        self.inner.begin_batch();
    }

    pub fn end_batch(&mut self) {
        self.inner.end_batch();
    }

    /// Get all intents as JSON: [{id, seeds, seeds_by_lang, learned_count, intent_type, metadata}]
    pub fn get_intents_json(&self) -> String {
        let mut ids = self.inner.intent_ids();
        ids.sort();
        let intents: Vec<serde_json::Value> = ids.iter().map(|id| {
            let all_seeds = self.inner.get_training(id).unwrap_or_default();
            let by_lang = self.inner.get_training_by_lang(id).cloned().unwrap_or_default();
            let learned = self.inner.get_vector(id)
                .map(|v| v.learned_term_count()).unwrap_or(0);
            let intent_type = self.inner.get_intent_type(id);
            let metadata = self.inner.get_metadata(id).cloned().unwrap_or_default();
            serde_json::json!({
                "id": id,
                "seeds": all_seeds,
                "seeds_by_lang": by_lang,
                "learned_count": learned,
                "intent_type": intent_type,
                "metadata": metadata
            })
        }).collect();
        serde_json::to_string(&intents).unwrap_or_default()
    }

    pub fn export_state(&self) -> String {
        self.inner.export_json()
    }

    pub fn import_state(&mut self, json: &str) -> bool {
        match Router::import_json(json) {
            Ok(r) => { self.inner = r; true }
            Err(_) => false
        }
    }

    /// Get supported languages as JSON: {"en": "English", ...}
    pub fn get_languages(&self) -> String {
        seed::supported_languages_json()
    }

    /// Build an LLM prompt for seed generation.
    /// languages_json: JSON array of language codes, e.g. ["en", "zh"]
    pub fn build_seed_prompt(&self, intent_id: &str, description: &str, languages_json: &str) -> String {
        let languages: Vec<String> = serde_json::from_str(languages_json).unwrap_or_default();
        seed::build_prompt(intent_id, description, &languages)
    }

    /// Parse an LLM response into seeds grouped by language.
    /// Returns JSON: {"seeds_by_lang": {...}, "total": N} or {"error": "..."}
    pub fn parse_seed_response(&self, response_text: &str, languages_json: &str) -> String {
        let languages: Vec<String> = serde_json::from_str(languages_json).unwrap_or_default();
        match seed::parse_response(response_text, &languages) {
            Ok(json) => json,
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }
}
