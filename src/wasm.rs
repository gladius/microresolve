//! WebAssembly bindings for MicroResolve (intent registry only).
//!
//! Routing is handled server-side by the Hebbian L1+L3 system.
//! This WASM module exposes the intent registry for browser-based management.

use wasm_bindgen::prelude::*;
use crate::{Router, IntentType, phrase};

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

    /// Add a single training phrase with duplicate guard. Returns JSON with result.
    pub fn add_phrase(&mut self, intent_id: &str, seed: &str) -> String {
        let result = self.inner.add_phrase_checked(intent_id, seed, "en");
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

    /// Set LLM instructions for an intent.
    pub fn set_instructions(&mut self, intent_id: &str, instructions: &str) {
        self.inner.set_instructions(intent_id, instructions);
    }

    /// Get LLM instructions for an intent.
    pub fn get_instructions(&self, intent_id: &str) -> String {
        self.inner.get_instructions(intent_id).to_string()
    }

    /// Set LLM persona for an intent.
    pub fn set_persona(&mut self, intent_id: &str, persona: &str) {
        self.inner.set_persona(intent_id, persona);
    }

    /// Get LLM persona for an intent.
    pub fn get_persona(&self, intent_id: &str) -> String {
        self.inner.get_persona(intent_id).to_string()
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

    /// Get all intents as JSON: [{id, phrases, phrases_by_lang, intent_type, metadata}]
    pub fn get_intents_json(&self) -> String {
        let mut ids = self.inner.intent_ids();
        ids.sort();
        let intents: Vec<serde_json::Value> = ids.iter().map(|id| {
            let all_phrases = self.inner.get_training(id).unwrap_or_default();
            let by_lang = self.inner.get_training_by_lang(id).cloned().unwrap_or_default();
            let intent_type = self.inner.get_intent_type(id);
            serde_json::json!({
                "id": id,
                "phrases": all_phrases,
                "phrases_by_lang": by_lang,
                "intent_type": intent_type,
                "instructions": self.inner.get_instructions(id),
                "persona": self.inner.get_persona(id),
                "guardrails": self.inner.get_guardrails(id),
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
        phrase::supported_languages_json()
    }

    /// Build an LLM prompt for phrase generation.
    pub fn build_phrase_prompt(&self, intent_id: &str, description: &str, languages_json: &str) -> String {
        let languages: Vec<String> = serde_json::from_str(languages_json).unwrap_or_default();
        phrase::build_prompt(intent_id, description, &languages)
    }

    /// Parse an LLM response into phrases grouped by language.
    pub fn parse_phrase_response(&self, response_text: &str, languages_json: &str) -> String {
        let languages: Vec<String> = serde_json::from_str(languages_json).unwrap_or_default();
        match phrase::parse_response(response_text, &languages) {
            Ok(json) => json,
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }
}
