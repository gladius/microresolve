//! Node.js bindings for MicroResolve via napi-rs.
//!
//! Usage:
//!   const { Router } = require('microresolve');
//!   const r = new Router();
//!   r.addIntent("cancel_order", ["cancel my order", "stop my order"]);
//!   const results = r.resolve("I want to cancel");  // [{ id: "cancel_order", score: 0.9 }]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

#[napi(object)]
pub struct ResolveMatch {
    pub id: String,
    pub score: f64,
}

#[napi(object)]
pub struct PhraseResult {
    pub added: bool,
    pub new_terms: Vec<String>,
    pub redundant: bool,
    pub warning: Option<String>,
}

#[napi]
pub struct Router {
    inner: microresolve_core::Router,
}

#[napi]
impl Router {
    #[napi(constructor)]
    pub fn new() -> Self {
        Router { inner: microresolve_core::Router::new() }
    }

    /// Add an intent with seed phrases.
    #[napi]
    pub fn add_intent(&mut self, id: String, seeds: Vec<String>) {
        let refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();
        self.inner.add_intent(&id, &refs);
    }

    /// Resolve a natural language query to matching intents. Returns matches sorted by score descending.
    #[napi]
    pub fn resolve(&self, query: String, threshold: Option<f64>, gap: Option<f64>) -> Vec<ResolveMatch> {
        let results = self.inner.resolve(&query, threshold.unwrap_or(0.3) as f32, gap.unwrap_or(1.5) as f32);
        results.into_iter().map(|(id, score)| ResolveMatch { id, score: score as f64 }).collect()
    }

    /// Correct a routing mistake: move query from wrong_intent to correct_intent.
    #[napi]
    pub fn correct(&mut self, query: String, wrong_intent: String, correct_intent: String) {
        self.inner.correct(&query, &wrong_intent, &correct_intent);
    }

    /// Set intent type: "action" or "context".
    #[napi]
    pub fn set_intent_type(&mut self, intent_id: String, intent_type: String) {
        let t = match intent_type.as_str() {
            "context" => microresolve_core::IntentType::Context,
            _ => microresolve_core::IntentType::Action,
        };
        self.inner.set_intent_type(&intent_id, t);
    }

    /// Export router state as JSON string.
    #[napi]
    pub fn export_json(&self) -> String {
        self.inner.export_json()
    }

    /// Import router state from JSON string.
    #[napi(factory)]
    pub fn import_json(json: String) -> Result<Router> {
        match microresolve_core::Router::import_json(&json) {
            Ok(r) => Ok(Router { inner: r }),
            Err(e) => Err(Error::from_reason(e)),
        }
    }

    /// Add a phrase with duplicate checking.
    #[napi]
    pub fn add_phrase(&mut self, intent_id: String, phrase: String, lang: Option<String>) -> PhraseResult {
        let result = self.inner.add_phrase_checked(&intent_id, &phrase, lang.as_deref().unwrap_or("en"));
        PhraseResult {
            added: result.added,
            new_terms: result.new_terms,
            redundant: result.redundant,
            warning: result.warning,
        }
    }

    /// Remove a phrase from an intent.
    #[napi]
    pub fn remove_phrase(&mut self, intent_id: String, phrase: String) -> bool {
        self.inner.remove_phrase(&intent_id, &phrase)
    }

    /// Set intent description.
    #[napi]
    pub fn set_description(&mut self, intent_id: String, description: String) {
        self.inner.set_description(&intent_id, &description);
    }

    /// Get intent description.
    #[napi]
    pub fn get_description(&self, intent_id: String) -> String {
        self.inner.get_description(&intent_id).to_string()
    }

    /// Delete an intent.
    #[napi]
    pub fn delete_intent(&mut self, id: String) {
        self.inner.remove_intent(&id);
    }

    /// List all intent IDs.
    #[napi]
    pub fn intent_ids(&self) -> Vec<String> {
        self.inner.intent_ids()
    }

}
