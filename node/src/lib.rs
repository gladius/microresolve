//! Node.js bindings for MicroResolve via napi-rs (intent registry only).
//!
//! Routing is handled server-side by the Hebbian L1+L3 system.
//!
//! Usage:
//!   const { Router } = require('asv-router');
//!   const r = new Router();
//!   r.addIntent("cancel_order", ["cancel my order", "stop my order"]);
//!   r.learn("I want to cancel", "cancel_order");  // store phrase for bootstrap

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

#[napi(object)]
pub struct DiscoveredCluster {
    pub name: String,
    pub size: u32,
    pub confidence: f64,
    pub top_terms: Vec<String>,
    pub representative_queries: Vec<String>,
}

#[napi(object)]
pub struct SeedResult {
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

    /// Store that query maps to intent_id (phrase stored for bootstrap).
    #[napi]
    pub fn learn(&mut self, query: String, intent_id: String) {
        self.inner.learn(&query, &intent_id);
    }

    /// Move query from wrong_intent to correct_intent.
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

    /// Add a seed with duplicate checking.
    #[napi]
    pub fn add_seed(&mut self, intent_id: String, seed: String, lang: Option<String>) -> SeedResult {
        let result = self.inner.add_seed_checked(&intent_id, &seed, lang.as_deref().unwrap_or("en"));
        SeedResult {
            added: result.added,
            new_terms: result.new_terms,
            redundant: result.redundant,
            warning: result.warning,
        }
    }

    /// Remove a seed from an intent.
    #[napi]
    pub fn remove_seed(&mut self, intent_id: String, seed: String) -> bool {
        self.inner.remove_seed(&intent_id, &seed)
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

    /// Begin batch mode (no-op, kept for API compat).
    #[napi]
    pub fn begin_batch(&mut self) {
        self.inner.begin_batch();
    }

    /// End batch mode (no-op, kept for API compat).
    #[napi]
    pub fn end_batch(&mut self) {
        self.inner.end_batch();
    }

    /// Discover intent clusters from unlabeled queries.
    #[napi]
    pub fn discover(queries: Vec<String>, expected_intents: Option<u32>) -> Vec<DiscoveredCluster> {
        let config = microresolve_core::discovery::DiscoveryConfig {
            expected_intents: expected_intents.unwrap_or(0) as usize,
            ..Default::default()
        };
        let clusters = microresolve_core::discovery::discover_intents(&queries, &config);
        clusters.iter().map(|c| DiscoveredCluster {
            name: c.suggested_name.clone(),
            size: c.size as u32,
            confidence: c.confidence as f64,
            top_terms: c.top_terms.clone(),
            representative_queries: c.representative_queries.clone(),
        }).collect()
    }
}
