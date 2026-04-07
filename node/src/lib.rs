//! Node.js bindings for ASV Router via napi-rs.
//!
//! Usage:
//!   const { Router } = require('asv-router');
//!   const r = new Router();
//!   r.addIntent("cancel_order", ["cancel my order", "stop my order"]);
//!   const result = r.route("I want to cancel");
//!   console.log(result[0].id);  // "cancel_order"

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

#[napi(object)]
pub struct RouteResult {
    pub id: String,
    pub score: f64,
}

#[napi(object)]
pub struct MultiRouteResult {
    pub id: String,
    pub score: f64,
    pub intent_type: String,
}

#[napi(object)]
pub struct MultiRouteOutput {
    pub confirmed: Vec<MultiRouteResult>,
}

#[napi(object)]
pub struct DiscoveredCluster {
    pub name: String,
    pub size: u32,
    pub confidence: f64,
    pub top_terms: Vec<String>,
    pub representative_queries: Vec<String>,
}

#[napi]
pub struct Router {
    inner: asv_router_core::Router,
}

#[napi]
impl Router {
    #[napi(constructor)]
    pub fn new() -> Self {
        Router { inner: asv_router_core::Router::new() }
    }

    /// Add an intent with seed phrases.
    #[napi]
    pub fn add_intent(&mut self, id: String, seeds: Vec<String>) {
        let refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();
        self.inner.add_intent(&id, &refs);
    }

    /// Route a query — returns array of {id, score}.
    #[napi]
    pub fn route(&self, query: String) -> Vec<RouteResult> {
        self.inner.route(&query).iter().map(|r| RouteResult {
            id: r.id.clone(),
            score: r.score as f64,
        }).collect()
    }

    /// Route with multi-intent detection.
    #[napi]
    pub fn route_multi(&self, query: String, threshold: Option<f64>) -> MultiRouteOutput {
        let t = threshold.unwrap_or(0.3) as f32;
        let output = self.inner.route_multi(&query, t);
        MultiRouteOutput {
            confirmed: output.intents.iter().map(|i| MultiRouteResult {
                id: i.id.clone(),
                score: i.score as f64,
                intent_type: format!("{:?}", i.intent_type).to_lowercase(),
            }).collect(),
        }
    }

    /// Learn a new paraphrase for an intent.
    #[napi]
    pub fn learn(&mut self, query: String, intent_id: String) {
        self.inner.learn(&query, &intent_id);
    }

    /// Correct a misroute.
    #[napi]
    pub fn correct(&mut self, query: String, wrong_intent: String, correct_intent: String) {
        self.inner.correct(&query, &wrong_intent, &correct_intent);
    }

    /// Set intent type: "action" or "context".
    #[napi]
    pub fn set_intent_type(&mut self, intent_id: String, intent_type: String) {
        let t = match intent_type.as_str() {
            "context" => asv_router_core::IntentType::Context,
            _ => asv_router_core::IntentType::Action,
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
        match asv_router_core::Router::import_json(&json) {
            Ok(r) => Ok(Router { inner: r }),
            Err(e) => Err(Error::from_reason(e)),
        }
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

    /// Begin batch mode.
    #[napi]
    pub fn begin_batch(&mut self) {
        self.inner.begin_batch();
    }

    /// End batch mode.
    #[napi]
    pub fn end_batch(&mut self) {
        self.inner.end_batch();
    }

    /// Discover intent clusters from unlabeled queries.
    #[napi]
    pub fn discover(queries: Vec<String>, expected_intents: Option<u32>) -> Vec<DiscoveredCluster> {
        let config = asv_router_core::discovery::DiscoveryConfig {
            expected_intents: expected_intents.unwrap_or(0) as usize,
            ..Default::default()
        };
        let clusters = asv_router_core::discovery::discover_intents(&queries, &config);
        clusters.iter().map(|c| DiscoveredCluster {
            name: c.suggested_name.clone(),
            size: c.size as u32,
            confidence: c.confidence as f64,
            top_terms: c.top_terms.clone(),
            representative_queries: c.representative_queries.clone(),
        }).collect()
    }
}
