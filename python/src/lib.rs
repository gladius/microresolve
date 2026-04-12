//! Python bindings for ASV Router via PyO3 (intent registry only).
//!
//! Routing is handled server-side by the Hebbian L1+L3 system.
//!
//! Usage:
//!   from asv_router import Router
//!   r = Router()
//!   r.add_intent("cancel_order", ["cancel my order", "stop my order"])
//!   r.learn("I want to cancel", "cancel_order")  # stores phrase for bootstrap

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Intent registry: stores phrases, types, descriptions, and metadata.
/// Routing is handled by the Hebbian L1+L3 server.
#[pyclass]
struct Router {
    inner: asv_router_core::Router,
}

#[pymethods]
impl Router {
    #[new]
    fn new() -> Self {
        Router { inner: asv_router_core::Router::new() }
    }

    /// Add an intent with seed phrases.
    fn add_intent(&mut self, id: &str, seeds: Vec<String>) {
        let refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();
        self.inner.add_intent(id, &refs);
    }

    /// Add an intent with seeds in multiple languages.
    fn add_intent_multilingual(&mut self, id: &str, seeds_by_lang: HashMap<String, Vec<String>>) {
        self.inner.add_intent_multilingual(id, seeds_by_lang);
    }

    /// Store that query maps to intent_id (phrase stored for bootstrap).
    fn learn(&mut self, query: &str, intent_id: &str) {
        self.inner.learn(query, intent_id);
    }

    /// Move query from wrong_intent to correct_intent.
    fn correct(&mut self, query: &str, wrong_intent: &str, correct_intent: &str) {
        self.inner.correct(query, wrong_intent, correct_intent);
    }

    /// Set intent type: "action" or "context".
    fn set_intent_type(&mut self, intent_id: &str, intent_type: &str) {
        let t = match intent_type {
            "context" => asv_router_core::IntentType::Context,
            _ => asv_router_core::IntentType::Action,
        };
        self.inner.set_intent_type(intent_id, t);
    }

    /// Set metadata for an intent.
    fn set_metadata(&mut self, intent_id: &str, key: &str, values: Vec<String>) {
        self.inner.set_metadata(intent_id, key, values);
    }

    /// Export router state as JSON string.
    fn export_json(&self) -> String {
        self.inner.export_json()
    }

    /// Import router state from JSON string.
    #[staticmethod]
    fn import_json(json: &str) -> PyResult<Router> {
        match asv_router_core::Router::import_json(json) {
            Ok(r) => Ok(Router { inner: r }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
        }
    }

    /// Add a seed with duplicate checking. Returns dict with added, redundant, warning.
    fn add_seed<'py>(&mut self, py: Python<'py>, intent_id: &str, seed: &str, lang: Option<&str>) -> PyResult<Bound<'py, PyDict>> {
        let result = self.inner.add_seed_checked(intent_id, seed, lang.unwrap_or("en"));
        let d = PyDict::new(py);
        d.set_item("added", result.added)?;
        d.set_item("new_terms", &result.new_terms)?;
        d.set_item("redundant", result.redundant)?;
        d.set_item("conflicts", Vec::<String>::new())?;
        d.set_item("warning", result.warning)?;
        Ok(d)
    }

    /// Check a seed without adding.
    fn check_seed<'py>(&self, py: Python<'py>, intent_id: &str, seed: &str) -> PyResult<Bound<'py, PyDict>> {
        let result = self.inner.check_seed(intent_id, seed);
        let d = PyDict::new(py);
        d.set_item("added", false)?;
        d.set_item("new_terms", &result.new_terms)?;
        d.set_item("redundant", result.redundant)?;
        d.set_item("conflicts", Vec::<String>::new())?;
        d.set_item("warning", result.warning)?;
        Ok(d)
    }

    /// Remove a seed phrase from an intent.
    fn remove_seed(&mut self, intent_id: &str, seed: &str) -> bool {
        self.inner.remove_seed(intent_id, seed)
    }

    /// Set intent description.
    fn set_description(&mut self, intent_id: &str, description: &str) {
        self.inner.set_description(intent_id, description);
    }

    /// Get intent description.
    fn get_description(&self, intent_id: &str) -> String {
        self.inner.get_description(intent_id).to_string()
    }

    /// Delete an intent.
    fn delete_intent(&mut self, id: &str) {
        self.inner.remove_intent(id);
    }

    /// List all intent IDs.
    fn intent_ids(&self) -> Vec<String> {
        self.inner.intent_ids()
    }

    /// Begin batch mode (no-op, kept for API compat).
    fn begin_batch(&mut self) {
        self.inner.begin_batch();
    }

    /// End batch mode (no-op, kept for API compat).
    fn end_batch(&mut self) {
        self.inner.end_batch();
    }

    /// Discover intent clusters from unlabeled queries.
    #[staticmethod]
    #[pyo3(signature = (queries, expected_intents=None))]
    fn discover<'py>(py: Python<'py>, queries: Vec<String>, expected_intents: Option<usize>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let config = asv_router_core::discovery::DiscoveryConfig {
            expected_intents: expected_intents.unwrap_or(0),
            ..Default::default()
        };
        let clusters = asv_router_core::discovery::discover_intents(&queries, &config);

        clusters.iter().map(|c| {
            let d = PyDict::new(py);
            d.set_item("name", &c.suggested_name)?;
            d.set_item("size", c.size)?;
            d.set_item("confidence", c.confidence)?;
            d.set_item("top_terms", &c.top_terms)?;
            d.set_item("representative_queries", &c.representative_queries)?;
            Ok(d)
        }).collect()
    }
}

/// ASV Router Python module.
#[pymodule]
fn asv_router(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Router>()?;
    Ok(())
}
