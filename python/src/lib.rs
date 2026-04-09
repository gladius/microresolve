//! Python bindings for ASV Router via PyO3.
//!
//! Usage:
//!   from asv_router import Router
//!   r = Router()
//!   r.add_intent("cancel_order", ["cancel my order", "stop my order"])
//!   result = r.route("I want to cancel")
//!   print(result[0]["id"])  # "cancel_order"

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Intent router: sub-millisecond, model-free, incremental learning.
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

    /// Route a query — returns list of dicts with 'id' and 'score'.
    fn route<'py>(&self, py: Python<'py>, query: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let results = self.inner.route(query);
        results.iter().map(|r| {
            let d = PyDict::new(py);
            d.set_item("id", &r.id)?;
            d.set_item("score", r.score)?;
            Ok(d)
        }).collect()
    }

    /// Route with multi-intent detection.
    #[pyo3(signature = (query, threshold=None))]
    fn route_multi<'py>(&self, py: Python<'py>, query: &str, threshold: Option<f32>) -> PyResult<Bound<'py, PyDict>> {
        let t = threshold.unwrap_or(0.3);
        let output = self.inner.route_multi(query, t);

        let dict = PyDict::new(py);

        let confirmed: Vec<Bound<'py, PyDict>> = output.intents.iter().map(|i| {
            let d = PyDict::new(py);
            d.set_item("id", &i.id).unwrap();
            d.set_item("score", i.score).unwrap();
            d.set_item("intent_type", format!("{:?}", i.intent_type).to_lowercase()).unwrap();
            d
        }).collect();

        dict.set_item("confirmed", confirmed)?;

        Ok(dict)
    }

    /// Learn a new paraphrase for an intent.
    fn learn(&mut self, query: &str, intent_id: &str) {
        self.inner.learn(query, intent_id);
    }

    /// Correct a misroute.
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

    /// Add a seed with collision guard. Returns dict with added, conflicts, etc.
    fn add_seed<'py>(&mut self, py: Python<'py>, intent_id: &str, seed: &str, lang: Option<&str>) -> PyResult<Bound<'py, PyDict>> {
        let result = self.inner.add_seed_checked(intent_id, seed, lang.unwrap_or("en"));
        let d = PyDict::new(py);
        d.set_item("added", result.added)?;
        d.set_item("new_terms", &result.new_terms)?;
        d.set_item("redundant", result.redundant)?;
        let conflicts: Vec<String> = result.conflicts.iter()
            .map(|c| format!("'{}' conflicts with {}", c.term, c.competing_intent))
            .collect();
        d.set_item("conflicts", conflicts)?;
        d.set_item("warning", result.warning)?;
        Ok(d)
    }

    /// Check a seed without adding (read-only collision check).
    fn check_seed<'py>(&self, py: Python<'py>, intent_id: &str, seed: &str) -> PyResult<Bound<'py, PyDict>> {
        let result = self.inner.check_seed(intent_id, seed);
        let d = PyDict::new(py);
        d.set_item("added", false)?;
        d.set_item("new_terms", &result.new_terms)?;
        d.set_item("redundant", result.redundant)?;
        let conflicts: Vec<String> = result.conflicts.iter()
            .map(|c| format!("'{}' conflicts with {}", c.term, c.competing_intent))
            .collect();
        d.set_item("conflicts", conflicts)?;
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

    /// Begin batch mode (defer index rebuild).
    fn begin_batch(&mut self) {
        self.inner.begin_batch();
    }

    /// End batch mode (rebuild index).
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
