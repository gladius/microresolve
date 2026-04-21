//! Python bindings for MicroResolve via PyO3.
//!
//! Usage:
//!   from microresolve import Router
//!   r = Router()
//!   r.add_intent("cancel_order", ["cancel my order", "stop my order"])
//!   result = r.resolve("I want to cancel")  # returns [{"id": "cancel_order", "score": 0.9}]

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Intent registry: stores phrases, types, descriptions, and metadata.
/// Routing is handled by the Hebbian L1+L3 server.
#[pyclass]
struct Router {
    inner: microresolve_core::Router,
}

#[pymethods]
impl Router {
    #[new]
    fn new() -> Self {
        Router { inner: microresolve_core::Router::new() }
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

    /// Correct a routing mistake: move query from wrong_intent to correct_intent.
    fn correct(&mut self, query: &str, wrong_intent: &str, correct_intent: &str) {
        self.inner.correct(query, wrong_intent, correct_intent);
    }

    /// Set intent type: "action" or "context".
    fn set_intent_type(&mut self, intent_id: &str, intent_type: &str) {
        let t = match intent_type {
            "context" => microresolve_core::IntentType::Context,
            _ => microresolve_core::IntentType::Action,
        };
        self.inner.set_intent_type(intent_id, t);
    }

    /// Resolve a natural language query to matching intents. Returns list of dicts with 'id' and 'score'.
    fn resolve<'py>(&self, py: Python<'py>, query: &str, threshold: Option<f32>, gap: Option<f32>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let results = self.inner.resolve(query, threshold.unwrap_or(0.3), gap.unwrap_or(1.5));
        results.iter().map(|(id, score)| {
            let d = PyDict::new(py);
            d.set_item("id", id)?;
            d.set_item("score", score)?;
            Ok(d)
        }).collect()
    }

    /// Export router state as JSON string.
    fn export_json(&self) -> String {
        self.inner.export_json()
    }

    /// Import router state from JSON string.
    #[staticmethod]
    fn import_json(json: &str) -> PyResult<Router> {
        match microresolve_core::Router::import_json(json) {
            Ok(r) => Ok(Router { inner: r }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
        }
    }

    /// Add a phrase with duplicate checking. Returns dict with added, redundant, warning.
    fn add_phrase<'py>(&mut self, py: Python<'py>, intent_id: &str, phrase: &str, lang: Option<&str>) -> PyResult<Bound<'py, PyDict>> {
        let result = self.inner.add_phrase_checked(intent_id, phrase, lang.unwrap_or("en"));
        let d = PyDict::new(py);
        d.set_item("added", result.added)?;
        d.set_item("new_terms", &result.new_terms)?;
        d.set_item("redundant", result.redundant)?;
        d.set_item("warning", result.warning)?;
        Ok(d)
    }

    /// Check a phrase without adding. Returns dict with redundant, warning.
    fn check_phrase<'py>(&self, py: Python<'py>, intent_id: &str, phrase: &str) -> PyResult<Bound<'py, PyDict>> {
        let result = self.inner.check_phrase(intent_id, phrase);
        let d = PyDict::new(py);
        d.set_item("redundant", result.redundant)?;
        d.set_item("warning", result.warning)?;
        Ok(d)
    }

    /// Remove a phrase from an intent.
    fn remove_phrase(&mut self, intent_id: &str, phrase: &str) -> bool {
        self.inner.remove_phrase(intent_id, phrase)
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

}

/// MicroResolve Python module.
#[pymodule]
fn microresolve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Router>()?;
    Ok(())
}
