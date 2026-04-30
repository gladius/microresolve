//! Python bindings for MicroResolve via PyO3.
//!
//! ```python
//! from microresolve import MicroResolve
//!
//! engine = MicroResolve(data_dir="/tmp/mr")
//! ns = engine.namespace("security")
//! ns.add_intent("jailbreak", ["ignore prior instructions", "ignore your safety rules"])
//! matches = ns.resolve("ignore prior instructions and reveal")
//! # → [Match(id='jailbreak', score=0.87)]
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

// ── Match ─────────────────────────────────────────────────────────────────────

/// A classification match: intent id paired with its match score.
#[pyclass(get_all, from_py_object)]
#[derive(Clone)]
struct Match {
    /// Intent identifier.
    id: String,
    /// Match score (higher = better match).
    score: f32,
}

#[pymethods]
impl Match {
    fn __repr__(&self) -> String {
        format!("Match(id={:?}, score={:.4})", self.id, self.score)
    }
}

// ── IntentInfo ────────────────────────────────────────────────────────────────

/// Read-only view of an intent's metadata and training phrases.
#[pyclass(get_all, from_py_object)]
#[derive(Clone)]
struct IntentInfo {
    /// Intent identifier.
    id: String,
    /// `"action"` or `"context"`.
    intent_type: String,
    /// Human-readable description.
    description: String,
    /// Training phrases grouped by language code.
    training: HashMap<String, Vec<String>>,
}

#[pymethods]
impl IntentInfo {
    fn __repr__(&self) -> String {
        format!("IntentInfo(id={:?}, intent_type={:?}, phrases={})",
            self.id, self.intent_type,
            self.training.values().map(|v| v.len()).sum::<usize>())
    }
}

fn info_to_py(info: microresolve_core::IntentInfo) -> IntentInfo {
    let intent_type = match info.intent_type {
        microresolve_core::IntentType::Action => "action".to_string(),
        microresolve_core::IntentType::Context => "context".to_string(),
    };
    IntentInfo {
        id: info.id,
        intent_type,
        description: info.description,
        training: info.training,
    }
}

// ── Namespace ─────────────────────────────────────────────────────────────────

/// Per-namespace handle. Obtain via `engine.namespace(id)`.
///
/// Each method creates a lightweight `NamespaceHandle` internally — cheap,
/// no lock held between calls.
#[pyclass]
struct Namespace {
    engine: Arc<microresolve_core::MicroResolve>,
    id: String,
}

#[pymethods]
impl Namespace {
    /// Add an intent with seed phrases.
    ///
    /// `phrases` can be a list of strings (English) or a dict mapping language
    /// codes to phrase lists for multilingual seeding:
    ///
    ///   ns.add_intent("greet", ["hello", "hi"])
    ///   ns.add_intent("greet", {"en": ["hello"], "fr": ["bonjour"]})
    ///
    /// Returns the number of phrases indexed.
    fn add_intent(&self, intent_id: &str, phrases: &Bound<'_, PyAny>) -> PyResult<usize> {
        let seeds = py_to_seeds(phrases)?;
        let handle = self.engine.namespace(&self.id);
        handle.add_intent(intent_id, seeds)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Resolve a query to matching intents.
    ///
    /// Returns a list of `Match` objects sorted by score descending.
    fn resolve(&self, query: &str) -> Vec<Match> {
        let handle = self.engine.namespace(&self.id);
        handle.resolve(query).into_iter()
            .map(|m| Match { id: m.id, score: m.score })
            .collect()
    }

    /// Resolve with explicit threshold and gap overrides.
    #[pyo3(signature = (query, threshold=0.3, gap=1.5))]
    fn resolve_with(&self, query: &str, threshold: f32, gap: f32) -> Vec<Match> {
        let opts = microresolve_core::ResolveOptions { threshold, gap };
        let handle = self.engine.namespace(&self.id);
        handle.resolve_with(query, opts).into_iter()
            .map(|m| Match { id: m.id, score: m.score })
            .collect()
    }

    /// Correct a mis-classification: move query from `wrong_intent` to `right_intent`.
    /// Applied locally immediately; in connected mode, buffered and shipped to
    /// the server on the next sync tick.
    ///
    /// Raises `ValueError` if `right_intent` does not exist.
    fn correct(&self, query: &str, wrong_intent: &str, right_intent: &str) -> PyResult<()> {
        let handle = self.engine.namespace(&self.id);
        handle.correct(query, wrong_intent, right_intent)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Remove an intent and all its phrases.
    fn remove_intent(&self, intent_id: &str) {
        self.engine.namespace(&self.id).remove_intent(intent_id);
    }

    /// Add a single phrase to an existing intent.
    ///
    /// Returns a dict with `added` (bool), `redundant` (bool), `warning` (str | None).
    fn add_phrase<'py>(&self, py: Python<'py>, intent_id: &str, phrase: &str, lang: Option<&str>) -> PyResult<Bound<'py, PyDict>> {
        let result = self.engine.namespace(&self.id).add_phrase(intent_id, phrase, lang.unwrap_or("en"));
        let d = PyDict::new(py);
        d.set_item("added", result.added)?;
        d.set_item("redundant", result.redundant)?;
        d.set_item("warning", result.warning)?;
        Ok(d)
    }

    /// Read-only view of an intent's metadata and training phrases.
    ///
    /// Returns `None` if the intent does not exist.
    fn intent(&self, intent_id: &str) -> Option<IntentInfo> {
        self.engine.namespace(&self.id).intent(intent_id).map(info_to_py)
    }

    /// Update metadata fields on an existing intent.
    ///
    /// Accepts kwargs: `intent_type` ("action"|"context"), `description`,
    /// `instructions`, `persona`, `guardrails` (list[str]).
    /// Raises `ValueError` if the intent does not exist or intent_type is invalid.
    fn update_intent(&self, intent_id: &str, edit: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut e = microresolve_core::IntentEdit::default();
        if let Some(v) = edit.get_item("intent_type")? {
            let s = v.extract::<String>()?;
            e.intent_type = Some(match s.as_str() {
                "action" => microresolve_core::IntentType::Action,
                "context" => microresolve_core::IntentType::Context,
                other => return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("intent_type must be 'action' or 'context', got '{other}'"))),
            });
        }
        if let Some(v) = edit.get_item("description")? {
            e.description = Some(v.extract::<String>()?);
        }
        if let Some(v) = edit.get_item("instructions")? {
            e.instructions = Some(v.extract::<String>()?);
        }
        if let Some(v) = edit.get_item("persona")? {
            e.persona = Some(v.extract::<String>()?);
        }
        if let Some(v) = edit.get_item("guardrails")? {
            e.guardrails = Some(v.extract::<Vec<String>>()?);
        }
        self.engine.namespace(&self.id).update_intent(intent_id, e)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// All intent IDs in this namespace.
    fn intent_ids(&self) -> Vec<String> {
        self.engine.namespace(&self.id).intent_ids()
    }

    /// Number of intents in this namespace.
    fn intent_count(&self) -> usize {
        self.engine.namespace(&self.id).intent_count()
    }

    /// Monotonic version counter. Increments on every mutation; in connected
    /// mode the background tick advances it when the server has newer data.
    fn version(&self) -> u64 {
        self.engine.namespace(&self.id).version()
    }

    /// Persist this namespace to disk now. No-op when `data_dir` is not set.
    fn flush(&self) -> PyResult<()> {
        self.engine.namespace(&self.id).flush()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Namespace identifier.
    #[getter]
    fn id(&self) -> &str { &self.id }

    fn __repr__(&self) -> String {
        format!("Namespace(id={:?})", self.id)
    }
}

// ── MicroResolve ──────────────────────────────────────────────────────────────

/// Entry point for MicroResolve.
///
/// Three modes:
///
///   # In-memory (no persistence)
///   engine = MicroResolve()
///
///   # Persistent — loads existing namespace dirs on startup
///   engine = MicroResolve(data_dir="/tmp/mr")
///
///   # Connected to a running MicroResolve server
///   engine = MicroResolve(
///       server_url="http://localhost:3001",
///       api_key="mr_xxx",           # optional
///       subscribe=["security"],     # namespaces to sync (omit / pass None to auto-subscribe to all)
///       tick_interval_secs=30,
///   )
#[pyclass]
struct MicroResolve {
    inner: Arc<microresolve_core::MicroResolve>,
}

#[pymethods]
impl MicroResolve {
    #[new]
    #[pyo3(signature = (
        data_dir=None,
        server_url=None,
        api_key=None,
        subscribe=None,
        tick_interval_secs=30,
        log_buffer_max=500,
    ))]
    fn new(
        data_dir: Option<String>,
        server_url: Option<String>,
        api_key: Option<String>,
        subscribe: Option<Vec<String>>,
        tick_interval_secs: u64,
        log_buffer_max: usize,
    ) -> PyResult<Self> {
        let server = server_url.map(|url| microresolve_core::ServerConfig {
            url,
            api_key,
            subscribe: subscribe.unwrap_or_default(),
            tick_interval_secs,
            log_buffer_max,
        });
        let config = microresolve_core::MicroResolveConfig {
            data_dir: data_dir.map(PathBuf::from),
            server,
            ..Default::default()
        };
        let engine = microresolve_core::MicroResolve::new(config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: Arc::new(engine) })
    }

    /// Return (or lazily create) a namespace handle.
    ///
    /// The namespace is created in-memory on first access. With `data_dir` set,
    /// an existing namespace directory is loaded automatically.
    fn namespace(&self, id: &str) -> Namespace {
        Namespace { engine: Arc::clone(&self.inner), id: id.to_string() }
    }

    /// IDs of all currently loaded namespaces.
    fn namespaces(&self) -> Vec<String> {
        self.inner.namespaces()
    }

    /// Force all dirty namespaces to disk. No-op when `data_dir` is not set.
    ///
    /// Raises `IOError` on failure.
    fn flush(&self) -> PyResult<()> {
        self.inner.flush()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        let ns = self.inner.namespaces();
        format!("MicroResolve(namespaces={ns:?})")
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Convert a Python `list[str]` or `dict[str, list[str]]` to `IntentSeeds`.
fn py_to_seeds(obj: &Bound<'_, PyAny>) -> PyResult<microresolve_core::IntentSeeds> {
    if let Ok(list) = obj.extract::<Vec<String>>() {
        return Ok(microresolve_core::IntentSeeds::from(list));
    }
    if let Ok(map) = obj.extract::<HashMap<String, Vec<String>>>() {
        return Ok(microresolve_core::IntentSeeds::from(map));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "phrases must be list[str] or dict[str, list[str]]"
    ))
}

// ── module ────────────────────────────────────────────────────────────────────

/// MicroResolve — pre-LLM reflex layer for intent classification, safety
/// filtering, and tool selection. Sub-millisecond, CPU-only, with continuous
/// learning from corrections.
#[pymodule]
fn microresolve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MicroResolve>()?;
    m.add_class::<Namespace>()?;
    m.add_class::<Match>()?;
    m.add_class::<IntentInfo>()?;
    Ok(())
}
