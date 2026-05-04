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

// ── IntentMatch ───────────────────────────────────────────────────────────────

/// A single intent in a resolve result.
#[pyclass(get_all)]
#[derive(Clone)]
struct IntentMatch {
    pub id: String,
    pub score: f32,
    /// Normalized confidence in [0,1].
    pub confidence: f32,
    /// Score band: `"High"`, `"Medium"`, or `"Low"`.
    pub band: String,
}

#[pymethods]
impl IntentMatch {
    fn __repr__(&self) -> String {
        format!(
            "IntentMatch(id={:?}, score={:.4}, confidence={:.3}, band={:?})",
            self.id, self.score, self.confidence, self.band
        )
    }
}

// ── ResolveResult ─────────────────────────────────────────────────────────────

/// Output of `Namespace.resolve()`.
#[pyclass(get_all)]
#[derive(Clone)]
struct ResolveResult {
    /// Ranked descending by score. May be empty.
    pub intents: Vec<IntentMatch>,
    /// `"Confident"`, `"LowConfidence"`, or `"NoMatch"`.
    pub disposition: String,
}

#[pymethods]
impl ResolveResult {
    fn __repr__(&self) -> String {
        format!(
            "ResolveResult(disposition={:?}, intents={})",
            self.disposition,
            self.intents.len()
        )
    }
}

fn core_result_to_py(r: microresolve_core::ResolveResult) -> ResolveResult {
    let disposition = match r.disposition {
        microresolve_core::Disposition::Confident => "Confident",
        microresolve_core::Disposition::LowConfidence => "LowConfidence",
        microresolve_core::Disposition::NoMatch => "NoMatch",
    }
    .to_string();
    let intents = r
        .intents
        .into_iter()
        .map(|m| IntentMatch {
            id: m.id,
            score: m.score,
            confidence: m.confidence,
            band: match m.band {
                microresolve_core::Band::High => "High",
                microresolve_core::Band::Medium => "Medium",
                microresolve_core::Band::Low => "Low",
            }
            .to_string(),
        })
        .collect();
    ResolveResult { intents, disposition }
}

// ── ResolveTrace ──────────────────────────────────────────────────────────────

/// Diagnostic trace from `Namespace.resolve_with_trace()`.
#[pyclass(get_all)]
struct ResolveTrace {
    pub tokens: Vec<String>,
    /// All scored intents as (id, score) tuples.
    pub all_scores: Vec<(String, f32)>,
    /// Per-round trace as a JSON string.
    pub multi_round_trace: String,
    pub negated: bool,
    pub threshold_applied: f32,
}

#[pymethods]
impl ResolveTrace {
    fn __repr__(&self) -> String {
        format!(
            "ResolveTrace(tokens={:?}, negated={}, threshold={})",
            self.tokens, self.negated, self.threshold_applied
        )
    }
}

// ── NamespaceInfo ─────────────────────────────────────────────────────────────

/// Read-only view of namespace-level metadata, including reflex-layer toggles.
///
/// Returned by `Namespace.namespace_info()`.
#[pyclass(get_all, from_py_object)]
#[derive(Clone)]
struct NamespaceInfo {
    /// Human-readable display name.
    name: String,
    /// Human-readable description.
    description: String,
    /// Per-namespace routing threshold override. `None` → use engine default.
    default_threshold: Option<f32>,
    /// Per-namespace voting-token gate override. `None` → use engine default (1, disabled).
    default_min_voting_tokens: Option<u32>,
}

#[pymethods]
impl NamespaceInfo {
    fn __repr__(&self) -> String {
        format!(
            "NamespaceInfo(name={:?}, default_threshold={:?}, default_min_voting_tokens={:?})",
            self.name, self.default_threshold, self.default_min_voting_tokens
        )
    }
}

fn ns_info_to_py(info: microresolve_core::NamespaceInfo) -> NamespaceInfo {
    NamespaceInfo {
        name: info.name,
        description: info.description,
        default_threshold: info.default_threshold,
        default_min_voting_tokens: info.default_min_voting_tokens,
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
    /// Returns a `ResolveResult` with `intents` (list of `IntentMatch`) and
    /// `disposition` (`"Confident"`, `"LowConfidence"`, or `"NoMatch"`).
    fn resolve(&self, query: &str) -> ResolveResult {
        let handle = self.engine.namespace(&self.id);
        core_result_to_py(handle.resolve(query))
    }

    /// Like `resolve` but also returns a `ResolveTrace` with per-round diagnostics.
    fn resolve_with_trace(&self, query: &str) -> (ResolveResult, ResolveTrace) {
        let handle = self.engine.namespace(&self.id);
        let (result, trace) = handle.resolve_with_trace(query);
        let trace_json = serde_json::to_string(&trace.multi_round_trace).unwrap_or_default();
        (
            core_result_to_py(result),
            ResolveTrace {
                tokens: trace.tokens,
                all_scores: trace.all_scores,
                multi_round_trace: trace_json,
                negated: trace.negated,
                threshold_applied: trace.threshold_applied,
            },
        )
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
    fn remove_intent(&self, intent_id: &str) -> PyResult<()> {
        self.engine.namespace(&self.id).remove_intent(intent_id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Add a single phrase to an existing intent.
    ///
    /// Returns a dict with `added` (bool), `redundant` (bool), `warning` (str | None).
    fn add_phrase<'py>(&self, py: Python<'py>, intent_id: &str, phrase: &str, lang: Option<&str>) -> PyResult<Bound<'py, PyDict>> {
        let result = self.engine.namespace(&self.id).add_phrase(intent_id, phrase, lang.unwrap_or("en"))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
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

    /// Read-only view of namespace-level metadata.
    fn namespace_info(&self) -> NamespaceInfo {
        let handle = self.engine.namespace(&self.id);
        ns_info_to_py(handle.namespace_info())
    }

    /// Patch namespace-level metadata fields.
    ///
    /// Accepts kwargs: `name`, `description`, `default_threshold` (float or None
    /// to clear). Fields not passed are left unchanged.
    ///
    /// Raises `ValueError` on invalid input.
    fn update_namespace(&self, edit: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut e = microresolve_core::NamespaceEdit::default();
        if let Some(v) = edit.get_item("name")? {
            e.name = Some(v.extract::<String>()?);
        }
        if let Some(v) = edit.get_item("description")? {
            e.description = Some(v.extract::<String>()?);
        }
        if let Some(v) = edit.get_item("default_threshold")? {
            if v.is_none() {
                e.default_threshold = Some(None);
            } else {
                e.default_threshold = Some(Some(v.extract::<f32>()?));
            }
        }
        let handle = self.engine.namespace(&self.id);
        handle.update_namespace(e)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    // ── Extended typed API ─────────────────────────────────────────────────

    /// Number of unique token→intent associations in the scoring index.
    fn vocab_size(&self) -> usize {
        self.engine.namespace(&self.id).vocab_size()
    }

    /// Per-intent normalized confidence for an already-scored result.
    ///
    /// `tokens` must be the tokenized form of the original query (use
    /// `resolve_with_trace(query)[1].tokens` to obtain them).
    fn confidence_for(&self, score: f32, tokens: Vec<String>, intent_id: &str) -> f32 {
        self.engine.namespace(&self.id).confidence_for(score, &tokens, intent_id)
    }

    /// Flat list of all training phrases for an intent (all languages combined).
    ///
    /// Returns `None` if the intent does not exist.
    fn training(&self, intent_id: &str) -> Option<Vec<String>> {
        self.engine.namespace(&self.id).training(intent_id)
    }

    /// Training phrases grouped by language code.
    ///
    /// Returns `None` if the intent does not exist.
    fn training_by_lang(&self, intent_id: &str) -> Option<HashMap<String, Vec<String>>> {
        self.engine.namespace(&self.id).training_by_lang(intent_id)
    }

    /// Export namespace state as a JSON string (for sync/backup).
    fn export_json(&self) -> String {
        self.engine.namespace(&self.id).export_json()
    }

    /// Check whether a phrase would be a useful addition (deduplication check).
    ///
    /// Returns a dict with `added` (bool), `redundant` (bool), `warning` (str | None).
    fn check_phrase<'py>(&self, py: Python<'py>, intent_id: &str, phrase: &str) -> PyResult<Bound<'py, PyDict>> {
        let result = self.engine.namespace(&self.id).check_phrase(intent_id, phrase);
        let d = PyDict::new(py);
        d.set_item("added", result.added)?;
        d.set_item("redundant", result.redundant)?;
        d.set_item("warning", result.warning)?;
        Ok(d)
    }

    /// Description for a specific domain prefix. Returns `None` if not set.
    fn domain_description(&self, domain: &str) -> Option<String> {
        self.engine.namespace(&self.id).domain_description(domain)
    }

    /// Set the description for a domain prefix.
    fn set_domain_description(&self, domain: &str, description: &str) -> PyResult<()> {
        self.engine.namespace(&self.id).set_domain_description(domain, description)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Remove a domain description.
    fn remove_domain_description(&self, domain: &str) -> PyResult<()> {
        self.engine.namespace(&self.id).remove_domain_description(domain)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }


    /// Reinforce specific query tokens toward `intent_id` (Hebbian-style update).
    fn reinforce_tokens(&self, words: Vec<String>, intent_id: &str) -> PyResult<()> {
        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        self.engine.namespace(&self.id).reinforce_tokens(&word_refs, intent_id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Rebuild the scoring index from stored training phrases.
    fn rebuild_index(&self) -> PyResult<()> {
        self.engine.namespace(&self.id).rebuild_index()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// EXPERIMENTAL: set voting-token gate. 1 = disabled. 2+ = require N distinct
    /// query tokens to back an intent before full-strength scoring.
    fn set_min_voting_tokens(&self, min: u32) -> PyResult<()> {
        self.engine.namespace(&self.id).set_min_voting_tokens(min)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Rebuild IDF table and in-memory caches (call after bulk `index_phrase` calls).
    fn rebuild_caches(&self) -> PyResult<()> {
        self.engine.namespace(&self.id).rebuild_caches()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Lower-level phrase ingestion: indexes without dedup check.
    ///
    /// Use `add_phrase` for user-driven additions; use `index_phrase` only for
    /// trusted, pre-validated phrases (e.g., from spec import or auto-learn).
    fn index_phrase(&self, intent_id: &str, phrase: &str) -> PyResult<()> {
        self.engine.namespace(&self.id).index_phrase(intent_id, phrase)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Anti-Hebbian decay: shrink L2 weights for `not_intents` on `queries`.
    ///
    /// `alpha` is clamped to `(0.0, 0.3]` internally.
    fn decay_for_intents(&self, queries: Vec<String>, not_intents: Vec<String>, alpha: f32) -> PyResult<()> {
        self.engine.namespace(&self.id).decay_for_intents(&queries, &not_intents, alpha)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Apply a review result (missed phrases, span learning, anti-Hebbian correction).
    ///
    /// Parameters:
    ///   - `missed_phrases`:   dict mapping intent_id → list of missed phrases
    ///   - `spans_to_learn`:   list of `[intent_id, span_text]` pairs
    ///   - `wrong_detections`: list of intent IDs that were wrongly detected
    ///   - `original_query`:   the original query text
    ///   - `negative_alpha`:   anti-Hebbian decay strength (0.0–0.3)
    ///
    /// Returns the number of phrases added.
    #[pyo3(signature = (missed_phrases, spans_to_learn, wrong_detections, original_query, negative_alpha=0.1))]
    fn apply_review(
        &self,
        missed_phrases: HashMap<String, Vec<String>>,
        spans_to_learn: Vec<(String, String)>,
        wrong_detections: Vec<String>,
        original_query: &str,
        negative_alpha: f32,
    ) -> PyResult<usize> {
        self.engine.namespace(&self.id).apply_review(
            &missed_phrases,
            &spans_to_learn,
            &wrong_detections,
            original_query,
            negative_alpha,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Remove a single phrase from an intent. Returns `True` if the phrase existed.
    fn remove_phrase(&self, intent_id: &str, phrase: &str) -> PyResult<bool> {
        self.engine.namespace(&self.id).remove_phrase(intent_id, phrase)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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
    m.add_class::<IntentMatch>()?;
    m.add_class::<ResolveResult>()?;
    m.add_class::<ResolveTrace>()?;
    m.add_class::<NamespaceInfo>()?;
    m.add_class::<IntentInfo>()?;
    Ok(())
}
