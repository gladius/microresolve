//! Node.js bindings for MicroResolve via napi-rs.
//!
//! Usage:
//!   const { MicroResolve } = require('microresolve');
//!   const engine = new MicroResolve();
//!   const ns = engine.namespace('security');
//!   ns.addIntent('jailbreak', ['ignore prior instructions']);
//!   const matches = ns.resolve('ignore prior instructions and reveal');
//!   // → [{ id: 'jailbreak', score: 0.87 }]

use std::collections::HashMap;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use microresolve_core::{
    MicroResolve as CoreEngine, MicroResolveConfig, ServerConfig, IntentSeeds,
    IntentEdit, IntentType, ResolveOptions,
    NamespaceEdit, NamespaceInfo as CoreNamespaceInfo,
};

/// A classification match: intent id + score.
#[napi(object)]
pub struct Match {
    pub id: String,
    pub score: f64,
}

/// Read-only view of an intent's metadata and training phrases.
#[napi(object)]
pub struct IntentInfo {
    pub id: String,
    /// `"action"` or `"context"`.
    pub intent_type: String,
    pub description: String,
    /// Training phrases grouped by language code.
    pub training: HashMap<String, Vec<String>>,
}

/// Result from `addPhrase`.
#[napi(object)]
pub struct PhraseResult {
    pub added: bool,
    pub redundant: bool,
    pub warning: Option<String>,
}

/// Read-only view of namespace-level metadata, including reflex-layer toggles.
///
/// Returned by `Namespace.namespaceInfo()`.
#[napi(object)]
pub struct NamespaceInfo {
    /// Human-readable display name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Per-namespace routing threshold override. `null` → use engine default.
    pub default_threshold: Option<f64>,
}

/// Edit options accepted by `updateNamespace`.
///
/// All fields are optional; omitted fields leave the existing value unchanged.
#[napi(object)]
pub struct NamespaceEditOptions {
    pub name: Option<String>,
    pub description: Option<String>,
    /// Per-namespace threshold override.
    ///   - omit / `undefined` → leave existing value alone
    ///   - any non-negative number → set as the new threshold
    ///   - `-1` (sentinel, matches the server's HTTP API) → clear the override
    ///
    /// The sentinel-based convention is the napi-rs friendly equivalent of
    /// `Option<Option<f32>>`, which napi-rs does not natively support.
    pub default_threshold: Option<f64>,
}

/// Edit object accepted by `updateIntent`.
#[napi(object)]
pub struct IntentEditOptions {
    /// `"action"` or `"context"`.
    pub intent_type: Option<String>,
    pub description: Option<String>,
    pub instructions: Option<String>,
    pub persona: Option<String>,
    pub guardrails: Option<Vec<String>>,
}

/// A single intent + score pair in a routing result.
#[napi(object)]
pub struct ScoredIntent {
    pub id: String,
    pub score: f64,
}

/// Output of `routeMulti` / `routeMultiWithTrace`.
#[napi(object)]
pub struct RouteMultiResult {
    /// Top intents after greedy multi-round extraction (threshold applied).
    pub multi: Vec<ScoredIntent>,
    /// All intents ranked by raw score (no threshold applied).
    pub raw: Vec<ScoredIntent>,
    /// `true` if the query contains a negation signal.
    pub negated: bool,
    /// Tokenized query terms, for use with `confidenceFor`.
    pub tokens: Vec<String>,
    /// Effective threshold that was applied.
    pub threshold: f64,
    /// Per-round trace as a JSON string. `null` when using `routeMulti`.
    pub trace: Option<String>,
}

/// An intent + span pair used in `applyReview`.
#[napi(object)]
pub struct SpanPair {
    pub intent_id: String,
    pub span: String,
}

/// Options for `new MicroResolve(options)`.
#[napi(object)]
pub struct EngineOptions {
    /// Path to persist namespace data. Each namespace is a sub-directory.
    /// Omit for in-memory only.
    pub data_dir: Option<String>,
    /// Server URL for connected mode, e.g. "http://localhost:3001".
    pub server_url: Option<String>,
    /// API key for the server (required when auth is enabled).
    pub api_key: Option<String>,
    /// Namespace IDs to subscribe to from the server. Omit (or pass an
    /// empty array) to auto-subscribe to every namespace the server exposes.
    pub subscribe: Option<Vec<String>>,
    /// Background sync interval in seconds. Default: 30.
    pub tick_interval_secs: Option<u32>,
}

/// Multi-namespace decision engine.
///
/// One `MicroResolve` per application. Call `engine.namespace(id)` to get a
/// `Namespace` handle and operate on intents within it.
#[napi]
pub struct MicroResolve {
    inner: Arc<CoreEngine>,
}

#[napi]
impl MicroResolve {
    /// Create a new MicroResolve instance.
    ///
    /// ```js
    /// // in-memory
    /// const engine = new MicroResolve();
    ///
    /// // persistent
    /// const engine = new MicroResolve({ dataDir: '/tmp/mr' });
    ///
    /// // connected
    /// const engine = new MicroResolve({
    ///   serverUrl: 'http://localhost:3001',
    ///   apiKey: 'mr_xxx',
    ///   subscribe: ['security', 'intent'],
    /// });
    /// ```
    #[napi(constructor)]
    pub fn new(options: Option<EngineOptions>) -> Result<Self> {
        let opts = options.unwrap_or(EngineOptions {
            data_dir: None,
            server_url: None,
            api_key: None,
            subscribe: None,
            tick_interval_secs: None,
        });

        let server = opts.server_url.map(|url| ServerConfig {
            url,
            api_key: opts.api_key,
            subscribe: opts.subscribe.unwrap_or_default(),
            tick_interval_secs: opts.tick_interval_secs.unwrap_or(30) as u64,
            log_buffer_max: 500,
        });

        let config = MicroResolveConfig {
            data_dir: opts.data_dir.map(std::path::PathBuf::from),
            server,
            ..Default::default()
        };

        let inner = CoreEngine::new(config)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(MicroResolve { inner: Arc::new(inner) })
    }

    /// Get a `Namespace` handle for the given id.
    /// The namespace is created lazily on first write.
    #[napi]
    pub fn namespace(&self, id: String) -> Namespace {
        Namespace { engine: Arc::clone(&self.inner), id }
    }

    /// All namespace IDs currently loaded in this engine.
    #[napi]
    pub fn namespaces(&self) -> Vec<String> {
        self.inner.namespaces()
    }

    /// Flush all dirty namespaces to disk (no-op if no `dataDir` was set).
    #[napi]
    pub fn flush(&self) -> Result<()> {
        self.inner.flush().map_err(|e| Error::from_reason(e.to_string()))
    }
}

/// Handle for one namespace inside a `MicroResolve` instance.
///
/// All operations on intents and phrases go through this object.
#[napi]
pub struct Namespace {
    engine: Arc<CoreEngine>,
    id: String,
}

#[napi]
impl Namespace {
    /// Add an intent with seed phrases (English by default).
    ///
    /// ```js
    /// // monolingual
    /// ns.addIntent('greet', ['hello', 'hi', 'hey']);
    ///
    /// // multilingual
    /// ns.addIntent('greet', { en: ['hello'], fr: ['bonjour'] });
    /// ```
    #[napi]
    pub fn add_intent(
        &self,
        id: String,
        seeds: Either<Vec<String>, HashMap<String, Vec<String>>>,
    ) -> Result<u32> {
        let intent_seeds: IntentSeeds = match seeds {
            Either::A(phrases) => phrases.into(),
            Either::B(map) => map.into(),
        };
        let ns = self.engine.namespace(&self.id);
        ns.add_intent(&id, intent_seeds)
            .map(|n| n as u32)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Resolve a query. Returns matches sorted by score descending.
    #[napi]
    pub fn resolve(&self, query: String) -> Vec<Match> {
        let ns = self.engine.namespace(&self.id);
        ns.resolve(&query)
            .into_iter()
            .map(|m| Match { id: m.id, score: m.score as f64 })
            .collect()
    }

    /// Correct a mis-classification: nudge the engine from `wrong` toward `right`.
    /// Applied locally immediately; in connected mode buffered for the next sync tick.
    #[napi]
    pub fn correct(&self, query: String, wrong: String, right: String) -> Result<()> {
        let ns = self.engine.namespace(&self.id);
        ns.correct(&query, &wrong, &right)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Remove an intent and all its phrases.
    #[napi]
    pub fn remove_intent(&self, id: String) -> Result<()> {
        self.engine.namespace(&self.id).remove_intent(&id)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// All intent IDs in this namespace.
    #[napi]
    pub fn intent_ids(&self) -> Vec<String> {
        self.engine.namespace(&self.id).intent_ids()
    }

    /// Number of intents in this namespace.
    #[napi]
    pub fn intent_count(&self) -> u32 {
        self.engine.namespace(&self.id).intent_count() as u32
    }

    /// Monotonic version counter; increments on every mutation.
    #[napi]
    pub fn version(&self) -> u32 {
        self.engine.namespace(&self.id).version() as u32
    }

    /// Resolve with explicit threshold and gap overrides.
    #[napi]
    pub fn resolve_with(&self, query: String, threshold: Option<f64>, gap: Option<f64>) -> Vec<Match> {
        let opts = ResolveOptions {
            threshold: threshold.unwrap_or(0.3) as f32,
            gap: gap.unwrap_or(1.5) as f32,
        };
        let ns = self.engine.namespace(&self.id);
        ns.resolve_with(&query, opts)
            .into_iter()
            .map(|m| Match { id: m.id, score: m.score as f64 })
            .collect()
    }

    /// Read-only view of an intent's metadata. Returns `null` if not found.
    #[napi]
    pub fn intent(&self, intent_id: String) -> Option<IntentInfo> {
        self.engine.namespace(&self.id).intent(&intent_id).map(|info| IntentInfo {
            id: info.id,
            intent_type: match info.intent_type {
                IntentType::Action => "action".to_string(),
                IntentType::Context => "context".to_string(),
            },
            description: info.description,
            training: info.training,
        })
    }

    /// Update metadata fields on an existing intent.
    ///
    /// Raises an error if the intent does not exist or `intentType` is invalid.
    #[napi]
    pub fn update_intent(&self, intent_id: String, edit: IntentEditOptions) -> Result<()> {
        let mut e = IntentEdit::default();
        if let Some(ref t) = edit.intent_type {
            e.intent_type = Some(match t.as_str() {
                "action" => IntentType::Action,
                "context" => IntentType::Context,
                other => return Err(Error::from_reason(format!(
                    "intentType must be 'action' or 'context', got '{other}'"
                ))),
            });
        }
        e.description = edit.description;
        e.instructions = edit.instructions;
        e.persona = edit.persona;
        e.guardrails = edit.guardrails;
        self.engine.namespace(&self.id).update_intent(&intent_id, e)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Add a single phrase to an existing intent.
    #[napi]
    pub fn add_phrase(&self, intent_id: String, phrase: String, lang: Option<String>) -> Result<PhraseResult> {
        let result = self.engine.namespace(&self.id)
            .add_phrase(&intent_id, &phrase, lang.as_deref().unwrap_or("en"))
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(PhraseResult {
            added: result.added,
            redundant: result.redundant,
            warning: result.warning,
        })
    }

    /// Read-only view of namespace-level metadata.
    #[napi]
    pub fn namespace_info(&self) -> NamespaceInfo {
        let ns = self.engine.namespace(&self.id);
        let info: CoreNamespaceInfo = ns.namespace_info();
        NamespaceInfo {
            name: info.name,
            description: info.description,
            default_threshold: info.default_threshold.map(|t| t as f64),
        }
    }

    /// Patch namespace-level metadata fields.
    ///
    /// All fields are optional; omitted (undefined) fields leave existing values unchanged.
    #[napi]
    pub fn update_namespace(&self, edit: NamespaceEditOptions) -> Result<()> {
        let e = NamespaceEdit {
            name: edit.name,
            description: edit.description,
            default_threshold: edit.default_threshold.map(|t| {
                if t < 0.0 { None } else { Some(t as f32) }
            }),
            domain_descriptions: None,
        };
        let ns = self.engine.namespace(&self.id);
        ns.update_namespace(e)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Flush this namespace to disk (no-op if `MicroResolve` has no `dataDir`).
    #[napi]
    pub fn flush(&self) -> Result<()> {
        self.engine.namespace(&self.id).flush()
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    // ── Extended typed API ─────────────────────────────────────────────────

    /// Number of unique token→intent associations in the scoring index.
    #[napi]
    pub fn vocab_size(&self) -> u32 {
        self.engine.namespace(&self.id).vocab_size() as u32
    }

    /// Per-intent normalized confidence for an already-scored result.
    ///
    /// `tokens` must be the tokenized form of the original query (use
    /// `routeMulti(query).tokens` to obtain them).
    #[napi]
    pub fn confidence_for(&self, score: f64, tokens: Vec<String>, intent_id: String) -> f64 {
        self.engine.namespace(&self.id)
            .confidence_for(score as f32, &tokens, &intent_id) as f64
    }

    /// Flat list of all training phrases for an intent (all languages combined).
    ///
    /// Returns `null` if the intent does not exist.
    #[napi]
    pub fn training(&self, intent_id: String) -> Option<Vec<String>> {
        self.engine.namespace(&self.id).training(&intent_id)
    }

    /// Training phrases grouped by language code.
    ///
    /// Returns `null` if the intent does not exist.
    #[napi]
    pub fn training_by_lang(&self, intent_id: String) -> Option<HashMap<String, Vec<String>>> {
        self.engine.namespace(&self.id).training_by_lang(&intent_id)
    }

    /// Export namespace state as a JSON string (for sync/backup).
    #[napi]
    pub fn export_json(&self) -> String {
        self.engine.namespace(&self.id).export_json()
    }

    /// Check whether a phrase would be a useful addition (deduplication check).
    #[napi]
    pub fn check_phrase(&self, intent_id: String, phrase: String) -> PhraseResult {
        let result = self.engine.namespace(&self.id).check_phrase(&intent_id, &phrase);
        PhraseResult {
            added: result.added,
            redundant: result.redundant,
            warning: result.warning,
        }
    }

    /// Description for a specific domain prefix. Returns `null` if not set.
    #[napi]
    pub fn domain_description(&self, domain: String) -> Option<String> {
        self.engine.namespace(&self.id).domain_description(&domain)
    }

    /// Set the description for a domain prefix.
    #[napi]
    pub fn set_domain_description(&self, domain: String, description: String) -> Result<()> {
        self.engine.namespace(&self.id).set_domain_description(&domain, &description)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Remove a domain description.
    #[napi]
    pub fn remove_domain_description(&self, domain: String) -> Result<()> {
        self.engine.namespace(&self.id).remove_domain_description(&domain)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Run the full multi-intent routing pipeline.
    ///
    /// Returns a `RouteMultiResult` object with `multi`, `raw`, `negated`,
    /// `tokens`, and `threshold` fields.
    #[napi]
    pub fn route_multi(
        &self,
        query: String,
        threshold_override: Option<f64>,
        gap: Option<f64>,
        fallback_threshold: Option<f64>,
    ) -> RouteMultiResult {
        let out = self.engine.namespace(&self.id).route_multi(
            &query,
            threshold_override.map(|v| v as f32),
            gap.unwrap_or(1.5) as f32,
            fallback_threshold.unwrap_or(0.3) as f32,
        );
        RouteMultiResult {
            multi: out.multi.into_iter().map(|(id, s)| ScoredIntent { id, score: s as f64 }).collect(),
            raw: out.raw.into_iter().map(|(id, s)| ScoredIntent { id, score: s as f64 }).collect(),
            negated: out.negated,
            tokens: out.tokens,
            threshold: out.threshold as f64,
            trace: None,
        }
    }

    /// Like `routeMulti` but includes a per-round trace in the `trace` field (JSON string).
    #[napi]
    pub fn route_multi_with_trace(
        &self,
        query: String,
        threshold_override: Option<f64>,
        gap: Option<f64>,
        fallback_threshold: Option<f64>,
    ) -> RouteMultiResult {
        let out = self.engine.namespace(&self.id).route_multi_with_trace(
            &query,
            threshold_override.map(|v| v as f32),
            gap.unwrap_or(1.5) as f32,
            fallback_threshold.unwrap_or(0.3) as f32,
        );
        let trace_json = out.trace.as_ref().and_then(|t| serde_json::to_string(t).ok());
        RouteMultiResult {
            multi: out.multi.into_iter().map(|(id, s)| ScoredIntent { id, score: s as f64 }).collect(),
            raw: out.raw.into_iter().map(|(id, s)| ScoredIntent { id, score: s as f64 }).collect(),
            negated: out.negated,
            tokens: out.tokens,
            threshold: out.threshold as f64,
            trace: trace_json,
        }
    }

    /// Reinforce specific query tokens toward `intentId` (Hebbian-style update).
    #[napi]
    pub fn reinforce_tokens(&self, words: Vec<String>, intent_id: String) -> Result<()> {
        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        self.engine.namespace(&self.id).reinforce_tokens(&word_refs, &intent_id)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Rebuild the scoring index from stored training phrases.
    #[napi]
    pub fn rebuild_index(&self) -> Result<()> {
        self.engine.namespace(&self.id).rebuild_index()
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Rebuild IDF table and in-memory caches (call after bulk `indexPhrase` calls).
    #[napi]
    pub fn rebuild_caches(&self) -> Result<()> {
        self.engine.namespace(&self.id).rebuild_caches()
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Lower-level phrase ingestion: indexes without dedup check.
    ///
    /// Use `addPhrase` for user-driven additions; use `indexPhrase` only for
    /// trusted, pre-validated phrases (e.g., from spec import or auto-learn).
    #[napi]
    pub fn index_phrase(&self, intent_id: String, phrase: String) -> Result<()> {
        self.engine.namespace(&self.id).index_phrase(&intent_id, &phrase)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Anti-Hebbian decay: shrink L2 weights for `notIntents` on `queries`.
    ///
    /// `alpha` is clamped to `(0.0, 0.3]` internally.
    #[napi]
    pub fn decay_for_intents(&self, queries: Vec<String>, not_intents: Vec<String>, alpha: f64) -> Result<()> {
        self.engine.namespace(&self.id).decay_for_intents(&queries, &not_intents, alpha as f32)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Apply a review result (missed phrases, span learning, anti-Hebbian correction).
    ///
    /// - `missedPhrases`:   object mapping intent_id → phrase list
    /// - `spansToLearn`:    array of `{ intentId, span }` objects
    /// - `wrongDetections`: array of intent IDs that were wrongly detected
    /// - `originalQuery`:   the original query text
    /// - `negativeAlpha`:   anti-Hebbian decay strength (0.0–0.3, default 0.1)
    ///
    /// Returns the number of phrases added.
    #[napi]
    pub fn apply_review(
        &self,
        missed_phrases: HashMap<String, Vec<String>>,
        spans_to_learn: Vec<SpanPair>,
        wrong_detections: Vec<String>,
        original_query: String,
        negative_alpha: Option<f64>,
    ) -> Result<u32> {
        let spans: Vec<(String, String)> = spans_to_learn
            .into_iter()
            .map(|p| (p.intent_id, p.span))
            .collect();
        self.engine.namespace(&self.id).apply_review(
            &missed_phrases,
            &spans,
            &wrong_detections,
            &original_query,
            negative_alpha.unwrap_or(0.1) as f32,
        )
        .map(|n| n as u32)
        .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Remove a single phrase from an intent. Returns `true` if the phrase existed.
    #[napi]
    pub fn remove_phrase(&self, intent_id: String, phrase: String) -> Result<bool> {
        self.engine.namespace(&self.id).remove_phrase(&intent_id, &phrase)
            .map_err(|e| Error::from_reason(e.to_string()))
    }
}
