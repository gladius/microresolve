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

/// Options for `new Engine(options)`.
#[napi(object)]
pub struct EngineOptions {
    /// Path to persist namespace data. Each namespace is a sub-directory.
    /// Omit for in-memory only.
    pub data_dir: Option<String>,
    /// Server URL for connected mode, e.g. "http://localhost:3001".
    pub server_url: Option<String>,
    /// API key for the server (required when auth is enabled).
    pub api_key: Option<String>,
    /// Namespace IDs to subscribe to from the server.
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
    #[napi]
    pub fn correct(&self, query: String, wrong: String, right: String) -> Result<()> {
        let ns = self.engine.namespace(&self.id);
        ns.correct(&query, &wrong, &right)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Remove an intent and all its phrases.
    #[napi]
    pub fn remove_intent(&self, id: String) {
        self.engine.namespace(&self.id).remove_intent(&id);
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
    pub fn add_phrase(&self, intent_id: String, phrase: String, lang: Option<String>) -> PhraseResult {
        let result = self.engine.namespace(&self.id)
            .add_phrase(&intent_id, &phrase, lang.as_deref().unwrap_or("en"));
        PhraseResult {
            added: result.added,
            redundant: result.redundant,
            warning: result.warning,
        }
    }

    /// Flush this namespace to disk (no-op if `MicroResolve` has no `dataDir`).
    #[napi]
    pub fn flush(&self) -> Result<()> {
        self.engine.namespace(&self.id).flush()
            .map_err(|e| Error::from_reason(e.to_string()))
    }
}
