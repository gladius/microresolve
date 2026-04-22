//! # MicroResolve
//!
//! Sub-millisecond intent resolution. No embeddings, no GPU, no retraining.
//! Scoring layers (morphology graph + scoring index) are in `src/scoring.rs`.
//!
//! ## Quick Start
//!
//! ```
//! use microresolve::Router;
//!
//! let mut router = Router::new();
//! router.add_intent("cancel_order", &["cancel my order", "stop my order"]);
//! router.add_intent("track_order",  &["where is my package", "track my order"]);
//!
//! let matches = router.resolve("I want to cancel", 0.3, 1.5);
//! assert_eq!(matches[0].0, "cancel_order");
//! ```

pub mod scoring;
pub mod entity;
pub mod import;
pub mod ngram;
pub mod phrase;
pub mod tokenizer;
pub mod types;

#[cfg(feature = "connect")]
pub mod connect;

// Router method modules (each contains `impl Router { ... }`)
mod router_core;
mod router_intents;
mod router_learning;
mod router_metadata;
mod router_persist;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use types::*;

use std::collections::HashMap;

/// Intent registry with incremental phrase learning.
///
/// Stores intent definitions, training phrases, types, descriptions, and metadata.
/// L0/L1/L2 routing layers are embedded directly — call `index_phrase` to update all
/// layers atomically, and `route` to run the full pipeline.
pub struct Router {
    /// Typo corrector: character n-gram index built from all known vocabulary.
    pub(crate) l0: crate::ngram::NgramIndex,
    /// Morphology graph: normalizes inflections, abbreviations, and synonyms before scoring.
    pub(crate) l1: crate::scoring::LexicalGraph,
    /// Scoring index: IDF-weighted word→intent associations with anti-Hebbian inhibition.
    pub(crate) l2: crate::scoring::IntentIndex,
    /// Raw training phrases per intent, grouped by language code.
    /// Structure: { intent_id: { lang_code: [phrases] } }
    /// This is the canonical intent list — `intent_ids()` reads from here.
    training: HashMap<String, HashMap<String, Vec<String>>>,
    /// Intent type per intent (Action or Context). Default: Action.
    intent_types: HashMap<String, IntentType>,
    /// Human-readable description per intent.
    /// Used by LLM prompts for Hebbian bootstrap context.
    descriptions: HashMap<String, String>,
    /// LLM instructions per intent (what to do when this intent fires).
    instructions: HashMap<String, String>,
    /// LLM persona per intent (tone and voice).
    persona: HashMap<String, String>,
    /// Import provenance: where this intent's definition came from.
    sources: HashMap<String, IntentSource>,
    /// Execution target: where to send when this intent fires.
    targets: HashMap<String, IntentTarget>,
    /// Tool/API schema for imported intents (JSON Schema format).
    schemas: HashMap<String, serde_json::Value>,
    /// Guardrail rules for this intent.
    guardrails: HashMap<String, Vec<String>>,
    /// Monotonic version counter. Incremented on every mutation.
    version: u64,
    /// When true, write operations are blocked (connected/read-only mode).
    connected: bool,
    /// Distributional similarity: term → [(similar_term, score)].
    /// Built from accumulated text. Used for analysis, not routing.
    similarity: HashMap<String, Vec<(String, f32)>>,
    /// Human-readable display name for this namespace.
    namespace_name: String,
    /// Human-readable description of this namespace.
    namespace_description: String,
    /// User-defined model registry for this namespace.
    namespace_models: Vec<NamespaceModel>,
    /// Default routing threshold for this namespace.
    /// Used as fallback when /api/route_multi requests omit `threshold`.
    /// `None` means "no override, use the compile-time default."
    /// `Some(0.0)` is a valid (degenerate) setting — accept all matches.
    namespace_default_threshold: Option<f32>,
    /// Per-namespace entity-detection configuration: which built-in patterns
    /// are enabled and any custom entities defined for this namespace.
    /// `None` means "entity layer disabled for this namespace."
    /// Persisted in `_entities.json`.
    namespace_entity_config: Option<crate::entity::EntityConfig>,
    /// Cached, pre-built EntityLayer for this namespace's config.
    /// Rebuilt only when set_entity_config is called or namespace is loaded.
    /// Avoids the ~2.5ms regex-compile cost on every routing call.
    cached_entity_layer: Option<crate::entity::EntityLayer>,
    /// Descriptions for domain prefixes (e.g., "billing" in "billing:cancel_order").
    domain_descriptions: HashMap<String, String>,
    // Legacy config fields kept for API compatibility.
    top_k: usize,
    max_intents: usize,
    batch_mode: bool,
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

// Router methods split across modules:
// router_core.rs      — constructor, config, persistence, accessors, route()
// router_intents.rs   — intent CRUD, phrase management
// router_learning.rs  — add_phrase_auto, correct
// router_metadata.rs  — intent types, descriptions, instructions, persona, sources, targets
// router_persist.rs   — directory-based persistence
