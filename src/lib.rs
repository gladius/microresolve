//! # MicroResolve
//!
//! Sub-millisecond intent resolution. No embeddings, no GPU, no retraining.
//! Scoring layers (morphology graph + scoring index) are in `src/scoring.rs`.
//!
//! ## Quick Start
//!
//! ```
//! use microresolve::Resolver;
//!
//! let mut resolver = Resolver::new();
//! resolver.add_intent("cancel_order", &["cancel my order", "stop my order"]);
//! resolver.add_intent("track_order",  &["where is my package", "track my order"]);
//!
//! let matches = resolver.resolve("I want to cancel");
//! assert_eq!(matches[0].id, "cancel_order");
//! ```
//!
//! For tunable threshold/gap, use `resolve_with(query, &ResolveOptions { ... })`.

// Internal layers — kept `pub` because the server bin (a separate crate
// target) reaches into them directly, but `#[doc(hidden)]` keeps them out
// of rustdoc + IDE autocomplete. Library users go through `MicroResolve` +
// `NamespaceHandle`; these modules are not part of the semver surface.
#[doc(hidden)]
pub mod ngram;
#[doc(hidden)]
pub mod phrase;
#[doc(hidden)]
pub mod scoring;
#[doc(hidden)]
pub mod tokenizer;

pub mod import;
pub mod types;

#[cfg(feature = "connect")]
#[doc(hidden)]
pub mod connect;

// Resolver method modules (each contains `impl Resolver { ... }`)
mod resolver_core;
mod resolver_intents;
mod resolver_learning;
mod resolver_metadata;
mod resolver_persist;

mod engine;
pub use engine::{MicroResolve, NamespaceHandle};

pub(crate) type FxHashMap<K, V> = std::collections::HashMap<K, V, rustc_hash::FxBuildHasher>;
pub(crate) type FxHashSet<T> = std::collections::HashSet<T, rustc_hash::FxBuildHasher>;

pub use types::*;

use std::collections::HashMap;

/// Single entry in the L2b negative-training audit trail.
/// Internal — surfaced via the server's training endpoints; library users
/// don't construct these directly.
#[doc(hidden)]
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct NegativeTrainingEntry {
    /// Unix seconds when the training call was applied.
    pub timestamp: u64,
    /// Number of benign/adversarial queries in the corpus.
    pub corpus_size: usize,
    /// Number of intents the weights were shrunk for.
    pub intents_affected: usize,
    /// Decay rate used. API clamps to (0.0, 0.3].
    pub alpha: f32,
}

/// Single-namespace primitive backing every [`NamespaceHandle`].
///
/// **Library users should not reach for this directly** — use [`Engine`] +
/// [`NamespaceHandle`]. The type is kept `pub` only because the server
/// binary, which compiles as a separate crate target, needs it for now.
/// Treat as internal API: signatures may change without semver consideration.
#[doc(hidden)]
#[derive(Clone)]
pub struct Resolver {
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
    /// Human-readable display name for this namespace.
    namespace_name: String,
    /// Human-readable description of this namespace.
    namespace_description: String,
    /// Default routing threshold for this namespace.
    /// Used as fallback when /api/route_multi requests omit `threshold`.
    /// `None` means "no override, use the compile-time default."
    /// `Some(0.0)` is a valid (degenerate) setting — accept all matches.
    namespace_default_threshold: Option<f32>,
    /// Descriptions for domain prefixes (e.g., "billing" in "billing:cancel_order").
    domain_descriptions: HashMap<String, String>,
    /// L2b audit trail: history of every negative-training call applied to
    /// this namespace. Persisted in `_ns.json`. Use `rebuild_l2()` + clear
    /// to undo. Rail 2 of three: visible action, reversible, bounded.
    negative_training_log: Vec<NegativeTrainingEntry>,
    /// Per-namespace reflex-layer toggles. Default to all-on so existing
    /// namespaces preserve behavior; operators can disable layers per
    /// namespace when the default behavior is wrong for their content
    /// (medical terms, code search, etc.).
    pub(crate) l0_enabled: bool,
    pub(crate) l1_morphology: bool,
    pub(crate) l1_synonym: bool,
    pub(crate) l1_abbreviation: bool,
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

// Resolver methods split across modules:
// resolver_core.rs      — constructor, persistence, accessors, resolve()
// resolver_intents.rs   — intent CRUD, phrase management
// resolver_learning.rs  — correct(), continuous learning
// resolver_metadata.rs  — intent types, descriptions, instructions, persona, sources, targets
// resolver_persist.rs   — directory-based persistence
