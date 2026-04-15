//! # ASV Router
//!
//! Intent registry and IDF-based scoring.
//! Training phrases and intent metadata are stored here; scoring layers
//! (L1 LexicalGraph + L2 IntentGraph) are in `src/scoring.rs`.
//!
//! ## Quick Start (server mode)
//!
//! ```
//! use asv_router::Router;
//!
//! let mut router = Router::new();
//!
//! // Register intents with training phrases (used for Hebbian bootstrap)
//! router.add_intent("cancel_order", &[
//!     "cancel my order",
//!     "I want to cancel",
//!     "stop my order",
//! ]);
//! router.add_intent("track_order", &[
//!     "where is my package",
//!     "track my order",
//!     "shipping status",
//! ]);
//! ```

pub mod scoring;
/// Backward compat alias
pub mod hebbian { pub use crate::scoring::*; }
pub mod discovery;
pub mod import;
pub mod connect;
pub mod ngram;
pub mod phrase;
pub mod tokenizer;
pub mod types;

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
/// Routing is handled externally by the Hebbian L1+L2 system.
/// This struct is the training-data store that feeds Hebbian bootstrap.
pub struct Router {
    /// Raw training phrases per intent, grouped by language code.
    /// Structure: { intent_id: { lang_code: [phrases] } }
    /// This is the canonical intent list — `intent_ids()` reads from here.
    training: HashMap<String, HashMap<String, Vec<String>>>,
    /// Intent type per intent (Action or Context). Default: Action.
    intent_types: HashMap<String, IntentType>,
    /// Human-readable description per intent.
    /// Used by LLM prompts for Hebbian bootstrap context.
    descriptions: HashMap<String, String>,
    /// Opaque metadata per intent. User-defined key-value pairs.
    metadata: HashMap<String, HashMap<String, Vec<String>>>,
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
// router_core.rs      — constructor, config, persistence, accessors
// router_intents.rs   — intent CRUD, phrase guard
// router_learning.rs  — learn, correct (phrase storage only)
// router_metadata.rs  — intent types, descriptions, metadata
// router_similarity.rs — distributional similarity
// router_persist.rs   — directory-based persistence
// router_situation.rs — situation pattern storage
