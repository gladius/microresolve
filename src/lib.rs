//! # ASV Router
//!
//! Model-free intent routing with incremental learning.
//! Sub-millisecond, no embeddings, no GPU, no neural network.
//!
//! ## Quick Start
//!
//! ```
//! use asv_router::Router;
//!
//! let mut router = Router::new();
//!
//! // Add intents with training phrases
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
//!
//! // Route a query
//! let result = router.route("I need to cancel something");
//! assert_eq!(result[0].id, "cancel_order");
//!
//! // Learn from user correction
//! router.learn("stop charging me", "cancel_order");
//!
//! // Now "stop charging me" routes correctly
//! let result = router.route("stop charging me");
//! assert_eq!(result[0].id, "cancel_order");
//! ```
//!
//! ## How It Works
//!
//! Each intent has a **dual-layer sparse vector**:
//! - **Phrase layer**: Generated from training phrases at setup time (immutable)
//! - **Learned layer**: Grows from user corrections (asymptotic toward 1.0)
//!
//! Routing tokenizes the query into unigrams + bigrams, looks up matching
//! intents via an inverted index, and scores by summing `max(phrase, learned)`
//! per term. The entire operation is a HashMap lookup — no matrix math,
//! no model inference.
//!
//! ## When to Use This
//!
//! - You have 10-1000 intents (support tickets, chatbot routing, command dispatch)
//! - You need sub-millisecond latency (edge, mobile, IoT)
//! - You want interpretable routing (see exactly why intent X was chosen)
//! - You want the system to learn from corrections without retraining
//! - You don't want to host an embedding model
//!
//! ## When NOT to Use This
//!
//! - You need semantic understanding ("stop charging me" won't match "cancel subscription" without training)
//! - You have 10K+ intents with heavy overlap
//! - You need deep semantic multilingual matching (CJK supported via Aho-Corasick, but coverage depends on phrase quality)

pub mod discovery;
pub mod import;
pub mod connect;
pub mod index;
pub mod multi;
pub mod phrase;
pub mod tokenizer;
pub mod types;
pub mod vector;

// Router method modules (each contains `impl Router { ... }`)
mod router_core;
mod router_intents;
mod router_routing;
mod router_learning;
mod router_metadata;
mod router_analytics;
mod router_similarity;
mod router_paraphrase;
mod router_situation;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use multi::{IntentRelation, MultiRouteOutput, MultiRouteResult};
pub use types::*;

use aho_corasick::AhoCorasick;
use index::InvertedIndex;
use std::collections::{HashMap, HashSet};
use tokenizer::{
    is_cjk, tokenize, training_to_terms, split_script_runs, generate_cjk_residual_bigrams,
    find_cjk_negated_regions, is_learnable_cjk_bigram, PositionedTerm, ScriptType,
};
use vector::LearnedVector;

/// Intent router with incremental learning.
///
/// The main entry point for the library. Manages intents, routing, and learning.
/// Supports both Latin and CJK scripts via a dual-path tokenization architecture:
/// Latin text uses whitespace tokenization; CJK text uses Aho-Corasick automaton
/// matching with character bigram fallback for novel terms.
pub struct Router {
    vectors: HashMap<String, LearnedVector>,
    index: InvertedIndex,
    /// Raw training phrases per intent, grouped by language code.
    /// Structure: { intent_id: { lang_code: [phrases] } }
    training: HashMap<String, HashMap<String, Vec<String>>>,
    top_k: usize,
    /// Aho-Corasick automaton for CJK term matching. None if no CJK terms exist.
    cjk_automaton: Option<AhoCorasick>,
    /// Pattern strings for the automaton. cjk_patterns[pattern_id] = term string.
    cjk_patterns: Vec<String>,
    /// When true, defers automaton rebuilds until `end_batch()` is called.
    batch_mode: bool,
    /// Tracks whether the automaton needs rebuilding (dirty during batch mode).
    cjk_dirty: bool,
    /// Intent type per intent (Action or Context). Default: Action.
    intent_types: HashMap<String, IntentType>,
    /// Human-readable description per intent.
    /// Used by LLM prompts to understand what the intent is about, even with zero phrases.
    descriptions: HashMap<String, String>,
    /// Opaque metadata per intent. User-defined key-value pairs.
    /// ASV stores and returns this data but never interprets it.
    metadata: HashMap<String, HashMap<String, Vec<String>>>,
    /// Co-occurrence counts: how often intent pairs fire together in route_multi.
    /// Key: (intent_a, intent_b) where a < b lexicographically. Value: count.
    co_occurrence: HashMap<(String, String), u32>,
    /// Temporal ordering: how often intent A appears BEFORE intent B in positional order.
    /// Key: (first_intent, second_intent) — NOT lexicographic, actual temporal order. Value: count.
    temporal_order: HashMap<(String, String), u32>,
    /// Full intent sequences observed in route_multi, for workflow/cluster discovery.
    /// Each entry is a sorted-by-position sequence of intent IDs from a single query.
    /// Capped at last 1000 observations to bound memory.
    intent_sequences: Vec<Vec<String>>,
    /// Paraphrase index: phrase (lowercase) -> (intent_id, weight).
    /// Multi-word phrase matching via Aho-Corasick automaton for dual-source confidence.
    paraphrase_phrases: HashMap<String, (String, f32)>,
    /// Aho-Corasick automaton for paraphrase matching.
    paraphrase_automaton: Option<AhoCorasick>,
    /// Pattern strings for paraphrase automaton.
    paraphrase_patterns: Vec<String>,
    /// Tracks whether paraphrase automaton needs rebuild (dirty during batch mode).
    paraphrase_dirty: bool,
    /// Monotonic version counter. Incremented on every mutation (learn, correct, add_intent, merge).
    version: u64,
    /// Maximum number of intents detected by route_multi. Default: 5.
    max_intents: usize,
    /// When true, write operations (add_intent, learn, correct) are blocked.
    /// Set in connected mode where the server manages state.
    connected: bool,
    /// Distributional similarity: term → Vec<(similar_term, cosine_score)>.
    /// Built from accumulated text (phrases + queries). Enables matching queries
    /// that use different words than the training phrases (e.g., "sent wrong" ≈ "return").
    similarity: HashMap<String, Vec<(String, f32)>>,
    /// Discount factor for similarity expansion. Default: 0.3.
    /// Lower = more conservative expansion, higher = more aggressive.
    expansion_discount: f32,
    /// Situation patterns: intent_id → [(pattern, weight)].
    /// Matched by direct substring search for state-description → action inference.
    /// Sits alongside the term index; scores are blended with SITUATION_ALPHA = 0.4.
    situation_patterns: HashMap<String, Vec<(String, f32)>>,
}


impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

// Router methods split across modules:
// router_core.rs      — constructor, config, persistence, accessors
// router_intents.rs   — intent CRUD, phrase guard
// router_routing.rs   — route, route_multi, route_best
// router_learning.rs  — learn, correct, reinforce, decay
// router_metadata.rs  — intent types, descriptions, metadata
// router_analytics.rs — co-occurrence, temporal, workflows
// router_similarity.rs — distributional similarity, merge
// router_paraphrase.rs — paraphrase index

