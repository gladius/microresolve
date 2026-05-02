//! Delta-sync op log.
//!
//! Every public `Resolver` mutation records one or more `Op` variants here.
//! The server hands these to delta-aware clients in place of a full export.

use std::collections::HashMap;

/// Maximum number of op entries retained in the oplog (FIFO eviction).
pub const OPLOG_MAX: usize = 1000;

/// A single mutation event in the oplog.
///
/// Variants use `#[serde(tag = "kind", rename_all = "snake_case")]` so the
/// JSON representation carries a `"kind"` field (e.g. `"intent_added"`).
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Op {
    // ── Structural (replay via existing dedup-aware public methods) ──────────
    /// New intent registered. Replay calls `add_intent`, which is idempotent.
    IntentAdded {
        id: String,
        phrases_by_lang: HashMap<String, Vec<String>>,
        intent_type: Option<String>,
        description: Option<String>,
        instructions: Option<String>,
        persona: Option<String>,
    },

    /// Intent fully removed. Replay = `remove_intent`. Idempotent.
    IntentRemoved { id: String },

    /// New phrase added to an existing intent. Replay = `add_phrase`. Idempotent.
    PhraseAdded {
        intent_id: String,
        phrase: String,
        lang: String,
    },

    /// Phrase removed. Sibling `WeightUpdates` carries the re-derived weights.
    PhraseRemoved { intent_id: String, phrase: String },

    // ── Numeric (carry POST-VALUES, idempotent on replay) ───────────────────
    /// Hebbian weight changes. Triplet: (token, intent_id, post_weight).
    /// Replay overwrites each listed weight to the post-value.
    WeightUpdates { changes: Vec<(String, String, f32)> },

    // ── Metadata (idempotent, last-write-wins) ──────────────────────────────
    /// Metadata edit on one intent. Replay = `update_intent(id, edit)`.
    IntentMetadataUpdated { id: String, edit_json: String },

    /// Namespace-level metadata edit. Replay = `update_namespace(edit)`.
    NamespaceMetadataUpdated { edit_json: String },

    /// Domain description set (`description = Some`) or removed (`None`).
    DomainDescription {
        domain: String,
        description: Option<String>,
    },
}
