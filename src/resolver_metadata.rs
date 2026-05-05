//! Resolver: intent metadata reads and writes.
//!
//! Public API: `intent(id)` for reads, `update_intent(id, IntentEdit)` for writes.
//! Per-field accessors are not exposed — the consolidated forms cover every use case.

use crate::*;

impl Resolver {
    /// Read all metadata for an intent.
    ///
    /// Returns `None` if the intent does not exist. The returned `IntentInfo`
    /// is owned and independent of the Resolver's borrow.
    pub fn intent(&self, intent_id: &str) -> Option<IntentInfo> {
        if !self.training.contains_key(intent_id) {
            return None;
        }
        Some(IntentInfo {
            id: intent_id.to_string(),
            description: self
                .descriptions
                .get(intent_id)
                .cloned()
                .unwrap_or_default(),
            instructions: self
                .instructions
                .get(intent_id)
                .cloned()
                .unwrap_or_default(),
            persona: self.persona.get(intent_id).cloned().unwrap_or_default(),
            source: self.sources.get(intent_id).cloned(),
            target: self.targets.get(intent_id).cloned(),
            schema: self.schemas.get(intent_id).cloned(),
            guardrails: self.guardrails.get(intent_id).cloned().unwrap_or_default(),
            training: self.training.get(intent_id).cloned().unwrap_or_default(),
        })
    }

    /// Update one or more metadata fields on an existing intent.
    ///
    /// Each field of `IntentEdit` is `Option<T>`: `None` leaves the value
    /// alone, `Some(_)` overwrites it. Empty strings clear the corresponding
    /// field (instructions, persona). Returns `Err(IntentNotFound)` if
    /// `intent_id` does not exist.
    pub fn update_intent(&mut self, intent_id: &str, edit: IntentEdit) -> Result<(), Error> {
        if !self.training.contains_key(intent_id) {
            return Err(Error::IntentNotFound(intent_id.to_string()));
        }
        // Serialize edit before consuming its fields.
        let edit_json = serde_json::to_string(&edit).unwrap_or_default();
        if let Some(d) = edit.description {
            self.descriptions.insert(intent_id.to_string(), d);
        }
        if let Some(i) = edit.instructions {
            if i.is_empty() {
                self.instructions.remove(intent_id);
            } else {
                self.instructions.insert(intent_id.to_string(), i);
            }
        }
        if let Some(p) = edit.persona {
            if p.is_empty() {
                self.persona.remove(intent_id);
            } else {
                self.persona.insert(intent_id.to_string(), p);
            }
        }
        if let Some(s) = edit.source {
            self.sources.insert(intent_id.to_string(), s);
        }
        if let Some(t) = edit.target {
            self.targets.insert(intent_id.to_string(), t);
        }
        if let Some(s) = edit.schema {
            self.schemas.insert(intent_id.to_string(), s);
        }
        if let Some(g) = edit.guardrails {
            self.guardrails.insert(intent_id.to_string(), g);
        }
        // Emit metadata op (idempotent on replay).
        self.bump_with_ops(vec![crate::oplog::Op::IntentMetadataUpdated {
            id: intent_id.to_string(),
            edit_json,
        }]);
        Ok(())
    }
}
