//! Router: intent types, descriptions, and metadata.

use crate::*;
use std::collections::HashMap;

impl Router {
    pub fn set_intent_type(&mut self, intent_id: &str, intent_type: IntentType) {
        self.require_local();
        self.intent_types.insert(intent_id.to_string(), intent_type);
    }

    /// Get the type of an intent. Defaults to Action if not set.
    pub fn get_intent_type(&self, intent_id: &str) -> IntentType {
        self.intent_types.get(intent_id).copied().unwrap_or(IntentType::Action)
    }

    /// Set a human-readable description for an intent.
    pub fn set_description(&mut self, intent_id: &str, description: &str) {
        self.descriptions.insert(intent_id.to_string(), description.to_string());
    }

    /// Get the description of an intent. Returns empty string if not set.
    pub fn get_description(&self, intent_id: &str) -> &str {
        self.descriptions.get(intent_id).map(|s| s.as_str()).unwrap_or("")
    }

    /// Set opaque metadata for an intent.
    ///
    /// ASV stores and returns this data but never interprets it.
    /// The application layer decides what to do with it.
    ///
    /// ```
    /// use asv_router::Router;
    ///
    /// let mut router = Router::new();
    /// router.add_intent("cancel_order", &["cancel my order"]);
    /// router.set_metadata("cancel_order", "context_intents", vec!["check_balance".into(), "track_order".into()]);
    /// ```
    pub fn set_metadata(&mut self, intent_id: &str, key: &str, values: Vec<String>) {
        self.require_local();
        self.metadata
            .entry(intent_id.to_string())
            .or_default()
            .insert(key.to_string(), values);
    }

    /// Get all metadata for an intent.
    pub fn get_metadata(&self, intent_id: &str) -> Option<&HashMap<String, Vec<String>>> {
        self.metadata.get(intent_id)
    }

    /// Get a specific metadata key for an intent.
    pub fn get_metadata_key(&self, intent_id: &str, key: &str) -> Option<&Vec<String>> {
        self.metadata.get(intent_id)?.get(key)
    }

    // Record co-occurrence for a set of intents detected together.
    // Call after route_multi to track which intents fire together.

}
