//! Router: intent types, descriptions, instructions, persona, and import metadata.

use crate::*;

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

    /// Set LLM instructions for an intent (what the LLM should do when this intent fires).
    pub fn set_instructions(&mut self, intent_id: &str, instructions: &str) {
        self.require_local();
        if instructions.is_empty() {
            self.instructions.remove(intent_id);
        } else {
            self.instructions.insert(intent_id.to_string(), instructions.to_string());
        }
    }

    /// Get LLM instructions for an intent.
    pub fn get_instructions(&self, intent_id: &str) -> &str {
        self.instructions.get(intent_id).map(|s| s.as_str()).unwrap_or("")
    }

    /// Set LLM persona for an intent (tone and voice).
    pub fn set_persona(&mut self, intent_id: &str, persona: &str) {
        self.require_local();
        if persona.is_empty() {
            self.persona.remove(intent_id);
        } else {
            self.persona.insert(intent_id.to_string(), persona.to_string());
        }
    }

    /// Get LLM persona for an intent.
    pub fn get_persona(&self, intent_id: &str) -> &str {
        self.persona.get(intent_id).map(|s| s.as_str()).unwrap_or("")
    }

    /// Set the import source for an intent.
    pub fn set_source(&mut self, intent_id: &str, source: IntentSource) {
        self.require_local();
        self.sources.insert(intent_id.to_string(), source);
    }

    /// Get the import source for an intent.
    pub fn get_source(&self, intent_id: &str) -> Option<&IntentSource> {
        self.sources.get(intent_id)
    }

    /// Set the execution target for an intent.
    pub fn set_target(&mut self, intent_id: &str, target: IntentTarget) {
        self.require_local();
        self.targets.insert(intent_id.to_string(), target);
    }

    /// Get the execution target for an intent.
    pub fn get_target(&self, intent_id: &str) -> Option<&IntentTarget> {
        self.targets.get(intent_id)
    }

    /// Set the tool/API schema for an intent (JSON Schema format).
    pub fn set_schema(&mut self, intent_id: &str, schema: serde_json::Value) {
        self.require_local();
        self.schemas.insert(intent_id.to_string(), schema);
    }

    /// Get the tool/API schema for an intent.
    pub fn get_schema(&self, intent_id: &str) -> Option<&serde_json::Value> {
        self.schemas.get(intent_id)
    }

    /// Set guardrail rules for an intent.
    pub fn set_guardrails(&mut self, intent_id: &str, rules: Vec<String>) {
        self.require_local();
        self.guardrails.insert(intent_id.to_string(), rules);
    }

    /// Get guardrail rules for an intent.
    pub fn get_guardrails(&self, intent_id: &str) -> &[String] {
        self.guardrails.get(intent_id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Set the namespace model registry.
    pub fn set_namespace_models(&mut self, models: Vec<NamespaceModel>) {
        self.namespace_models = models;
    }

    /// Get the namespace model registry.
    pub fn get_namespace_models(&self) -> &[NamespaceModel] {
        &self.namespace_models
    }
}
