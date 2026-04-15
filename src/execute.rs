//! Intent Programming runtime.
//!
//! Resolves which intent is active for a conversation turn, assembles
//! a focused system prompt from that intent's metadata, and provides
//! the context needed for LLM execution.
//!
//! # How it works
//!
//! Each intent carries metadata: instructions, guardrails, persona, tools.
//! At each conversation turn:
//!
//! 1. **Route** the user message through the scoring engine.
//! 2. **Resolve** the active intent: new detection → transition, nothing
//!    detected → continue with previous intent from history.
//! 3. **Assemble** a system prompt from the active intent's metadata.
//! 4. The caller sends `[system_prompt, ...history, user_message]` to an LLM.
//!
//! Instructions are loaded fresh per turn — never stored in history.
//! History carries only `role`, `content`, optional `intent` (on transitions),
//! and optional `remark` (LLM-generated audit trail).
//!
//! # Example
//!
//! ```ignore
//! let turn = execute::resolve_turn(
//!     &intent_graph,
//!     Some(&lexical_graph),
//!     &router,
//!     "I need to cancel my appointment",
//!     &history,
//!     0.3,  // threshold
//! );
//!
//! // turn.intent = "cancel_appointment"
//! // turn.is_transition = true
//! // turn.system_prompt = "Handle cancellation. Ask when..."
//! // turn.messages = [system, ...history, user]  (ready for LLM)
//! ```

use crate::scoring::{IntentGraph, LexicalGraph, RouteResult};
use crate::Router;

/// Result of resolving a conversation turn.
#[derive(Debug, Clone)]
pub struct TurnResult {
    /// The active intent for this turn.
    pub intent: Option<String>,
    /// True if ASV detected a new/different intent (entry point or topic switch).
    /// False if continuing the previous intent from history.
    pub is_transition: bool,
    /// Assembled system prompt from the active intent's metadata.
    pub system_prompt: String,
    /// Messages ready for LLM: [system_prompt, ...clean_history, user_query].
    /// History is stripped of `intent` and `remark` fields.
    pub messages: Vec<serde_json::Value>,
    /// Raw routing result (scores, disposition, ranked candidates).
    pub route_result: RouteResult,
}

/// Minimum score ratio for a detected intent to trigger a transition.
/// Prevents weak/accidental matches from breaking an active flow.
/// A detected intent must score at least this fraction of the scoring
/// engine's typical strong match to override the current flow.
const TRANSITION_THRESHOLD_RATIO: f32 = 1.5;

/// Resolve the active intent for a conversation turn.
///
/// - If ASV detects a strong, different intent → transition (new instructions loaded).
/// - If ASV detects nothing or a weak match → continue previous intent from history.
/// - If no history and no detection → no match.
///
/// Instructions are loaded fresh from `router`'s metadata — never from history.
pub fn resolve_turn(
    ig: &IntentGraph,
    l1: Option<&LexicalGraph>,
    router: &Router,
    query: &str,
    history: &[serde_json::Value],
    threshold: f32,
) -> TurnResult {
    // Route the query
    let route_result = ig.route(l1, query, threshold, 5);

    // Find the previous intent from history (last user message with intent tag)
    let previous_intent: Option<String> = history.iter().rev()
        .find_map(|msg| msg.get("intent").and_then(|v| v.as_str()).map(String::from));

    // Decide: transition or continue?
    let (intent, is_transition) = resolve_intent(
        &route_result, previous_intent.as_deref(), threshold,
    );

    // Assemble system prompt from active intent's metadata
    let system_prompt = match &intent {
        Some(id) => assemble_prompt(router, id),
        None => String::new(),
    };

    // Build LLM-ready messages
    let messages = build_messages(&system_prompt, history, query);

    TurnResult {
        intent,
        is_transition,
        system_prompt,
        messages,
        route_result,
    }
}

/// Decide whether to transition to a new intent or continue the current one.
///
/// Transitions only when:
/// - ASV detected an intent AND
/// - It's DIFFERENT from the current intent AND
/// - It scores strongly (above threshold × TRANSITION_THRESHOLD_RATIO)
///
/// This prevents weak matches (e.g., "Saturday" → pricing) from breaking
/// an active flow (cancel_appointment).
fn resolve_intent(
    route_result: &RouteResult,
    previous_intent: Option<&str>,
    threshold: f32,
) -> (Option<String>, bool) {
    if let Some((detected_id, detected_score)) = route_result.confirmed.first() {
        let is_different = previous_intent.map(|p| p != detected_id).unwrap_or(true);
        let is_strong = *detected_score >= threshold * TRANSITION_THRESHOLD_RATIO;
        let is_first_message = previous_intent.is_none();

        if is_first_message || (is_different && is_strong) {
            // Transition: new intent detected with strong confidence
            return (Some(detected_id.clone()), true);
        }
    }

    // Continue with previous intent (or no match if no history)
    match previous_intent {
        Some(id) => (Some(id.to_string()), false),
        None => (None, false),
    }
}

/// Assemble a system prompt from an intent's metadata blocks.
///
/// Blocks are concatenated in order: instructions, guardrails, persona, tools.
/// Only present blocks are included. Appends a remark instruction for audit trail.
pub fn assemble_prompt(router: &Router, intent_id: &str) -> String {
    let mut parts = Vec::new();

    if let Some(meta) = router.get_metadata(intent_id) {
        if let Some(v) = meta.get("instructions") {
            parts.push(v.join("\n"));
        }
        if let Some(v) = meta.get("guardrails") {
            let rules: Vec<String> = v.iter().map(|g| format!("- {}", g)).collect();
            parts.push(format!("\nConstraints:\n{}", rules.join("\n")));
        }
        if let Some(v) = meta.get("persona") {
            parts.push(format!("\nTone: {}", v.join(", ")));
        }
        if let Some(v) = meta.get("tools") {
            let tools: Vec<String> = v.iter().map(|t| format!("- {}", t)).collect();
            parts.push(format!("\nAvailable tools:\n{}", tools.join("\n")));
        }
    }

    // Fallback to description if no metadata
    if parts.is_empty() {
        let desc = router.get_description(intent_id);
        if !desc.is_empty() {
            parts.push(desc.to_string());
        }
    }

    if parts.is_empty() {
        return "You are a helpful assistant.".to_string();
    }

    let mut prompt = parts.join("\n");
    prompt.push_str("\n\nAfter your response, add on a new line: [REMARK: one sentence explaining your reasoning]");
    prompt
}

/// Build LLM-ready messages array: [system_prompt, ...clean_history, user_query].
///
/// Strips `intent` and `remark` fields from history — the LLM sees only
/// `role` and `content`. This keeps the LLM focused on the conversation,
/// not on routing metadata.
fn build_messages(
    system_prompt: &str,
    history: &[serde_json::Value],
    query: &str,
) -> Vec<serde_json::Value> {
    let mut messages = vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
    ];

    // Clean history: only role + content
    for msg in history {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
        messages.push(serde_json::json!({"role": role, "content": content}));
    }

    messages.push(serde_json::json!({"role": "user", "content": query}));
    messages
}

/// Extract `[REMARK: ...]` from LLM response text.
///
/// Returns `(clean_response, optional_remark)`. The remark is stripped from
/// the user-facing response and stored separately in history for audit.
pub fn extract_remark(text: &str) -> (String, Option<String>) {
    if let Some(idx) = text.rfind("[REMARK:") {
        let before = text[..idx].trim().to_string();
        let remark_raw = &text[idx..];
        let remark = remark_raw
            .trim_start_matches("[REMARK:")
            .trim_end_matches(']')
            .trim()
            .to_string();
        (before, if remark.is_empty() { None } else { Some(remark) })
    } else {
        (text.to_string(), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_remark_present() {
        let (text, remark) = extract_remark(
            "I'd be happy to help.\n\n[REMARK: Customer wants to cancel, asking for details]"
        );
        assert_eq!(text, "I'd be happy to help.");
        assert_eq!(remark.unwrap(), "Customer wants to cancel, asking for details");
    }

    #[test]
    fn extract_remark_absent() {
        let (text, remark) = extract_remark("Just a normal response.");
        assert_eq!(text, "Just a normal response.");
        assert!(remark.is_none());
    }

    #[test]
    fn resolve_intent_first_message() {
        let route = RouteResult {
            confirmed: vec![("cancel".to_string(), 2.0)],
            ranked: vec![],
            disposition: "confident".to_string(),
            has_negation: false,
        };
        let (intent, is_trans) = resolve_intent(&route, None, 0.3);
        assert_eq!(intent.unwrap(), "cancel");
        assert!(is_trans);
    }

    #[test]
    fn resolve_intent_continue_same() {
        let route = RouteResult {
            confirmed: vec![("cancel".to_string(), 0.5)],
            ranked: vec![],
            disposition: "confident".to_string(),
            has_negation: false,
        };
        // Same intent, weak score → continue, not transition
        let (intent, is_trans) = resolve_intent(&route, Some("cancel"), 0.3);
        assert_eq!(intent.unwrap(), "cancel");
        assert!(!is_trans);
    }

    #[test]
    fn resolve_intent_strong_transition() {
        let route = RouteResult {
            confirmed: vec![("booking".to_string(), 3.0)],
            ranked: vec![],
            disposition: "confident".to_string(),
            has_negation: false,
        };
        // Different intent, strong score → transition
        let (intent, is_trans) = resolve_intent(&route, Some("cancel"), 0.3);
        assert_eq!(intent.unwrap(), "booking");
        assert!(is_trans);
    }

    #[test]
    fn resolve_intent_weak_different_no_transition() {
        let route = RouteResult {
            confirmed: vec![("pricing".to_string(), 0.4)],
            ranked: vec![],
            disposition: "low_confidence".to_string(),
            has_negation: false,
        };
        // Different intent but WEAK score → don't transition, continue current
        let (intent, is_trans) = resolve_intent(&route, Some("cancel"), 0.3);
        assert_eq!(intent.unwrap(), "cancel");
        assert!(!is_trans);
    }

    #[test]
    fn resolve_intent_no_detection_continue() {
        let route = RouteResult {
            confirmed: vec![],
            ranked: vec![],
            disposition: "no_match".to_string(),
            has_negation: false,
        };
        let (intent, is_trans) = resolve_intent(&route, Some("cancel"), 0.3);
        assert_eq!(intent.unwrap(), "cancel");
        assert!(!is_trans);
    }

    #[test]
    fn build_messages_strips_metadata() {
        let history = vec![
            serde_json::json!({"role": "user", "content": "cancel please", "intent": "cancel"}),
            serde_json::json!({"role": "assistant", "content": "Sure.", "remark": "processing"}),
        ];
        let msgs = build_messages("System prompt", &history, "tomorrow");
        assert_eq!(msgs.len(), 4); // system + 2 history + user
        // History messages should NOT have intent/remark
        assert!(msgs[1].get("intent").is_none());
        assert!(msgs[2].get("remark").is_none());
    }
}
