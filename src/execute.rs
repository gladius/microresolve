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

    // Option H: check the last assistant message for a `next_intent` directive
    // (set when the previous turn emitted `→ intent_id`). This is a forced
    // handoff that overrides ASV routing for this turn.
    let last_assistant = history.iter().rev()
        .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant"));
    let forced_next: Option<String> = last_assistant
        .and_then(|m| m.get("next_intent")).and_then(|v| v.as_str())
        .map(String::from);
    let briefing: Option<String> = last_assistant
        .and_then(|m| m.get("context")).and_then(|v| v.as_str())
        .map(String::from);

    // Decide the active intent.
    //
    // Design commitment: ASV is ENTRY-ONLY. It fires only when there's no
    // previous intent in history (first user message). Once a conversation
    // is in progress, intent transitions are driven solely by the intent
    // programming layer via `→ <intent_id>` handoff directives from the
    // previous assistant turn. ASV does not re-route mid-conversation.
    //
    // Priority:
    //   1. Forced handoff (previous assistant emitted `next_intent`)
    //   2. Continue previous intent (if any)
    //   3. ASV routing (only when history has no intent — entry point)
    let (intent, is_transition) = if let Some(id) = forced_next {
        // Validate the forced intent exists in router
        if router.intent_ids().iter().any(|x| x == &id) {
            (Some(id), true)
        } else {
            // Hallucinated intent name: ignore, fall through to normal logic
            if let Some(prev) = previous_intent.as_deref() {
                (Some(prev.to_string()), false)
            } else {
                resolve_intent(&route_result, None, threshold)
            }
        }
    } else if let Some(prev) = previous_intent.as_deref() {
        // Conversation in progress: stay in the current intent. No ASV hijack.
        (Some(prev.to_string()), false)
    } else {
        // Entry point: no history → ASV decides
        resolve_intent(&route_result, None, threshold)
    };

    // Assemble system prompt from active intent's metadata.
    // Only inject briefing on the FIRST turn after handoff (when transitioning into the new intent).
    let system_prompt = match &intent {
        Some(id) => {
            let b = if is_transition { briefing.as_deref() } else { None };
            assemble_prompt(router, id, b)
        }
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

/// Returns true if the intent is marked as a fact-returning node.
/// Fact intents have `metadata.mode = ["fact"]` and their instructions ARE
/// the literal answer (or a parameterized template).
pub fn is_fact_intent(router: &Router, intent_id: &str) -> bool {
    router.get_metadata(intent_id)
        .and_then(|m| m.get("mode"))
        .map(|v| v.iter().any(|s| s == "fact"))
        .unwrap_or(false)
}

/// List all fact-mode intents in the namespace as `id: one-line description`,
/// for inclusion in conversational intents' system prompts.
pub fn list_fact_intents(router: &Router) -> Vec<(String, String)> {
    router.intent_ids().into_iter()
        .filter(|id| is_fact_intent(router, id))
        .map(|id| {
            let desc = router.get_description(&id).to_string();
            (id, desc)
        })
        .collect()
}

/// Assemble a system prompt from an intent's metadata blocks.
///
/// Blocks are concatenated in order: instructions, guardrails, persona, tools.
/// Only present blocks are included. Appends a remark instruction for audit trail.
/// If `briefing` is provided, prepends it as a "carried-forward context" block.
/// Appends Option H handoff convention.
pub fn assemble_prompt(router: &Router, intent_id: &str, briefing: Option<&str>) -> String {
    let mut parts = Vec::new();

    if let Some(b) = briefing {
        if !b.trim().is_empty() {
            parts.push(format!(
                "Briefing from previous intent: {}\n\nUse this briefing — do not re-ask the user for information already in it.",
                b.trim()
            ));
        }
    }

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

    if parts.is_empty() || (parts.len() == 1 && briefing.is_some()) {
        let desc = router.get_description(intent_id);
        if !desc.is_empty() {
            parts.push(desc.to_string());
        }
    }

    if parts.is_empty() {
        return "You are a helpful assistant.".to_string();
    }

    let mut prompt = parts.join("\n");
    // Fact-intent registry: list available fact-returning intents the LLM can look up.
    let facts = list_fact_intents(router);
    if !facts.is_empty() {
        prompt.push_str("\n\nFACT INTENTS AVAILABLE (look these up for accurate, authoritative answers):\n");
        for (id, desc) in &facts {
            let line = if desc.is_empty() {
                format!("- {}\n", id)
            } else {
                format!("- {}: {}\n", id, desc)
            };
            prompt.push_str(&line);
        }
        prompt.push_str(
            "\nWhen you need one of these facts to answer accurately, emit on its own line:\n\
             lookup: <intent_id>\n\
             You may emit multiple lookups (each on its own line). The system will \
             execute them and return results to you in the next round, then you \
             produce the final reply. Do NOT invent values for facts available via lookup.\n"
        );
    }

    prompt.push_str(
        "\n\nHANDOFF CONVENTION: When you have finished this intent's job and want to hand off to another intent, \
         end your response with these two lines exactly (each on its own line):\n\
         → <intent_id>\n\
         context: <one short paragraph briefing the next intent on what you have learned>\n\n\
         If you are NOT handing off, do NOT write `→` at all and do NOT write any meta-commentary \
         about whether you are handing off. Just continue the conversation naturally.\n\n\
         After your reply, on a new line (BEFORE any handoff lines), add: \
         [REMARK: one sentence explaining your reasoning]"
    );
    prompt
}

/// Extract `lookup: <intent_id>` directives from LLM response.
/// Returns (cleaned_text, list_of_lookup_ids).
pub fn extract_lookups(text: &str) -> (String, Vec<String>) {
    let mut ids = Vec::new();
    let mut keep = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("lookup:") {
            let id = rest.trim().trim_matches(|c: char| c == '`' || c == '*').to_string();
            if !id.is_empty() { ids.push(id); }
            continue;
        }
        keep.push(line);
    }
    (keep.join("\n").trim().to_string(), ids)
}

/// Get the literal body of a fact intent (its `instructions` field, or
/// description as fallback). Used when a lookup directive resolves a fact intent.
pub fn fact_body(router: &Router, intent_id: &str) -> String {
    if let Some(meta) = router.get_metadata(intent_id) {
        if let Some(v) = meta.get("instructions") {
            return v.join("\n");
        }
    }
    router.get_description(intent_id).to_string()
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

/// Extract Option H handoff directives `→ <intent_id>` and `context: <text>`
/// from the trailing lines of an LLM response.
///
/// Returns `(clean_response, optional_handoff_intent, optional_context)`.
/// Stripped from user-facing text. Either or both directives may be absent.
/// The arrow can be `→` or `->`.
pub fn extract_handoff(text: &str) -> (String, Option<String>, Option<String>) {
    let mut handoff: Option<String> = None;
    let mut context: Option<String> = None;
    let mut keep_lines: Vec<&str> = Vec::new();

    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix('→').or_else(|| t.strip_prefix("->")) {
            let id = rest.trim().trim_matches(|c: char| c == '`' || c == '*').to_string();
            if !id.is_empty() { handoff = Some(id); }
            continue;
        }
        if let Some(rest) = t.to_lowercase().strip_prefix("context:") {
            // strip_prefix on lowercase gives wrong byte offsets; use case-insensitive find
            let _ = rest;
            if let Some(colon) = t.find(':') {
                let body = t[colon+1..].trim().to_string();
                if !body.is_empty() { context = Some(body); }
            }
            continue;
        }
        keep_lines.push(line);
    }

    let clean = keep_lines.join("\n").trim().to_string();
    (clean, handoff, context)
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
