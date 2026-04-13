//! LLM integration: call_llm, phrase pipeline, full 3-turn review.

use axum::http::StatusCode;
use std::collections::HashMap;
use asv_router::Router;
use crate::state::*;

/// Turn 1 (Judge): id + description only. No phrases — descriptions are the interface contract.
/// Flat cost regardless of how many phrases have been learned.
pub fn build_intent_labels(router: &Router) -> String {
    let mut ids = router.intent_ids();
    ids.sort();
    ids.iter().map(|id| {
        let desc = router.get_description(id);
        if desc.is_empty() {
            format!("- {} [NO DESCRIPTION — cannot classify reliably]", id)
        } else {
            format!("- {} ({})", id, desc)
        }
    }).collect::<Vec<_>>().join("\n")
}

/// Turn 2 context: phrases for specific intents, capped to avoid token bloat.
/// Shows most-recently-added phrases first (they reflect what was learned, not just bootstrap seeds).
fn intent_phrases_context(router: &Router, intent_ids: &[String], cap: usize) -> String {
    intent_ids.iter().map(|id| {
        let desc = router.get_description(id);
        let phrases = router.get_training(id).unwrap_or_default();
        let shown: Vec<&String> = phrases.iter().rev().take(cap).collect();
        let desc_str = if desc.is_empty() { String::new() } else { format!(" ({})", desc) };
        format!("  {}{}: {:?}", id, desc_str, shown)
    }).collect::<Vec<_>>().join("\n")
}

/// Legacy: kept for any callers outside the review pipeline.
pub fn build_intent_descriptions(router: &Router) -> String {
    build_intent_labels(router)
}

/// Extract JSON from LLM response that may be wrapped in markdown code fences.
/// Handles both object `{}` and array `[]` top-level values.
pub fn extract_json(text: &str) -> &str {
    let trimmed = text.trim();

    // First: try to extract from a ```json ... ``` code fence — most reliable signal.
    // The LLM often wraps JSON in ``` blocks; extract whatever is between the fences.
    if let Some(fence_start) = trimmed.find("```") {
        // skip past the opening fence line (```json\n or ```\n)
        let after_fence = &trimmed[fence_start + 3..];
        let content_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
        let content = &after_fence[content_start..];
        // trim at the closing fence
        let content = if let Some(end) = content.find("```") {
            content[..end].trim()
        } else {
            content.trim()
        };
        // Now extract array or object from the fenced content
        let arr = content.find('[');
        let obj = content.find('{');
        let use_array = match (arr, obj) {
            (Some(a), Some(o)) => a <= o,
            (Some(_), None)    => true,
            _                  => false,
        };
        if use_array {
            if let (Some(s), Some(e)) = (content.find('['), content.rfind(']')) {
                return &content[s..=e];
            }
        }
        if let (Some(s), Some(e)) = (content.find('{'), content.rfind('}')) {
            return &content[s..=e];
        }
    }

    // Fallback: scan for the last `[` ... `]` or `{` ... `}` that look like JSON
    // (the last `]` / `}` is more reliable than the first `[` / `{` in preamble text)
    let last_array_end  = trimmed.rfind(']');
    let last_object_end = trimmed.rfind('}');
    match (last_array_end, last_object_end) {
        (Some(ae), Some(oe)) if ae > oe => {
            // array ends later — extract array
            if let (Some(s), Some(e)) = (trimmed.find('['), trimmed.rfind(']')) {
                return &trimmed[s..=e];
            }
        }
        (None, Some(_)) | (Some(_), Some(_)) => {
            if let (Some(s), Some(e)) = (trimmed.find('{'), trimmed.rfind('}')) {
                return &trimmed[s..=e];
            }
        }
        (Some(_), None) => {
            if let (Some(s), Some(e)) = (trimmed.find('['), trimmed.rfind(']')) {
                return &trimmed[s..=e];
            }
        }
        _ => {}
    }
    trimmed
}

/// Call LLM for text generation. Supports two formats:
/// - "anthropic": Anthropic Messages API (/v1/messages)
/// - "openai" (or anything else): OpenAI Chat Completions format (/v1/chat/completions)
///   Covers: OpenAI, Ollama, Groq, Together, DeepSeek, vLLM, LM Studio, any compatible endpoint.
pub async fn call_llm(
    state: &ServerState,
    prompt: &str,
    max_tokens: u32,
) -> Result<String, (StatusCode, String)> {
    let key = state.llm_key.as_ref().ok_or_else(|| {
        (StatusCode::SERVICE_UNAVAILABLE, "LLM_API_KEY not set. Add it to .env file.".to_string())
    })?;

    let provider = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "anthropic".to_string());
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());
    let url = std::env::var("LLM_API_URL").unwrap_or_else(|_| {
        if provider == "anthropic" {
            "https://api.anthropic.com/v1/messages".to_string()
        } else {
            "https://api.openai.com/v1/chat/completions".to_string()
        }
    });

    let messages = serde_json::json!([{"role": "user", "content": prompt}]);

    let resp = if provider == "anthropic" {
        // Anthropic Messages API
        let body = serde_json::json!({
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        });
        state.http
            .post(&url)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
    } else {
        // OpenAI Chat Completions format (works with OpenAI, Ollama, Groq, etc.)
        let body = serde_json::json!({
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        });
        state.http
            .post(&url)
            .header("Authorization", format!("Bearer {}", key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
    }.map_err(|e| (StatusCode::BAD_GATEWAY, format!("LLM request failed: {}", e)))?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return Err((StatusCode::BAD_GATEWAY, format!("LLM API {}: {}", status, text)));
    }

    let data: serde_json::Value = resp.json().await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Bad response: {}", e)))?;

    // Extract text: Anthropic uses content[0].text, OpenAI uses choices[0].message.content
    if provider == "anthropic" {
        data["content"][0]["text"]
            .as_str()
            .map(|s| s.trim().to_string())
            .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No text in response".to_string()))
    } else {
        data["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.trim().to_string())
            .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No text in response".to_string()))
    }
}

// --- Shared phrase pipeline: guard + one LLM retry ---

/// Result of the phrase pipeline.
#[derive(Debug, Clone, serde::Serialize)]
#[derive(serde::Deserialize)]
pub struct PhrasePipelineResult {
    /// Phrases that were successfully added: (intent_id, phrase)
    pub added: Vec<(String, String)>,
    /// Phrases permanently blocked after all retry rounds: (intent_id, phrase, reason)
    pub blocked: Vec<(String, String, String)>,
    /// How many phrases initially hit the collision guard
    pub initially_blocked: usize,
    /// How many of those were rescued by retry (added via alternative phrase)
    pub recovered_by_retry: usize,
    /// LLM-suggested alternatives for user-edited phrases (when auto_apply=false)
    pub suggestions: Vec<(String, String)>, // (intent_id, suggested_phrase)
}

/// Shared phrase pipeline used by all flows that add phrases.
///
/// 1. Runs each phrase through `add_phrase_checked`
/// 2. Collects blocked phrases with reasons
/// 3. If LLM available: one retry with collision info
/// 4. auto_apply_retry=true: silently apply alternatives (for LLM-generated phrases)
///    auto_apply_retry=false: return alternatives as suggestions (for user-edited phrases)
pub async fn phrase_pipeline(
    state: &AppState,
    app_id: &str,
    phrases_by_intent: &HashMap<String, Vec<String>>,
    auto_apply_retry: bool,
    lang: &str,
) -> PhrasePipelineResult {
    let mut added = Vec::new();
    let mut blocked_for_retry: Vec<(String, String, String)> = Vec::new(); // (intent, phrase, reason)
    let mut blocked_final = Vec::new();

    // Step 1: Try all phrases through the guard
    {
        let mut routers = state.routers.write().unwrap();
        if let Some(router) = routers.get_mut(app_id) {
            for (intent_id, phrases) in phrases_by_intent {
                for phrase in phrases {
                    let s = phrase.trim();
                    if s.is_empty() { continue; }
                    let result = router.add_phrase_checked(intent_id, s, lang);
                    if result.added {
                        added.push((intent_id.clone(), s.to_string()));
                    } else if !result.conflicts.is_empty() {
                        // Collision — candidate for retry
                        let reason = result.conflicts.iter()
                            .map(|c| format!("'{}' conflicts with {}", c.term, c.competing_intent))
                            .collect::<Vec<_>>().join("; ");
                        blocked_for_retry.push((intent_id.clone(), s.to_string(), reason));
                    } else if result.redundant {
                        // Redundant — silently skip, not worth retrying
                    } else if let Some(warning) = result.warning {
                        blocked_final.push((intent_id.clone(), s.to_string(), warning));
                    }
                }
            }
            if !added.is_empty() {
                maybe_persist(state, app_id, router);
            }
        }
    }

    // Steps 2+3: Up to 2 retry rounds for collisions — each round generates 3 alternatives
    // per blocked phrase, giving more chances to escape a crowded intent space.
    let initially_blocked = blocked_for_retry.len();
    let added_before_retry = added.len();
    let mut suggestions = Vec::new();
    let mut still_blocked = blocked_for_retry;

    for _round in 0..2 {
        if still_blocked.is_empty() || state.llm_key.is_none() { break; }

        let blocked_desc: String = still_blocked.iter()
            .map(|(intent, phrase, reason)| format!("  \"{}\" → {}: {}", phrase, intent, reason))
            .collect::<Vec<_>>().join("\n");

        let retry_prompt = format!(
            "These training phrases were REJECTED by a collision guard:\n{}\n\n\
             The guard blocks phrases whose terms are exclusive to a competing intent.\n\
             For each rejected phrase, suggest 3 alternatives that:\n\
             - Use completely different vocabulary (avoid ALL terms flagged above)\n\
             - Still express the same user intent\n\
             - Are diverse from each other\n\n\
             {}\n\n\
             Respond with ONLY valid JSON:\n\
             {{\"phrases_by_intent\": {{\"intent_name\": [\"alt1\", \"alt2\", \"alt3\"]}}}}\n",
            blocked_desc, asv_router::phrase::PHRASE_QUALITY_RULES
        );

        let mut next_blocked: Vec<(String, String, String)> = Vec::new();

        if let Ok(response) = call_llm(state, &retry_prompt, 1500).await {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(extract_json(&response)) {
                if let Some(sbi) = parsed.get("phrases_by_intent").and_then(|v| v.as_object()) {
                    let mut routers = state.routers.write().unwrap();
                    if let Some(router) = routers.get_mut(app_id) {
                        for (intent_id, phrases) in sbi {
                            if let Some(arr) = phrases.as_array() {
                                let mut any_passed = false;
                                for phrase in arr {
                                    if let Some(s) = phrase.as_str() {
                                        if auto_apply_retry {
                                            let result = router.add_phrase_checked(intent_id, s, lang);
                                            if result.added {
                                                added.push((intent_id.clone(), s.to_string()));
                                                any_passed = true;
                                            } else if !any_passed {
                                                // All alternatives so far blocked — carry forward for next round
                                                let reason = if !result.conflicts.is_empty() {
                                                    result.conflicts.iter()
                                                        .map(|c| format!("'{}' conflicts with {}", c.term, c.competing_intent))
                                                        .collect::<Vec<_>>().join("; ")
                                                } else {
                                                    result.warning.clone().unwrap_or_else(|| "still blocked".to_string())
                                                };
                                                next_blocked.push((intent_id.clone(), s.to_string(), reason));
                                            }
                                        } else {
                                            suggestions.push((intent_id.clone(), s.to_string()));
                                        }
                                    }
                                }
                                if any_passed { next_blocked.retain(|(id, _, _)| id != intent_id); }
                            }
                        }
                        if auto_apply_retry && !added.is_empty() {
                            maybe_persist(state, app_id, router);
                        }
                    }
                }
            }
        }

        // Deduplicate next_blocked (keep one per intent)
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        next_blocked.retain(|(id, _, _)| seen.insert(id.clone()));
        still_blocked = next_blocked;
    }

    // Anything still blocked after 2 rounds is final
    blocked_final.extend(still_blocked);

    let recovered_by_retry = added.len().saturating_sub(added_before_retry);
    PhrasePipelineResult { added, blocked: blocked_final, initially_blocked, recovered_by_retry, suggestions }
}

// --- Shared full review: 3 turns + guard + apply ---

/// Result of a full 3-turn review.
#[derive(Debug, Clone, serde::Serialize)]
#[derive(serde::Deserialize)]
pub struct FullReviewResult {
    /// Turn 1: correct intents identified by LLM
    pub correct_intents: Vec<String>,
    /// Turn 1: wrongly detected intents
    pub wrong_detections: Vec<String>,
    /// Turn 1: intents the query expresses but router missed
    pub missed_intents: Vec<String>,
    /// Turn 1: detected languages
    pub languages: Vec<String>,
    /// True when Turn 1 found no issues — Turns 2+3 were skipped
    pub detection_perfect: bool,
    /// Turn 2 (phrase mode): phrases to add (passed guard + retry)
    pub phrases_to_add: HashMap<String, Vec<String>>,
    /// Turn 2: phrases blocked by guard
    pub phrases_blocked: Vec<(String, String, String)>, // (intent, phrase, reason)
    /// Turn 3: discriminating suppressor words per wrong intent — added as L2 suppressor edges.
    /// These are query words that specifically indicate the query does NOT belong to that intent.
    pub suppressor_words: HashMap<String, Vec<String>>, // wrong_intent → [words]
    /// Turn 3: summary
    pub summary: String,
}


/// Run the full review for a query.
/// Used by: auto-learn (applies immediately), manual review (returns for approval), auto-improve.
pub async fn full_review(
    state: &AppState,
    app_id: &str,
    query: &str,
    detected: &[String],
) -> Result<FullReviewResult, String> {
    // Turn 1 context: intent labels (id + description only — no phrases)
    let intent_labels = {
        let routers = state.routers.read().unwrap();
        routers.get(app_id).map(|r| build_intent_labels(r)).unwrap_or_default()
    };

    // L2 scores for detected intents — helps Turn 1 judge confidence level
    let detected_with_scores: String = {
        let ig_map = state.intent_graph.read().unwrap();
        let heb_map = state.hebbian.read().unwrap();
        if let (Some(ig), Some(heb)) = (ig_map.get(app_id), heb_map.get(app_id)) {
            let pre = heb.preprocess(query);
            let (all_scores, _) = ig.score_multi_normalized(&pre.expanded, 0.0, 100.0);
            let score_map: HashMap<&str, f32> = all_scores.iter().map(|(id, s)| (id.as_str(), *s)).collect();
            if detected.is_empty() {
                "(none detected)".to_string()
            } else {
                detected.iter().map(|id| {
                    let score = score_map.get(id.as_str()).copied().unwrap_or(0.0);
                    format!("  {} (L2 score: {:.2})", id, score)
                }).collect::<Vec<_>>().join("\n")
            }
        } else if detected.is_empty() {
            "(none detected)".to_string()
        } else {
            detected.iter().map(|id| format!("  {}", id)).collect::<Vec<_>>().join("\n")
        }
    };

    // L1 expansion context — show if the query was normalized before routing
    let l1_context: String = {
        let heb_map = state.hebbian.read().unwrap();
        if let Some(heb) = heb_map.get(app_id) {
            let pre = heb.preprocess(query);
            if pre.was_modified && !pre.injected.is_empty() {
                format!("Router expanded query via synonyms: injected {:?} → processed as \"{}\"\n", pre.injected, pre.expanded)
            } else {
                String::new()
            }
        } else {
            String::new()
        }
    };

    // === Turn 1: Judge ===
    let turn1_prompt = format!(
        "Customer query: \"{query}\"\n\
         {l1_context}\
         Router detected:\n{detected_with_scores}\n\n\
         Available intents:\n{intent_labels}\n\n\
         Which intents does this query EXPLICITLY express? Only literal, not implied.\n\
         Which detected intents are WRONG (false positives)?\n\
         Which intents does the query express that were NOT detected (missed)?\n\
         What language is the query in?\n\
         Respond with ONLY JSON:\n\
         {{\"correct_intents\": [\"intent_id\"], \"wrong_detections\": [\"intent_id\"], \"missed_intents\": [\"intent_id\"], \"languages\": [\"en\"]}}\n"
    );

    let t1_response = call_llm(state, &turn1_prompt, 256).await
        .map_err(|e| format!("Turn 1 failed: {}", e.1))?;
    let t1_parsed: serde_json::Value = serde_json::from_str(extract_json(&t1_response))
        .map_err(|e| format!("Turn 1 parse failed: {}", e))?;

    let correct_intents: Vec<String> = t1_parsed["correct_intents"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let wrong_detections: Vec<String> = t1_parsed["wrong_detections"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let missed_intents: Vec<String> = t1_parsed["missed_intents"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let languages: Vec<String> = t1_parsed["languages"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_else(|| vec!["en".to_string()]);

    eprintln!("[full_review] Turn 1: correct={:?}, wrong={:?}, missed={:?}, langs={:?}",
        correct_intents, wrong_detections, missed_intents, languages);

    // === Early exit: detection is perfect — skip Turns 2 + 3 ===
    if wrong_detections.is_empty() && missed_intents.is_empty() {
        eprintln!("[full_review] Detection perfect — skipping Turns 2+3");
        return Ok(FullReviewResult {
            correct_intents,
            wrong_detections,
            missed_intents,
            languages,
            detection_perfect: true,

            phrases_to_add: HashMap::new(),
            phrases_blocked: Vec::new(),
            suppressor_words: HashMap::new(),
            summary: "Detection correct, no changes needed.".to_string(),
        });
    }

    // === Turn 2: Fix misses — generate training phrases for missed intents ===
    // Show the MISSED intents' current phrases so LLM avoids duplicating what's already there.
    // Include descriptions so the LLM understands each intent's scope.
    let all_relevant_intents: Vec<String> = missed_intents.iter()
        .chain(correct_intents.iter())
        .cloned()
        .collect::<std::collections::LinkedList<_>>()
        .into_iter()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let existing_phrases: String = {
        let routers = state.routers.read().unwrap();
        if let Some(router) = routers.get(app_id) {
            intent_phrases_context(router, &all_relevant_intents, 15)
        } else { String::new() }
    };

    // Language instruction: if non-English, generate phrases in that language too
    let detected_lang = languages.first().map(|s| s.as_str()).unwrap_or("en");
    let lang_instruction = if detected_lang == "en" {
        String::new()
    } else {
        format!("\nThe query is in \"{detected_lang}\". Generate phrases in \"{detected_lang}\" for missed intents.\n")
    };

    let missed_labels: String = {
        let routers = state.routers.read().unwrap();
        if let Some(router) = routers.get(app_id) {
            missed_intents.iter().map(|id| {
                let desc = router.get_description(id);
                if desc.is_empty() { format!("  - {}", id) }
                else { format!("  - {} ({})", id, desc) }
            }).collect::<Vec<_>>().join("\n")
        } else {
            missed_intents.iter().map(|id| format!("  - {}", id)).collect::<Vec<_>>().join("\n")
        }
    };

    let turn2_prompt = format!(
        "{guidelines}\n\n\
         Customer query: \"{query}\"\n\
         Missed intents (router failed to detect these):\n{missed_labels}\n\n\
         Current phrases already in the system (do not duplicate):\n{existing_phrases}\n\
         {lang_instruction}\
         {quality}\n\n\
         Generate 2-3 new training phrases per missed intent that would help route this query.\n\
         Respond with ONLY JSON:\n\
         {{\"phrases_by_intent\": {{\"intent_id\": [\"phrase1\", \"phrase2\"]}}}}\n",
        guidelines = asv_router::phrase::REVIEW_FIX_GUIDELINES,
        quality = asv_router::phrase::PHRASE_QUALITY_RULES,
    );

    let t2_response = call_llm(state, &turn2_prompt, 300).await
        .map_err(|e| format!("Turn 2 failed: {}", e.1))?;
    let t2_parsed: serde_json::Value = serde_json::from_str(extract_json(&t2_response))
        .map_err(|e| {
            eprintln!("[full_review] Turn 2 parse error: {}. Raw: {}", e, &t2_response[..t2_response.len().min(300)]);
            format!("Turn 2 parse failed: {}", e)
        })?;

    // Pre-validate phrases through guard (read-only check + retry)
    let mut phrases_to_add: HashMap<String, Vec<String>> = HashMap::new();
    let mut phrases_blocked = Vec::new();

    if let Some(sbi) = t2_parsed.get("phrases_by_intent").and_then(|v| v.as_object()) {
        let mut blocked_for_retry: Vec<(String, String, String)> = Vec::new();

        {
            let routers = state.routers.read().unwrap();
            if let Some(router) = routers.get(app_id) {
                for (intent_id, phrases) in sbi {
                    if let Some(arr) = phrases.as_array() {
                        for phrase in arr {
                            if let Some(s) = phrase.as_str() {
                                let check = router.check_phrase(intent_id, s);
                                if check.conflicts.is_empty() && !check.redundant
                                    && check.warning.as_deref() != Some("No content terms after tokenization")
                                {
                                    phrases_to_add.entry(intent_id.clone()).or_default().push(s.to_string());
                                } else if !check.conflicts.is_empty() {
                                    let reason = check.conflicts.iter()
                                        .map(|c| format!("'{}' conflicts with {}", c.term, c.competing_intent))
                                        .collect::<Vec<_>>().join("; ");
                                    blocked_for_retry.push((intent_id.clone(), s.to_string(), reason));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Feedback loop: one retry for blocked phrases
        if !blocked_for_retry.is_empty() && state.llm_key.is_some() {
            let blocked_desc: String = blocked_for_retry.iter()
                .map(|(intent, phrase, reason)| format!("  \"{}\" → {}: {}", phrase, intent, reason))
                .collect::<Vec<_>>().join("\n");

            let retry_prompt = format!(
                "These training phrases were REJECTED by the collision guard:\n{}\n\n\
                 The guard blocks phrases containing terms exclusive to other intents.\n\
                 For each rejected phrase, suggest ONE alternative that avoids the conflicting terms.\n\
                 Use completely different vocabulary that still captures the same meaning.\n\n\
                 {}\n\n\
                 Respond with ONLY JSON:\n\
                 {{\"phrases_by_intent\": {{\"intent_name\": [\"alternative_phrase\"]}}}}\n",
                blocked_desc, asv_router::phrase::PHRASE_QUALITY_RULES
            );

            if let Ok(retry_resp) = call_llm(state, &retry_prompt, 300).await {
                if let Ok(retry_parsed) = serde_json::from_str::<serde_json::Value>(extract_json(&retry_resp)) {
                    if let Some(retry_sbi) = retry_parsed.get("phrases_by_intent").and_then(|v| v.as_object()) {
                        let routers = state.routers.read().unwrap();
                        if let Some(router) = routers.get(app_id) {
                            for (intent_id, phrases) in retry_sbi {
                                if let Some(arr) = phrases.as_array() {
                                    for phrase in arr {
                                        if let Some(s) = phrase.as_str() {
                                            let check = router.check_phrase(intent_id, s);
                                            if check.conflicts.is_empty() && !check.redundant {
                                                phrases_to_add.entry(intent_id.clone()).or_default().push(s.to_string());
                                            } else {
                                                phrases_blocked.push((intent_id.clone(), s.to_string(), "still blocked after retry".to_string()));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    eprintln!("[full_review] Turn 2: phrases_to_add={:?}, blocked={}", phrases_to_add, phrases_blocked.len());

    // Turn 3 removed: count/log-odds model handles false positives without explicit suppressors.
    // As more correct phrases are learned, wrong intents' log-odds naturally decrease.
    let summary = String::new();

    Ok(FullReviewResult {
        correct_intents,
        wrong_detections,
        missed_intents,
        languages,
        detection_perfect: false,
        phrases_to_add,
        phrases_blocked,
        suppressor_words: HashMap::new(),
        summary,
    })
}


/// Apply a full review result: add phrases, update L2 edges, add L1 synonyms.
pub async fn apply_review(
    state: &AppState,
    app_id: &str,
    result: &FullReviewResult,
    original_query: &str,
) -> (usize, usize) { // (phrases_added, suppressors_added)
    let mut added = 0;
    let mut suppressors_added = 0;

    // Apply phrases_to_add through pipeline — use detected language, not hardcoded "en"
    if !result.phrases_to_add.is_empty() {
        let lang = result.languages.first().map(|s| s.as_str()).unwrap_or("en");
        let pipeline_result = phrase_pipeline(state, app_id, &result.phrases_to_add, true, lang).await;
        added = pipeline_result.added.len();
    }

    // ── L2 update ────────────────────────────────────────────────────────────
    // Count model: learn_phrase() on new phrases — counts increment, log-odds recompute.
    //              One phrase = immediate discriminative update. No delta tuning.
    // Legacy model: reinforce() with delta nudges (kept for backward compat).
    const DELTA_MISS: f32      =  0.15;
    const DELTA_REINFORCE: f32 =  0.05;
    const DELTA_SUPPRESS: f32  = -0.05;

    let has_l3_update = !result.correct_intents.is_empty()
        || !result.missed_intents.is_empty()
        || !result.wrong_detections.is_empty();

    if has_l3_update {
        let normalized = {
            let hebbian = state.hebbian.read().unwrap();
            if let Some(g) = hebbian.get(app_id) {
                g.preprocess(original_query).expanded
            } else {
                original_query.to_string()
            }
        };
        let words: Vec<String> = asv_router::tokenizer::tokenize(&normalized);
        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

        eprintln!("[auto-learn/L2] query='{}' | expanded='{}'", original_query, normalized);
        eprintln!("[auto-learn/L2] tokens: {:?}", word_refs);

        let new_vocabulary: Vec<String> = {
            let ig_map = state.intent_graph.read().unwrap();
            if let Some(ig) = ig_map.get(app_id) {
                word_refs.iter()
                    .filter(|&&w| !ig.word_intent.contains_key(w))
                    .map(|&w| w.to_string())
                    .collect()
            } else { vec![] }
        };

        {
            let mut ig_map = state.intent_graph.write().unwrap();
            if let Some(ig) = ig_map.get_mut(app_id) {
                if ig.is_count_model() {
                    // ── Count model: learn phrases directly ────────────────
                    // Correct intents: learn the query as an example phrase (positive reinforcement).
                    for intent_id in &result.correct_intents {
                        ig.learn_phrase(&word_refs, intent_id);
                        eprintln!("[auto-learn/L2] correct  → learn_phrase {:?} → '{}'", word_refs, intent_id);
                    }
                    // Missed intents: learn new phrases generated by Turn 2.
                    for (intent_id, phrases) in &result.phrases_to_add {
                        for phrase in phrases {
                            let phrase_words: Vec<String> = asv_router::tokenizer::tokenize(phrase);
                            let phrase_refs: Vec<&str> = phrase_words.iter().map(|s| s.as_str()).collect();
                            ig.learn_phrase(&phrase_refs, intent_id);
                            eprintln!("[auto-learn/L2] missed   → learn_phrase {:?} → '{}'", phrase_refs, intent_id);
                        }
                    }
                    // Wrong detections: no explicit action — as correct phrases accumulate,
                    // the false positive's log-odds naturally decrease relative to correct intent.
                    for intent_id in &result.wrong_detections {
                        eprintln!("[auto-learn/L2] wrong    → '{}' (no action — log-odds self-correct)", intent_id);
                    }
                } else {
                    // ── Legacy model: delta nudges ─────────────────────────
                    for intent_id in &result.correct_intents {
                        ig.reinforce(&word_refs, intent_id, DELTA_REINFORCE);
                    }
                    for intent_id in &result.missed_intents {
                        ig.reinforce(&word_refs, intent_id, DELTA_MISS);
                    }
                    for intent_id in &result.wrong_detections {
                        if let Some(sup_words) = result.suppressor_words.get(intent_id) {
                            let sup_refs: Vec<&str> = sup_words.iter().map(|s| s.as_str()).collect();
                            ig.add_suppressors(&sup_refs, intent_id, DELTA_SUPPRESS.abs());
                            suppressors_added += sup_words.len();
                        } else {
                            ig.reinforce(&word_refs, intent_id, DELTA_SUPPRESS);
                        }
                    }
                    // Conjunction reinforcement (legacy only)
                    let fired = ig.fired_conjunction_indices(&word_refs);
                    for idx in &fired {
                        let intent = ig.conjunctions[*idx].intent.clone();
                        if result.correct_intents.contains(&intent) {
                            ig.reinforce_conjunction(*idx, DELTA_REINFORCE);
                        } else if result.wrong_detections.contains(&intent) {
                            ig.reinforce_conjunction(*idx, DELTA_SUPPRESS);
                        }
                    }
                }

                if let Some(ref dir) = state.data_dir {
                    let path = format!("{}/{}/_intent_graph.json", dir, app_id);
                    ig.save(&path).ok();
                    eprintln!("[auto-learn/L2] graph persisted → {}", path);
                }
            }
        }

        // ── L1 synonym reinforcement ─────────────────────────────────────────
        // Reinforce L1 synonym edges that fired and contributed to correct routing.
        // L1 only gets positive updates — synonyms are global, suppressing them
        // could hurt other intents that use the same synonym correctly.
        if !result.correct_intents.is_empty() {
            let mut heb_map = state.hebbian.write().unwrap();
            if let Some(heb) = heb_map.get_mut(app_id) {
                let pre = heb.preprocess(original_query);
                let orig_words = asv_router::hebbian::HebbianGraph::l1_tokens_pub(original_query);
                for word in &orig_words {
                    if let Some(edges) = heb.edges.get(word.as_str()).cloned() {
                        for edge in edges {
                            if matches!(edge.kind, asv_router::hebbian::EdgeKind::Synonym)
                                && pre.injected.contains(&edge.target)
                            {
                                heb.reinforce(word, &edge.target, DELTA_REINFORCE);
                                eprintln!("[auto-learn/L1] synonym '{}' → '{}' reinforce ({:+.2})", word, edge.target, DELTA_REINFORCE);
                            }
                        }
                    }
                }
                if let Some(ref dir) = state.data_dir {
                    let path = format!("{}/{}/_hebbian.json", dir, app_id);
                    heb.save(&path).ok();
                }
            }
        }

        // ── L1 learning: morphology + synonym discovery run in parallel ──────
        // Both are independent LLM calls — no shared state, safe to join.
        let do_morph = !new_vocabulary.is_empty();
        let do_syn = !result.missed_intents.is_empty();

        match (do_morph, do_syn) {
            (true, true) => {
                eprintln!("[auto-learn/L1] parallel: morphology({:?}) + synonym discovery", new_vocabulary);
                tokio::join!(
                    learn_l1_morphology(state, app_id, &new_vocabulary, original_query),
                    learn_l1_synonyms(state, app_id, &word_refs, &result.missed_intents, original_query)
                );
            }
            (true, false) => {
                eprintln!("[auto-learn/L1] morphology only: {:?}", new_vocabulary);
                learn_l1_morphology(state, app_id, &new_vocabulary, original_query).await;
            }
            (false, true) => {
                learn_l1_synonyms(state, app_id, &word_refs, &result.missed_intents, original_query).await;
            }
            (false, false) => {}
        }
    }

    (added, suppressors_added)
}

/// For missed intents, map query words → existing L2 vocabulary words as new L1 synonym edges.
/// Handles cross-lingual and paraphrase cases: "suscripción"→"subscription" routes via the
/// already-strong "subscription"→cancel_subscription L2 edge instead of starting a weak new one.
async fn learn_l1_synonyms(
    state: &AppState,
    app_id: &str,
    query_words: &[&str],
    missed_intents: &[String],
    context_query: &str,
) {
    if state.llm_key.is_none() || query_words.is_empty() { return; }

    // Get L2 vocabulary that already activates the missed intents — these are
    // the target words we want new synonyms to map TO.
    let graph_vocab: Vec<String> = {
        let ig_map = state.intent_graph.read().unwrap();
        match ig_map.get(app_id) {
            Some(ig) => ig.word_intent.iter()
                .filter(|(_, acts)| acts.iter().any(|(id, _)| missed_intents.contains(id)))
                .map(|(w, _)| w.clone())
                .collect(),
            None => return,
        }
    };

    if graph_vocab.is_empty() { return; }

    // Only consider query words not already in L2 — words that ARE in L2 don't need a synonym path.
    let unknown_words: Vec<&str> = {
        let ig_map = state.intent_graph.read().unwrap();
        match ig_map.get(app_id) {
            Some(ig) => query_words.iter()
                .filter(|&&w| !ig.word_intent.contains_key(w))
                .copied()
                .collect(),
            None => query_words.to_vec(),
        }
    };

    if unknown_words.is_empty() {
        eprintln!("[auto-learn/L1] all query words already in L2 — synonym pass skipped");
        return;
    }

    let words_str = unknown_words.join(", ");
    let vocab_str = graph_vocab.join(", ");
    let intents_str = missed_intents.join(", ");

    let prompt = format!(
        "An intent router missed routing this query to the correct intent(s).\n\
         Query: \"{context_query}\"\n\
         Missed intent(s): {intents_str}\n\n\
         Query words NOT yet in the graph: [{words_str}]\n\
         Existing graph vocabulary for these intents: [{vocab_str}]\n\n\
         Map each query word to an existing graph word if they are synonyms, translations, or paraphrases.\n\
         Only map to words from the graph vocabulary list above — do not invent new words.\n\
         Skip query words that have no good match in the graph vocabulary.\n\n\
         Respond with ONLY JSON:\n\
         {{\"synonyms\": {{\"query_word\": \"graph_word\"}}}}\n\
         Example: {{\"synonyms\": {{\"suscripción\": \"subscription\", \"cancelar\": \"cancel\"}}}}\n\
         If no useful mappings exist: {{\"synonyms\": {{}}}}"
    );

    match call_llm(state, &prompt, 300).await {
        Ok(response) => {
            let json_str = extract_json(&response);
            match serde_json::from_str::<serde_json::Value>(json_str) {
                Ok(parsed) => {
                    if let Some(syn_map) = parsed["synonyms"].as_object() {
                        let mut heb_map = state.hebbian.write().unwrap();
                        if let Some(graph) = heb_map.get_mut(app_id) {
                            let mut learned = 0usize;
                            for (from_word, to_word) in syn_map {
                                if let Some(target) = to_word.as_str() {
                                    let target = target.trim().to_lowercase();
                                    let from = from_word.trim().to_lowercase();
                                    if !target.is_empty() && target != from
                                        && graph_vocab.contains(&target)
                                    {
                                        graph.add(&from, &target, 0.88,
                                            asv_router::hebbian::EdgeKind::Synonym);
                                        eprintln!("[auto-learn/L1] synonym {} → {} (semantic/cross-lingual)", from, target);
                                        learned += 1;
                                    }
                                }
                            }
                            if learned > 0 {
                                if let Some(ref dir) = state.data_dir {
                                    let path = format!("{}/{}/_hebbian.json", dir, app_id);
                                    graph.save(&path).ok();
                                    eprintln!("[auto-learn/L1] {} synonym edges saved → {}", learned, path);
                                }
                            } else {
                                eprintln!("[auto-learn/L1] no synonym mappings found for {:?}", unknown_words);
                            }
                        }
                    }
                }
                Err(e) => eprintln!("[auto-learn/L1] synonym parse error: {} — raw: {}", e, &response[..response.len().min(200)]),
            }
        }
        Err((_, e)) => eprintln!("[auto-learn/L1] synonym LLM call failed: {}", e),
    }
}

/// Ask the LLM for morphological variants of newly discovered words and add them to L1.
/// This closes the gap where L2 learns "ping"→send_message but L1 doesn't know "pinging"→"ping".
async fn learn_l1_morphology(
    state: &AppState,
    app_id: &str,
    new_words: &[String],
    context_query: &str,
) {
    if state.llm_key.is_none() { return; }

    let words_str = new_words.join(", ");
    let prompt = format!(
        "These words appeared in a user query (\"{context_query}\") and were just learned by an intent router:\n\
         Words: [{words_str}]\n\n\
         For each word, list ONLY morphological variants (inflected forms) that users would naturally type:\n\
         - verb forms: -ing, -ed, -s, -ion, -er suffixes\n\
         - Do NOT include synonyms or semantically related words — only inflected forms of the same word\n\
         - Do NOT include the word itself\n\
         - Skip words that have no useful variants (e.g. nouns like 'team')\n\n\
         Respond with ONLY JSON:\n\
         {{\"variants\": {{\"canonical_word\": [\"variant1\", \"variant2\"]}}}}\n\
         Example: {{\"variants\": {{\"ping\": [\"pinging\", \"pinged\", \"pings\", \"pinged\"]}}}}"
    );

    match call_llm(state, &prompt, 400).await {
        Ok(response) => {
            let json_str = extract_json(&response);
            match serde_json::from_str::<serde_json::Value>(json_str) {
                Ok(parsed) => {
                    if let Some(variants_map) = parsed["variants"].as_object() {
                        let mut hebbian = state.hebbian.write().unwrap();
                        if let Some(graph) = hebbian.get_mut(app_id) {
                            let mut learned = 0usize;
                            for (canonical, var_list) in variants_map {
                                if let Some(arr) = var_list.as_array() {
                                    for v in arr {
                                        if let Some(variant) = v.as_str() {
                                            let variant = variant.trim().to_lowercase();
                                            if !variant.is_empty() && variant != canonical.as_str() {
                                                graph.add(&variant, canonical, 0.97,
                                                    asv_router::hebbian::EdgeKind::Morphological);
                                                eprintln!("[auto-learn/L1] {} → {} (morphological)", variant, canonical);
                                                learned += 1;
                                            }
                                        }
                                    }
                                }
                            }
                            if learned > 0 {
                                if let Some(ref dir) = state.data_dir {
                                    let path = format!("{}/{}/_hebbian.json", dir, app_id);
                                    graph.save(&path).ok();
                                    eprintln!("[auto-learn/L1] {} edges added, graph persisted → {}", learned, path);
                                }
                            } else {
                                eprintln!("[auto-learn/L1] no morphological variants found for {:?}", new_words);
                            }
                        }
                    }
                }
                Err(e) => eprintln!("[auto-learn/L1] parse error: {} — raw: {}", e, &response[..response.len().min(200)]),
            }
        }
        Err((_, e)) => eprintln!("[auto-learn/L1] LLM call failed: {}", e),
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_extract_json_fenced_array() {
        let s = "```json\n[\n  {\"from\": \"x\", \"to\": \"y\"}\n]\n```";
        let r = extract_json(s);
        assert!(r.starts_with('['), "expected array, got: {:?}", r);
    }
    #[test]
    fn test_extract_json_preamble_then_fence() {
        let s = "Here are edges for {cancel_sub, create_repo}:\n```json\n[\n  {\"from\": \"x\"}\n]\n```";
        let r = extract_json(s);
        assert!(r.starts_with('['), "expected array, got: {:?}", r);
    }
}
