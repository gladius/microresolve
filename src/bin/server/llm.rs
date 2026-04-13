//! LLM integration: call_llm, phrase pipeline, full 3-turn review.

use axum::http::StatusCode;
use std::collections::HashMap;
use asv_router::Router;
use crate::state::*;

pub fn build_intent_descriptions(router: &Router) -> String {
    router.intent_ids().iter().map(|id| {
        let desc = router.get_description(id);
        let seeds = router.get_training(id)
            .unwrap_or_default()
            .into_iter()
            .take(4)
            .collect::<Vec<_>>()
            .join("\", \"");
        if desc.is_empty() {
            format!("- {}: [\"{}\"]", id, seeds)
        } else {
            format!("- {} ({}): [\"{}\"]", id, desc, seeds)
        }
    }).collect::<Vec<_>>().join("\n")
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
    /// Turn 2 (concept mode): signals to add to concepts
    #[serde(default)]
    pub signals_to_add: Vec<(String, String)>, // (concept, signal)
    /// Turn 2 (phrase mode): phrases to add (passed guard + retry)
    pub phrases_to_add: HashMap<String, Vec<String>>,
    /// Turn 2: phrases blocked by guard
    pub phrases_blocked: Vec<(String, String, String)>, // (intent, phrase, reason)
    /// Turn 3: phrases to replace in wrong intents
    pub phrases_to_replace: Vec<PhraseReplacement>,
    /// Turn 3: safe to apply
    pub safe_to_apply: bool,
    /// Turn 3: summary
    pub summary: String,
}

#[derive(Debug, Clone, serde::Serialize)]
#[derive(serde::Deserialize)]
pub struct PhraseReplacement {
    pub intent: String,
    pub old_phrase: String,
    pub new_phrase: String,
    pub reason: String,
}

/// Run the full 3-turn review for a query.
/// Used by: auto-learn (applies immediately), manual review (returns for approval), auto-improve.
pub async fn full_review(
    state: &AppState,
    app_id: &str,
    query: &str,
    detected: &[String],
) -> Result<FullReviewResult, String> {
    let intent_descriptions = {
        let routers = state.routers.read().unwrap();
        routers.get(app_id).map(|r| build_intent_descriptions(r)).unwrap_or_default()
    };

    // === Turn 1: Diagnose ===
    let turn1_prompt = format!(
        "Customer query: \"{}\"\nDetected intents: {:?}\n\n\
         Available intents (with descriptions and example seeds):\n{}\n\n\
         Which intents does this query EXPLICITLY express? Only literal requests.\n\
         Which detected intents are WRONG for this query?\n\
         Which intents does the query express that were NOT detected (missed)?\n\
         Respond with ONLY JSON:\n\
         {{\"correct_intents\": [\"intent1\"], \"wrong_detections\": [\"wrong1\"], \"missed_intents\": [\"missed1\"], \"languages\": [\"en\"]}}\n",
        query, detected, intent_descriptions
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
            signals_to_add: Vec::new(),
            phrases_to_add: HashMap::new(),
            phrases_blocked: Vec::new(),
            phrases_to_replace: Vec::new(),
            safe_to_apply: true,
            summary: "Detection correct, no changes needed.".to_string(),
        });
    }

    // === Branch: concept registry or term index? ===
    let has_concepts = state.concepts.read().unwrap().contains_key(app_id);

    // === CONCEPT PATH: Turn 2 + 3 via signal learning ===
    if has_concepts {
        return concept_review_turns(state, app_id, query, &correct_intents, &wrong_detections, &missed_intents, &languages).await;
    }

    // === TERM INDEX PATH: Turn 2: Fix misses (phrases to add) ===
    let existing_phrases: String = {
        let routers = state.routers.read().unwrap();
        if let Some(router) = routers.get(app_id) {
            correct_intents.iter()
                .map(|id| format!("  {}: {:?}", id, router.get_training(id).unwrap_or_default()))
                .collect::<Vec<_>>().join("\n")
        } else { String::new() }
    };

    let lang_instruction = if languages.is_empty() || languages == vec!["en"] {
        String::new()
    } else {
        format!("\nThe query is in {:?}. Generate phrases in the detected language.\n", languages)
    };

    let turn2_prompt = format!(
        "{}\n\n\
         Customer query: \"{}\"\n\
         Correct intents: {:?}\n\n\
         Current phrases in the index:\n{}\n\n\
         {}\
         {}\n\n\
         Respond with ONLY JSON:\n\
         {{\"phrases_by_intent\": {{\"intent_name\": [\"phrase1\", \"phrase2\"]}}}}\n",
        asv_router::phrase::REVIEW_FIX_GUIDELINES,
        query, correct_intents, existing_phrases, lang_instruction,
        asv_router::phrase::PHRASE_QUALITY_RULES
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

    // === Turn 3: Fix false positives (replace broad phrases) ===
    let mut phrases_to_replace = Vec::new();
    let mut safe_to_apply = true;
    let mut summary = String::new();

    if !wrong_detections.is_empty() {
        let wrong_intent_phrases: String = {
            let routers = state.routers.read().unwrap();
            if let Some(router) = routers.get(app_id) {
                wrong_detections.iter().map(|id| {
                    let phrases = router.get_training(id).unwrap_or_default();
                    format!("- {}: {:?}", id, phrases.iter().take(10).collect::<Vec<_>>())
                }).collect::<Vec<_>>().join("\n")
            } else { String::new() }
        };

        let turn3_prompt = format!(
            "Customer query: \"{}\"\n\n\
             These intents were WRONGLY detected (false positives):\n{}\n\n\
             Phrases being added for correct intents:\n{}\n\n\
             For each wrong intent:\n\
             1. Which phrase caused the false match?\n\
             2. Suggest a MORE SPECIFIC replacement phrase that keeps the intent's coverage but stops matching this query.\n\
             Do NOT suggest removing phrases — suggest replacements that narrow the match.\n\n\
             Replacement phrase quality rules:\n\
             {}\n\n\
             Respond with ONLY JSON:\n\
             {{\n\
               \"replacements\": [{{\"intent\": \"name\", \"old_phrase\": \"the broad phrase\", \"new_phrase\": \"more specific version\", \"reason\": \"why\"}}],\n\
               \"safe_to_apply\": true,\n\
               \"summary\": \"one sentence\"\n\
             }}\n\
             If no replacement needed, use empty replacements array.\n",
            query,
            wrong_intent_phrases,
            phrases_to_add.iter().map(|(id, phrases)| format!("  {}: {:?}", id, phrases)).collect::<Vec<_>>().join("\n"),
            asv_router::phrase::PHRASE_QUALITY_RULES,
        );

        if let Ok(t3_response) = call_llm(state, &turn3_prompt, 1024).await {
            if let Ok(t3_parsed) = serde_json::from_str::<serde_json::Value>(extract_json(&t3_response)) {
                if let Some(replacements) = t3_parsed["replacements"].as_array() {
                    for r in replacements {
                        if let (Some(intent), Some(old), Some(new), Some(reason)) = (
                            r["intent"].as_str(), r["old_phrase"].as_str(),
                            r["new_phrase"].as_str(), r["reason"].as_str(),
                        ) {
                            phrases_to_replace.push(PhraseReplacement {
                                intent: intent.to_string(),
                                old_phrase: old.to_string(),
                                new_phrase: new.to_string(),
                                reason: reason.to_string(),
                            });
                        }
                    }
                }
                safe_to_apply = t3_parsed["safe_to_apply"].as_bool().unwrap_or(true);
                summary = t3_parsed["summary"].as_str().unwrap_or("").to_string();
            }
        }
    }

    eprintln!("[full_review] Turn 3: replacements={}, safe={}, summary={}", phrases_to_replace.len(), safe_to_apply, summary);

    Ok(FullReviewResult {
        correct_intents,
        wrong_detections,
        missed_intents,
        languages,
        detection_perfect: false,
        signals_to_add: Vec::new(),
        phrases_to_add,
        phrases_blocked,
        phrases_to_replace,
        safe_to_apply,
        summary,
    })
}

// ── Concept-aware review turns ────────────────────────────────────────────────

async fn concept_review_turns(
    state: &AppState,
    app_id: &str,
    query: &str,
    correct_intents: &[String],
    wrong_detections: &[String],
    missed_intents: &[String],
    languages: &[String],
) -> Result<FullReviewResult, String> {

    // Build concept summary and what fired for this query
    let (concept_summary, fired_summary) = {
        let concepts = state.concepts.read().unwrap();
        let reg = concepts.get(app_id);
        let concept_summary = reg.map(|r| {
            r.concepts.iter()
                .map(|(name, sigs)| format!("  {}: [{}]", name, sigs.iter().take(6).cloned().collect::<Vec<_>>().join(", ")))
                .collect::<Vec<_>>().join("\n")
        }).unwrap_or_default();
        let fired_summary = reg.map(|r| {
            r.explain(query).iter()
                .map(|a| format!("  {} (matched: {})", a.concept, a.matched_signals.join(", ")))
                .collect::<Vec<_>>().join("\n")
        }).unwrap_or_default();
        (concept_summary, fired_summary)
    };

    // === Concept Turn 2: identify missing signals ===
    let turn2_prompt = format!(
        r#"An intent router uses semantic concepts. A query was misrouted.

Query: "{query}"
Correct intent(s): {correct_intents:?}
Missed intent(s): {missed_intents:?}

Available concepts and their current signals:
{concept_summary}

Concepts that fired for this query:
{fired_summary}

The query failed because signals are missing from the concept registry.
Identify up to 3 signals to add — the key words/phrases from this query that should activate the correct concept(s).

Rules:
- Signals are 1-4 words, lowercase
- Pick words that are semantically meaningful for the correct intent
- Prefer specific phrases over single generic words
- Only add signals that aren't already present

Return ONLY JSON:
{{"signals_to_add": [{{"concept": "concept_name", "signal": "the phrase"}}]}}"#
    );

    let t2_response = call_llm(state, &turn2_prompt, 256).await
        .map_err(|e| format!("Concept Turn 2 failed: {}", e.1))?;
    let t2_parsed: serde_json::Value = serde_json::from_str(extract_json(&t2_response))
        .unwrap_or_default();

    let signals_to_add: Vec<(String, String)> = t2_parsed["signals_to_add"]
        .as_array().unwrap_or(&vec![])
        .iter()
        .filter_map(|s| {
            let concept = s["concept"].as_str()?.to_string();
            let signal = s["signal"].as_str()?.to_string();
            if concept.is_empty() || signal.is_empty() { return None; }
            Some((concept, signal))
        })
        .collect();

    eprintln!("[concept_review] Turn 2: signals_to_add={:?}", signals_to_add);

    // === Concept Turn 3: identify false positive concept signals ===
    // For now: log wrong detections, skip complex signal removal
    // (conjunction scoring already handles most false positives)
    if !wrong_detections.is_empty() {
        eprintln!("[concept_review] Turn 3: wrong_detections={:?} — conjunction scoring handles these", wrong_detections);
    }

    let summary = if !signals_to_add.is_empty() {
        format!("Adding {} signal(s) to concept registry", signals_to_add.len())
    } else {
        "No signal changes needed".to_string()
    };

    Ok(FullReviewResult {
        correct_intents: correct_intents.to_vec(),
        wrong_detections: wrong_detections.to_vec(),
        missed_intents: missed_intents.to_vec(),
        languages: languages.to_vec(),
        detection_perfect: false,
        signals_to_add,
        phrases_to_add: HashMap::new(),
        phrases_blocked: Vec::new(),
        phrases_to_replace: Vec::new(),
        safe_to_apply: true,
        summary,
    })
}

/// Apply a full review result: add phrases + replace broad phrases.
/// `original_query` is the failing query — used to learn situation n-grams for each
/// intent that gets phrases added (CJK always; Latin only if intent has situation patterns).
pub async fn apply_review(
    state: &AppState,
    app_id: &str,
    result: &FullReviewResult,
    original_query: &str,
) -> (usize, usize) { // (phrases_added, phrases_replaced)
    let mut added = 0;
    let mut replaced = 0;

    // === Concept path: apply signals directly ===
    if !result.signals_to_add.is_empty() {
        let mut concepts = state.concepts.write().unwrap();
        if let Some(reg) = concepts.get_mut(app_id) {
            for (concept, signal) in &result.signals_to_add {
                eprintln!("[apply_review] concept signal: '{}' → '{}'", signal, concept);
                reg.add_signal(concept, signal);
                added += 1;
            }
            if let Some(ref dir) = state.data_dir {
                let _ = reg.save(&format!("{}/{}/_concepts.json", dir, app_id));
            }
        }
    }

    // Apply phrases_to_add through pipeline
    if !result.phrases_to_add.is_empty() {
        let pipeline_result = phrase_pipeline(state, app_id, &result.phrases_to_add, true, "en").await;
        added = pipeline_result.added.len();

    }

    // Apply phrase replacements (remove old + add new through guard)
    if !result.phrases_to_replace.is_empty() {
        let mut routers = state.routers.write().unwrap();
        if let Some(router) = routers.get_mut(app_id) {
            for replacement in &result.phrases_to_replace {
                // Verify old phrase exists before removing
                let old_exists = router.get_training(&replacement.intent)
                    .map(|phrases| phrases.contains(&replacement.old_phrase))
                    .unwrap_or(false);

                if old_exists {
                    // Add new phrase first (through guard)
                    let check = router.add_phrase_checked(&replacement.intent, &replacement.new_phrase, "en");
                    if check.added {
                        // Only remove old if new was accepted
                        router.remove_phrase(&replacement.intent, &replacement.old_phrase);
                        replaced += 1;
                        eprintln!("[apply_review] Replaced in {}: \"{}\" → \"{}\"",
                            replacement.intent, replacement.old_phrase, replacement.new_phrase);
                    } else {
                        eprintln!("[apply_review] Replacement blocked for {}: new phrase \"{}\" rejected",
                            replacement.intent, replacement.new_phrase);
                    }
                }
            }
            maybe_persist(state, app_id, router);
        }
    }

    // ── Hebbian L2: full bidirectional learning ──────────────────────────────
    //
    // Weight derivation — all values anchored to the routing threshold (0.3):
    //
    //   DELTA_MISS = threshold / (min_phrase_words × expected_idf)
    //              = 0.30 / (2 × 1.0) = 0.15
    //   → A 2-word missed phrase accumulates 2 × 0.15 × 1.0 = 0.30, just above threshold.
    //   → A 3-word phrase gives 0.45, comfortably routable.
    //
    //   DELTA_REINFORCE = DELTA_MISS / 3 = 0.05
    //   → Asymptotic nudge: Δw = 0.05 × (1 - w). Diminishing returns prevent
    //     runaway growth — 1000 reinforcements converge to 1.0, never exceed it.
    //
    //   DELTA_SUPPRESS = -DELTA_REINFORCE = -0.05
    //   → Asymptotic decay: w = w × (1 - 0.05) = w × 0.95. Approaches 0,
    //     never goes negative. Strong edges (right 100 times) survive a few
    //     wrong routings — intentional.
    //
    // If the routing threshold changes, these three values should scale together.
    const DELTA_MISS: f32       =  0.15;  // threshold / (2 * IDF_avg)
    const DELTA_REINFORCE: f32  =  0.05;  // DELTA_MISS / 3
    const DELTA_SUPPRESS: f32   = -0.05;  // -DELTA_REINFORCE
    //
    // missed_intents is the key new path: adds word→intent edges that didn't
    // exist before, so L2 learns completely unknown vocabulary from real queries.
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

        // Content words: >2 chars, not pure numbers
        let words: Vec<String> = normalized
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| w.len() > 2 && !w.chars().all(|c| c.is_numeric()))
            .collect();
        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

        eprintln!("[auto-learn/L2] query='{}' | L1-expanded='{}'", original_query, normalized);
        eprintln!("[auto-learn/L2] content words: {:?}", word_refs);

        // Track words that are BRAND NEW to L3 (no existing edge for any intent).
        // These need L1 morphological learning — we don't want "pinging" to miss
        // the newly added "ping" edge because L1 doesn't know "pinging"→"ping".
        let new_vocabulary: Vec<String> = {
            let ig_map = state.intent_graph.read().unwrap();
            if let Some(ig) = ig_map.get(app_id) {
                word_refs.iter()
                    .filter(|&&w| !ig.word_intent.contains_key(w))
                    .map(|&w| w.to_string())
                    .collect()
            } else {
                vec![]
            }
        };

        {
            let mut ig_map = state.intent_graph.write().unwrap();
            if let Some(ig) = ig_map.get_mut(app_id) {
                for intent_id in &result.correct_intents {
                    ig.reinforce(&word_refs, intent_id, DELTA_REINFORCE);
                    eprintln!("[auto-learn/L2] correct  → reinforce {:?} → '{}' ({:+.2})", word_refs, intent_id, DELTA_REINFORCE);
                }
                for intent_id in &result.missed_intents {
                    ig.reinforce(&word_refs, intent_id, DELTA_MISS);
                    eprintln!("[auto-learn/L2] missed   → reinforce {:?} → '{}' ({:+.2})", word_refs, intent_id, DELTA_MISS);
                }
                for intent_id in &result.wrong_detections {
                    ig.reinforce(&word_refs, intent_id, DELTA_SUPPRESS);
                    eprintln!("[auto-learn/L2] wrong    → suppress  {:?} → '{}' ({:+.2})", word_refs, intent_id, DELTA_SUPPRESS);
                }

                // Show resulting edge weights for transparency
                for w in &word_refs {
                    if let Some(edges) = ig.word_intent.get(*w) {
                        eprintln!("[auto-learn/L2] edges after: '{}' → {:?}", w,
                            edges.iter().map(|(id, wt)| format!("{}:{:.2}", id, wt)).collect::<Vec<_>>());
                    }
                }

                // ── L2 conjunction reinforcement ────────────────────────────
                // Which conjunction rules fired for this query?
                // Reinforce for correct intents, suppress for wrong intents.
                let fired = ig.fired_conjunction_indices(&word_refs);
                for idx in &fired {
                    let intent = ig.conjunctions[*idx].intent.clone();
                    if result.correct_intents.contains(&intent) {
                        ig.reinforce_conjunction(*idx, DELTA_REINFORCE);
                        eprintln!("[auto-learn/L2] conjunction[{}] → '{}' reinforce ({:+.2})", idx, intent, DELTA_REINFORCE);
                    } else if result.wrong_detections.contains(&intent) {
                        ig.reinforce_conjunction(*idx, DELTA_SUPPRESS);
                        eprintln!("[auto-learn/L2] conjunction[{}] → '{}' suppress ({:+.2})", idx, intent, DELTA_SUPPRESS);
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
                let orig_words = crate::hebbian::HebbianGraph::l1_tokens_pub(original_query);
                for word in &orig_words {
                    if let Some(edges) = heb.edges.get(word.as_str()) {
                        for edge in edges.clone() {
                            if matches!(edge.kind, crate::hebbian::EdgeKind::Synonym)
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

        // ── L1 morphology: learn variants of brand-new vocabulary ────────────
        // If "ping" was just added to L3, we need L1 to know "pinging","pinged","pings"→"ping"
        // so those variants route correctly on the next query.
        if !new_vocabulary.is_empty() {
            eprintln!("[auto-learn/L1] new vocabulary found: {:?} — requesting morphological variants", new_vocabulary);
            learn_l1_morphology(state, app_id, &new_vocabulary, original_query).await;
        }
    }

    (added, replaced)
}

/// Ask the LLM for morphological variants of newly discovered words and add them to L1.
/// This closes the gap where L3 learns "ping"→send_message but L1 doesn't know "pinging"→"ping".
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
