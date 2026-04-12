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
pub fn extract_json(text: &str) -> &str {
    let trimmed = text.trim();
    // Strip ```json ... ``` or ``` ... ```
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return &trimmed[start..=end];
        }
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
    /// Turn 2: phrases to add (passed guard + retry)
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
            phrases_to_add: HashMap::new(),
            phrases_blocked: Vec::new(),
            phrases_to_replace: Vec::new(),
            safe_to_apply: true,
            summary: "Detection correct, no changes needed.".to_string(),
        });
    }

    // === Turn 2: Fix misses (phrases to add) ===
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
        phrases_to_add,
        phrases_blocked,
        phrases_to_replace,
        safe_to_apply,
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

    // Apply phrases_to_add through pipeline
    if !result.phrases_to_add.is_empty() {
        let pipeline_result = phrase_pipeline(state, app_id, &result.phrases_to_add, true, "en").await;
        added = pipeline_result.added.len();

        // Learn situation n-grams from the failing query for each intent that got phrases.
        // This lets the situation index pick up state signals from real failed queries.
        if added > 0 {
            let seen_intents: std::collections::HashSet<String> =
                pipeline_result.added.iter().map(|(intent, _)| intent.clone()).collect();
            let mut routers = state.routers.write().unwrap();
            if let Some(router) = routers.get_mut(app_id) {
                for intent_id in &seen_intents {
                    router.learn_situation(original_query, intent_id);
                }
                maybe_persist(state, app_id, router);
            }
        }
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

    (added, replaced)
}

