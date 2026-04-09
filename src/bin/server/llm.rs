//! LLM integration: call_llm, seed pipeline, full 3-turn review.

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

// --- Shared seed pipeline: guard + one LLM retry ---

/// Result of the seed pipeline.
#[derive(Debug, Clone, serde::Serialize)]
#[derive(serde::Deserialize)]
pub struct SeedPipelineResult {
    /// Seeds that were successfully added: (intent_id, seed)
    pub added: Vec<(String, String)>,
    /// Seeds permanently blocked after retry: (intent_id, seed, reason)
    pub blocked: Vec<(String, String, String)>,
    /// How many seeds were retried via LLM
    pub retried: usize,
    /// LLM-suggested alternatives for user-edited seeds (when auto_apply=false)
    pub suggestions: Vec<(String, String)>, // (intent_id, suggested_seed)
}

/// Shared seed pipeline used by all flows that add seeds.
///
/// 1. Runs each seed through `add_seed_checked`
/// 2. Collects blocked seeds with reasons
/// 3. If LLM available: one retry with collision info
/// 4. auto_apply_retry=true: silently apply alternatives (for LLM-generated seeds)
///    auto_apply_retry=false: return alternatives as suggestions (for user-edited seeds)
pub async fn seed_pipeline(
    state: &AppState,
    app_id: &str,
    seeds_by_intent: &HashMap<String, Vec<String>>,
    auto_apply_retry: bool,
) -> SeedPipelineResult {
    let mut added = Vec::new();
    let mut blocked_for_retry: Vec<(String, String, String)> = Vec::new(); // (intent, seed, reason)
    let mut blocked_final = Vec::new();

    // Step 1: Try all seeds through the guard
    {
        let mut routers = state.routers.write().unwrap();
        if let Some(router) = routers.get_mut(app_id) {
            for (intent_id, seeds) in seeds_by_intent {
                for seed in seeds {
                    let s = seed.trim();
                    if s.is_empty() { continue; }
                    let result = router.add_seed_checked(intent_id, s, "en");
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

    // Step 2: If any collisions and LLM available, one retry
    let mut retried = 0;
    let mut suggestions = Vec::new();

    if !blocked_for_retry.is_empty() && state.llm_key.is_some() {
        let blocked_desc: String = blocked_for_retry.iter()
            .map(|(intent, seed, reason)| format!("  \"{}\" → {}: {}", seed, intent, reason))
            .collect::<Vec<_>>().join("\n");

        let retry_prompt = format!(
            "These seed phrases were REJECTED by the collision guard:\n{}\n\n\
             The guard blocks seeds containing terms that are exclusive to other intents.\n\
             For each rejected seed, suggest ONE alternative that:\n\
             - Avoids the specific conflicting terms mentioned above\n\
             - Uses completely different vocabulary\n\
             - Still captures the same meaning\n\n\
             {}\n\n\
             Respond with ONLY JSON:\n\
             {{\"seeds_by_intent\": {{\"intent_name\": [\"alternative_seed\"]}}}}\n",
            blocked_desc, asv_router::seed::SEED_QUALITY_RULES
        );

        retried = blocked_for_retry.len();

        if let Ok(response) = call_llm(state, &retry_prompt, 500).await {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(extract_json(&response)) {
                if let Some(sbi) = parsed.get("seeds_by_intent").and_then(|v| v.as_object()) {
                    let mut routers = state.routers.write().unwrap();
                    if let Some(router) = routers.get_mut(app_id) {
                        for (intent_id, seeds) in sbi {
                            if let Some(arr) = seeds.as_array() {
                                for seed in arr {
                                    if let Some(s) = seed.as_str() {
                                        if auto_apply_retry {
                                            let result = router.add_seed_checked(intent_id, s, "en");
                                            if result.added {
                                                added.push((intent_id.clone(), s.to_string()));
                                            } else {
                                                // Retry also blocked — give up on this one
                                                let reason = if !result.conflicts.is_empty() {
                                                    result.conflicts.iter()
                                                        .map(|c| format!("'{}' conflicts with {}", c.term, c.competing_intent))
                                                        .collect::<Vec<_>>().join("; ")
                                                } else {
                                                    result.warning.unwrap_or("still blocked".to_string())
                                                };
                                                blocked_final.push((intent_id.clone(), s.to_string(), reason));
                                            }
                                        } else {
                                            // Return as suggestion for user approval
                                            suggestions.push((intent_id.clone(), s.to_string()));
                                        }
                                    }
                                }
                            }
                        }
                        if auto_apply_retry && !added.is_empty() {
                            maybe_persist(state, app_id, router);
                        }
                    }
                }
            }
        }
    } else {
        // No LLM available — blocked seeds are final
        blocked_final.extend(blocked_for_retry);
    }

    SeedPipelineResult { added, blocked: blocked_final, retried, suggestions }
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
    /// Turn 1: detected languages
    pub languages: Vec<String>,
    /// Turn 2: seeds to add (passed guard + retry)
    pub seeds_to_add: HashMap<String, Vec<String>>,
    /// Turn 2: seeds blocked by guard
    pub seeds_blocked: Vec<(String, String, String)>, // (intent, seed, reason)
    /// Turn 3: seeds to replace in wrong intents
    pub seeds_to_replace: Vec<SeedReplacement>,
    /// Turn 3: safe to apply
    pub safe_to_apply: bool,
    /// Turn 3: summary
    pub summary: String,
}

#[derive(Debug, Clone, serde::Serialize)]
#[derive(serde::Deserialize)]
pub struct SeedReplacement {
    pub intent: String,
    pub old_seed: String,
    pub new_seed: String,
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
         Respond with ONLY JSON:\n\
         {{\"correct_intents\": [\"intent1\"], \"wrong_detections\": [\"wrong1\"], \"languages\": [\"en\"]}}\n",
        query, detected, intent_descriptions
    );

    let t1_response = call_llm(state, &turn1_prompt, 200).await
        .map_err(|e| format!("Turn 1 failed: {}", e.1))?;
    let t1_parsed: serde_json::Value = serde_json::from_str(extract_json(&t1_response))
        .map_err(|e| format!("Turn 1 parse failed: {}", e))?;

    let correct_intents: Vec<String> = t1_parsed["correct_intents"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let wrong_detections: Vec<String> = t1_parsed["wrong_detections"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let languages: Vec<String> = t1_parsed["languages"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_else(|| vec!["en".to_string()]);

    eprintln!("[full_review] Turn 1: correct={:?}, wrong={:?}, langs={:?}", correct_intents, wrong_detections, languages);

    // === Turn 2: Fix misses (seeds to add) ===
    let existing_seeds: String = {
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
        format!("\nThe query is in {:?}. Generate seeds in the detected language.\n", languages)
    };

    let turn2_prompt = format!(
        "{}\n\n\
         Customer query: \"{}\"\n\
         Correct intents: {:?}\n\n\
         Current seeds in the index:\n{}\n\n\
         {}\
         {}\n\n\
         Respond with ONLY JSON:\n\
         {{\"seeds_by_intent\": {{\"intent_name\": [\"seed1\", \"seed2\"]}}}}\n",
        asv_router::seed::REVIEW_FIX_GUIDELINES,
        query, correct_intents, existing_seeds, lang_instruction,
        asv_router::seed::SEED_QUALITY_RULES
    );

    let t2_response = call_llm(state, &turn2_prompt, 300).await
        .map_err(|e| format!("Turn 2 failed: {}", e.1))?;
    let t2_parsed: serde_json::Value = serde_json::from_str(extract_json(&t2_response))
        .map_err(|e| {
            eprintln!("[full_review] Turn 2 parse error: {}. Raw: {}", e, &t2_response[..t2_response.len().min(300)]);
            format!("Turn 2 parse failed: {}", e)
        })?;

    // Pre-validate seeds through guard (read-only check + retry)
    let mut seeds_to_add: HashMap<String, Vec<String>> = HashMap::new();
    let mut seeds_blocked = Vec::new();

    if let Some(sbi) = t2_parsed.get("seeds_by_intent").and_then(|v| v.as_object()) {
        let mut blocked_for_retry: Vec<(String, String, String)> = Vec::new();

        {
            let routers = state.routers.read().unwrap();
            if let Some(router) = routers.get(app_id) {
                for (intent_id, seeds) in sbi {
                    if let Some(arr) = seeds.as_array() {
                        for seed in arr {
                            if let Some(s) = seed.as_str() {
                                let check = router.check_seed(intent_id, s);
                                if check.conflicts.is_empty() && !check.redundant
                                    && check.warning.as_deref() != Some("No content terms after tokenization")
                                {
                                    seeds_to_add.entry(intent_id.clone()).or_default().push(s.to_string());
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

        // Feedback loop: one retry for blocked seeds
        if !blocked_for_retry.is_empty() && state.llm_key.is_some() {
            let blocked_desc: String = blocked_for_retry.iter()
                .map(|(intent, seed, reason)| format!("  \"{}\" → {}: {}", seed, intent, reason))
                .collect::<Vec<_>>().join("\n");

            let retry_prompt = format!(
                "These seed phrases were REJECTED by the collision guard:\n{}\n\n\
                 The guard blocks seeds containing terms exclusive to other intents.\n\
                 For each rejected seed, suggest ONE alternative that avoids the conflicting terms.\n\
                 Use completely different vocabulary that still captures the same meaning.\n\n\
                 {}\n\n\
                 Respond with ONLY JSON:\n\
                 {{\"seeds_by_intent\": {{\"intent_name\": [\"alternative_seed\"]}}}}\n",
                blocked_desc, asv_router::seed::SEED_QUALITY_RULES
            );

            if let Ok(retry_resp) = call_llm(state, &retry_prompt, 300).await {
                if let Ok(retry_parsed) = serde_json::from_str::<serde_json::Value>(extract_json(&retry_resp)) {
                    if let Some(retry_sbi) = retry_parsed.get("seeds_by_intent").and_then(|v| v.as_object()) {
                        let routers = state.routers.read().unwrap();
                        if let Some(router) = routers.get(app_id) {
                            for (intent_id, seeds) in retry_sbi {
                                if let Some(arr) = seeds.as_array() {
                                    for seed in arr {
                                        if let Some(s) = seed.as_str() {
                                            let check = router.check_seed(intent_id, s);
                                            if check.conflicts.is_empty() && !check.redundant {
                                                seeds_to_add.entry(intent_id.clone()).or_default().push(s.to_string());
                                            } else {
                                                seeds_blocked.push((intent_id.clone(), s.to_string(), "still blocked after retry".to_string()));
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

    eprintln!("[full_review] Turn 2: seeds_to_add={:?}, blocked={}", seeds_to_add, seeds_blocked.len());

    // === Turn 3: Fix false positives (replace broad seeds) ===
    let mut seeds_to_replace = Vec::new();
    let mut safe_to_apply = true;
    let mut summary = String::new();

    if !wrong_detections.is_empty() {
        let wrong_intent_seeds: String = {
            let routers = state.routers.read().unwrap();
            if let Some(router) = routers.get(app_id) {
                wrong_detections.iter().map(|id| {
                    let seeds = router.get_training(id).unwrap_or_default();
                    format!("- {}: {:?}", id, seeds.iter().take(10).collect::<Vec<_>>())
                }).collect::<Vec<_>>().join("\n")
            } else { String::new() }
        };

        let turn3_prompt = format!(
            "Customer query: \"{}\"\n\n\
             These intents were WRONGLY detected (false positives):\n{}\n\n\
             Seeds being added for correct intents:\n{}\n\n\
             For each wrong intent:\n\
             1. Which seed caused the false match?\n\
             2. Suggest a MORE SPECIFIC replacement seed that keeps the intent's coverage but stops matching this query.\n\
             Do NOT suggest removing seeds — suggest replacements that narrow the match.\n\n\
             Replacement seed quality rules:\n\
             {}\n\n\
             Respond with ONLY JSON:\n\
             {{\n\
               \"replacements\": [{{\"intent\": \"name\", \"old_seed\": \"the broad seed\", \"new_seed\": \"more specific version\", \"reason\": \"why\"}}],\n\
               \"safe_to_apply\": true,\n\
               \"summary\": \"one sentence\"\n\
             }}\n\
             If no replacement needed, use empty replacements array.\n",
            query,
            wrong_intent_seeds,
            seeds_to_add.iter().map(|(id, seeds)| format!("  {}: {:?}", id, seeds)).collect::<Vec<_>>().join("\n"),
            asv_router::seed::SEED_QUALITY_RULES,
        );

        if let Ok(t3_response) = call_llm(state, &turn3_prompt, 1024).await {
            if let Ok(t3_parsed) = serde_json::from_str::<serde_json::Value>(extract_json(&t3_response)) {
                if let Some(replacements) = t3_parsed["replacements"].as_array() {
                    for r in replacements {
                        if let (Some(intent), Some(old), Some(new), Some(reason)) = (
                            r["intent"].as_str(), r["old_seed"].as_str(),
                            r["new_seed"].as_str(), r["reason"].as_str(),
                        ) {
                            seeds_to_replace.push(SeedReplacement {
                                intent: intent.to_string(),
                                old_seed: old.to_string(),
                                new_seed: new.to_string(),
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

    eprintln!("[full_review] Turn 3: replacements={}, safe={}, summary={}", seeds_to_replace.len(), safe_to_apply, summary);

    Ok(FullReviewResult {
        correct_intents,
        wrong_detections,
        languages,
        seeds_to_add,
        seeds_blocked,
        seeds_to_replace,
        safe_to_apply,
        summary,
    })
}

/// Apply a full review result: add seeds + replace broad seeds.
pub async fn apply_review(
    state: &AppState,
    app_id: &str,
    result: &FullReviewResult,
) -> (usize, usize) { // (seeds_added, seeds_replaced)
    let mut added = 0;
    let mut replaced = 0;

    // Apply seeds_to_add through pipeline
    if !result.seeds_to_add.is_empty() {
        let pipeline_result = seed_pipeline(state, app_id, &result.seeds_to_add, true).await;
        added = pipeline_result.added.len();
    }

    // Apply seed replacements (remove old + add new through guard)
    if !result.seeds_to_replace.is_empty() {
        let mut routers = state.routers.write().unwrap();
        if let Some(router) = routers.get_mut(app_id) {
            for replacement in &result.seeds_to_replace {
                // Verify old seed exists before removing
                let old_exists = router.get_training(&replacement.intent)
                    .map(|seeds| seeds.contains(&replacement.old_seed))
                    .unwrap_or(false);

                if old_exists {
                    // Add new seed first (through guard)
                    let check = router.add_seed_checked(&replacement.intent, &replacement.new_seed, "en");
                    if check.added {
                        // Only remove old if new was accepted
                        router.remove_seed(&replacement.intent, &replacement.old_seed);
                        replaced += 1;
                        eprintln!("[apply_review] Replaced in {}: \"{}\" → \"{}\"",
                            replacement.intent, replacement.old_seed, replacement.new_seed);
                    } else {
                        eprintln!("[apply_review] Replacement blocked for {}: new seed \"{}\" rejected",
                            replacement.intent, replacement.new_seed);
                    }
                }
            }
            maybe_persist(state, app_id, router);
        }
    }

    (added, replaced)
}

