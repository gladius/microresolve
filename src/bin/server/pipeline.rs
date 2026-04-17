//! Auto-learn pipeline: the single unified learning path used by all flows.
//!
//! # Pipeline
//! Every learning trigger (auto worker, simulate, manual, review) calls:
//!   1. `full_review` — judges the routing result and generates phrases for misses.
//!      - With `ground_truth: Some(gt)` (simulate/training): Turn 1 LLM is skipped;
//!        correct/missed/wrong are computed by set math — cheaper and exact.
//!      - With `ground_truth: None` (auto worker / review): Turn 1 LLM judges.
//!      - Turn 2 LLM always runs for missed intents → generates candidate phrases.
//!   2. `apply_review` — applies the `FullReviewResult`:
//!      - phrase_pipeline → adds phrases to the L0 inverted-index router
//!      - L2 Hebbian learn_phrase for accepted phrases
//!      - L3 anti-Hebbian learn_inhibition for false-positive pairs
//!      - L1 synonym reinforcement for edges that fired
//!      - L1 synonym + morphology discovery (LLM) for new vocabulary in misses
//!
//! # LLM utilities
//! `call_llm`, `call_llm_smart`, `extract_json` at the bottom of this file.

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
/// Call the "smart" model for one-time, high-quality tasks (e.g. L1 import seeding).
/// Uses LLM_SMART_MODEL if set, otherwise falls back to LLM_MODEL.
/// Set in .env: LLM_SMART_MODEL=claude-sonnet-4-6
#[allow(dead_code)]
pub async fn call_llm_smart(
    state: &ServerState,
    prompt: &str,
    max_tokens: u32,
) -> Result<String, (StatusCode, String)> {
    let smart = std::env::var("LLM_SMART_MODEL").ok();
    if let Some(model) = smart {
        eprintln!("[llm] using smart model: {}", model);
        call_llm_with_model(state, prompt, max_tokens, &model).await
    } else {
        call_llm(state, prompt, max_tokens).await
    }
}

pub async fn call_llm(
    state: &ServerState,
    prompt: &str,
    max_tokens: u32,
) -> Result<String, (StatusCode, String)> {
    let provider = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "anthropic".to_string());
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| match provider.as_str() {
        "gemini" => "gemini-2.5-flash".to_string(),
        _ => "claude-haiku-4-5-20251001".to_string(),
    });
    call_llm_with_model(state, prompt, max_tokens, &model).await
}

/// Call LLM with a full messages array (system + history + user).
#[allow(dead_code)]
pub async fn call_llm_with_messages(
    state: &ServerState,
    messages: &[serde_json::Value],
    max_tokens: u32,
) -> Result<String, (StatusCode, String)> {
    let key = state.llm_key.as_ref().ok_or_else(|| {
        (StatusCode::SERVICE_UNAVAILABLE, "LLM_API_KEY not set".to_string())
    })?;
    let provider = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "anthropic".to_string());
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| match provider.as_str() {
        "gemini" => "gemini-2.5-flash".to_string(),
        _ => "claude-haiku-4-5-20251001".to_string(),
    });

    let resp = match provider.as_str() {
        "anthropic" => {
            // Anthropic: system goes in top-level, not in messages
            let system = messages.iter()
                .find(|m| m["role"].as_str() == Some("system"))
                .and_then(|m| m["content"].as_str())
                .unwrap_or("");
            let non_system: Vec<&serde_json::Value> = messages.iter()
                .filter(|m| m["role"].as_str() != Some("system"))
                .collect();
            let url = std::env::var("LLM_API_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com/v1/messages".to_string());
            let body = serde_json::json!({
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": non_system,
            });
            state.http.post(&url)
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&body).send().await
        }
        _ => {
            // OpenAI-compatible (Groq, OpenAI, etc.) — messages array as-is
            let url = std::env::var("LLM_API_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string());
            let body = serde_json::json!({
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            });
            state.http.post(&url)
                .header("Authorization", format!("Bearer {}", key))
                .header("content-type", "application/json")
                .json(&body).send().await
        }
    }.map_err(|e| (StatusCode::BAD_GATEWAY, format!("LLM request failed: {}", e)))?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return Err((StatusCode::BAD_GATEWAY, format!("LLM API {}: {}", status, text)));
    }

    let data: serde_json::Value = resp.json().await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Bad response: {}", e)))?;

    // Extract text based on provider
    if provider == "anthropic" {
        data["content"][0]["text"].as_str()
            .map(|s| s.trim().to_string())
            .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No text in response".to_string()))
    } else {
        data["choices"][0]["message"]["content"].as_str()
            .map(|s| s.trim().to_string())
            .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No text in response".to_string()))
    }
}

async fn call_llm_with_model(
    state: &ServerState,
    prompt: &str,
    max_tokens: u32,
    model: &str,
) -> Result<String, (StatusCode, String)> {
    match call_llm_once(state, prompt, max_tokens, model).await {
        Ok(text) => Ok(text),
        Err((status, msg)) => {
            // One retry after short wait for rate limits (free tier APIs)
            let is_rate_limit = status == StatusCode::TOO_MANY_REQUESTS
                || msg.contains("429") || msg.contains("rate") || msg.contains("quota")
                || status == StatusCode::SERVICE_UNAVAILABLE
                || msg.contains("503") || msg.contains("overloaded");

            if is_rate_limit {
                eprintln!("[llm] rate limited, waiting 3s then retrying once");
                tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
                call_llm_once(state, prompt, max_tokens, model).await
            } else {
                Err((status, msg))
            }
        }
    }
}

async fn call_llm_once(
    state: &ServerState,
    prompt: &str,
    max_tokens: u32,
    model: &str,
) -> Result<String, (StatusCode, String)> {
    let key = state.llm_key.as_ref().ok_or_else(|| {
        (StatusCode::SERVICE_UNAVAILABLE, "LLM_API_KEY not set. Add it to .env file.".to_string())
    })?;

    let provider = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "anthropic".to_string());

    // Gemini thinking models need higher output budget (thinking tokens count against limit)
    // Gemini models: use reasonable output budget (not too high — causes slow thinking)
    let effective_max = if provider == "gemini" { max_tokens.max(512) } else { max_tokens };

    let resp = match provider.as_str() {
        "gemini" => {
            // Google Gemini API: key in query param, different body format
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                model, key
            );
            let body = serde_json::json!({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": effective_max,
                    "temperature": 0.3,
                    "thinkingConfig": { "thinkingBudget": 0 }
                }
            });
            state.http
                .post(&url)
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
        }
        "anthropic" => {
            let url = std::env::var("LLM_API_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com/v1/messages".to_string());
            let body = serde_json::json!({
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            });
            state.http
                .post(&url)
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
        }
        _ => {
            // OpenAI-compatible (OpenAI, Ollama, Groq, DeepSeek, etc.)
            let url = std::env::var("LLM_API_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string());
            let body = serde_json::json!({
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            });
            state.http
                .post(&url)
                .header("Authorization", format!("Bearer {}", key))
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
        }
    }.map_err(|e| (StatusCode::BAD_GATEWAY, format!("LLM request failed: {}", e)))?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return Err((StatusCode::BAD_GATEWAY, format!("LLM API {}: {}", status, text)));
    }

    let data: serde_json::Value = resp.json().await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Bad response: {}", e)))?;

    // Extract text based on provider response format
    match provider.as_str() {
        "gemini" => {
            // Gemini: candidates[0].content.parts[0].text
            data["candidates"][0]["content"]["parts"][0]["text"]
                .as_str()
                .map(|s| s.trim().to_string())
                .ok_or_else(|| {
                    let raw = serde_json::to_string(&data).unwrap_or_default();
                    (StatusCode::BAD_GATEWAY, format!("Invalid JSON from LLM: {}", &raw[..raw.len().min(200)]))
                })
        }
        "anthropic" => {
            data["content"][0]["text"]
                .as_str()
                .map(|s| s.trim().to_string())
                .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No text in response".to_string()))
        }
        _ => {
            data["choices"][0]["message"]["content"]
                .as_str()
                .map(|s| s.trim().to_string())
                .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No text in response".to_string()))
        }
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

/// Add phrases to Router storage. No collision guard — IDF self-regulates.
/// Duplicates and empty phrases are silently skipped.
pub async fn phrase_pipeline(
    state: &AppState,
    app_id: &str,
    phrases_by_intent: &HashMap<String, Vec<String>>,
    _auto_apply_retry: bool,
    lang: &str,
) -> PhrasePipelineResult {
    let mut added = Vec::new();
    let blocked_final = Vec::new();

    {
        let mut routers = state.routers.write().unwrap();
        if let Some(router) = routers.get_mut(app_id) {
            for (intent_id, phrases) in phrases_by_intent {
                for phrase in phrases {
                    let s = phrase.trim();
                    if s.is_empty() { continue; }
                    // add_phrase_checked still skips duplicates and empty — just no collision blocking
                    let result = router.add_phrase_checked(intent_id, s, lang);
                    if result.added {
                        added.push((intent_id.clone(), s.to_string()));
                    }
                }
            }
            if !added.is_empty() {
                maybe_persist(state, app_id, router);
            }
        }
    }

    PhrasePipelineResult {
        added,
        blocked: blocked_final,
        initially_blocked: 0,
        recovered_by_retry: 0,
        suggestions: vec![],
    }
}

// --- Shared full review: 3 turns + guard + apply ---

/// A phrase that was blocked by the collision guard.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockedPhrase {
    pub intent: String,
    pub phrase: String,
    pub reason: String,
}

/// Result of a full review — returned by `full_review`, consumed by `apply_review`.
///
/// This is the single data contract between the review step and the apply step,
/// used by all flows: auto worker, simulate, manual, and review queue.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FullReviewResult {
    /// Intents that were correctly detected
    pub correct_intents: Vec<String>,
    /// Intents that were incorrectly detected (false positives) — triggers L3 inhibition
    pub wrong_detections: Vec<String>,
    /// Intents the query expressed but the router missed — triggers phrase gen + L2 learn
    pub missed_intents: Vec<String>,
    /// Detected query languages (used to pick phrase generation language)
    pub languages: Vec<String>,
    /// True when detection was perfect — no learning needed, Turns 2+ were skipped
    pub detection_perfect: bool,
    /// Phrases to add per intent (passed the collision guard in Turn 2)
    pub phrases_to_add: HashMap<String, Vec<String>>,
    /// Phrases blocked by the collision guard after retry
    pub phrases_blocked: Vec<BlockedPhrase>,
    /// Human-readable summary (empty when detection_perfect=true)
    pub summary: String,
    /// LLM-extracted spans from the query that express each intent.
    /// Used to learn precise n-gram patterns from the customer's own words.
    #[serde(default)]
    pub spans_to_learn: Vec<(String, String)>,  // (intent_id, span text)
}


/// Run the full review for a query.
///
/// `ground_truth`: when provided (simulate/manual flows), Turn 1 LLM is skipped entirely —
/// correct/missed/wrong are computed by set math, which is both cheaper and more accurate.
/// When `None` (auto worker), Turn 1 LLM judges correctness from detected intents alone.
pub async fn full_review(
    state: &AppState,
    app_id: &str,
    query: &str,
    detected: &[String],
    ground_truth: Option<&[String]>,
) -> Result<FullReviewResult, String> {

    // ── When ground truth is known: skip Turn 1 LLM entirely ──────────────────
    if let Some(gt) = ground_truth {
        use std::collections::HashSet;
        let gt_set:  HashSet<&str> = gt.iter().map(|s| s.as_str()).collect();
        let det_set: HashSet<&str> = detected.iter().map(|s| s.as_str()).collect();

        let correct_intents:  Vec<String> = detected.iter().filter(|s| gt_set.contains(s.as_str())).cloned().collect();
        let wrong_detections: Vec<String> = detected.iter().filter(|s| !gt_set.contains(s.as_str())).cloned().collect();
        let missed_intents:   Vec<String> = gt.iter().filter(|s| !det_set.contains(s.as_str())).cloned().collect();

        eprintln!("[full_review] ground_truth provided — skipping Turn 1. correct={:?} wrong={:?} missed={:?}",
            correct_intents, wrong_detections, missed_intents);

        if wrong_detections.is_empty() && missed_intents.is_empty() {
            eprintln!("[full_review] detection perfect");
            return Ok(FullReviewResult {
                correct_intents,
                wrong_detections,
                missed_intents,
                languages: vec!["en".to_string()],
                detection_perfect: true,
                phrases_to_add: HashMap::new(),
                phrases_blocked: Vec::new(),
                summary: "Detection correct, no changes needed.".to_string(),
                spans_to_learn: vec![],
            });
        }

        // Run Turn 2 with the computed sets (reuse shared helper below)
        return full_review_from_sets(
            state, app_id, query,
            correct_intents, wrong_detections, missed_intents,
            vec!["en".to_string()],
        ).await;
    }

    // ── Turn 1 LLM judge (auto worker — ground truth unknown) ─────────────────
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
    // Also asks: is this query relevant to this namespace at all?
    // out_of_scope = true means the query has no domain signal for any intent here —
    // pure conversational filler or a completely different domain. Skip all learning.
    let turn1_prompt = format!(
        "Customer query: \"{query}\"\n\
         {l1_context}\
         Router detected:\n{detected_with_scores}\n\n\
         Available intents:\n{intent_labels}\n\n\
         Which intents does this query EXPLICITLY express? Only literal, not implied.\n\
         Which detected intents are WRONG (false positives)?\n\
         Which intents does the query express that were NOT detected (missed)?\n\
         What language is the query in?\n\
         Is this query completely irrelevant to all available intents? (pure filler, wrong domain, only pronouns with no actionable signal)\n\
         Respond with ONLY JSON:\n\
         {{\"correct_intents\": [\"intent_id\"], \"wrong_detections\": [\"intent_id\"], \"missed_intents\": [\"intent_id\"], \"languages\": [\"en\"], \"out_of_scope\": false}}\n"
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
    let out_of_scope = t1_parsed["out_of_scope"].as_bool().unwrap_or(false);

    eprintln!("[full_review] Turn 1: correct={:?}, wrong={:?}, missed={:?}, langs={:?}, out_of_scope={}",
        correct_intents, wrong_detections, missed_intents, languages, out_of_scope);

    // === Early exit: out of scope — query irrelevant to this namespace ===
    if out_of_scope {
        eprintln!("[full_review] Query out of scope for namespace — skipping all learning");
        return Ok(FullReviewResult {
            correct_intents: vec![],
            wrong_detections: vec![],
            missed_intents: vec![],
            languages,
            detection_perfect: false,
            phrases_to_add: HashMap::new(),
            phrases_blocked: Vec::new(),
            summary: "Query out of scope for this namespace — no learning applied.".to_string(),
            spans_to_learn: vec![],
        });
    }

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
            summary: "Detection correct, no changes needed.".to_string(),
            spans_to_learn: vec![],
        });
    }

    full_review_from_sets(
        state, app_id, query,
        correct_intents, wrong_detections, missed_intents, languages,
    ).await
}

/// Shared Turn 2 logic — runs after correct/missed/wrong are known (either from Turn 1 LLM or set math).
async fn full_review_from_sets(
    state: &AppState,
    app_id: &str,
    query: &str,
    correct_intents: Vec<String>,
    wrong_detections: Vec<String>,
    missed_intents: Vec<String>,
    languages: Vec<String>,
) -> Result<FullReviewResult, String> {
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

    // Build missed intent labels with phrase count — informs Turn 2 how saturated each intent is.
    // Intents with many phrases need more targeted additions; sparsely covered ones need breadth.
    let missed_labels: String = {
        let routers = state.routers.read().unwrap();
        if let Some(router) = routers.get(app_id) {
            missed_intents.iter().map(|id| {
                let desc  = router.get_description(id);
                let count = router.get_training(id).unwrap_or_default().len();
                let coverage = if count >= 20 {
                    format!(" [{} phrases — well covered, be very targeted]", count)
                } else if count >= 10 {
                    format!(" [{} phrases — add vocabulary not yet represented]", count)
                } else {
                    format!(" [{} phrases — add diverse new vocabulary]", count)
                };
                if desc.is_empty() { format!("  - {}{}", id, coverage) }
                else { format!("  - {} ({}){}", id, desc, coverage) }
            }).collect::<Vec<_>>().join("\n")
        } else {
            missed_intents.iter().map(|id| format!("  - {}", id)).collect::<Vec<_>>().join("\n")
        }
    };

    let turn2_prompt = format!(
        "{guidelines}\n\n\
         Customer query: \"{query}\"\n\n\
         Intents that need more training coverage:\n{missed_labels}\n\n\
         Phrases already in the system for these intents (do not duplicate):\n{existing_phrases}\n\
         {lang_instruction}\
         {quality}\n\n\
         For each intent, respond with a phrase AND the key words from the customer query.\n\
         Respond with ONLY JSON:\n\
         {{\"phrases_by_intent\": {{\"intent_id\": \"new phrase\"}}, \"key_words\": {{\"intent_id\": [\"word1\", \"word2\"]}}}}\n",
        guidelines = asv_router::phrase::REVIEW_FIX_GUIDELINES,
        quality = asv_router::phrase::PHRASE_QUALITY_RULES,
        query = query,
    );

    let t2_response = call_llm(state, &turn2_prompt, 150).await
        .map_err(|e| format!("Turn 2 failed: {}", e.1))?;
    let t2_parsed: serde_json::Value = serde_json::from_str(extract_json(&t2_response))
        .map_err(|e| {
            eprintln!("[full_review] Turn 2 parse error: {}. Raw: {}", e, &t2_response[..t2_response.len().min(300)]);
            format!("Turn 2 parse failed: {}", e)
        })?;

    // Pre-validate phrases through guard (read-only check + retry)
    let mut phrases_to_add: HashMap<String, Vec<String>> = HashMap::new();
    let mut phrases_blocked = Vec::new();
    let mut spans_to_learn: Vec<(String, String)> = Vec::new(); // (intent_id, span from query)

    if let Some(sbi) = t2_parsed.get("phrases_by_intent").and_then(|v| v.as_object()) {
        let mut blocked_for_retry: Vec<(String, String, String)> = Vec::new();

        {
            let routers = state.routers.read().unwrap();
            if let Some(router) = routers.get(app_id) {
                for (intent_id, phrase_val) in sbi {
                    // Accept string or single-element array
                    let phrase_str = if let Some(s) = phrase_val.as_str() {
                        Some(s.to_string())
                    } else if let Some(arr) = phrase_val.as_array() {
                        arr.first().and_then(|v| v.as_str()).map(|s| s.to_string())
                    } else {
                        None
                    };
                    if let Some(s) = phrase_str {
                        let s = s.trim();
                        if s.is_empty() { continue; }
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

        // One retry for blocked phrases — ask for a single alternative string
        if !blocked_for_retry.is_empty() && state.llm_key.is_some() {
            let blocked_desc: String = blocked_for_retry.iter()
                .map(|(intent, phrase, reason)| format!("  \"{}\" → {}: {}", phrase, intent, reason))
                .collect::<Vec<_>>().join("\n");

            let retry_prompt = format!(
                "These training phrases were REJECTED by the collision guard:\n{}\n\n\
                 The guard blocks phrases containing terms exclusive to other intents.\n\
                 For each, suggest ONE alternative using completely different vocabulary.\n\n\
                 {}\n\n\
                 Respond with ONLY JSON:\n\
                 {{\"phrases_by_intent\": {{\"intent_name\": \"alternative phrase\"}}}}\n",
                blocked_desc, asv_router::phrase::PHRASE_QUALITY_RULES
            );

            if let Ok(retry_resp) = call_llm(state, &retry_prompt, 150).await {
                if let Ok(retry_parsed) = serde_json::from_str::<serde_json::Value>(extract_json(&retry_resp)) {
                    if let Some(retry_sbi) = retry_parsed.get("phrases_by_intent").and_then(|v| v.as_object()) {
                        let routers = state.routers.read().unwrap();
                        if let Some(router) = routers.get(app_id) {
                            for (intent_id, phrase_val) in retry_sbi {
                                let phrase_str = if let Some(s) = phrase_val.as_str() {
                                    Some(s.to_string())
                                } else if let Some(arr) = phrase_val.as_array() {
                                    arr.first().and_then(|v| v.as_str()).map(|s| s.to_string())
                                } else {
                                    None
                                };
                                if let Some(s) = phrase_str {
                                    let check = router.check_phrase(intent_id, s.trim());
                                    if check.conflicts.is_empty() && !check.redundant {
                                        phrases_to_add.entry(intent_id.clone()).or_default().push(s.trim().to_string());
                                    } else {
                                        phrases_blocked.push(BlockedPhrase { intent: intent_id.clone(), phrase: s, reason: "still blocked after retry".to_string() });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Extract key_words from LLM response — intent-bearing words from the query.
    // Format: {"key_words": {"intent_id": ["word1", "word2"]}}
    if let Some(kw_obj) = t2_parsed.get("key_words").and_then(|v| v.as_object()) {
        for (intent_id, words_val) in kw_obj {
            if let Some(words_arr) = words_val.as_array() {
                let words: Vec<String> = words_arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.trim().to_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect();
                if !words.is_empty() {
                    // Join words into a single string for span-style learning
                    spans_to_learn.push((intent_id.clone(), words.join(" ")));
                }
            }
        }
        eprintln!("[full_review] Turn 2: extracted {} key_word sets", spans_to_learn.len());
    }
    // Also check legacy "spans" key for backward compatibility
    if let Some(spans_obj) = t2_parsed.get("spans").and_then(|v| v.as_object()) {
        for (intent_id, span_val) in spans_obj {
            if let Some(span) = span_val.as_str() {
                let span = span.trim();
                if !span.is_empty() {
                    spans_to_learn.push((intent_id.clone(), span.to_string()));
                }
            }
        }
    }

    eprintln!("[full_review] Turn 2: phrases_to_add={:?}, blocked={}", phrases_to_add, phrases_blocked.len());

    let summary = String::new();

    Ok(FullReviewResult {
        correct_intents,
        wrong_detections,
        missed_intents,
        languages,
        detection_perfect: false,
        phrases_to_add,
        phrases_blocked,
        summary,
        spans_to_learn,
    })
}


/// Apply a full review result: add phrases, update L2 edges, add L1 synonyms.
pub async fn apply_review(
    state: &AppState,
    app_id: &str,
    result: &FullReviewResult,
    original_query: &str,
) -> usize {
    let mut added = 0;

    // Add phrases to Router storage (phrase registry) — collision guard applies.
    // L2 learning is NOT gated by the Router — word_intent learns from ALL phrases.
    if !result.phrases_to_add.is_empty() {
        let lang = result.languages.first().map(|s| s.as_str()).unwrap_or("en");
        let pipeline_result = phrase_pipeline(state, app_id, &result.phrases_to_add, true, lang).await;
        added = pipeline_result.added.len();
    }

    // ── L2 update ────────────────────────────────────────────────────────────
    // L2 learns all phrases + LLM-confirmed query words into word_intent.
    const DELTA_REINFORCE: f32 =  0.05;
    const DELTA_SUPPRESS:  f32 = -0.05;

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

        {
            let mut ig_map = state.intent_graph.write().unwrap();
            // Create IntentGraph for this namespace if it doesn't exist yet (fresh namespace).
            let ig = ig_map.entry(app_id.to_string())
                .or_insert_with(asv_router::scoring::IntentGraph::new);

            // Missed intents: learn LLM-generated phrase WORDS into L2.
            // Each phrase is tokenized and its words are learned as 1-grams.
            for (intent_id, phrases) in &result.phrases_to_add {
                for phrase in phrases {
                    let phrase_words: Vec<String> = asv_router::tokenizer::tokenize(phrase);
                    let phrase_refs: Vec<&str> = phrase_words.iter().map(|s| s.as_str()).collect();
                    ig.learn_phrase(&phrase_refs, intent_id);
                    eprintln!("[auto-learn/L2] missed → learn phrase words {:?} → '{}'", phrase_refs, intent_id);
                }
            }
            // Learn intent-bearing words from LLM-extracted query spans.
            // These are the ACTUAL user words the LLM confirmed as intent-relevant.
            // Critical for vocabulary growth: teaches the exact words users use.
            for (span_intent, span_text) in &result.spans_to_learn {
                let span_words: Vec<String> = asv_router::tokenizer::tokenize(span_text);
                let span_refs: Vec<&str> = span_words.iter().map(|s| s.as_str()).collect();
                ig.learn_query_words(&span_refs, span_intent);
                eprintln!("[auto-learn/query] span words {:?} → '{}'", span_refs, span_intent);
            }

            // Wrong detections: L3 inhibition — correct intent suppresses false positive.
            for wrong in &result.wrong_detections {
                for correct in &result.correct_intents {
                    ig.learn_inhibition(correct, wrong);
                    eprintln!("[auto-learn/L3] inhibit → '{}' suppresses '{}'", correct, wrong);
                }
            }

            // Direct L3 suppression: when GT confirms a detection is wrong AND the correct
            // intents weren't detected (correct_intents is empty), normal inhibition can't fire.
            // Instead, directly suppress the query's vocabulary from activating the wrong intents.
            // "fire together, wire apart" — these query words should NOT activate these intents.
            if result.correct_intents.is_empty() && !result.wrong_detections.is_empty() {
                for wrong in &result.wrong_detections {
                    ig.reinforce(&word_refs, wrong, DELTA_SUPPRESS);
                    eprintln!("[auto-learn/L3] direct suppress query tokens from '{}'", wrong);
                }
            }

            if let Some(ref dir) = state.data_dir {
                let path = format!("{}/{}/_intent_graph.json", dir, app_id);
                ig.save(&path).ok();
                eprintln!("[auto-learn/L2] graph persisted → {}", path);
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
                let orig_words = asv_router::scoring::LexicalGraph::l1_tokens_pub(original_query);
                for word in &orig_words {
                    if let Some(edges) = heb.edges.get(word.as_str()).cloned() {
                        for edge in edges {
                            if matches!(edge.kind, asv_router::scoring::EdgeKind::Synonym)
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

        // ── L1 learning: morphology + synonym discovery, only on misses ─────
        // Both are independent LLM calls — no shared state, safe to join.
        // Only runs when there are missed intents — that's when new vocabulary coverage matters.
        // Correct routing needs no L1 expansion; false positives are handled by L3 alone.
        if !result.missed_intents.is_empty() {
            // Words from the query not yet in L1 edges — candidates for morphology expansion
            let new_to_l1: Vec<String> = {
                let heb_map = state.hebbian.read().unwrap();
                if let Some(heb) = heb_map.get(app_id) {
                    word_refs.iter()
                        .filter(|&&w| !heb.edges.contains_key(w))
                        .map(|&w| w.to_string())
                        .collect()
                } else {
                    word_refs.iter().map(|&w| w.to_string()).collect()
                }
            };
            let do_morph = !new_to_l1.is_empty();

            if do_morph {
                eprintln!("[auto-learn/L1] parallel: morphology({:?}) + synonym discovery", new_to_l1);
                tokio::join!(
                    learn_l1_morphology(state, app_id, &new_to_l1, original_query),
                    learn_l1_synonyms(state, app_id, &word_refs, &result.missed_intents, original_query)
                );
            } else {
                eprintln!("[auto-learn/L1] synonym discovery only (all words already in L1)");
                learn_l1_synonyms(state, app_id, &word_refs, &result.missed_intents, original_query).await;
            }
        }
    }

    added
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
                                            asv_router::scoring::EdgeKind::Synonym);
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
                                                    asv_router::scoring::EdgeKind::Morphological);
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
