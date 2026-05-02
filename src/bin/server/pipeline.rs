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
//!      - phrase_pipeline → adds accepted phrases via `Resolver::index_phrase`
//!      - L2 Hebbian reinforcement on those phrases
//!      - Anti-Hebbian shrink (negative training) on tokens of wrong detections
//!
//! # LLM utilities
//! `call_llm`, `call_llm_smart`, `extract_json` at the bottom of this file.

use crate::state::*;
use axum::http::StatusCode;
use std::collections::HashMap;

/// Turn 1 (Judge): id + description only. No phrases — descriptions are the interface contract.
/// Flat cost regardless of how many phrases have been learned.
pub fn build_intent_labels(h: &microresolve::NamespaceHandle<'_>) -> String {
    let mut ids = h.intent_ids();
    ids.sort();
    ids.iter()
        .map(|id| {
            let desc = h.intent(id).map(|i| i.description).unwrap_or_default();
            if desc.is_empty() {
                format!("- {} [NO DESCRIPTION — cannot classify reliably]", id)
            } else {
                format!("- {} ({})", id, desc)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Turn 2 context: phrases for specific intents, capped to avoid token bloat.
/// Shows most-recently-added phrases first (they reflect what was learned, not just bootstrap seeds).
fn intent_phrases_context(
    h: &microresolve::NamespaceHandle<'_>,
    intent_ids: &[String],
    cap: usize,
) -> String {
    intent_ids
        .iter()
        .map(|id| {
            let desc = h.intent(id).map(|i| i.description).unwrap_or_default();
            let phrases = h.training(id).unwrap_or_default();
            let shown: Vec<&String> = phrases.iter().rev().take(cap).collect();
            let desc_str = if desc.is_empty() {
                String::new()
            } else {
                format!(" ({})", desc)
            };
            format!("  {}{}: {:?}", id, desc_str, shown)
        })
        .collect::<Vec<_>>()
        .join("\n")
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
            (Some(_), None) => true,
            _ => false,
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
    let last_array_end = trimmed.rfind(']');
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
    let provider = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "anthropic".to_string());
    let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| match provider.as_str() {
        "gemini" => "gemini-2.5-flash".to_string(),
        _ => "claude-haiku-4-5-20251001".to_string(),
    });
    call_llm_with_model(state, prompt, max_tokens, &model).await
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
                || msg.contains("429")
                || msg.contains("rate")
                || msg.contains("quota")
                || status == StatusCode::SERVICE_UNAVAILABLE
                || msg.contains("503")
                || msg.contains("overloaded");

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
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "LLM_API_KEY not set. Add it to .env file.".to_string(),
        )
    })?;

    let provider = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "anthropic".to_string());

    // Gemini thinking models need higher output budget (thinking tokens count against limit)
    // Gemini models: use reasonable output budget (not too high — causes slow thinking)
    let effective_max = if provider == "gemini" {
        max_tokens.max(512)
    } else {
        max_tokens
    };

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
            state
                .http
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
            state
                .http
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
            state
                .http
                .post(&url)
                .header("Authorization", format!("Bearer {}", key))
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
        }
    }
    .map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            format!("LLM request failed: {}", e),
        )
    })?;

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            format!("LLM API {}: {}", status, text),
        ));
    }

    let data: serde_json::Value = resp
        .json()
        .await
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
                    (
                        StatusCode::BAD_GATEWAY,
                        format!("Invalid JSON from LLM: {}", &raw[..raw.len().min(200)]),
                    )
                })
        }
        "anthropic" => data["content"][0]["text"]
            .as_str()
            .map(|s| s.trim().to_string())
            .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No text in response".to_string())),
        _ => data["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.trim().to_string())
            .ok_or_else(|| (StatusCode::BAD_GATEWAY, "No text in response".to_string())),
    }
}

// --- Shared phrase pipeline: guard + one LLM retry ---

/// Result of the phrase pipeline.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

/// Add phrases to Resolver storage. No collision guard — IDF self-regulates.
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

    if let Some(h) = state.engine.try_namespace(app_id) {
        for (intent_id, phrases) in phrases_by_intent {
            for phrase in phrases {
                let s = phrase.trim().to_string();
                if s.is_empty() {
                    continue;
                }
                let result = h.add_phrase(intent_id, &s, lang);
                if result.added {
                    added.push((intent_id.clone(), s));
                }
            }
        }
        if !added.is_empty() {
            maybe_commit(state, app_id);
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
    /// Intents the query expressed but the engine missed — triggers phrase gen + L2 learn
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
    pub spans_to_learn: Vec<(String, String)>, // (intent_id, span text)
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
        let gt_set: HashSet<&str> = gt.iter().map(|s| s.as_str()).collect();
        let det_set: HashSet<&str> = detected.iter().map(|s| s.as_str()).collect();

        let correct_intents: Vec<String> = detected
            .iter()
            .filter(|s| gt_set.contains(s.as_str()))
            .cloned()
            .collect();
        let wrong_detections: Vec<String> = detected
            .iter()
            .filter(|s| !gt_set.contains(s.as_str()))
            .cloned()
            .collect();
        let missed_intents: Vec<String> = gt
            .iter()
            .filter(|s| !det_set.contains(s.as_str()))
            .cloned()
            .collect();

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
            state,
            app_id,
            query,
            correct_intents,
            wrong_detections,
            missed_intents,
            vec!["en".to_string()],
        )
        .await;
    }

    // ── Confidence short-circuit (UI setting: review_skip_threshold) ──────────
    // If top detected intent scores above threshold, trust routing — skip Turn 1 LLM.
    let skip_threshold = state.ui_settings.read().unwrap().review_skip_threshold;
    if skip_threshold > 0.0 && !detected.is_empty() {
        let top_score = state
            .engine
            .try_namespace(app_id)
            .map(|h| {
                let (all_scores, _) = h.score_all(query);
                detected
                    .iter()
                    .filter_map(|id| all_scores.iter().find(|(s, _)| s == id).map(|(_, sc)| *sc))
                    .fold(0.0f32, f32::max)
            })
            .unwrap_or(0.0);
        if top_score >= skip_threshold {
            eprintln!("[full_review] confidence short-circuit: top score {:.2} >= threshold {:.2} — skipping Turn 1", top_score, skip_threshold);
            return Ok(FullReviewResult {
                correct_intents: detected.to_vec(),
                wrong_detections: vec![],
                missed_intents: vec![],
                languages: vec!["en".to_string()],
                detection_perfect: true,
                phrases_to_add: HashMap::new(),
                phrases_blocked: Vec::new(),
                summary: format!(
                    "High confidence ({:.0}%) — routing trusted, Turn 1 skipped.",
                    top_score * 100.0
                ),
                spans_to_learn: vec![],
            });
        }
    }

    // ── Turn 1 LLM judge (auto worker — ground truth unknown) ─────────────────
    let intent_labels = state
        .engine
        .try_namespace(app_id)
        .map(|h| build_intent_labels(&h))
        .unwrap_or_default();

    let detected_with_scores: String = state
        .engine
        .try_namespace(app_id)
        .map(|h| {
            let (all_scores, _) = h.score_all(query);
            let score_map: HashMap<&str, f32> =
                all_scores.iter().map(|(id, s)| (id.as_str(), *s)).collect();
            if detected.is_empty() {
                "(none detected)".to_string()
            } else {
                detected
                    .iter()
                    .map(|id| {
                        let score = score_map.get(id.as_str()).copied().unwrap_or(0.0);
                        format!("  {} (L2 score: {:.2})", id, score)
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        })
        .unwrap_or_else(|| {
            if detected.is_empty() {
                "(none detected)".to_string()
            } else {
                detected
                    .iter()
                    .map(|id| format!("  {}", id))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        });

    // === Turn 1: Judge ===
    // Also asks: is this query relevant to this namespace at all?
    // out_of_scope = true means the query has no domain signal for any intent here —
    // pure conversational filler or a completely different domain. Skip all learning.
    let turn1_prompt = format!(
        "Customer query: \"{query}\"\n\
         Resolver detected:\n{detected_with_scores}\n\n\
         Available intents:\n{intent_labels}\n\n\
         Which intents does this query EXPLICITLY express? Only literal, not implied.\n\
         Which detected intents are WRONG (false positives)?\n\
         Which intents does the query express that were NOT detected (missed)?\n\
         What language is the query in?\n\
         Is this query completely irrelevant to all available intents? (pure filler, wrong domain, only pronouns with no actionable signal)\n\
         Respond with ONLY JSON:\n\
         {{\"correct_intents\": [\"intent_id\"], \"wrong_detections\": [\"intent_id\"], \"missed_intents\": [\"intent_id\"], \"languages\": [\"en\"], \"out_of_scope\": false}}\n"
    );

    let t1_response = call_llm(state, &turn1_prompt, 256)
        .await
        .map_err(|e| format!("Turn 1 failed: {}", e.1))?;
    let t1_parsed: serde_json::Value = serde_json::from_str(extract_json(&t1_response))
        .map_err(|e| format!("Turn 1 parse failed: {}", e))?;

    let correct_intents: Vec<String> = t1_parsed["correct_intents"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let wrong_detections: Vec<String> = t1_parsed["wrong_detections"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let missed_intents: Vec<String> = t1_parsed["missed_intents"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let languages: Vec<String> = t1_parsed["languages"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_else(|| vec!["en".to_string()]);
    let out_of_scope = t1_parsed["out_of_scope"].as_bool().unwrap_or(false);

    eprintln!(
        "[full_review] Turn 1: correct={:?}, wrong={:?}, missed={:?}, langs={:?}, out_of_scope={}",
        correct_intents, wrong_detections, missed_intents, languages, out_of_scope
    );

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
        state,
        app_id,
        query,
        correct_intents,
        wrong_detections,
        missed_intents,
        languages,
    )
    .await
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
    // Fix #1 (2026-04-24): Turn 2 has no job when there are no missed intents.
    // Previously, Turn 2 still ran if wrong_detections was non-empty — leaving
    // the prompt with an empty "intents that need coverage" list, which led
    // Haiku to invent intents (observed: "account_balance_inquiry" on benign
    // CLINC queries). Skip Turn 2 entirely in the (wrong≠∅, missed=∅) case.
    // wrong_detections still reach apply_review's anti-Hebbian branch.
    if missed_intents.is_empty() {
        eprintln!(
            "[full_review] no missed intents — skipping Turn 2 (no phrase-generation needed)"
        );
        return Ok(FullReviewResult {
            correct_intents,
            wrong_detections,
            missed_intents,
            languages,
            detection_perfect: false,
            phrases_to_add: HashMap::new(),
            phrases_blocked: Vec::new(),
            summary: String::new(),
            spans_to_learn: Vec::new(),
        });
    }

    // === Turn 2: Fix misses — generate training phrases for missed intents ===
    // Show the MISSED intents' current phrases so LLM avoids duplicating what's already there.
    // Include descriptions so the LLM understands each intent's scope.
    let all_relevant_intents: Vec<String> = missed_intents
        .iter()
        .chain(correct_intents.iter())
        .cloned()
        .collect::<std::collections::LinkedList<_>>()
        .into_iter()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let existing_phrases: String = state
        .engine
        .try_namespace(app_id)
        .map(|h| intent_phrases_context(&h, &all_relevant_intents, 15))
        .unwrap_or_default();

    // Language instruction: if non-English, generate phrases in that language too
    let detected_lang = languages.first().map(|s| s.as_str()).unwrap_or("en");
    let lang_instruction = if detected_lang == "en" {
        String::new()
    } else {
        format!("\nThe query is in \"{detected_lang}\". Generate phrases in \"{detected_lang}\" for missed intents.\n")
    };

    let missed_labels: String = state
        .engine
        .try_namespace(app_id)
        .map(|h| {
            missed_intents
                .iter()
                .map(|id| {
                    let desc = h.intent(id).map(|i| i.description).unwrap_or_default();
                    let count = h.training(id).unwrap_or_default().len();
                    let coverage = if count >= 20 {
                        format!(" [{} phrases — well covered, be very targeted]", count)
                    } else if count >= 10 {
                        format!(" [{} phrases — add vocabulary not yet represented]", count)
                    } else {
                        format!(" [{} phrases — add diverse new vocabulary]", count)
                    };
                    if desc.is_empty() {
                        format!("  - {}{}", id, coverage)
                    } else {
                        format!("  - {} ({}){}", id, desc, coverage)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_else(|| {
            missed_intents
                .iter()
                .map(|id| format!("  - {}", id))
                .collect::<Vec<_>>()
                .join("\n")
        });

    let turn2_prompt = format!(
        "{guidelines}\n\n\
         Customer query: \"{query}\"\n\n\
         Intents that need more training coverage:\n{missed_labels}\n\n\
         Phrases already in the system for these intents (do not duplicate):\n{existing_phrases}\n\
         {lang_instruction}\
         Respond with ONLY JSON:\n\
         {{\"phrases_by_intent\": {{\"intent_id\": \"extracted span\"}}}}\n",
        guidelines = microresolve::phrase::REVIEW_FIX_GUIDELINES,
        query = query,
    );

    let t2_response = call_llm(state, &turn2_prompt, 150)
        .await
        .map_err(|e| format!("Turn 2 failed: {}", e.1))?;
    let t2_parsed: serde_json::Value =
        serde_json::from_str(extract_json(&t2_response)).map_err(|e| {
            eprintln!(
                "[full_review] Turn 2 parse error: {}. Raw: {}",
                e,
                &t2_response[..t2_response.len().min(300)]
            );
            format!("Turn 2 parse failed: {}", e)
        })?;

    // Pre-validate phrases through guard (read-only check + retry)
    let mut phrases_to_add: HashMap<String, Vec<String>> = HashMap::new();
    let phrases_blocked = Vec::new();
    let spans_to_learn: Vec<(String, String)> = Vec::new();

    if let Some(sbi) = t2_parsed
        .get("phrases_by_intent")
        .and_then(|v| v.as_object())
    {
        if let Some(h) = state.engine.try_namespace(app_id) {
            for (intent_id, phrase_val) in sbi {
                let exists = h.training(intent_id).is_some();
                if !exists {
                    eprintln!("[auto-learn/guard] skipping LLM-hallucinated intent '{}' (not in namespace)", intent_id);
                    continue;
                }
                let phrase_str = if let Some(s) = phrase_val.as_str() {
                    Some(s.to_string())
                } else if let Some(arr) = phrase_val.as_array() {
                    arr.first().and_then(|v| v.as_str()).map(|s| s.to_string())
                } else {
                    None
                };
                if let Some(s) = phrase_str {
                    let s = s.trim().to_string();
                    if s.is_empty() {
                        continue;
                    }
                    let check = h.check_phrase(intent_id, &s);
                    if !check.redundant
                        && check.warning.as_deref() != Some("No content terms after tokenization")
                    {
                        phrases_to_add.entry(intent_id.clone()).or_default().push(s);
                    }
                }
            }
        }
    }

    // spans_to_learn is now empty — span extraction was merged into phrases_by_intent.
    // The LLM extracts the intent-bearing span from the user's query and returns it
    // as the phrase directly, so it is stored in the phrase registry and survives rebuilds.

    eprintln!(
        "[full_review] Turn 2: phrases_to_add={:?}, blocked={}",
        phrases_to_add,
        phrases_blocked.len()
    );

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

/// Apply a full review result: add phrases, update L2 weights, anti-Hebbian
/// shrink on wrong detections.
pub async fn apply_review(
    state: &AppState,
    app_id: &str,
    result: &FullReviewResult,
    original_query: &str,
) -> usize {
    let mut added = 0;

    // Add phrases to Resolver storage (phrase registry) — collision guard applies.
    // L2 learning is NOT gated by the Resolver — word_intent learns from ALL phrases.
    if !result.phrases_to_add.is_empty() {
        let lang = result.languages.first().map(|s| s.as_str()).unwrap_or("en");
        let pipeline_result =
            phrase_pipeline(state, app_id, &result.phrases_to_add, true, lang).await;
        added = pipeline_result.added.len();
    }

    // ── Resolver-local mutations ──────────────────────────────────────────────
    // The deterministic learning core lives in microresolve::Resolver — server
    // just calls it. Bindings (python/node) get the same behaviour for free.
    let has_learning = !result.correct_intents.is_empty()
        || !result.missed_intents.is_empty()
        || !result.wrong_detections.is_empty();

    if has_learning {
        let Some(h) = state.engine.try_namespace(app_id) else {
            return added;
        };

        let no_phrases: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        h.apply_review_local(
            &no_phrases,
            &result.spans_to_learn,
            &result.wrong_detections,
            original_query,
            0.1,
        );
        if !result.wrong_detections.is_empty() {
            eprintln!(
                "[auto-learn/L2b] shrink weights on query tokens for wrong intents: {:?}",
                result.wrong_detections
            );
        }
        for (span_intent, span_text) in &result.spans_to_learn {
            eprintln!(
                "[auto-learn/query] span '{}' → '{}'",
                span_text, span_intent
            );
        }

        if let Err(e) = h.flush() {
            eprintln!("[auto-learn/L2] flush error: {}", e);
        } else {
            eprintln!("[auto-learn/L2] state persisted for '{}'", app_id);
        }
    }

    added
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
