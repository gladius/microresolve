//! Phrase generation — prompt building and response parsing.
//!
//! All LLM prompt logic lives here. The WASM layer exposes these functions
//! so the UI only handles HTTP transport.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Language configuration loaded from languages/languages.json.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LangConfig {
    pub name: String,
    pub hint: Option<String>,
}

/// Get all language configs, lazily loaded and cached.
fn lang_configs() -> &'static HashMap<String, LangConfig> {
    static DATA: OnceLock<HashMap<String, LangConfig>> = OnceLock::new();
    DATA.get_or_init(|| {
        let json = include_str!("../languages/languages.json");
        serde_json::from_str(json).expect("invalid languages.json")
    })
}

/// Get supported languages as JSON: {"en": "English", "es": "Spanish", ...}
pub fn supported_languages_json() -> String {
    let configs = lang_configs();
    let names: HashMap<&str, &str> = configs
        .iter()
        .map(|(code, cfg)| (code.as_str(), cfg.name.as_str()))
        .collect();
    serde_json::to_string(&names).unwrap_or_default()
}

/// Phrase quality rules — shared DO NOTs for all phrase generation.
pub const PHRASE_QUALITY_RULES: &str = r#"DO NOT:
- Use the customer's exact message as a seed
- Repeat the same structure with word swaps ("cancel my order" / "cancel my purchase" / "cancel my item")
- Use overly polished corporate language
- Include order numbers, names, dates, or specific products
- Generate translations of the same phrases across languages — each language should have culturally natural expressions"#;

/// Review fix prompt — gap-filling, not broad generation.
/// Callers must fill in: query, intent, existing_seeds.
pub const REVIEW_FIX_GUIDELINES: &str = r#"You are fixing a failed intent match in a keyword-matching router.

The router failed because the customer used words/phrases that don't overlap with the existing seeds.
Your job: identify what vocabulary is MISSING and suggest 1-2 short seed phrases to fill the gap.

Rules:
- Look at the existing seeds and the customer query
- Find words/phrases in the query that have NO overlap with existing seeds
- Generate 1-2 seed phrases that cover ONLY this gap
- Each seed should introduce NEW vocabulary, not repeat what's already covered
- Keep seeds short (3-8 words)
- Do NOT generate generic paraphrases of existing seeds"#;

const BASE_GUIDELINES: &str = r#"Generate realistic seed phrases for an intent routing system. These phrases train a keyword-matching router (not an LLM), so vocabulary diversity is critical.

Intent ID: {intent_id}
Description: {description}

Generate exactly 10 phrases per language. Each phrase must be something a real human would actually type in a chat box or support ticket. Requirements:

VARIETY IN LENGTH:
- 2-3 short phrases (2-4 words): "cancel order", "refund status"
- 4-5 medium phrases (5-10 words): "I need to cancel the order I placed"
- 2-3 long/conversational phrases (10+ words): "hey I ordered something yesterday and I changed my mind, can you cancel it"

VARIETY IN STYLE:
- Formal: "I would like to request a cancellation"
- Casual: "yo can I cancel this thing"
- Frustrated: "why is it so hard to cancel an order around here"
- Question form: "how do I cancel my recent order"
- Command form: "cancel order 12345"
- Contextual/story: "I found a better price elsewhere so I need to cancel"

VOCABULARY DIVERSITY (most important):
- Use different verbs for the same action (cancel/terminate/revoke/withdraw/undo)
- Use different nouns (order/purchase/transaction/item)
- Include phrases that describe the SITUATION not just the action ("I changed my mind", "ordered by mistake")
- Include emotional/frustrated variants that real users type

DO NOT:
- Repeat the same structure with word swaps ("cancel my order" / "cancel my purchase" / "cancel my item")
- Use overly polished corporate language
- Generate translations of the same phrases across languages — each language should have culturally natural expressions"#;

/// Build an LLM prompt for seed generation.
///
/// Returns the full prompt string to send to the LLM.
pub fn build_prompt(intent_id: &str, description: &str, languages: &[String]) -> String {
    let configs = lang_configs();

    let guidelines = BASE_GUIDELINES
        .replace("{intent_id}", if intent_id.is_empty() { "(unnamed)" } else { intent_id })
        .replace("{description}", description);

    if languages.len() == 1 {
        let lang = &languages[0];
        let lang_name = configs
            .get(lang.as_str())
            .map(|c| c.name.as_str())
            .unwrap_or(lang.as_str());
        let hint = configs
            .get(lang.as_str())
            .and_then(|c| c.hint.as_deref())
            .map(|h| format!("\n\n{}", h))
            .unwrap_or_default();

        format!(
            "{}{}\n\nLanguage: {}\n\nReturn ONLY a JSON array of strings. No markdown, no explanation.",
            guidelines, hint, lang_name
        )
    } else {
        let lang_names: Vec<&str> = languages
            .iter()
            .map(|l| {
                configs
                    .get(l.as_str())
                    .map(|c| c.name.as_str())
                    .unwrap_or(l.as_str())
            })
            .collect();
        let lang_list = lang_names.join(", ");

        let hints: Vec<&str> = languages
            .iter()
            .filter_map(|l| configs.get(l.as_str()))
            .filter_map(|c| c.hint.as_deref())
            .collect();

        let hints_block = if hints.is_empty() {
            String::new()
        } else {
            let items: Vec<String> = hints.iter().map(|h| format!("- {}", h)).collect();
            format!("\n\nLanguage-specific instructions:\n{}", items.join("\n"))
        };

        format!(
            "{}\n\nLanguages: {}\nFor non-English languages: write how native speakers actually type in chat, not translations of English phrases. Include slang, colloquialisms, and culturally natural expressions.{}\n\nReturn ONLY a JSON object mapping language codes to arrays. No markdown, no explanation. Example:\n{{\"en\": [\"phrase one\", \"long conversational phrase here\"], \"es\": [\"frase natural\", \"frase larga y conversacional aquí\"]}}",
            guidelines, lang_list, hints_block
        )
    }
}

/// Parse the LLM response into phrases grouped by language.
///
/// Returns JSON: {"phrases_by_lang": {"en": [...], "es": [...]}, "total": N}
pub fn parse_response(response_text: &str, languages: &[String]) -> Result<String, String> {
    let phrases_by_lang: HashMap<String, Vec<String>>;

    if languages.len() == 1 {
        // Expect a JSON array
        let array_str = extract_json_array(response_text)
            .ok_or_else(|| "Could not parse response as JSON array".to_string())?;
        let parsed: Vec<String> =
            serde_json::from_str(&array_str).map_err(|e| format!("JSON parse error: {}", e))?;
        let mut map = HashMap::new();
        map.insert(languages[0].clone(), parsed);
        phrases_by_lang = map;
    } else {
        // Expect a JSON object
        let obj_str = extract_json_object(response_text)
            .ok_or_else(|| "Could not parse response as JSON object".to_string())?;
        let parsed: HashMap<String, Vec<String>> =
            serde_json::from_str(&obj_str).map_err(|e| format!("JSON parse error: {}", e))?;
        phrases_by_lang = parsed;
    }

    let total: usize = phrases_by_lang.values().map(|v| v.len()).sum();

    let result = serde_json::json!({
        "phrases_by_lang": phrases_by_lang,
        "total": total,
    });
    serde_json::to_string(&result).map_err(|e| format!("Serialization error: {}", e))
}

/// Extract the first JSON array `[...]` from text.
fn extract_json_array(text: &str) -> Option<String> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end > start {
        Some(text[start..=end].to_string())
    } else {
        None
    }
}

/// Extract the first JSON object `{...}` from text.
fn extract_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end > start {
        Some(text[start..=end].to_string())
    } else {
        None
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Situation pattern generation
// ──────────────────────────────────────────────────────────────────────────────

const SITUATION_GUIDELINES: &str = r#"You are configuring a keyword router that detects intents from state-description text.

Intent ID: {intent_id}
Description: {description}

Existing action seeds (DO NOT repeat these — they cover "user requesting the action"):
{seeds}

Your task: generate 6-8 SITUATION SIGNAL PATTERNS — short keywords or phrases that appear in text when someone is describing a state that implies this intent, NOT when they are requesting the action directly.

WHAT THESE ARE:
- Domain-specific error codes, status words, technical terms, and state nouns
- Things a user describes HAPPENING, not things they are ASKING FOR
- Example for "create_issue": "build failed", "OOM", "502", "crash", "prod down" — NOT "report a bug" (that's a seed)
- Example for "charge_card": "declined", "402", "payment failed", "card rejected" — NOT "charge my card"
- Example for "merge_pr": "LGTM", "approved", "two approvals", "review passed" — NOT "merge this PR"

WEIGHT GUIDE:
- 1.0: Highly domain-specific — virtually only appears in context of this intent
- 0.7: Moderately specific — usually implies this intent, occasionally ambiguous
- 0.4: Generic signal — common state word, needs a partner pattern to be confident

{lang_hint}

Return ONLY a JSON array. No markdown, no explanation. Example format:
[{"pattern": "build failed", "weight": 1.0}, {"pattern": "OOM", "weight": 1.0}, {"pattern": "prod", "weight": 0.7}, {"pattern": "error", "weight": 0.4}]"#;

/// Build an LLM prompt for situation pattern generation.
///
/// `seeds` are the existing action seeds — the LLM uses them to understand
/// what NOT to generate (seeds cover action vocabulary; situations cover state vocabulary).
/// `languages` drives whether CJK pattern hints are included.
pub fn build_situation_prompt(
    intent_id: &str,
    description: &str,
    seeds: &[String],
    languages: &[String],
) -> String {
    let configs = lang_configs();

    let seeds_block = if seeds.is_empty() {
        "(none yet)".to_string()
    } else {
        seeds.iter().take(12).map(|s| format!("  - {}", s)).collect::<Vec<_>>().join("\n")
    };

    let lang_hint = if languages.iter().any(|l| matches!(l.as_str(), "zh" | "ja" | "ko")) {
        let cjk: Vec<&str> = languages.iter()
            .filter(|l| matches!(l.as_str(), "zh" | "ja" | "ko"))
            .map(|l| configs.get(l.as_str()).map(|c| c.name.as_str()).unwrap_or(l.as_str()))
            .collect();
        format!(
            "CJK PATTERNS: Include 2-3 patterns in {} script naturally alongside Latin ones. \
             Use the actual ideographs/kana, e.g. \"付款\" not \"payment\" when a Chinese-specific \
             compound is more domain-specific.",
            cjk.join("/")
        )
    } else {
        String::new()
    };

    SITUATION_GUIDELINES
        .replace("{intent_id}", if intent_id.is_empty() { "(unnamed)" } else { intent_id })
        .replace("{description}", description)
        .replace("{seeds}", &seeds_block)
        .replace("{lang_hint}", &lang_hint)
}

/// Parsed situation pattern entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SituationPattern {
    pub pattern: String,
    pub weight: f32,
}

/// Parse an LLM response for situation patterns.
///
/// Returns a Vec of (pattern, weight) pairs, clamped to [0.1, 1.0].
pub fn parse_situation_response(response_text: &str) -> Result<Vec<SituationPattern>, String> {
    let array_str = extract_json_array(response_text)
        .ok_or_else(|| "Could not find JSON array in response".to_string())?;

    let raw: Vec<serde_json::Value> =
        serde_json::from_str(&array_str).map_err(|e| format!("JSON parse error: {}", e))?;

    let mut patterns = Vec::new();
    for v in raw {
        let pattern = v["pattern"]
            .as_str()
            .ok_or_else(|| format!("missing 'pattern' field in: {}", v))?
            .trim()
            .to_string();
        let weight = v["weight"]
            .as_f64()
            .ok_or_else(|| format!("missing 'weight' field in: {}", v))?
            as f32;
        if !pattern.is_empty() {
            patterns.push(SituationPattern {
                pattern,
                weight: weight.clamp(0.1, 1.0),
            });
        }
    }

    Ok(patterns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_languages_includes_expected() {
        let json = supported_languages_json();
        let map: HashMap<String, String> = serde_json::from_str(&json).unwrap();
        assert_eq!(map.get("en").unwrap(), "English");
        assert_eq!(map.get("zh").unwrap(), "Chinese");
        assert_eq!(map.get("ta").unwrap(), "Tamil");
        assert!(map.len() >= 12);
    }

    #[test]
    fn build_prompt_single_lang() {
        let prompt = build_prompt("cancel", "cancel order", &["en".to_string()]);
        assert!(prompt.contains("Intent ID: cancel"));
        assert!(prompt.contains("Language: English"));
        assert!(prompt.contains("JSON array"));
    }

    #[test]
    fn build_prompt_multi_lang_includes_hints() {
        let prompt = build_prompt(
            "cancel",
            "cancel order",
            &["en".to_string(), "zh".to_string(), "ta".to_string()],
        );
        assert!(prompt.contains("Languages: English, Chinese, Tamil"));
        assert!(prompt.contains("simplified Chinese"));
        assert!(prompt.contains("traditional Chinese"));
        assert!(prompt.contains("pure Tamil script"));
        assert!(prompt.contains("JSON object"));
    }

    #[test]
    fn parse_response_single_lang() {
        let response = r#"["cancel my order", "stop the order"]"#;
        let result = parse_response(response, &["en".to_string()]).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["total"], 2);
        assert_eq!(parsed["phrases_by_lang"]["en"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn parse_response_multi_lang() {
        let response = r#"{"en": ["cancel"], "es": ["cancelar", "anular"]}"#;
        let result =
            parse_response(response, &["en".to_string(), "es".to_string()]).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["total"], 3);
    }

    #[test]
    fn parse_response_with_surrounding_text() {
        let response = "Here are the seeds:\n[\"phrase one\", \"phrase two\"]\nDone.";
        let result = parse_response(response, &["en".to_string()]).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["total"], 2);
    }

    #[test]
    fn parse_response_bad_input() {
        let result = parse_response("no json here", &["en".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn build_situation_prompt_includes_intent_and_seeds() {
        let seeds = vec!["report a bug".to_string(), "log an issue".to_string()];
        let prompt = build_situation_prompt("create_issue", "Report bugs and incidents", &seeds, &["en".to_string()]);
        assert!(prompt.contains("create_issue"));
        assert!(prompt.contains("Report bugs and incidents"));
        assert!(prompt.contains("report a bug"));
        assert!(prompt.contains("JSON array"));
    }

    #[test]
    fn build_situation_prompt_includes_cjk_hint() {
        let prompt = build_situation_prompt("charge_card", "Charge payment", &[], &["zh".to_string()]);
        assert!(prompt.contains("CJK PATTERNS"));
        assert!(prompt.contains("Chinese"));
    }

    #[test]
    fn parse_situation_response_valid() {
        let response = r#"[{"pattern": "OOM", "weight": 1.0}, {"pattern": "prod", "weight": 0.7}]"#;
        let patterns = parse_situation_response(response).unwrap();
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].pattern, "OOM");
        assert!((patterns[0].weight - 1.0).abs() < 0.01);
        assert_eq!(patterns[1].pattern, "prod");
        assert!((patterns[1].weight - 0.7).abs() < 0.01);
    }

    #[test]
    fn parse_situation_response_clamps_weight() {
        let response = r#"[{"pattern": "oops", "weight": 2.5}]"#;
        let patterns = parse_situation_response(response).unwrap();
        assert!((patterns[0].weight - 1.0).abs() < 0.01);
    }

    #[test]
    fn parse_situation_response_strips_surrounding_text() {
        let response = "Sure! Here:\n[{\"pattern\": \"502\", \"weight\": 1.0}]\nDone.";
        let patterns = parse_situation_response(response).unwrap();
        assert_eq!(patterns[0].pattern, "502");
    }

    #[test]
    fn parse_situation_response_bad_input() {
        assert!(parse_situation_response("no json here").is_err());
    }
}
