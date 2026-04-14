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

/// Review fix prompt — intent-anchored phrase generation.
pub const REVIEW_FIX_GUIDELINES: &str = r#"You are expanding training coverage for an intent router.

A customer query failed to route correctly. Turn 1 has already identified which intents need more coverage.
Your job: generate new standalone training phrases for those intents.

Rules:
- Generate phrases based on the intent's description and what a user would say
- Phrases must be self-contained: meaningful without any prior conversation context
- Avoid pronouns and vague references ("them", "it", "that one")
- Introduce vocabulary diversity — different verbs, styles, and phrasings
- Keep phrases short to medium (2-10 words)
- Do NOT duplicate existing phrases already in the system"#;

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

}
