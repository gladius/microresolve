//! Global stop words — stored in {data_dir}/_stopwords/{lang}.json
//! Language-agnostic: Japanese particles are the same regardless of namespace.

use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{delete, get, post},
    Json,
};
use std::collections::HashMap;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/stopwords", get(list_stopwords))
        .route("/api/stopwords/generate", post(generate_stopwords))
        .route("/api/stopwords/{lang}", get(get_stopwords_for_lang))
        .route("/api/stopwords/{lang}/add", post(add_stopword))
        .route("/api/stopwords/{lang}/{word}", delete(remove_stopword))
}

/// English stop words — pre-seeded, no LLM needed.
pub const EN_STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "need",
    "dare", "ought", "used", "it", "its", "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "his", "she", "her", "they", "their", "what", "which", "who",
    "whom", "how", "when", "where", "why", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "same", "so", "than", "too", "very", "just",
    "about", "up", "out", "get", "got", "make", "made", "go", "goes", "went", "come", "came",
    "see", "said", "know", "think", "also", "there", "then", "now", "like", "new", "one", "two",
    "any", "want", "use", "into", "over",
];

/// Load all available stop word sets from disk.
/// `en.json` on disk overrides the built-in EN_STOPWORDS — that's how
/// add/delete edits to the English list get persisted.
pub fn load_all_stopwords(data_dir: &str) -> HashMap<String, Vec<String>> {
    let mut map = HashMap::new();
    // Built-in English baseline (overridden below if en.json exists)
    map.insert(
        "en".to_string(),
        EN_STOPWORDS.iter().map(|s| s.to_string()).collect(),
    );
    let dir = format!("{}/_stopwords", data_dir);
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let lang = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            if lang.is_empty() {
                continue;
            }
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(words) = serde_json::from_str::<Vec<String>>(&content) {
                    map.insert(lang, words);
                }
            }
        }
    }
    map
}

/// Source label: "built-in" iff `en` is being served from EN_STOPWORDS
/// (no en.json on disk), "custom" if en.json overrides, "generated" for
/// other languages produced by the LLM.
fn source_for(lang: &str, data_dir: &str) -> &'static str {
    let path = format!("{}/_stopwords/{}.json", data_dir, lang);
    let on_disk = std::path::Path::new(&path).exists();
    match (lang, on_disk) {
        ("en", false) => "built-in",
        ("en", true) => "custom",
        (_, _) => "generated",
    }
}

fn write_stopwords(data_dir: &str, lang: &str, words: &[String]) -> Result<(), String> {
    let dir = format!("{}/_stopwords", data_dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = format!("{}/{}.json", dir, lang);
    let json = serde_json::to_string_pretty(words).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())
}

// ── GET /api/stopwords ────────────────────────────────────────────────────────

pub async fn list_stopwords(State(state): State<AppState>) -> Json<serde_json::Value> {
    let Some(ref data_dir) = state.data_dir else {
        return Json(serde_json::json!({}));
    };
    let map = load_all_stopwords(data_dir);
    let result: serde_json::Value = map
        .iter()
        .map(|(lang, words)| {
            (
                lang.clone(),
                serde_json::json!({
                    "count": words.len(),
                    "source": source_for(lang, data_dir),
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into();
    Json(result)
}

// ── GET /api/stopwords/{lang} ────────────────────────────────────────────────

pub async fn get_stopwords_for_lang(
    State(state): State<AppState>,
    Path(lang): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let data_dir = state
        .data_dir
        .clone()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "no data dir".into()))?;
    let map = load_all_stopwords(&data_dir);
    let words = map
        .get(&lang)
        .cloned()
        .ok_or((StatusCode::NOT_FOUND, format!("no stop words for {}", lang)))?;
    Ok(Json(serde_json::json!({
        "lang": lang,
        "words": words,
        "count": words.len(),
        "source": source_for(&lang, &data_dir),
    })))
}

// ── POST /api/stopwords/{lang}/add ──────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct AddRequest {
    word: String,
}

pub async fn add_stopword(
    State(state): State<AppState>,
    Path(lang): Path<String>,
    Json(req): Json<AddRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let data_dir = state
        .data_dir
        .clone()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "no data dir".into()))?;
    let word = req.word.trim().to_lowercase();
    if word.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "word must not be empty".into()));
    }
    let mut map = load_all_stopwords(&data_dir);
    let words = map.entry(lang.clone()).or_default();
    if !words.iter().any(|w| w == &word) {
        words.push(word);
        words.sort();
    }
    let words = words.clone();
    write_stopwords(&data_dir, &lang, &words)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(serde_json::json!({
        "lang": lang,
        "count": words.len(),
        "source": source_for(&lang, &data_dir),
    })))
}

// ── DELETE /api/stopwords/{lang}/{word} ─────────────────────────────────────

pub async fn remove_stopword(
    State(state): State<AppState>,
    Path((lang, word)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let data_dir = state
        .data_dir
        .clone()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "no data dir".into()))?;
    let mut map = load_all_stopwords(&data_dir);
    let words = map.entry(lang.clone()).or_default();
    let target = word.to_lowercase();
    let before = words.len();
    words.retain(|w| w != &target);
    if words.len() == before {
        return Err((StatusCode::NOT_FOUND, format!("'{}' not in list", word)));
    }
    let words = words.clone();
    write_stopwords(&data_dir, &lang, &words)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(serde_json::json!({
        "lang": lang,
        "count": words.len(),
        "source": source_for(&lang, &data_dir),
    })))
}

// ── POST /api/stopwords/generate ─────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct GenerateRequest {
    lang: String,
}

pub async fn generate_stopwords(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if req.lang == "en" {
        return Ok(Json(serde_json::json!({
            "lang": "en", "count": EN_STOPWORDS.len(), "source": "built-in"
        })));
    }

    let data_dir = state
        .data_dir
        .clone()
        .ok_or_else(|| (StatusCode::INTERNAL_SERVER_ERROR, "no data dir".to_string()))?;

    // Language name for prompt (best-effort)
    let lang_name = lang_display_name(&req.lang);

    let prompt = format!(
        "List the 60 most common {lang_name} stop words: function words, particles, articles, \
        prepositions, conjunctions, and auxiliary verbs that carry no semantic meaning for \
        intent classification. These will be excluded from phrase matching in a routing system.\n\
        Return ONLY a JSON array of lowercase strings. No explanations, no markdown.\n\
        Example format: [\"word1\", \"word2\", ...]",
        lang_name = lang_name,
    );

    let response = crate::pipeline::call_llm(&state, &prompt, 800)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("LLM failed: {}", e.1),
            )
        })?;

    let json_str = crate::pipeline::extract_json(&response);
    let words: Vec<String> = serde_json::from_str(json_str).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "parse failed: {}. Raw: {}",
                e,
                &response[..response.len().min(200)]
            ),
        )
    })?;

    // Save to _stopwords/{lang}.json
    let dir = format!("{}/_stopwords", data_dir);
    std::fs::create_dir_all(&dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let path = format!("{}/{}.json", dir, req.lang);
    std::fs::write(&path, serde_json::to_string_pretty(&words).unwrap())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    eprintln!(
        "[stopwords] generated {} words for {}",
        words.len(),
        req.lang
    );

    Ok(Json(serde_json::json!({
        "lang": req.lang,
        "count": words.len(),
        "source": "generated",
    })))
}

fn lang_display_name(code: &str) -> &str {
    match code {
        "ja" => "Japanese",
        "ko" => "Korean",
        "zh" => "Chinese",
        "es" => "Spanish",
        "fr" => "French",
        "de" => "German",
        "pt" => "Portuguese",
        "it" => "Italian",
        "nl" => "Dutch",
        "ar" => "Arabic",
        "hi" => "Hindi",
        "ru" => "Russian",
        "tr" => "Turkish",
        "pl" => "Polish",
        "sv" => "Swedish",
        "da" => "Danish",
        "fi" => "Finnish",
        "nb" => "Norwegian",
        "ta" => "Tamil",
        "te" => "Telugu",
        "bn" => "Bengali",
        "th" => "Thai",
        "vi" => "Vietnamese",
        other => other,
    }
}
