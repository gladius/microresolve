//! Semantic layer endpoints.
//!
//! POST /api/semantic/build   — train a semantic model for a namespace (model_type: mini|nano|hierarchical)
//! GET  /api/semantic/status  — check which models are built for a namespace
//! POST /api/semantic/score   — score a query against a trained model
//! POST /api/semantic/compare — score a query against all three models side-by-side
//! POST /api/semantic/pairs   — generate LLM pairs without training (diagnostic)

use axum::{
    extract::State,
    http::{StatusCode, HeaderMap},
    routing::{get, post},
    Json,
};
use std::collections::HashMap;
use asv_router::semantic::{MiniEncoder, MiniEncoderConfig, NanoEncoder, NanoEncoderConfig, HierarchicalEncoder};
use crate::state::*;
use crate::llm::call_llm;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/semantic/build",   post(semantic_build))
        .route("/api/semantic/status",  get(semantic_status))
        .route("/api/semantic/score",   post(semantic_score))
        .route("/api/semantic/compare", post(semantic_compare))
        .route("/api/semantic/pairs",   post(semantic_pairs_preview))
}

// ── Build ─────────────────────────────────────────────────────────────────────

fn default_model_type() -> String { "mini".to_string() }
fn default_n_paraphrases() -> usize { 10 }
fn default_epochs() -> usize { 100 }
fn default_n_pairs() -> usize { 50 }
fn default_pair_epochs() -> usize { 20 }

#[derive(serde::Deserialize)]
pub struct BuildRequest {
    /// Which encoder architecture to train. One of: "mini", "nano", "hierarchical".
    /// Default: "mini".
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// If true, call LLM to generate exclusive vocabulary terms per intent before training.
    #[serde(default)]
    pub augment_with_llm: bool,
    /// How many LLM vocabulary terms to generate per intent. Default: 10.
    #[serde(default = "default_n_paraphrases")]
    pub n_paraphrases: usize,
    /// Training epochs for triplet loss. Default: 100.
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    /// If true, call LLM to generate semantic similarity pairs for a refinement pass.
    #[serde(default)]
    pub refine_with_pairs: bool,
    /// How many semantic pairs to request from the LLM. Default: 50.
    #[serde(default = "default_n_pairs")]
    pub n_pairs: usize,
    /// Epochs for the pair-refinement pass. Default: 20.
    #[serde(default = "default_pair_epochs")]
    pub pair_epochs: usize,
}

pub async fn semantic_build(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<BuildRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    let (intent_phrases, intent_descriptions) = collect_phrases(&state, &app_id)?;

    if intent_phrases.len() < 2 {
        return Err((StatusCode::UNPROCESSABLE_ENTITY,
            "need at least 2 intents with phrases to build a semantic model".to_string()));
    }

    let augmented = if req.augment_with_llm {
        augment_with_llm(&state, &intent_phrases, &intent_descriptions, req.n_paraphrases).await?
    } else {
        intent_phrases.clone()
    };

    let total_phrases: usize = augmented.values().map(|v| v.len()).sum();
    let n_intents = augmented.len();

    let pairs = if req.refine_with_pairs {
        generate_semantic_pairs(&state, &intent_phrases, &intent_descriptions, req.n_pairs).await?
    } else {
        Vec::new()
    };
    let n_pairs = pairs.len();

    match req.model_type.as_str() {
        "nano" => {
            let cfg = NanoEncoderConfig {
                epochs: req.epochs,
                pair_epochs: req.pair_epochs,
                ..NanoEncoderConfig::default()
            };
            let mut model = NanoEncoder::train(&augmented, &cfg)
                .ok_or_else(|| (StatusCode::INTERNAL_SERVER_ERROR, "nano training failed".to_string()))?;
            if !pairs.is_empty() {
                model.refine_with_pairs(&pairs, &augmented, &cfg);
            }
            state.semantic_nano.write().unwrap().insert(app_id.clone(), model);
        }
        "hierarchical" => {
            let cfg = MiniEncoderConfig {
                epochs: req.epochs,
                pair_epochs: req.pair_epochs,
                ..MiniEncoderConfig::default()
            };
            let mut model = HierarchicalEncoder::train(&augmented, &cfg)
                .ok_or_else(|| (StatusCode::INTERNAL_SERVER_ERROR, "hierarchical training failed".to_string()))?;
            if !pairs.is_empty() {
                model.refine_with_pairs(&intent_phrases, &pairs, &cfg);
            }
            state.semantic_hier.write().unwrap().insert(app_id.clone(), model);
        }
        _ => {
            // "mini" (default)
            let cfg = MiniEncoderConfig {
                epochs: req.epochs,
                pair_epochs: req.pair_epochs,
                ..MiniEncoderConfig::default()
            };
            let mut model = MiniEncoder::train(&augmented, &cfg)
                .ok_or_else(|| (StatusCode::INTERNAL_SERVER_ERROR, "mini training failed".to_string()))?;
            if !pairs.is_empty() {
                model.refine_with_pairs(&pairs, &augmented, &cfg);
            }
            state.semantic.write().unwrap().insert(app_id.clone(), model);
        }
    }

    Ok(Json(serde_json::json!({
        "app_id": app_id,
        "model_type": req.model_type,
        "intents": n_intents,
        "total_phrases": total_phrases,
        "augmented": req.augment_with_llm,
        "epochs": req.epochs,
        "pair_refined": req.refine_with_pairs,
        "n_pairs": n_pairs,
        "status": "ok"
    })))
}

fn collect_phrases(
    state: &AppState,
    app_id: &str,
) -> Result<(HashMap<String, Vec<String>>, HashMap<String, String>), (StatusCode, String)> {
    let routers = state.routers.read().unwrap();
    let router = routers.get(app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
    let mut phrases = HashMap::new();
    let mut descs = HashMap::new();
    for id in router.intent_ids() {
        let p = router.get_training(&id).unwrap_or_default();
        if !p.is_empty() {
            descs.insert(id.clone(), router.get_description(&id).to_string());
            phrases.insert(id, p);
        }
    }
    Ok((phrases, descs))
}

/// LLM distillation via semantic vocabulary expansion.
///
/// Instead of paraphrase sentences, we ask the LLM for the *vocabulary* —
/// individual words and short phrases a user would say for this intent.
/// This directly trains each term's n-grams toward the intent centroid,
/// unlike full sentences where signal is diluted across many words.
///
/// Example output for "stripe:cancel_subscription":
///   ["quit", "opt out", "deactivate", "stop billing", "end membership",
///    "turn off renewal", "close account", "unsubscribe", "terminate", ...]
async fn augment_with_llm(
    state: &ServerState,
    intent_phrases: &HashMap<String, Vec<String>>,
    intent_descriptions: &HashMap<String, String>,
    n: usize,
) -> Result<HashMap<String, Vec<String>>, (StatusCode, String)> {
    let mut augmented = intent_phrases.clone();

    // Build a full summary of ALL intents (id + description + examples)
    // so the LLM has precise boundaries between closely related intents
    let all_intents_block: String = intent_phrases.iter().map(|(id, phrases)| {
        let desc = intent_descriptions.get(id).map(|s| s.as_str()).unwrap_or("");
        let ex = phrases.iter().take(2).cloned().collect::<Vec<_>>().join("\", \"");
        if desc.is_empty() {
            format!("  - {id}: e.g. \"{ex}\"")
        } else {
            format!("  - {id} ({desc}): e.g. \"{ex}\"")
        }
    }).collect::<Vec<_>>().join("\n");

    for (intent_id, seed_phrases) in intent_phrases {
        let desc = intent_descriptions.get(intent_id).map(|s| s.as_str()).unwrap_or("");
        let seeds = seed_phrases.iter().take(5).cloned().collect::<Vec<_>>().join("\", \"");

        let desc_line = if desc.is_empty() {
            String::new()
        } else {
            format!("Description: {desc}\n")
        };

        let prompt = format!(
            "You are building an EXCLUSIVE semantic vocabulary for an intent classifier.\n\n\
             ALL intents in this system (read carefully to understand boundaries):\n\
             {all_intents_block}\n\n\
             YOUR TARGET INTENT: {intent_id}\n\
             {desc_line}\
             Example phrases: [\"{seeds}\"]\n\n\
             Generate {n} individual words and SHORT phrases (1-4 words each) that:\n\
             1. Unambiguously indicate THIS intent specifically\n\
             2. Would NOT be confused with any of the other intents above\n\n\
             Rules:\n\
             - Understand the precise meaning of this intent vs its siblings\n\
             - Include: synonyms, domain jargon, oblique user expressions, action words, nouns\n\
             - Exclude: terms shared with other intents, full sentences\n\
             - Think: what word would make a human immediately think of THIS specific action?\n\n\
             Output ONLY a JSON array of strings, no other text.\n\
             Example: [\"quit\", \"opt out\", \"stop billing\", \"deactivate\", \"terminate\"]"
        );

        match call_llm(state, &prompt, 1024).await {
            Ok(response) => {
                let json_str = extract_json_array(&response);
                if let Ok(terms) = serde_json::from_str::<Vec<String>>(json_str) {
                    augmented.entry(intent_id.clone())
                        .or_default()
                        .extend(terms);
                }
            }
            Err(_) => {}
        }
    }

    Ok(augmented)
}

/// Ask the LLM to generate semantic similarity pairs over the domain vocabulary.
///
/// Each pair `(t1, t2, sim)` becomes an MSE cosine loss target during the refinement pass.
/// The LLM is instructed to focus on vocabulary-gap pairs: common user words that never
/// appear in seed phrases but should route to a specific intent (e.g. "debited" → refund).
async fn generate_semantic_pairs(
    state: &ServerState,
    intent_phrases: &HashMap<String, Vec<String>>,
    intent_descriptions: &HashMap<String, String>,
    n: usize,
) -> Result<Vec<(String, String, f32)>, (StatusCode, String)> {
    let all_intents_block: String = intent_phrases.iter().map(|(id, phrases)| {
        let desc = intent_descriptions.get(id).map(|s| s.as_str()).unwrap_or("");
        let ex = phrases.iter().take(3).cloned().collect::<Vec<_>>().join("\", \"");
        if desc.is_empty() {
            format!("  - {id}: e.g. \"{ex}\"")
        } else {
            format!("  - {id} ({desc}): e.g. \"{ex}\"")
        }
    }).collect::<Vec<_>>().join("\n");

    let prompt = format!(
        "You are generating semantic similarity training data for a lightweight intent-routing embedding model.\n\n\
         The model needs to learn which EVERYDAY words and phrases belong to which intent,\n\
         even when those words never appear in the seed phrases.\n\n\
         Intents:\n{all_intents_block}\n\n\
         Generate {n} semantic similarity pairs from the DOMAIN VOCABULARY.\n\n\
         Output format: [{{\"t1\": \"...\", \"t2\": \"...\", \"sim\": 0.0}}]\n\n\
         Similarity score meaning:\n\
         - 0.8–1.0  near-synonyms that BOTH unambiguously indicate the SAME intent\n\
         - 0.0–0.2  terms from CLEARLY DIFFERENT intents — especially confusable pairs\n\
         - 0.3–0.7  topically related but belonging to distinct intents\n\n\
         CRITICAL — prioritize vocabulary-gap pairs:\n\
         These are words a real user would say that do NOT appear in the seed phrases above.\n\
         Example: if 'debited' should map to a refund intent → (\"debited\", \"refund\", 0.88)\n\
         Example: if 'stolen card' should map to dispute → (\"stolen card\", \"chargeback\", 0.90)\n\
         Example: if 'deactivate' should map to cancel → (\"deactivate\", \"cancel\", 0.85)\n\n\
         Use 60% high-similarity pairs (pulling synonyms together) and 40% low-similarity pairs\n\
         (pushing cross-intent terms apart). Use 1-4 word terms only. No full sentences.\n\n\
         Output ONLY the JSON array, no other text."
    );

    #[derive(serde::Deserialize)]
    struct RawPair { t1: String, t2: String, sim: f32 }

    match call_llm(state, &prompt, 2048).await {
        Ok(response) => {
            let json_str = extract_json_array(&response);
            match serde_json::from_str::<Vec<RawPair>>(json_str) {
                Ok(raw) => {
                    let pairs: Vec<(String, String, f32)> = raw.into_iter()
                        .filter(|p| p.sim >= 0.0 && p.sim <= 1.0
                            && !p.t1.trim().is_empty() && !p.t2.trim().is_empty())
                        .map(|p| (p.t1, p.t2, p.sim))
                        .collect();
                    Ok(pairs)
                }
                Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to parse LLM pair output: {e}"))),
            }
        }
        Err(e) => Err(e),
    }
}

/// Extract a JSON array from LLM output that may have surrounding text or fences.
fn extract_json_array(text: &str) -> &str {
    let trimmed = text.trim();
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            return &trimmed[start..=end];
        }
    }
    trimmed
}

// ── Pairs preview (diagnostic) ───────────────────────────────────────────────

/// `POST /api/semantic/pairs` — generate and return LLM pairs without training.
/// Useful for diagnosing what vocabulary-gap signal the LLM produces for a namespace.
#[derive(serde::Deserialize)]
pub struct PairsPreviewRequest {
    #[serde(default = "default_n_pairs")]
    pub n_pairs: usize,
}

pub async fn semantic_pairs_preview(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<PairsPreviewRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let (intent_phrases, intent_descriptions) = collect_phrases(&state, &app_id)?;
    let pairs = generate_semantic_pairs(&state, &intent_phrases, &intent_descriptions, req.n_pairs).await?;
    let out: Vec<serde_json::Value> = pairs.iter()
        .map(|(t1, t2, sim)| serde_json::json!({"t1": t1, "t2": t2, "sim": sim}))
        .collect();
    Ok(Json(serde_json::json!({
        "app_id": app_id,
        "n": out.len(),
        "pairs": out,
    })))
}

// ── Status ────────────────────────────────────────────────────────────────────

pub async fn semantic_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let mini  = state.semantic.read().unwrap().contains_key(&app_id);
    let nano  = state.semantic_nano.read().unwrap().contains_key(&app_id);
    let hier  = state.semantic_hier.read().unwrap().contains_key(&app_id);
    Json(serde_json::json!({
        "app_id": app_id,
        "built": mini || nano || hier,
        "models": { "mini": mini, "nano": nano, "hierarchical": hier },
    }))
}

// ── Score ─────────────────────────────────────────────────────────────────────

fn default_model_type_score() -> String { "mini".to_string() }
fn default_top_k() -> usize { 5 }

#[derive(serde::Deserialize)]
pub struct ScoreRequest {
    pub query: String,
    /// Which model to score against: "mini", "nano", or "hierarchical". Default: "mini".
    #[serde(default = "default_model_type_score")]
    pub model_type: String,
    /// Only return top-N results. Default: 5.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

pub async fn semantic_score(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ScoreRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    let scores = match req.model_type.as_str() {
        "nano" => {
            let sem = state.semantic_nano.read().unwrap();
            let model = sem.get(&app_id).ok_or_else(|| {
                (StatusCode::NOT_FOUND, format!("no nano model for '{}' — build it first", app_id))
            })?;
            model.score_query(&req.query)
        }
        "hierarchical" => {
            let sem = state.semantic_hier.read().unwrap();
            let model = sem.get(&app_id).ok_or_else(|| {
                (StatusCode::NOT_FOUND, format!("no hierarchical model for '{}' — build it first", app_id))
            })?;
            model.score_query(&req.query)
        }
        _ => {
            let sem = state.semantic.read().unwrap();
            let model = sem.get(&app_id).ok_or_else(|| {
                (StatusCode::NOT_FOUND, format!("no mini model for '{}' — build it first", app_id))
            })?;
            model.score_query(&req.query)
        }
    };

    let results: Vec<serde_json::Value> = scores.into_iter()
        .take(req.top_k)
        .map(|(id, score)| serde_json::json!({ "intent_id": id, "score": score }))
        .collect();

    Ok(Json(serde_json::json!({
        "query": req.query,
        "model_type": req.model_type,
        "results": results,
    })))
}

// ── Compare ───────────────────────────────────────────────────────────────────

/// `POST /api/semantic/compare` — score a query against all built models simultaneously.
/// Returns side-by-side top-K results for mini, nano, and hierarchical.
#[derive(serde::Deserialize)]
pub struct CompareRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

pub async fn semantic_compare(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CompareRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let top_k = req.top_k;

    let fmt_scores = |scores: Vec<(String, f32)>| -> Vec<serde_json::Value> {
        scores.into_iter().take(top_k)
            .map(|(id, s)| serde_json::json!({ "intent_id": id, "score": s }))
            .collect()
    };

    let mini_results = state.semantic.read().unwrap()
        .get(&app_id).map(|m| fmt_scores(m.score_query(&req.query)));
    let nano_results = state.semantic_nano.read().unwrap()
        .get(&app_id).map(|m| fmt_scores(m.score_query(&req.query)));
    let hier_results = state.semantic_hier.read().unwrap()
        .get(&app_id).map(|m| fmt_scores(m.score_query(&req.query)));

    Json(serde_json::json!({
        "query": req.query,
        "mini":         mini_results,
        "nano":         nano_results,
        "hierarchical": hier_results,
    }))
}
