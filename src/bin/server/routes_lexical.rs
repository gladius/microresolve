//! Lexical group CRUD — per-namespace morph + abbrev normalization.
//!
//! Each group has a `kind` (`morph` or `abbrev`), a `lang` (language code),
//! a `canonical` form, and a list of `variants`. At tokenization time
//! (both index-time when seeds are added, and query-time at resolve),
//! every token gets normalized to its canonical form before scoring.
//!
//! Mutations rebuild the IntentIndex (the existing seeds need to be
//! re-tokenized through the new lexical group set). Each mutation lands
//! in the audit log.
//!
//! See `_internal/V0_3_LEXICAL_GROUPS_PLAN.md` for design + history.

use crate::pipeline::{call_llm, extract_json};
use crate::state::*;
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Extension, Json,
};
use microresolve::lexical::{LexicalGroup, LexicalKind};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/lexical-groups", get(list).post(add))
        .route(
            "/api/lexical-groups/{idx}",
            axum::routing::delete(remove).patch(update),
        )
        .route("/api/lexical-groups/suggest", post(suggest))
}

#[derive(serde::Deserialize)]
pub struct GroupPayload {
    pub kind: LexicalKind,
    #[serde(default = "default_lang")]
    pub lang: String,
    pub canonical: String,
    pub variants: Vec<String>,
}

fn default_lang() -> String {
    "en".to_string()
}

fn group_to_json(idx: usize, g: &LexicalGroup) -> serde_json::Value {
    serde_json::json!({
        "idx": idx,
        "kind": g.kind,
        "lang": g.lang,
        "canonical": g.canonical,
        "variants": g.variants,
    })
}

pub async fn list(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let h = state.engine.try_namespace(&app_id).ok_or((
        StatusCode::NOT_FOUND,
        format!("namespace '{}' not found", app_id),
    ))?;
    let groups = h.list_lexical_groups();
    let arr: Vec<serde_json::Value> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| group_to_json(i, g))
        .collect();
    Ok(Json(serde_json::json!({ "lexical_groups": arr })))
}

pub async fn add(
    State(state): State<AppState>,
    headers: HeaderMap,
    Extension(KeyName(kid)): Extension<KeyName>,
    Json(req): Json<GroupPayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let h = state.engine.try_namespace(&app_id).ok_or((
        StatusCode::NOT_FOUND,
        format!("namespace '{}' not found", app_id),
    ))?;

    let canonical_audit = req.canonical.clone();
    let variants_audit = req.variants.clone();
    let lang_audit = req.lang.clone();
    let kind_audit = req.kind;

    let group = LexicalGroup {
        kind: req.kind,
        lang: req.lang,
        canonical: req.canonical,
        variants: req.variants,
    };
    let idx = h
        .add_lexical_group(group)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    audit_mutation(
        &state,
        &kid,
        &app_id,
        "lexical_group.add",
        serde_json::json!({
            "idx": idx,
            "kind": kind_audit,
            "lang": lang_audit,
            "canonical": canonical_audit,
            "variants": variants_audit,
        }),
    );

    let _ = h.flush();
    maybe_commit(&state, &app_id);

    Ok(Json(serde_json::json!({ "idx": idx })))
}

pub async fn remove(
    State(state): State<AppState>,
    headers: HeaderMap,
    Extension(KeyName(kid)): Extension<KeyName>,
    Path(idx): Path<usize>,
) -> Result<StatusCode, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let h = state.engine.try_namespace(&app_id).ok_or((
        StatusCode::NOT_FOUND,
        format!("namespace '{}' not found", app_id),
    ))?;
    let removed = h
        .remove_lexical_group(idx)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    audit_mutation(
        &state,
        &kid,
        &app_id,
        "lexical_group.remove",
        serde_json::json!({
            "idx": idx,
            "kind": removed.kind,
            "lang": removed.lang,
            "canonical": removed.canonical,
            "variants": removed.variants,
        }),
    );
    let _ = h.flush();
    maybe_commit(&state, &app_id);
    Ok(StatusCode::NO_CONTENT)
}

pub async fn update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Extension(KeyName(kid)): Extension<KeyName>,
    Path(idx): Path<usize>,
    Json(req): Json<GroupPayload>,
) -> Result<StatusCode, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let h = state.engine.try_namespace(&app_id).ok_or((
        StatusCode::NOT_FOUND,
        format!("namespace '{}' not found", app_id),
    ))?;
    let canonical_audit = req.canonical.clone();
    let variants_audit = req.variants.clone();
    let lang_audit = req.lang.clone();
    let kind_audit = req.kind;
    let group = LexicalGroup {
        kind: req.kind,
        lang: req.lang,
        canonical: req.canonical,
        variants: req.variants,
    };
    h.update_lexical_group(idx, group)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    audit_mutation(
        &state,
        &kid,
        &app_id,
        "lexical_group.update",
        serde_json::json!({
            "idx": idx,
            "kind": kind_audit,
            "lang": lang_audit,
            "canonical": canonical_audit,
            "variants": variants_audit,
        }),
    );
    let _ = h.flush();
    maybe_commit(&state, &app_id);
    Ok(StatusCode::NO_CONTENT)
}

#[derive(serde::Deserialize)]
pub struct SuggestPayload {
    /// `morph` or `abbrev` — different prompts target each.
    pub kind: LexicalKind,
    #[serde(default = "default_lang")]
    pub lang: String,
}

/// Operator-triggered LLM proposal of lexical groups for the namespace's
/// current vocabulary. Returns proposals as JSON; operator approves
/// individually via separate POST /api/lexical-groups calls.
pub async fn suggest(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SuggestPayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let h = state.engine.try_namespace(&app_id).ok_or((
        StatusCode::NOT_FOUND,
        format!("namespace '{}' not found", app_id),
    ))?;

    // Gather current vocabulary (tokens present in the index) + intent
    // descriptions. The LLM uses both to ground proposals in real pack data.
    let (vocab, intent_descs): (Vec<String>, Vec<(String, String)>) = h.with_resolver(|r| {
        let mut tokens: Vec<&String> = r.index().word_intent.keys().collect();
        tokens.sort();
        let vocab: Vec<String> = tokens.iter().take(400).map(|s| s.to_string()).collect();
        let descs: Vec<(String, String)> = r
            .intent_ids()
            .into_iter()
            .filter_map(|id| {
                r.intent(&id)
                    .map(|info| (id.clone(), info.description.clone()))
            })
            .collect();
        (vocab, descs)
    });

    let intent_block: String = intent_descs
        .iter()
        .map(|(id, d)| {
            let short = d.chars().take(140).collect::<String>();
            format!("  - {}: {}", id, short)
        })
        .collect::<Vec<_>>()
        .join("\n");

    let vocab_str = vocab.join(", ");
    let lang = &req.lang;

    let prompt = match req.kind {
        LexicalKind::Morph => format!(
            "You are extending a per-namespace lexical dictionary for an intent classifier.\n\n\
             Namespace intents:\n{intent_block}\n\n\
             Tokens currently in the index (lowercase): {vocab_str}\n\n\
             Language: {lang}\n\n\
             For language {lang}, identify which of these tokens are inflectional\n\
             variants of the same root word (the lexeme). Group them so the engine\n\
             can normalize variants to a canonical form.\n\n\
             VALID examples (these ARE inflectional variants of one root):\n\
             - {{\"canonical\": \"child\", \"variants\": [\"child\", \"children\"]}}\n\
             - {{\"canonical\": \"predict\", \"variants\": [\"predict\", \"predicts\", \"predicting\"]}}\n\n\
             INVALID examples (these are different words, not inflections):\n\
             - {{\"canonical\": \"act\", \"variants\": [\"act\", \"action\", \"active\"]}}  ← \"active\" is NOT an inflection of \"act\"\n\
             - {{\"canonical\": \"police\", \"variants\": [\"police\", \"policy\"]}}  ← \"policy\" is NOT inflection of \"police\"\n\n\
             Be conservative. Skip a token if you're unsure.\n\n\
             Output: ONLY a JSON array of groups. No preamble, no commentary.\n\
             Format: [{{\"canonical\": \"...\", \"variants\": [\"...\", \"...\"]}}, ...]"
        ),
        LexicalKind::Abbrev => format!(
            "You are identifying domain abbreviations in a namespace's vocabulary.\n\n\
             Namespace intents:\n{intent_block}\n\n\
             Tokens / phrases currently in the index: {vocab_str}\n\n\
             Language: {lang}\n\n\
             Find tokens that are abbreviations or acronyms of longer phrases\n\
             relevant to this namespace. The canonical form is the FULL phrase\n\
             (lowercase); the abbreviation is the shortened variant.\n\n\
             Examples:\n\
             - {{\"canonical\": \"real-time biometric identification\", \"variants\": [\"rbi\"]}}\n\
             - {{\"canonical\": \"non-consensual intimate imagery\", \"variants\": [\"ncii\"]}}\n\
             - {{\"canonical\": \"child sexual abuse material\", \"variants\": [\"csam\"]}}\n\n\
             Skip ambiguous abbreviations (multiple plausible expansions).\n\n\
             Output: ONLY a JSON array of groups. Format same as above."
        ),
    };

    let response = call_llm(&state, &prompt, 1500).await?;
    let json_str = extract_json(&response);

    // Parse loose JSON — tolerate either bare array or {"groups": [...]}.
    let parsed: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("LLM JSON parse: {}", e)))?;
    let arr = parsed
        .as_array()
        .or_else(|| parsed.get("groups").and_then(|g| g.as_array()))
        .cloned()
        .unwrap_or_default();

    // Build proposals tagged with kind + lang. Don't auto-apply — operator
    // approves via POST /api/lexical-groups.
    let proposals: Vec<serde_json::Value> = arr
        .iter()
        .filter_map(|g| {
            let canonical = g.get("canonical").and_then(|c| c.as_str())?;
            let variants = g.get("variants").and_then(|v| v.as_array())?;
            let v: Vec<String> = variants
                .iter()
                .filter_map(|x| x.as_str().map(|s| s.to_lowercase()))
                .collect();
            if v.is_empty() {
                return None;
            }
            Some(serde_json::json!({
                "kind": req.kind,
                "lang": req.lang,
                "canonical": canonical.to_lowercase(),
                "variants": v,
            }))
        })
        .collect();

    Ok(Json(serde_json::json!({
        "proposals": proposals,
        "count": proposals.len(),
    })))
}
