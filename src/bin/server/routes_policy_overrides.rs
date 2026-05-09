//! Policy override CRUD — narrow declarative escape hatch per namespace.
//!
//! A policy override is a hard rule: when ALL listed words appear in the
//! normalised query, add `bonus` to the target intent. Designed for
//! pack-author encoded policy assertions that are too important or too rare
//! to wait for the auto-learn loop to teach them — explicit Article 5
//! carve-outs, CSAM detection vs generation, similar externally-specified
//! distinctions. Usage guideline: ≤10 per pack. Most compositional logic
//! belongs in per-intent structured expansion (LLM-distilled at pack-build),
//! not here.
//!
//! Internally still backed by `ConjunctionRule` — the runtime mechanism is a
//! token-conjunction. The renamed surface signals the intended *role*: hard
//! policy override, not a Swiss Army knife.
//!
//! Every mutation lands in the audit log so operators can see who added /
//! removed which rule when.

use crate::state::*;
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::{delete, get},
    Extension, Json,
};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/policy-overrides", get(list).post(add))
        .route(
            "/api/policy-overrides/{idx}",
            delete(remove).patch(update),
        )
}

#[derive(serde::Deserialize)]
pub struct PolicyOverridePayload {
    pub words: Vec<String>,
    pub intent: String,
    pub bonus: f32,
}

fn rule_to_json(idx: usize, r: &microresolve::scoring::ConjunctionRule) -> serde_json::Value {
    serde_json::json!({
        "idx": idx,
        "words": r.words,
        "intent": r.intent,
        "bonus": r.bonus,
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
    let rules = h.list_conjunctions();
    let arr: Vec<serde_json::Value> = rules
        .iter()
        .enumerate()
        .map(|(i, r)| rule_to_json(i, r))
        .collect();
    Ok(Json(serde_json::json!({ "policy_overrides": arr })))
}

pub async fn add(
    State(state): State<AppState>,
    headers: HeaderMap,
    Extension(KeyName(kid)): Extension<KeyName>,
    Json(req): Json<PolicyOverridePayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let h = state.engine.try_namespace(&app_id).ok_or((
        StatusCode::NOT_FOUND,
        format!("namespace '{}' not found", app_id),
    ))?;

    let words_for_audit = req.words.clone();
    let intent_for_audit = req.intent.clone();
    let bonus_for_audit = req.bonus;

    let idx = h
        .add_conjunction(req.words, req.intent, req.bonus)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    audit_mutation(
        &state,
        &kid,
        &app_id,
        "policy_override.add",
        serde_json::json!({
            "idx": idx,
            "words": words_for_audit,
            "intent": intent_for_audit,
            "bonus": bonus_for_audit,
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
        .remove_conjunction(idx)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    audit_mutation(
        &state,
        &kid,
        &app_id,
        "policy_override.remove",
        serde_json::json!({
            "idx": idx,
            "words": removed.words,
            "intent": removed.intent,
            "bonus": removed.bonus,
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
    Json(req): Json<PolicyOverridePayload>,
) -> Result<StatusCode, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let h = state.engine.try_namespace(&app_id).ok_or((
        StatusCode::NOT_FOUND,
        format!("namespace '{}' not found", app_id),
    ))?;
    let words_for_audit = req.words.clone();
    let intent_for_audit = req.intent.clone();
    let bonus_for_audit = req.bonus;
    h.update_conjunction(idx, req.words, req.intent, req.bonus)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    audit_mutation(
        &state,
        &kid,
        &app_id,
        "policy_override.update",
        serde_json::json!({
            "idx": idx,
            "words": words_for_audit,
            "intent": intent_for_audit,
            "bonus": bonus_for_audit,
        }),
    );
    let _ = h.flush();
    maybe_commit(&state, &app_id);
    Ok(StatusCode::NO_CONTENT)
}
