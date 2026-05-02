//! Namespace and domain management endpoints.
//!
//! Hierarchy: Namespace → Domain → Intent
//!   Namespace: isolated Resolver instance, selected via X-Namespace-ID header
//!   Domain:    logical intent group derived from "domain:intent_id" prefix
//!   Intent:    leaf routing target

use crate::data_git;
use crate::state::*;
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    routing::{delete, get, patch, post},
    Json,
};
use std::path::PathBuf;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/namespaces", get(list_namespaces))
        .route("/api/namespaces", post(create_namespace))
        .route("/api/namespaces", delete(delete_namespace))
        .route("/api/namespaces", patch(update_namespace))
        .route("/api/namespaces/train_negative", post(train_negative))
        .route("/api/namespaces/rebuild", post(rebuild_namespace))
        .route("/api/namespaces/{id}/history", get(namespace_history))
        .route("/api/namespaces/{id}/rollback", post(namespace_rollback))
        .route("/api/namespaces/{id}/diff", get(namespace_diff))
        .route("/api/domains", get(list_domain_groups))
        .route("/api/domains", post(create_domain))
        .route("/api/domains", patch(update_domain))
        .route("/api/domains", delete(delete_domain))
}

// ── Git history endpoints ───────────────────────────────────────────────────

#[derive(serde::Deserialize, Default)]
pub struct HistoryParams {
    #[serde(default)]
    limit: Option<usize>,
}

/// Last N commits affecting a namespace's data dir.
pub async fn namespace_history(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(params): Query<HistoryParams>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let dir = state.data_dir.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "server has no data_dir; history unavailable".into(),
    ))?;
    if !state.engine.has_namespace(&id) {
        return Err((
            StatusCode::NOT_FOUND,
            format!("namespace '{}' not found", id),
        ));
    }
    let limit = params.limit.unwrap_or(20).min(200);
    let commits = data_git::log(&PathBuf::from(dir), &id, limit);
    Ok(Json(serde_json::json!({
        "namespace_id": id,
        "commits": commits,
    })))
}

#[derive(serde::Deserialize)]
pub struct RollbackRequest {
    pub sha: String,
}

/// Hard-reset the data dir to `sha` and reload affected namespaces from disk.
///
/// Note: a rollback is repo-wide (git resets the whole data dir). The reload
/// re-imports every namespace currently loaded so they reflect the new state.
pub async fn namespace_rollback(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<RollbackRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let dir = state.data_dir.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "server has no data_dir; rollback unavailable".into(),
    ))?;
    if !state.engine.has_namespace(&id) {
        return Err((
            StatusCode::NOT_FOUND,
            format!("namespace '{}' not found", id),
        ));
    }
    let dir_path = PathBuf::from(dir);
    data_git::rollback(&dir_path, &req.sha).map_err(|e| (StatusCode::BAD_REQUEST, e))?;

    // Reload every loaded namespace from the rolled-back disk state.
    // Also pick up any namespace dirs that may have come back via the reset.
    let mut all_ids: std::collections::HashSet<String> =
        state.engine.namespaces().into_iter().collect();
    if let Ok(rd) = std::fs::read_dir(&dir_path) {
        for entry in rd.flatten() {
            let p = entry.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Skip dot-dirs (.git) and engine-reserved underscore-dirs.
            if !name.starts_with('_') && !name.starts_with('.') && p.is_dir() {
                all_ids.insert(name.to_string());
            }
        }
    }

    let mut reloaded = Vec::new();
    let mut dropped = Vec::new();
    for ns_id in all_ids {
        match state.engine.reload_namespace(&ns_id) {
            Ok(true) => reloaded.push(ns_id),
            Ok(false) => dropped.push(ns_id),
            Err(e) => eprintln!("[rollback] reload {} failed: {}", ns_id, e),
        }
    }
    Ok(Json(serde_json::json!({
        "rolled_back_to": req.sha,
        "reloaded_namespaces": reloaded,
        "dropped_namespaces": dropped,
    })))
}

// ── Diff ─────────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct DiffParams {
    from: String,
    to: String,
}

pub async fn namespace_diff(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(params): Query<DiffParams>,
) -> Result<Json<data_git::NamespaceDiff>, (StatusCode, String)> {
    let dir = state.data_dir.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "server has no data_dir".into(),
    ))?;
    if !state.engine.has_namespace(&id) {
        return Err((
            StatusCode::NOT_FOUND,
            format!("namespace '{}' not found", id),
        ));
    }
    // Reject non-hex shas.
    for sha in [&params.from, &params.to] {
        if sha.is_empty() || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err((StatusCode::BAD_REQUEST, format!("invalid sha: {}", sha)));
        }
    }
    let result = data_git::diff(&PathBuf::from(dir), &id, &params.from, &params.to)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(result))
}

/// L2b anti-Hebbian v2: feed queries as negative examples.
#[derive(serde::Deserialize)]
pub struct TrainNegativeRequest {
    namespace_id: String,
    queries: Vec<String>,
    #[serde(default)]
    not_intents: Vec<String>,
    #[serde(default = "default_alpha")]
    alpha: f32,
}

fn default_alpha() -> f32 {
    0.1
}

pub async fn train_negative(
    State(state): State<AppState>,
    Json(req): Json<TrainNegativeRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if req.alpha <= 0.0 || req.alpha > 0.3 {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
            "alpha must be in (0.0, 0.3]. Got {}. Recommended: 0.05 (gentle) to 0.15 (aggressive). \
             If you think you need more, run multiple rounds instead.",
            req.alpha
        ),
        ));
    }
    let h = state
        .engine
        .try_namespace(&req.namespace_id)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("namespace '{}' not found", req.namespace_id),
            )
        })?;
    let not_intents: Vec<String> = if req.not_intents.is_empty() {
        h.intent_ids()
    } else {
        req.not_intents
    };
    let queries_len = req.queries.len();
    let affected = not_intents.len();
    h.train_negative(&req.queries, &not_intents, req.alpha);
    maybe_commit(&state, &req.namespace_id);
    Ok(Json(serde_json::json!({
        "trained": req.namespace_id,
        "queries": queries_len,
        "intents_affected": affected,
        "alpha": req.alpha,
    })))
}

/// Rail 3 — reset: wipe L2 weights and rebuild from stored training phrases.
#[derive(serde::Deserialize)]
pub struct RebuildRequest {
    namespace_id: String,
}

pub async fn rebuild_namespace(
    State(state): State<AppState>,
    Json(req): Json<RebuildRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let h = state
        .engine
        .try_namespace(&req.namespace_id)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("namespace '{}' not found", req.namespace_id),
            )
        })?;
    let n_phrases_before: usize = h
        .intent_ids()
        .iter()
        .filter_map(|id| h.training(id))
        .map(|v| v.len())
        .sum();
    h.rebuild_l2();
    maybe_commit(&state, &req.namespace_id);
    Ok(Json(serde_json::json!({
        "rebuilt": req.namespace_id,
        "phrases_reindexed": n_phrases_before,
        "note": "L2 weights rebuilt from training phrases; negative-training log cleared.",
    })))
}

// --- Namespaces ---

pub async fn list_namespaces(State(state): State<AppState>) -> Json<serde_json::Value> {
    let modes = state.review_mode.read().unwrap();
    let mut namespaces: Vec<serde_json::Value> = state
        .engine
        .namespaces()
        .into_iter()
        .map(|id| {
            let h = state.engine.namespace(&id);
            let (info, version, intent_count) = (h.namespace_info(), h.version(), h.intent_count());
            serde_json::json!({
                "id": id,
                "name": info.name,
                "description": info.description,
                "auto_learn": modes.get(&id).map(|m| m == "auto").unwrap_or(false),
                "default_threshold": info.default_threshold,
                "version": version,
                "intent_count": intent_count,
            })
        })
        .collect();
    namespaces.sort_by(|a, b| {
        a["id"]
            .as_str()
            .unwrap_or("")
            .cmp(b["id"].as_str().unwrap_or(""))
    });
    Json(serde_json::json!(namespaces))
}

#[derive(serde::Deserialize)]
pub struct CreateNamespaceRequest {
    namespace_id: String,
    #[serde(default)]
    description: String,
}

pub async fn create_namespace(
    State(state): State<AppState>,
    Json(req): Json<CreateNamespaceRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let id = &req.namespace_id;
    if id.is_empty() || id.len() > 40 {
        return Err((
            StatusCode::BAD_REQUEST,
            "namespace ID must be 1–40 characters".to_string(),
        ));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "namespace ID must contain only lowercase letters, digits, hyphens, and underscores"
                .to_string(),
        ));
    }
    if state.engine.has_namespace(id.as_str()) {
        return Err((
            StatusCode::CONFLICT,
            format!("namespace '{}' already exists", id),
        ));
    }
    let h = state.engine.namespace(id);
    let _ = h.update_namespace(microresolve::NamespaceEdit {
        description: Some(req.description.clone()),
        ..Default::default()
    });
    maybe_commit(&state, &req.namespace_id);
    Ok(Json(serde_json::json!({"created": req.namespace_id})))
}

#[derive(serde::Deserialize)]
pub struct DeleteNamespaceRequest {
    namespace_id: String,
}

pub async fn delete_namespace(
    State(state): State<AppState>,
    Json(req): Json<DeleteNamespaceRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if req.namespace_id == "default" {
        return Err((
            StatusCode::BAD_REQUEST,
            "cannot delete default namespace".to_string(),
        ));
    }
    if !state.engine.remove_namespace(&req.namespace_id) {
        return Err((
            StatusCode::NOT_FOUND,
            format!("namespace '{}' not found", req.namespace_id),
        ));
    }
    if let Some(ref dir) = state.data_dir {
        let _ = std::fs::remove_dir_all(format!("{}/{}", dir, req.namespace_id));
        let _ = std::fs::remove_file(format!("{}/{}.json", dir, req.namespace_id));
    }
    state.log_store.lock().unwrap().drop_app(&req.namespace_id);
    state.review_mode.write().unwrap().remove(&req.namespace_id);
    Ok(Json(serde_json::json!({"deleted": req.namespace_id})))
}

#[derive(serde::Deserialize)]
pub struct UpdateNamespaceRequest {
    namespace_id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    auto_learn: Option<bool>,
    #[serde(default)]
    default_threshold: Option<f32>,
}

pub async fn update_namespace(
    State(state): State<AppState>,
    Json(req): Json<UpdateNamespaceRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    {
        let h = state
            .engine
            .try_namespace(&req.namespace_id)
            .ok_or_else(|| {
                (
                    StatusCode::NOT_FOUND,
                    format!("namespace '{}' not found", req.namespace_id),
                )
            })?;
        let edit = microresolve::NamespaceEdit {
            name: req.name.clone(),
            description: req.description.clone(),
            default_threshold: req
                .default_threshold
                .map(|t| if t < 0.0 { None } else { Some(t) }),
            ..Default::default()
        };
        let _ = h.update_namespace(edit);
        maybe_commit(&state, &req.namespace_id);
    }
    if let Some(auto_learn) = req.auto_learn {
        let mode = if auto_learn { "auto" } else { "manual" };
        state
            .review_mode
            .write()
            .unwrap()
            .insert(req.namespace_id.clone(), mode.to_string());
        if auto_learn {
            state.worker_notify.notify_one();
        }
    }
    Ok(Json(serde_json::json!({"updated": req.namespace_id})))
}

// --- Domains ---

pub async fn list_domain_groups(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let namespace_id = app_id_from_headers(&headers);
    let domains: Vec<serde_json::Value> = {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut names: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut domain_descs: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        if let Some(h) = state.engine.try_namespace(&namespace_id) {
            for id in h.intent_ids() {
                if let Some(colon) = id.find(':') {
                    *counts.entry(id[..colon].to_string()).or_default() += 1;
                    names.insert(id[..colon].to_string());
                }
            }
            for (domain, desc) in h.namespace_info().domain_descriptions {
                names.insert(domain.clone());
                domain_descs.insert(domain, desc);
            }
        }
        let mut names: Vec<String> = names.into_iter().collect();
        names.sort();
        names
            .into_iter()
            .map(|name| {
                let desc = domain_descs.get(&name).cloned().unwrap_or_default();
                serde_json::json!({
                    "name": name,
                    "description": desc,
                    "intent_count": counts.get(&name).copied().unwrap_or(0),
                })
            })
            .collect()
    };
    Json(serde_json::json!(domains))
}

#[derive(serde::Deserialize)]
pub struct CreateDomainRequest {
    domain: String,
    #[serde(default)]
    description: String,
}

pub async fn create_domain(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateDomainRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let id = &req.domain;
    if id.is_empty() || id.len() > 40 {
        return Err((
            StatusCode::BAD_REQUEST,
            "domain ID must be 1–40 characters".to_string(),
        ));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "domain ID must contain only lowercase letters, digits, hyphens, and underscores"
                .to_string(),
        ));
    }
    let namespace_id = app_id_from_headers(&headers);
    let h = state.engine.try_namespace(&namespace_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("namespace '{}' not found", namespace_id),
        )
    })?;
    let already_exists = h.domain_description(&req.domain).is_some();
    if already_exists {
        return Err((
            StatusCode::CONFLICT,
            format!("domain '{}' already exists", req.domain),
        ));
    }
    h.set_domain_description(&req.domain, &req.description);
    maybe_commit(&state, &namespace_id);
    Ok(Json(serde_json::json!({"created": req.domain})))
}

#[derive(serde::Deserialize)]
pub struct DeleteDomainRequest {
    domain: String,
}

pub async fn delete_domain(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DeleteDomainRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let namespace_id = app_id_from_headers(&headers);
    let h = state.engine.try_namespace(&namespace_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("namespace '{}' not found", namespace_id),
        )
    })?;
    h.remove_domain_description(&req.domain);
    let prefix = format!("{}:", req.domain);
    let to_remove: Vec<String> = h
        .intent_ids()
        .into_iter()
        .filter(|id| id.starts_with(&prefix))
        .collect();
    for id in to_remove {
        h.remove_intent(&id);
    }
    maybe_commit(&state, &namespace_id);
    if let Some(ref dir) = state.data_dir {
        let _ = std::fs::remove_dir_all(format!("{}/{}/{}", dir, namespace_id, req.domain));
    }
    Ok(Json(serde_json::json!({"deleted": req.domain})))
}

#[derive(serde::Deserialize)]
pub struct UpdateDomainRequest {
    domain: String,
    description: String,
}

pub async fn update_domain(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<UpdateDomainRequest>,
) -> Json<serde_json::Value> {
    let namespace_id = app_id_from_headers(&headers);
    if let Some(h) = state.engine.try_namespace(&namespace_id) {
        h.set_domain_description(&req.domain, &req.description);
        maybe_commit(&state, &namespace_id);
    }
    Json(serde_json::json!({"updated": req.domain}))
}
