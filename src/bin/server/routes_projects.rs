//! Namespace and domain management endpoints.
//!
//! Hierarchy: Namespace → Domain → Intent
//!   Namespace: isolated Router workspace, selected via X-Namespace-ID header
//!   Domain:    logical intent group derived from "domain:intent_id" prefix
//!   Intent:    leaf routing target

use asv_router::Router;
use axum::{
    extract::State,
    http::{StatusCode, HeaderMap},
    routing::{get, post, delete, patch},
    Json,
};
use crate::state::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/namespaces", get(list_namespaces))
        .route("/api/namespaces", post(create_namespace))
        .route("/api/namespaces", delete(delete_namespace))
        .route("/api/namespaces", patch(update_namespace))
        .route("/api/domains", get(list_domain_groups))
        .route("/api/domains", post(create_domain))
        .route("/api/domains", patch(update_domain))
        .route("/api/domains", delete(delete_domain))
}

// --- Namespaces ---

pub async fn list_namespaces(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let routers = state.routers.read().unwrap();
    let modes = state.review_mode.read().unwrap();
    let mut namespaces: Vec<serde_json::Value> = routers.iter()
        .map(|(id, r)| serde_json::json!({
            "id": id,
            "name": r.namespace_name(),
            "description": r.namespace_description(),
            "auto_learn": modes.get(id).map(|m| m == "auto").unwrap_or(false),
        }))
        .collect();
    namespaces.sort_by(|a, b| {
        a["id"].as_str().unwrap_or("").cmp(b["id"].as_str().unwrap_or(""))
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
    let mut routers = state.routers.write().unwrap();
    if routers.contains_key(&req.namespace_id) {
        return Err((StatusCode::CONFLICT, format!("namespace '{}' already exists", req.namespace_id)));
    }
    let mut router = Router::new();
    router.set_namespace_description(&req.description);
    maybe_persist(&state, &req.namespace_id, &router);
    routers.insert(req.namespace_id.clone(), router);
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
        return Err((StatusCode::BAD_REQUEST, "cannot delete default namespace".to_string()));
    }
    let mut routers = state.routers.write().unwrap();
    if routers.remove(&req.namespace_id).is_none() {
        return Err((StatusCode::NOT_FOUND, format!("namespace '{}' not found", req.namespace_id)));
    }
    drop(routers);
    if let Some(ref dir) = state.data_dir {
        let _ = std::fs::remove_dir_all(format!("{}/{}", dir, req.namespace_id));
        // Also remove old flat file if it exists (migration cleanup)
        let _ = std::fs::remove_file(format!("{}/{}.json", dir, req.namespace_id));
    }
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
}

pub async fn update_namespace(
    State(state): State<AppState>,
    Json(req): Json<UpdateNamespaceRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    {
        let mut routers = state.routers.write().unwrap();
        let router = routers.get_mut(&req.namespace_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("namespace '{}' not found", req.namespace_id)))?;
        if let Some(ref name) = req.name { router.set_namespace_name(name); }
        if let Some(ref desc) = req.description { router.set_namespace_description(desc); }
        maybe_persist(&state, &req.namespace_id, router);
    }
    if let Some(auto_learn) = req.auto_learn {
        let mode = if auto_learn { "auto" } else { "manual" };
        state.review_mode.write().unwrap().insert(req.namespace_id.clone(), mode.to_string());
        if auto_learn { state.worker_notify.notify_one(); }
    }
    Ok(Json(serde_json::json!({"updated": req.namespace_id})))
}

// --- Domains ---

pub async fn list_domain_groups(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let namespace_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();

    let domains: Vec<serde_json::Value> = {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut names: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut domain_descs: std::collections::HashMap<String, String> = std::collections::HashMap::new();

        if let Some(router) = routers.get(&namespace_id) {
            for id in router.intent_ids() {
                if let Some(colon) = id.find(':') {
                    *counts.entry(id[..colon].to_string()).or_default() += 1;
                    names.insert(id[..colon].to_string());
                }
            }
            // Include explicitly-created domains (with descriptions but no intents yet)
            for (domain, desc) in router.domain_descriptions() {
                names.insert(domain.clone());
                domain_descs.insert(domain.clone(), desc.clone());
            }
        }
        let mut names: Vec<String> = names.into_iter().collect();
        names.sort();
        names.into_iter().map(|name| {
            let desc = domain_descs.get(&name).cloned().unwrap_or_default();
            serde_json::json!({
                "name": name,
                "description": desc,
                "intent_count": counts.get(&name).copied().unwrap_or(0),
            })
        }).collect()
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
    let namespace_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&namespace_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("namespace '{}' not found", namespace_id)))?;
    if router.domain_description(&req.domain).is_some() {
        return Err((StatusCode::CONFLICT, format!("domain '{}' already exists", req.domain)));
    }
    router.set_domain_description(&req.domain, &req.description);
    maybe_persist(&state, &namespace_id, router);
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
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&namespace_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("namespace '{}' not found", namespace_id)))?;
    router.remove_domain_description(&req.domain);
    // Remove all intents with this domain prefix
    let prefix = format!("{}:", req.domain);
    let to_remove: Vec<String> = router.intent_ids()
        .into_iter()
        .filter(|id| id.starts_with(&prefix))
        .collect();
    for id in to_remove {
        router.remove_intent(&id);
    }
    maybe_persist(&state, &namespace_id, router);
    // Remove the domain folder from disk (save_to_dir cleanup handles intent files;
    // the folder itself is removed here so empty domain dirs don't linger)
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
    let mut routers = state.routers.write().unwrap();
    if let Some(router) = routers.get_mut(&namespace_id) {
        router.set_domain_description(&req.domain, &req.description);
        maybe_persist(&state, &namespace_id, router);
    }
    Json(serde_json::json!({"updated": req.domain}))
}
