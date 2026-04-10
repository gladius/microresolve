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
    let descriptions = state.namespace_descriptions.read().unwrap();
    let mut namespaces: Vec<serde_json::Value> = routers.keys()
        .map(|id| serde_json::json!({
            "id": id,
            "description": descriptions.get(id).cloned().unwrap_or_default(),
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
    let router = Router::new();
    routers.insert(req.namespace_id.clone(), router);
    drop(routers);
    state.namespace_descriptions.write().unwrap()
        .insert(req.namespace_id.clone(), req.description.clone());
    // Create folder + _ns.json + empty _router.json
    if let Some(ref dir) = state.data_dir {
        let ns_dir = format!("{}/{}", dir, req.namespace_id);
        std::fs::create_dir_all(&ns_dir).ok();
        let ns_meta = serde_json::json!({"description": req.description});
        let _ = std::fs::write(
            format!("{}/_ns.json", ns_dir),
            serde_json::to_string_pretty(&ns_meta).unwrap_or_default(),
        );
    }
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
    state.namespace_descriptions.write().unwrap().remove(&req.namespace_id);
    state.domain_descriptions.write().unwrap().remove(&req.namespace_id);
    if let Some(ref dir) = state.data_dir {
        // Remove namespace folder (new format)
        let _ = std::fs::remove_dir_all(format!("{}/{}", dir, req.namespace_id));
        // Also remove old flat file if it exists (migration cleanup)
        let _ = std::fs::remove_file(format!("{}/{}.json", dir, req.namespace_id));
    }
    Ok(Json(serde_json::json!({"deleted": req.namespace_id})))
}

#[derive(serde::Deserialize)]
pub struct UpdateNamespaceRequest {
    namespace_id: String,
    description: String,
}

pub async fn update_namespace(
    State(state): State<AppState>,
    Json(req): Json<UpdateNamespaceRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if !state.routers.read().unwrap().contains_key(&req.namespace_id) {
        return Err((StatusCode::NOT_FOUND, format!("namespace '{}' not found", req.namespace_id)));
    }
    state.namespace_descriptions.write().unwrap()
        .insert(req.namespace_id.clone(), req.description.clone());
    if let Some(ref dir) = state.data_dir {
        let ns_dir = format!("{}/{}", dir, req.namespace_id);
        std::fs::create_dir_all(&ns_dir).ok();
        let meta = serde_json::json!({"description": req.description});
        let _ = std::fs::write(
            format!("{}/_ns.json", ns_dir),
            serde_json::to_string_pretty(&meta).unwrap_or_default(),
        );
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
    let domain_descs = state.domain_descriptions.read().unwrap();
    let ns_domain_descs = domain_descs.get(&namespace_id);

    let domains: Vec<serde_json::Value> = {
        // Intent-derived domain counts
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        if let Some(router) = routers.get(&namespace_id) {
            for id in router.intent_ids() {
                if let Some(colon) = id.find(':') {
                    *counts.entry(id[..colon].to_string()).or_default() += 1;
                }
            }
        }
        // Union with metadata-only domains (created explicitly but no intents yet)
        let mut names: std::collections::HashSet<String> = counts.keys().cloned().collect();
        if let Some(meta) = ns_domain_descs {
            for k in meta.keys() { names.insert(k.clone()); }
        }
        let mut names: Vec<String> = names.into_iter().collect();
        names.sort();
        names.into_iter().map(|name| {
            let desc = ns_domain_descs.and_then(|m| m.get(&name)).cloned().unwrap_or_default();
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
    if !state.routers.read().unwrap().contains_key(&namespace_id) {
        return Err((StatusCode::NOT_FOUND, format!("namespace '{}' not found", namespace_id)));
    }
    let already_exists = state.domain_descriptions.read().unwrap()
        .get(&namespace_id)
        .map(|m| m.contains_key(&req.domain))
        .unwrap_or(false);
    if already_exists {
        return Err((StatusCode::CONFLICT, format!("domain '{}' already exists", req.domain)));
    }
    state.domain_descriptions.write().unwrap()
        .entry(namespace_id.clone())
        .or_default()
        .insert(req.domain.clone(), req.description.clone());
    // Create the domain folder + _domain.json
    if let Some(ref dir) = state.data_dir {
        let domain_dir = format!("{}/{}/{}", dir, namespace_id, req.domain);
        std::fs::create_dir_all(&domain_dir).ok();
        let meta = serde_json::json!({"description": req.description});
        let _ = std::fs::write(
            format!("{}/_domain.json", domain_dir),
            serde_json::to_string_pretty(&meta).unwrap_or_default(),
        );
    }
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
    state.domain_descriptions.write().unwrap()
        .entry(namespace_id.clone())
        .or_default()
        .remove(&req.domain);
    // Remove all intents with this domain prefix from the router
    let prefix = format!("{}:", req.domain);
    {
        let mut routers = state.routers.write().unwrap();
        if let Some(router) = routers.get_mut(&namespace_id) {
            let to_remove: Vec<String> = router.intent_ids()
                .into_iter()
                .filter(|id| id.starts_with(&prefix))
                .collect();
            for id in to_remove {
                router.remove_intent(&id);
            }
            maybe_persist(&state, &namespace_id, router);
        }
    }
    // Remove the domain folder from disk
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
    state.domain_descriptions.write().unwrap()
        .entry(namespace_id.clone())
        .or_default()
        .insert(req.domain.clone(), req.description.clone());
    if let Some(ref dir) = state.data_dir {
        let domain_dir = format!("{}/{}/{}", dir, namespace_id, req.domain);
        std::fs::create_dir_all(&domain_dir).ok();
        let meta = serde_json::json!({"description": req.description});
        let _ = std::fs::write(
            format!("{}/_domain.json", domain_dir),
            serde_json::to_string_pretty(&meta).unwrap_or_default(),
        );
    }
    Json(serde_json::json!({"updated": req.domain}))
}
