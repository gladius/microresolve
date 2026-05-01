//! API key management endpoints. UI-driven CRUD on
//! `~/.config/microresolve/keys.json`.

use crate::key_store::KeyScope;
use crate::state::*;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route(
            "/api/auth/keys",
            axum::routing::get(list_keys).post(create_key),
        )
        .route("/api/auth/keys/{name}", axum::routing::delete(revoke_key))
}

/// List all keys (redacted — never returns full secret).
pub async fn list_keys(State(state): State<AppState>) -> Json<serde_json::Value> {
    let store = state.key_store.read().unwrap();
    Json(serde_json::json!({
        "enabled": store.is_enabled(),
        "keys": store.list_redacted(),
    }))
}

#[derive(serde::Deserialize)]
pub struct CreateKeyRequest {
    pub name: String,
    /// Optional scope — defaults to `library` if omitted. UI passes this
    /// explicitly when minting `admin` keys (e.g. for a teammate who
    /// needs full Studio access).
    #[serde(default)]
    pub scope: Option<KeyScope>,
}

/// Generate a new key. Returns the full key ONCE — caller must save it.
pub async fn create_key(
    State(state): State<AppState>,
    Json(req): Json<CreateKeyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if req.name.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "name required".to_string()));
    }
    let scope = req.scope.unwrap_or(KeyScope::Library);
    let mut store = state.key_store.write().unwrap();
    let key = store
        .create(req.name.trim(), scope)
        .map_err(|e| (StatusCode::CONFLICT, e))?;
    Ok(Json(serde_json::json!({
        "key": key,
        "name": req.name.trim(),
        "scope": scope,
        "warning": "This key is shown once. Save it now — it cannot be retrieved later.",
    })))
}

/// Revoke a key by name.
pub async fn revoke_key(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let mut store = state.key_store.write().unwrap();
    store
        .revoke(&name)
        .map_err(|e| (StatusCode::NOT_FOUND, e))?;
    Ok(StatusCode::NO_CONTENT)
}
