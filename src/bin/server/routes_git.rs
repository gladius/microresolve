//! Git remote configuration + manual push endpoints.

use crate::data_git;
use crate::state::*;
use axum::{extract::State, http::StatusCode, Json};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route(
            "/api/settings/git",
            axum::routing::get(get_git_settings).put(put_git_settings),
        )
        .route("/api/git/push", axum::routing::post(push_now))
}

#[derive(serde::Serialize)]
pub struct GitSettings {
    pub remote_url: Option<String>,
    pub auto_push: bool,
    pub has_repo: bool,
}

pub async fn get_git_settings(State(state): State<AppState>) -> Json<GitSettings> {
    let remote_url = state.git_remote.read().unwrap().clone();
    let auto_push = remote_url.is_some();
    let has_repo = state
        .data_dir
        .as_deref()
        .map(|d| std::path::Path::new(d).join(".git").exists())
        .unwrap_or(false);
    Json(GitSettings {
        remote_url,
        auto_push,
        has_repo,
    })
}

#[derive(serde::Deserialize)]
pub struct PutGitRequest {
    remote_url: Option<String>,
}

pub async fn put_git_settings(
    State(state): State<AppState>,
    Json(req): Json<PutGitRequest>,
) -> Result<Json<GitSettings>, (StatusCode, String)> {
    let Some(ref data_dir) = state.data_dir else {
        return Err((
            StatusCode::BAD_REQUEST,
            "Server has no data directory configured".to_string(),
        ));
    };
    let dir = std::path::Path::new(data_dir);

    if let Some(ref url) = req.remote_url {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            data_git::set_remote(dir, trimmed)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
            *state.git_remote.write().unwrap() = Some(trimmed.to_string());
        } else {
            // Empty string → clear remote
            let _ = std::process::Command::new("git")
                .args(["remote", "remove", "origin"])
                .current_dir(dir)
                .status();
            *state.git_remote.write().unwrap() = None;
        }
    } else {
        // null → clear remote
        let _ = std::process::Command::new("git")
            .args(["remote", "remove", "origin"])
            .current_dir(dir)
            .status();
        *state.git_remote.write().unwrap() = None;
    }

    let remote_url = state.git_remote.read().unwrap().clone();
    let auto_push = remote_url.is_some();
    let has_repo = dir.join(".git").exists();
    Ok(Json(GitSettings {
        remote_url,
        auto_push,
        has_repo,
    }))
}

pub async fn push_now(State(state): State<AppState>) -> Json<serde_json::Value> {
    let Some(ref data_dir) = state.data_dir else {
        return Json(serde_json::json!({ "ok": false, "error": "No data directory configured" }));
    };
    let dir = std::path::Path::new(data_dir);
    match data_git::push(dir) {
        Ok(()) => Json(serde_json::json!({ "ok": true, "error": null })),
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e })),
    }
}
