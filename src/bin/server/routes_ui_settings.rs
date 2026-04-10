//! UI settings — persisted on the server so the browser has no local state.

use axum::{extract::State, routing::get, Json};
use crate::state::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/settings", get(get_settings).patch(patch_settings))
}

pub async fn get_settings(State(state): State<AppState>) -> Json<UiSettings> {
    Json(state.ui_settings.read().unwrap().clone())
}

#[derive(serde::Deserialize)]
pub struct PatchSettings {
    pub selected_namespace_id: Option<String>,
    pub selected_domain: Option<String>,
    pub threshold: Option<f32>,
    pub languages: Option<Vec<String>>,
}

pub async fn patch_settings(
    State(state): State<AppState>,
    Json(req): Json<PatchSettings>,
) -> Json<UiSettings> {
    {
        let mut s = state.ui_settings.write().unwrap();
        if let Some(v) = req.selected_namespace_id { s.selected_namespace_id = v; }
        if let Some(v) = req.selected_domain { s.selected_domain = v; }
        if let Some(v) = req.threshold { s.threshold = v; }
        if let Some(mut v) = req.languages {
            if !v.contains(&"en".to_string()) { v.insert(0, "en".to_string()); }
            s.languages = v;
        }
    }
    save_ui_settings(&state);
    Json(state.ui_settings.read().unwrap().clone())
}
