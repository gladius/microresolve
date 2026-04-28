//! Query log endpoints — backed by LogStore.

use crate::log_store::LogQuery;
use crate::state::*;
use axum::{
    extract::{Query, State},
    http::HeaderMap,
    routing::get,
    Json,
};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/logs", get(get_logs))
        .route("/api/logs/stats", get(log_stats))
}

#[derive(serde::Deserialize)]
pub struct LogParams {
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
    /// "true" = resolved only, "false" = unresolved only, omit = all
    resolved: Option<bool>,
}
fn default_limit() -> usize {
    100
}

pub async fn get_logs(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<LogParams>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let result = state.log_store.lock().unwrap().query(&LogQuery {
        app_id: Some(app_id),
        resolved: params.resolved,
        since_ms: None,
        limit: params.limit,
        offset: params.offset,
    });

    Json(serde_json::json!({
        "total": result.total,
        "offset": params.offset,
        "limit": params.limit,
        "entries": result.records,
    }))
}

pub async fn log_stats(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let store = state.log_store.lock().unwrap();
    Json(serde_json::json!({
        "app_id": app_id,
        "total": store.count_total(&app_id),
        "unresolved": store.count_alive(&app_id),
        "all_apps": store.stats(),
    }))
}
