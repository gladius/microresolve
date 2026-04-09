//! Query log and accuracy endpoints.

use axum::{
    extract::{State, Query},
    http::{StatusCode, HeaderMap},
    routing::{get, post, delete},
    Json,
};
use std::collections::HashMap;
use asv_router::{Router, IntentType};
use crate::state::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/logs", get(get_logs))
        .route("/api/logs", delete(clear_logs))
        .route("/api/logs/stats", get(log_stats))
        .route("/api/logs/accuracy", post(check_accuracy))
}

#[derive(serde::Deserialize)]
pub struct LogQuery {
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

pub fn default_limit() -> usize { 100 }

pub async fn get_logs(Query(params): Query<LogQuery>) -> Json<serde_json::Value> {
    let content = std::fs::read_to_string(LOG_FILE).unwrap_or_default();
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();

    // Most recent first
    let entries: Vec<serde_json::Value> = lines.iter().rev()
        .skip(params.offset)
        .take(params.limit)
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    Json(serde_json::json!({
        "total": total,
        "offset": params.offset,
        "limit": params.limit,
        "entries": entries,
    }))
}

pub async fn log_stats() -> Json<serde_json::Value> {
    let count = std::fs::read_to_string(LOG_FILE)
        .map(|c| c.lines().count())
        .unwrap_or(0);
    let size = std::fs::metadata(LOG_FILE)
        .map(|m| m.len())
        .unwrap_or(0);

    Json(serde_json::json!({
        "count": count,
        "size_bytes": size,
        "file": LOG_FILE,
    }))
}

pub async fn clear_logs(State(state): State<AppState>) -> StatusCode {
    // Truncate the log file
    if let Ok(mut file) = state.log.lock() {
        if let Ok(f) = std::fs::File::create(LOG_FILE) {
            *file = f;
        }
    }
    StatusCode::OK
}

// --- Accuracy check: re-route all logged queries against current router ---

pub async fn check_accuracy(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let content = std::fs::read_to_string(LOG_FILE).unwrap_or_default();

    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!({"error": "app not found"})),
    };

    let mut high = 0u32;
    let mut medium = 0u32;
    let mut low = 0u32;
    let mut miss = 0u32;
    let mut total = 0u32;
    let mut sample_misses: Vec<serde_json::Value> = Vec::new();

    for line in content.lines() {
        let entry: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let query = match entry["query"].as_str() {
            Some(q) => q,
            None => continue,
        };

        total += 1;
        let result = router.route_multi(query, 0.3);

        if result.intents.is_empty() {
            miss += 1;
            if sample_misses.len() < 10 {
                sample_misses.push(serde_json::json!({
                    "query": &query[..query.len().min(80)],
                }));
            }
        } else {
            // Best confidence among detected intents
            let best = result.intents.iter()
                .map(|i| i.confidence.as_str())
                .min_by_key(|c| match *c {
                    "high" => 0,
                    "medium" => 1,
                    _ => 2,
                })
                .unwrap_or("low");

            match best {
                "high" => high += 1,
                "medium" => medium += 1,
                _ => {
                    low += 1;
                    if sample_misses.len() < 10 {
                        let intents: Vec<String> = result.intents.iter()
                            .map(|i| format!("{}({:.1})", i.id, i.score))
                            .collect();
                        sample_misses.push(serde_json::json!({
                            "query": &query[..query.len().min(80)],
                            "detected": intents,
                        }));
                    }
                }
            }
        }
    }

    Json(serde_json::json!({
        "total": total,
        "high": high,
        "medium": medium,
        "low": low,
        "miss": miss,
        "high_pct": if total > 0 { high as f64 / total as f64 * 100.0 } else { 0.0 },
        "medium_pct": if total > 0 { medium as f64 / total as f64 * 100.0 } else { 0.0 },
        "low_pct": if total > 0 { low as f64 / total as f64 * 100.0 } else { 0.0 },
        "miss_pct": if total > 0 { miss as f64 / total as f64 * 100.0 } else { 0.0 },
        "pass_pct": if total > 0 { (high + medium) as f64 / total as f64 * 100.0 } else { 0.0 },
        "sample_issues": sample_misses,
    }))
}

