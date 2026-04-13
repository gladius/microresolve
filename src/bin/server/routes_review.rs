//! Review system: queue from log store, fix, analyze.
//!
//! The review queue is no longer a separate in-memory structure.
//! It is a filtered view of the log store: unresolved entries with flags.

use axum::{
    extract::{State, Query},
    http::{StatusCode, HeaderMap},
    routing::{get, post},
    Json,
};
use std::collections::HashMap;
use crate::state::*;
use crate::log_store::{LogQuery, LogRecord};
use crate::llm::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/review/queue",        get(review_queue))
        .route("/api/review/reject",       post(review_reject))
        .route("/api/review/fix",          post(review_fix))
        .route("/api/review/analyze",      post(review_analyze))
        .route("/api/review/intent_phrases", post(review_intent_phrases))
        .route("/api/review/stats",        get(review_stats))
        .route("/api/review/mode",         get(get_review_mode))
        .route("/api/review/mode",         post(set_review_mode))
}

// ─── Queue (filtered log view) ───────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct ReviewQueueParams {
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
    /// Filter by flag: "miss", "low_confidence", "false_positive"
    flag: Option<String>,
}
fn default_limit() -> usize { 50 }

pub async fn review_queue(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<ReviewQueueParams>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let result = state.log_store.lock().unwrap().query(&LogQuery {
        app_id: Some(app_id),
        flag: params.flag,
        flagged_only: true,  // review queue = flagged entries only
        resolved: Some(false),
        since_ms: None,
        limit: params.limit,
        offset: params.offset,
    });

    let items: Vec<serde_json::Value> = result.records.iter().map(|r| serde_json::json!({
        "id": r.id,
        "query": r.query,
        "detected": r.detected_intents,
        "flag": r.flag,
        "confidence": r.confidence,
        "timestamp": r.timestamp_ms,
        "session_id": r.session_id,
    })).collect();

    Json(serde_json::json!({
        "total": result.total,
        "items": items,
    }))
}

// ─── Actions ─────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct ReviewActionRequest {
    id: u64,
}

/// Dismiss: mark as resolved without applying any fix.
pub async fn review_reject(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReviewActionRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let resolved = state.log_store.lock().unwrap().resolve(&app_id, req.id);
    if resolved {
        Ok(Json(serde_json::json!({"status": "ok"})))
    } else {
        Err((StatusCode::NOT_FOUND, "log entry not found".to_string()))
    }
}

#[derive(serde::Deserialize)]
pub struct ReviewFixRequest {
    id: u64,
    phrases_by_intent: HashMap<String, Vec<PhraseWithLang>>,
}

#[derive(serde::Deserialize)]
pub struct PhraseWithLang {
    phrase: String,
    #[serde(default = "default_lang")]
    lang: String,
}

/// Apply seed fixes, then resolve this entry and re-check similar unresolved entries.
pub async fn review_fix(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReviewFixRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    // Extract original query and verify entry exists
    let original_query = {
        let mut store = state.log_store.lock().unwrap();
        let result = store.query(&LogQuery {
            app_id: Some(app_id.clone()),
            resolved: Some(false),
            ..Default::default()
        });
        result.records.iter()
            .find(|r| r.id == req.id)
            .map(|r| r.query.clone())
            .ok_or_else(|| (StatusCode::NOT_FOUND, "log entry not found".to_string()))?
    };

    // Run phrase pipeline
    let seeds_map: HashMap<String, Vec<String>> = req.phrases_by_intent.iter()
        .map(|(id, phrases)| (id.clone(), phrases.iter().map(|s| s.phrase.clone()).collect()))
        .collect();

    let pipeline = phrase_pipeline(&state, &app_id, &seeds_map, true, "en").await;

    // Resolve this entry
    state.log_store.lock().unwrap().resolve(&app_id, req.id);

    // Re-check other unresolved flagged entries against updated router
    let mut auto_resolved = 0usize;
    if !pipeline.added.is_empty() {
        let unresolved = state.log_store.lock().unwrap().query(&LogQuery {
            app_id: Some(app_id.clone()),
            resolved: Some(false),
            limit: 1000,
            ..Default::default()
        });
        for record in &unresolved.records {
            // Re-check via Hebbian L2
            let passes = {
                let ig_map = state.intent_graph.read().unwrap();
                let heb_map = state.hebbian.read().unwrap();
                if let (Some(ig), Some(heb)) = (ig_map.get(&app_id), heb_map.get(&app_id)) {
                    let pre = heb.preprocess(&record.query);
                    let (scores, _) = ig.score_multi_normalized(&pre.expanded, 0.3, 1.5);
                    !scores.is_empty()
                } else {
                    false
                }
            };
            if passes {
                state.log_store.lock().unwrap().resolve(&app_id, record.id);
                auto_resolved += 1;
            }
        }
    }

    let blocked: Vec<serde_json::Value> = pipeline.blocked.iter()
        .map(|(intent, phrase, reason)| serde_json::json!({"phrase": phrase, "intent": intent, "reason": reason}))
        .collect();

    Ok(Json(serde_json::json!({
        "status": "ok",
        "added": pipeline.added.len(),
        "blocked": blocked,
        "initially_blocked": pipeline.initially_blocked,
        "recovered_by_retry": pipeline.recovered_by_retry,
        "auto_resolved": auto_resolved,
    })))
}

/// Run 3-turn LLM analysis on a log entry.
pub async fn review_analyze(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReviewActionRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    let (query, detected) = {
        let mut store = state.log_store.lock().unwrap();
        let all = store.query(&LogQuery {
            app_id: Some(app_id.clone()),
            resolved: None,
            limit: 5000,
            ..Default::default()
        });
        let record = all.records.into_iter()
            .find(|r| r.id == req.id)
            .ok_or((StatusCode::NOT_FOUND, "log entry not found".to_string()))?;
        (record.query, record.detected_intents)
    };

    let review = full_review(&state, &app_id, &query, &detected).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(serde_json::json!({
        "correct_intents": review.correct_intents,
        "wrong_detections": review.wrong_detections,
        "languages": review.languages,
        "phrases_to_add": review.phrases_to_add,
        "phrases_blocked": review.phrases_blocked.iter().map(|(i,s,r)| serde_json::json!({"intent":i,"phrase":s,"reason":r})).collect::<Vec<_>>(),
        "phrases_to_replace": review.phrases_to_replace.iter().map(|r| serde_json::json!({
            "intent": r.intent, "old_phrase": r.old_phrase, "new_phrase": r.new_phrase, "reason": r.reason,
        })).collect::<Vec<_>>(),
        "safe_to_apply": review.safe_to_apply,
        "summary": review.summary,
    })))
}

// ─── Supporting ──────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct IntentPhrasesRequest {
    intent_ids: Vec<String>,
}

pub async fn review_intent_phrases(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<IntentPhrasesRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let mut result: HashMap<String, Vec<String>> = HashMap::new();
    if let Some(router) = routers.get(&app_id) {
        for id in &req.intent_ids {
            result.insert(id.clone(), router.get_training(id).unwrap_or_default());
        }
    }
    Json(serde_json::json!(result))
}

pub async fn review_stats(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let store = state.log_store.lock().unwrap();
    let unresolved = store.count_alive(&app_id);
    let total = store.count_total(&app_id);
    Json(serde_json::json!({
        "total": total,
        "pending": unresolved,
    }))
}

pub async fn get_review_mode(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let mode = get_ns_mode(&state, &app_id);
    Json(serde_json::json!({"mode": mode, "app_id": app_id}))
}

#[derive(serde::Deserialize)]
pub struct SetReviewModeRequest { mode: String }

pub async fn set_review_mode(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetReviewModeRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if !["manual", "auto"].contains(&req.mode.as_str()) {
        return Err((StatusCode::BAD_REQUEST, "mode must be 'manual' or 'auto'".to_string()));
    }
    let app_id = app_id_from_headers(&headers);
    state.review_mode.write().unwrap().insert(app_id.clone(), req.mode.clone());
    state.worker_notify.notify_one(); // wake worker in case mode just switched to auto
    Ok(Json(serde_json::json!({"mode": req.mode, "app_id": app_id})))
}
