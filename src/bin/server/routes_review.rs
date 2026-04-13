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
        .route("/api/review/queue",          get(review_queue))
        .route("/api/review/reject",         post(review_reject))
        .route("/api/review/fix",            post(review_fix))
        .route("/api/review/analyze",        post(review_analyze))
        .route("/api/review/intent_phrases", post(review_intent_phrases))
        .route("/api/review/stats",          get(review_stats))
        .route("/api/learn/now",             post(learn_now))
        .route("/api/report",                post(report_query))
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
        "suppressor_words": review.suppressor_words,
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

// ─── Synchronous learn (instant queue) ───────────────────────────────────────
// Bypasses the background worker. Runs full_review + apply_review immediately,
// fires SSE events so the UI live feed updates, and returns the result.
// Used by Manual and Simulate tabs where the user wants instant feedback.

#[derive(serde::Deserialize)]
pub struct LearnNowRequest {
    query: String,
    #[serde(default)]
    detected_intents: Vec<String>,
}

pub async fn learn_now(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LearnNowRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    // Ensure namespace exists
    if !state.routers.read().unwrap().contains_key(&app_id) {
        return Err((StatusCode::NOT_FOUND, format!("namespace '{}' not found", app_id)));
    }

    let version_before = state.routers.read().unwrap()
        .get(&app_id).map(|r| r.version()).unwrap_or(0);

    // Broadcast that we're starting (id=0 = ad-hoc, not from log store)
    let _ = state.event_tx.send(StudioEvent::LlmStarted {
        id: 0,
        query: req.query.clone(),
    });

    let review = full_review(&state, &app_id, &req.query, &req.detected_intents).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let (phrases_added, suppressors_added) =
        apply_review(&state, &app_id, &review, &req.query).await;

    let version_after = state.routers.read().unwrap()
        .get(&app_id).map(|r| r.version()).unwrap_or(0);

    let model = std::env::var("LLM_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());

    let _ = state.event_tx.send(StudioEvent::LlmDone {
        id: 0,
        correct: review.correct_intents.clone(),
        wrong: review.wrong_detections.clone(),
        phrases_added,
        summary: review.summary.clone(),
    });

    if phrases_added > 0 || suppressors_added > 0 {
        let _ = state.event_tx.send(StudioEvent::FixApplied {
            id: 0,
            phrases_added,
            phrases_replaced: suppressors_added,
            version_before,
            version_after,
        });
    }

    Ok(Json(serde_json::json!({
        "correct_intents":  review.correct_intents,
        "wrong_detections": review.wrong_detections,
        "missed_intents":   review.missed_intents,
        "phrases_added":    phrases_added,
        "suppressors_added": suppressors_added,
        "summary":          review.summary,
        "languages":        review.languages,
        "version_before":   version_before,
        "version_after":    version_after,
        "model":            model,
    })))
}

// ─── Explicit report (for simulate/client-side flagging) ─────────────────────
// Adds a query to the review queue with an explicit flag.
// Returns the log entry ID so the caller can track it via SSE.

#[derive(serde::Deserialize)]
pub struct ReportRequest {
    query: String,
    #[serde(default)]
    detected: Vec<String>,
    #[serde(default)]
    flag: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
}

pub async fn report_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReportRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let flag = req.flag.unwrap_or_else(|| {
        if req.detected.is_empty() { "miss".to_string() } else { "low_confidence".to_string() }
    });
    let log_id = log_query(&state, LogRecord {
        id: 0,
        query: req.query.clone(),
        app_id: app_id.clone(),
        detected_intents: req.detected,
        confidence: if flag == "miss" { "none".to_string() } else { "low".to_string() },
        flag: Some(flag),
        session_id: req.session_id,
        timestamp_ms: now_ms(),
        router_version: state.routers.read().unwrap()
            .get(&app_id).map(|r| r.version()).unwrap_or(0),
        source: "client_report".to_string(),
    });
    state.worker_notify.notify_one();
    Json(serde_json::json!({ "id": log_id }))
}
