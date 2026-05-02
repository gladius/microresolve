//! Review queue management and direct-learn endpoints.
//!
//! # Review queue
//! A filtered view of the log store: unresolved entries with flags (miss/low_confidence/false_positive).
//! No separate in-memory state — the log store IS the queue.
//!
//! # Endpoints
//! - `/api/review/queue` — list pending entries
//! - `/api/review/reject` — dismiss without learning
//! - `/api/review/fix` — apply user-supplied phrases via full pipeline (L0→L1→L2→L3)
//! - `/api/review/analyze` — run `full_review` (Turn 1 LLM judge + Turn 2 phrase gen)
//! - `/api/learn/now` — synchronous learn: `full_review` + `apply_review` without queueing
//! - `/api/report` — add a query to the review queue for LLM-judge review

use crate::log_store::{LogQuery, LogRecord};
use crate::pipeline::*;
use crate::state::*;
use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json,
};
use std::collections::HashMap;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/review/queue", get(review_queue))
        .route("/api/review/reject", post(review_reject))
        .route("/api/review/fix", post(review_fix))
        .route("/api/review/analyze", post(review_analyze))
        .route("/api/review/intent_phrases", post(review_intent_phrases))
        .route("/api/review/stats", get(review_stats))
        .route("/api/learn/now", post(learn_now))
        .route("/api/learn/words", post(learn_words))
        .route("/api/report", post(report_query))
}

// ─── Queue (filtered log view) ───────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct ReviewQueueParams {
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}
fn default_limit() -> usize {
    50
}

pub async fn review_queue(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<ReviewQueueParams>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    // Review-everything model: queue shows ALL unresolved entries. When auto-learn is
    // on, the worker resolves them automatically; when off, they stay here for manual
    // triage. No flag filter — the sidebar badge must match what the user sees.
    let result = state.log_store.lock().unwrap().query(&LogQuery {
        app_id: Some(app_id),
        resolved: Some(false),
        since_ms: None,
        limit: params.limit,
        offset: params.offset,
    });

    let items: Vec<serde_json::Value> = result
        .records
        .iter()
        .map(|r| {
            serde_json::json!({
                "id": r.id,
                "query": r.query,
                "detected": r.detected_intents,
                "confidence": r.confidence,
                "timestamp": r.timestamp_ms,
                "session_id": r.session_id,
            })
        })
        .collect();

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
    /// Intents correctly detected — used to fire L3 inhibition against wrong_detections.
    #[serde(default)]
    correct_intents: Vec<String>,
    /// Intents incorrectly detected (false positives) — L3 will suppress these.
    #[serde(default)]
    wrong_detections: Vec<String>,
}

#[derive(serde::Deserialize)]
pub struct PhraseWithLang {
    phrase: String,
    #[serde(default = "default_lang")]
    lang: String,
}

/// Apply fixes from the review queue, then resolve this entry.
///
/// Delegates to the unified auto-learn pipeline (`apply_review`):
/// phrase_pipeline (L0) → L2 Hebbian phrase learn → L3 inhibition → L1 synonym/morphology.
pub async fn review_fix(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReviewFixRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    // Read the log entry: get original query and detected intents
    let (original_query, detected) = {
        let mut store = state.log_store.lock().unwrap();
        let result = store.query(&LogQuery {
            app_id: Some(app_id.clone()),
            resolved: Some(false),
            ..Default::default()
        });
        result
            .records
            .iter()
            .find(|r| r.id == req.id)
            .map(|r| (r.query.clone(), r.detected_intents.clone()))
            .ok_or_else(|| (StatusCode::NOT_FOUND, "log entry not found".to_string()))?
    };

    // Build FullReviewResult from user's input — apply_review handles all layers
    let det_set: std::collections::HashSet<&str> = detected.iter().map(|s| s.as_str()).collect();
    let phrases_to_add: HashMap<String, Vec<String>> = req
        .phrases_by_intent
        .iter()
        .map(|(id, phrases)| {
            (
                id.clone(),
                phrases.iter().map(|p| p.phrase.clone()).collect(),
            )
        })
        .collect();
    let missed_intents: Vec<String> = req
        .phrases_by_intent
        .keys()
        .filter(|id| !det_set.contains(id.as_str()))
        .cloned()
        .collect();
    let lang = req
        .phrases_by_intent
        .values()
        .flat_map(|ps| ps.iter())
        .map(|p| p.lang.as_str())
        .find(|l| !l.is_empty())
        .unwrap_or("en")
        .to_string();

    let review_result = crate::pipeline::FullReviewResult {
        correct_intents: req.correct_intents,
        wrong_detections: req.wrong_detections,
        missed_intents,
        languages: vec![lang],
        detection_perfect: false,
        phrases_to_add,
        phrases_blocked: Vec::new(),
        summary: String::new(),
        spans_to_learn: vec![],
    };

    let phrases_added = apply_review(&state, &app_id, &review_result, &original_query).await;

    // Resolve this entry
    state.log_store.lock().unwrap().resolve(&app_id, req.id);

    // Re-check other unresolved entries: if something was learned, auto-resolve now-passing entries
    let mut auto_resolved = 0usize;
    if phrases_added > 0 {
        let unresolved = state.log_store.lock().unwrap().query(&LogQuery {
            app_id: Some(app_id.clone()),
            resolved: Some(false),
            limit: 1000,
            ..Default::default()
        });
        for record in &unresolved.records {
            let passes = state
                .engine
                .try_namespace(&app_id)
                .map(|h| !h.resolve(&record.query).is_empty())
                .unwrap_or(false);
            if passes {
                state.log_store.lock().unwrap().resolve(&app_id, record.id);
                auto_resolved += 1;
            }
        }
    }

    Ok(Json(serde_json::json!({
        "status": "ok",
        "added": phrases_added,
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
        let record = all
            .records
            .into_iter()
            .find(|r| r.id == req.id)
            .ok_or((StatusCode::NOT_FOUND, "log entry not found".to_string()))?;
        (record.query, record.detected_intents)
    };

    let review = full_review(&state, &app_id, &query, &detected, None)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(serde_json::to_value(&review).unwrap()))
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
    let mut result: HashMap<String, Vec<String>> = HashMap::new();
    if let Some(h) = state.engine.try_namespace(&app_id) {
        for id in &req.intent_ids {
            result.insert(
                id.clone(),
                h.training(id).unwrap_or_default(),
            );
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
    /// When provided (simulate/training flows), Turn 1 LLM is skipped entirely —
    /// correct/missed/wrong are computed by set math (free + exact).
    /// When absent (auto, manual without GT), Turn 1 LLM judges the routing.
    #[serde(default)]
    ground_truth: Option<Vec<String>>,
}

pub async fn learn_now(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LearnNowRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    if !state.engine.has_namespace(&app_id) {
        return Err((
            StatusCode::NOT_FOUND,
            format!("namespace '{}' not found", app_id),
        ));
    }

    let version_before = state
        .engine
        .try_namespace(&app_id)
        .map(|h| h.version())
        .unwrap_or(0);

    // Broadcast that we're starting (id=0 = ad-hoc, not from log store)
    let _ = state.event_tx.send(StudioEvent::LlmStarted {
        id: 0,
        query: req.query.clone(),
    });

    let gt_ref: Option<&[String]> = req.ground_truth.as_deref();
    let review = full_review(&state, &app_id, &req.query, &req.detected_intents, gt_ref)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let phrases_added = apply_review(&state, &app_id, &review, &req.query).await;

    let version_after = state
        .engine
        .try_namespace(&app_id)
        .map(|h| h.version())
        .unwrap_or(0);

    let model =
        std::env::var("LLM_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());

    let _ = state.event_tx.send(StudioEvent::LlmDone {
        id: 0,
        correct: review.correct_intents.clone(),
        wrong: review.wrong_detections.clone(),
        phrases_added,
        summary: review.summary.clone(),
    });

    if phrases_added > 0 {
        let _ = state.event_tx.send(StudioEvent::FixApplied {
            id: 0,
            phrases_added,
            phrases_replaced: 0,
            version_before,
            version_after,
        });
    }

    Ok(Json(serde_json::json!({
        "correct_intents":  review.correct_intents,
        "wrong_detections": review.wrong_detections,
        "missed_intents":   review.missed_intents,
        "phrases_added":    phrases_added,
        "summary":          review.summary,
        "languages":        review.languages,
        "version_before":   version_before,
        "version_after":    version_after,
        "model":            model,
    })))
}

// ─── Explicit report (for simulate/client-side flagging) ─────────────────────
// Adds a query to the review queue for LLM-judge review.
// Returns the log entry ID so the caller can track it via SSE.

#[derive(serde::Deserialize)]
pub struct ReportRequest {
    query: String,
    #[serde(default)]
    detected: Vec<String>,
    #[serde(default)]
    session_id: Option<String>,
}

pub async fn report_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReportRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let confidence = if req.detected.is_empty() {
        "none"
    } else {
        "low"
    };
    let log_id = log_query(
        &state,
        LogRecord {
            id: 0,
            query: req.query.clone(),
            app_id: app_id.clone(),
            detected_intents: req.detected,
            confidence: confidence.to_string(),
            session_id: req.session_id,
            timestamp_ms: now_ms(),
            router_version: state
                .engine
                .try_namespace(&app_id)
                .map(|h| h.version())
                .unwrap_or(0),
            source: "client_report".to_string(),
        },
    );
    state.worker_notify.notify_one();
    Json(serde_json::json!({ "id": log_id }))
}

// ── Learn words directly (no LLM) ────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct LearnWordsRequest {
    intent_id: String,
    words: Vec<String>,
}

/// Learn specific words for an intent. No LLM call — direct word→intent learning.
/// Used for testing and for systems that extract key_words externally.
pub async fn learn_words(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<LearnWordsRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    if !state.engine.has_namespace(&app_id) {
        return Err((
            StatusCode::NOT_FOUND,
            format!("namespace '{}' not found", app_id),
        ));
    }

    let word_refs: Vec<&str> = req.words.iter().map(|s| s.as_str()).collect();
    let count = word_refs.len();

    state
        .engine
        .namespace(&app_id)
        .learn_query_words(&word_refs, &req.intent_id);

    if let Some(h) = state.engine.try_namespace(&app_id) {
        h.flush().ok();
    }

    Ok(Json(serde_json::json!({
        "learned": count,
        "intent": req.intent_id,
    })))
}
