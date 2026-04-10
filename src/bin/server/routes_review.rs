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
        .route("/api/review/intent_seeds", post(review_intent_seeds))
        .route("/api/review/stats",        get(review_stats))
        .route("/api/review/mode",         get(get_review_mode))
        .route("/api/review/mode",         post(set_review_mode))
        .route("/api/similarity/build",    post(build_similarity))
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
    seeds_by_intent: HashMap<String, Vec<SeedWithLang>>,
}

#[derive(serde::Deserialize)]
pub struct SeedWithLang {
    seed: String,
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

    // Run seed pipeline
    let seeds_map: HashMap<String, Vec<String>> = req.seeds_by_intent.iter()
        .map(|(id, seeds)| (id.clone(), seeds.iter().map(|s| s.seed.clone()).collect()))
        .collect();

    let pipeline = seed_pipeline(&state, &app_id, &seeds_map, true, "en").await;

    // For each intent that got seeds added, learn situation n-grams from the original
    // failing query. CJK queries always learn; Latin queries learn only if the intent
    // already has situation patterns (avoiding noise from generic Latin bigrams).
    if !pipeline.added.is_empty() {
        let mut routers = state.routers.write().unwrap();
        if let Some(router) = routers.get_mut(&app_id) {
            let seen_intents: std::collections::HashSet<String> =
                pipeline.added.iter().map(|(intent, _)| intent.clone()).collect();
            for intent_id in &seen_intents {
                router.learn_situation(&original_query, intent_id);
            }
            maybe_persist(&state, &app_id, router);
        }
    }

    // Resolve this entry
    state.log_store.lock().unwrap().resolve(&app_id, req.id);

    // Re-check other unresolved flagged entries against updated router
    let mut auto_resolved = 0usize;
    if !pipeline.added.is_empty() {
        let routers = state.routers.read().unwrap();
        if let Some(router) = routers.get(&app_id) {
            let unresolved = state.log_store.lock().unwrap().query(&LogQuery {
                app_id: Some(app_id.clone()),
                resolved: Some(false),
                limit: 1000,
                ..Default::default()
            });
            for record in &unresolved.records {
                let result = router.route_multi(&record.query, 0.3);
                let passes = result.intents.iter()
                    .any(|i| i.confidence == "high" || i.confidence == "medium");
                if passes {
                    state.log_store.lock().unwrap().resolve(&app_id, record.id);
                    auto_resolved += 1;
                }
            }
        }
    }

    let blocked: Vec<serde_json::Value> = pipeline.blocked.iter()
        .map(|(intent, seed, reason)| serde_json::json!({"seed": seed, "intent": intent, "reason": reason}))
        .collect();

    Ok(Json(serde_json::json!({
        "status": "ok",
        "added": pipeline.added.len(),
        "blocked": blocked,
        "retried": pipeline.retried,
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
        "seeds_to_add": review.seeds_to_add,
        "seeds_blocked": review.seeds_blocked.iter().map(|(i,s,r)| serde_json::json!({"intent":i,"seed":s,"reason":r})).collect::<Vec<_>>(),
        "seeds_to_replace": review.seeds_to_replace.iter().map(|r| serde_json::json!({
            "intent": r.intent, "old_seed": r.old_seed, "new_seed": r.new_seed, "reason": r.reason,
        })).collect::<Vec<_>>(),
        "safe_to_apply": review.safe_to_apply,
        "summary": review.summary,
    })))
}

// ─── Supporting ──────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct IntentSeedsRequest {
    intent_ids: Vec<String>,
}

pub async fn review_intent_seeds(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<IntentSeedsRequest>,
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

pub async fn get_review_mode(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mode = state.review_mode.read().unwrap().clone();
    Json(serde_json::json!({"mode": mode}))
}

#[derive(serde::Deserialize)]
pub struct SetReviewModeRequest { mode: String }

pub async fn set_review_mode(
    State(state): State<AppState>,
    Json(req): Json<SetReviewModeRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if !["manual", "auto"].contains(&req.mode.as_str()) {
        return Err((StatusCode::BAD_REQUEST, "mode must be 'manual' or 'auto'".to_string()));
    }
    *state.review_mode.write().unwrap() = req.mode.clone();
    Ok(Json(serde_json::json!({"mode": req.mode})))
}

#[derive(serde::Deserialize)]
pub struct BuildSimilarityRequest {
    #[serde(default)]
    corpus: Vec<String>,
}

pub async fn build_similarity(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<BuildSimilarityRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let mut texts: Vec<String> = Vec::new();

    {
        let routers = state.routers.read().unwrap();
        if let Some(router) = routers.get(&app_id) {
            for intent_id in router.intent_ids() {
                if let Some(training) = router.get_training(&intent_id) {
                    texts.extend(training);
                }
            }
        }
    }

    // Include recent queries from log as real-world vocabulary
    {
        let store = state.log_store.lock().unwrap();
        let result = state.log_store.lock().unwrap().query(&LogQuery {
            app_id: Some(app_id.clone()),
            resolved: None,
            limit: 2000,
            ..Default::default()
        });
        for r in result.records {
            texts.push(r.query);
        }
    }

    texts.extend(req.corpus);
    let text_count = texts.len();

    {
        let mut routers = state.routers.write().unwrap();
        if let Some(router) = routers.get_mut(&app_id) {
            router.build_similarity(&texts);
            maybe_persist(&state, &app_id, router);
        }
    }

    Json(serde_json::json!({"status": "built", "texts_used": text_count}))
}
