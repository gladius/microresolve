//! Review system: report, queue, fix, analyze.

use axum::{
    extract::{State, Query},
    http::{StatusCode, HeaderMap},
    routing::{get, post, delete},
    Json,
};
use std::collections::HashMap;
use asv_router::{Router, IntentType};
use crate::state::*;
use crate::llm::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/report", post(report_query))
        .route("/api/review/queue", get(review_queue))
        .route("/api/review/approve", post(review_approve))
        .route("/api/review/reject", post(review_reject))
        .route("/api/review/fix", post(review_fix))
        .route("/api/review/analyze", post(review_analyze))
        .route("/api/review/intent_seeds", post(review_intent_seeds))
        .route("/api/review/stats", get(review_stats))
        .route("/api/review/mode", get(get_review_mode))
        .route("/api/review/mode", post(set_review_mode))
        .route("/api/similarity/build", post(build_similarity))
}


#[derive(serde::Deserialize)]
pub struct ReportRequest {
    query: String,
    detected: Vec<String>,
    flag: String,
    #[serde(default)]
    session_id: Option<String>,
}

pub async fn report_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReportRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let id = state.review_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let item = ReviewItem {
        id,
        query: req.query.clone(),
        detected: req.detected,
        flag: req.flag.clone(),
        suggested_intent: None,
        suggested_seed: None,
        app_id: app_id.clone(),
        timestamp: now_ms(),
        session_id: req.session_id,
    };

    // In auto_learn or auto_review mode, run full 3-turn review
    let mode = state.review_mode.read().unwrap().clone();
    let mut item = item;

    if (mode == "auto_learn" || mode == "auto_review") && state.llm_key.is_some() {
        if let Ok(review) = full_review(&state, &app_id, &req.query, &item.detected).await {
            item.suggested_intent = Some(review.correct_intents.join(", "));
            if let Some(first_seeds) = review.seeds_to_add.values().next() {
                item.suggested_seed = first_seeds.first().cloned();
            }

            // Auto-learn: apply everything (seeds + replacements)
            if mode == "auto_learn" {
                let (added, replaced) = apply_review(&state, &app_id, &review).await;
                eprintln!("[auto_learn] query=\"{}\" added={} replaced={}", &req.query[..req.query.len().min(50)], added, replaced);
                return Json(serde_json::json!({"id": id, "status": "auto_applied", "added": added, "replaced": replaced}));
            }
        }
    }

    let mut queue = state.review_queue.write().unwrap();
    queue.push(item);

    // Cap queue at 10000 items
    if queue.len() > 10000 {
        let excess = queue.len() - 10000;
        queue.drain(..excess);
    }

    Json(serde_json::json!({"id": id, "status": "queued"}))
}

#[derive(serde::Deserialize)]
pub struct ReviewQueueParams {
    #[serde(default = "default_review_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

pub fn default_review_limit() -> usize { 50 }

pub async fn review_queue(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<ReviewQueueParams>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let queue = state.review_queue.read().unwrap();

    let filtered: Vec<&ReviewItem> = queue.iter()
        .filter(|item| item.app_id == app_id)
        .collect();

    let total = filtered.len();
    let items: Vec<serde_json::Value> = filtered.iter()
        .skip(params.offset)
        .take(params.limit)
        .map(|item| serde_json::json!({
            "id": item.id,
            "query": item.query,
            "detected": item.detected,
            "flag": item.flag,
            "suggested_intent": item.suggested_intent,
            "suggested_seed": item.suggested_seed,
            "timestamp": item.timestamp,
            "session_id": item.session_id,
        }))
        .collect();

    Json(serde_json::json!({
        "total": total,
        "items": items,
    }))
}

#[derive(serde::Deserialize)]
pub struct ReviewActionRequest {
    id: u64,
}

pub async fn review_approve(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReviewActionRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    // Extract data and remove from queue
    let (query, intent) = {
        let queue = state.review_queue.read().unwrap();
        let item = queue.iter()
            .find(|i| i.id == req.id && i.app_id == app_id)
            .ok_or((StatusCode::NOT_FOUND, "review item not found".to_string()))?;
        let intent = item.suggested_intent.clone()
            .ok_or((StatusCode::BAD_REQUEST, "no suggestion to approve".to_string()))?;
        (item.query.clone(), intent)
    };

    // Apply the fix
    {
        let mut routers = state.routers.write().unwrap();
        if let Some(router) = routers.get_mut(&app_id) {
            router.learn(&query, &intent);
            maybe_persist(&state, &app_id, router);
        }
    }

    // Remove from queue
    let mut queue = state.review_queue.write().unwrap();
    queue.retain(|i| !(i.id == req.id && i.app_id == app_id));

    Ok(Json(serde_json::json!({"status": "ok", "intent": intent})))
}

pub async fn review_reject(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReviewActionRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut queue = state.review_queue.write().unwrap();

    let existed = queue.iter().any(|i| i.id == req.id && i.app_id == app_id);
    if !existed {
        return Err((StatusCode::NOT_FOUND, "review item not found".to_string()));
    }

    queue.retain(|i| !(i.id == req.id && i.app_id == app_id));

    Ok(Json(serde_json::json!({"status": "ok"})))
}

#[derive(serde::Deserialize)]
pub struct ReviewFixRequest {
    id: u64,
    /// Map of intent_id → list of {seed, lang} to add
    /// e.g. {"return_item": [{"seed": "received wrong item", "lang": "en"}]}
    seeds_by_intent: HashMap<String, Vec<SeedWithLang>>,
}

#[derive(serde::Deserialize)]
pub struct SeedWithLang {
    seed: String,
    #[serde(default = "default_lang")]
    lang: String,
}

pub async fn review_fix(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReviewFixRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    // Verify item exists
    {
        let queue = state.review_queue.read().unwrap();
        if !queue.iter().any(|i| i.id == req.id && i.app_id == app_id) {
            return Err((StatusCode::NOT_FOUND, "review item not found".to_string()));
        }
    }

    // Convert to pipeline input format
    let seeds_map: HashMap<String, Vec<String>> = req.seeds_by_intent.iter()
        .map(|(id, seeds)| (id.clone(), seeds.iter().map(|s| s.seed.clone()).collect()))
        .collect();

    // Run through shared pipeline (guard + one LLM retry for collisions)
    let pipeline_result = seed_pipeline(&state, &app_id, &seeds_map, true).await;

    // Re-check ALL items in queue against updated index
    let mut resolved_count = 0;
    if !pipeline_result.added.is_empty() {
        let routers_read = state.routers.read().unwrap();
        if let Some(router) = routers_read.get(&app_id) {
            let mut queue = state.review_queue.write().unwrap();
            let before = queue.len();
            queue.retain(|item| {
                if item.app_id != app_id { return true; }
                let result = router.route_multi(&item.query, 0.3);
                let passes = result.intents.iter()
                    .any(|i| i.confidence == "high" || i.confidence == "medium");
                !passes
            });
            resolved_count = before - queue.len();
        }
    }

    let blocked: Vec<serde_json::Value> = pipeline_result.blocked.iter()
        .map(|(intent, seed, reason)| serde_json::json!({
            "seed": seed, "intent": intent, "reason": reason,
        })).collect();

    let suggestions: Vec<serde_json::Value> = pipeline_result.suggestions.iter()
        .map(|(intent, seed)| serde_json::json!({
            "intent": intent, "seed": seed,
        })).collect();

    Ok(Json(serde_json::json!({
        "status": "ok",
        "added": pipeline_result.added.len(),
        "blocked": blocked,
        "retried": pipeline_result.retried,
        "suggestions": suggestions,
        "resolved_count": resolved_count,
    })))
}

pub async fn review_stats(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let queue = state.review_queue.read().unwrap();

    let app_items: Vec<&ReviewItem> = queue.iter().filter(|i| i.app_id == app_id).collect();
    Json(serde_json::json!({
        "total": app_items.len(),
        "pending": app_items.len(),
    }))
}

/// Analyze a review item using full 3-turn review.
/// Returns the complete analysis for the UI to display.
pub async fn review_analyze(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReviewActionRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    let (query, detected) = {
        let queue = state.review_queue.read().unwrap();
        let item = queue.iter()
            .find(|i| i.id == req.id && i.app_id == app_id)
            .ok_or((StatusCode::NOT_FOUND, "review item not found".to_string()))?;
        (item.query.clone(), item.detected.clone())
    };

    let review = full_review(&state, &app_id, &query, &detected).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(serde_json::json!({
        "correct_intents": review.correct_intents,
        "wrong_detections": review.wrong_detections,
        "languages": review.languages,
        "seeds_to_add": review.seeds_to_add,
        "seeds_blocked": review.seeds_blocked.iter().map(|(i, s, r)| serde_json::json!({"intent": i, "seed": s, "reason": r})).collect::<Vec<_>>(),
        "seeds_to_replace": review.seeds_to_replace.iter().map(|r| serde_json::json!({
            "intent": r.intent, "old_seed": r.old_seed, "new_seed": r.new_seed, "reason": r.reason,
        })).collect::<Vec<_>>(),
        "safe_to_apply": review.safe_to_apply,
        "summary": review.summary,
    })))
}

/// Get current seeds for specific intents (used by review UI to show what's in the index)
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
            let seeds = router.get_training(id).unwrap_or_default();
            result.insert(id.clone(), seeds);
        }
    }

    Json(serde_json::json!(result))
}

#[derive(serde::Deserialize)]
pub struct BuildSimilarityRequest {
    #[serde(default)]
    corpus: Vec<String>,
}

/// Build distributional similarity index from seeds + review queries + optional corpus.
pub async fn build_similarity(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<BuildSimilarityRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);

    // Collect texts: seeds + review queue queries
    let mut texts: Vec<String> = Vec::new();

    // Add all seed phrases
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

    // Add review queue queries (real customer language)
    {
        let queue = state.review_queue.read().unwrap();
        for item in queue.iter() {
            if item.app_id == app_id {
                texts.push(item.query.clone());
            }
        }
    }

    // Add external corpus if provided
    texts.extend(req.corpus);

    let text_count = texts.len();

    // Build similarity
    {
        let mut routers = state.routers.write().unwrap();
        if let Some(router) = routers.get_mut(&app_id) {
            router.build_similarity(&texts);
            maybe_persist(&state, &app_id, router);
        }
    }

    Json(serde_json::json!({
        "status": "built",
        "texts_used": text_count,
        "has_similarity": true,
    }))
}

pub async fn get_review_mode(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let mode = state.review_mode.read().unwrap().clone();
    Json(serde_json::json!({"mode": mode}))
}

#[derive(serde::Deserialize)]
pub struct SetReviewModeRequest {
    mode: String,
}

pub async fn set_review_mode(
    State(state): State<AppState>,
    Json(req): Json<SetReviewModeRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let valid = ["manual", "auto_review", "auto_learn"];
    if !valid.contains(&req.mode.as_str()) {
        return Err((StatusCode::BAD_REQUEST, format!("mode must be one of: {:?}", valid)));
    }
    *state.review_mode.write().unwrap() = req.mode.clone();
    Ok(Json(serde_json::json!({"mode": req.mode})))
}

