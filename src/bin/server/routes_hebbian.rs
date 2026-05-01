//! Hebbian graph persistence utilities + L1 CRUD API endpoints.

use crate::state::*;
use axum::{
    extract::State,
    http::HeaderMap,
    routing::{delete, get, post},
    Json,
};
use microresolve::scoring::EdgeKind;

// ── Routes ────────────────────────────────────────────────────────────────────

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/layers/info", get(layers_info))
        .route("/api/layers/l0", get(l0_info))
        .route("/api/layers/l0/correct", post(l0_correct))
        .route("/api/layers/l1/edges", get(l1_list_edges))
        .route("/api/layers/l1/edges", post(l1_add_edge))
        .route("/api/layers/l1/edges", delete(l1_delete_edge))
        .route("/api/layers/l1/distill", post(l1_distill))
        .route("/api/layers/l2/probe", post(l2_probe))
}

// ── GET /api/layers/l0 ────────────────────────────────────────────────────────

async fn l0_info(State(state): State<AppState>, headers: HeaderMap) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let Some(h) = state.engine.try_namespace(&app_id) else {
        return Json(serde_json::json!({ "error": "namespace not found" }));
    };
    h.with_resolver(|router| {
        let l0 = router.l0();
        Json(serde_json::json!({
            "namespace": app_id,
            "vocab_size": l0.len(),
            "ngram_size": l0.ngram_size(),
            "min_term_len": l0.min_term_len(),
            "vocab_sample": l0.vocab_sample(200),
        }))
    })
}

// ── POST /api/layers/l0/correct ───────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct L0CorrectRequest {
    query: String,
}

async fn l0_correct(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<L0CorrectRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let Some(h) = state.engine.try_namespace(&app_id) else {
        return Json(serde_json::json!({ "error": "namespace not found" }));
    };
    h.with_resolver(|router| {
        let corrected = router.l0().correct_query(&req.query);
        Json(serde_json::json!({
            "query": req.query,
            "corrected": corrected,
            "changed": corrected != req.query,
        }))
    })
}

// ── GET /api/layers/info ──────────────────────────────────────────────────────

async fn layers_info(State(state): State<AppState>, headers: HeaderMap) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let Some(h) = state.engine.try_namespace(&app_id) else {
        return Json(serde_json::json!({ "error": "namespace not found" }));
    };

    h.with_resolver(|router| {
        let l1 = router.l1();
        let l2 = router.l2();
        let l0_vocab = router.l0().len();
        let edge_counts = l1.edges.values().fold((0usize, 0, 0), |mut acc, edges| {
            for e in edges {
                match e.kind {
                    EdgeKind::Morphological => acc.0 += 1,
                    EdgeKind::Abbreviation => acc.1 += 1,
                    EdgeKind::Synonym => acc.2 += 1,
                    _ => {}
                }
            }
            acc
        });
        Json(serde_json::json!({
            "l0": { "vocab_size": l0_vocab as usize },
            "l1": {
                "terms": l1.edges.len(),
                "edges_morphological": edge_counts.0,
                "edges_abbreviation":  edge_counts.1,
                "edges_synonym":       edge_counts.2,
                "edges_total": edge_counts.0 + edge_counts.1 + edge_counts.2,
            },
            "l2": {
                "words": l2.word_intent.len(),
                "intents": router.intent_ids().len(),
            },
        }))
    })
}

// ── GET /api/layers/l1/edges ──────────────────────────────────────────────────

async fn l1_list_edges(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let Some(h) = state.engine.try_namespace(&app_id) else {
        return Json(serde_json::json!({ "edges": [], "total": 0 }));
    };
    let filter = params
        .get("q")
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    let kind_filter = params
        .get("kind")
        .map(|s| s.as_str())
        .unwrap_or("all")
        .to_string();
    let limit: usize = params
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    h.with_resolver(|router| {
        let all_edges: Vec<serde_json::Value> = router.l1().edges.iter()
            .flat_map(|(from, targets)| targets.iter().map(move |e| {
                let kind = match e.kind {
                    EdgeKind::Morphological => "morphological",
                    EdgeKind::Abbreviation  => "abbreviation",
                    EdgeKind::Synonym       => "synonym",
                    _                       => "semantic",
                };
                (from.clone(), e.target.clone(), kind, e.weight)
            }))
            .filter(|(from, to, kind, _)| {
                if kind_filter != "all" && *kind != kind_filter.as_str() { return false; }
                if !filter.is_empty() && !from.contains(filter.as_str()) && !to.contains(filter.as_str()) { return false; }
                true
            })
            .take(limit)
            .map(|(from, to, kind, weight)| serde_json::json!({ "from": from, "to": to, "kind": kind, "weight": weight }))
            .collect();
        Json(serde_json::json!({ "edges": all_edges, "total": router.l1().edges.len() }))
    })
}

// ── POST /api/layers/l1/edges ─────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct AddEdgeRequest {
    from: String,
    to: String,
    kind: String,
    weight: Option<f32>,
}

async fn l1_add_edge(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddEdgeRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let from = req.from.trim().to_lowercase();
    let to = req.to.trim().to_lowercase();
    if from.is_empty() || to.is_empty() || from == to {
        return Json(serde_json::json!({ "ok": false, "error": "invalid edge" }));
    }
    let kind = match req.kind.as_str() {
        "morphological" => EdgeKind::Morphological,
        "abbreviation" => EdgeKind::Abbreviation,
        _ => EdgeKind::Synonym,
    };
    let weight = req.weight.unwrap_or(match kind {
        EdgeKind::Morphological => 0.98,
        EdgeKind::Abbreviation => 0.99,
        _ => 0.88,
    });

    let Some(h) = state.engine.try_namespace(&app_id) else {
        return Json(serde_json::json!({ "ok": false, "error": "namespace not found" }));
    };
    h.with_resolver_mut(|r| r.l1_mut().add(&from, &to, weight, kind));
    h.flush().ok();
    Json(serde_json::json!({ "ok": true }))
}

// ── DELETE /api/layers/l1/edges ───────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct DeleteEdgeRequest {
    from: String,
    to: String,
}

async fn l1_delete_edge(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DeleteEdgeRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let from = req.from.trim().to_lowercase();
    let to = req.to.trim().to_lowercase();

    let Some(h) = state.engine.try_namespace(&app_id) else {
        return Json(serde_json::json!({ "ok": false }));
    };
    h.with_resolver_mut(|r| {
        if let Some(edges) = r.l1_mut().edges.get_mut(&from) {
            edges.retain(|e| e.target != to);
            if edges.is_empty() {
                r.l1_mut().edges.remove(&from);
            }
        }
    });
    h.flush().ok();
    Json(serde_json::json!({ "ok": true }))
}

// ── POST /api/layers/l1/distill ───────────────────────────────────────────────

async fn l1_distill(State(state): State<AppState>, headers: HeaderMap) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    // Collect all accepted phrases from this namespace
    let Some(h) = state.engine.try_namespace(&app_id) else {
        return Json(serde_json::json!({ "ok": false, "error": "namespace not found" }));
    };
    let accepted: Vec<(String, String)> = h.with_resolver(|router| {
        router
            .intent_ids()
            .into_iter()
            .flat_map(|id| {
                let phrases = router.training(&id).unwrap_or_default();
                phrases.into_iter().map(move |p| (id.clone(), p))
            })
            .collect()
    });

    if accepted.is_empty() {
        return Json(serde_json::json!({ "ok": false, "error": "no training phrases" }));
    }

    // Reuse the same LLM distillation pipeline from import
    crate::routes_import::seed_into_l1_pub(&state, &app_id, &accepted).await;

    let edge_total: usize = state
        .engine
        .try_namespace(&app_id)
        .map(|h| h.with_resolver(|r| r.l1().edges.values().map(|v| v.len()).sum()))
        .unwrap_or(0);
    Json(serde_json::json!({ "ok": true, "edges_total": edge_total }))
}

// ── POST /api/layers/l2/probe ─────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct L2ProbeRequest {
    query: String,
}

async fn l2_probe(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<L2ProbeRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let Some(h) = state.engine.try_namespace(&app_id) else {
        return Json(serde_json::json!({ "tokens": [], "scores": [] }));
    };

    h.with_resolver(|router| {
        let q0 = router.l0().correct_query(&req.query);
        let preprocessed = router.l1().preprocess(&q0);
        let tokens: Vec<String> = microresolve::tokenizer::tokenize(&preprocessed.expanded);
        let (scores, _) = router.l2().score_normalized(&preprocessed.expanded);
        const DEFAULT_THRESHOLD: f32 = 0.3;
        let (final_intents, has_negation, trace) = router.l2().score_multi_normalized_traced(
            &preprocessed.expanded, DEFAULT_THRESHOLD, 0.0, true,
        );
        Json(serde_json::json!({
            "l0_corrected": q0,
            "l1_normalized": preprocessed.normalized,
            "l1_expanded": preprocessed.expanded,
            "l1_injected": preprocessed.injected,
            "tokens": tokens,
            "scores": scores.iter().take(10).map(|(id, s)| serde_json::json!({"id": id, "score": s})).collect::<Vec<_>>(),
            "multi": {
                "rounds": trace.as_ref().map(|t| &t.rounds),
                "stop_reason": trace.as_ref().map(|t| t.stop_reason.clone()),
                "final_intents": final_intents,
                "has_negation": has_negation,
            },
        }))
    })
}
