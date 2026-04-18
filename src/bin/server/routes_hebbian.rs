//! Hebbian graph persistence utilities + L1 CRUD API endpoints.

use asv_router::scoring::{LexicalGraph, EdgeKind};
use axum::{extract::State, http::HeaderMap, routing::{get, post, delete}, Json};
use crate::state::*;

// ── Startup utility ───────────────────────────────────────────────────────────

/// Load the global L1 base graph (data/l1_base.json).
pub fn load_l1_base(data_dir: &str) -> Option<LexicalGraph> {
    let path = format!("{}/l1_base.json", data_dir);
    match LexicalGraph::load(&path) {
        Ok(g) => {
            let edge_count: usize = g.edges.values().map(|v| v.len()).sum();
            println!("Loaded L1 base graph: {} terms, {} edges", g.edges.len(), edge_count);
            Some(g)
        }
        Err(e) => {
            println!("L1 base graph not found at {} — run scripts/generate_l1_base.py", path);
            None
        }
    }
}

// ── Routes ────────────────────────────────────────────────────────────────────

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/layers/info",       get(layers_info))
        .route("/api/layers/l1/edges",   get(l1_list_edges))
        .route("/api/layers/l1/edges",   post(l1_add_edge))
        .route("/api/layers/l1/edges",   delete(l1_delete_edge))
        .route("/api/layers/l1/distill", post(l1_distill))
        .route("/api/layers/l2/probe",   post(l2_probe))
}

// ── GET /api/layers/info ──────────────────────────────────────────────────────

async fn layers_info(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let Some(router) = routers.get(&app_id) else {
        return Json(serde_json::json!({ "error": "namespace not found" }));
    };
    let l1 = router.l1();
    let l2 = router.l2();
    let l0_vocab = router.l0().len();

    let edge_counts = l1.edges.values().fold((0usize, 0, 0), |mut acc, edges| {
        for e in edges {
            match e.kind {
                EdgeKind::Morphological => acc.0 += 1,
                EdgeKind::Abbreviation  => acc.1 += 1,
                EdgeKind::Synonym       => acc.2 += 1,
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
}

// ── GET /api/layers/l1/edges ──────────────────────────────────────────────────

async fn l1_list_edges(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let Some(router) = routers.get(&app_id) else {
        return Json(serde_json::json!({ "edges": [] }));
    };
    let edges: Vec<serde_json::Value> = router.l1().edges.iter()
        .flat_map(|(from, targets)| targets.iter().map(move |e| {
            let kind = match e.kind {
                EdgeKind::Morphological => "morphological",
                EdgeKind::Abbreviation  => "abbreviation",
                EdgeKind::Synonym       => "synonym",
                _                       => "semantic",
            };
            serde_json::json!({ "from": from, "to": e.target, "kind": kind, "weight": e.weight })
        }))
        .collect();
    Json(serde_json::json!({ "edges": edges }))
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
    let to   = req.to.trim().to_lowercase();
    if from.is_empty() || to.is_empty() || from == to {
        return Json(serde_json::json!({ "ok": false, "error": "invalid edge" }));
    }
    let kind = match req.kind.as_str() {
        "morphological" => EdgeKind::Morphological,
        "abbreviation"  => EdgeKind::Abbreviation,
        _               => EdgeKind::Synonym,
    };
    let weight = req.weight.unwrap_or(match kind {
        EdgeKind::Morphological => 0.98,
        EdgeKind::Abbreviation  => 0.99,
        _                       => 0.88,
    });

    let mut routers = state.routers.write().unwrap();
    let Some(router) = routers.get_mut(&app_id) else {
        return Json(serde_json::json!({ "ok": false, "error": "namespace not found" }));
    };
    router.l1_mut().add(&from, &to, weight, kind);

    if let Some(ref dir) = state.data_dir {
        let ns_dir = std::path::Path::new(dir).join(&app_id);
        router.save_to_dir(&ns_dir).ok();
    }
    Json(serde_json::json!({ "ok": true }))
}

// ── DELETE /api/layers/l1/edges ───────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct DeleteEdgeRequest { from: String, to: String }

async fn l1_delete_edge(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DeleteEdgeRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let from = req.from.trim().to_lowercase();
    let to   = req.to.trim().to_lowercase();

    let mut routers = state.routers.write().unwrap();
    let Some(router) = routers.get_mut(&app_id) else {
        return Json(serde_json::json!({ "ok": false }));
    };
    if let Some(edges) = router.l1_mut().edges.get_mut(&from) {
        edges.retain(|e| e.target != to);
        if edges.is_empty() { router.l1_mut().edges.remove(&from); }
    }
    if let Some(ref dir) = state.data_dir {
        let ns_dir = std::path::Path::new(dir).join(&app_id);
        router.save_to_dir(&ns_dir).ok();
    }
    Json(serde_json::json!({ "ok": true }))
}

// ── POST /api/layers/l1/distill ───────────────────────────────────────────────

async fn l1_distill(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    // Collect all accepted phrases from this namespace
    let accepted: Vec<(String, String)> = {
        let routers = state.routers.read().unwrap();
        let Some(router) = routers.get(&app_id) else {
            return Json(serde_json::json!({ "ok": false, "error": "namespace not found" }));
        };
        router.intent_ids().into_iter()
            .flat_map(|id| {
                let phrases = router.get_training(&id).unwrap_or_default();
                phrases.into_iter().map(move |p| (id.clone(), p))
            })
            .collect()
    };

    if accepted.is_empty() {
        return Json(serde_json::json!({ "ok": false, "error": "no training phrases" }));
    }

    // Reuse the same LLM distillation pipeline from import
    crate::routes_import::seed_into_l1_pub(&state, &app_id, &accepted).await;

    let edge_total: usize = {
        let routers = state.routers.read().unwrap();
        routers.get(&app_id)
            .map(|r| r.l1().edges.values().map(|v| v.len()).sum())
            .unwrap_or(0)
    };
    Json(serde_json::json!({ "ok": true, "edges_total": edge_total }))
}

// ── POST /api/layers/l2/probe ─────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct L2ProbeRequest { query: String }

async fn l2_probe(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<L2ProbeRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let routers = state.routers.read().unwrap();
    let Some(router) = routers.get(&app_id) else {
        return Json(serde_json::json!({ "tokens": [], "scores": [] }));
    };
    let q0 = router.l0().correct_query(&req.query);
    let preprocessed = router.l1().preprocess(&q0);
    let tokens: Vec<String> = asv_router::tokenizer::tokenize(&preprocessed.expanded);
    let (scores, _) = router.l2().score_normalized(&preprocessed.expanded);
    Json(serde_json::json!({
        "l0_corrected": q0,
        "l1_normalized": preprocessed.normalized,
        "l1_expanded": preprocessed.expanded,
        "l1_injected": preprocessed.injected,
        "tokens": tokens,
        "scores": scores.iter().take(10).map(|(id, s)| serde_json::json!({"id": id, "score": s})).collect::<Vec<_>>(),
    }))
}
