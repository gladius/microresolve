//! Hebbian association graph endpoints.
//!
//! POST /api/hebbian/bootstrap         — LLM generates L1 graph for a namespace
//! POST /api/hebbian/bootstrap_intent  — LLM generates L3 intent graph
//! GET  /api/hebbian/expand            — debug: show L1 preprocessing result for a query
//! GET  /api/hebbian/score             — debug: show L3 spreading activation scores
//! POST /api/hebbian/reinforce         — manually strengthen an L1 edge
//! GET  /api/hebbian                   — L1 graph stats
//! GET  /api/hebbian/intent_graph      — L3 graph stats

use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json,
};
use std::collections::HashMap;
use asv_router::hebbian::{HebbianGraph, EdgeKind, IntentGraph, ConjunctionRule};
use crate::state::*;
use crate::llm::{call_llm, extract_json};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/hebbian/bootstrap",        post(bootstrap))
        .route("/api/hebbian/bootstrap_intent", post(bootstrap_intent))
        .route("/api/hebbian/expand",           get(expand_query))
        .route("/api/hebbian/score",            get(score_query))
        .route("/api/hebbian/reinforce",        post(reinforce_edge))
        .route("/api/hebbian",                  get(get_graph))
        .route("/api/hebbian/intent_graph",     get(get_intent_graph))
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct BootstrapRequest {
    pub synonym_threshold: Option<f32>,
}

pub async fn bootstrap(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<BootstrapRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    let intent_ids: Vec<String> = {
        let routers = state.routers.read().unwrap();
        routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Namespace '{}' not found", app_id)))?
            .intent_ids()
    };

    let intents_str = intent_ids.join(", ");
    let context = format!("Intents: {}", intents_str);

    eprintln!("[hebbian/bootstrap] {} — generating graph for {} intents", app_id, intent_ids.len());

    let prompt = format!(
r#"Generate a Hebbian word association graph for query preprocessing in an intent routing system.

{}

Your task: for each important domain term, list its word associations.

Association kinds:
- "morphological": inflected form of same word (weight 0.97-0.99). Examples: "canceling"→"cancel", "shipped"→"ship", "creating"→"create"
- "abbreviation": short form → full form (weight 0.97-0.99). Examples: "pr"→"pull request", "repo"→"repository", "sub"→"subscription"
- "synonym": different word, same domain meaning (weight 0.80-0.95). Examples: "terminate"→"cancel", "ping"→"send", "spin up"→"create"
- "semantic": related but context-dependent — for confidence signal only, NOT query expansion (weight 0.60-0.79)

Rules:
1. Cover all key domain verbs and their morphological variants (ing/ed/s/ion forms)
2. Cover common abbreviations users will type
3. For synonyms: only include words that UNAMBIGUOUSLY mean the target in this domain. Avoid short words that appear in compound terms (e.g. "pull" appears in "pull request" — exclude it as a synonym)
4. Semantic edges are for soft signals only — use sparingly
5. Each edge is directional: informal/variant → canonical
6. Include negation word mappings for any non-English languages this domain supports.
   Map foreign negation words to their English equivalent ("not", "never", "without").
   This enables multilingual negation detection via the existing pipeline.
   Examples: {{"from": "nicht", "to": "not", "weight": 0.99, "kind": "morphological"}},
             {{"from": "jamais", "to": "never", "weight": 0.99, "kind": "morphological"}},
             {{"from": "nunca", "to": "never", "weight": 0.99, "kind": "morphological"}}
   Only include if the domain actually serves those languages.

Output a JSON array only, no explanation:
[
  {{"from": "canceling", "to": "cancel", "weight": 0.99, "kind": "morphological"}},
  {{"from": "terminate", "to": "cancel", "weight": 0.92, "kind": "synonym"}},
  {{"from": "pr", "to": "pull request", "weight": 0.99, "kind": "abbreviation"}},
  ...
]

Generate 60-120 edges covering morphology, abbreviations, synonyms, and multilingual negation for this domain."#,
        context
    );

    // 60-120 edges × ~20 tokens each = 2400+ tokens; use 6000 to be safe
    let raw = call_llm(&state, &prompt, 6000).await
        .map_err(|(_, e)| (StatusCode::BAD_GATEWAY, e))?;

    let json_str = extract_json(&raw);
    // If truncated (no closing ']'), try to recover by closing the array after the last '},'
    let recovered;
    let json_str = if json_str.trim_start().starts_with('[') && !json_str.trim_end().ends_with(']') {
        // Find last complete object
        if let Some(last_obj_end) = json_str.rfind('}') {
            recovered = format!("{}}}\n]", &json_str[..last_obj_end]);
            recovered.as_str()
        } else {
            json_str
        }
    } else {
        json_str
    };
    let edges: Vec<serde_json::Value> = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY,
            format!("LLM parse error: {} — extracted: {:?}",
                e, &json_str[..json_str.len().min(200)])))?;

    let mut graph = HebbianGraph::new();
    if let Some(t) = req.synonym_threshold {
        graph.synonym_threshold = t;
    }

    let mut counts = HashMap::<String, usize>::new();
    let mut skipped = 0usize;

    for edge in &edges {
        let from = edge["from"].as_str().unwrap_or("").trim().to_lowercase();
        let to   = edge["to"].as_str().unwrap_or("").trim().to_lowercase();
        let weight = edge["weight"].as_f64().unwrap_or(0.0) as f32;
        let kind_str = edge["kind"].as_str().unwrap_or("");

        if from.is_empty() || to.is_empty() || weight <= 0.0 {
            skipped += 1;
            continue;
        }

        let kind = match kind_str {
            "morphological" => EdgeKind::Morphological,
            "abbreviation"  => EdgeKind::Abbreviation,
            "synonym"       => EdgeKind::Synonym,
            "semantic"      => EdgeKind::Semantic,
            _ => { skipped += 1; continue; }
        };

        graph.add(&from, &to, weight, kind);
        *counts.entry(kind_str.to_string()).or_default() += 1;
    }

    eprintln!("[hebbian/bootstrap] {} — {} edges ({:?}), {} skipped",
        app_id, edges.len() - skipped, counts, skipped);

    // Persist
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}/_hebbian.json", dir, app_id);
        std::fs::create_dir_all(format!("{}/{}", dir, app_id)).ok();
        if let Err(e) = graph.save(&path) {
            eprintln!("[hebbian/bootstrap] save error: {}", e);
        }
    }

    let total = edges.len() - skipped;
    state.hebbian.write().unwrap().insert(app_id, graph);

    Ok(Json(serde_json::json!({
        "edges": total,
        "morphological": counts.get("morphological").copied().unwrap_or(0),
        "abbreviation":  counts.get("abbreviation").copied().unwrap_or(0),
        "synonym":       counts.get("synonym").copied().unwrap_or(0),
        "semantic":      counts.get("semantic").copied().unwrap_or(0),
        "skipped": skipped,
    })))
}

// ── Debug: expand a query ─────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct ExpandParams {
    pub query: String,
}

pub async fn expand_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<ExpandParams>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let hebbian = state.hebbian.read().unwrap();
    let Some(graph) = hebbian.get(&app_id) else {
        return Json(serde_json::json!({"error": "no hebbian graph — call /api/hebbian/bootstrap first"}));
    };

    let result = graph.preprocess(&params.query);
    Json(serde_json::json!({
        "original":      result.original,
        "normalized":    result.normalized,
        "expanded":      result.expanded,
        "injected":      result.injected,
        "semantic_hits": result.semantic_hits.iter()
            .map(|(s, t, w)| serde_json::json!({"from": s, "to": t, "weight": w}))
            .collect::<Vec<_>>(),
        "was_modified":  result.was_modified,
    }))
}

// ── Inspect graph ─────────────────────────────────────────────────────────────

pub async fn get_graph(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let hebbian = state.hebbian.read().unwrap();
    let Some(graph) = hebbian.get(&app_id) else {
        return Json(serde_json::json!({"error": "no hebbian graph"}));
    };

    let mut morph = 0usize; let mut abbrev = 0usize;
    let mut synonym = 0usize; let mut semantic = 0usize;
    for edges in graph.edges.values() {
        for e in edges {
            match e.kind {
                EdgeKind::Morphological => morph += 1,
                EdgeKind::Abbreviation  => abbrev += 1,
                EdgeKind::Synonym       => synonym += 1,
                EdgeKind::Semantic      => semantic += 1,
            }
        }
    }

    Json(serde_json::json!({
        "source_terms": graph.edges.len(),
        "edges": { "morphological": morph, "abbreviation": abbrev,
                   "synonym": synonym, "semantic": semantic,
                   "total": morph + abbrev + synonym + semantic },
        "synonym_threshold": graph.synonym_threshold,
    }))
}

// ── Manual reinforce ──────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct ReinforceRequest {
    pub from: String,
    pub to:   String,
    pub delta: Option<f32>,
}

pub async fn reinforce_edge(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ReinforceRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let delta = req.delta.unwrap_or(0.03);
    {
        let mut hebbian = state.hebbian.write().unwrap();
        let Some(graph) = hebbian.get_mut(&app_id) else {
            return Json(serde_json::json!({"error": "no hebbian graph"}));
        };
        graph.reinforce(&req.from, &req.to, delta);
    }

    // Persist
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}/_hebbian.json", dir, app_id);
        let hebbian = state.hebbian.read().unwrap();
        if let Some(g) = hebbian.get(&app_id) {
            g.save(&path).ok();
        }
    }

    Json(serde_json::json!({"reinforced": {"from": req.from, "to": req.to, "delta": delta}}))
}

/// Load a hebbian graph from disk for a namespace. Called at startup.
pub fn load_hebbian(data_dir: &str, namespace: &str) -> Option<HebbianGraph> {
    let path = format!("{}/{}/_hebbian.json", data_dir, namespace);
    HebbianGraph::load(&path).ok()
}

/// Load an intent graph from disk for a namespace. Called at startup.
pub fn load_intent_graph(data_dir: &str, namespace: &str) -> Option<IntentGraph> {
    let path = format!("{}/{}/_intent_graph.json", data_dir, namespace);
    IntentGraph::load(&path).ok()
}

// ── Bootstrap L3 intent graph (count/log-odds model) ─────────────────────
//
// The LLM generates diverse training phrases per intent.
// We tokenize them, build a count table, and derive log-odds weights.
// One phrase = one meaningful discriminative update. No hand-tuned weights.

pub async fn bootstrap_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(_req): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    // Collect intents with descriptions and seed phrases (for LLM context)
    let intent_defs: Vec<(String, String, Vec<String>)> = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Namespace '{}' not found", app_id)))?;
        let mut ids = router.intent_ids();
        ids.sort();
        ids.into_iter().map(|id| {
            let desc = router.get_description(&id).to_string();
            let seeds: Vec<String> = router.get_training(&id).unwrap_or_default()
                .into_iter().take(5).collect();
            (id, desc, seeds)
        }).collect()
    };

    let n_intents = intent_defs.len();
    eprintln!("[bootstrap_intent] {} — generating phrase corpus for {} intents", app_id, n_intents);

    // Build intent list for the prompt
    let intent_list = intent_defs.iter().map(|(id, desc, seeds)| {
        let desc_str = if desc.is_empty() { String::new() } else { format!(" ({})", desc) };
        let seeds_str = if seeds.is_empty() { String::new() }
            else { format!("\n  existing phrases: {}", seeds.join(" | ")) };
        format!("- {}{}{}", id, desc_str, seeds_str)
    }).collect::<Vec<_>>().join("\n");

    let prompt = format!(
r#"You are building training data for an intent classification system.

For each intent below, generate 20 diverse, realistic phrases a user might say to express that intent.
Include: short queries, long sentences, informal language, indirect phrasing, different vocabulary.
Do NOT repeat the existing phrases — generate NEW ones that cover different vocabulary.

Intents:
{intent_list}

Return ONLY a JSON object mapping each intent ID to an array of 20 phrase strings:
{{
  "intent_id_1": ["phrase 1", "phrase 2", ...],
  "intent_id_2": ["phrase 1", "phrase 2", ...],
  ...
}}

Rules:
- Use exact intent IDs as keys
- All 20 phrases per intent must be distinct
- Vary vocabulary, style, and length across phrases
- Return ONLY the JSON object, no explanation"#
    );

    let raw = call_llm(&state, &prompt, 16000).await
        .map_err(|(_, e)| (StatusCode::BAD_GATEWAY, e))?;

    let json_str = extract_json(&raw);
    let phrase_map: std::collections::HashMap<String, Vec<String>> = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY,
            format!("LLM parse error: {} — raw[..200]: {:?}", e, &raw[..raw.len().min(200)])))?;

    // Start from an empty count model (full rebuild from LLM corpus + existing seeds)
    let mut graph = asv_router::hebbian::IntentGraph::new();

    // Also include the existing seed phrases from the router
    let all_phrases: Vec<(String, Vec<String>)> = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id).unwrap();
        let mut ids = router.intent_ids();
        ids.sort();
        ids.into_iter().map(|id| {
            let mut phrases: Vec<String> = router.get_training(&id).unwrap_or_default();
            if let Some(llm_phrases) = phrase_map.get(&id) {
                phrases.extend(llm_phrases.iter().cloned());
            }
            (id, phrases)
        }).collect()
    };

    // L1 preprocessing for tokenization consistency
    let l1_graph = {
        state.hebbian.read().unwrap()
            .get(&app_id).cloned()
            .unwrap_or_default()
    };

    let mut total_phrases = 0usize;
    let mut total_words = 0usize;

    for (intent_id, phrases) in &all_phrases {
        for phrase in phrases {
            // L1 preprocess then tokenize — same pipeline as routing
            let preprocessed = l1_graph.preprocess(phrase);
            let words = asv_router::tokenizer::tokenize(&preprocessed.expanded);
            let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
            if !word_refs.is_empty() {
                graph.learn_phrase(&word_refs, intent_id);
                total_words += word_refs.len();
                total_phrases += 1;
            }
        }
    }

    eprintln!("[bootstrap_intent] {} — {} intents, {} phrases, {} word tokens, {} unique words",
        app_id, n_intents, total_phrases, total_words, graph.counts.len());

    // Persist
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}/_intent_graph.json", dir, app_id);
        std::fs::create_dir_all(format!("{}/{}", dir, app_id)).ok();
        if let Err(e) = graph.save(&path) {
            eprintln!("[bootstrap_intent] save error: {}", e);
        }
    }

    let (unique_words, _) = graph.count_stats();
    state.intent_graph.write().unwrap().insert(app_id.clone(), graph);

    Ok(Json(serde_json::json!({
        "model": "count_log_odds",
        "intents": n_intents,
        "phrases": total_phrases,
        "word_tokens": total_words,
        "unique_words": unique_words,
        "threshold": 1.0,
        "gap": 5.0,
    })))
}

// ── Debug: score a query via L3 ──────────────────────────────────────────────

pub async fn score_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<ExpandParams>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);

    let layer1_preprocessed = {
        let hebbian = state.hebbian.read().unwrap();
        hebbian.get(&app_id).map(|g| g.preprocess(&params.query))
    };

    let normalized = layer1_preprocessed.as_ref()
        .map(|p| p.expanded.clone())
        .unwrap_or_else(|| params.query.clone());

    let intent_graph = state.intent_graph.read().unwrap();
    let Some(ig) = intent_graph.get(&app_id) else {
        return Json(serde_json::json!({"error": "no intent graph — call /api/hebbian/bootstrap_intent first"}));
    };

    let (scores, has_negation) = ig.score_normalized(&normalized);

    Json(serde_json::json!({
        "query": params.query,
        "normalized": normalized,
        "has_negation": has_negation,
        "scores": scores.iter().map(|(id, s)| serde_json::json!({"intent": id, "score": s})).collect::<Vec<_>>(),
    }))
}

// ── Inspect L3 intent graph ───────────────────────────────────────────────────

pub async fn get_intent_graph(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let intent_graph = state.intent_graph.read().unwrap();
    let Some(ig) = intent_graph.get(&app_id) else {
        return Json(serde_json::json!({"error": "no intent graph"}));
    };
    let (words, activations, suppressors, conjunctions) = ig.stats();
    Json(serde_json::json!({
        "vocabulary_size": words,
        "activation_edges": activations,
        "suppressor_edges": suppressors,
        "conjunctions": conjunctions,
    }))
}
