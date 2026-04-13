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

    // Collect intent IDs + concept signals for context
    let intent_ids: Vec<String> = {
        let routers = state.routers.read().unwrap();
        routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Namespace '{}' not found", app_id)))?
            .intent_ids()
    };

    let concept_signals: String = {
        let concepts = state.concepts.read().unwrap();
        if let Some(reg) = concepts.get(&app_id) {
            reg.concepts.iter()
                .map(|(name, sigs)| format!("  {}: {}", name, sigs[..sigs.len().min(6)].join(", ")))
                .collect::<Vec<_>>().join("\n")
        } else {
            String::new()
        }
    };

    let intents_str = intent_ids.join(", ");
    let context = if concept_signals.is_empty() {
        format!("Intents: {}", intents_str)
    } else {
        format!("Intents: {}\n\nConcept signals (sample):\n{}", intents_str, concept_signals)
    };

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

// ── Bootstrap L3 intent graph ─────────────────────────────────────────────

pub async fn bootstrap_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(_req): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    // Collect intent IDs
    let intent_ids: Vec<String> = {
        let routers = state.routers.read().unwrap();
        routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Namespace '{}' not found", app_id)))?
            .intent_ids()
    };

    // Collect L1 Hebbian edges summary (so LLM knows which synonyms are already handled)
    let l1_summary: String = {
        let hebbian = state.hebbian.read().unwrap();
        if let Some(g) = hebbian.get(&app_id) {
            let synonyms: Vec<String> = g.edges.iter()
                .flat_map(|(from, edges)| edges.iter().map(move |e| format!("{}→{}", from, e.target)))
                .take(40)
                .collect();
            format!("Layer 1 already maps: {}", synonyms.join(", "))
        } else {
            "No Layer 1 graph — include morphological variants in activate words.".to_string()
        }
    };

    let intents_str = intent_ids.join(", ");
    eprintln!("[hebbian/bootstrap_intent] {} — generating L2 graph for {} intents", app_id, intent_ids.len());

    let prompt = format!(
r#"Generate a Layer 2 Hebbian intent graph for spreading activation intent routing.

Intents: {intents_str}

Context: {l1_summary}

Rules:
1. Use CANONICAL terms only in activate/suppress/conjunction words — Layer 1 already handles morphological variants and synonyms
2. "activate" words: content words that indicate this intent. Weight 0.7-1.0 for highly discriminative words, 0.3-0.6 for weak signals
3. "suppress" words: words that appear with SIMILAR intents but NOT this one — used for disambiguation. Only add if genuinely confusing
4. "conjunctions": pairs of words that TOGETHER strongly confirm this intent (bonus on top of individual activations). Use sparingly — only the most reliable pairs
5. Cover every intent listed above

Output a JSON array only, no explanation:
[
  {{
    "intent": "cancel_subscription",
    "activate": [
      {{"word": "cancel", "weight": 0.90}},
      {{"word": "subscription", "weight": 0.85}},
      {{"word": "plan", "weight": 0.55}}
    ],
    "suppress": [
      {{"word": "order", "weight": 0.50}}
    ],
    "conjunctions": [
      {{"words": ["cancel", "subscription"], "bonus": 0.50}}
    ]
  }},
  ...
]"#
    );

    let raw = call_llm(&state, &prompt, 8000).await
        .map_err(|(_, e)| (StatusCode::BAD_GATEWAY, e))?;

    let json_str = extract_json(&raw);
    let entries: Vec<serde_json::Value> = serde_json::from_str(json_str)
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY,
            format!("LLM parse error: {} — raw[..200]: {:?}", e, &raw[..raw.len().min(200)])))?;

    // Merge into existing graph rather than replacing it.
    // add_activation / add_suppressor both use max(), so any edge that was
    // reinforced above the bootstrap weight is preserved. New words from the
    // LLM are added on top. Conjunctions are replaced (no live learning there).
    let mut graph = {
        state.intent_graph.read().unwrap()
            .get(&app_id).cloned()
            .unwrap_or_default()
    };
    // Clear only conjunctions — these come entirely from the bootstrap prompt
    // and there's no live conjunction learning to preserve.
    graph.conjunctions.clear();

    let mut activation_count = 0usize;
    let mut suppressor_count = 0usize;
    let mut conjunction_count = 0usize;

    for entry in &entries {
        let intent = match entry["intent"].as_str() {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => continue,
        };

        if let Some(activations) = entry["activate"].as_array() {
            for a in activations {
                let word = a["word"].as_str().unwrap_or("").trim().to_lowercase();
                let weight = a["weight"].as_f64().unwrap_or(0.0) as f32;
                if !word.is_empty() && weight > 0.0 {
                    graph.add_activation(&word, &intent, weight);
                    activation_count += 1;
                }
            }
        }

        if let Some(suppressors) = entry["suppress"].as_array() {
            for s in suppressors {
                let word = s["word"].as_str().unwrap_or("").trim().to_lowercase();
                let weight = s["weight"].as_f64().unwrap_or(0.0) as f32;
                if !word.is_empty() && weight > 0.0 {
                    graph.add_suppressor(&word, &intent, weight);
                    suppressor_count += 1;
                }
            }
        }

        if let Some(conjunctions) = entry["conjunctions"].as_array() {
            for c in conjunctions {
                let words: Vec<String> = c["words"].as_array()
                    .map(|arr| arr.iter()
                        .filter_map(|w| w.as_str())
                        .map(|w| w.trim().to_lowercase())
                        .filter(|w| !w.is_empty())
                        .collect())
                    .unwrap_or_default();
                let bonus = c["bonus"].as_f64().unwrap_or(0.0) as f32;
                if words.len() >= 2 && bonus > 0.0 {
                    graph.conjunctions.push(ConjunctionRule { words, intent: intent.clone(), bonus });
                    conjunction_count += 1;
                }
            }
        }
    }

    eprintln!("[hebbian/bootstrap_intent] {} — {} activations, {} suppressors, {} conjunctions",
        app_id, activation_count, suppressor_count, conjunction_count);

    // Persist
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}/_intent_graph.json", dir, app_id);
        std::fs::create_dir_all(format!("{}/{}", dir, app_id)).ok();
        if let Err(e) = graph.save(&path) {
            eprintln!("[hebbian/bootstrap_intent] save error: {}", e);
        }
    }

    state.intent_graph.write().unwrap().insert(app_id, graph);

    Ok(Json(serde_json::json!({
        "intents": entries.len(),
        "activation_edges": activation_count,
        "suppressor_edges": suppressor_count,
        "conjunctions": conjunction_count,
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
