//! Concept-signal intent routing — primary routing engine.
//!
//! POST /api/concepts/bootstrap  — LLM generates concept registry (two-phase for large namespaces)
//! GET  /api/concepts             — get current registry
//! POST /api/concepts/signal      — add a signal to a concept (manual learning)
//! POST /api/concepts/explain     — show which signals fired for a query

use axum::{
    extract::State,
    http::HeaderMap,
    routing::{get, post},
    Json,
};
use std::collections::HashMap;
use asv_router::concept::ConceptRegistry;
use crate::state::*;
use crate::llm::{call_llm, extract_json};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/concepts/bootstrap", post(bootstrap))
        .route("/api/concepts",           get(get_registry))
        .route("/api/concepts/signal",    post(add_signal))
        .route("/api/concepts/explain",   post(explain_query))
}

// ── Bootstrap (two-phase for large namespaces) ────────────────────────────────
//
// Phase 1: LLM defines concepts + signals (~20 concepts, small output)
// Phase 2: LLM assigns intent profiles in batches of 25 intents
// Merged into one ConceptRegistry, saved to {data_dir}/{namespace}/_concepts.json

const PROFILE_BATCH_SIZE: usize = 25;

#[derive(serde::Deserialize)]
pub struct BootstrapRequest {
    pub intent_ids: Option<Vec<String>>,
}

#[derive(serde::Serialize)]
pub struct BootstrapResponse {
    pub concepts: usize,
    pub intents_covered: usize,
    pub total_signals: usize,
}

pub async fn bootstrap(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<BootstrapRequest>,
) -> Result<Json<BootstrapResponse>, (axum::http::StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    let intent_list: Vec<(String, String, Vec<String>)> = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (axum::http::StatusCode::NOT_FOUND,
                format!("Namespace '{}' not found", app_id)))?;
        let ids = req.intent_ids.unwrap_or_else(|| router.intent_ids());
        ids.into_iter().map(|id| {
            let desc = router.get_description(&id).to_string();
            let phrases = router.get_training(&id).unwrap_or_default();
            (id, desc, phrases)
        }).collect()
    };

    if intent_list.is_empty() {
        return Err((axum::http::StatusCode::BAD_REQUEST, "No intents in namespace".to_string()));
    }

    eprintln!("[concepts/bootstrap] {} — phase 1: defining concepts for {} intents",
        app_id, intent_list.len());

    // ── Phase 1: define concepts ──────────────────────────────────────────────

    let intent_summary: String = intent_list.iter().map(|(id, desc, phrases)| {
        let samples = phrases.iter().take(3).cloned().collect::<Vec<_>>().join("\", \"");
        if desc.is_empty() {
            format!("  - {}", id)
        } else if samples.is_empty() {
            format!("  - {} | {}", id, desc)
        } else {
            format!("  - {} | {} | e.g. [\"{}\"]", id, desc, samples)
        }
    }).collect::<Vec<_>>().join("\n");

    let phase1_prompt = format!(
        r#"You are building a semantic routing layer for an AI assistant.

Given these intents (id | description | example phrases):
{intent_summary}

Define the CONCEPT layer — named semantic units that cover the meaning space of all intents.

Rules:
1. Define 20-35 concepts in TWO categories that together uniquely identify any intent:

   DOMAIN concepts — what entity or topic the intent is about.
   Derive these from the intents themselves. Each domain concept should cover a distinct subject area.

   ACTION concepts — what operation the user wants to perform.
   These are CRITICAL for distinguishing intents that share the same domain.
   Derive action concepts from the verbs in the intents: create/add, list/fetch/show, update/edit, delete/remove, cancel/stop, deploy/publish, search/find, etc.
   Include all synonyms and verb forms for each action.

   Together, domain + action uniquely route any intent. Example: domain=subscription + action=cancel → cancel_subscription. domain=subscription + action=list → list_subscriptions.

2. Always include:
   - "user_is_requesting": first-person request signals
     signals: ["i want", "i need", "please", "can you", "help me", "i would like", "how do i", "show me", "get me", "fetch", "retrieve", "create", "make", "set up", "add", "remove", "delete", "update", "list", "find"]
   - "technical_context": UI/code discussion (NOT user intent)
     signals: ["button", "the ui", "interface", "on the page", "the menu", "the form", "dropdown", "modal", "element", "component", "the api", "endpoint", "the code", "the screen"]

3. For each concept provide 15-30 signals (words or short phrases):
   - Synonyms, informal variants, morphological variants
   - For action concepts: include all verb forms (e.g. create, creating, created, make, new, add, set up)
   - Lowercase only

Return ONLY valid JSON:
{{
  "concepts": {{
    "concept_name": ["signal1", "signal2", ...],
    "user_is_requesting": ["i want", "i need", ...],
    "technical_context": ["button", "the ui", ...]
  }}
}}"#
    );

    let phase1_response = call_llm(&state, &phase1_prompt, 4096).await?;
    let phase1_json = extract_json(&phase1_response);
    let phase1_parsed: serde_json::Value = serde_json::from_str(phase1_json)
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY,
            format!("Phase 1 invalid JSON: {}. Raw: {}", e, &phase1_response[..phase1_response.len().min(300)])))?;

    let concepts: HashMap<String, Vec<String>> = serde_json::from_value(
        phase1_parsed["concepts"].clone()
    ).map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, format!("Phase 1 bad concepts: {}", e)))?;

    eprintln!("[concepts/bootstrap] {} — phase 1 done: {} concepts defined", app_id, concepts.len());

    // ── Phase 2: assign intent profiles in batches ────────────────────────────

    let concept_names: Vec<&str> = concepts.keys().map(|s| s.as_str()).collect();
    let concept_summary: String = concepts.iter().map(|(name, signals)| {
        let sample = signals.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
        format!("  {}: [{}]", name, sample)
    }).collect::<Vec<_>>().join("\n");

    let mut all_profiles: HashMap<String, HashMap<String, f32>> = HashMap::new();

    for (batch_idx, batch) in intent_list.chunks(PROFILE_BATCH_SIZE).enumerate() {
        eprintln!("[concepts/bootstrap] {} — phase 2 batch {}: {} intents",
            app_id, batch_idx + 1, batch.len());

        let batch_summary: String = batch.iter().map(|(id, desc, phrases)| {
            let samples = phrases.iter().take(3).cloned().collect::<Vec<_>>().join("\", \"");
            if desc.is_empty() {
                format!("  - {}", id)
            } else if samples.is_empty() {
                format!("  - {} | {}", id, desc)
            } else {
                format!("  - {} | {} | e.g. [\"{}\"]", id, desc, samples)
            }
        }).collect::<Vec<_>>().join("\n");

        let concept_list = concept_names.join(", ");

        let phase2_prompt = format!(
            r#"You are assigning concept weights for intent routing.

Available concepts:
{concept_summary}

For each intent below, assign weights (0.0 to 1.0) for the concepts that apply.
Rules:
- 1.0 = this concept strongly signals this intent
- 0.5 = partial signal
- 0.0 = omit (no relation needed)
- Use -1.0 for "technical_context" on all action intents
- Only include non-zero weights

Intents to profile (id | description | example phrases):
{batch_summary}

Return ONLY valid JSON — only the intent_profiles for these intents:
{{
  "intent_profiles": {{
    "intent_id": {{"concept_name": 1.0, "technical_context": -1.0}},
    ...
  }}
}}

Use only these concept names: {concept_list}"#
        );

        let phase2_response = call_llm(&state, &phase2_prompt, 4096).await?;
        let phase2_json = extract_json(&phase2_response);
        let phase2_parsed: serde_json::Value = serde_json::from_str(phase2_json)
            .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY,
                format!("Phase 2 batch {} invalid JSON: {}", batch_idx + 1, e)))?;

        let batch_profiles: HashMap<String, HashMap<String, f32>> = serde_json::from_value(
            phase2_parsed["intent_profiles"].clone()
        ).map_err(|e| (axum::http::StatusCode::BAD_GATEWAY,
            format!("Phase 2 batch {} bad profiles: {}", batch_idx + 1, e)))?;

        all_profiles.extend(batch_profiles);
    }

    let total_signals: usize = concepts.values().map(|s| s.len()).sum();
    let n_concepts = concepts.len();
    let n_intents = all_profiles.len();

    let registry = ConceptRegistry { concepts, intent_profiles: all_profiles };

    // Persist
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}/_concepts.json", dir, app_id);
        if let Err(e) = registry.save(&path) {
            eprintln!("[concepts/bootstrap] save failed: {}", e);
        }
    }

    state.concepts.write().unwrap().insert(app_id.clone(), registry);

    eprintln!("[concepts/bootstrap] {} — done: {} concepts, {} intents, {} signals",
        app_id, n_concepts, n_intents, total_signals);

    Ok(Json(BootstrapResponse { concepts: n_concepts, intents_covered: n_intents, total_signals }))
}

// ── Get registry ──────────────────────────────────────────────────────────────

async fn get_registry(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let concepts = state.concepts.read().unwrap();
    match concepts.get(&app_id) {
        Some(reg) => Json(serde_json::to_value(reg).unwrap_or_default()),
        None => Json(serde_json::json!({ "error": "no concept registry — call /api/concepts/bootstrap first" })),
    }
}

// ── Add signal (manual continuous learning) ───────────────────────────────────

#[derive(serde::Deserialize)]
struct AddSignalRequest {
    concept: String,
    signal: String,
}

async fn add_signal(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddSignalRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let mut concepts = state.concepts.write().unwrap();
    match concepts.get_mut(&app_id) {
        None => Json(serde_json::json!({ "ok": false, "error": "no concept registry" })),
        Some(reg) => {
            reg.add_signal(&req.concept, &req.signal);
            if let Some(ref dir) = state.data_dir {
                let _ = reg.save(&format!("{}/{}/_concepts.json", dir, app_id));
            }
            eprintln!("[concepts/signal] {}: added '{}' to '{}'", app_id, req.signal, req.concept);
            Json(serde_json::json!({
                "ok": true,
                "concept": req.concept,
                "signal": req.signal,
                "total_signals": reg.concepts.get(&req.concept).map(|s| s.len()).unwrap_or(0),
            }))
        }
    }
}

// ── Explain query ─────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct ExplainRequest {
    query: String,
}

async fn explain_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ExplainRequest>,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    let concepts = state.concepts.read().unwrap();
    match concepts.get(&app_id) {
        None => Json(serde_json::json!({ "error": "no concept registry" })),
        Some(reg) => {
            let activations = reg.explain(&req.query);
            let scores = reg.score_query(&req.query);
            Json(serde_json::json!({
                "query": req.query,
                "concepts_fired": activations.iter().map(|a| serde_json::json!({
                    "concept": a.concept,
                    "score": a.score,
                    "matched_signals": a.matched_signals,
                })).collect::<Vec<_>>(),
                "intent_scores": scores.iter().map(|(id, s)| serde_json::json!({
                    "intent": id, "score": s
                })).collect::<Vec<_>>(),
            }))
        }
    }
}

// ── Helpers used by other modules ─────────────────────────────────────────────

/// Load concept registry from disk for a namespace, if it exists.
pub fn load_concepts(data_dir: &str, app_id: &str) -> Option<ConceptRegistry> {
    ConceptRegistry::load(&format!("{}/{}/_concepts.json", data_dir, app_id)).ok()
}

/// Called after LLM confirms the correct intent for a failed query.
/// Asks LLM which specific signal to add to which concept.
/// Falls back to heuristic word extraction if LLM call fails.
pub async fn learn_from_correction(
    state: &AppState,
    app_id: &str,
    query: &str,
    correct_intent: &str,
) {
    // Get current concept names and which concept dominates this intent
    let (concept_names, dominant_concept) = {
        let concepts = state.concepts.read().unwrap();
        let Some(reg) = concepts.get(app_id) else { return };
        let Some(profile) = reg.intent_profiles.get(correct_intent) else { return };
        let dominant = profile.iter()
            .filter(|(_, &w)| w > 0.0)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(c, _)| c.clone());
        let names: Vec<String> = reg.concepts.keys().cloned().collect();
        (names, dominant)
    };

    let Some(dominant) = dominant_concept else { return };

    // Ask LLM which signal to add and to which concept
    let concept_list = concept_names.join(", ");
    let prompt = format!(
        r#"A query was misrouted. The correct intent is "{correct_intent}".

Query: "{query}"

Available concepts: {concept_list}

Which SHORT signal phrase (1-3 words) from this query most uniquely identifies the intent "{correct_intent}"?
Which concept should it be added to?

Return ONLY valid JSON:
{{"concept": "concept_name", "signal": "the signal phrase"}}"#
    );

    let signals_to_add: Vec<(String, String)> = match call_llm(state, &prompt, 128).await {
        Ok(response) => {
            let json_str = extract_json(&response);
            match serde_json::from_str::<serde_json::Value>(json_str) {
                Ok(parsed) => {
                    let concept = parsed["concept"].as_str().unwrap_or(&dominant).to_string();
                    let signal = parsed["signal"].as_str().unwrap_or("").to_string();
                    if !signal.is_empty() && concept_names.contains(&concept) {
                        vec![(concept, signal)]
                    } else {
                        heuristic_signals(state, app_id, query, &dominant)
                    }
                }
                Err(_) => heuristic_signals(state, app_id, query, &dominant),
            }
        }
        Err(_) => heuristic_signals(state, app_id, query, &dominant),
    };

    if signals_to_add.is_empty() { return; }

    let mut concepts = state.concepts.write().unwrap();
    let Some(reg) = concepts.get_mut(app_id) else { return };

    for (concept, signal) in &signals_to_add {
        eprintln!("[concepts/learn] {}/{}: '{}' → concept '{}'", app_id, correct_intent, signal, concept);
        reg.add_signal(concept, signal);
    }

    if let Some(ref dir) = state.data_dir {
        let _ = reg.save(&format!("{}/{}/_concepts.json", dir, app_id));
    }
}

/// Fallback: extract unmatched content words from query as signals.
fn heuristic_signals(state: &AppState, app_id: &str, query: &str, dominant_concept: &str) -> Vec<(String, String)> {
    const STOP: &[&str] = &["a","an","the","i","to","of","for","my","me","is","it",
        "in","on","at","be","do","if","or","and","not","no","we","us","our","can",
        "this","that","with","from","by","as","so","but","up","you","your","have",
        "has","had","will","would","could","should","may","might","am","are","was","were"];

    let concepts = state.concepts.read().unwrap();
    let Some(reg) = concepts.get(app_id) else { return vec![] };
    let activations = reg.explain(query);
    let fired: Vec<String> = activations.iter()
        .flat_map(|a| a.matched_signals.iter().cloned())
        .collect();

    query.to_lowercase().split_whitespace()
        .filter(|w| w.len() > 2 && !STOP.contains(w))
        .filter(|w| !fired.iter().any(|s| s.contains(*w)))
        .take(2)
        .map(|w| (dominant_concept.to_string(), w.to_string()))
        .collect()
}
