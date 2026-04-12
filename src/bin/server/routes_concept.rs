//! Concept-signal intent routing endpoints.
//!
//! POST /api/concepts/bootstrap  — LLM generates concept registry from intent list
//! GET  /api/concepts             — get current registry for namespace
//! POST /api/concepts/route       — route a query using concept registry
//! POST /api/concepts/signal      — add a signal to a concept (continuous learning)
//! GET  /api/concepts/explain     — show which signals fired for a query

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
        .route("/api/concepts/route",     post(route_query))
        .route("/api/concepts/signal",    post(add_signal))
        .route("/api/concepts/explain",   post(explain_query))
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
//
// Calls LLM with the namespace's intent list + descriptions.
// LLM returns a full ConceptRegistry as JSON.
// Saves to {data_dir}/{namespace}/_concepts.json.

#[derive(serde::Deserialize)]
struct BootstrapRequest {
    /// Optional: override which intents to include. Defaults to all intents in namespace.
    intent_ids: Option<Vec<String>>,
}

#[derive(serde::Serialize)]
struct BootstrapResponse {
    concepts: usize,
    intents_covered: usize,
    total_signals: usize,
    registry: serde_json::Value,
}

async fn bootstrap(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<BootstrapRequest>,
) -> Result<Json<BootstrapResponse>, (axum::http::StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);

    // Build intent descriptions from the namespace
    let intent_list: Vec<(String, String, Vec<String>)> = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (axum::http::StatusCode::NOT_FOUND, format!("Namespace '{}' not found", app_id)))?;

        let ids = req.intent_ids.unwrap_or_else(|| router.intent_ids());
        ids.into_iter().map(|id| {
            let desc = router.get_description(&id);
            let phrases = router.get_training(&id).unwrap_or_default();
            (id, desc.to_string(), phrases)
        }).collect()
    };

    if intent_list.is_empty() {
        return Err((axum::http::StatusCode::BAD_REQUEST, "No intents in namespace".to_string()));
    }

    let intent_summary: String = intent_list.iter().map(|(id, desc, phrases)| {
        let samples = phrases.iter().take(5).cloned().collect::<Vec<_>>().join("\", \"");
        if desc.is_empty() {
            format!("  - {}: example phrases: [\"{}\"]", id, samples)
        } else {
            format!("  - {} | {} | example phrases: [\"{}\"]", id, desc, samples)
        }
    }).collect::<Vec<_>>().join("\n");

    let prompt = format!(
        r#"You are building a semantic routing layer for an AI assistant.

Given these intents (id | description | example phrases):
{intent_summary}

Define a concept registry that lets a fast local system detect which intent a user query expresses.

Instructions:
1. Define 8-20 SEMANTIC CONCEPTS that cover the meaning space of all intents above.
   Each concept is a named semantic unit (e.g. "wants_to_stop_service", "billing_dispute").
   Concepts should be distinct — minimal overlap between them.

2. Always include these two meta-concepts:
   - "user_is_requesting": signals the user is making a first-person request (not describing UI)
     Signals: "i want", "i need", "please", "can you", "help me", "i would like", "how do i", "i am trying to"
   - "technical_context": signals UI/code discussion rather than user intent
     Signals: "button", "the ui", "interface", "on the page", "the menu", "the form", "dropdown", "modal", "element", "component"

3. For each concept provide 15-30 signals (words or short phrases):
   - Include synonyms, informal language, common variants
   - Include morphological variants (cancel, canceling, cancelled, cancellation)
   - Lowercase only
   - Signals matched as whole words (word-boundary aware)

4. For each intent assign concept weights (0.0 to 1.0):
   - 1.0 = this concept strongly signals this intent
   - 0.5 = partial signal
   - 0.0 = no relation
   - Use -1.0 for "technical_context" on all action intents (UI discussion ≠ user request)

Return ONLY valid JSON, no explanation:
{{
  "concepts": {{
    "concept_name": ["signal1", "signal2", "signal3"],
    "user_is_requesting": ["i want", "i need", "please", "can you", "help me", "i would like", "how do i"],
    "technical_context": ["button", "the ui", "interface", "on the page", "dropdown"]
  }},
  "intent_profiles": {{
    "intent_id": {{"concept_name": 1.0, "technical_context": -1.0}},
    ...
  }}
}}"#
    );

    let response = call_llm(&state, &prompt, 8192).await?;
    let json_str = extract_json(&response);

    let parsed: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY,
            format!("LLM returned invalid JSON: {}. Raw: {}", e, &response[..response.len().min(500)])))?;

    // Build ConceptRegistry from parsed JSON
    let concepts: HashMap<String, Vec<String>> = serde_json::from_value(
        parsed["concepts"].clone()
    ).map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, format!("Bad concepts: {}", e)))?;

    let intent_profiles: HashMap<String, HashMap<String, f32>> = serde_json::from_value(
        parsed["intent_profiles"].clone()
    ).map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, format!("Bad intent_profiles: {}", e)))?;

    let total_signals: usize = concepts.values().map(|s| s.len()).sum();
    let n_concepts = concepts.len();
    let n_intents = intent_profiles.len();

    let registry = ConceptRegistry { concepts, intent_profiles };

    // Persist to disk
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}/_concepts.json", dir, app_id);
        if let Err(e) = registry.save(&path) {
            eprintln!("[concepts/bootstrap] Save failed: {}", e);
        }
    }

    // Update in-memory state
    state.concepts.write().unwrap().insert(app_id.clone(), registry.clone());

    let registry_json = serde_json::to_value(&registry).unwrap_or_default();

    eprintln!("[concepts/bootstrap] {} — {} concepts, {} intents, {} signals",
        app_id, n_concepts, n_intents, total_signals);

    Ok(Json(BootstrapResponse {
        concepts: n_concepts,
        intents_covered: n_intents,
        total_signals,
        registry: registry_json,
    }))
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
        None => Json(serde_json::json!({ "error": "no concept registry for this namespace" })),
    }
}

// ── Route query ───────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct RouteRequest {
    query: String,
    #[serde(default = "default_threshold")]
    threshold: f32,
    #[serde(default = "default_gap")]
    gap: f32,
}
fn default_threshold() -> f32 { 0.3 }
fn default_gap() -> f32 { 1.5 }

#[derive(serde::Serialize)]
struct RouteResponse {
    intents: Vec<IntentScore>,
    concepts_fired: Vec<ConceptFired>,
    has_registry: bool,
}

#[derive(serde::Serialize)]
struct IntentScore {
    id: String,
    score: f32,
}

#[derive(serde::Serialize)]
struct ConceptFired {
    concept: String,
    score: f32,
    signals: Vec<String>,
}

async fn route_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RouteRequest>,
) -> Json<RouteResponse> {
    let app_id = app_id_from_headers(&headers);
    let concepts = state.concepts.read().unwrap();

    match concepts.get(&app_id) {
        None => Json(RouteResponse {
            intents: Vec::new(),
            concepts_fired: Vec::new(),
            has_registry: false,
        }),
        Some(reg) => {
            let intents = reg.score_query_multi(&req.query, req.threshold, req.gap)
                .into_iter()
                .map(|(id, score)| IntentScore { id, score })
                .collect();

            let concepts_fired = reg.explain(&req.query)
                .into_iter()
                .map(|a| ConceptFired {
                    concept: a.concept,
                    score: a.score,
                    signals: a.matched_signals,
                })
                .collect();

            Json(RouteResponse { intents, concepts_fired, has_registry: true })
        }
    }
}

// ── Add signal (continuous learning) ─────────────────────────────────────────

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

            // Persist immediately
            if let Some(ref dir) = state.data_dir {
                let path = format!("{}/{}/_concepts.json", dir, app_id);
                let _ = reg.save(&path);
            }

            eprintln!("[concepts/signal] {}/{}: added '{}' to '{}'",
                app_id, req.concept, req.signal, req.concept);

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

/// Load concept registry from disk for a namespace, if it exists.
pub fn load_concepts(data_dir: &str, app_id: &str) -> Option<ConceptRegistry> {
    let path = format!("{}/{}/_concepts.json", data_dir, app_id);
    ConceptRegistry::load(&path).ok()
}

/// After an LLM review confirms an intent, update the concept registry.
/// Adds unmatched content words from the query to the dominant concept of the confirmed intent.
pub fn learn_from_correction(
    state: &AppState,
    app_id: &str,
    query: &str,
    correct_intent: &str,
) {
    let mut concepts = state.concepts.write().unwrap();
    let Some(reg) = concepts.get_mut(app_id) else { return };

    // Find the dominant concept for this intent (highest weight)
    let Some(profile) = reg.intent_profiles.get(correct_intent) else { return };
    let Some(dominant_concept) = profile.iter()
        .filter(|(_, &w)| w > 0.0)
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(c, _)| c.clone())
    else { return };

    // Find words in query that didn't fire any signal
    let activations = reg.explain(query);
    let fired_signals: Vec<String> = activations.iter()
        .flat_map(|a| a.matched_signals.iter().cloned())
        .collect();

    // Simple stop words
    const STOP: &[&str] = &["a","an","the","i","to","of","for","my","me","is","it",
        "in","on","at","be","do","if","or","and","not","no","we","us","our","can",
        "this","that","with","from","by","as","so","but","up","you","your","have",
        "has","had","will","would","could","should","may","might","am","are","was","were"];

    let lower = query.to_lowercase();
    let new_signals: Vec<String> = lower.split_whitespace()
        .filter(|w| w.len() > 2 && !STOP.contains(w))
        .filter(|w| !fired_signals.iter().any(|s| s.contains(*w)))
        .map(|w| w.to_string())
        .collect();

    if new_signals.is_empty() { return; }

    eprintln!("[concepts/learn] {}/{}: adding {:?} to concept '{}'",
        app_id, correct_intent, new_signals, dominant_concept);

    for signal in &new_signals {
        reg.add_signal(&dominant_concept, signal);
    }

    // Persist
    if let Some(ref dir) = state.data_dir {
        let path = format!("{}/{}/_concepts.json", dir, app_id);
        let _ = reg.save(&path);
    }
}
