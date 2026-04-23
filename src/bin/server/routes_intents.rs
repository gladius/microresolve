//! Intent management endpoints.

use axum::{
    extract::{State, Path},
    http::{StatusCode, HeaderMap},
    routing::{get, post, patch},
    Json,
};
use microresolve::NamespaceModel;
use microresolve::{Router, IntentType};
use crate::state::*;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        // Modern RESTful surface (preferred for new callers)
        .route("/api/intents",                  get(list_intents).post(add_intent))
        .route("/api/intents/{id}",             patch(patch_intent).delete(delete_intent_by_id))
        .route("/api/intents/{id}/phrases",     post(add_phrase_to_intent).delete(remove_phrase_from_intent))
        // Multilingual seed payload (POST body shape differs from POST /intents)
        .route("/api/intents/multilingual",     post(add_intent_multilingual))
        // Cross-cutting operations
        .route("/api/intents/discriminate",     post(discriminate_intents))
        .route("/api/ns/models",                get(get_ns_models).post(set_ns_models))

        // ── DEPRECATED ─────────────────────────────────────────────────────
        // Per-field POST endpoints kept for backward compatibility during
        // Phase 2 of the API consolidation. UI is migrating to PATCH /intents/{id}.
        // To be removed in Phase 3 of the consolidation (next session).
        .route("/api/intents/delete",           post(delete_intent))
        .route("/api/intents/phrase",           post(add_phrase))
        .route("/api/intents/phrase/remove",    post(remove_phrase))
        .route("/api/intents/type",             post(set_intent_type))
        .route("/api/intents/description",      post(set_intent_description))
        .route("/api/intents/instructions",     post(set_intent_instructions))
        .route("/api/intents/persona",          post(set_intent_persona))
        .route("/api/intents/guardrails",       post(set_intent_guardrails))
        .route("/api/intents/target",           post(set_intent_target))
}

// ── New RESTful handlers ────────────────────────────────────────────────────

/// Partial update of an intent. Any subset of fields may be provided.
#[derive(serde::Deserialize)]
pub struct PatchIntentRequest {
    #[serde(default)] pub intent_type: Option<IntentType>,
    #[serde(default)] pub description: Option<String>,
    #[serde(default)] pub instructions: Option<String>,
    #[serde(default)] pub persona: Option<String>,
    #[serde(default)] pub guardrails: Option<Vec<String>>,
    #[serde(default)] pub target: Option<microresolve::IntentTarget>,
}

pub async fn patch_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PatchIntentRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or((StatusCode::NOT_FOUND, format!("namespace '{}' not found", app_id)))?;

    if router.get_training(&id).is_none() {
        return Err((StatusCode::NOT_FOUND, format!("intent '{}' not found", id)));
    }

    if let Some(t) = req.intent_type   { router.set_intent_type(&id, t); }
    if let Some(d) = req.description   { router.set_description(&id, &d); }
    if let Some(i) = req.instructions  { router.set_instructions(&id, &i); }
    if let Some(p) = req.persona       { router.set_persona(&id, &p); }
    if let Some(g) = req.guardrails    { router.set_guardrails(&id, g); }
    if let Some(t) = req.target        { router.set_target(&id, t); }

    maybe_persist(&state, &app_id, router);
    Ok(StatusCode::NO_CONTENT)
}

pub async fn delete_intent_by_id(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.remove_intent(&id);
    maybe_persist(&state, &app_id, router);
    StatusCode::NO_CONTENT
}

#[derive(serde::Deserialize)]
pub struct PhrasePayload {
    pub phrase: String,
    #[serde(default = "default_lang")]
    pub lang: String,
}

pub async fn add_phrase_to_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PhrasePayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Reuse existing handler logic by adapting the request shape.
    add_phrase(
        State(state),
        headers,
        Json(AddPhraseRequest { intent_id: id, phrase: req.phrase, lang: req.lang }),
    ).await
}

pub async fn remove_phrase_from_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<PhrasePayload>,
) -> Result<StatusCode, (StatusCode, String)> {
    remove_phrase(
        State(state),
        headers,
        Json(RemovePhraseRequest { intent_id: id, phrase: req.phrase }),
    ).await
}

pub async fn list_intents(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let app_id = app_id_from_headers(&headers);
    ensure_app(&state, &app_id);
    let routers = state.routers.read().unwrap();
    let router = match routers.get(&app_id) {
        Some(r) => r,
        None => return Json(serde_json::json!([])),
    };
    let mut ids = router.intent_ids();
    ids.sort();
    let intents: Vec<serde_json::Value> = ids
        .iter()
        .map(|id| {
            let seeds = router.get_training(id).unwrap_or_default();
            let by_lang = router.get_training_by_lang(id).cloned().unwrap_or_default();
            let learned = 0usize;
            let intent_type = router.get_intent_type(id);
            let description = router.get_description(id);
            let instructions = router.get_instructions(id);
            let persona = router.get_persona(id);
            let source = router.get_source(id);
            let target = router.get_target(id);
            let schema = router.get_schema(id);
            let guardrails = router.get_guardrails(id);
            serde_json::json!({
                "id": id,
                "description": description,
                "phrases": seeds,
                "phrases_by_lang": by_lang,
                "learned_count": learned,
                "intent_type": intent_type,
                "instructions": instructions,
                "persona": persona,
                "source": source,
                "target": target,
                "schema": schema,
                "guardrails": guardrails,
            })
        })
        .collect();
    Json(serde_json::json!(intents))
}

#[derive(serde::Deserialize)]
pub struct AddIntentRequest {
    id: String,
    #[serde(default)]
    phrases: Vec<String>,
    #[serde(default)]
    intent_type: Option<IntentType>,
}

pub async fn add_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddIntentRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.entry(app_id.clone()).or_insert_with(Router::new);
    let seed_refs: Vec<&str> = req.phrases.iter().map(|s| s.as_str()).collect();
    router.add_intent(&req.id, &seed_refs);
    if let Some(t) = req.intent_type {
        router.set_intent_type(&req.id, t);
    }
    maybe_persist(&state, &app_id, router);
    StatusCode::CREATED
}

#[derive(serde::Deserialize)]
pub struct AddPhraseRequest {
    intent_id: String,
    phrase: String,
    #[serde(default = "default_lang")]
    lang: String,
}

pub fn default_lang() -> String { "en".to_string() }

pub async fn add_phrase(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddPhraseRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;

    if router.get_training(&req.intent_id).is_none() {
        return Err((StatusCode::NOT_FOUND, format!("intent '{}' not found", req.intent_id)));
    }

    let result = router.add_phrase_checked(&req.intent_id, &req.phrase, &req.lang);

    if result.added {
        maybe_persist(&state, &app_id, router);
    }

    let counts = router.seed_counts_by_lang(&req.intent_id);
    drop(routers);

    Ok(Json(serde_json::json!({
        "added": result.added,
        "counts": counts,
        "new_terms": result.new_terms,
        "conflicts": result.conflicts.iter().map(|c| serde_json::json!({
            "term": c.term,
            "competing_intent": c.competing_intent,
            "severity": c.severity,
        })).collect::<Vec<_>>(),
        "redundant": result.redundant,
        "reason": result.warning,
    })))
}

#[derive(serde::Deserialize)]
pub struct RemovePhraseRequest {
    intent_id: String,
    phrase: String,
}

pub async fn remove_phrase(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RemovePhraseRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.get_mut(&app_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("app '{}' not found", app_id)))?;
    if router.remove_phrase(&req.intent_id, &req.phrase) {
        maybe_persist(&state, &app_id, router);
        Ok(StatusCode::OK)
    } else {
        Err((StatusCode::NOT_FOUND, "phrase not found".to_string()))
    }
}

#[derive(serde::Deserialize)]
pub struct AddIntentMultilingualRequest {
    id: String,
    phrases_by_lang: std::collections::HashMap<String, Vec<String>>,
    #[serde(default)]
    intent_type: Option<IntentType>,
    #[serde(default)]
    description: Option<String>,
}

pub async fn add_intent_multilingual(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<AddIntentMultilingualRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    ensure_app(&state, &app_id);

    {
        let mut routers = state.routers.write().unwrap();
        let router = routers.get_mut(&app_id).unwrap();
        router.add_intent_multilingual(&req.id, req.phrases_by_lang);
        if let Some(t) = req.intent_type {
            router.set_intent_type(&req.id, t);
        }
        if let Some(desc) = req.description {
            if !desc.is_empty() {
                router.set_description(&req.id, &desc);
            }
        }
        maybe_persist(&state, &app_id, router);
    }

    StatusCode::CREATED
}

#[derive(serde::Deserialize)]
pub struct SetIntentTypeRequest {
    intent_id: String,
    intent_type: IntentType,
}

pub async fn set_intent_type(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetIntentTypeRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.set_intent_type(&req.intent_id, req.intent_type);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
pub struct SetDescriptionRequest {
    intent_id: String,
    description: String,
}

pub async fn set_intent_description(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetDescriptionRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.set_description(&req.intent_id, &req.description);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
pub struct DeleteIntentRequest {
    id: String,
}

pub async fn delete_intent(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DeleteIntentRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.remove_intent(&req.id);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
pub struct SetInstructionsRequest {
    intent_id: String,
    instructions: String,
}

pub async fn set_intent_instructions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetInstructionsRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.set_instructions(&req.intent_id, &req.instructions);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
pub struct SetPersonaRequest {
    intent_id: String,
    persona: String,
}

pub async fn set_intent_persona(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetPersonaRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.set_persona(&req.intent_id, &req.persona);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
pub struct SetGuardrailsRequest {
    intent_id: String,
    guardrails: Vec<String>,
}

pub async fn set_intent_guardrails(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetGuardrailsRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.set_guardrails(&req.intent_id, req.guardrails);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

#[derive(serde::Deserialize)]
pub struct SetTargetRequest {
    intent_id: String,
    target: microresolve::IntentTarget,
}

pub async fn set_intent_target(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<SetTargetRequest>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = match routers.get_mut(&app_id) {
        Some(r) => r,
        None => return StatusCode::NOT_FOUND,
    };
    router.set_target(&req.intent_id, req.target);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

pub async fn get_ns_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<Vec<NamespaceModel>> {
    let app_id = app_id_from_headers(&headers);
    ensure_app(&state, &app_id);
    let routers = state.routers.read().unwrap();
    let models = routers.get(&app_id)
        .map(|r| r.get_namespace_models().to_vec())
        .unwrap_or_default();
    Json(models)
}

pub async fn set_ns_models(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(models): Json<Vec<NamespaceModel>>,
) -> StatusCode {
    let app_id = app_id_from_headers(&headers);
    let mut routers = state.routers.write().unwrap();
    let router = routers.entry(app_id.clone()).or_insert_with(microresolve::Router::new);
    router.set_namespace_models(models);
    maybe_persist(&state, &app_id, router);
    StatusCode::OK
}

// ─── Discriminate ─────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct DiscriminateRequest {
    /// Only analyze intents in this domain prefix (e.g. "notion"). None = all intents.
    #[serde(default)]
    domain: Option<String>,
    /// Jaccard token overlap threshold above which a pair is considered confused. Default 0.15.
    #[serde(default = "default_threshold")]
    threshold: f32,
    /// Phrases to generate per intent per confused pair. Default 5.
    #[serde(default = "default_phrases_per_pair")]
    phrases_per_pair: usize,
    /// If true, find pairs but don't call LLM or add phrases.
    #[serde(default)]
    dry_run: bool,
}

fn default_threshold() -> f32 { 0.15 }
fn default_phrases_per_pair() -> usize { 5 }

#[derive(serde::Serialize)]
struct ConfusedPair {
    intent_a: String,
    intent_b: String,
    overlap: f32,
    phrases_added_a: usize,
    phrases_added_b: usize,
}

pub async fn discriminate_intents(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DiscriminateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    use std::collections::HashSet;

    let app_id = app_id_from_headers(&headers);
    ensure_app(&state, &app_id);

    // ── Snapshot intent data (release lock before async LLM calls) ────────────
    struct IntentSnap { id: String, description: String, phrases: Vec<String> }

    let snaps: Vec<IntentSnap> = {
        let routers = state.routers.read().unwrap();
        let router = routers.get(&app_id)
            .ok_or_else(|| (StatusCode::NOT_FOUND, "namespace not found".to_string()))?;

        let mut ids = router.intent_ids();
        ids.sort();

        // Filter by domain prefix if requested
        if let Some(ref domain) = req.domain {
            let prefix = format!("{}:", domain);
            ids.retain(|id| id.starts_with(&prefix));
        }

        ids.into_iter().map(|id| IntentSnap {
            description: router.get_description(&id).to_string(),
            phrases: router.get_training(&id).unwrap_or_default(),
            id,
        }).collect()
    };

    if snaps.len() < 2 {
        return Ok(Json(serde_json::json!({
            "pairs_analyzed": 0, "phrases_added": 0, "pairs": []
        })));
    }

    // ── Load stop words (global, language-agnostic per language file) ────────
    let stop_words: HashSet<String> = if let Some(ref data_dir) = state.data_dir {
        crate::routes_stopwords::load_all_stopwords(data_dir)
            .into_values()
            .flatten()
            .collect()
    } else {
        crate::routes_stopwords::EN_STOPWORDS.iter().map(|s| s.to_string()).collect()
    };

    // ── Build token sets per intent ───────────────────────────────────────────
    // Split on non-alphanumeric, lowercase, min 3 bytes (keeps CJK single chars @ 3 bytes each).
    let token_sets: Vec<HashSet<String>> = snaps.iter().map(|s| {
        let mut tokens = HashSet::new();
        for phrase in &s.phrases {
            for tok in phrase.split(|c: char| !c.is_alphanumeric()) {
                let t = tok.to_lowercase();
                if t.len() >= 3 && !stop_words.contains(&t) {
                    tokens.insert(t);
                }
            }
        }
        tokens
    }).collect();

    // ── Find confused pairs via Jaccard overlap ───────────────────────────────
    let mut confused: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..snaps.len() {
        for j in (i + 1)..snaps.len() {
            let a = &token_sets[i];
            let b = &token_sets[j];
            if a.is_empty() || b.is_empty() { continue; }
            let inter = a.intersection(b).count() as f32;
            let union = a.union(b).count() as f32;
            let jaccard = inter / union;
            if jaccard >= req.threshold {
                confused.push((i, j, jaccard));
            }
        }
    }
    confused.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    if req.dry_run || confused.is_empty() {
        let pairs: Vec<serde_json::Value> = confused.iter().map(|(i, j, score)| serde_json::json!({
            "intent_a": snaps[*i].id,
            "intent_b": snaps[*j].id,
            "overlap": (score * 100.0) as u32,
            "phrases_added_a": 0,
            "phrases_added_b": 0,
        })).collect();
        return Ok(Json(serde_json::json!({
            "pairs_analyzed": pairs.len(),
            "phrases_added": 0,
            "pairs": pairs,
            "dry_run": true,
        })));
    }

    // ── LLM: generate discriminative phrases per pair ─────────────────────────
    let mut result_pairs: Vec<ConfusedPair> = Vec::new();
    let mut total_added = 0usize;
    let n = req.phrases_per_pair;

    for (i, j, overlap) in &confused {
        let a = &snaps[*i];
        let b = &snaps[*j];

        let sample_a = a.phrases.iter().take(6).cloned().collect::<Vec<_>>().join(", ");
        let sample_b = b.phrases.iter().take(6).cloned().collect::<Vec<_>>().join(", ");

        let prompt = format!(
            "You are helping an intent router distinguish between two similar intents.\n\n\
            Intent A: {id_a}\nDescription: {desc_a}\nExample phrases: [{sample_a}]\n\n\
            Intent B: {id_b}\nDescription: {desc_b}\nExample phrases: [{sample_b}]\n\n\
            Generate {n} user queries that clearly match Intent A and would NOT be confused with Intent B.\n\
            Generate {n} user queries that clearly match Intent B and would NOT be confused with Intent A.\n\
            Each phrase should be a realistic user request (1-2 sentences).\n\
            Return ONLY valid JSON: {{\"intent_a\": [...{n} phrases...], \"intent_b\": [...{n} phrases...]}}",
            id_a = a.id, desc_a = a.description, sample_a = sample_a,
            id_b = b.id, desc_b = b.description, sample_b = sample_b,
            n = n,
        );

        let response = match crate::pipeline::call_llm(&state, &prompt, 600).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[discriminate] LLM failed for ({}, {}): {}", a.id, b.id, e.1);
                continue;
            }
        };

        let json_str = crate::pipeline::extract_json(&response);
        let parsed: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[discriminate] JSON parse failed: {}. Raw: {}", e, &response[..response.len().min(200)]);
                continue;
            }
        };

        let mut added_a = 0usize;
        let mut added_b = 0usize;

        // Add phrases — add_phrase_checked indexes into L2 atomically
        {
            let mut routers = state.routers.write().unwrap();
            let router = match routers.get_mut(&app_id) {
                Some(r) => r,
                None => continue,
            };
            if let Some(phrases) = parsed["intent_a"].as_array() {
                for p in phrases {
                    if let Some(s) = p.as_str() {
                        if router.add_phrase_checked(&a.id, s, "en").added { added_a += 1; }
                    }
                }
            }
            if let Some(phrases) = parsed["intent_b"].as_array() {
                for p in phrases {
                    if let Some(s) = p.as_str() {
                        if router.add_phrase_checked(&b.id, s, "en").added { added_b += 1; }
                    }
                }
            }
        }

        eprintln!("[discriminate] ({} ↔ {}) overlap={:.0}% added_a={} added_b={}",
            a.id, b.id, overlap * 100.0, added_a, added_b);
        total_added += added_a + added_b;
        result_pairs.push(ConfusedPair {
            intent_a: a.id.clone(),
            intent_b: b.id.clone(),
            overlap: (overlap * 100.0).round(),
            phrases_added_a: added_a,
            phrases_added_b: added_b,
        });
    }

    // Persist once after all pairs processed
    if total_added > 0 {
        let mut routers = state.routers.write().unwrap();
        if let Some(router) = routers.get_mut(&app_id) {
            maybe_persist(&state, &app_id, router);
        }
    }

    Ok(Json(serde_json::json!({
        "pairs_analyzed": result_pairs.len(),
        "phrases_added": total_added,
        "pairs": result_pairs,
    })))
}
