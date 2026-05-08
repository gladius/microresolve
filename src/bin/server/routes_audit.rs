//! Audit log HTTP endpoints — minimal surface.
//!
//! - `GET    /api/audit/heads`             — current chain heads per key (kid, count, head_hash)
//! - `GET    /api/audit/config`            — server-wide audit mode (off | default)
//! - `POST   /api/audit/verify`            — re-walk all chains, recompute hashes
//!
//! Cut from v0.2.2 in the scope-trim:
//! - per-namespace audit mode override (`PUT /api/audit/ns/{ns}/mode`)
//! - custom application events (`POST /api/audit/event`)
//! - audit entries getter (`GET /api/audit/entries`)
//!
//! The chain primitive is the foundation we keep; those elaborations
//! were compliance flavour for an audience MicroResolve doesn't target
//! (we're an intent classifier that ships compliance packs, not a
//! compliance platform). The verify CLI covers the audit needs that
//! the packs require.

use crate::audit_verify;
use crate::state::*;
use axum::{extract::State, routing::*, Json};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/audit/heads", get(audit_heads))
        .route("/api/audit/config", get(audit_config))
        .route("/api/audit/verify", post(audit_verify_endpoint))
}

/// Current chain heads — what each key has written so far.
async fn audit_heads(State(state): State<AppState>) -> Json<serde_json::Value> {
    let heads = state.audit_log.heads();
    Json(serde_json::json!({ "heads": heads }))
}

/// Server-wide audit mode.
async fn audit_config(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "mode": state.audit_log.mode(),
        "modes_available": ["off", "default"],
    }))
}

/// Re-walk all chains and verify integrity. Returns per-chain status.
async fn audit_verify_endpoint(State(state): State<AppState>) -> Json<serde_json::Value> {
    let report = audit_verify::verify_all(state.data_dir.as_deref());
    Json(serde_json::to_value(report).unwrap_or_default())
}
