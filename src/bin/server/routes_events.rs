//! SSE (Server-Sent Events) endpoint for real-time Studio feed.
//!
//! `GET /api/events` — browser connects once, receives `StudioEvent` JSON objects
//! as they happen (queued, llm_started, llm_done, fix_applied, escalated).

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::get,
};
use std::convert::Infallible;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt as _;
use crate::state::{AppState, StudioEvent};

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/events", get(sse_handler))
}

async fn sse_handler(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.event_tx.subscribe();
    let broadcast = BroadcastStream::new(rx);

    let stream = broadcast.filter_map(|res| {
        match res {
            Ok(event) => {
                let data = serde_json::to_string(&event).unwrap_or_default();
                Some(Ok(Event::default().data(data)))
            }
            Err(_) => None, // lagged or closed — skip
        }
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Convenience: emit an ItemQueued event and wake the worker.
/// Called from `routes_core.rs` after logging each query.
pub fn emit_queued(state: &AppState, id: u64, query: &str, app_id: &str) {
    let _ = state.event_tx.send(StudioEvent::ItemQueued {
        id,
        query: query.to_string(),
        app_id: app_id.to_string(),
    });
    state.worker_notify.notify_one();
}
