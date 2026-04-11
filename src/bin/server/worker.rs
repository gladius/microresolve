//! Background auto-learn worker.
//!
//! Woken by `state.worker_notify` whenever a new item enters the log.
//! In "auto" mode: calls `full_review` for every unreviewed flagged entry,
//! applies fixes, and broadcasts `StudioEvent`s to SSE subscribers.
//! In "manual" mode: sits idle.

use std::sync::Arc;
use tokio::sync::Notify;
use crate::state::{AppState, StudioEvent};
use crate::log_store::ReviewStatus;
use crate::llm::{full_review, apply_review};

pub async fn run_worker(state: AppState, notify: Arc<Notify>) {
    loop {
        // Wait for a signal that there's work to do.
        notify.notified().await;

        let mode = state.review_mode.read().unwrap().clone();
        if mode != "auto" {
            continue;
        }

        // Collect pending (app_id, id) pairs — read-only snapshot.
        let pending = state.log_store.lock().unwrap().pending_worker_ids(None);

        for (app_id, id) in pending {
            // Re-check mode before each item (user might have switched mid-batch)
            if state.review_mode.read().unwrap().as_str() != "auto" {
                break;
            }

            // Read the full record (query + detected intents)
            let record = match state.log_store.lock().unwrap().get_record(&app_id, id) {
                Some(r) => r,
                None => continue,
            };

            // Mark as processing so we don't pick it up again
            state.log_store.lock().unwrap().set_review_status(&app_id, id, ReviewStatus {
                status: "processing".to_string(),
                version_before: {
                    state.routers.read().unwrap()
                        .get(&app_id).map(|r| r.version()).unwrap_or(0)
                },
                ..ReviewStatus::pending()
            });

            let _ = state.event_tx.send(StudioEvent::LlmStarted {
                id,
                query: record.query.clone(),
            });

            let version_before = state.routers.read().unwrap()
                .get(&app_id).map(|r| r.version()).unwrap_or(0);

            let model = std::env::var("LLM_MODEL")
                .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());

            match full_review(&state, &app_id, &record.query, &record.detected_intents).await {
                Ok(review_result) => {
                    let (phrases_added, phrases_replaced) =
                        apply_review(&state, &app_id, &review_result, &record.query).await;

                    let version_after = state.routers.read().unwrap()
                        .get(&app_id).map(|r| r.version()).unwrap_or(0);

                    // Resolve log entry (tombstone the binary record)
                    state.log_store.lock().unwrap().resolve(&app_id, id);

                    state.log_store.lock().unwrap().set_review_status(&app_id, id, ReviewStatus {
                        status: "done".to_string(),
                        llm_reviewed: true,
                        llm_model: Some(model),
                        llm_result: serde_json::to_value(&review_result).ok(),
                        applied: phrases_added > 0 || phrases_replaced > 0,
                        phrases_added,
                        version_before,
                        version_after: if version_after != version_before { Some(version_after) } else { None },
                        summary: if review_result.summary.is_empty() { None } else { Some(review_result.summary.clone()) },
                    });

                    let _ = state.event_tx.send(StudioEvent::LlmDone {
                        id,
                        correct: review_result.correct_intents.clone(),
                        wrong: review_result.wrong_detections.clone(),
                        phrases_added,
                        summary: review_result.summary,
                    });

                    if phrases_added > 0 || phrases_replaced > 0 {
                        let _ = state.event_tx.send(StudioEvent::FixApplied {
                            id,
                            phrases_added,
                            phrases_replaced,
                            version_before,
                            version_after,
                        });
                    }
                }
                Err(reason) => {
                    state.log_store.lock().unwrap().set_review_status(&app_id, id, ReviewStatus {
                        status: "escalated".to_string(),
                        ..ReviewStatus::pending()
                    });
                    let _ = state.event_tx.send(StudioEvent::Escalated { id, reason });
                }
            }
        }
    }
}
