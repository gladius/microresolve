//! Connected mode: pull-based sync, log shipping, multi-app routing.
//!
//! In connected mode, routing happens locally (zero latency) but the library
//! periodically syncs with a central MicroResolve server to pick up model improvements,
//! and ships anonymized query logs back for the server's LLM review pipeline.
//!
//! ## Architecture
//! - Each app_id gets its own `Arc<Router>`, hot-swapped atomically on version update
//! - Background thread: polls `/api/sync` every 30s, flushes logs every 30s
//! - No tokio required — uses `reqwest::blocking` in a plain `std::thread`
//!
//! ## Feature gate
//! Enable with `features = ["connect"]` in Cargo.toml.

pub mod app_router;
pub mod types;
pub(crate) mod sync;

pub use app_router::AppRouter;
pub use types::{ConnectConfig, LogEntry};
