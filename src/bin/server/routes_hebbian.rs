//! Hebbian graph persistence utilities — called at startup by main.rs.
//!
//! L1 (LexicalGraph): seeded via routes_import::seed_into_l1.
//! L2 (IntentGraph):  seeded automatically on intent add/import via routes_import::seed_into_l2.
//!
//! No API routes — these are internal implementation layers.

use asv_router::scoring::{LexicalGraph, IntentGraph};

/// Load a hebbian graph from disk for a namespace. Called at startup.
pub fn load_hebbian(data_dir: &str, namespace: &str) -> Option<LexicalGraph> {
    let path = format!("{}/{}/_hebbian.json", data_dir, namespace);
    LexicalGraph::load(&path).ok()
}

/// Load an intent graph from disk for a namespace. Called at startup.
pub fn load_intent_graph(data_dir: &str, namespace: &str) -> Option<IntentGraph> {
    let path = format!("{}/{}/_intent_graph.json", data_dir, namespace);
    IntentGraph::load(&path).ok()
}
