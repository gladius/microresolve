//! Routing profiler — routing is now Hebbian L3 (IntentGraph), not Router.
//! This binary is a placeholder; real profiling happens in the server process.

fn main() {
    eprintln!("profile_route: BM25 routing removed. Hebbian L3 routing runs server-side.");
    eprintln!("Start the server and use /api/route_multi to measure latency.");
}
