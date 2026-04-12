//! ASV Router — Benchmark Runner (stub)
//!
//! BM25 routing removed. Benchmarks now run against Hebbian L3 via the server.
//! Start the server and use POST /api/route_multi to evaluate latency and accuracy.

fn main() {
    eprintln!("benchmark: BM25 routing removed. Hebbian L3 routing runs server-side.");
    eprintln!("Start the server (cargo run --release --bin server --features server)");
    eprintln!("and POST to /api/route_multi to benchmark.");
}
