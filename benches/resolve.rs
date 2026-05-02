//! In-process latency benchmark for `MicroResolve::namespace().resolve()`.
//!
//! Measures the engine itself — no HTTP, no JSON, no Python. This is the
//! number the library claim ("sub-millisecond classification") rests on.
//! Run: `cargo bench --bench resolve`
//!
//! The HTTP-end-to-end equivalent (which adds ~500µs of TCP/JSON overhead
//! a real Python client sees) is in `benchmarks/latency.py`.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use microresolve::{MicroResolve, MicroResolveConfig};
use std::hint::black_box;

const VERBS: &[&str] = &[
    "cancel", "list", "show", "create", "update", "delete", "fetch", "stop", "start", "renew",
    "refund", "track", "send", "reset", "approve", "decline",
];
const NOUNS: &[&str] = &[
    "order",
    "subscription",
    "invoice",
    "payment",
    "account",
    "shipment",
    "ticket",
    "user",
    "session",
    "report",
    "notification",
    "transaction",
    "discount",
    "product",
    "customer",
    "review",
];
const QUALIFIERS: &[&str] = &[
    "my", "the", "this", "that", "current", "recent", "latest", "pending",
];

const N_INTENTS: usize = 100;

/// Build a deterministic 100-intent × 5-seeds engine for benchmarking.
fn build_engine() -> MicroResolve {
    let engine = MicroResolve::new(MicroResolveConfig::default()).expect("engine init");
    let ns = engine.namespace("bench");
    for i in 0..N_INTENTS {
        let verb = VERBS[i % VERBS.len()];
        let noun = NOUNS[(i / VERBS.len()) % NOUNS.len()];
        let intent_id = format!("{verb}_{noun}_{i:03}");
        let seeds: Vec<String> = vec![
            format!("{verb} my {noun}"),
            format!("{verb} the {noun}"),
            format!("please {verb} {noun}"),
            format!("can you {verb} my {noun}"),
            format!("i want to {verb} {noun}"),
        ];
        ns.add_intent(&intent_id, seeds).expect("add_intent");
    }
    engine
}

/// Build a query that exercises the indexed vocabulary.
fn make_query(seed: u64) -> String {
    let v = VERBS[seed as usize % VERBS.len()];
    let n = NOUNS[((seed / 7) as usize) % NOUNS.len()];
    let q = QUALIFIERS[((seed / 13) as usize) % QUALIFIERS.len()];
    format!("{v} {q} {n}")
}

fn bench_resolve(c: &mut Criterion) {
    let engine = build_engine();
    let mut group = c.benchmark_group("resolve");

    // Single-call latency on a representative query — what users see per
    // classification call. Criterion handles warmup + outlier detection.
    group.bench_function("single_query", |b| {
        let ns = engine.namespace("bench");
        b.iter_batched(
            || "cancel my subscription".to_string(),
            |q| black_box(ns.resolve(black_box(&q))),
            BatchSize::SmallInput,
        );
    });

    // Throughput: how many calls/sec on a single thread, varied queries.
    group.bench_function("varied_queries", |b| {
        let ns = engine.namespace("bench");
        let mut seed: u64 = 0;
        b.iter(|| {
            seed = seed.wrapping_add(1);
            let q = make_query(seed);
            black_box(ns.resolve(black_box(&q)))
        });
    });

    group.finish();
}

criterion_group!(benches, bench_resolve);
criterion_main!(benches);
