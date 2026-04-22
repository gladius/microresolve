# ASV Benchmarks

## Structure

```
benchmarks/
  datasets/       — seed + test JSON files for each benchmark
  results/        — saved results (JSON + markdown summaries)
  scripts/        — Python benchmark runners (HTTP pipeline)
  src/            — Rust benchmark runners (direct, true latency)
```

## Benchmark Types

### Python (HTTP) — `scripts/`
Tests full pipeline including auto-learn. Measures wall-clock latency.
Requires server running on localhost:3001.

### Rust (direct) — `src/`  
Tests routing only, no HTTP. Measures true <50µs routing latency.
Run with: `cargo bench --features bench`

## Datasets

| Dataset | Intents | Test queries | Type |
|---------|---------|--------------|------|
| CLINC150 | 150 | 4500 | Single-intent, banking/finance |
| MixSNIPS | 7 | 2199 | Multi-intent, voice assistant |
| MixATIS | 17 | 828 | Multi-intent, airline |
| BANKING77 | 77 | 3080 | Single-intent, banking |
| HWU64 | 64 | 2152 | Single-intent, home assistant |
| MASSIVE-EN | — | — | Multilingual |
| mcp_routing | TBD | TBD | Real-world: MCP tool dispatch |
| customer_support | TBD | TBD | Real-world: SaaS support |

## Running

```bash
# Single-intent learning curve (CLINC150)
python3 scripts/bench_learning_curve.py --dataset clinc150 --rounds 3

# Multi-intent auto-learn (MixSNIPS, uses Haiku)
python3 scripts/bench_mixsnips_autolearn.py --rounds 2

# MixATIS cold start
python3 scripts/bench_mixatis.py

# Rust routing latency
cargo bench --features bench
```

## Results Summary

See `results/` for full JSON. Key numbers:

| Benchmark | Cold Start | After Learning | Latency |
|-----------|-----------|----------------|---------|
| CLINC150 top-1 | 74.8% | 97.6% | ~2ms |
| MixSNIPS F1 | TBD | TBD | ~300µs |
| MixSNIPS Exact | TBD | TBD | — |
| Routing (Rust) | — | — | <50µs |
