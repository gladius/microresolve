# Contributing to ASV Router

## Development Setup

```bash
git clone https://github.com/gladius/asv-router.git
cd asv-router
cargo test            # run all 169 tests
cargo run --example rust_basic  # verify library works
```

### Server

```bash
cargo run --release --bin server --features server -- --data ./data
```

### UI

```bash
cd ui && npm install && npm run dev
```

### Python bindings

```bash
cd python
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release
python ../examples/python/basic.py
```

### Node.js bindings

```bash
cd node
npm install
npx napi build --release
node ../examples/node/basic.js
```

## Running Tests

```bash
cargo test                          # all tests (169)
cargo test discovery                # discovery module only
cargo test scenario_regression      # 30-scenario regression
cargo test --test integration       # integration tests
```

## Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix warnings
- No `unsafe` code in the library
- Doc comments on all public items

## Pull Request Process

1. Fork and create a feature branch
2. Write tests for new functionality
3. Ensure `cargo test` passes
4. Ensure `cargo clippy` has no warnings
5. Update README.md if adding user-facing features
6. Submit PR with clear description of what and why

## Architecture

- `src/lib.rs` — Router struct, main API
- `src/tokenizer.rs` — query tokenization (Latin + CJK)
- `src/index.rs` — inverted index (BM25-inspired scoring)
- `src/vector.rs` — dual-layer sparse vectors (seed + learned)
- `src/multi.rs` — multi-intent decomposition + relations
- `src/discovery.rs` — auto-discovery from unlabeled queries
- `src/seed.rs` — LLM prompt building for seed generation
- `src/bin/server.rs` — Axum HTTP server (feature-gated)

The library (`src/lib.rs` + modules) has zero network dependencies. The server binary adds HTTP via the `server` feature flag.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include: Rust version, OS, minimal reproduction steps
- For security issues, email directly (do not open a public issue)
