# Changelog

## 0.1.0 (2026-04-07)

Initial release.

### Core Library
- Intent routing via inverted index with BM25-inspired scoring
- Dual-layer sparse vectors: seed (immutable) + learned (incremental)
- Multi-intent decomposition with relation detection (parallel, sequential, conditional, negation)
- Negation-aware tokenization (12 languages including CJK)
- Intent types: Action vs Context with projected context
- Opaque metadata per intent
- Co-occurrence tracking and workflow discovery
- State export/import (JSON serialization)
- Auto-discovery of intent clusters from unlabeled queries (PMI-anchored clustering)

### Server
- Axum HTTP server with 40+ endpoints
- Multi-app support via `X-App-ID` header
- Auto-persistence with `--data <dir>`
- LLM-powered features: seed generation, routing review, training arena (requires Anthropic API key)
- Discovery endpoints with LLM-assisted cluster naming

### UI
- React + Tailwind dashboard
- Live routing with intent span visualization
- Intent management with type toggles and metadata
- Learn mode with LLM-reviewed routing
- App selector for multi-app management
- Discovery page: upload queries, review clusters, apply as intents

### Bindings
- Python via PyO3 (native speed, pip-installable)
- Node.js via napi-rs (native speed, npm-installable)
- WASM via wasm-bindgen (browser/edge deployment)

### Benchmarks
- CLINC150: 87.5% accuracy (100 seeds/intent), 22μs avg latency
- BANKING77: 85.5% accuracy (130 seeds/intent), 21μs avg latency
- Zero ML training, zero GPU
