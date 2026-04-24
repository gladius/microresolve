# Changelog

All notable changes to MicroResolve are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Entity layer: detection, augmentation, extraction, and masking of PII and custom entities.
- MCP tool prefiltering via the same lexical engine.
- Per-namespace default threshold with cascade.
- IDF cache built at index time (O(1) per token at query).

### Changed
- README and tagline: "Pre-LLM decision layer" framing.

### Removed
- L3 inhibition layer (replaced by simpler cascade).

---

## [0.1.0] — TBD

Initial public release.

### Core
- Model-free intent routing via inverted index + learned sparse vectors.
- Latin + CJK dual-path tokenization.
- Multi-intent decomposition with relation detection (sequential, conditional, negation, parallel).
- Continuous on-device learning from live queries and historic logs.
- CRDT merge for cross-replica state sync.

### Bindings
- Rust library (`microresolve` on crates.io).
- Python bindings via PyO3 + maturin (`microresolve` on PyPI).
- Node.js bindings via napi-rs (`microresolve` on npm).
- HTTP server (axum-based) for language-agnostic access.

### Benchmarks
- CLINC150: 95.1% exact (continuous learning), 84.9% seed-only; ~76μs per query.
- BANKING77: 89.7% exact (continuous learning), 80.4% seed-only; ~50μs per query.
- Multi-intent (44 compound queries): 95.5% detection, 90.9% ordering.

---

[Unreleased]: https://github.com/gladius/microresolve/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gladius/microresolve/releases/tag/v0.1.0
