# Changelog

All notable changes to MicroResolve are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Empty `subscribe` list now auto-subscribes to all namespaces** the
  server exposes. Pass `subscribe=[]` (Python: omit / `None`; Node:
  omit / `[]`) and the library queries `GET /api/namespaces` at connect
  time, then pulls each one. Explicit lists still work as an allow-list
  for multi-tenant cases. Zero-config for solo / single-team setups.

---

## [0.1.5] — 2026-04-30

### Breaking

- **Connected-mode protocol unified.** The three v0.1.4 endpoints
  (`GET /api/sync`, `POST /api/ingest`, `POST /api/correct`) are
  removed and replaced by a single `POST /api/sync` that carries the
  library's buffered logs + corrections + per-namespace local versions
  in one request, and returns deltas in one response. v0.1.4 clients
  cannot talk to v0.1.5 servers; pin matching versions for now.
- `MicroResolve.correct(...)` no longer makes an immediate HTTP call.
  The correction is applied locally on the spot and shipped to the
  server on the next sync tick. The server reconciles within
  `tick_interval_secs`. Eliminates the per-correction stampede risk
  at scale and gives clean eventual-consistency semantics.
- v0.1 is pre-stable; expect more breaking protocol changes before 1.0.

---

## [0.1.4.1] — 2026-04-30

### Changed

- **Studio binary now ships with UI embedded.** The downloaded GitHub
  Release tarball is a single executable — no sibling `ui/` directory
  required, no `npm run dev` step. Implementation: `rust-embed` with
  compression bakes `ui/dist/` into the binary at compile time; the
  `server` feature now implies the new `bundled-ui` feature so building
  the studio is `cargo build --release --features server --bin
  microresolve-studio` (single flag). Build prerequisite from source:
  `cd ui && npm run build` once before cargo build, so `ui/dist/` exists.
- Crate description latency claim updated `~30μs` → `~50μs` to match
  the v0.1.4 README.

### Note

- Library users (PyPI / npm / crates.io) are unaffected — `bundled-ui`
  only activates when the `server` feature is enabled, which embedded
  bindings never enable.

---

## [0.1.4] — 2026-04-30

### Added

- **Connect-mode examples** — `examples/connected.rs` (Rust) and
  `python/examples/connected.py` (Python): minimal end-to-end demo of
  a library instance subscribing to a running Studio, resolving
  locally, pushing a correction, and watching the version bump on the
  next sync tick. This loop — Studio writes, library subscribes, live
  sync — is the headline feature of this release.
- **Prebuilt server binaries** via GitHub Releases: linux-gnu x86\_64 +
  aarch64, linux-musl x86\_64, darwin x86\_64 + aarch64,
  windows-msvc x86\_64. Install: download the tarball for your platform
  and put `microresolve-studio` on `$PATH`.

### Changed

- **Binary renamed** — `server` → `microresolve-studio` in Cargo.toml;
  the old `server` bin name is removed. Update any scripts or service
  files that reference the old name.
- **TCP bind failure** — server now exits cleanly with a descriptive
  error when the port is already in use; previously it panicked.
- **README** — hero rewritten to "learnable reflex layer for LLM apps";
  added "In the box" block (Studio / Library / online learning / native
  imports / multilingual / multi-namespace); Quick Start moved above
  the comparison table; v0.1 disclaimer condensed to one line; median
  resolve latency updated to 50µs.

---

## [0.1.3] — 2026-04-28

### Breaking

- **`Engine` renamed to `MicroResolve`** across all bindings (Rust, Python, Node.js).
  Update all imports and call sites:
  - Rust: `Engine` → `MicroResolve`, `EngineConfig` → `MicroResolveConfig`
  - Python: `from microresolve import Engine` → `from microresolve import MicroResolve`
  - Node.js: `const { Engine } = require('microresolve')` → `const { MicroResolve } = require('microresolve')`

---

## [0.1.0] — TBD

Initial public release. **MicroResolve** is a pre-LLM reflex layer for
intent classification, safety filtering, and tool selection — sub-millisecond,
CPU-only, with continuous learning from corrections.

### Library API

- **`MicroResolve` + `NamespaceHandle`** — multi-namespace decision engine. One
  instance per application can run several classifiers in parallel: security
  / mood / MCP tool selection / intent. The single public entry point.
  (Previously named `Engine`; renamed in v0.1.3.)
- **`MicroResolveConfig`** with cascade: `data_dir`, `default_threshold`,
  `languages`, optional LLM config, optional server config for connected
  mode. (Previously named `EngineConfig`; renamed in v0.1.3.)
- **`NamespaceConfig`** — per-namespace overrides for threshold,
  languages, LLM model.
- **Connected mode** — set `MicroResolveConfig.server` to a `ServerConfig` and
  the engine pulls subscribed namespaces from a server on startup,
  spawns a single background sync thread, buffers query logs for
  shipping, and pushes corrections inline. Replaces the legacy
  `AppRouter` surface.
- **Typed errors** — `Error::IntentNotFound`, `Io`, `Parse`,
  `Persistence`, `Connect`. No `Result<_, String>` in the public API.
- **`add_intent`** accepts `&[&str]` (English) or
  `HashMap<lang, Vec<phrase>>` (multilingual). Returns
  `Result<usize, Error>` (count of phrases indexed).
- **`update_intent` / `update_namespace`** patch metadata fields with
  `IntentEdit` / `NamespaceEdit` structs.
- **`correct(query, wrong, right)`** — one call moves a phrase between
  intents and reinforces; in connected mode, also pushes to the server.

### Bindings

- **Rust crate** `microresolve` on crates.io. Default features include
  connected mode (`reqwest`); embedded users can opt out via
  `default-features = false`.
- **Python** `microresolve` on PyPI — full `MicroResolve` + `Namespace` API
  via PyO3 / maturin. Mono and multilingual `add_intent`, `resolve`,
  `correct`, `intent`, `update_intent`, `add_phrase`, `version`,
  `flush`. Connected mode supported.
- **Node.js** `microresolve` on npm — `MicroResolve` + `Namespace` API via
  napi-rs. Connected mode supported.

### Server

- **HTTP API** (`microresolve-server`, `--features server`) wraps the
  same `MicroResolve` instance for multi-tenant deployments.
  - `/api/route_multi` — classify a query.
  - `/api/intents`, `/api/namespaces`, `/api/domains` — CRUD.
  - `/api/import/openapi/*`, `/api/import/mcp/*` — bulk-import tools.
  - `/api/sync`, `/api/ingest`, `/api/correct` — connected-mode endpoints
    for SDK clients.
  - `/api/auth/keys` — API key management for connected-mode auth.
- **Background auto-learn worker** — ingested query logs are
  LLM-judged in batch; corrections applied to the live namespace.
- **API key auth** — `X-Api-Key` middleware; keys stored at
  `~/.config/microresolve/keys.json` (separate from data dir, never
  git-tracked).

### Git-versioned training data

- The `--data` directory is automatically a git repo. Every namespace
  mutation auto-commits with a meaningful message.
- `GET /api/namespaces/{id}/history?limit=N` — list commits scoped to
  a namespace.
- `GET /api/namespaces/{id}/diff?from=&to=` — semantic diff between two
  commits: intents added/removed, phrases added/removed, metadata
  changes, weight-update count. Filters out raw L1/L2 weight noise.
- `POST /api/namespaces/{id}/rollback` — `git reset --hard` + reload
  affected resolvers from disk.
- `GET /api/settings/git` / `PUT /api/settings/git` — runtime configure
  a `git remote origin` so each commit pushes to a real GitHub/GitLab
  repo. Auth uses whatever git is configured with on the server.
- `POST /api/git/push` — manual push trigger.

### Studio UI

- React + Vite single-page app served by the binary or run via
  `vite dev` against an external server.
- **Namespaces, Intents, Domains, Layers, Languages, Models** management
  pages.
- **Studio (Resolve / Simulate / Review)** — live classification, batch
  simulation, query review queue, manual + auto-learn corrections.
- **History** — first-class sidebar entry. List-sidebar + main-pane
  layout. For each selected commit: sha, relative time, message, and
  author; amber warning when rolling back will discard newer commits;
  semantic diff panel showing affected intents, sample phrases under
  each added/removed intent, phrases-added/removed grouped by intent,
  metadata changes side-by-side (before/after), and training-weight
  summary; one-click rollback with confirmation.
- **Settings → Data sync (Git)** — set/clear the git remote URL,
  manual push button, status indicator.
- **Settings → Auth Keys** — generate, list, revoke API keys.
- **Import** — OpenAPI / MCP / LangChain / OpenAI function-tool import
  flows with LLM-assisted seed-phrase generation.

### Internal

- `Resolver` and the `scoring` / `ngram` / `phrase` / `tokenizer` /
  `connect` modules are `#[doc(hidden)]`. Library users see only
  `MicroResolve`, `NamespaceHandle`, and the public types in rustdoc.
- Server bin migrated off direct `Resolver` use to `MicroResolve` API
  internally — single source of truth for namespace state.
- Cargo.toml `exclude` list keeps the published crate at ~174 KB
  (only `src/`, `examples/*.rs`, `tests/*.rs`, `languages/*.json`,
  `LICENSE-*`, `README.md`, `CHANGELOG.md`).
- License: dual MIT / Apache 2.0 across all artifacts (root crate,
  Python wheel, Node tarball, UI bundle, docs site).

### Benchmarks

Benchmark numbers will be added to this section after the v0.1.0
benchmark sweep runs. End-to-end smoke against Llama 3.3 / Groq
verifies multi-namespace MCP import, cross-namespace isolation,
manual corrections, and the auto-learn loop in ~75 seconds.

### Known limitations

- Server bin still uses `pub fn with_resolver(_mut)` closure escape
  hatches (marked `#[doc(hidden)]`). True `pub(crate) struct Resolver`
  is a v0.2 cleanup that doesn't change behaviour.
- Node binding is missing `resolve_with`, `intent`, `update_intent`,
  `add_phrase` methods that Python has (v0.2).
- History UI shows commit list + diff but no commit-graph visualisation
  or filter chips.
- No CLA — contributions are accepted under the dual MIT/Apache license.

---

[0.1.0]: https://github.com/gladius/microresolve/releases/tag/v0.1.0
