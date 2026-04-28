# Changelog

All notable changes to MicroResolve are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
