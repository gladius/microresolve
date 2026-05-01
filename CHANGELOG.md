# Changelog

All notable changes to MicroResolve are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.9] — 2026-05-01

### Breaking

- **Studio auth model: every `/api/*` route now requires `X-Api-Key`** —
  including the UI's own fetches. The previous setup left UI fetches
  unauthenticated and only `/api/sync` enforced auth. Combined with a
  default bind to `0.0.0.0`, that meant any LAN host could mint keys
  via the unauthenticated `/api/auth/keys` POST. Closed.
- **`POST /api/auth/keys` now requires auth** like everything else.
  No more bootstrap exception.
- **Renamed CLI flag `--no-open` → `--no-browser`**. More
  self-documenting; matches Jupyter's convention.

### Added

- **Auto-bootstrap admin key on first boot.** When `keys.json` is empty,
  the server mints `mr_studio-admin_<hex>` (Admin scope), prints it to
  stdout in a visible banner, and persists it to
  `~/.config/microresolve/admin-key.txt` (mode 0600) so operators who
  redirect stdout (Docker, systemd) can `cat` it later.
- **Studio paste-screen** — first browser visit renders a key entry
  form. Pasted key lives in `localStorage`; every subsequent fetch
  carries `X-Api-Key`. Single global `window.fetch` wrapper (in
  `ui/src/api/client.ts`) attaches it to every relative `/api/*`
  request, including the dozen-plus raw `fetch()` call sites scattered
  across the import wizards and Layout polls.
- **`KeyScope` schema on every API key** — `admin` or `library`.
  Persisted in `keys.json`, returned by `GET /api/auth/keys`, accepted
  by `POST /api/auth/keys`, surfaced in the Auth Keys page (column
  badge + scope dropdown on Generate). **Enforcement is permissive in
  v0.1.9** (every scope grants every route); v0.2 will land
  route-level scope checks. Existing keys without a scope field load
  as `Admin` (backwards-compat default).
- **`--keys-file <path>` CLI flag** for keystore isolation. Used by
  the integration test harness to give every spawned server its own
  `keys.json`, so parallel test binaries don't race on the global
  `~/.config/microresolve/keys.json`.

### Removed

- **Home page "Common shapes" use-case tile section.** Replaced with a
  one-line stats strip (namespaces · intents · connected clients ·
  pending review) and a single "see use cases" link in the empty
  state — exactly where new operators look for examples, with no
  clutter on the persistent home view for repeat visitors.
- **Home page "Try It" widget** (already removed in earlier branch
  work, formalized in v0.1.9). The dedicated `/resolve` page provides
  the same with full L0/L1/L2 trace.
- **"Start with" dropdown in the New Namespace modal.** Three options
  that just navigated to different pages — friction at the moment of
  creation. Now: create + close, namespace appears in the list, user
  picks where to go from there.

### Changed

- **Hero on Studio Home** rewritten from "Pre-LLM intent routing" →
  "The pre-LLM reflex layer" to match the README. Intent routing is
  one use case; reflex layer covers the broader pitch (tool selection,
  guardrail dispatch, refusal classification).
- **Test harness (`tests/common/mod.rs`)** captures the bootstrap
  admin key from each spawned server's `admin-key.txt` and auto-injects
  it via a thread-local on every helper call. Tests stay terse;
  unauthorized-path tests bypass the auto-inject by setting
  `X-Api-Key` explicitly.

## [0.1.8] — 2026-05-01

### Fixed

- **Python wheels**: `python/pyproject.toml` was out of sync with
  `python/Cargo.toml` and `Cargo.toml`, causing maturin to build
  v0.1.7 release artifacts as `microresolve-0.1.6-*.whl`. PyPI's
  `--skip-existing` silently skipped them. v0.1.8 syncs the
  pyproject version and re-publishes the Python wheel. Crates.io
  and npm v0.1.7 are unaffected; tarball release for v0.1.7 is also
  intact, but skipped in favor of v0.1.8 for cross-registry parity.

## [0.1.7] — 2026-05-01

### Added

- **Per-namespace reflex-layer toggles.** Each namespace now carries
  four boolean fields — `l0_enabled`, `l1_morphology`, `l1_synonym`,
  `l1_abbreviation` — accepted by `Resolver::update_namespace` and
  exposed on `NamespaceInfo`. All default to `true`; namespaces saved
  before this change load with all layers on. Flags are honored at
  both index time and resolve time so trained vectors and runtime
  preprocessing stay in sync. Use cases: turn L0 off for medical /
  legal namespaces (auto-correcting domain terms is dangerous); turn
  L1 abbreviations off for code search (short tokens carry literal
  meaning).
- **`PATCH /api/namespaces` body** accepts the four new fields;
  `GET /api/namespaces` returns them per namespace.
- **Studio: layer toggles in three places.** Namespaces edit modal has
  all four switches (atomic save, setup flow); L0 page has an inline
  toggle; L1 page has a top-level `on / off / partial` status badge
  plus a per-column compact toggle for each edge kind. Sidebar L0 / L1
  nav items show an `off` or `partial` pill when any layer is disabled
  for the active namespace.
- **`LayerToggle` shared component** + `AppContext.layerStatus` —
  single optimistic-update store for the active namespace's toggles,
  so changing them on any page updates the sidebar instantly without
  an extra GET. Reverts on PATCH failure.
- **`LexicalGraph::preprocess_with_kinds` /
  `preprocess_grounded_with_kinds`** — public variants of the
  preprocess methods that take three booleans gating Morphological,
  Abbreviation, and Synonym edge kinds. The original `preprocess` /
  `preprocess_grounded` are unchanged thin wrappers — no break to the
  published API.

## [0.1.6] — 2026-05-01

### Added

- **Empty `subscribe` list now auto-subscribes to all namespaces** the
  server exposes. Pass `subscribe=[]` (Python: omit / `None`; Node:
  omit / `[]`) and the library queries `GET /api/namespaces` at connect
  time, then pulls each one. Explicit lists still work as an allow-list
  for multi-tenant cases. Zero-config for solo / single-team setups.
- **Studio shows connected library clients.** A new top-level "LIVE →
  Connected" sidebar item with a live count badge, plus a dedicated
  `/connected` page listing every authenticated client currently
  syncing (name, library version, subscribed namespaces, tick interval,
  last sync, expires-in countdown). Auto-refreshes every 3s.
- **`GET /api/connected_clients`** — read-only roster endpoint backing
  the UI. Lazy GC on read: any entry older than `2 × tick_interval_secs`
  is dropped before responding.
- **`POST /api/sync` body fields** (optional, advisory):
  `tick_interval_secs` lets the server use each client's own freshness
  window; `library_version` (e.g. `microresolve-py/0.1.6`) surfaces in
  the connected-clients panel for "who's still on the old client?"
  triage.

### Changed

- **API key format now embeds the label**: `mr_<name>_<64 hex chars>`.
  The server extracts the name from the key string itself — no separate
  index lookup needed for attribution. Old opaque keys (`mr_<hex>`
  without the embedded name) are no longer accepted; v0.1 is pre-stable
  and operators must regenerate keys via the Studio UI / `/api/auth/keys`.
- Auth-key names must be slug-safe: `[a-z0-9][a-z0-9-]{0,30}` (no
  underscore — that's the field separator).

### Note

- Connected-clients tracking is **only active when API keys are
  configured**. In open mode (no keys), the panel is empty by design —
  there's no identity to attribute connections to.

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
