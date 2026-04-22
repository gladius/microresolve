# Pre-Launch Review — Three Architectural Concerns

Completed review of persistence, API surface, and bindings parity on 2026-04-22.
Use this as a working checklist — mark items as **DONE** / **SKIP** / **DEFERRED** as we work through them.

---

## 1. Persistence — loaded/saved vs in-memory

### What IS persisted on disk (`save_to_dir`)

| File (per namespace) | Contents | When written |
|----------------------|----------|--------------|
| `_ns.json` | name, description, models, default_threshold | Every namespace mutation |
| `_entities.json` | enabled built-in labels + custom entities | Every entity config change |
| `_l1.json` | full LexicalGraph (morphology + synonyms, **including learned edges**) | Every phrase add/remove |
| `_l2.json` | full IntentIndex (word→intent weights **including all learning**) | Every phrase add/remove |
| `<intent>.json` | description, type, phrases, instructions, persona, guardrails, source, target, schema | Every intent mutation |
| `<domain>/_domain.json` | domain description | Every domain mutation |

### What is NOT persisted (deliberately — rebuilt on load)

| Field | Rebuild mechanism |
|-------|-------------------|
| `l0` (NgramIndex) | Rebuilt from L1+L2 vocabulary on load |
| `cached_entity_layer` | Rebuilt from `entity_config` via `rebuild_entity_cache()` |
| `idf_cache` (in L2) | Rebuilt via `rebuild_idf()` on load |
| `known_intents` (in L2) | Rebuilt from posting lists |

### What is NOT persisted AND is a concern

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 1.1 | `similarity` HashMap on Router — dead code, written only in clone/restore, never read for routing | Low (just noise) | ✅ **DONE** (commit 3ccf36a) — removed field, struct init, export/import paths |
| 1.2 | `version` counter resets to 0 on each load (because it's not serialized) | Medium — connected-mode clients tracking `version` would refetch after every server restart | **[ ] Either persist `version` or document the reset behavior** |
| 1.3 | Log store persistence location unclear — does `DELETE /api/namespaces` cleanly remove logs too? | Medium if enterprise privacy matters | **[ ] Audit log-store cleanup on namespace delete** |

### When does the server actually save?

`maybe_persist()` is called from every mutation route (phrase add, intent add, entity config update, namespace patch, etc.) plus the auto-learn worker. **Routing itself never modifies state.** All learning paths flow through phrase-add which auto-persists — no "in-memory only learning" bug.

### Verdict

✅ **Persistence model is solid for the routing/PII/entity story.** One real cleanup (`similarity`), one minor documentation/persistence decision (`version`), one audit item (log cleanup on namespace delete). No data loss risk for users.

---

## 2. API surface review — what's used, what's noise, library vs server

**80 HTTP endpoints total.** Sorted by status:

### Keep — core product

- `POST /api/route_multi`
- `POST /api/entities/{detect,extract,mask,distill}`
- `GET/PATCH /api/entities`, `POST/DELETE /api/entities/custom`, `GET /api/entities/builtin`
- `POST /api/intents` (+ description, instructions, persona, guardrails, target, type, delete, multilingual)
- `POST/PATCH/DELETE /api/namespaces`
- `POST /api/intents/phrase`, `POST /api/intents/phrase/remove`
- `GET /api/intents`, `GET/POST/DELETE /api/domains`

### Keep — operational

- `GET /api/health`, `GET /api/version`, `GET /api/events` (SSE), `GET /api/llm/status`
- `GET/POST /api/settings`, `GET/POST /api/stopwords`

### Keep — import/onboarding

- `POST /api/import/{spec,parse,apply}`
- `POST /api/import/mcp/{search,fetch,parse,apply}`, `GET /api/import/params`

### Keep — continuous learning loop

- `GET /api/review/queue`, `POST /api/review/{analyze,fix,reject,intent_phrases}`
- `POST /api/learn/{now,words}`
- `POST /api/training/{generate,run,review,apply}`
- `POST /api/simulate/{turn,respond}`, `GET /api/simulate/history`

### Keep — layer inspection (for the UI and transparency)

- `GET /api/layers/info`
- `GET/POST/DELETE /api/layers/l1/edges`
- `POST /api/layers/l1/distill`, `POST /api/layers/l2/probe`

### Keep — logs / observability

- `GET /api/logs`, `GET /api/logs/stats`
- `POST /api/ingest`, `POST /api/report`

### Keep — connected mode

- `GET /api/sync`

### Candidates to remove or hide

| # | Endpoint | Issue | Action |
|---|----------|-------|--------|
| 2.1 | `POST /api/reset` | Was UI-callable but only by the now-removed Reset button | ✅ **DONE** (commit 834b958) — endpoint and UI button removed |
| 2.2 | `DELETE /api/data/all` | "Clear All Data" button keeps it; wipes everything by design | ✅ **KEPT INTENTIONALLY** — typed-confirmation modal is sufficient guard for self-hosted dev tool convention |
| 2.3 | `POST /api/intents/load_defaults` | 364 lines of hardcoded e-commerce demo intents from before the current product story | ✅ **DONE** (commit 834b958) — endpoint, handler, and 'Reset to Defaults' button removed |
| 2.4 | `POST /api/phrase/{parse,prompt}` | UI never called them; server handlers were dead. `/api/phrase/generate` is kept (used by IntentsPage). | ✅ **DONE** (commit f123b26) — removed 2 server routes + handlers + UI client funcs |
| 2.5 | `GET /api/export`, `POST /api/import` | Whole-state dump/restore. Useful for backup but also a data-exfiltration risk without auth. | **[ ] Document risk OR gate behind auth** |
| 2.6 | `GET /api/ns/models` | Per-namespace model registry lookup. Unclear if UI uses it. | **[ ] Confirm usage; delete if unused** |

### Library vs server boundary — what's missing from the library

Currently exposed in Rust lib core: intent CRUD, routing, correction, description, phrase CRUD, export/import JSON.

**Missing from the Rust library public API (or missing from Python/Node bindings):**

| # | Feature | In Rust core? | Python binding? | Node binding? | WASM? |
|---|---------|---------------|-----------------|---------------|-------|
| 2.7 | Entity layer (detect/extract/mask/augment) | Yes (`EntityLayer::detect`...) | **No** | **No** | **No** |
| 2.8 | Per-namespace entity config | Yes | **No** | **No** | No |
| 2.9 | Threshold cascade | Yes (`resolve_threshold`) | **No** | **No** | No |
| 2.10 | Persistence (`save_to_dir`, `load_from_dir`) | Yes | **No** | **No** | No |
| 2.11 | Built-in pattern introspection (list patterns, categories) | Yes | **No** | **No** | No |
| 2.12 | LLM-mediated features (distill, training/generate) | No (server-only; needs API key + async runtime) | N/A | N/A | N/A |

**Legitimately server-only** (should stay): HTTP routes, LLM API calls, SSE, background worker, log_store (though could be exposed later), MCP/OpenAPI network imports.

### Verdict

⚠️ **~6 endpoints need review/cleanup.** The bigger issue is that **entity layer, threshold cascade, and persistence are missing from Python and Node bindings** — the launch story says "embedded library" but the v1.0 headline features aren't embeddable yet.

---

## 3. Bindings parity — Rust vs Python vs Node vs WASM

### Method-by-method comparison

| Method | Rust core | Python | Node | WASM |
|--------|-----------|--------|------|------|
| `new` | ✓ | ✓ | ✓ | ✓ |
| `add_intent` | ✓ | ✓ | ✓ | ✓ |
| `add_intent_multilingual` | ✓ | ✓ | ✗ | ✓ |
| `resolve` / `route` | ✓ | ✓ | ✓ | ✗ (different shape) |
| `add_phrase` | ✓ | ✓ | ✓ | ✓ |
| `remove_phrase` | ✓ | ✓ | ✓ | ✗ |
| `delete_intent` / `remove_intent` | ✓ | ✓ | ✓ | ✓ |
| `intent_ids` | ✓ | ✓ | ✓ | ✗ |
| `set_intent_type` / `get_intent_type` | ✓ | ✓ (set) | ✓ (set) | ✓ |
| `set/get_description` | ✓ | ✓ | ✓ | ✗ |
| `set/get_instructions/persona` | ✓ | ✗ | ✗ | ✓ |
| `correct` | ✓ | ✓ | ✓ | ✓ |
| `export_json` / `import_json` | ✓ | ✓ | ✓ | ✓ |
| `check_phrase` | ✓ | ✓ | ✗ | ✗ |
| **Entity ops** (detect/extract/mask) | ✓ | **✗** | **✗** | **✗** |
| **Per-namespace threshold** | ✓ | **✗** | **✗** | **✗** |
| **Per-namespace entity config** | ✓ | **✗** | **✗** | **✗** |
| **`save_to_dir` / `load_from_dir`** | ✓ | **✗** | **✗** | **✗** |

### Action items

| # | Finding | Action |
|---|---------|--------|
| 3.1 | Python binding missing entity layer + threshold cascade + persistence | **[ ] Add `EntityLayer` / `detect` / `extract` / `mask` / `augment`, `set_namespace_default_threshold`, `save_to_dir` / `load_from_dir` to Python binding** |
| 3.2 | Node binding missing same features | **[ ] Add same to Node binding** |
| 3.3 | Node missing `add_intent_multilingual`, `check_phrase` (parity gaps with Python) | **[ ] Add for Python↔Node parity** |
| 3.4 | Python missing `set/get_instructions/persona` (WASM has them) | **[ ] Add to Python** |
| 3.5 | WASM missing `resolve` — has old API shape | **[ ] Either add modern resolve shape OR document WASM as intentionally minimal** |
| 3.6 | WASM as a target — keep or shrink? | **[ ] Decide: keep for browser/edge demos with minimal API OR invest to match Python/Node** |

### Verdict

❌ **The launch story ("drop into Python/Node/Rust/WASM as an embedded library") is not true for the v1.0 features.** Bindings are stuck at "v0.5" — they expose basic intent CRUD but none of the entity layer, threshold, or persistence work. This is the biggest gap.

---

## Priority order for work

### Must-fix before launch
- **[ ] 3.1, 3.2** — Python + Node bindings: add entity layer + threshold + persistence. ~3-4 hours.

### Should-do before launch
- **[ ] 2.1, 2.2** — hide/remove destructive unguarded endpoints (`/api/reset`, `/api/data/all`). ~15 min.
- **✅ 1.1** — remove dead `similarity` field. **DONE** in commit 3ccf36a.
- **[ ] 2.3, 2.4, 2.6** — audit legacy endpoints (load_defaults, phrase/*, ns/models) — remove if dead. ~15 min.

### Nice-to-have
- **[ ] 1.2** — persist `version` counter across restarts OR document. ~10 min if persist.
- **[ ] 1.3** — audit log store cleanup on namespace delete. ~10 min.
- **[ ] 3.3, 3.4** — bindings parity polish.
- **[ ] 3.5, 3.6** — WASM decision + documentation.

### Defer
- **[ ] 2.5** — auth for export/import endpoints (post-launch security work).

---

## How to use this document

Mark each `[ ]` checkbox as we complete items. When a section is fully addressed, add `✅ DONE` at the section header. Keep this file updated as we work through the list.
