# ASV Router — Implementation Plan

> Updated 2026-04-02. Simplified from brainstorming — cut over-engineering, focused on what to build.

---

## 1. Architecture

**One Rust core, multiple distribution paths.** No reimplementation. Each binding is ~100-200 lines of glue calling the same `Router` struct.

**Two layers, clean split:**

| Layer | Code | LLM? | Purpose |
|-------|------|------|---------|
| **Core library** | `src/lib.rs` + modules | **No** | Route, learn, correct, export/import. Pure math. No network, no API keys. |
| **Server binary** | `src/bin/server.rs` | **Yes** | Wraps core library + LLM integration (`call_anthropic()`), Training Arena, seed generation, review endpoints. |

**Client bindings (Python, Node.js, WASM) wrap the core library only.** They never touch server code or LLM APIs. If a user wants LLM-driven training (Training Arena, auto-seeding), they run the server.

**Three usage modes:**

1. **Library standalone** — `pip install asv-router`. Route + `learn()`/`correct()` manually. No LLM. No server. Save/load local files.
2. **Server API** — `docker run asv-router`. Any HTTP client calls the API. Server has the LLM key, does all training. Clients are thin.
3. **Library + Server** — Library routes locally at 30µs. Reports corrections to server via HTTP. Server does LLM training + merges learning. Library pulls updated state periodically.

**State is JSON.** All paths use the same `export_json()`/`import_json()` format. A router trained via the server loads in Python, Node.js, browser, or CLI. Same file, same results. ~1.5MB for 150 intents.

**Sync is simple.** Corrections flow UP (client → server). State flows DOWN (server → client). Server is the learning hub. No custom protocol, no S3, no git integration. Users wrap the file in whatever infrastructure they already have.

---

## 2. Distribution Paths

| Priority | Target | Method | Who it serves | Effort |
|----------|--------|--------|---------------|--------|
| **1** | **Server (HTTP API)** | Polish existing Axum server | Enterprise, any-language teams | ~2-3 days |
| **2** | **Python library** | PyO3 + maturin | AI developers, LangChain | ~2-3 days |
| **3** | **Browser (demo only)** | WASM (existing) + npm packaging | Marketing demos, offline PWAs | ~1 day |
| **4** | **Node.js library** | napi-rs | Server-side JS | ~2 days |
| **5** | **CLI** | Rust binary | Developer tooling | ~1 day |

---

## 3. Server — What Exists vs What to Add

### Already built (src/bin/server.rs, 30+ endpoints)

**Core routing:**
- `POST /api/route` — single intent
- `POST /api/route_multi` — multi-intent with relations + negation
- `GET /api/intents` — list intents

**Intent management:**
- `POST /api/intents` — add intent
- `POST /api/intents/delete` — remove intent
- `POST /api/intents/add_seed` — add seeds
- `POST /api/intents/multilingual` — multilingual intent
- `POST /api/intents/type` — set action/context
- `POST /api/intents/load_defaults` — demo intents

**Learning:**
- `POST /api/learn` — teach phrase → intent
- `POST /api/correct` — move signal wrong → right

**State:**
- `GET /api/export` — full state as JSON
- `POST /api/import` — full state overwrite
- `POST /api/reset` — clear all

**Metadata, LLM integration, Training Arena, Simulation, Logs:**
- 15+ additional endpoints already built

### What to add

**Multi-app support:**
- `X-App-ID` header on every request — isolates routers per app
- No header = `default` app (backwards compatible, single-user/demo)
- `default` app always exists, created on first startup
- Apps must be explicitly registered — unknown `X-App-ID` returns 404
- Server state: `HashMap<String, Router>` — one Router per app
- Storage: one file per app in data directory

```
POST /api/apps  {"app_id": "support-bot"}   → creates data/support-bot.json
GET  /api/apps                               → ["default", "support-bot"]
DELETE /api/apps  {"app_id": "support-bot"}  → removes app + file

X-App-ID: support-bot     → data/support-bot.json
X-App-ID: sales-agent     → 404 (not registered)
(no header)                → data/default.json
```

**Sync endpoints (2 new):**
- `GET /api/state/version` — current version number (client polls to check for updates)
- `POST /api/feedback` — client reports correction (server learns + optionally runs LLM training)

Existing endpoints already handle the rest:
- `GET /api/export` — client pulls full state (already exists)
- `POST /api/learn` — server learns phrase → intent (already exists)
- `POST /api/correct` — server corrects wrong → right (already exists)

All scoped by `X-App-ID` — each app has independent state, sync, versioning.

**Note:** `export_learned_only()` / `import_learned_merge()` remain in the core library for advanced use cases (library-to-library CRDT merge, offline sync). But in the standard client-server flow, clients don't learn locally — they report to the server.

**Auto-persistence:**
- `--data <dir>` CLI flag — directory for per-app state files
- Auto-load all `*.json` from data dir on startup
- Auto-save on shutdown (graceful SIGTERM)
- Auto-save after every N mutations (configurable, default: 10)

**Deployment:**
- Dockerfile (multi-stage, minimal image)
- docker-compose.yml (server + optional UI)

**Documentation:**
- Quick-start with curl examples
- OpenAPI spec for the API

**Optional (later):**
- API key authentication (`X-API-Key` header, per-app)
- `POST /api/route_batch` — batch routing
- `GET /api/metrics` — Prometheus-compatible
- `--read-only` mode
- `GET /api/apps` — list all app IDs

---

## 4. Sync Design

### Philosophy

ASV provides sync **primitives**, not sync **infrastructure**.

The server saves a local file. Clients talk to the server via HTTP. That's it. If users want git versioning, S3 backup, Kafka streaming — they wrap the file or HTTP calls in their own infrastructure. ASV doesn't know or care.

### What exists in the core library (already built, no LLM)

```rust
// Routing
router.route(query)               // single intent, 30µs
router.route_multi(query)         // multi-intent with relations + negation

// Online learning (pure math, no LLM)
router.learn(phrase, intent)      // teach phrase → intent
router.correct(phrase, wrong, right) // move signal wrong → right

// State management
router.export_json()              // full state → String (1.5MB)
router.import_json(json)          // full state ← String (overwrite)
router.export_learned_only()      // learned delta → String (~10KB)
router.import_learned_merge(json) // learned delta ← String (CRDT merge)
router.merge_learned(&other)      // merge another Router's learned weights
router.version()                  // monotonic mutation counter
```

### What exists in the server only (LLM-dependent)

```
call_anthropic()                  // LLM API calls (reads ANTHROPIC_API_KEY)
Training Arena endpoints          // LLM-driven seed generation + review
Auto-seeding from corrections     // LLM generates related phrases
Intent review / quality checks    // LLM validates intent configurations
```

**This is the boundary.** Bindings wrap the core library. Server wraps core library + LLM code. Client libraries never need an API key.

### How sync works (one server, many clients)

```
Library clients                        Server (has LLM key)
──────────────                        ──────────────────────
Route locally at 30µs                 Holds truth per app
Do NOT learn locally                  Does ALL learning (learn/correct + LLM)
                                      Saves to data/{app-id}.json

  1. Client reports correction:
     POST /api/correct                 Server learns from correction
     X-App-ID: support-bot    ──────►  Server runs LLM training if needed
     body: {query, wrong, right}       Server auto-saves

  2. Client pulls latest state:
     GET /api/export                   Server returns full state
     X-App-ID: support-bot    ◄──────  (includes all learning from all clients)
     response: full JSON

  3. Client checks if update needed:
     GET /api/state/version            Server returns version number
     X-App-ID: support-bot    ◄──────  Client compares to local version
```

**Key: corrections flow UP, state flows DOWN. Server is the single learning hub.**

### How sync works (library only, no server)

```python
# Save state to file
with open("router.json", "w") as f:
    f.write(router.export_json())

# Load state from file
with open("router.json") as f:
    router = Router.import_json(f.read())
```

If the user wants to sync multiple library instances without a server, they save/load files. If they want versioning, they put the file in a git repo. If they want remote storage, they copy it to S3. None of that is ASV's job.

### What we do NOT build

- S3 integration
- Git integration
- WebSocket streaming
- Event replay protocol
- Sync scheduling / cron
- Custom transport adapters
- Server-to-server sync
- Conflict resolution UI

---

## 5. Python Binding (PyO3)

### Why PyO3
- Same `pyo3` crate we already use (but in "extension-module" direction: Python calls Rust)
- Native speed, zero-copy, native Python objects
- Not WASM-in-Python (2-3x slower, awkward API)
- Not C FFI (manual memory management pain)

### Structure

```
bindings/python/
  Cargo.toml        ← crate-type = ["cdylib"], depends on asv-router
  pyproject.toml    ← maturin build config
  src/lib.rs        ← PyO3 wrapper (~150 lines)
  tests/test_router.py
```

### API surface

```python
from asv_router import Router

r = Router()
r.add_intent("cancel", ["cancel my order", "stop my order"])

result = r.route("cancel my order")         # single intent
results = r.route_multi("cancel and track")  # multi-intent

r.learn("stop that", "cancel")              # online learning
r.save("router.json")                       # local file
r.load("router.json")                       # local file
```

### Build

```bash
cd bindings/python
maturin develop          # install into venv
maturin build --release  # build wheel
maturin publish          # upload to PyPI
```

---

## 6. Node.js Binding (napi-rs)

### Why napi-rs, not WASM
- Native `.node` addon, full speed
- WASM in Node.js is ~1.5-2x slower with serialization overhead
- napi-rs auto-builds for linux-x64, darwin-arm64, win32-x64

### Structure

```
bindings/node/
  Cargo.toml        ← napi-rs, depends on asv-router
  package.json
  src/lib.rs        ← napi-rs wrapper (~150 lines)
  __tests__/router.test.js
```

---

## 7. WASM (Browser) — Demo Only

Already exists in `src/wasm.rs`. **Not for production.**

**Why:** WASM ships the full router state (~1.5MB) to the browser. Anyone can inspect devtools and extract all intent vectors, learned weights, seed phrases — your entire routing logic. That's an IP and security risk.

**Production browsers should call the server API** (`POST /api/route`). The 30µs→~50ms latency difference is invisible to users.

**WASM is only for:**
- Live demo on website ("try ASV in the browser, no signup")
- Offline PWAs where there's no server
- Marketing / conference demos

Move to `bindings/wasm/` later for clean separation. Add `package.json` + wasm-pack config for npm publishing.

---

## 8. CLI

New binary at `src/bin/cli.rs`. Lowest priority.

```bash
asv-router route "cancel my order and track my package"
asv-router learn "stop that" cancel
asv-router --state router.json route "..."
```

---

## 9. Folder Structure

```
asv/
  Cargo.toml              ← add [workspace] for bindings
  Dockerfile              ← NEW
  docker-compose.yml      ← NEW
  src/
    lib.rs                ← core (untouched)
    bin/
      server.rs           ← polish: auto-save, sync endpoints
      cli.rs              ← NEW
  bindings/
    python/               ← PyO3
    node/                 ← napi-rs
    wasm/                 ← move from src/wasm.rs
  tests/                  ← untouched
  ui/                     ← untouched
```

---

## 10. Binary Sizes

| Target | Size |
|--------|------|
| Server | ~4-6 MB |
| Python wheel | ~2-3 MB |
| Node.js addon | ~3-5 MB |
| WASM | ~300-500 KB |
| CLI | ~2-4 MB |

No model files. No data downloads. No runtime dependencies.

---

## 11. Enterprise Deployment

### Simple (one server, multiple apps)

```
Server (port 3001)
  ├── data/
  │   ├── support-bot.json      ← X-App-ID: support-bot
  │   ├── sales-agent.json      ← X-App-ID: sales-agent
  │   └── default.json          ← no header
  ├── auto-loads all on startup
  ├── auto-saves on shutdown + every N mutations
  └── all clients hit same endpoints, scoped by X-App-ID
```

`docker run -v ./data:/data -p 3001:3001 asv-router --data /data`

### With edge clients (library + server)

```
Server (central learning hub, has LLM key)
  ├── each app trains independently
  ├── clients report corrections: POST /api/correct + X-App-ID
  ├── server learns + runs LLM training (Training Arena, auto-seeding)
  ├── clients pull latest state: GET /api/export + X-App-ID
  └── server auto-saves per app
```

### Scaling (user's infrastructure)

- Want versioning? Put `data/` dir in a git repo.
- Want backup? Copy `data/` to S3.
- Want multi-server? Shared volume mount for `data/`. CRDT merge handles divergence.

None of this is ASV code.

---

## 12. Open Questions

1. **Package naming:** `asv-router` everywhere? `asv_router` for Python?
2. **Monorepo vs multi-repo:** Current plan: monorepo with workspace.
3. **CI/CD:** GitHub Actions for cross-platform builds.
4. **LangChain integration:** Separate package or built into Python binding?
5. **App auth:** API key per app? Or defer to reverse proxy (nginx)?
6. **App limits:** Max apps per server? Max state size per app?

---

*Decisions made:*
- *Core library = pure routing + learning. No LLM, no network, no API keys.*
- *Server binary = core library + LLM integration. Only place LLM code lives.*
- *Client bindings wrap core library only. Never need API keys.*
- *Corrections flow UP (client → server), state flows DOWN (server → client)*
- *Server is the single learning hub — clients don't learn locally in connected mode*
- *Multi-app via `X-App-ID` header, default app for no-header requests*
- *No S3, no git, no WebSocket, no event replay, no server-to-server sync*
- *Server saves per-app JSON files in a data directory*
- *Library standalone reads/writes local files, learns locally*
- *Everything else is the user's infrastructure*
