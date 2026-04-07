# ASV Architecture & Implementation Plan

## Two Modes

### Local Mode (standalone)
- `Router(data_path="./asv.json")`
- Full control: add intents, remove intents, add seeds, learn, correct
- Saves to local JSON file
- No network, no server, no dependencies
- User manages everything

### Connected Mode (server URL configured)
- `Router(server="http://central:3001", app_id="support-bot")`
- READ-ONLY routing node
- Pulls intent data from server on startup
- Periodically pulls updates (server may have new intents/seeds)
- Routes locally at full speed (30μs)
- NO local writes: no add_intent, no learn, no correct, no add_seed
- All intent management happens on the server via UI or API
- Client is a pure routing engine

### Why no writes in connected mode?
- Server is single source of truth — no conflicts, no merge, no CRDT
- Privacy: queries stay on the client, never sent to server
- Simplicity: sync is one-directional pull, nothing to coordinate
- LLM features (seed generation, review) live on server where the API key is

---

## Components

### 1. Library (`asv`)
**What it does:** Routes queries to intents. Fast. Local.

**Local mode API:**
```
Router::new(config)           → create empty router
Router::load(path)            → load from JSON file
router.save(path)             → save to JSON file
router.add_intent(id, seeds)  → add intent
router.remove_intent(id)      → remove intent  
router.learn(query, intent)   → learn from correction
router.correct(query, wrong, right) → fix misroute
router.route(query)           → route, returns matches
router.route_multi(query)     → multi-intent routing
router.export_json()          → serialize to string
Router::import_json(string)   → deserialize from string
```

**Connected mode API:**
```
Router::connect(server_url, app_id) → pull state from server
router.pull()                       → refresh from server
router.route(query)                 → route locally
router.route_multi(query)           → multi-intent locally

# These are BLOCKED in connected mode (return error):
router.add_intent()    → Error: "managed by server"
router.learn()         → Error: "managed by server"  
router.correct()       → Error: "managed by server"
router.save()          → Error: "managed by server"
```

**Single package. Mode determined by constructor args.**

### 2. Server (`asv-server`)
**What it does:** Centralized intent management + UI.

- Multi-app support (X-App-ID)
- Full CRUD: intents, seeds, metadata
- LLM features: seed generation, routing review, training
- Auto-persistence to disk
- Discovery: upload queries, get clusters, apply as intents
- Serves config to connected library instances via existing export API

**Endpoints used by connected clients:**
```
GET /api/export          → full router state as JSON (client pulls this)
GET /api/health          → check server is alive
```
That's it. Connected clients only need these two endpoints.
Everything else (add_intent, learn, review, discover) is for the UI/admin.

### 3. UI (React dashboard)
**What it does:** Visual management of intents for the server.

Pages:
- Router: test queries interactively
- Intents: add/edit/delete intents and seeds
- Discovery: upload queries, auto-discover clusters
- Projections: visualize intent relationships
- Training: LLM-powered scenario testing
- Debug: query logs
- Settings: mode, app selector

---

## Package Structure (single repo)

```
asv/
├── src/
│   ├── lib.rs          ← core library (routing, learning, export/import)
│   ├── connected.rs    ← connected mode (pull from server, block writes)  [NEW]
│   ├── discovery.rs    ← auto-discovery
│   ├── index.rs        ← inverted index
│   ├── multi.rs        ← multi-intent decomposition
│   ├── tokenizer.rs    ← tokenization
│   ├── vector.rs       ← sparse vectors
│   ├── seed.rs         ← LLM prompt building
│   └── bin/
│       └── server.rs   ← HTTP server
├── python/             ← PyO3 bindings
├── node/               ← napi-rs bindings
├── ui/                 ← React dashboard
├── examples/           ← per-language examples
├── tests/
└── docs/
```

---

## Sync Protocol (connected mode)

### On startup:
1. Client calls `GET /api/export` with `X-App-ID` header
2. Server returns full router JSON
3. Client deserializes into local Router instance
4. Client is ready to route

### Periodic refresh:
1. Every N seconds (configurable, default 30s):
   - Client calls `GET /api/export`
   - If server version > local version → replace local state
   - Routing continues uninterrupted during refresh (read lock / swap)

### Server pushes new intent/seed:
1. Admin adds intent via UI
2. Server state updated immediately
3. Connected clients pick it up on next pull (within 30s)

### No push from client to server. Ever.
- Client never sends queries, never sends learned data
- Client only pulls config
- Server only serves config

---

## What's Built vs What's New

| Component | Status |
|-----------|--------|
| Core routing library | DONE |
| Local mode (file load/save) | DONE (export_json/import_json) |
| Server with UI | DONE |
| Multi-app | DONE |
| Discovery | DONE |
| Python bindings | DONE |
| Node bindings | DONE |
| Connected mode (pull from server) | NEW — needs connected.rs |
| Block writes in connected mode | NEW — guard in Router methods |
| Periodic pull with version check | NEW — background thread/timer |
| RouterConfig struct | NEW — constructor with settings |

---

## Open Questions

1. Should connected mode pull on a timer (background thread) or only on explicit `router.pull()` call?
   - Timer: automatic, but adds threading complexity in Python/Node
   - Explicit: simpler, user controls when to refresh
   - Compromise: pull on startup + explicit pull() + optional timer

2. Should the server have a version endpoint (`GET /api/version`) so clients can check cheaply without pulling full state?
   - Saves bandwidth if nothing changed
   - Simple: return `{"version": 42}`, client compares with local version

3. File format: current export_json() works but is it the right format for config files?
   - Maybe add a simpler human-readable format for local mode?
   - Or keep JSON and rely on the UI for human-friendly editing?
