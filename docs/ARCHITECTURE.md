# MicroResolve Architecture

## Overview

MicroResolve is an intent extraction system with a library (fast local routing) and
a server (central management, review, learning). Both run inside the
company's infrastructure. Nothing leaves the company network.

```
Company's Infrastructure
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  Service A ──→ MicroResolve Library ──→ routes locally (30μs)     │
│  Service B ──→ MicroResolve Library ──→ routes locally (30μs)     │
│  Service C ──→ MicroResolve Library ──→ routes locally (30μs)     │
│       │              │              │                     │
│       └──────────────┼──────────────┘                     │
│                      ↓                                    │
│              MicroResolve Server (central)                         │
│              - Receives ALL queries + results             │
│              - Flags failures (low confidence, miss)      │
│              - LLM auto-learning or human review          │
│              - Pushes updated intents to all libraries    │
│              - Dashboard, analytics, workflows            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Two Modes

### Local Mode (dev/testing/edge)
- `Router::new()` or `Router::load("asv.json")`
- Full control: add intents, seeds, learn, save
- No server, no network
- For: getting started, development, edge/IoT, simple use cases

### Connected Mode (production)
- `Router::with_config(RouterConfig { server: "http://asv-server:3001", app_id: "my-app" })`
- Routes locally at full speed (30μs)
- Sends EVERY query + results to server (full text, full results)
- Server manages intents, reviews failures, pushes updates
- Library pulls updated config periodically

## Failure Detection

The library already computes confidence. A query is flagged as "failed" when:

```
MISS:      route_multi returns empty confirmed list
LOW_CONF:  best score < threshold
AMBIGUOUS: top 2 scores within 10% of each other
```

In connected mode, ALL queries are sent to the server, but flagged queries
are prioritized for review.

```json
POST /api/report
{
  "query": "I ordered a blue jacket and got a red hoodie",
  "session_id": "S003",
  "results": {
    "confirmed": [],
    "candidates": [{"id": "change_order", "score": 0.4}]
  },
  "flag": "miss",           // miss | low_confidence | ambiguous | ok
  "expected_intents": null,  // null until reviewed
  "timestamp": "2026-04-08T12:34:56Z"
}
```

## Three Review Modes (server-side setting)

### Auto-Learn
```
Library flags low confidence query → sends to server
Server calls LLM: "This query was routed to change_order but confidence is low.
  Here are all intents: [cancel_order, return_item, refund, ...].
  What is the correct intent? Generate a seed phrase."
LLM responds: { intent: "return_item", seed: "received wrong item" }
Server adds seed automatically
All library instances get updated on next pull
```
No human involved. Fast. Risk: LLM makes mistakes.

### Auto-Review
```
Same as auto-learn, but LLM prepares the fix
Fix is queued in the UI for human approval
Human clicks "Approve" or "Reject" or edits
Approved fixes are applied, pushed to all instances
```
Human in the loop. Slower but safer.

### Manual
```
Failed queries appear in the Review tab
Human reads the query, decides the correct intent
Human adds seed phrases manually
Changes pushed to all instances
```
Full human control. Slowest but most accurate.

## Server Components

### Endpoints for connected libraries
```
POST /api/report         — library sends query + results + flag
GET  /api/version        — cheap version check (returns version number)
GET  /api/export         — full config pull (when version changed)
```

### Endpoints for UI/admin
```
GET  /api/review/queue   — flagged queries pending review
POST /api/review/approve — approve LLM-suggested fix
POST /api/review/reject  — reject suggestion
POST /api/review/fix     — manually assign intent + seed
GET  /api/review/stats   — how many pending, auto-fixed, rejected
```

### Existing endpoints (unchanged)
```
POST /api/route_multi    — test routing from UI
GET  /api/intents        — list intents
POST /api/intents        — add intent
POST /api/learn          — add seed
GET  /api/co_occurrence  — analytics
GET  /api/projections    — analytics
GET  /api/workflows      — analytics
GET  /api/temporal_order — analytics
GET  /api/escalation_patterns — analytics
POST /api/discover       — auto-discover from uploaded queries
```

## Data Flow

```
1. Customer sends message to Service A
2. Service A calls MicroResolve Library: route_multi("I got the wrong item")
3. Library returns: {confirmed: [], candidates: [{id: "change_order", score: 0.4}]}
   Library flags: "miss" (empty confirmed)
4. Service A uses the result (even if low confidence)
5. Library sends report to MicroResolve Server: {query, results, flag: "miss"}
6. Server queues for review

In auto-learn mode:
7. Server calls LLM: "correct intent for 'I got the wrong item'?"
8. LLM: "return_item" + seed "received wrong item"
9. Server adds seed to return_item
10. Server increments version
11. Next library pull: version changed → pull new config
12. All libraries now route "I got the wrong item" → return_item

In auto-review mode:
7-8. Same as above
9. Fix queued in UI → admin reviews → approves
10-12. Same as above

In manual mode:
7. Admin sees "I got the wrong item" in Review tab
8. Admin assigns to return_item, types seed "received wrong item"
9-11. Same as above
```

## What's Built vs What's New

| Component | Status |
|-----------|--------|
| Core routing library | DONE |
| Local mode (load/save/route/learn) | DONE |
| Server (intents, multi-app, persistence) | DONE |
| Dashboard (co-occurrence, workflows, temporal, escalation) | DONE |
| Python/Node bindings | DONE |
| Discovery | DONE |
| Simulation (20 sessions, 50 turns) | DONE |
| RouterConfig + connected mode constructor | DONE |
| Write guards in connected mode | DONE |
| Version endpoint (GET /api/version) | DONE |
| POST /api/report (library → server reporting) | NEW |
| Review queue (store flagged queries) | NEW |
| Review UI tab (see failures, approve fixes) | NEW |
| LLM auto-review/auto-learn | NEW |
| Periodic config pull in library | NEW |
| Review stats on dashboard | NEW |

## Implementation Priority

1. POST /api/report + review queue storage (server)
2. Review UI tab (see flagged queries, manual fix)
3. LLM auto-review (prepare fixes for approval)
4. LLM auto-learn (auto-apply fixes)
5. Library: send reports in connected mode
6. Library: periodic config pull
