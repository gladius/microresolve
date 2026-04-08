# Dashboard Implementation Plan

## Overview

Replace the current Projections page with a full analytics dashboard that surfaces
all emergent intelligence from routing data: co-occurrence, workflows, temporal
ordering, escalation detection, anomaly alerts, and intent suggestions.

All analytics data is already computed by the library. This is purely UI + minor
server additions for anomaly detection. No new library methods needed except
anomaly tracking.

---

## What We Have (backend complete)

| Feature | Library method | Server endpoint | Data source |
|---------|---------------|-----------------|-------------|
| Co-occurrence | `get_co_occurrence()` | `GET /api/co_occurrence` | `record_co_occurrence()` called in `route_multi` |
| Projected Context | Built from co-occurrence | `GET /api/projections` | Same as above |
| Temporal Ordering | `get_temporal_order()` | `GET /api/temporal_order` | `record_intent_sequence()` called in `route_multi` |
| Workflow Discovery | `discover_workflows()` | `GET /api/workflows` | Accumulated intent sequences |
| Escalation Patterns | `detect_escalation_patterns()` | `GET /api/escalation_patterns` | Accumulated intent sequences |
| Intent Suggestions | `suggest_intents()` | None — needs endpoint | Co-occurrence data |
| Simulation | N/A | Uses `route_multi` | 55 hardcoded multi-intent queries |

## What We Need to Build

### 1. Library additions (src/lib.rs)

#### 1a. Intent frequency tracking
```rust
/// Track per-intent routing counts for volume monitoring.
/// Call after each route_multi result.
pub fn record_intent_hit(&mut self, intent_id: &str) {
    // Increment per-intent counter
    // Store timestamp bucket (hourly) for trend detection
}

/// Get intent hit counts.
pub fn get_intent_stats(&self) -> Vec<IntentStat> {
    // Returns: intent_id, total_hits, hits_last_hour, avg_hits_per_hour
}
```

#### 1b. Confidence tracking
```rust
/// Track routing confidence distribution.
pub fn record_confidence(&mut self, confidence: f32) {
    // Bucket into: high (>0.8), medium (0.4-0.8), low (<0.4), miss (0)
    // Store last N values for moving average
}

/// Get confidence distribution.
pub fn get_confidence_stats(&self) -> ConfidenceStats {
    // Returns: total_queries, high_pct, medium_pct, low_pct, miss_pct
}
```

#### 1c. Anomaly detection
```rust
/// Detect anomalies in recent routing data.
pub fn detect_anomalies(&self) -> Vec<Anomaly> {
    // Check:
    // 1. Volume spikes: any intent >3x its average
    // 2. Escalation spikes: escalation pattern frequency > 2x average
    // 3. Confidence drop: low confidence rate > 2x average
    // 4. New pattern: unseen co-occurrence pairs appearing frequently
}

pub struct Anomaly {
    pub anomaly_type: AnomalyType, // VolumeSpike, EscalationSpike, ConfidenceDrop, NewPattern
    pub description: String,
    pub severity: f32,             // 0.0-1.0
    pub intent_id: Option<String>, // which intent is affected
}
```

**Estimated: ~150 lines in lib.rs, ~30 lines for server endpoints.**

### 2. Server additions (src/bin/server.rs)

New endpoints:
```
GET /api/stats          → intent hit counts + confidence distribution
GET /api/anomalies      → detected anomalies
POST /api/record        → record intent hits from connected clients (just intent IDs, no query text)
```

The `/api/record` endpoint is how connected mode clients report back.
It receives ONLY intent IDs — no query text, no PII.

```json
POST /api/record
{
  "intents": ["cancel_order", "order_history"],
  "app_id": "support-bot"
}
```

Server calls `record_co_occurrence()`, `record_intent_sequence()`,
`record_intent_hit()` for each intent. Privacy-safe.

**Estimated: ~60 lines.**

### 3. API client additions (ui/src/api/client.ts)

```typescript
// New API methods
getWorkflows: () => get('/workflows'),
getTemporalOrder: () => get('/temporal_order'),
getEscalationPatterns: () => get('/escalation_patterns'),
getStats: () => get('/stats'),
getAnomalies: () => get('/anomalies'),
```

**Estimated: ~10 lines.**

### 4. Dashboard UI (ui/src/pages/DashboardPage.tsx)

Replaces current ProjectionsPage. Full analytics dashboard with sections:

#### 4a. Overview Strip (top)
```
┌─────────────┬──────────────┬──────────────┬──────────────┐
│ 1,247       │ 82%          │ 12           │ 2 alerts     │
│ observations│ high conf    │ workflows    │              │
└─────────────┴──────────────┴──────────────┴──────────────┘
```
Key metrics at a glance. Pulls from `/api/stats` and `/api/anomalies`.

#### 4b. Anomaly Alerts
```
⚠ cancel_order volume up 3.2x in last hour
⚠ Low confidence rate increased from 5% to 18%
```
Red/amber alert cards. Only shown when anomalies detected. Pulls from `/api/anomalies`.

#### 4c. Projected Context (existing, enhanced)
```
cancel_order → order_history (82%) → check_balance (45%)
refund       → check_balance (71%) → order_history (55%)
```
Enhanced version of current Projections page. Each action shows which context
intents co-occur. Strength bars + counts. Already built, just move here.

#### 4d. Workflow Discovery
```
Workflow 1: cancel_order → order_history → refund → check_balance
            (observed 34 times, 4 intents)

Workflow 2: billing_issue → contact_human → schedule_callback
            (observed 28 times, 3 intents)
```
Connected node visualization. Each workflow is a card showing the sequence
of intents that form a business process. Pulls from `/api/workflows`.

#### 4e. Temporal Flow
```
           ┌─ refund (75%) ──┐
cancel ────┤                 ├── check_balance
           └─ track (25%) ───┘
```
Arrow diagram showing which intents follow which. Direction + probability.
Pulls from `/api/temporal_order`. Rendered as simple HTML/CSS arrows,
not a graph library (no new dependencies).

#### 4f. Escalation Patterns
```
🔴 billing_issue → contact_human → schedule_callback
   12 occurrences (8% of sessions) — potential process gap

🟡 password_reset → contact_human
   8 occurrences (5%) — expected escalation
```
Flagged sequences with severity color. Pulls from `/api/escalation_patterns`.

#### 4g. Simulation (existing, move here)
```
[Run Simulation]  55/55 ████████████████ Complete
```
Same button + progress bar from current Projections page. Runs 55 queries
to populate all analytics data for demo purposes.

**Estimated: ~400-500 lines for the full dashboard page.**

### 5. Navigation update

```
[Playground]  [Intents]  [Dashboard]  [Discovery]  [Debug]  [Settings]
```

"Projections" renamed to "Dashboard". Route stays at `/projections` or
changes to `/dashboard`.

---

## Implementation Order

```
Step 1: Library — add intent stats + confidence tracking + anomaly detection
        Test: unit tests for each new method
        ~150 lines, ~30 min

Step 2: Server — add /api/stats, /api/anomalies, /api/record endpoints
        Test: curl tests
        ~60 lines, ~15 min

Step 3: API client — add new methods
        ~10 lines, ~5 min

Step 4: Dashboard UI — build the page with all sections
        Test: run simulation, verify all sections populate
        ~400 lines, ~1-2 hours

Step 5: Navigation — rename Projections → Dashboard
        ~1 line change
```

Total estimated: ~620 lines, 2-3 hours.

---

## Demo Script

For blog post / video / HN demo:

1. Start server with defaults: `./server --data ./data`
2. Open Dashboard — empty, no data yet
3. Click "Run Simulation" — watch 55 queries flow
4. Watch in real-time:
   - Co-occurrence numbers increase
   - Workflow clusters form
   - Temporal arrows appear
   - Escalation patterns get flagged
5. Point out: "Zero configuration. These patterns emerged purely from routing data."
6. Show anomaly detection: manually send a burst of one intent type → alert appears
7. Show projected context: "When cancel_order fires, the system already knows to preload order_history and check_balance — learned from usage, not programmed."

The "wow" moment: patterns appearing in real-time from a simulation button press,
with zero setup, zero ML, zero configuration.

---

## Data Privacy in Connected Mode

In connected mode, clients send ONLY intent IDs to `/api/record`:
```json
{"intents": ["cancel_order", "order_history"]}
```

NO query text. NO user data. NO PII. Just the same intent names the admin created.
The dashboard works entirely from intent-level metadata, never from query content.

---

## Simulation Queries

Current: 55 hardcoded multi-intent queries in ProjectionsPage.tsx.
These cover: refunds, cancellations, billing issues, fraud reports, plan changes,
reorders, shipping updates, returns, warranty, complaints, gift cards, coupons.

Consider: Allow users to paste their own queries for simulation (reuse Discovery
page's paste/upload UI). This lets them see patterns from their own data without
going to production.
