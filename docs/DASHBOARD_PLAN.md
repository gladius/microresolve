# Dashboard Implementation Plan

## Overview

Replace the current Projections page with a full analytics dashboard that surfaces
all emergent intelligence from routing data: co-occurrence, workflows, temporal
ordering, escalation detection, anomaly alerts, intent suggestions, and query
intelligence signals (agency, causation, certainty, similarity).

The library already computes most analytics. New additions: query intelligence
signals (agency, causation, certainty), intent similarity detection, anomaly
tracking, and confidence/frequency stats.

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

**Estimated: ~150 lines.**

#### 1d. Query Intelligence Signals (agency, causation, certainty)

Detected during routing from the query text. Returned as metadata on each
routing result. No ML — lexicon-based pattern matching.

```rust
/// Query intelligence signals extracted during routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySignals {
    /// Who is acting: Requesting, Reporting, Commanding
    pub agency: Agency,
    /// Causal markers found: "because X", "due to Y"
    pub causes: Vec<String>,    // intent IDs that caused this intent
    /// Certainty level: 0.0 (very uncertain) to 1.0 (very certain)
    pub certainty: f32,
}

pub enum Agency {
    Requesting,  // "I want to cancel" — active, user wants action
    Reporting,   // "My order was cancelled" — passive, something happened
    Commanding,  // "Cancel the order" — imperative, direct instruction
    Unknown,
}
```

**Agency detection** — verb form analysis:
```
Active + first person: "I want", "I need", "I'd like"    → Requesting
Passive: "was cancelled", "has been charged", "got denied" → Reporting
Imperative: "Cancel", "Show me", "Track"                   → Commanding
```
Implementation: ~20 regex patterns on the query string. Runs during tokenization.

**Causation detection** — marker words:
```
"because", "since", "due to", "as a result", "so", "therefore",
"that's why", "caused by", "after", "which led to"
```
When found, split query at the marker. Route both halves. Mark the
second intent as CAUSED BY the first.
```
"I want to cancel because I was charged twice"
  → cancel_order (primary) + billing_issue (cause)
  → causes: ["billing_issue"]
```
Implementation: ~15 marker words, split + re-route logic.

**Certainty detection** — hedge vs commitment lexicon:
```
Hedges (low certainty):     "maybe", "possibly", "thinking about", "might",
                            "not sure", "wondering", "could I", "is it possible"
Commitment (high certainty): "definitely", "absolutely", "must", "need",
                            "right now", "immediately", "I demand", "ASAP"
Neutral:                     no markers → 0.5
```
Implementation: ~30 words, score = (commitment_count - hedge_count) / total, clamped to [0,1].

**Analytics from signals:**
```
Agency distribution:     "72% requesting, 18% reporting, 10% commanding"
Causation chains:        "billing_issue causes 47% of cancellations"
Certainty per intent:    "refund requests avg 0.8 certainty, track requests avg 0.4"
```
These aggregate stats populate dashboard visualizations without any PII.

**Estimated: ~100 lines in tokenizer.rs or a new signals.rs module.**

#### 1e. Intent Similarity Detection

Detect duplicate/overlapping intents from two signals:

**Seed similarity (Jaccard):**
```
cancel_order seeds:  {cancel, order, stop, want}
stop_order seeds:    {stop, order, halt, cancel}
Jaccard = |intersection| / |union| = 3/5 = 0.60 → ⚠ high overlap
```

**Co-occurrence rate:**
```
If cancel_order and stop_order fire together >90% of the time
→ they're probably the same intent
```

```rust
/// Detect potentially duplicate or overlapping intents.
pub fn detect_similar_intents(&self, threshold: f32) -> Vec<SimilarPair> {
    // Compare all intent pairs by:
    // 1. Seed term Jaccard similarity
    // 2. Co-occurrence rate (if available)
    // Return pairs above threshold
}

pub struct SimilarPair {
    pub intent_a: String,
    pub intent_b: String,
    pub seed_similarity: f32,    // 0.0-1.0 Jaccard
    pub co_occurrence_rate: f32, // 0.0-1.0 how often they fire together
    pub suggestion: String,      // "Consider merging" or "Review overlap"
}
```

**Estimated: ~60 lines.**

---

**Total library additions: ~460 lines**
- Intent stats + confidence: ~150 lines
- Anomaly detection: ~80 lines (included in 150 above)
- Query signals (agency, causation, certainty): ~100 lines
- Similarity detection: ~60 lines

### 2. Server additions (src/bin/server.rs)

New endpoints:
```
GET /api/stats              → intent hit counts + confidence distribution
GET /api/anomalies          → detected anomalies
GET /api/similar_intents    → duplicate/overlap detection
POST /api/record            → record intent hits from connected clients (just intent IDs, no query text)
```

Modify existing `route_multi` response to include query signals:
```json
{
  "confirmed": [...],
  "candidates": [...],
  "relations": [...],
  "signals": {
    "agency": "requesting",
    "causes": ["billing_issue"],
    "certainty": 0.85
  }
}
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

#### 4g. Query Intelligence (NEW)
```
Agency Distribution          Certainty by Intent         Causation Map
┌────────────────────┐      ┌──────────────────────┐    ┌─────────────────────────┐
│ ████████░░ 72% req │      │ refund         ██ 0.8│    │ billing_issue           │
│ ███░░░░░░░ 18% rep │      │ cancel_order   ██ 0.7│    │   → 47% cancel_order    │
│ ██░░░░░░░░ 10% cmd │      │ track_order    █░ 0.4│    │   → 23% contact_human   │
└────────────────────┘      └──────────────────────┘    │ delivery_issue          │
                                                         │   → 61% refund          │
                                                         └─────────────────────────┘
```
Three visualizations from query signals:
- Agency: pie/bar chart of requesting vs reporting vs commanding
- Certainty: per-intent average certainty score
- Causation: which intents CAUSE which (root cause analysis)
Pulls from aggregated signal data accumulated during routing.

#### 4h. Intent Health (NEW)
```
⚠ cancel_order and stop_order are 87% similar — consider merging
⚠ track_order and shipping_status co-occur 94% of the time
✓ All other intents are well-separated
```
Intent similarity/overlap alerts. Helps admin maintain clean intent taxonomy.
Pulls from `/api/similar_intents`.

#### 4i. Simulation (existing, move here)
```
[Run Simulation]  55/55 ████████████████ Complete
```
Same button + progress bar from current Projections page. Runs 55 queries
to populate all analytics data for demo purposes.

**Estimated: ~600-700 lines for the full dashboard page.**

### 5. Webhook Alerts (future, not in MVP)

HTTP webhook system for real-time alerting. When anomalies, escalations,
or confidence drops are detected, POST to a configured URL.

**Configuration** (server .env or --webhook flag):
```
WEBHOOK_URL=https://hooks.slack.com/services/T00/B00/xxx
WEBHOOK_EVENTS=escalation,anomaly,confidence_drop
```

**Payload** (no customer data — intent-level signals only):
```json
{
  "type": "escalation_spike",
  "severity": "high",
  "app_id": "support-bot",
  "description": "billing_issue → contact_human escalation 3.2x above baseline",
  "details": {
    "pattern": ["billing_issue", "contact_human"],
    "current_rate": 0.24,
    "baseline_rate": 0.075
  },
  "timestamp": "2026-04-08T14:32:00Z"
}
```

**Alert types:**
- `anomaly` — volume spike, new unknown pattern, intent drift
- `escalation` — escalation pattern rate above threshold
- `confidence_drop` — low confidence rate above threshold
- `similarity` — new intent overlap detected above threshold

**Implementation:** ~50 lines in server. Check anomalies after each route_multi batch
(debounced, not every query). POST to webhook URL if triggered. Fire-and-forget
(don't block routing on webhook response).

**UI:** Webhook URL configuration on Settings page. Test webhook button.

### 6. Navigation update

```
[Playground]  [Intents]  [Dashboard]  [Discovery]  [Debug]  [Settings]
```

"Projections" renamed to "Dashboard". Route stays at `/projections` or
changes to `/dashboard`.

---

## Implementation Order

```
Step 1: Query signals — agency, causation, certainty detection
        Add to tokenizer.rs or new signals.rs module
        Integrate into route_multi output
        Test: unit tests for each signal type
        ~100 lines, ~30 min

Step 2: Intent similarity detection
        Add detect_similar_intents() to lib.rs
        Test: unit tests with overlapping intents
        ~60 lines, ~20 min

Step 3: Intent stats + confidence tracking + anomaly detection
        Add to lib.rs, integrate into route_multi recording
        Test: unit tests for stats and anomaly detection
        ~150 lines, ~30 min

Step 4: Server endpoints
        Add /api/stats, /api/anomalies, /api/similar_intents, /api/record
        Modify route_multi response to include signals
        Test: curl tests
        ~80 lines, ~20 min

Step 5: API client — add new methods
        ~15 lines, ~5 min

Step 6: Dashboard UI — build the full page with all 9 sections
        Overview, Anomalies, Projected Context, Workflows,
        Temporal Flow, Escalation, Query Intelligence,
        Intent Health, Simulation
        Test: run simulation, verify all sections populate
        ~600 lines, ~2-3 hours

Step 7: Navigation — rename Projections → Dashboard
        ~1 line change
```

Total estimated: ~1000 lines, 4-5 hours.

### Priority order (if time-constrained)

Must-have (MVP dashboard):
- Steps 3-4: Stats + anomalies (backend)
- Step 6 sections a-d, g, i: Overview, Anomalies, Projections, Workflows, Simulation

Should-have:
- Steps 1-2: Query signals + similarity
- Step 6 sections e-f, h: Temporal, Escalation, Intent Health

Nice-to-have:
- Step 6 section g: Full query intelligence visualization (agency, causation, certainty)
- Causation detection is the most complex signal

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
