# Studio Page — Implementation Plan

## Terminology
- Hierarchy: **Namespace → Domains → Intents** (no "app" concept anywhere)
- `X-Namespace-ID` header on all API calls selects the active namespace
- Server internally uses `app_id_from_headers()` — implementation detail only

---

## Four Studio Tabs — Final Design

```
┌─────────────────────────────────────────────────────────────────┐
│  [ Manual ]  [ Simulate ]  [ Review ]  [ Auto ]                │
│──────────────────────────────────────────────────────────────────│
│  LEFT PANEL (tab-specific)   │  RIGHT PANEL (tab-aware)        │
│                              │                                  │
│  Manual:   chat routing UI   │  Manual/Simulate: SSE feed      │
│  Simulate: gen→test→learn    │    (learn_now events only)      │
│  Review:   flagged queue     │  Review: queue strip + detail   │
│  Auto:     worker stats      │  Auto: worker SSE feed          │
└─────────────────────────────────────────────────────────────────┘
```

### Tab behaviour (FINAL, agreed)
| Tab | Auto-learns? | Who triggers? | Right panel |
|-----|-------------|---------------|-------------|
| Manual | YES, always | User clicks "Learn Now" | `learn_now` SSE events |
| Simulate | YES, always | Each failure → `learn_now` loop | `learn_now` SSE events |
| Review | NO | Human: Analyze → Apply | Selected item detail |
| Auto | YES, if namespace `auto_learn=true` | Background worker | Worker SSE events |

**Manual and Simulate always auto-learn** — `learn_now` is synchronous, bypasses queue.
**Review** is always manual — human analyst only.
**Auto** is the background worker — only fires when namespace has `auto_learn=true`.

---

## No Global Review Mode (AGREED, to implement)

The old `review_mode: manual|auto` per-namespace flag is **removed entirely**.

- **Before**: server had `review_mode` in AppState, `GET/POST /api/review/mode` endpoints controlled the worker
- **After**: worker checks namespace `auto_learn: bool` directly — the flag already exists on the namespace model

### Changes needed:
- [x] **Server**: Remove `GET /api/review/mode` and `POST /api/review/mode` endpoints from `routes_review.rs`
- [x] **UI client.ts**: Remove `getReviewMode()` and `setReviewMode()`
- [x] **UI AutoPanel**: Toggle `auto_learn` via `PATCH /api/namespaces/{id}` instead of `setReviewMode()`
- [x] **UI Settings page**: Remove "Review Mode" section entirely
- [x] **UI Layout.tsx**: Remove the AUTO badge (was driven by review_mode)
- [x] **UI**: Delete dead `ReviewPage.tsx` and `AutoImprovePage.tsx` files

Note: `review_mode` HashMap stays in `AppState` as the internal backing store.
Worker still uses `get_ns_mode()` which reads it. The `PATCH /api/namespaces` handler
writes to it via `auto_learn` field. HTTP endpoints for direct access are gone.

---

## What is Done ✓

### Step 1 — Simulate tab fixed ✓
- Removed `setReviewMode('auto')` / `api.report()` / worker-wait pattern
- Now uses `learn_now` loop: `for failure in failures: await api.learnNow(...)`
- No queue, no mode switching, SSE fires per failure
- Queue stays unchanged during simulate

### Step 2 — Auto tab added ✓
- Left panel: ON/OFF toggle, queue stats, live worker event feed
- Right panel: tab-aware — each tab shows relevant SSE events only
- Review tab: queue strip only shown on Review tab
- Tab-specific empty states (no more "what is 2+2 — Analyze with AI" bleeding into Manual)

### Step 3 — `log: false` on route_multi ✓
- Manual tab routes with `log: false` — zero queue pollution
- Added `log: bool` field to `RouteMultiRequest` in server

### Step 4 — `/api/report` endpoint added ✓
- Added `report_query` handler to `routes_review.rs`
- Accepts query + detected + flag, adds to log store, returns id

### Nav cleanup ✓
- Removed "Review" and "Learn" from nav
- Nav: Namespaces | Intents | Studio | Import | Settings

---

## What Remains

### Priority 1 — Remove review_mode (clean up the confusion) ✓ DONE
HTTP endpoints removed. `auto_learn` on namespace PATCH is the only public control.
Worker still reads internal `review_mode` HashMap (set by namespace PATCH handler).

### Priority 2 — 2-turn LLM triage gate
Replace 3-turn `full_review` with:
- Turn 1: "Is this query relevant to any intent?" → if NO, mark irrelevant, stop
- Turn 2: "Which intent? Generate phrases." → apply, resolve

Prevents LLM waste on gibberish. Affects `full_review` in `llm.rs`.

Log record additions:
```rust
irrelevant: bool,  // triage said not related to namespace
escalated: bool,   // worker gave up, needs human
```

### Priority 3 — Escalation state for Review tab
- Worker marks entries `escalated=true` when LLM can't fix after triage
- Review tab filter: show escalated-only (when auto_learn=true) or all flagged (when false)
- `GET /api/review/queue?escalated=true` filter

### Priority 4 — Simulate history
- Save run results to `{data_dir}/{namespace}/simulations.json` (max 20)
- Show history in Simulate tab (collapsible past runs)

---

## Server Endpoints State

### Keep
- `POST /api/learn/now` — sync learn, Manual + Simulate tabs
- `POST /api/training/generate` — LLM query generation
- `POST /api/training/run` — batch routing evaluation (no logging)
- `GET /api/review/queue` — review queue
- `GET /api/review/stats` — queue depth
- `POST /api/review/analyze` — LLM analysis of entry
- `POST /api/review/fix` — apply phrases to resolve entry
- `GET /api/events` — SSE stream
- `POST /api/report` — explicit queue entry (for future use)

### Remove
- `GET /api/review/mode` — replaced by namespace `auto_learn` flag
- `POST /api/review/mode` — replaced by namespace update

### Namespace update for auto_learn toggle
- Use existing `PATCH /api/namespaces/{id}` or add `auto_learn` field to namespace update

---

## Simulate Flow (correct, implemented)
```
Step 1: POST /api/training/generate  → LLM generates N queries with ground_truth
Step 2: POST /api/training/run       → baseline (no logging)
Step 3: for failure in failures:
          POST /api/learn/now        → sync LLM fix (SSE fires per call)
Step 4: POST /api/training/run       → retest
Step 5: Show before%/after%, fixed/stuck, per-failure learn log
```

## What NOT to build
- No separate in-memory queue for Manual/Simulate
- No `/api/report` usage from Manual/Simulate tabs (use `learn_now`)
- No mode switching from UI for Manual/Simulate (never touch auto_learn mode)
- No complex ID tracking in Simulate (learn_now is synchronous)
