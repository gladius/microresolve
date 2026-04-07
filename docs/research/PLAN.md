# ASV Implementation Plan

## Straightforward (do first)

### S1. Replace prerequisites with intent type + opaque metadata [DONE]
- Remove `PrerequisiteKind`, `Prerequisite` struct, `prerequisites` HashMap
- Add `intent_type: IntentType` field on intents (enum: `Action`, `Context`)
  - Action: user explicitly wants this done (cancel_order, refund)
  - Context: supporting data for fulfillment (check_balance, track_order)
  - Default: Action
- Add `metadata: HashMap<String, Vec<String>>` on intents (opaque, user-defined)
  - Two conventional keys: `"context_intents"`, `"action_intents"` (suggestions)
  - User can add anything else, ASV stores and returns, never interprets
- Update: Router struct, RouterState, export/import, server.rs, wasm.rs, API client, UI
- Update all prerequisite tests → intent type + metadata tests
- Files: lib.rs, multi.rs, server.rs, wasm.rs, client.ts, IntentsPage.tsx

### S2. Query highlighting in RouterPage [DONE]
- Use existing span data to color-code query text per detected intent
- Card-based response format:
  - Top: highlighted query with colored segments per intent
  - Below: intent cards with id, score, type badge (action/context)
- Each intent gets a consistent color from a palette
- Makes architecture self-evident to demo viewers
- Files: RouterPage.tsx

### S3. Score gap filtering [DONE]
- After route_multi, compute relative scores (each score / best score)
- Auto-flag low-confidence matches (below 30% of top score)
- Show flagged intents dimmed/strikethrough in UI, still visible but marked
- This is UI-only initially, no algorithm change in Rust
- Files: RouterPage.tsx

### S4. Settings page + global mode toggle [DONE]
- New SettingsPage.tsx with:
  - Mode selector: Production / Learn (radio or toggle)
  - API key field (for LLM, saved to localStorage)
  - Confidence threshold slider
  - Log file path display
- Mode indicator on navbar (colored badge: green=production, amber=learn)
- Store mode in React context or zustand, accessible from all pages
- Files: SettingsPage.tsx, Layout.tsx, App.tsx, new context/store

### S5. Query logging (persist to disk) [DONE]
- Server-side: append every route/route_multi call to JSONL file
- Log entry: `{timestamp, query, threshold, results: [{id, score, span}], mode}`
- New endpoint: `GET /api/logs` (paginated, most recent first)
- New endpoint: `DELETE /api/logs` (clear log)
- Log file: `asv_queries.jsonl` in working directory
- Files: server.rs, client.ts

### S6. Intent type in create/edit UI [DONE]
- AddIntentPanel: radio buttons for Action/Context type
- IntentDetailPanel: show type badge, allow changing
- IntentListItem: visual indicator for context vs action intents
- Files: IntentsPage.tsx, client.ts

## Experimental (do after straightforward)

### E1. Learn mode — LLM-reviewed routing [DONE]
- In learn mode, after every route, send to LLM:
  - The query
  - All detected intents with scores and spans
  - All intent definitions (id, type, seeds)
  - Current threshold
- LLM returns structured analysis:
  - Correct intents (confirmed)
  - False positives (should remove)
  - Missed intents (should have matched)
  - Suggested corrections (learn/correct/add_seed calls)
  - Reasoning
- UI shows LLM analysis as a suggestion card below routing result
- One-click approve buttons for each suggestion
- Before/after comparison: re-route same query to show improvement
- Files: RouterPage.tsx, server.rs (prompt building), client.ts

### E2. Intent co-occurrence tracking [DONE]
- Track which intents fire together on every route_multi call
- Data structure: `HashMap<String, HashMap<String, u32>>` (pair counts)
- Persist alongside query log or in router state
- API endpoint: `GET /api/co_occurrence`
- UI: co-occurrence matrix or graph on Debug page
- Future: use co-occurrence to adjust thresholds dynamically
  (if A and B co-occur 80% of the time, lower threshold for B when A is present)
- Files: lib.rs, server.rs, client.ts, DebugPage.tsx

### E3. Auto-tuning feedback loop (paper feature) [DONE]
- Combine S5 (logging) + E1 (LLM review) into closed loop
- In learn mode: every query → route → LLM review → suggestion → approve → learn
- Track improvement metrics: accuracy before/after, number of corrections needed
- "Cold start to production" metric: how many reviewed queries until accuracy plateaus
- Export tuning report for paper
- Files: TuningPage.tsx or integrated into RouterPage

### E4. Context intent discovery [DONE]
- When route_multi fires, if low-score matches are context-type intents,
  promote them as "suggested context" rather than filtering as false positives
- Mathematical basis: term overlap between action and context intents reveals
  functional dependencies (e.g., refund → needs check_balance)
- Separate section in routing output: `actions: [...]`, `context: [...]`
- Files: lib.rs (or server-side grouping), RouterPage.tsx

## Execution order

```
S1 → S6 → S2 → S3 → S4 → S5 → E1 → E2 → E3 → E4
│         │              │         │
│         │              │         └─ logging enables learn mode
│         └─ highlighting└─ settings enables mode toggle
└─ type system enables everything else
```
