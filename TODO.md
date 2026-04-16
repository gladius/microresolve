# TODO — ASV + Intent Programming

Open work surfaced during design + live tests (2026-04-15 / 2026-04-16).
See also: `docs/LANGUAGE_AS_PROGRAM.md`, `INTENT_PROGRAMMING.md`.

## Completed in this session

### [x] ASV is entry-only, not a continuous router
Done in `src/execute.rs::resolve_turn`. After the first user message, intent transitions are driven by `→` handoff directives; ASV no longer re-routes mid-conversation. Validated across scenarios 1, 2, 4 — no hijack seen when vocabulary overlapped.

### [x] Option H handoff runtime (arrow + context briefing)
Done in `src/execute.rs` + `src/bin/server/routes_core.rs`. LLM emits `→ <intent_id>` and `context: <paragraph>`, server validates and carries forward. Briefing is auto-injected into the receiving intent's system prompt.

### [x] Router metadata persistence
Fixed in `src/bin/server/routes_build.rs`. The builder's `create_intent` and `update_intent` tool handlers now call `maybe_persist`. Router's existing `save_to_dir` already writes per-intent JSON files; startup loader already calls `load_from_dir`. The gap was just that the builder never triggered the save. Verified end-to-end: build agent → restart server → namespace loaded with all intents, phrases, instructions, persona intact → execute works on loaded agent.

### [x] Inter-intent reference via `lookup:`
Done as a proof-of-concept (`src/execute.rs::extract_lookups`). One intent can read another intent's body inline; server executes up to 3 rounds. Works for composition (multiple lookups in one turn). **Remove the `mode: ["fact"]` type distinction** — keep the mechanism as pure inter-intent reference; all intents are just intents with short or long bodies.

---

## High priority

### [ ] Drop the fact/action type distinction
The `mode: ["fact"]` metadata, `is_fact_intent()` helper, and special listing in system prompts add unwanted cracks in the "everything is an intent" principle. Keep the `lookup:` mechanism — it's just "read another intent's body" — but treat all intents uniformly. The optimization of reading short bodies directly (vs calling the LLM) can be automatic, not a declared type.

### [ ] Simplify hallucination story
Two validated mitigations: (a) explicit exhaustive prose ("we do NOT offer X, Y, Z"), (b) `lookup:` to an authoritative intent body. Both work without introducing new type systems. Document the pattern in `LANGUAGE_AS_PROGRAM.md` as the official approach; de-prioritize the side-channel-tool idea until we see real failures that only it solves.

### [ ] Builder support for inter-intent reference
`/api/build` currently creates conversational intents with flow instructions. It doesn't yet know about creating short-body reference intents (`price_bath` = `$40`) or wiring `lookup:` references into other intents' prose. Update the system prompt so the builder LLM can choose: make a new intent for a shared fact, or inline it.

---

## Medium priority

### [ ] Separate repo / release track for ASV vs Intent Programming
Commit to the two-project framing. ASV becomes a standalone routing library (fast classification). Intent Programming uses ASV as a dependency. Consider either a Cargo workspace with two published crates, or two separate repos. Makes each story cleaner and lets ASV ship for audiences that don't want the agent runtime.

### [ ] ASV reliability — seed-quality surface
ASV routing quality is a function of seed phrase coverage. Make this observable:
- Show per-intent seed coverage (how many phrases, how distinctive the vocabulary)
- LLM-assisted phrase generation as a first-class UI flow
- Auto-learn visibility — user sees their agent getting better over time
- Conversion metrics: `% of queries routed confidently at namespace level`

This converts "I don't trust ASV" into "I can see it improving."

### [ ] Confidence tier labeling for public launch
Label features by readiness:
- **Stable:** entry-point routing, core scoring, multilingual tokenization
- **Beta:** auto-learn, CRDT merge
- **Experimental:** intent programming runtime, handoff conventions, `lookup:`

Don't rely on community to fix algorithmic cores. Do invite community contributions to seed packs, benchmarks, integrations, UI, and domain docs.

### [ ] Forced-handoff disposition
When `next_intent` from previous turn fires, current routing shows `disposition: no_match` (because ASV didn't fire). Should surface as `forced_handoff` for readable logs.

### [ ] Handoff tracing UI
Context paragraphs, handoff chains, remarks generate a rich audit trail but there's no UI. A "Runs" tab on the Intents page could show conversation traces with handoff arrows rendered as graph edges, searchable by failure.

### [ ] Scenario test harness
Current tests are ad-hoc Python scripts. Commit a proper integration-test harness: build intents → run fixed conversation scripts → assert on handoff/remark/context/routing.

### [ ] Fix builder's prompt-following quirks
Scenario 1 had the LLM dropping the `→` arrow while still emitting `context:` — partial handoff. Tighten the system prompt wording so these stay paired, or accept context-only as a valid signal (scratchpad).

### [ ] Seed-coverage fallback
Scenario 3 returned empty replies because ASV didn't match informal phrasings ("crashing" vs seeded "crashed"). When routing returns no match, fall back to a generic clarify/greet intent instead of returning empty.

---

## Speculative / deferred

- **Side-channel tool calls** — deferred. Current inline `lookup:` works; revisit if we hit real bottlenecks with history bloat or latency.
- **Action intents + idempotency/integration patterns** — park until there's a real use case. Today's conversational + short-body intents cover most scenarios.
- **Parameterized intent bodies** (`price_bath(size)` templating) — only if a concrete use case appears.
- **Typed slot escape valve** — optional structured `slots:` alongside prose `context:`. Wait for a real precision boundary that prose can't handle.
- **Multi-intent parallel handoff** (`→ [a, b]`) — probably not needed.
- **Scale test** — 50-100 intents in one namespace. Untested but likely fine given ASV's index. Worth a benchmark when we have real data.
- **Adversarial guardrail red-teaming** — wait for external users.
- **Render intent graph visually** — static analysis over `→` references across all intent prose. Slick demo when ready.

---

## Launch-track decisions pending

- Which project launches first — ASV standalone or bundled?
- Public naming: "ASV" stays? "Intent Programming" or something else for the runtime?
- Repo split: monorepo with two crates, or two repos?
- What's the minimum-viable persistence + UI story before any public hand-off?
- Community contribution channels — issue templates, seed-pack PR flow, benchmark submission.
