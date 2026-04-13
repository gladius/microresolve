# Auto-Learn Optimization Plan

## Context

The auto-learn pipeline was designed around a term-index router. Every turn was
oriented toward phrase management: add a phrase → term index picks it up → routing
improves. The router is now Hebbian L1+L2, which learns directly from query
vocabulary. The phrase store still exists for human readability and re-bootstrap
seeding, but it no longer drives live routing. The LLM turn structure needs to
be redesigned around the new architecture.

---

## 1. Intent Descriptions Are Required — No Fallback

**Current state**: `build_intent_descriptions()` sends id + description + 4 sample
phrases. If no description exists, phrases act as a substitute.

**Problem**: Sending phrases in the judge turn is noise. The 4-phrase fallback
masks missing descriptions instead of enforcing them.

**Decision**: Descriptions are mandatory. No fallback to phrases. Flag at entry points:

- **Import flow** (`/api/intents/multilingual`, MCP import, OpenAPI import): if
  `description` field is absent or empty, the intent is created but marked with a
  warning flag in the response. The UI shows an amber badge on flagged intents.
- **Intents page**: intents without descriptions show a "missing description"
  warning and are ineligible for bootstrap until one is added.
- **Auto-learn Turn 1**: if any intent in the namespace has no description, log a
  warning and skip it from the judge context (it cannot be correctly classified
  anyway).

---

## 2. Turn 1 — Judge: id + description only

**Current**: sends id + description + up to 4 seed phrases per intent.

**Fixed**: sends `id (description)` one line per intent, nothing else.

```
billing:charge_card (Charge a payment card)
billing:refund (Issue a full or partial refund)
shipping:track_order (Track the location or status of an order)
...
```

**Why this is correct**: Turn 1 is a classification task. It asks the LLM "which
of these intents does this query express?" This is exactly how the industry runs
LLM-based routing — OpenAI function calling, Anthropic tool use, LangChain agents
all send a tool name + description, no examples. Examples belong in fine-tuning,
not in a zero-shot classification prompt.

Sending phrases in Turn 1 costs tokens, adds noise, and creates a false impression
that the LLM needs to see training data to judge intent. It does not. The
description is the interface contract.

**Scale**: 100 intents × ~60 chars per line = ~6000 chars = ~1500 tokens. Flat cost
regardless of how many phrases have been learned. Does not grow with usage.

---

## 3. Turn 2 — Fix Misses: are phrases still the right output?

### What Turn 2 currently does

LLM is shown the missed intents + their current phrases, and asked to suggest new
phrases to add. Those phrases go through `phrase_pipeline` → Router training store.
Separately, `apply_review` updates L2 edges directly from the original query's
content words (not from the suggested phrases).

### The architecture mismatch

The phrase store and L2 edges are two separate knowledge representations. Adding a
phrase to the store does not immediately update L2 — that gap is tracked in the
codebase. The routing improvement from Turn 2 currently comes entirely from the
L2 direct edge update in `apply_review`, not from the phrases Turn 2 suggests.

So Turn 2's phrase suggestions are:
- ✓ Useful for training store quality and future re-bootstrap
- ✓ Human-readable audit trail of what the system learned
- ✗ Not the mechanism that fixes routing right now

### Are phrases still the best output?

Yes, but they need to propagate automatically to L2. The fix (tracked separately):
when `phrase_pipeline` adds a new phrase, tokenize it and immediately write L2 edges
at a base weight (0.7). This closes the gap — phrases become the human-readable
interface that also drives immediate routing improvement, same as the old term index.

Once that fix is in place, Turn 2's phrase suggestions are both persistent (stored)
and immediately effective (L2 seeded).

### Turn 2 scope and efficiency

- Only send missed intents + their current phrases (capped at **15 most recent**)
- Do NOT send all intents — Turn 1 already classified; Turn 2 drills down
- Language instruction: if Turn 1 detected a non-English language, tell Turn 2 to
  generate phrases in that language
- Phrase cap rationale: after 15 phrases, an intent's coverage is well-defined.
  Sending 60 learned phrases wastes tokens and produces redundant suggestions.

---

## 4. Turn 3 — False Positive Narrowing: is it still needed?

### What Turn 3 currently does

Shown wrong intents + their phrases, asked to suggest phrase replacements that
narrow coverage and stop matching the current query.

### The architecture mismatch

Phrase replacement → Router training store update. But routing is driven by L2 edges,
not phrase text matching. Replacing a phrase in the store does not change any L2
edge weights. The suppression that actually fixes false positives happens in
`apply_review`: `wrong_detections` content words get suppressed in L2 directly.

So Turn 3 is solving the wrong problem. The L2 suppression already fires without
Turn 3. Turn 3 only improves the training store (for future re-bootstrap quality),
at the cost of an additional LLM call for every false positive.

### Decision: Turn 3 is optional, not always run

**Skip Turn 3** when:
- Wrong detections are marginal (score close to threshold) — L2 suppression alone is
  sufficient, the edge will decay naturally
- The namespace is small (< 20 intents) — false positive rate is low

**Run Turn 3** only when:
- A wrong detection is high-confidence (strong L2 score for the wrong intent)
  meaning the training store itself has an overly broad phrase that will regenerate
  the bad L2 edge on next bootstrap
- The false positive is recurring across multiple queries

**Alternative to phrase replacement in Turn 3**: ask the LLM for specific suppressor
words instead. "Which words in this query most strongly indicate NOT [wrong intent]?"
These map directly to L2 suppressor edges and take effect immediately without
touching the phrase store.

---

## 5. Proposed Flow (Revised)

```
Query arrives → route_multi → miss or low_confidence → log flagged
                                                              │
                                                     worker wakes (auto mode)
                                                              │
                                          ┌───────────────────┘
                                          │
                                   Turn 1: Judge
                                   ALL intents: id + description only
                                   → correct / wrong / missed / language
                                          │
                              ┌───────────┴────────────┐
                              │                        │
                   missed intents?              wrong detections?
                              │                        │
                       Turn 2: Fix                L2 Suppress
                   missed intents only:         (already fires in
                   description + phrases        apply_review — no
                   (cap 15) + language          extra turn needed)
                   → new phrases to add               │
                              │                  Turn 3 (optional):
                   phrase_pipeline +             only if high-confidence
                   L2 edge seed                  false positive — ask for
                   (when gap fixed)              suppressor words → L2
                              │
                   L2 Direct Update (always):
                   content words → reinforce/suppress
                              │
                   L1 Synonym (parallel, missed only):
                   query words not in L2 → map to existing L2 vocab
                   (handles multilingual + paraphrase recovery)
                              │
                   L1 Morphology (parallel, new vocab only):
                   brand new words → inflected forms
```

---

## 6. LLM Call Count Per Event

| Scenario | Current | Optimized |
|----------|---------|-----------|
| Miss (simple) | T1 + T2 + L1-morph = 3 calls | T1 + T2 + L1-syn + L1-morph = 4 calls* |
| Miss + false positive | T1 + T2 + T3 + L1-morph = 4 calls | T1 + T2 + L1-syn + L1-morph = 4 calls (T3 skipped if marginal) |
| Perfect detection | T1 only = 1 call | T1 only = 1 call |

*L1-syn and L1-morph can run in parallel (tokio::join!) — wall-clock time unchanged.

---

## 7. Implementation Order

1. **Strip phrases from Turn 1** — 2-line change in `build_intent_descriptions()`
   or a new `build_intent_labels()` function used only in Turn 1
2. **Enforce descriptions** — flag in import endpoints + intents page UI
3. **Phrase cap in Turn 2** — limit `router.get_training(id)` to last 15 phrases
4. **Turn 3 gating** — only fire when wrong detection score > threshold × 2
5. **Phrase → L2 seeding** — in `phrase_pipeline`, tokenize new phrases and write
   L2 edges at base weight 0.7 immediately (closes the phrase/L2 gap)
6. **Turn 3 suppressor rewrite** — when Turn 3 does fire, ask for suppressor words
   instead of phrase replacements (better mapping to actual L2 mechanics)

## 8. Open Questions

- Should descriptions be auto-generated by LLM on intent creation if not provided?
  Or is requiring a human-written description important for quality control?
- What is the right phrase cap — 15 is a guess. Needs measurement.
- Should Turn 3 be removed entirely and replaced with a dedicated "suppressor
  discovery" pass that only fires for high-confidence recurring false positives?
