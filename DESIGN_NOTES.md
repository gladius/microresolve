# ASV Router — Design Notes & Decisions

Running record of architectural discussions, test findings, and open questions.
Most recent entries at the top of each section.

---

## 2026-04-10 — Multi-App Testing & Honest System Limits

### What was built
- `tests/multi_app_test.py` — 100 tests across 5 apps (stripe/github/slack/shopify/calendar),
  20 intents each, minimal seeds. Tests single-intent, multi-intent, negative.
- `tests/cross_app_learn_test.py` — 20 cross-app multi-intent queries with auto-learn loop,
  plus 10 hard-tier observation cases.

### Honest baseline (seeds only, no tuning)
| Category | Score | Notes |
|---|---|---|
| Single-intent (natural phrasing) | 16/46 (34%) | Fails when user phrasing shares no words with seeds |
| Multi-intent same-app (implicit) | 4/15 (26%) | Compound intent with no conjunction barely works |
| Multi-intent cross-app | 6/20 (30%) | Catches ~half on first contact |
| Negative (no-intent queries) | 25/25 (100%) | No false positives on gibberish |
| After one learn pass (same queries) | 20/20 (100%) | Memorizes fast, doesn't generalize |

### Hard tier findings (10 genuinely difficult queries)
| Category | Result | Root cause |
|---|---|---|
| IMPLICIT ("payment bounced") | Nothing fired | No action verb, system blind to situation descriptions |
| RAMBLING (50-word buried intent) | Partial (1/2) | Term density survives noise; accidental false positives from unrelated terms |
| NEGATION ("process card, hold invoice") | Correct | Negation suppression working |
| JARGON ("cut the release, push channel") | Wrong intent | "Push" matched add_collaborator seeds — jargon remapped to wrong meaning |
| SHORT ("ship it") | Nothing | Two words below scoring threshold |
| AMBIGUOUS ("cancel it") | Correctly nothing | Right behavior — ambiguous pronoun → low confidence |
| TYPO ("creat an isue") | Correct | Tokenizer survives if supporting terms intact |
| PRONOUN ("do the usual") | Nothing | No session memory — architectural limit |

---

## 2026-04-10 — Seed vs Learned Terms Discussion

### Current design
`LearnedVector` has two separate maps:
- `seed_terms: HashMap<String, f32>` — set at setup, immutable during operation
- `learned_terms: HashMap<String, f32>` — grown from `learn()` / `correct()` calls
- Scoring: `sum of max(seed_weight, learned_weight)` per query term

### Why the separation exists
1. **Unlearn safety**: `correct()` calls `unlearn()` on wrong intent. If seeds and learned were one
   map, a correction could degrade a core seed term (e.g. "charge" weight drops because it appeared
   in a misrouted query). Currently: `unlearn()` never touches `seed_terms`.
2. **CRDT delta sync**: `export_learned_only()` / `import_learned_merge()` ship only learned weights
   between connected nodes, keeping seeds as local authoritative config. Seeds are set by the
   deploying org; learned weights can sync across instances.
3. **Decay isolation**: Periodic `decay()` only shrinks learned terms. Seed weights are stable
   anchors. If merged, long inactivity would silently degrade seed coverage.

### Merge proposal — is it safe?
Yes, with one change. Replace the two maps with:
- `terms: HashMap<String, f32>` — single unified map
- `protected_terms: HashSet<String>` — term names that `unlearn()` will never touch

When adding a seed phrase, its tokenized terms go into `protected_terms`. Unlearn skips any protected term. Same safety guarantee, simpler code.

The only thing lost: distinguishing "how many terms came from seeds vs learning" (the `learned_count` field in the API). If you don't need that observability, merge is fine.

If CRDT delta sync becomes important again, you'd need a different tagging mechanism. For now (full export/import sync), merge is fine.

**Decision: deferred — not urgent, seeds work correctly either way.**

---

## 2026-04-10 — Learned Vectors vs Embeddings

### What learned vectors are
Sparse term-weight maps that grow from user feedback. When `learn(query, intent)` is called:
- Each tokenized term in the query gets: `weight += 0.15 * (1 - weight)` (asymptotic toward 1.0)
- Terms accumulate in `learned_terms` per intent
- Next query with overlapping terms scores higher for that intent

### Key difference from embeddings
| | Learned Vectors | Embeddings (BERT/etc.) |
|---|---|---|
| Semantic understanding | No — only term overlap | Yes — "loop in" ≈ "invite" |
| Cold start | Blind until it sees the term | Works from day 1 |
| Speed | Microseconds | Milliseconds or API call |
| Model size | KB per intent | GB |
| Interpretable | Fully — readable weight map | Opaque |
| Online learning | Native | Requires fine-tuning |
| Infrastructure | Zero | Embedding server/API |

### Domain data loading (discussed earlier)
Loading raw text (API docs, support logs, domain glossaries) to pre-expand seeds via
co-occurrence mining before any user traffic. This is the offline version of what learned
vectors do online. Combined: load domain corpus → extract terms → seed learned vectors →
refine with real traffic.

### LLM-generated seed expansion (recommended)
For each intent, prompt LLM: "What are 20 ways an engineer might say [intent label]?"
Add responses as seeds. Pre-populates jargon vocabulary without needing real traffic.
Cost: one LLM call per intent at setup time. With 20 intents per app × 5 apps = 100 calls.

---

## 2026-04-10 — Situation→Action Inference (Open Problem)

See detailed design analysis below (next section).

---

## 2026-04-10 — Multi-App Architecture

### How it works
- Server holds `routers: RwLock<HashMap<String, Router>>` — one Router per app_id
- `X-App-ID` header selects which router to use per request
- Default app_id = "default" — backward compatible
- Routing is always scoped to one app. Cross-app dispatch is the caller's responsibility.

### Log store
Binary append log: `[u8: alive][u32 LE: len][json payload]`
- Per-app files: `{data_dir}/logs/{app_id}.bin`
- In-memory index (`Vec<LogMeta>`) rebuilt at startup — 50 bytes/entry, 1M entries ≈ 50MB
- `resolve()` flips the alive byte in-place (seek + write 1 byte)
- Review queue = log store query with `flagged_only=true, resolved=false`
- No separate in-memory review queue — log IS the review queue

### Connect mode (library side)
- `AppRouter` manages per-app `Arc<Router>` with atomic hot-swap
- Background thread polls `GET /api/sync?version=N` every 30s
- On version change: build new Router in background (no lock during rebuild), then atomic Arc swap
- Log shipping: `POST /api/ingest` batched every 60s
- reqwest always-on — no feature gate. Connect mode = runtime config (set server_url), not compile-time.

---

## Situation → Action Inference — Design Analysis (2026-04-10)

### The problem
Normal routing: user says what they want (command form).
Situation routing: user describes what is happening (state form). System must infer the intended action.

Examples:
- "the payment bounced" → implies retry → `stripe.charge_card`
- "customer churned last night" → implies cancel → `stripe.cancel_subscription`
- "we're getting 402s on the payments endpoint" → implies log → `github.create_issue` + alert → `slack.send_message`
- "the inventory hit zero" → implies restock → `shopify.update_inventory`

### Is it feasible as a seeded+learned index?

Yes, but the design must differ from action routing.

**Why the same approach doesn't directly work:**
Action routing indexes terms that *describe* the action ("charge", "bill", "process").
Situation routing must index terms that *describe a state* that implies the action ("bounced",
"declined", "failed", "rejected"). These are different vocabularies with no overlap.
A single inverted index conflates them — "bounced" would score for charge_card, but
accidentally match "the ball bounced" in any context.

**Proposed architecture: SituationIndex (parallel to InvertedIndex)**

Each intent has two scoring channels:
1. Action channel (existing) — seeds + learned terms describing the action
2. Situation channel (new) — situation phrases that imply this intent

Combined score:
```
total_score(intent, query) = action_score + 0.4 * situation_score
```

Situation score only contributes additively. It can promote an intent that action scoring
would have missed. It cannot suppress an intent that action scoring found.

**Situation phrase storage (phrase-level, not term-level):**

Rather than individual terms, store whole situation phrases per intent:
```
stripe.charge_card:
  situation_phrases: [
    "payment bounced", "card declined", "payment failed",
    "charge rejected", "payment didn't go through",
    "transaction failed", "card not working"
  ]

github.create_issue:
  situation_phrases: [
    "something's broken", "regression in prod", "getting errors",
    "not working", "keeps crashing", "404 on the endpoint",
    "memory leak in", "performance degraded"
  ]
```

Matching: fuzzy substring or n-gram overlap (not exact match, so "payment got bounced" still hits).

**Why phrases not terms:**
- "failed" alone matches too broadly (payment failed, build failed, login failed, rocket failed)
- "payment failed" is specific to the payment domain
- Phrase-level matching keeps precision high. Term-level is too noisy.

**Seeding:**
One LLM call per intent at setup time:
> "List 15 situation descriptions (things that are happening, not actions to take) that would
> make someone want to [intent_label]. Each should be 2-5 words. Focus on [app_domain] domain."

Stripe's charge_card gets payment failure phrases. GitHub's create_issue gets engineering
failure phrases. Totally separate vocabularies.

**Learning situation phrases from corrections:**
When a query routes incorrectly AND gets corrected:
1. Detect it's a situation query (no action verb, or low action score)
2. Add the query (or key n-grams from it) to the correct intent's situation_phrases
3. These grow the situation vocabulary over time

This is the same mechanism as learned vectors but operating on the phrase level.

**Mathematical analysis:**

Let Q = query, I = set of intents, for each intent i:
- A(i, Q) = action score (existing BM25-style)
- S(i, Q) = situation score (phrase overlap with situation_phrases[i])
- Total(i, Q) = A(i, Q) + α * S(i, Q)  where α ∈ [0.3, 0.5]

For a pure situation query (A ≈ 0 for all intents):
- Winner = argmax S(i, Q) * α
- Confidence is lower (multiply by α) — intentional, situation inference is less certain

For a mixed query ("the payment failed, retry it"):
- A(charge_card, Q) > 0 from "retry"
- S(charge_card, Q) > 0 from "payment failed"
- Both channels agree → highest total score, high confidence

For a pure action query (normal case):
- A(intent, Q) dominates
- S scores near zero (no situation phrases match)
- Behavior identical to current system

**Negatives and failure modes:**

1. **Many-to-many ambiguity.** "Payment failed" could imply charge_card (retry), 
   refund_payment (give up), create_issue (log), send_message (alert). All are valid.
   System should return multiple candidates at reduced confidence, not pick one.
   Resolution: threshold for situation routing is lower (e.g. 0.15 instead of 0.25);
   return all intents above threshold; let the LLM or human decide.

2. **Domain leakage.** If situation phrases aren't scoped tightly, "failed" in Stripe's 
   charge_card situation phrases could match GitHub queries about failing tests.
   Resolution: situation phrases are scoped per app (each app has its own situation index).
   Route against the app_id first, then situation inference within that app.

3. **Situation phrases are longer → lower per-term IDF weight.**
   A 5-word situation phrase has lower per-term weight than a 1-word action seed.
   Resolution: score situation phrases as a unit (phrase match score) not per-term.
   Binary match (phrase present/absent) + phrase length normalization.

4. **Learning signal is indirect.**
   For action routing, if routing was correct, the user did the thing they asked.
   For situation routing, you don't know if the inference was right without explicit feedback.
   Resolution: use the correction API (`/api/correct`) as the learning signal.
   Don't auto-learn from situation routing — only learn from confirmed corrections.

5. **Action verb detection for mode switching is imperfect.**
   "Ship it" has an action verb ("ship") but is ambiguous (shopify vs github).
   "The team is shipping next week" has "shipping" but it's not a direct command.
   Resolution: don't binary-switch between action/situation mode. Run both always,
   combine scores. Action mode naturally dominates when action verbs are present.

**Enterprise simulation:**

| Query | Situation phrases match | Action score | Combined routing |
|---|---|---|---|
| "payment bounced" | stripe.charge_card (high) | 0 | charge_card at reduced confidence |
| "customer churned" | stripe.cancel_subscription (high) | 0 | cancel_subscription at reduced confidence |
| "the auth is broken" | github.create_issue (medium) | 0 | create_issue at reduced confidence |
| "we're going live" | github.create_release (medium), slack.send_message (medium) | 0 | both at reduced confidence |
| "cancel the subscription" | stripe.cancel_subscription (medium, from "subscription") | high | cancel_subscription high confidence |
| "the customer wants to cancel" | stripe.cancel_subscription (high) | medium ("cancel","wants") | cancel_subscription high confidence |
| "it's down" | github.create_issue (low) | 0 | weak signal, low confidence (correct) |
| "banana" | nothing | 0 | nothing (correct) |

**Implementation path:**
1. Add `situation_phrases: Vec<String>` to intent storage (parallel to existing seeds)
2. Add phrase-level matching in routing (n-gram overlap with normalization)
3. Add `POST /api/intents/situation` endpoint for adding situation phrases
4. LLM generation at app setup time: `generate_situation_phrases(app_id, intent_id)`
5. Learn from corrections: when `correct()` is called on a situation query, extract
   n-grams and add to `situation_phrases` of the correct intent

**NOT recommended:**
- Term-level situation index (too much noise, cross-domain pollution)
- Auto-learning from every query (need confirmed signal)
- Replacing action routing with situation routing (complement, not replace)

---

## Pronoun & Context Resolution (2026-04-10)

"Do the usual for the new client" — "the usual" references something outside this query.

This is an architectural limit, not a vocabulary problem. No amount of seeds solves it.
Resolution requires one of:
- Session memory: previous messages in the conversation
- User-level stored procedures: "for this user, 'the usual' = create_customer + create_subscription"
- Clarification: detect zero-information queries, return clarification request

In production AI agents: LLM with conversation history resolves pronouns before the
router sees the query. Router only receives explicit action descriptions. Pronoun
resolution is upstream of routing, not inside it.

---

## Jargon Coverage (2026-04-10)

### The gap
"Cut the release, push to the eng channel" — "cut" and "push" are not in seeds.
System routes to wrong intents (add_collaborator, invite_user) because "push" matched
"give access to repo" / "add user to channel" seeds.

### Fix options (in order of practicality)

1. **LLM seed expansion at setup** — generate jargon paraphrases per intent, add as seeds.
   One LLM call per intent, zero runtime cost. Best ROI.

2. **Learning from corrections** — when user corrects a jargon routing, terms get learned.
   Works but requires real traffic and correction feedback loop.

3. **Domain glossary** — per-app synonym table: "cut" → "create", "push" → "release", "land" → "merge".
   Applied as a pre-processing step. Simple and fast, but high maintenance.

4. **Embeddings** — semantic similarity, understands "cut" ≈ "create" without seeing it.
   High infrastructure cost. Best generalization. For later.
