# Situation→Action Inference — Experiment Tracking

## Hypothesis

A query like "the payment bounced" contains no action verb and no seed overlap, yet a human
instantly knows it implies `stripe.charge_card`. The system currently scores 0 because it only
indexes action vocabulary ("charge", "bill", "process").

**Hypothesis:** A parallel phrase-level index keyed on *situation vocabulary* ("bounced",
"declined", "churned", "broken in prod") can:

1. Detect the correct **app** from a situation description alone (routing layer 1)
2. Boost the correct **intent** within that app (routing layer 2)
3. Combine with existing action scoring to improve mixed queries

This is independent of embeddings — pure sparse phrase matching, seeded and learned
the same way as the term index. Zero runtime model cost.

---

## Why a Separate Binary

- No risk to the main routing system during experimentation
- Can tune scoring aggressively (thresholds, α values, phrase lengths)
- Can test app detection and intent detection as separate metrics
- Fast iteration: change algorithm, recompile, re-run in seconds
- Once we know the architecture works, we integrate the winning design into Router

**Binary:** `src/bin/situation_exp.rs`
Uses ASV's tokenizer (same library) but implements its own SituationStore.

---

## Architecture of the Experiment

### SituationStore

```
HashMap<(app_id, intent_id), Vec<SituationPhrase>>

SituationPhrase {
    phrase: String,           // raw phrase e.g. "payment bounced"
    tokens: HashSet<String>,  // pre-tokenized, stop words removed
    weight: f32,              // 1.0 for seeded, 0.4 for learned
}
```

### Scoring (per query Q)

```
For each (app, intent) with situation phrases:
  intent_score = 0
  for each phrase P:
    overlap = |tokens(P) ∩ tokens(Q)| / |tokens(P)|
    if overlap >= 0.6:
      intent_score += overlap × P.weight × sqrt(len(P.tokens))
      # sqrt(len) rewards longer, more specific phrases

app_score[app] = sum of intent_scores for all intents in that app
                 (not max — many matching intents = higher app confidence)
```

### Two-level detection

**Layer 1 — App detection:**
Which app does this situation belong to?
`top_app = argmax app_score`

Situation vocabulary is naturally domain-separated:
- "payment bounced / declined / chargeback" → Stripe world
- "PR approved / build red / memory leak" → GitHub world
- "channel noisy / team needs to know" → Slack world
- "inventory depleted / customer returned" → Shopify world
- "double booked / can't make it / going on vacation" → Calendar world

**Layer 2 — Intent detection (within app):**
Given app, which specific intent?
`top_intent = argmax intent_score within detected_app`

Or: without app constraint → global argmax across all (app, intent) pairs.

### Combined scoring with action routing

```
total(app, intent, query) = action_score(intent, query)
                           + α × situation_score(app, intent, query)
```

α is tuned experimentally. Default start: 0.4.

For a pure situation query: action_score ≈ 0, situation score dominates.
For a pure action query: situation score ≈ 0, action dominates (no regression).
For a mixed query: both contribute, boosting confidence.

### Learning

From a corrected query (query, correct_app, correct_intent):
1. Tokenize query, remove stop words
2. Extract all bigrams and trigrams from content words
3. Add top 3 n-grams (by length) as new situation phrases at weight 0.4
4. Do NOT learn from auto-detected routing — only from corrections

---

## Test Cases Design

### Category A — Pure situation queries (no action verb)
Query describes what is happening. System must infer action purely from situation vocabulary.
These are the hardest. Expected: system fails most in Round 1, improves after learning.

20 test cases across 5 apps.

### Category B — Cross-app ambiguous situations
Query could plausibly belong to multiple apps. Test whether app detection still works.
5 test cases.

### Category C — Mixed (situation + action)
Query has both a situation description and an action verb. Test that combined scoring
is better than either alone.
5 test cases.

### Category D — Negative (no situation implied)
Queries with no clear situation. Test that the system stays low-confidence.
5 test cases.

---

## Phases

### Phase 1 — Core engine ✅ (implement first)
- `SituationStore` with phrase storage and scoring
- Token overlap scoring with length bonus
- App-level aggregate scoring
- Intent-level scoring
- Hand-seeded situation phrases for 5 apps × key intents

### Phase 2 — Test harness ✅ (implement first)
- Load 35 test cases (20 + 5 + 5 + 5)
- Round 1: seed phrases only → measure baseline
- Learning pass: extract n-grams from failed cases, add as phrases
- Round 2: measure improvement
- Output: precision/recall/F1 per app, per category

### Phase 3 — Combined scoring (after Phase 1+2 show signal)
- Import Router from asv_router library
- Load same 5-app setup from cross_app_learn_test.py
- Run action scoring on same test queries
- Tune α ∈ {0.2, 0.3, 0.4, 0.5}
- Compare: situation-only vs action-only vs combined
- Measure lift on the hard-tier cases specifically

### Phase 4 — LLM seed generation (after Phase 3 confirms architecture)
- For each (app, intent): prompt LLM for 15 situation descriptions
- Auto-generate the situation phrase corpus
- Re-run Phase 2 with LLM-seeded phrases vs hand-seeded phrases
- Measure: does LLM seeding outperform hand-seeding?

---

## Success Criteria

| Metric | Minimum to proceed | Target |
|---|---|---|
| App detection (situation only) | > 60% accuracy | > 80% |
| Intent detection within app (situation only) | > 40% | > 60% |
| Lift on hard-tier queries vs action-only | > 3 new cases fixed | > 6 |
| No regression on clean action queries | 0 regression | 0 regression |
| Learning loop improvement | > 5% after one pass | > 15% |

If app detection < 60%: situation vocabulary is not domain-separated enough. Need longer phrases.
If no regression found: safe to integrate.

---

## Integration Plan (after experiment succeeds)

1. Add `situation_phrases: HashMap<String, Vec<String>>` to Router (per intent)
2. At `route_multi()` time: run situation scoring in parallel, add α × score to each intent
3. Add `POST /api/intents/situation` endpoint to add/manage situation phrases
4. Add LLM situation phrase generation to the review/improve pipeline
5. Learn situation phrases from `correct()` calls (n-gram extraction)

**Not in scope for integration:** changing the inverted index, changing seed/learned structure,
embedding models.

---

## Results Log

### Round 0 — Before experiment (known from hard-tier test)
- "payment bounced" → nothing fired
- "customer churned" → nothing fired
- "cut the release" → wrong intents (jargon mismatch)
- System completely blind to situation vocabulary

### Round 1 — (fill in after running Phase 1+2)

### Round 2 — (fill in after learning pass)

### Phase 3 — Combined scoring results (fill in)

---

## Open Questions

1. Should situation scores combine ADDITIVELY (current plan) or via a separate confidence channel?
   Additive is simple but could let situation score override a very confident action routing.
   Alternative: situation score only applies when action_score < threshold.

2. What minimum phrase token length is right? Currently: 0.6 overlap of ≥2 token phrases.
   "Failed" (1 token) is too broad. "Payment failed" (2 tokens) is good.
   Could enforce minimum 2 content tokens per phrase.

3. Does app-level aggregate scoring (sum of intent scores) or max help more for app detection?
   Sum rewards apps where multiple intents match (broad situational relevance).
   Max rewards apps where one intent matches strongly (specific situational relevance).
   The experiment will tell us.

4. When the system detects an app from a situation query but is uncertain about the intent —
   should it return the app with low-confidence, or wait for intent certainty?
   Useful for product: "I know this is Stripe, ask user what they want to do."
