# ASV Router — Weaknesses and Mitigations

## Critical (must address before shipping)

### 1. Semantic Blindness

The biggest weakness. ASV is purely lexical — it matches words, not meaning.

```
"cancel my order"        → terms: [cancel, order]     → CancelOrder ✓
"I changed my mind"      → terms: [changed, mind]     → ??? (zero overlap)
"stop charging me"       → terms: [stop, charging]     → ??? (zero overlap)
```

Same intent, completely different words. ASV scores 0.

**Integration test finding (March 2026):** The gap is **narrower than originally claimed**. With 4 seed phrases per intent, partial term overlap catches more than expected:
- "give me my cash back" → routes to `refund` via "back" overlap with seed "money back" (score 0.52)
- "let me talk to your manager" → routes to `contact_human` via "talk" overlap with seed "talk to a human" (score 0.95)
- "my login isn't working anymore" → routes to `reset_password` via "log"+"in" from seed "can't log in" (score 0.63)
- "I changed my mind about that purchase" → routes to `cancel_order` via "purchase" from seed "cancel the purchase" (score 0.52)

**Still fails on truly zero-overlap paraphrases:**
- "I changed my mind completely" → no route (no seed terms present)
- "authentication keeps failing" → no route
- "I demand to escalate" → no route
- "put the amount on my card again" → no route

**Mitigation:**
- LLM seeder generates 50-100 paraphrases per intent, covering synonyms
- `router.expand_seeds()` function using synonym substitution (no LLM needed for basic coverage)
- Document honestly — the coverage is visible and auditable
- With aggressive seeding (4+ phrases), partial overlap covers more than zero-overlap analysis suggests

### 2. Negation Blindness

```
"cancel my order"       → [cancel, order]    → CancelOrder
"DON'T cancel my order" → [cancel, order]    → CancelOrder  (same!)
```

ASV strips "don't" as a stop word. Opposite intent, identical score.

**Integration test finding:** The apostrophe in "don't" causes a **compounding problem** in multi-intent routing. The tokenizer splits "don't" into "don" + "t". The orphaned "t" can be consumed by unrelated intents — e.g., `reset_password` consumes "t" because its seed "can't log in" also produces a "t" term. This collapses the gap words between intent spans, breaking relation detection:

```
"cancel my order but don't reset my password"
                        ^^^^^^
Expected gap: "but don t" → Negation detected
Actual gap:   "but don"   → Parallel (wrong!) because "t" consumed by reset_password
```

**Workaround:** Use "except"/"without" instead of "don't" for negation — these are single tokens that survive intact.

**Mitigation — negation-aware tokenizer:**
```
"don't cancel" → ["cancel", "not_cancel"]   // negative marker term
"never cancel" → ["cancel", "not_cancel"]
```

The `not_cancel` term won't match any intent seeded with "cancel", creating natural separation. ~20 lines in the tokenizer. This would also fix the "t" orphan problem by consuming the negation as a unit.

### 3. Correction Cannot Override Strong Seeds

```
router.add_intent("cancel_order", &["stop my order", ...]);  // "stop" in seed at weight 0.62
router.learn("stop the delivery", "cancel_order");            // learns it as cancel
router.correct("stop the delivery", "cancel_order", "refund"); // tries to move to refund
router.route("stop the delivery");  // → STILL cancel_order (0.62 seed > 0.45 learned)
```

`correct()` only modifies the **learned layer** via `unlearn()`. It cannot touch the **seed layer** (by design — seeds are immutable). When a query shares terms with a strong seed, correction is powerless.

**Impact:** Medium. In practice this means some term associations are permanently locked by seeding. Users must re-seed the intent with better phrases rather than relying on `correct()`.

**Mitigation:**
- Document clearly: `correct()` works for learned associations, not seed conflicts
- Add `router.update_seeds(intent_id, new_phrases)` to allow re-seeding
- Or: add a `seed_override` layer that can suppress specific seed terms per intent

### 4. No Dialogue Context

```
Agent: "Should I cancel your order?"
User:  "Yes"
```

ASV sees "yes" — a stop word. Returns empty. No memory of what "yes" refers to.

**Mitigation:**
- Add `route_with_context(query, previous_intent)` that prepends previous intent's terms at reduced weight
- Or: document as out-of-scope. ASV routes single turns. Conversation manager sits above it.

## Important (address before v1.0)

### 5. No Typo Tolerance

```
"cancl my order" → [cancl, order] → misses "cancel" entirely
```

One typo breaks routing. Real users misspell constantly.

**Mitigation:**
- Edit-distance fuzzy matching: if a query term doesn't match any posting, find closest term within Levenshtein distance 1-2
- ~40 lines. Big usability win for real deployments.

### 6. No Confidence Score

Score 2.3 — is that good? If top-1 scores 2.3 and top-2 scores 2.2, that's ambiguous. If top-1 scores 2.3 and top-2 scores 0.1, that's confident.

**Mitigation:**
```rust
pub struct RouteResult {
    pub id: String,
    pub score: f32,
    pub confidence: f32,  // top1_score / (top1_score + top2_score)
}
```

Simple ratio. If confidence < 0.6, caller knows to ask for clarification or escalate to LLM.

### 7. English-Only Tokenizer

Stop words, bigram logic, word boundaries — all English. Won't work for Hindi, Chinese, Japanese, Arabic.

**Mitigation:**
- Pluggable tokenizer: `Router::with_tokenizer(custom_fn)`
- Ship English as default, users provide their own for other languages
- For v1.0: document as "English-only, pluggable tokenizer planned"

## Minor (nice to have)

| Weakness | Impact | Fix |
|----------|--------|-----|
| Homonym confusion ("bank" = money or river) | Low — bigrams disambiguate | "river bank" vs "bank account" already separate |
| No entity extraction | Out of scope | ASV routes, doesn't extract. Pair with regex or NER |
| Full rebuild on add_intent | Slow at 10K+ intents | Already have `update_intent()`, skip full rebuild |
| No term weighting decay in seeds | Stale seeds after months | Seeds are immutable — regenerate them |

## Cold Start

New intents with 1-2 seed phrases perform poorly. Needs either:
- LLM-generated diverse paraphrases for seeding (50-100 per intent)
- Or: several rounds of user correction via `learn()` / `correct()`

The system is designed to improve over time, but the first few interactions will be rough without good seeding.

## Predicted Benchmark Performance

Based on integration testing and architectural analysis:

| Dataset | Intents | Seed-Only | After Learning | Why |
|---------|---------|-----------|----------------|-----|
| SNIPS | 7 | ~85-95% | ~95%+ | Distinct domains, clear keywords |
| ATIS | 8 | ~80-90% | ~90%+ | Single domain but distinct verbs |
| CLINC150 | 150 | ~30-50% | ~60-75% | Many overlapping intents across 10 domains |
| HWU64 | 64 | ~35-55% | ~65-80% | Intra-domain overlap (alarm_set vs alarm_query) |
| BANKING77 | 77 | ~20-35% | ~50-65% | Single domain, near-identical vocabulary |
| HINT3 | 21-59 | ~15-30% | ~40-55% | Real production queries, heavy OOS |

These predictions will be validated by benchmarks. The key comparison points:
- Rasa DIET (full data): 89.4% CLINC150, 89.9% BANKING77
- GPT-4 3-shot: ~90% CLINC150, 83.1% BANKING77
- SetFit 8-shot cosine: 85.9% CLINC150, 77.9% BANKING77
- TF-IDF baseline: ~81-85% CLINC150
