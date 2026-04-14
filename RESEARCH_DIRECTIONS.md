# ASV Multi-Intent Research — Findings & Architecture

## Session Findings (2026-04-14)

### Benchmark Results

IT Helpdesk domain: 24 intents, 5 personas, 41 frozen queries.

| Stage | Exact Pass | Notes |
|-------|-----------|-------|
| Baseline (with 30% ratio rule) | 2.4% | 97.6% partial — correct intents drowned in 6-11 false positives |
| After removing 30% ratio rule | 12.2% | 80.5% partial — false positives reduced, false negatives exposed |
| After auto-learn (wrong endpoint) | 12.2% | Zero change — was calling `/api/review/learn/now` (404), fixed to `/api/learn/now` |
| With learning (correct endpoint) | NOT YET RUN | — |

### Gap Filter Fix

Removed the OR branch from `apply_multi_filter` in `src/hebbian.rs`:

**Before:** `score >= threshold AND (top - score <= gap OR score >= top * 0.30)`  
**After:** `score >= threshold AND top - score <= gap`

The 30% ratio rule meant: if top=5.0, include anything scoring ≥1.5. For long chatty queries
this included 6-11 intents (most of which were wrong).

### Experiment Results (Directions 1-3)

Three experiments built and run: OMP, Coordinator Split, Co-occurrence PMI.

| Experiment | Result | Why |
|-----------|--------|-----|
| OMP (`experiment_omp.rs`) | +0 cases (67% → 67%) | Clean test cases already handled by baseline gap filter |
| Coord Split (`experiment_coord_split.rs`) | +0 cases (78% → 78%) | Same — baseline handles coordinator queries fine |
| PMI (`experiment_cooccur.rs`) | -1 case (60% → 40%) | Over-boosting caused new false positive |

**Key insight from experiments**: The test cases were too clean. They used vocabulary that closely
matches seed phrases. The real benchmark failures are different — users express intent through
word combinations and conversational patterns, not through domain vocabulary.

---

## The Real Problem: Unigram Blindness

### What the system currently learns

```
word_intent: HashMap<String, Vec<(String, f32)>>

"password"  → [("account:reset_password", 0.85)]
"vpn"       → [("network:vpn", 0.80)]
"escalate"  → [("tickets:escalate_ticket", 0.75)]
```

Every token is independent. The system cannot distinguish:
- "been waiting" → frustration → escalate
- "been" → nothing
- "waiting" → check_ticket_status (marginally)

### What humans actually say

Users don't say "escalate my ticket". They say:

| What they say | What it means | Tokens that matter |
|---|---|---|
| "I've been waiting all morning" | escalate_ticket | "been waiting" (bigram) |
| "this is ridiculous" | escalate_ticket | "this is ridiculous" (trigram) |
| "I'm done with this" | escalate_ticket | "done with" (bigram) |
| "literally going to cry" | escalate_ticket | "going to cry" (trigram) |
| "can't get into my account" | reset_password | "can't get into" (trigram) |
| "thing that connects me" | vpn or wifi | "connects me" (bigram) |

No individual token carries the meaning. The **combination** does.

### The finite space argument

In a constrained domain (24 IT helpdesk intents):
- Each intent has maybe 100-300 common expression patterns
- Total: ~5,000-7,000 n-gram patterns across all intents
- This is **finite and learnable**
- With enough LLM-supervised simulation, the system converges
- New unseen patterns become increasingly rare
- The system saturates to near-perfect accuracy within the domain

This is provably convergent: the space of expression patterns is bounded by domain
constraints. Each LLM correction adds patterns. The coverage monotonically increases.

---

## Direction 4: Word N-gram Intent Learning (PRIMARY DIRECTION)

### Architecture Change

Extend IntentGraph to learn multi-word patterns alongside single tokens:

```rust
pub struct IntentGraph {
    // Existing: unigram → intent associations
    pub word_intent: HashMap<String, Vec<(String, f32)>>,
    
    // NEW: n-gram → intent associations (n=2,3,4,5)
    pub phrase_intent: HashMap<String, Vec<(String, f32)>>,
    
    // ... existing fields unchanged
}
```

N-gram keys use underscore joining: `"been_waiting"`, `"this_is_ridiculous"`,
`"can't_get_into"`.

### Scoring Formula

```
score(I, Q) = Σ_{unigram ∈ Q}  w(unigram→I) × IDF(unigram)        # existing
            + Σ_{ngram ∈ Q}    w(ngram→I)   × IDF(ngram) × bonus   # NEW
```

Where:
- N-grams are generated from Q at lengths 2, 3, 4, 5
- `IDF(ngram)` is naturally high because n-grams are more discriminative
  (fewer intents share "been_waiting" than share "waiting")
- `bonus` = length-based multiplier (bigram: 1.5×, trigram: 2.0×, 4-gram: 2.5×)
  because longer matches are stronger evidence

### Why N-grams Solve Both Problems

**False negatives (missing intents):**
- "I've been waiting all morning" → "been_waiting" bigram scores for escalate_ticket
- Even without the word "escalate", the PATTERN fires
- Each learned n-gram is highly discriminative (high IDF) → strong score contribution
- Secondary intents rise above threshold

**False positives (wrong intents):**
- N-grams are MORE specific than unigrams → fewer false matches
- "software not working" as a trigram → troubleshoot_app ONLY
- vs. "software" alone → install, license, update, uninstall, troubleshoot (all score)
- Higher specificity = fewer false positives

### LLM Distillation Change

**Current distillation** (auto-learn):
```
LLM input:  "user said X, missed intent Y"
LLM output: "new phrase for Y" → tokenized into unigrams → learned
```

**Proposed distillation**:
```
LLM input:  "user said X, correct intent is Y. Which SPANS of the query express Y?"
LLM output: ["been waiting all morning", "getting ridiculous"]
→ Learn as n-grams: "been_waiting_all_morning" → Y, "getting_ridiculous" → Y
→ Also learn sub-n-grams: "been_waiting" → Y, "all_morning" → Y
```

This distills the LLM's *reasoning* (which parts of the query map to which intent)
not just its *conclusion* (generate a synonym phrase).

### N-gram Generation From Query

For query "I've been waiting all morning":
Tokens: ["been", "waiting", "all", "morning"]

Generated n-grams:
- Bigrams: "been_waiting", "waiting_all", "all_morning"
- Trigrams: "been_waiting_all", "waiting_all_morning"
- 4-grams: "been_waiting_all_morning"

Score each against `phrase_intent`. Most will miss (not learned yet).
The ones that hit carry strong, discriminative signal.

### Saturation Curve (Expected)

```
Queries seen    Coverage    Accuracy (est.)
0               seed only   12%
50              +patterns   30-40%
200             common      55-65%
500             most        75-85%
1000            saturating  90-95%
2000+           saturated   95%+
```

Each LLM-supervised query adds 1-3 n-gram patterns. The domain has ~5000 total
patterns. After ~1500 diverse queries, 90%+ of patterns are covered.

### Implementation Plan

**Phase 1 — experiment** (`src/bin/experiment_ngram_intent.rs`):
1. Extend IntentGraph with `phrase_intent` field
2. Implement `learn_ngram(tokens: &[&str], intent, n_range: 2..=5)`
3. Implement `score_with_ngrams(query) → Vec<(String, f32)>`
4. Seed: normal unigram phrases + hand-crafted n-gram patterns for escalate_ticket
5. Test: queries that fail with unigrams but succeed with n-grams
6. Verify: no regression on clean single-intent queries

**Phase 2 — library** (`src/hebbian.rs`):
- Add `phrase_intent: HashMap<String, Vec<(String, f32)>>` to IntentGraph
- Modify `score_normalized()` to also scan n-grams
- Add `learn_ngram_phrase()` method
- Serialization: include phrase_intent in save/load

**Phase 3 — LLM distillation** (`src/bin/server/pipeline.rs`):
- Change auto-learn prompt: ask LLM for intent-bearing spans, not new phrases
- Extract spans → generate n-grams → learn into phrase_intent
- Keep existing unigram learning as fallback

**Phase 4 — saturation tracking**:
- Track `phrase_intent` size over time
- Log "new patterns learned" per session
- When new patterns/session drops below threshold → domain is saturated

---

## Revised Architecture

```
Query
  │
  ├─ L0: Typo correction (character n-gram, existing)
  │
  ├─ L1: Synonym expansion (LexicalGraph, existing)
  │
  ├─ L2: Intent scoring (EXTENDED)
  │    ├─ Unigram scoring: word_intent (existing)
  │    ├─ N-gram scoring: phrase_intent (NEW — bigram through 5-gram)
  │    └─ Combined score with n-gram bonus weighting
  │
  ├─ L3: Lateral inhibition (existing, unchanged)
  │
  ├─ Multi-intent filter: gap-based (existing, fixed)
  │
  └─ If confidence < threshold:
       └─ LLM fallback (existing pattern, ~20% of queries)
```

The n-gram layer slots cleanly into L2. No new layers needed.
The LLM distillation change is in the auto-learn pipeline only.

---

## Directions 1-3: Status and Role

### Direction 1: OMP Residual Detection
**Status**: Experiment built, +0 improvement on clean data.
**Role after n-grams**: May still help for queries where one intent's unigrams dominate.
Worth re-testing after n-gram integration — n-grams may make OMP unnecessary since
n-gram scoring naturally distributes signal across intent-bearing phrases.

### Direction 2: Coordinator Segmentation
**Status**: Experiment built, +0 improvement on clean data.
**Role after n-grams**: Complementary. Splitting + n-gram scoring per fragment could be
powerful. Low implementation cost. Worth keeping as Phase 2.

### Direction 3: Co-occurrence PMI
**Status**: Experiment built, -1 regression.
**Role after n-grams**: Lower priority. PMI boost is fragile (causes false positives).
Only useful after Phase 1 (n-grams) proves insufficient. Deprioritize.

---

## Execution Order

```
Priority 1: N-gram intent learning (Direction 4)
  └─ experiment_ngram_intent.rs  ← BUILD FIRST
  └─ Integrate into hebbian.rs
  └─ Change LLM distillation to extract spans

Priority 2: Coordinator segmentation (Direction 2)
  └─ Only if n-grams alone don't reach 80%+ on benchmark
  
Priority 3: OMP residual (Direction 1)
  └─ Re-test after n-gram integration

Priority 4: Co-occurrence PMI (Direction 3)
  └─ Only if still needed after 1-3
```

---

## Benchmark Validation

All changes validated against frozen `/tmp/asv_benchmark.json` (41 queries, IT helpdesk).

| Metric | Baseline (current) | Target after n-grams | Target final |
|--------|-------------------|---------------------|--------------|
| Exact pass | 12.2% | >35% | >60% |
| Recall | ~50% (est.) | >75% | >90% |
| Precision | ~65% (est.) | >80% | >90% |
| F1 | ~56% (est.) | >77% | >90% |

The 60% exact pass target acknowledges that some benchmark queries have implicit intents
(frustration → escalation) that require multiple learning cycles to capture via n-gram
patterns. With enough LLM-supervised simulation, even these should be learnable.
