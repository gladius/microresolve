# ASV Router — Multi-Intent Problem

## The Problem

ASV assumes one intent per query. When users type paragraphs with multiple intents, scoring breaks.

### Example

> "Cancel my order for the blue shoes and also track the red jacket I ordered last week, oh and can you check if I have any store credit?"

3 intents: `cancel_order`, `track_order`, `check_credit`.

### Why Single-Intent Scoring Breaks

For single-intent query `q` with terms `{t₁, t₂, ...tₙ}`:

```
S(I, q) = Σᵢ max(seed_I(tᵢ), learned_I(tᵢ))
```

All terms relate to one intent. Every matching term is signal.

For multi-intent query `q = q₁ ∪ q₂ ∪ q₃`:

```
S(I₁, q) = Σ over ALL terms (q₁ ∪ q₂ ∪ q₃) max(seed_I₁(tᵢ), learned_I₁(tᵢ))
```

Intent I₁ gets scored against terms from q₂ and q₃. If they share a term (like "order"), it inflates scores with noise. Scores become a muddy sum, not clean signals.

### Concrete Simulation

```
Intent definitions:
  cancel_order:  {cancel: 0.95, order: 0.80, stop: 0.60, return: 0.50}
  track_order:   {track: 0.95, order: 0.80, shipping: 0.70, where: 0.60}
  check_credit:  {credit: 0.90, store: 0.60, balance: 0.70, check: 0.50}
  refund:        {refund: 0.95, money: 0.80, back: 0.50}

Query tokens: [cancel, order, blue, shoes, track, red, jacket, check, store, credit]

Naive scoring:
  cancel_order:  cancel(0.95) + order(0.80) = 1.75
  track_order:   track(0.95) + order(0.80) = 1.75
  check_credit:  credit(0.90) + store(0.60) + check(0.50) = 2.00
  refund:        0.00

Result: check_credit "wins", cancel_order and track_order silently dropped.
```

## Solutions Analyzed

### Option A: Sentence Splitting (Preprocessing)

Split paragraph into sentences, route each independently.

```
"Cancel my order for the blue shoes."     → cancel_order (1.75)
"Also track the red jacket."             → track_order (0.95)
"Can you check if I have store credit?"  → check_credit (2.00)
```

**Pros**: Simple, works for clean text.
**Cons**: Breaks on "Cancel my order and track my other one" (one sentence, two intents). Fails on voice input with no punctuation.

### Option B: Sliding Window Scoring

Score overlapping N-term windows. Detect multiple peaks.

```
Window size 4, stride 2:
  [cancel, order, blue, shoes]     → cancel_order: 1.75
  [blue, shoes, track, red]        → track_order: 0.95
  [track, red, jacket, check]      → track_order: 0.95
  [check, store, credit]           → check_credit: 2.00
```

**Pros**: Handles unstructured text.
**Cons**: Window size tuning, boundary artifacts, duplicate detections.

### Option C: Score Distribution Analysis (Entropy)

High entropy in score distribution = multiple intents.

```
Multi-intent: scores = [2.00, 1.75, 1.75, 0.00]
  normalized = [0.364, 0.318, 0.318, 0.000]
  H = -Σ pᵢ log(pᵢ) = 1.08  (high entropy → multi-intent)

Single-intent: scores = [1.75, 0.80, 0.00, 0.00]
  normalized = [0.686, 0.314, 0.000, 0.000]
  H = 0.63  (low entropy → single intent)
```

**Pros**: Good detection signal.
**Cons**: Detects multi-intent but doesn't decompose WHICH terms belong to WHICH intent.

### Option D: Greedy Term Consumption (Recommended)

1. Score all intents against full query
2. Take highest-scoring intent
3. Remove its matching terms from the query
4. Score remaining terms
5. Repeat until remaining terms score below threshold

```
Pass 1: full query → check_credit wins (2.00)
  Matched terms: [credit, store, check]
  Remove them. Remaining: [cancel, order, blue, shoes, track, red, jacket]

Pass 2: remaining → cancel_order wins (1.75)
  Matched terms: [cancel, order]
  Remove them. Remaining: [blue, shoes, track, red, jacket]

Pass 3: remaining → track_order wins (0.95)
  Matched terms: [track]
  Remove them. Remaining: [blue, shoes, red, jacket]

Pass 4: remaining → no intent scores above threshold (0.5)
  Stop.

Result: [check_credit, cancel_order, track_order] ✓ All three detected.
```

## Why Greedy Consumption Wins

| Criterion | Splitting | Window | Entropy | Greedy |
|-----------|-----------|--------|---------|--------|
| Handles no punctuation | No | Yes | Partial | Yes |
| Decomposes intents | Yes | Partial | No | Yes |
| No tuning parameters | Yes | No | Yes | Yes |
| Handles any # of intents | Yes | Yes | No | Yes |
| Sub-millisecond | Yes | Yes | Yes | Yes |
| Simple to implement | Yes | No | Yes | Yes |

## Algorithm: Greedy Term Consumption

```
function route_multi(query, intents, threshold=0.5):
    terms = tokenize(query)
    detected = []

    while terms is not empty:
        scores = score_all_intents(terms)
        best = max(scores)

        if best.score < threshold:
            break

        detected.append(best)

        # Remove consumed terms
        for term in terms:
            if intent_has_term(best.id, term):
                terms.remove(term)

    return detected
```

**Complexity**: O(k × n) where k = intents detected, n = terms.
Typical paragraph: k=3, n=15 → 45 lookups. Still sub-millisecond.

## Combining with Entropy (Hybrid)

Use entropy as a fast pre-check:

```
1. Score all intents against full query
2. Compute entropy of score distribution
3. If entropy < 0.8 → single intent, return top-1 (fast path)
4. If entropy ≥ 0.8 → multi-intent, run greedy consumption
```

This avoids the multi-pass overhead for the common case (single intent).

## Novel Contribution

Greedy term consumption for multi-intent decomposition appears to be novel. Key properties:

1. **No segmentation model needed** — works on raw token stream
2. **Naturally handles any number of intents** — no predefined limit
3. **Returns intents in priority order** — highest confidence first
4. **Remaining unmatched terms** indicate potential unknown intents
5. **Pure arithmetic** — no neural network, no LLM

This could be a standalone paper section: "Greedy Sparse Decomposition for Multi-Intent Detection."
