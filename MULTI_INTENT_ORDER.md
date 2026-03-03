# ASV Router — Multi-Intent Positional Decomposition

## The Ordering Problem

Greedy consumption detects multiple intents but returns them by **score strength**, not **position in the user's sentence**.

```
"Cancel my order and also check my store credit"

Greedy score order:    [check_credit (2.00), cancel_order (1.75)]
User's actual order:   [cancel_order, check_credit]
```

The user said "cancel" first — that's their primary intent. Score order gets this wrong.

## Solution: Position-Aware Term Tracking

Every token has a position in the original query. Track it through greedy consumption.

```
"Cancel my order and also track the red jacket and check store credit"

Positioned tokens (after stop word removal):
  pos 0:  cancel
  pos 2:  order
  pos 5:  track
  pos 7:  red
  pos 8:  jacket
  pos 10: check
  pos 11: store
  pos 12: credit
```

During greedy consumption, each intent consumes specific terms. Record their positions:

```
Pass 1: check_credit consumes check(10), store(11), credit(12) → min_pos = 10
Pass 2: cancel_order consumes cancel(0), order(2)              → min_pos = 0
Pass 3: track_order consumes track(5)                          → min_pos = 5
```

Re-sort by min_position:

```
[cancel_order (pos 0), track_order (pos 5), check_credit (pos 10)]
```

Matches the user's actual ordering perfectly.

## Result Structure

```rust
pub struct MultiRouteResult {
    pub id: String,
    pub score: f32,
    pub position: usize,        // first word position in original query
    pub span: (usize, usize),   // (start, end) word positions consumed
}
```

## Algorithm

```
function route_multi(query):
    positioned_terms = tokenize_with_positions(query)
    remaining = positioned_terms.clone()
    detected = []

    while remaining is not empty:
        scores = score_all_intents(remaining.terms_only())
        best = max(scores)

        if best.score < threshold:
            break

        consumed_positions = []
        new_remaining = []

        for (term, pos) in remaining:
            if intent_has_term(best.id, term):
                consumed_positions.push(pos)
            else:
                new_remaining.push((term, pos))

        detected.push({
            id: best.id,
            score: best.score,
            position: min(consumed_positions),
            span: (min(consumed_positions), max(consumed_positions)),
        })

        remaining = new_remaining

    detected.sort_by(|a, b| a.position.cmp(&b.position))
    return detected
```

Complexity: O(k × n) where k = detected intents, n = terms. Still sub-millisecond.

## Intent Relation Detection

Knowing positions enables detecting **relationships** between intents by scanning the gap words between spans.

### Example

```
"Transfer my balance AND THEN close the account"
                      ^^^^^^^^
                      gap words between span 1 (pos 0-3) and span 2 (pos 5-7)

Intent 1: transfer_balance (span 0..3)
Intent 2: close_account (span 5..7)
Gap text: ["and", "then"]
Relation: Sequential
```

### Relation Types

| Gap Pattern | Relation | Meaning |
|-------------|----------|---------|
| "and then", "after that", "once done", "next", "followed by" | **Sequential** | Execute in order |
| "and", "also", "plus", "as well" | **Parallel** | Independent, any order |
| "but first", "before that", "first" | **Reverse** | Second intent executes first |
| "or", "otherwise", "if not", "failing that" | **Conditional** | Second is fallback |
| "but don't", "except", "not", "without" | **Negation** | Second intent is excluded |

### Detection Logic

```
function detect_relation(original_words, span1, span2):
    gap = original_words[span1.end + 1 .. span2.start]
    gap_lower = gap.join(" ").lowercase()

    if contains_any(gap_lower, ["then", "after", "next", "once done", "followed by"]):
        return Sequential(first=span1.intent, then=span2.intent)

    if contains_any(gap_lower, ["or", "otherwise", "if not", "failing that"]):
        return Conditional(primary=span1.intent, fallback=span2.intent)

    if contains_any(gap_lower, ["but first", "before that"]):
        return Reverse(do_second=span1.intent, do_first=span2.intent)

    if contains_any(gap_lower, ["but don't", "except", "without", "not"]):
        return Negation(do_this=span1.intent, not_this=span2.intent)

    return Parallel  // default
```

### Result Structure

```rust
pub struct MultiRouteOutput {
    /// Detected intents in positional order (left to right in user's message).
    pub intents: Vec<MultiRouteResult>,
    /// Relations between consecutive intents.
    pub relations: Vec<IntentRelation>,
}

pub enum IntentRelation {
    /// "and", "also" — independent, any execution order
    Parallel,
    /// "and then", "after that" — must execute in order
    Sequential { first: usize, then: usize },
    /// "or", "otherwise" — second is fallback if first fails
    Conditional { primary: usize, fallback: usize },
    /// "but first" — reverse the stated order
    Reverse { stated_first: usize, execute_first: usize },
    /// "but don't", "except" — exclude second intent
    Negation { do_this: usize, not_this: usize },
}
```

### Full Example

```
Input: "Transfer my balance, then close the account, but don't delete my profile"

Detected intents (positional order):
  [0] transfer_balance  (span 0..3)
  [1] close_account     (span 5..7)
  [2] delete_profile    (span 10..12)

Relations:
  [0→1] Sequential (gap contains "then")
  [1→2] Negation (gap contains "don't")

Execution plan:
  1. transfer_balance   ← execute
  2. close_account      ← execute after transfer completes
  3. delete_profile     ← DO NOT execute
```

## Complexity Summary

| Operation | Time | Notes |
|-----------|------|-------|
| Tokenize with positions | O(n) | Single pass over query |
| Greedy consumption (k passes) | O(k × n) | k = intents detected |
| Position sort | O(k log k) | Typically k ≤ 5 |
| Relation detection | O(k × g) | g = avg gap size, typically small |
| **Total** | **O(k × n)** | k=3, n=15 → 45 lookups. Sub-millisecond. |

## Novel Contribution

This combination appears to be novel:

1. **Greedy sparse decomposition** — multi-intent detection via term consumption
2. **Positional tracking** — recover user's original ordering from consumed term positions
3. **Relation detection** — infer execution semantics (sequential, conditional, negation) from gap words
4. **Pure arithmetic** — no segmentation model, no NER, no LLM

All three operate on the same token stream. No additional models or preprocessing. The entire multi-intent pipeline adds ~150 lines to the router and stays sub-millisecond.
