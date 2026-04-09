# Seed Guard: Insertion-Time Collision Detection

## Problem
Adding a seed phrase to an intent can introduce terms that are primary vocabulary in other intents, diluting their discriminative power. Example: adding "refund to visa" to `refund` puts "visa" (a core `payment_method` term) into the refund index.

No existing system solves this at insertion time. IDF mitigates it at search time but doesn't prevent it. This is novel territory.

## Research Basis
- **TF-IDF-rho** (Zhang & Ge, 2019): class discriminative strength as a multiplier. Terms concentrated in one class get rho~1, terms spread across many get rho~0.
- **Complement Naive Bayes** (Rennie et al., ICML 2003): explicitly uses negative class evidence — terms in OTHER classes count against classification.
- **Chi-square / PMI**: standard feature selection metrics that score how discriminative a term is for a category.
- **c-TF-IDF** (BERTopic): class-level IDF that measures distinctiveness per class, not per document.

All operate on batch data. None provide real-time insertion gating. Our contribution: apply discrimination scoring at `add_seed` time.

## Design

### Core Metric: Discrimination Ratio
For each term in a new seed, compute:

```
discrimination(term, target_intent) = weight_in_target / total_weight_across_all_intents
```

- Ratio close to 1.0 → term is unique to this intent (safe)
- Ratio close to 0.0 → term is primarily in OTHER intents (dangerous)
- Term not in any intent → new vocabulary (safe, introduces fresh signal)

### Three Checks at Insertion Time

**Check 1: Collision Detection**
For each content term in the seed (after tokenization, stop-word removal):
- Look up the term in the inverted index
- If it exists with weight > 0.5 in another intent, flag as collision
- Report: which term, which intent, what weight

**Check 2: Redundancy Detection**
- Tokenize the new seed
- Check how many of its content terms already exist in the target intent
- If ALL content terms are already covered → seed is redundant, adds nothing
- Report: "all terms already in index"

**Check 3: Stop-Word-Only Detection**
- After tokenization + stop word removal, check if any content terms remain
- If zero content terms → seed is useless (e.g., "I want to" → all stop words)
- Report: "no content terms"

### Return Type

```rust
pub struct SeedCheckResult {
    /// Whether the seed was added to the intent
    pub added: bool,
    /// New terms this seed introduces (not previously in this intent)
    pub new_terms: Vec<String>,
    /// Terms that collide with other intents: (term, other_intent, weight_in_other)
    pub conflicts: Vec<(String, String, f32)>,
    /// True if all content terms already exist in this intent
    pub redundant: bool,
    /// Warning message if any issues detected
    pub warning: Option<String>,
}
```

### Behavior

| Scenario | Action | UI Display |
|----------|--------|-----------|
| All terms new or safe | Add, no warning | Green: "Added. New terms: [back, original]" |
| Some terms collide | Add, warn | Yellow: "Added. Warning: 'visa' also primary in payment_method (0.95)" |
| All terms redundant | Skip | Red: "Skipped: all terms already covered by existing seeds" |
| No content terms | Skip | Red: "Skipped: no content terms after stop-word removal" |

**Key decision: always add (except redundant/empty), always warn.** Don't block. The user sees the warning and can remove the seed if they disagree. Corrections via `learn()`/`correct()` always add without blocking — user ground truth overrides the guard.

### Where It Applies

| Method | Collision Check? | Block on Collision? |
|--------|-----------------|-------------------|
| `add_seed()` | Yes | Warn, don't block |
| `learn()` | Yes | Warn only (ground truth) |
| `correct()` | Yes | Warn only (ground truth) |
| `import_json()` | No (bulk, too expensive) | — |
| `merge_learned()` | No (CRDT, trust remote) | — |

### Implementation Steps

**Step 1: Add `check_seed()` method to Router (~1 hour)**
```rust
pub fn check_seed(&self, intent_id: &str, seed: &str) -> SeedCheckResult
```
- Tokenize the seed
- For each term, look up in inverted index
- Compute discrimination ratio
- Return SeedCheckResult

**Step 2: Modify `add_seed()` to return SeedCheckResult (~30 min)**
- Currently returns `bool` (added or not due to limit)
- Change to return `SeedCheckResult`
- Add redundancy check: if all terms exist, return added=false
- Add collision detection: populate conflicts
- Still add the seed (unless redundant)

**Step 3: Update server endpoint (~30 min)**
- `POST /api/intents/add_seed` returns the full SeedCheckResult
- Review fix endpoint collects warnings from all seeds added
- Return warnings to UI

**Step 4: Update UI (~1 hour)**
- Review page: show warnings next to each applied seed
- Intent page: show warning when manually adding seeds
- Color-coded: green (clean), yellow (collision warning), red (skipped)

**Step 5: Add discrimination score to index stats (~30 min)**
- `Router.term_discrimination(term) -> HashMap<String, f32>` — shows how each intent claims this term
- Useful for debugging: "why is 'order' causing problems?" → shows it's in 14 intents

### What This Does NOT Do
- Does not block seed addition (except redundant/empty)
- Does not require LLM calls
- Does not change search-time scoring (IDF still handles that)
- Does not affect import/merge (bulk operations skip the check)
- Does not solve the fundamental shared-vocabulary problem (IDF does that) — it just warns the user

### Future Enhancement: Automatic Term Weighting
Instead of just warning, the system could automatically downweight conflicting terms in the new seed. If "visa" is 0.95 in payment_method, and you add "refund to visa" to refund, "visa" could get weight 0.2 in refund instead of the default. This is Rocchio negative centroid in spirit. Not in v1 — just warning first, see if users want automatic adjustment.

## References
- Zhang & Ge (2019). Class Specific TF-IDF Boosting. ACM.
- Rennie et al. (2003). Tackling the Poor Assumptions of Naive Bayes. ICML.
- BERTopic c-TF-IDF. https://maartengr.github.io/BERTopic/getting_started/ctfidf/
- Forman (2003). Extensive Empirical Study of Feature Selection Metrics. JMLR.
- Sparck Jones (1972). A Statistical Interpretation of Term Specificity. JDoc.
