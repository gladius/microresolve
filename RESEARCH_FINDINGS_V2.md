# ASV Research Findings V2 — IDF + Token Consumption + Auto-Learn (2026-04-15)

## Final Architecture

```
L0: Typo correction (character trigram Jaccard, unchanged)
L1: Morphology normalization (LexicalGraph, keeps for Latin)
L2: word_intent IDF scoring + token consumption multi-intent
    - score = Σ weight(word,intent) × ln(N/df)
    - Multi-pass: confirm top (within 10%), consume tokens, re-score
    - Gate: round N+1 must score ≥ 55% of round 1 top
L3: Anti-Hebbian inhibition (unchanged)
Auto-learn: LLM extracts key_words from query → learn_query_words()
```

## Scale Test Results (98 intents, 4 domains, 55 queries)

### Exact Match (system returns ONLY the correct intent set)

| Category | Seed only | After learning |
|----------|:---------:|:--------------:|
| Single (29) | 48% | 62% |
| Multi (8) | 62% | 88% |
| Cross-domain (8) | 50% | 38% |
| Indirect (10) | 10% | 80% |
| **TOTAL (55)** | **44%** | **65%** |

### Recall@N (correct intent in top N results)

| Category | Top 1 | Top 2 | Top 3 | Top 5 |
|----------|:-----:|:-----:|:-----:|:-----:|
| Single (29) | 90% | 97% | 97% | 100% |
| Multi (8) | 0% | 100% | 100% | 100% |
| Cross-domain (8) | 88% | 100% | 100% | 100% |
| Indirect (10) | 90% | 90% | 100% | 100% |
| **TOTAL (55)** | **75%** | **98%** | **98%** | **100%** |

### Confidence Distribution
- 65% of queries: clear winner → return directly (zero LLM cost)
- 35% of queries: ambiguous → top 3-5 candidates for LLM disambiguation

### Key Findings

1. **IDF + targeted query word learning works.** Learning actual user vocabulary
   (not LLM paraphrases) keeps IDF discriminative. System reaches 80-100%
   on controlled tests.

2. **Token consumption is the multi-intent solution.** Confirm top intent,
   remove its tokens, re-score remaining. Handles both single-intent precision
   and multi-intent recall in one mechanism.

3. **Bigrams from tokenizer ARE discriminative.** "cancel_order" as a single
   token maps only to cancel_order (IDF=4.58). When verb+object are adjacent
   after stop-word removal, exact match is near-perfect.

4. **Cross-domain is a context problem, not scoring.** "list customers" correctly
   matches both Stripe and Shopify. Disambiguation requires context (namespace,
   conversation history, or LLM from shortlist).

5. **Top-2 candidates achieve 98%.** For the 35% ambiguous cases, sending
   2-3 candidates to LLM for disambiguation is cheap and effective.

6. **Self-regulating IDF.** Over time: intent-specific words float up (reinforced,
   high IDF), noise words sink (spread across intents, IDF drops). No manual
   cleanup needed.

## Learning Model

**What to learn:** LLM-confirmed intent-bearing words from the user's actual query.
NOT LLM-generated paraphrases (different vocabulary, causes IDF dilution).

**Growth curve (from experiment_growth.rs, 8 intents, 30 queries):**
```
Seed:    30% → Wave 1: 57% → Wave 2: 80% → Wave 3: 90% → Final: 100%
```

**Scale (from experiment_scale.rs, 98 intents, 70 queries):**
```
Seed:    27% → Wave 1: 39% → Wave 5: 74% → Final: 80%
Remaining: 14 partial (cross-provider ambiguity, correct intent always found)
```

## Test Validity

- Scale test uses 98 real-world MCP intents (Stripe/Shopify/Linear/Vercel)
- Queries are natural language, not seed phrase paraphrases
- Learning simulates LLM key_word extraction (all content words — slightly noisy)
- Cross-domain test is artificially hard (all 98 intents in one namespace)
- Multi-intent tested with explicit two-intent queries
- 82 unit tests pass
