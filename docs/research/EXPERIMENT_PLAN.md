# ASV Multi-Intent Accuracy — Experiment Plan

## Problem Statement

ASV routes short queries (1-5 words) well (53% pass) but degrades to 0-8% for
realistic customer support messages (11+ words). The root cause: additive scoring
of generic vocabulary. Terms like "order", "account", "charge" appear across many
intents and accumulate scores that produce 5-10 false positive intents per query.

**Baseline: 10.1% pass rate on 30 multi-turn scenarios (138 turns)**

The correct intent IS detected 84% of the time — it's buried under false positives.
This is a specificity problem, not a sensitivity problem.

---

## Experimental Ideas (A through H)

Each experiment is independently testable against the 30 scenarios.
No LLM cost. All operate on existing ASV data structures.


### Test A: SymSpell Correction (ASV vocabulary as dictionary)

**Idea:** Use inverted index keys as the spell-check dictionary. For each query
term not found in the index, find the closest known term by edit distance (<=2).
SymSpell precomputes all possible edits — lookup is O(1).

**Why it could work:** Real users type "cancl", "refnd", "ordr". These miss the
index entirely, losing signal. Correcting them recovers it.

**Extension:** Also include learned-layer terms as dictionary entries (with caution
— only terms with weight > 0.5 to avoid noise).

**Implementation:**
- Build SymSpell dictionary from `index.terms()` at router init
- Before tokenization, run each whitespace-split token through SymSpell
- Measure: pass rate with vs without correction
- Also test: introduce random 1-2 char errors into scenario messages, measure delta

**Risk:** Low. Can only map to known vocabulary. No hallucination.

**Expected impact:** Moderate for production (real typos), minimal for current
test scenarios (messages are correctly spelled).


### Test B: IDF Noise Gate (multi-intent only)

**Idea:** Before multi-intent scoring, classify every query term by IDF.
Terms with IDF below a threshold are "noise" — they match many intents weakly.
Exclude noise terms entirely from the multi-intent greedy loop.
Single-intent routing (route()) keeps all terms unchanged.

**Why it could work:** The top false positives (cancel_order 42x, remove_item 25x,
product_availability 19x) are all driven by low-IDF terms: "order", "item",
"product", "account", "charge". Removing these from multi-intent scoring
eliminates the noise accumulation at its source.

**Implementation:**
- Compute IDF for each query term: `1 + 0.5 * ln(N / df)`
- With N=36 intents: IDF ranges from 1.0 (df=36) to 2.79 (df=1)
- Test multiple cutoffs: median IDF, 25th percentile, fixed values (1.3, 1.5, 1.8)
- In multi::route_multi, filter positioned terms to only those above cutoff
- Measure: pass rate at each cutoff level

**Risk:** Medium. Too aggressive a cutoff removes legitimate signal terms.
Need to find the sweet spot empirically.

**Expected impact:** HIGH. Directly addresses root cause. Could move pass rate
from 10% to 30-50%.


### Test C: Per-Intent Confidence Calibration

**Idea:** Track mean and standard deviation of scores per intent when confirmed
correct (from learn/correct calls). At routing time, filter intents whose scores
fall below `mean - 1*stddev` for that intent.

**Why it could work:** cancel_order scores 5-8 when truly intended, 1-3 as noise.
Per-intent thresholds would catch this. Different intents have different
"normal" score ranges.

**Implementation:**
- During setup, run all seed phrases through route() per intent
- Record score distribution (mean, stddev) per intent from these self-test results
- At routing time, reject intents scoring below `mean - 1.5 * stddev`
- Measure: pass rate with calibrated thresholds

**Risk:** Medium-high. Score distributions shift as vocabulary grows through
learning. Needs enough data per intent (seed phrases provide a starting point).
May not generalize well from seed-to-seed self-testing to real queries.

**Expected impact:** Moderate. Helps with clear outliers but may not distinguish
borderline cases where correct and noise scores overlap.


### Test D: Coverage Ratio as Escalation Signal

**Idea:** For each query, compute: what fraction of content terms are known to
the inverted index? High coverage (>60%) = trust ASV. Low coverage (<40%) =
high uncertainty, flag for LLM escalation.

**Why it could work:** A query full of unknown words means ASV is guessing.
A query where most terms hit the index means ASV has real signal.

**Implementation:**
- After tokenization, count: known_terms (in index) vs total_terms
- coverage_ratio = known_terms / total_terms
- On the 30 scenarios, compute coverage per turn
- Correlate with pass/fail: is coverage a reliable predictor of accuracy?
- Set escalation threshold empirically

**Risk:** Low. This is a diagnostic signal, not a fix. It tells you when to
trust ASV vs when to escalate.

**Expected impact:** Low for accuracy directly. High for production reliability —
prevents confidently wrong routing on out-of-vocabulary queries.


### Test E: Anti-Co-occurrence Filter

**Idea:** Build a learned matrix of intent pairs that should NOT appear together.
When multi-intent returns N intents, check every pair against this matrix.
Suppress the weaker intent in historically invalid pairs.

**Why it could work:** From the 30 scenarios, we can identify all false positive
pairs: billing_issue + apply_coupon, track_order + cancel_order in non-cancel
contexts, etc. These patterns are stable across queries.

**Two data structures:**
- `valid_pairs: HashMap<(String, String), u32>` — confirmed co-detections
- `invalid_pairs: HashMap<(String, String), u32>` — confirmed false pairings

**Implementation:**
- From scenario_turns_detail.json, extract all (ground_truth, extra) pairs
- Build invalid_pairs from extras, valid_pairs from ground truth combos
- Post-filter: for each detected pair, if invalid count >> valid count, suppress weaker
- Measure: pass rate with filtering vs without

**Risk:** Medium. Bootstrapping from 30 scenarios is limited data. Could
over-fit to test set. Needs production traffic to truly converge.

**Expected impact:** Moderate-high for known intent combinations. Zero for
unseen combinations. Gets better over time — this is the learning flywheel.


### Test F: Full Pipeline Combined

**Idea:** Run SymSpell -> noise-gated multi-intent -> confidence calibration ->
anti-co-occurrence filter -> coverage-based escalation as a full pipeline.

**Implementation:** Combine winners from Tests A-E. Measure aggregate pass rate.
Identify which components are additive vs redundant.

**Expected impact:** Sum of individual improvements minus overlap.


### Test G: Intent-Anchored Scoring (Novel Algorithm)

**Idea:** Fundamentally change multi-intent detection from "score everything,
filter later" to "anchor on discriminative terms, score in local windows."

**Algorithm:**
1. Precompute "anchor terms" per intent: top 2-3 terms with highest
   discrimination score (lowest df). For cancel_order: "cancel", "cancellation".
   For track_order: "track", "tracking", "package".
2. Scan query for anchor terms. Only intents with an anchor present become
   candidates. No anchor = no detection, regardless of generic term overlap.
3. For each anchored intent, compute ASV score using only terms within a
   window (e.g., 5-7 words) around the anchor position.
4. This naturally segments the query around intent-defining words without
   needing punctuation or grammar.

**Why it could work:** "I was charged twice for the same order and I still havent
received my package" — "charged" anchors billing_issue (window: "was charged
twice for the same"), "received"/"package" anchors track_order (window: "havent
received my package"). "Order" alone CANNOT anchor cancel_order or apply_coupon
because it's not a discriminative term for those intents.

**Implementation:**
- For each intent, compute anchor terms: terms where df <= (N/15).max(3) and
  weight >= 0.5 (high-weight + low-df = defining term)
- At query time, scan positioned terms for anchor matches
- For each anchor match, extract window of positioned terms around it
- Score the intent using only window terms
- Merge across anchors: keep best score per intent

**Risk:** Medium. Some intents may not have good anchor terms (too generic).
Edge case: what if the user expresses an intent without any anchor term?
("I need my money back" for refund — does "money" qualify as an anchor?)

**Expected impact:** HIGH. This is the cleanest architectural fix. Eliminates
noise at detection time rather than filtering it after. Could be the novel
contribution for the paper.


### Test H: Session-Based Bayesian Prior

**Idea:** In multi-turn conversations, the previous turn's detected intent(s)
inform what's likely next. Boost intents that commonly follow the previous
intent, suppress those that never follow.

**Implementation:**
- Build transition matrix from co-occurrence and scenario data
- Before scoring, apply prior: multiply candidate scores by transition probability
- Measure: per-turn accuracy within multi-turn scenarios

**Risk:** Medium. First turn has no prior. Wrong prior propagates errors.
Only works in conversational (multi-turn) contexts.

**Expected impact:** Moderate for multi-turn scenarios. Zero for single queries.

---

## Execution Order (by expected impact)

| Priority | Test | Expected Impact | Complexity | Risk |
|----------|------|----------------|------------|------|
| 1 | B: Noise Gate | HIGH | Low | Medium |
| 2 | G: Anchor-Based | HIGH | Medium | Medium |
| 3 | E: Anti-Co-occurrence | Medium-High | Low | Medium |
| 4 | A: SymSpell | Medium | Low | Low |
| 5 | C: Confidence Cal | Moderate | Medium | Medium-High |
| 6 | H: Session Prior | Moderate | Medium | Medium |
| 7 | D: Coverage Ratio | Low (diagnostic) | Low | Low |
| 8 | F: Full Pipeline | Sum of above | High | Low |

**Recommended sequence:**
1. Run Test B first (noise gate) — likely single biggest win, simplest to implement
2. Run Test G (anchor-based) — the novel algorithm, most research value
3. If B or G alone achieves >50% pass, combine them for Test F
4. Add SymSpell (A) and anti-co-occurrence (E) as production polish
5. Session prior (H) and confidence calibration (C) are optimizations for mature deployments

---

## Success Criteria

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Pass rate (exact match) | 10.1% | 40% | 60%+ |
| Pass + Partial rate | 84.1% | 85%+ | 90%+ |
| Fail rate (missed GT) | 15.9% | <10% | <5% |
| False positives per turn | ~4.2 avg | <1.5 | <0.5 |
| Latency impact | 0 (baseline) | <5ms added | <1ms added |

---

## Data Files

- `tests/scenarios/scenarios.json` — 30 scenarios, 138 turns, 6 categories
- `scenario_baseline_report.txt` — Full baseline report with threshold sweep
- `scenario_turns_detail.json` — Per-turn JSON: message, ground truth, detected,
  matched, missed, extra, status, word count, scores

---

## Key Insight for Paper

ASV's own data structures — inverted index, IDF scores, discrimination scores,
co-occurrence matrix, learned layer — contain all the information needed to solve
the multi-intent noise problem. No external models, no GPU, no embeddings.
The solution is self-referential: use the router's knowledge to improve the
router's accuracy. This is the "ASV-distilled discrimination" thesis.
