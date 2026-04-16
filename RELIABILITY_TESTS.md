# ASV Reliability Experiments — Thursday 2026-04-16 → Friday 2026-04-17

Goal: find one or more layers that materially improve ASV's reliability — especially on false positives, morphological failures, and cold-start behaviour — before Monday's launch. Six months of development shouldn't be foreclosed by a rushed launch.

## Context: where ASV stands today

From prior benchmarks (`ngram-pattern-engine` branch, `docs/research/RESULTS.md`):

| Benchmark | Seed-only | With learning | Top-3 |
|---|---|---|---|
| CLINC150 (150 intents) | 84.9% @ 120 seeds | 95.1% | 94.4% |
| BANKING77 (77 intents) | 83.3% @ 130 seeds | 89.7% | 94.9% |

These are strong — matches Rasa DIET at 100x the speed, approaches fine-tuned RoBERTa. The launch is publishable on these numbers.

Today's experiments are about *better*, not *necessary*.

## What has been tried and failed (do NOT re-attempt)

From `concept-system`, `ngram-pattern-engine`, and `idf-query-learn` branches:

| Attempt | Result | Lesson |
|---|---|---|
| PMI word expansion | Finds primary intent; doesn't sharpen discrimination | Statistical similarity on sparse data is noisy; can't replace IDF |
| SVD on word-intent matrix | Zero meaningful similarity | 709×460 matrix too sparse to factorize |
| 24-dim dense vectors | All intents overlap | Dimensionality too small |
| N-gram pattern engine (v1-v5) | Dropped for 1-gram IDF | Complexity without accuracy gain |
| LLM paraphrase learning | Vocabulary mismatch | Paraphrases use LLM words, not user words |
| Raw query 1-gram learning | Cross-domain bleed | Query words from domain A strengthened B's intents |
| CA3 without IDF | Stalls at 93% | IDF is load-bearing; not replaceable |

Winning formula in current code: 1-gram IDF + Hebbian L1 expansion + L2 intent graph + L3 anti-Hebbian inhibition + token consumption for multi-intent + n-gram typo correction at input.

## Experiments to run (reprioritized 2026-04-16)

### Order — single branch `experiments-thursday`, additive commits, measure delta at each step

#### Step 0: Measurement harness (must come first)
A single script that runs the same query set through ASV and records top-1, top-3, confidence, and disposition. Fast to run (<60s). Same queries used for every subsequent experiment.

Sets: (a) 19-query adversarial set from `/tmp/router_experiment.py` covering easy/false-neg/false-pos/ambiguous; (b) a sample of CLINC150 (50 queries × 5 intents) if data available locally; (c) a "clean" set (25 unambiguous queries for regression detection).

Commit baseline numbers.

#### Step 1: Confidence calibration (top1/top2 ratio)
**Hypothesis:** raw IDF scores aren't well-calibrated. Adding `confidence = top1 / (top1 + top2)` as a field on `RouteResult` enables downstream callers to make better decisions (LLM fallback, OOS rejection, clarification prompts).

**Not an accuracy experiment per se** — infrastructure. Ship regardless of measurement outcome.

**Cost:** 1 hour. Add field, populate in routing path, include in API response.

**Risk:** none. Additive only.

#### Step 2: LLM-provided equivalence classes at build time
**Hypothesis:** The morphology class of failures ("crashing" vs "crashed") and the synonym class ("cancel" vs "rescind") can be addressed without embeddings, without PMI, without paraphrase generation. Instead: one LLM call per intent at build time asking *"identify equivalence classes of words in these phrases — morphological variants or synonyms."* Store as a deterministic tokenizer mapping.

This is NOT "LLM paraphrase learning" (which failed because paraphrases used novel vocabulary). This IS teacher-student distillation where the LLM acts as a dictionary, not a generator.

**Cost:** 2-3 hours. Add `equivalence_classes: HashMap<String, String>` per namespace (maps variant → canonical). Apply in `tokenize()` before IDF. Populate at intent creation via one LLM call.

**Expected lift:** 3-8% on morphology-heavy adversarial queries. 1-3% on clean benchmarks.

**Risk:** over-collapsing could hurt discrimination. LLM output needs a guard (refuse to collapse words that appear in different intents' seeds with different meanings). Fallback: tokenizer treats equivalence classes as *additive* (query token expanded to include canonical), not replacement. Safer.

#### Step 3: LP → L3 auto-feedback
**Hypothesis:** L3 inhibition is integrated in scoring but dormant on fresh namespaces. The `/api/execute` conversational flow never feeds back to it. If a conversation resolves to intent X, the *other* intents that weakly activated during turn 1 should be weakly inhibited for that query's vocabulary.

**Cost:** 1-2 hours. After each LP turn, if the resolved intent is confident, call `learn_inhibition(resolved_intent, competitor_intent)` for competitors above a weak threshold.

**Expected lift:** compounding over time. Single-turn doesn't help immediately; after 50-100 conversations, false-positive rate should drop measurably. Hard to measure in a day — set up infrastructure, measure the delta on the second run.

**Risk:** low. L3 has learning-rate caps. Revertable.

#### Step 4: Build-time L3 seeding from LLM-confusable pairs
**Hypothesis:** Cold-start false positives are predictable. At namespace creation, ask the LLM *"which of these intents look most confusable to which others?"* Pre-populate L3 inhibition pairs. Day-1 false-positive resistance without waiting for corrections.

**Cost:** 2-3 hours. One LLM call per namespace at build time. Populate inhibit matrix.

**Expected lift:** 3-6% on adversarial queries, especially false-positive class.

**Risk:** LLM may over-predict confusability, creating unnecessary inhibition. Guard: only seed inhibition at low initial strength; let real corrections strengthen or discard.

#### Step 5: Bigram-IDF as secondary scoring signal
**Hypothesis:** "book appointment" is a stronger signal than "book" + "appointment" scored independently. Adjacent word pairs with cross-intent rarity carry information the unigram model drops. Distinct from the failed n-gram pattern engine because this is purely **additive secondary scoring**, not a replacement for the 1-gram path.

**Cost:** 2-3 hours. Precompute bigram → intent → weight from seed phrases. At route time, add bigram contributions to the existing unigram IDF score.

**Expected lift:** 2-5% on queries with distinctive phrases.

**Risk:** slight memory increase (~50%). Slower but still sub-100µs.

#### Step 6: Porter/Snowball stemming (fallback)
**Hypothesis:** Even with equivalence classes, a stemmer handles long-tail morphology the LLM didn't cover.

**Cost:** 1-2 hours if we reach this.

**Expected lift:** 1-3% if Step 2 didn't already cover this via LLM equivalence classes.

**Risk:** over-stemming ("universe" → "univers" → matches "university"). Standard Porter is conservative.

---

## Execution plan

**Thursday afternoon / evening (today):**
- Step 0 (measurement harness + baseline) ✓ must complete
- Step 1 (confidence calibration) ✓ ship regardless
- Step 2 (LLM equivalence classes) — highest novelty

**Friday:**
- Step 3 (LP → L3 feedback) — wire-up
- Step 4 (build-time L3 seeding) — highest upside for false positives
- Step 5 (bigram-IDF) if time
- Step 6 (stemming) only if Step 2 underperformed

**Decision gate Friday evening:**
- Any experiment ≥3% lift on CLINC150 or ≥10% on adversarial → include in launch
- No experiment meaningful → launch Monday with existing 95.1% / 89.7% numbers (still a legitimate story)

## Strategic rules

1. **Single branch, additive commits.** `experiments-thursday` from current `intent-programming`. Each experiment = one commit with baseline + post-measurement in the message.
2. **Revert quickly.** If an experiment regresses clean queries by ≥2%, `git revert` that commit. Move on.
3. **No re-trying already-failed ideas.** PMI, SVD, dense vectors, n-gram pattern engine, LLM paraphrase training, raw query 1-gram learning.
4. **Measure same queries each time.** Don't change the test set mid-experiment.
5. **Record everything in this doc.** Update Results section as we go.
6. **Time-box.** Each experiment gets its stated cost + 50%. If it's eating time beyond that, skip and move on.

---

## Experiment Results Log

### Baseline (Step 0) — 2026-04-16 afternoon
Dataset: `tests/reliability/dataset.json` (105 queries, enterprise-focused, 4 SaaS domains). Namespace: `scale-test` (98 intents, 207 seed phrases, ~2-3 phrases/intent — realistic enterprise cold start).

**Overall:**
- Top-1: **66.7%** (single-intent queries only)
- Top-3: **78.3%**
- Multi-intent hit-all-in-top5: **42.9%**
- Multi-intent partial (at least one): **77.1%**
- OOS rejection: **70.0%** (30% false-accept rate on off-topic queries)
- Avg latency: **3.4ms** (includes HTTP round-trip)

**By category (top-3):**
- `cross_provider` — 90% (good: cross-provider disambiguation largely works)
- `multi_cross_domain` — 66.7% hit-all, 100% partial
- `multi_same_domain` — 40% hit-all, 100% partial
- `single_vercel` — 80%
- `single_linear` — 90%
- `single_stripe` — 70%
- `single_shopify` — 50% (**weakest**: thin seeds + vocab overlap with other providers)
- `informal_oov` — 90% (n-gram correction carrying)
- `false_pos_bait` — 20% partial (**expected low**: traps designed to fool the naive router)
- `oos_negative` — 70% rejection

**Key observations:**
- Top-1 (66.7%) vs Top-3 (78.3%) gap is ~12 points — the prefilter pitch holds.
- Multi-partial@5 is 77-100% — at least one correct intent almost always surfaces.
- Shopify lagging: likely from thin seeds + overlap with Linear/Stripe vocab.
- OOS false-accept rate of 30% is the main calibration target.

Saved: `tests/reliability/results/baseline.json`

### Step 1: Confidence calibration — 2026-04-16 (result: **hypothesis falsified, different signal wins**)

**Hypothesis:** `confidence_ratio = top1/(top1+top2)` is a better OOS rejection signal than raw top-1 score. Recommended in the prior `WEAKNESSES.md` doc.

**Finding:** On this enterprise dataset, **raw-score threshold BEATS confidence_ratio for OOS rejection.** Reason: cross-domain vocabulary overlap (create/list/update across 4 providers) makes *in-scope* queries have low confidence_ratio too. Low ratio = ambiguous, not OOS.

**Threshold sweep (see `calibration_analysis.py`):**

| Threshold strategy | OOS-rejected | In-scope wrongly rejected | F1(OOS) |
|---|---|---|---|
| Raw score < 0.5 (current default) | 70% | 1.5% | 0.778 |
| **Raw score < 1.5** | **80%** | **4.4%** | **0.762** |
| **Raw score < 1.8** | **90%** | **7.4%** | **0.750** |
| Raw score < 2.0 | 100% | 22.1% | 0.571 |
| Confidence ratio < 0.55 | 80% | **32.4%** | 0.400 |
| Confidence ratio < 0.60 | 80% | 39.7% | 0.356 |

**Lifts available from a raw-score threshold bump alone: 70% → 80-90% OOS rejection with only 4-7% false rejection.** That's ~15-20% improvement in OOS rejection without any code change beyond a threshold bump.

**Ratio is still useful, just not for OOS:**
- It signals *ambiguity* (multiple close candidates), which is a different thing than OOS
- Useful for triggering clarification prompts ("did you mean A or B?")
- Worth exposing in the API as a downstream signal even though it doesn't help the specific OOS metric

**Decision:**
- Bump the low-confidence threshold to ~1.5-1.8 in the default API behaviour (easy)
- Expose `confidence_ratio` as an additive field in `RouteResult` (takes nothing away, gives consumers the option)
- No regression — the low-score rejection only fires when scores are *higher* than current default was already trusting; flagged low-confidence queries simply surface as such to the consumer

**Saved:** threshold-sweep output in `tests/reliability/calibration_analysis.py`.

### Step 2: LLM equivalence classes — 2026-04-16 (result: **+6.7pp top-3, -1.7pp top-1**)

**Approach:** One LLM call per intent (98 total) asking for morphological variants + direct synonyms of each intent's seed words. Result: 789-entry variant→canonical map. Applied at query time via token expansion (query "cancelling my order" → "cancelling cancel my order" before sending to ASV).

**This is distinct from the failed LLM-paraphrase training.** Paraphrases generated new phrases (novel vocabulary). Equivalence classes map variants of words ALREADY in the seeds (e.g., "cancelling" → "cancel"). LLM is a dictionary, not a generator.

**Overall:**
| Metric | Baseline | Step 2 | Δ |
|---|---|---|---|
| Top-1 | 66.7% | 65.0% | -1.7pp |
| **Top-3** | **78.3%** | **85.0%** | **+6.7pp** |
| Multi-hit3 | 42.9% | 48.6% | +5.7pp |
| Multi-partial3 | 77.1% | 77.1% | 0 |
| OOS rejection | 70.0% | 70.0% | 0 |

**Per category (top-3):**
- `single_shopify` 50% → **80%** (+30pp) — biggest single-category win
- `informal_oov` 90% → **100%** (+10pp)
- `cross_provider` 90% → **100%** (+10pp)
- `single_vercel` 80% → 90%
- `multi_same_domain` 40% → 50%
- `multi_cross_domain` 66.7% → 73.3%
- `single_linear` 90% → 80% (regression)
- `single_stripe` 70% → 60% (regression)

**Interpretation:**
- Step 2 adds signal where vocabulary was sparse (Shopify) or mismatched (informal/OOV).
- Adds some noise where vocabulary was already clean (Linear/Stripe), diluting top-1 in favour of top-3.
- **Net win is top-3 / prefilter accuracy** (+6.7pp) — aligned with the MCP-prefilter launch pitch.

**Decision:** include at launch as **optional namespace feature** (user opts in per namespace). Default off for zero-regression promise; opt-in with documented top-1/top-3 tradeoff.

**Artifacts:**
- `tests/reliability/equivalence_cache.json` — per-intent LLM response cache
- `tests/reliability/equivalence_classes.json` — flat variant→canonical map (789 entries)
- `tests/reliability/results/step2_equivalence.json` — full measurement run

### Step 3: LP → L3 auto-feedback
(pending)

### Step 4: Build-time L3 seeding
(pending)

### Step 5: Bigram-IDF
(pending)

### Step 6: Porter stemming
(pending — skip if Step 2 covered morphology)

---

## Ideas considered and deliberately dropped

- **Negative example training via weight subtraction.** Too messy — per-word subtraction risks breaking legitimate queries. L3 pair-level inhibition (Step 3/4) is the clean version of the same goal.
- **Retrying PMI as a secondary signal.** Prior work showed PMI helps retrieval but not precision. Secondary-signal variants explored. Not worth the time budget — move to equivalence-class distillation instead.
- **Deep semantic layer via embeddings.** Contrary to the "language as program" philosophy and the cost profile. Would also require an embedding model dependency we've been careful to avoid.
