# Building MicroResolve: 14 Experiments, 12 Failures, 2 Wins

> A candid engineering log from building a sub-millisecond intent router.
> Published 2026-04-17.

## TL;DR

We built **MicroResolve**, a fast intent router in Rust — no embeddings, no GPU, no external models. Before getting to the final design, we tried **fourteen architectural experiments**. Twelve didn't work. Two did. The two that worked surprised us.

**What actually matters for intent routing accuracy:** not the scoring algorithm, not the clever post-processing layers, not the learned embeddings. **It's seed phrase density.** At 12 phrases per intent, simple 1-gram IDF with Hebbian reinforcement hits **91% top-1 / 97% top-3** on held-out enterprise queries in under a millisecond. At 2-3 phrases per intent, the same algorithm scores 40%. No cleverness bridges that gap.

If you're building a classifier and tempted to reach for embeddings, SVD, PMI, or 5-layer neural nets — read this first. We tried all of them.

---

## The setup

MicroResolve is a lexical intent router. Given a user query, it returns a ranked list of matching intents from a namespace. Use cases:

- Pre-filtering MCP tool routing (pick top-5 from 100+ tools before an LLM disambiguates)
- Command palette classifiers
- SaaS in-app help search
- Multi-agent orchestrator dispatch

Design constraints we picked early:
- **Sub-millisecond in-process** (no LLM on the routing hot path)
- **Zero infrastructure** (no GPU, no vector database, no fine-tuning pipeline)
- **Learns online** from corrections, no batch retraining
- **Multi-intent aware** — handle compound queries like *"refund the charge and cancel the order"*

With those constraints fixed, the question became: what's the scoring algorithm? The obvious candidates were TF-IDF, BM25, dense embeddings, or learned classifiers. We started with a TF-IDF-style inverted index with per-word IDF weighting — a simple baseline. Then we spent months trying to beat it.

---

## Part 1: the early experiments (things that failed in obvious ways)

### 1. PMI word expansion

**Hypothesis.** LLMs know which words are synonyms. Compute PMI (pointwise mutual information) from a corpus of LLM-generated paraphrases per intent. At query time, expand the query with high-PMI neighbors of each token so "reschedule" gets matched to "book_appointment" via its PMI neighbors.

**Implementation.** Per-namespace PMI matrix. At route time, expand each query word with its top-k highest-PMI neighbors, feed the expanded token set to the IDF scorer.

**Result.** PMI finds plausible related intents but doesn't sharpen top-1. On held-out validation, top-3 stayed roughly the same. Top-1 got slightly worse because expanded tokens added competing scores to already-similar intents.

**Why.** PMI on sparse data is noisy. With 12 phrases per intent, you have maybe 60-100 unique tokens per namespace — nowhere near enough for stable pairwise statistics. The expanded tokens add weight to competing intents more often than they sharpen the correct one. We tried applying PMI only when confidence was low; same outcome. PMI is a useful tool when you have millions of documents; not when you have hundreds of phrases.

### 2. SVD on the word-intent matrix

**Hypothesis.** Low-rank factorization of the word-intent co-occurrence matrix would capture latent intent structure. Project query words into the low-rank space, compare via cosine.

**Implementation.** 709 × 460 sparse matrix (words × intents). SVD at ranks 8, 16, 32, 64.

**Result.** Zero meaningful word-word similarity at any rank. At rank 8, every word mapped to the same corner of the space. At rank 64, pure noise.

**Why.** SVD needs a dense matrix. With ~4 non-zero cells per word on average, you're factorizing mostly zeros. The principal components that emerge are noise shapes. To make SVD work here you'd need hundreds of training examples per intent — which is the regime where a fine-tuned BERT would just be better anyway.

### 3. 24-dimensional dense vectors

**Hypothesis.** Train per-intent embeddings in a small dimension (24) and route via cosine similarity. Avoids the "SVD needs dense data" problem because you'd learn the embeddings, not factor an existing matrix.

**Implementation.** Per-intent 24-dim vectors. Train via gradient descent minimizing cosine loss between query bag-of-words and positive intent, maximizing distance to negatives.

**Result.** Every intent ended up at roughly cosine 0.5 from every query. The embedding space collapsed to something uniform.

**Why.** 24 dimensions is too small for ~100 intents sharing substantial vocabulary. The optimization has nowhere to put them that satisfies all the constraints simultaneously, so it finds a local minimum where everything is "medium-similar" to everything else. To separate 100 classes in an embedding space, you typically need 128-512 dimensions — which is approaching the regime of pre-trained sentence embeddings, which we had ruled out by design.

### 4. N-gram pattern engine (five iterations)

**Hypothesis.** Word bigrams and trigrams carry more signal than unigrams. *"book appointment"* is distinctive in a way that *"book"* and *"appointment"* separately are not.

**Implementation.** Five rounds of increasing complexity:
- v1: straight bigram index alongside unigrams
- v2: bigram IDF weighting
- v3: phrase position encoding
- v4: co-occurrence bonuses when multiple distinctive bigrams fire
- v5: dynamic pattern discovery from training queries

Each version is 400-600 lines of Rust. The pipeline had to match, score, and merge bigrams alongside unigrams at query time, which roughly doubled tokenization and index lookups.

**Result.** **1-2 percentage points of accuracy improvement** over the 1-gram baseline. Latency roughly 2x. Index size roughly 3x.

**Why.** At the scale we care about (dozens to low-hundreds of intents, 5-15 seed phrases each), most distinctive bigrams are already reconstructed from unigram matches — if a query has both "book" and "appointment" as tokens, the intent with both in its seeds already wins. The marginal cases where the bigram order matters are rare enough that the complexity isn't paid back. We killed this line after v5 and never looked back. **Complexity without proportional accuracy gain is just complexity.**

### 5. LLM paraphrase learning

**Hypothesis.** LLMs can generate 50-100 paraphrases per intent. Use those as additional seed phrases. Should give the router much richer vocabulary coverage.

**Implementation.** Batch call an LLM with each intent's description + existing seeds. Get back varied paraphrases. Add them to the training corpus.

**Result.** Validation accuracy **decreased**. Substantially in some domains.

**Why.** This was a sharp lesson. **LLMs generate paraphrases using LLM vocabulary.** LLM-generated user messages tend to be grammatical, complete, professional. Real user queries are terse, slangy, typo-prone, full of domain jargon. Adding LLM paraphrases to the training corpus **shifts the model toward the wrong vocabulary distribution**. Real users say "gimme the stripe customers" — and after LLM augmentation, that query matches less well because the router now expects longer, grammatical forms.

The fix turned out to be the opposite: learn directly from real user queries (when corrections come in), not from LLM-imagined queries.

### 6. Raw 1-gram learning from user queries

**Hypothesis.** After #5 taught us real queries are the right training signal, learn 1-grams directly from every correctly-routed query. Simple reinforcement.

**Implementation.** On successful routing, add each query word to the target intent's index with a small positive weight.

**Result.** Cross-domain vocabulary contamination. Common words like "help", "please", "with", "need" accumulated strong weights across unrelated intents. IDF weakened. Routing degraded.

**Why.** Without discriminative filtering, every word becomes a positive signal for every intent it appears in. If a user says *"please help me with my account"* and we route to `account_details`, then "please", "help", "with" all get learning signal toward `account_details` — even though none of those words discriminate `account_details` from any other intent.

The fix: only learn words that are **LLM-confirmed intent-bearing** (ask an LLM "which words in this query actually signal the intent"). That worked. Raw query 1-gram learning doesn't.

### 7. CA3-style scoring without IDF

**Hypothesis.** The brain-inspired literature suggests Hebbian learning alone (no explicit IDF) should be enough. Word-intent co-occurrence strengths accumulate through reinforcement; the strong associations dominate.

**Implementation.** Pure Hebbian: weight strengthening on each observed co-occurrence, asymptotic saturation. No IDF term at query time.

**Result.** Accuracy plateaus at **93%** on CLINC150. With IDF: **95.1%**.

**Why.** IDF is load-bearing. Rare words carry more information than common words. Hebbian learning alone treats all observed co-occurrences as equally informative — a word with thousands of positive reinforcements across many intents contributes as much as a word with 20 reinforcements in one intent.

The 2% gap is concentrated in the hardest queries: the ones where multiple intents share most vocabulary and only one distinctive word separates them. Without IDF weighting on that distinctive word, scoring goes sideways.

We shipped 1-gram IDF + Hebbian reinforcement. Hebbian gives you adaptability; IDF gives you discrimination. You need both.

---

## Part 2: this week's interventions (things that failed subtly)

The first wave of experiments was "try radically different architectures." The second wave, more recent, was the opposite: **try cheap additive layers on top of the 1-gram IDF baseline**. A confidence threshold, a rejection filter, a tiebreaker, equivalence classes, cross-provider inhibition, bigram re-ranking. Each one tested in isolation, each one looked like an easy 2-5pp win on development queries.

Then we started running held-out validation. **Everything broke.**

### 8. Confidence ratio for out-of-scope rejection

**Hypothesis.** `top1 / (top1 + top2)` is a better OOS rejection signal than raw top-1 score. When the top two candidates score close, the router is genuinely uncertain.

**Result.** On our enterprise dataset (98 intents across 4 providers), in-scope queries *also* have low confidence ratios because of cross-provider vocabulary overlap. In-scope "list stripe customers" has `stripe:list_customers` and `shopify:list_customers` both scoring high. Using ratio for OOS rejection wrongly rejected 32% of in-scope queries.

**Why.** Ratio measures ambiguity, not familiarity. A query can be ambiguous between two valid intents and still be in-scope. Raw score threshold (simple low-score cutoff) outperformed ratio-based rejection by 15-20 points on F1.

**Lesson.** We re-read our own `WEAKNESSES.md` where an earlier researcher had recommended "use confidence ratio for OOS." The document was wrong. Testing it on real data showed the opposite. **Prior documentation is not evidence.**

### 9. LLM equivalence classes via query-time expansion

**Hypothesis.** Ask an LLM once per intent: "what are morphological variants and synonyms of these seed words?" Get back maps like `{"crashing": "crash", "booking": "book"}`. At query time, expand query tokens with canonical forms.

**Result on dev set:** **+6.7pp top-3**. Specifically +30pp on the weakest domain (Shopify). Looked like a clear win.

**Result on held-out validation:** **0pp lift.** Zero. Same top-3 accuracy as baseline.

**Why.** The dev queries and the seed phrases had been authored by the same person (me). Equivalence classes from the same author's vocabulary help queries written in the same author's vocabulary. On held-out queries written deliberately in different phrasings, equivalence classes don't bridge the gap because the gap isn't morphology — it's vocabulary choice.

**Lesson.** **Dev-set overfitting is real and sneaky.** A +6.7pp number looks compelling until you test on queries outside the design distribution. Always run a held-out validation.

### 10. Seed-phrase augmentation (variants as phrases)

**Hypothesis.** Same equivalence classes, but add variants as extra phrases in the router's index instead of expanding at query time. Persistent rather than transient.

**Result.** Neutral to slightly negative on held-out. ~3300 variant phrases added to the namespace; accuracy unchanged.

**Why.** Same underlying problem as #9 — variants help for vocabulary the seeds already cover, don't help for vocabulary they don't. Adding them to the index also increases IDF denominator (more intents contain any given word), which dilutes existing scoring weights. Small net regression.

### 11. L3 cross-provider inhibition (hand-seeded pairs)

**Hypothesis.** Same-action-different-provider pairs are mutually confusable — `stripe:list_customers` vs `shopify:list_customers`. Pre-seed anti-Hebbian inhibition between them so provider context disambiguates.

**Result.** +3.3pp top-1 on dev, **-4.6pp top-3** on validation. Trades precision for recall, and worse on the metric that matters for prefilter use cases.

**Why.** Inhibition suppresses. When the query genuinely spans both providers ("list customers on both stripe and shopify"), inhibition eats the correct multi-intent answer. The pair-level heuristic is too coarse.

### 12. L3 via LLM-identified confusable pairs

**Hypothesis.** Instead of hand-picking confusable pairs, have an LLM identify them from intent descriptions.

**Result.** Neutral to negative on held-out. LLM-picked pairs weren't better than our heuristic pairs.

**Why.** The LLM picks pairs that *look* confusable — cancel vs refund, create vs update. But surface similarity ≠ actual routing confusion. Real routing confusion happens where vocabulary genuinely overlaps, which is already captured by MicroResolve's internal scoring. Adding LLM-picked inhibition on top adds noise, not signal.

### 13. Bigram-IDF re-ranking

**Hypothesis.** Re-rank MicroResolve's top-K results using a bigram overlap bonus. Doesn't replace scoring, just tweaks the order at the top.

**Result.** +3.3pp dev top-1, -2.3pp val top-1. Classic overfit.

**Why.** The bigrams that correlate with dev queries happen to over-fire on dev query patterns. On held-out phrasings, those same bigrams are distractors.

### 14. N-gram FP filter

**Hypothesis.** After MicroResolve returns top-1, check whether the query contains any **distinctive** bigram for that intent. If not, demote top-1. Keeps confident correct matches, rejects confident wrong matches.

**Result.** **-20.6pp top-1 on dense validation.** Catastrophic.

**Why.** "Distinctive bigram" is a high bar. Many perfectly correct queries use general vocabulary — "cancel it", "show me", "refund please". None of those has a distinctive bigram. The filter demoted them indiscriminately. The intent wasn't wrong; our filter's bar was wrong.

**Lesson.** Filters based on hard signals need extremely conservative thresholds. If the filter fires on 40% of queries, it's wrong for 40% of queries regardless of how precise the threshold seems in isolation.

### 15. Warm learning with 30 corrections

**Hypothesis.** Apply 30 corrections for previously-failing queries. Learning improves over time.

**Result.** **-2.3pp top-1, -6.8pp top-3** on held-out validation.

**Why.** Sparse corrections cause collateral damage. Each correction strengthens the target intent for specific words — but those same words appear in other intents too. The net effect of 30 corrections on 98 intents is that adjacent intents lose accuracy as fast as the corrected ones gain it.

**Lesson.** Learning scales with volume. 30 corrections is below the threshold where noise averages out. Either do zero corrections or do hundreds. 30 is the worst number.

---

## What actually worked

### A. Seed density

**The single biggest lever**: going from 2-3 phrases per intent (thin, realistic cold start) to 12 phrases per intent (rich, realistic production) moved held-out top-1 accuracy from **40% to 91%**. That's +50 percentage points from data alone, no algorithmic changes.

| Seeds per intent | Top-1 | Top-3 | OOS rejection |
|---|---|---|---|
| 2-3 | 40.9% | 59.1% | 20.0% |
| 12 | **91.2%** | **97.1%** | **66.7%** |
| 120 (CLINC150) | 95.1% | 94.4% | - |

The lesson: **if your team is asking "how do we improve accuracy," the answer is almost always "add more phrases to your intents."** Not a better algorithm. Not a cleverer layer. More data, where the data is high-signal seed phrases.

### B. Token consumption for multi-intent

MicroResolve's distinctive capability is decomposing compound queries. *"Refund the stripe charge and cancel the shopify order"* produces two top-ranked intents — one from each SaaS provider — because MicroResolve's scoring consumes tokens as it fires intents, so subsequent intents score on what's left.

On 125 intents × 5 domains × 58 compound queries: **94.8% of queries had all expected intents in top-5, 97.7% average recall@5**. At sub-millisecond latency.

Most classifiers can't do multi-intent at all. They pick one winning intent. Token-consumption decomposition is the algorithmic differentiator.

### C. Character n-gram Jaccard tiebreaker

The one new layer from this round of experiments that shipped. When MicroResolve's top-1 and top-2 score close (ratio < 0.65), compute character 4-gram Jaccard similarity between the query and each candidate's seed phrases. Re-rank using `original_score + 0.5 × jaccard`.

On thin-seed held-out: **+4.6-6.7pp top-1**. On rich-seed: dormant (rarely fires, zero regression). Derived from seed phrases at index time — no training needed.

The reason this one survived where #13 (bigram re-ranking) failed: char-ngram catches morphological similarity that word-level matching misses (typos, verb forms), and the tiebreaker only activates on ambiguous queries. It doesn't try to "improve" confident cases.

---

## Meta-lessons

### 1. Held-out validation kills dev-set optimism

We consistently saw 15-25 percentage point gaps between dev accuracy and held-out validation accuracy. If a team is optimizing against the same queries they wrote the seeds for, they're measuring authoring consistency, not routing capability.

Build two disjoint query sets. Measure on both. Only the held-out number is real.

### 2. Data density is the primary lever

Every architectural experiment we ran this week fought for 1-5 percentage points. Seed density alone gave us +50. The ratio is absurd. **If you're worried about accuracy, invest in data quality before algorithms.**

### 3. Complexity costs are underestimated

Our failed n-gram pattern engine was ~3,000 lines across 5 iterations. Our failed SVD and dense vector experiments were hundreds of lines each. When the marginal gain is 1-2 points and the maintenance cost is a module, delete the module.

### 4. Simple baselines are deceptively strong

1-gram IDF is a 1970s algorithm. Hebbian reinforcement is a 1940s algorithm. We spent months trying to beat them with PMI, SVD, embeddings, neural nets, and n-gram engines. The baselines won. They did not win because they were optimal — they won because the more complex alternatives were worse.

### 5. Prior documentation is not evidence

Halfway through this project we were implementing suggestions from our own earlier research notes. "Use confidence ratio for OOS rejection." "Add PMI expansion." When we tested these on held-out data, most were falsified. The documentation reflected intuitions from earlier iterations, not current measured behavior.

Re-test everything before trusting prior claims.

---

## The numbers (cold-start enterprise, held-out validation)

| Metric | Value |
|---|---|
| Top-1 accuracy (12 phrases/intent) | **91.2%** |
| Top-3 accuracy (12 phrases/intent) | **97.1%** |
| Multi-intent partial@5 (58 compound queries, 125 intents) | **84.2%** |
| Multi-intent all-expected@5 | **70.0%** |
| Avg routing latency (in-process Rust) | **300-900µs** |
| Avg latency over HTTP (local) | **2-4ms** |
| Namespace memory (1000 intents × 20 phrases) | **~10MB** |
| Throughput (single core) | **100K-150K queries/second** |

Public benchmarks (from prior work on the n-gram-engine branch):
- CLINC150 (150 intents): 84.9% seed-only → 95.1% with 30 corrections → 94.4% top-3
- BANKING77 (77 intents): 83.3% seed-only → 89.7% with learning → 94.9% top-3

Independent numbers (not author-measured). Reproducible benchmark harness coming with the v0.1 release.

---

## Takeaways, if you're building a router

1. **Start with 1-gram IDF**. It's the honest baseline. If you can't beat it reliably, don't replace it.
2. **Invest in seed quality before scoring cleverness.** 12+ phrases per intent, each distinct in vocabulary, written by a real user of the target domain if possible.
3. **Write held-out validation queries**. Disjoint from seeds. Phrased differently. Run every "improvement" against it.
4. **Token consumption for multi-intent is cheap and distinctive.** If your classifier only returns one intent per query, you're leaving a product capability on the table for compound queries.
5. **Character-level signals help thin-seed scenarios.** Char 4-gram Jaccard as a tiebreaker on close calls. Free morphology handling.
6. **Avoid embedding dependencies unless you already need them.** GPUs and vector databases are real infrastructure costs. Simple lexical + learning gets you 90%+ on most enterprise use cases at 0.1% of the serving cost.
7. **Publish your failures.** They'll save someone else six months.

---

*Code: MicroResolve is open source at [link-to-repo]. Benchmarks + test suite included. Contributions welcome — especially ones that break our numbers, honestly reported.*
