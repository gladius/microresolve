# Semantic Layer Research — Findings & Architecture Notes

*Recorded after multi-turn LLM-driven continuous learning experiments.*  
*Date: 2026-04-12*

---

## What We Built

Three CPU-trainable, pure-Rust semantic encoders. No embedding API. All distillable from an LLM via vocabulary-gap pairs.

### MiniEncoder
Word unigrams + bigrams → FNV-1a hash → embedding table [30k × 64] → mean pool → W[64×64] → tanh → L2 normalize → cosine against intent centroids.

Bag-of-words MLP. Fast to train (< 1s for 7 intents × 200 epochs). No char n-grams (avoids contamination between similar words like "charge" / "chargeback"). Each word maps to exactly one hash bucket — clean pair training with no cross-bucket pollution.

### NanoEncoder
Same input → word embeddings E[n × dim] → Q=W_q·E, K=W_k·E, V=W_v·E → scaled dot-product attention → context-weighted pool → W_proj → tanh → L2.

True single-head self-attention. Same word gets different embedding depending on surrounding words. Theoretically superior for context-dependent routing.

### HierarchicalEncoder
L1 MiniEncoder: maps text → domain (stripe / support / deploy).  
L2 MiniEncoder per domain: maps text → intent within that domain.  
Inference: L1 score × L2 score. LLM pairs applied to both levels independently.

---

## Multi-Turn LLM Distillation Results

Dataset: 7 intents × 3 domains, 20 queries (IVQ + OOV + MIXED).

**Learning curve (Pass 0 → Pass 3):**

```
Model           P0    P1    P2    P3   Gain   Notes
MiniEncoder     15 →  17 →  18 →  17   +2    Converges in 1 pass, then plateaus
NanoEncoder     10 →   9 →   9 →  10   +0    Oscillates; attention unstable
HierarchicalEncoder 14 →  17 →  17 →  17   +3    Best gain; MIXED queries improve most
```

**Category breakdown (representative run, Pass 3):**

```
               IVQ      OOV      MIXED    Total
MiniEncoder    7/7      7/7      3/6      17/20
NanoEncoder    7/7      3/7      2/6      12/20
Hierarchical   7/7      7/7      4/6      18/20
```

**Key observation:** HierarchicalEncoder is the only model that improves on MIXED queries (cross-domain vocabulary) after pair refinement. It goes from 2/6 MIXED at P0 to 4/6 by P1. This is the designed use case: L1 domain routing acts as a soft Bayesian prior that reduces the confusion space before L2 makes the fine-grained call.

---

## Critical Bugs Found During Testing

### 1. Char N-gram Contamination (TinyEmbedder, fixed)
"chargeback" and "charge" share char n-grams `cha, har, arc, rge`. Pair training for ("chargeback" → dispute) corrupted the ("charge" → refund) path through shared hash buckets. Fixed by removing char n-grams entirely in MiniEncoder (word-only features).

### 2. NanoEncoder Attention Collapse Under Pair Refinement (fixed)
LLM pairs are 1-4 word snippets. The attention matrix for a 1-word input is trivially [[1.0]] — all attention on itself. Running 3000 gradient steps (60 pairs × 50 epochs) through degenerate 1×1 attention destroyed the attention patterns learned during triplet training. Symptoms: attention converges to stop words ("is", "the") getting 100% weight.

**Fix:** `pair_attn_lr_scale: 0.05` — W_q/W_k/W_v receive 5% of the learning rate during pair refinement. W_proj and word_emb receive full LR, since those don't depend on multi-word context to learn correctly.

### 3. Anisotropy in All-English Embeddings (fixed in TinyEmbedder)
English text shares common char n-grams. All embeddings cluster near the same direction. Fixed with mean-centering: subtract the mean centroid from both query and intent centroids before cosine similarity.

---

## Why NanoEncoder Underperforms at This Scale

The attention mechanism adds W_q, W_k, W_v — three extra 64×64 matrices. With only 6-7 seed phrases per intent, there is insufficient data to learn discriminative attention patterns. The extra parameters hurt: the model is overparameterized for this dataset size.

**The crossover threshold** (estimated): NanoEncoder starts outperforming MiniEncoder when each intent has ≥ 50 phrases. With 6-7 phrases, MiniEncoder's simpler inductive bias (uniform word weighting) is actually better.

With LLM pair distillation generating 60+ vocabulary-gap pairs per namespace, NanoEncoder's advantage should emerge — but only if the pairs are full phrases (not 1-4 word snippets), so the attention has real context to learn from.

---

## Emergence Observations

### Attention Peak Behavior
On mixed-domain queries, NanoEncoder peaks its attention on one word (1.000 weight):
- "I was double charged and want to dispute it" → peaks on "charged" → routes wrong (bug not dispute)
- "deploy failed and the app is crashing" → peaks on "deploy" or "failed"
- "rollback the release the system is broken" → peaks on "rollback" or "is"

The peaking on "is" (a stop word) after heavy pair refinement is the attention collapse signature. After the fix (scaled attn_lr), peaking on content words ("deploy", "failed", "rollback") is correct — but single-head still can't resolve competing signals.

### Hierarchical Implicit Domain Priors
Even without explicit domain information in test queries, L1 learns implicit priors. "cancel" has higher weight toward stripe-domain phrases than deploy-domain phrases, because "cancel subscription" appears in stripe training but not deploy training. This creates soft domain-routing even for ambiguous queries.

---

## What LLM Distillation Actually Teaches

Each pair `(term1, term2, similarity)` is a **knowledge transfer unit**:

- `("debited", "refund", 0.88)` → teaches the model that "debited" is in the semantic neighborhood of refund, without that word ever appearing in seed phrases
- `("chargeback", "dispute", 0.90)` + `("chargeback", "cancel", 0.08)` → draws a boundary between intents in embedding space
- `("revert", "rollback", 0.88)` + `("revert", "release", 0.08)` → teaches opposites at the deployment domain boundary

This is vocabulary-gap bridging: the LLM knows what "debited" means in context; we extract that knowledge as a geometric constraint in the embedding space. The model doesn't need to see "debited" in a training phrase — it only needs to know it should be near "refund".

---

## Phase 2 Improvements (2026-04-12)

### 1. Hard Negative Mining for MiniEncoder

**Problem:** Triplet training picks random negatives. Many are easy (already far from anchor),
so gradients are wasted on triplets the model already handles correctly.

**Fix:** `MiniEncoderConfig::hard_neg_start` (non-zero = epoch to switch). After that epoch,
every `hard_neg_freq` epochs: compute current centroid embeddings, find the closest wrong
centroid for each anchor, use that intent's phrases as negatives. Forces the model to sharpen
the boundaries it actually gets wrong.

**Result:** MIXED query accuracy 2/6 → 3/6 (+1). IVQ/OOV unaffected (already 7/7).
Gain of +1 on top of existing +1 from pair refinement.

### 2. NanoEncoder Transfer Initialization

**Problem:** NanoEncoder trained from scratch on 6-7 phrases/intent fails to learn useful
attention patterns. The model routes via word embeddings anyway, making attention irrelevant.

**Key insight:** MiniEncoder already learns intent-discriminative word embeddings via triplet
loss. Copying these into NanoEncoder's word embedding table (frozen), then training only
Q/K/V/W_proj, gives attention a semantic head start.

**Implementation:** `NanoEncoder::from_mini(mini, intent_phrases, cfg)`:
- Copies `mini.word_emb` → `nano.word_emb`
- Initializes Q/K/V with Kaiming-scale random (1/√dim)
- Trains with `word_emb_lr = 0.0` — only attention + W_proj update
- Requires matching `n_buckets` and `dim`; falls back to `NanoEncoder::train` otherwise

**Result (vs scratch NanoEncoder, same benchmark):**

```
                     P0      after pairs   gain
Scratch NanoEncoder  12/20   12/20          0
Transfer NanoEncoder 13/20   16/20         +3
```

Transfer advantage: **+4 after pair refinement**. IVQ: 5→7/7, OOV: 5→6/7, MIXED: 2→3/6.

**Why it works:** Attention starts with meaningful word geometries. Q/K projections can learn
to discriminate immediately rather than first needing to encode basic semantics from scratch.
The frozen word embeddings prevent pair refinement from corrupting the transfer knowledge.

### 3. Full Phase 2 Model Comparison

Final benchmark (20 queries: 7 IVQ + 7 OOV + 6 MIXED), after all 3 LLM pair passes:

```
Model                            P0    post   notes
MiniEncoder (baseline)           15    18     random negatives
MiniEncoder + hard neg           16    18     hard neg from epoch 100
NanoEncoder (transfer from mini) 15    18     frozen word_emb, attention only
HierarchicalEncoder + hard neg   12    17     L1+L2 both use hard neg
```

**Observations:**
- All three improved models converge to 18/20 — appears to be the ceiling for this 7-intent dataset
- Transfer-init NanoEncoder now matches MiniEncoder at 18/20 (was 12/20 from scratch)
- HierarchicalEncoder underperforms at P0 (12/20) but recovers to 17/20 with pairs
- Hard negative mining provides cleaner P0 start (16 vs 15) but same ceiling

**Known ceiling:** 2/6 MIXED queries resist all models. These involve cross-domain vocabulary
that requires both domain detection (L1) and intent disambiguation (L2) to be simultaneously
correct. The 6/20 irreducible error floor.

---

## MultiNanoEncoder (2026-04-12) — Experimental

2-layer 4-head self-attention encoder. Implements full transformer block: per-head Q/K/V,
scaled dot-product attention, W_o projection, residual connection between layers.

**Result:** Matches MiniEncoder exactly (17/20 P0 → 17/20 after pairs). No benefit vs
MiniEncoder at this dataset scale. Training 4-8x slower. Attention stays uniform — model
routes via word embeddings without needing attention diversity.

**Verdict:** Architecture is correct (tests pass, routing works), but requires ≥50
phrases/intent and full-sentence pairs to show benefit over MiniEncoder. Not recommended
for current deployment scale.

---

## Multi-Intent Detection (2026-04-12)

Single query → multiple intents, without retraining. Critical for enterprise routing where users
express compound needs: "cancel my account and roll back the last deployment."

### Approach: Per-Word NanoEncoder + Word-Level Centroids

**Problem with naive approach:** Running `nano_forward` on individual content words and scoring
against phrase-level centroids fails across domains. Root cause: phrase centroids are
mean-pooled multi-token embeddings; per-word scoring uses single-token embeddings. Different
distributions — cross-domain words like "rollback" score poorly against phrase centroids built
from full sentences.

**Fix: Word-Level Centroids**
- New field: `word_centroids: Option<HashMap<String, Vec<f32>>>` on `NanoEncoder`
- Built at training time: for each intent, run `nano_forward([single_hash])` on each content
  word in training phrases, then average. These live in the same space as per-word query embeddings.
- `score_query_multi` uses `word_centroids` (when available) instead of `centroids`.
- Stop words filtered consistently on both centroid-building and query-scoring sides.

**Gap detection:** after max-pooling per-word scores, return all intents within 0.15 of top score.
Genuine second intents score close to first; noise is typically 0.20+ lower.

### Results (transfer-init NanoEncoder, threshold=0.25)

```
Query type                                          Detected
─────────────────────────────────────────────────── ────────
cancel + refund (same domain)                       ✓ (both at 0.95+)
deploy release + rollback (same domain)             ✓ (0.91 + 0.90)
cancel account + rollback deployment (cross-domain) ✓ (0.96 + 0.85)
bug crashing + refund (cross-domain)                ✗ (refund dominates)
cancel subscription + deploy version (cross-domain) ✗ (cancel dominates)

Recall: 3/5   Precision: 2/3 (top intent always correct)
```

**What works:** Same-domain multi-intent and cross-domain pairs where both intents have
strong, distinct content words ("cancel" vs "rollback"). Word-level centroids give each word
a discriminative embedding that maps cleanly to the right intent.

**Failure mode:** When one intent's words ("cancel", "subscription") dominate the query
embedding space and the other intent's words ("deploy", "version") score lower because of
semantic ambiguity or shared vocabulary with the dominant intent.

**HierarchicalEncoder recall: 0/5** — L1 multi-domain softening (relative floor 30%)
sometimes rejects the secondary domain because L1 scores are too close (noise threshold issue),
or the secondary L2 model doesn't surface the right intent above threshold. Not fixed.

### Architecture Note
Word-level centroids are automatically built by `train()` and `from_mini()`. `rebuild_centroids()`
(called after pair refinement) also rebuilds them. `load()` sets them to `None` — save/load
only persists phrase centroids; word centroids are recomputed after loading if needed.

---

## Open Architecture Questions

Resolved or superseded:

1. ~~Multi-head attention~~ → MultiNanoEncoder implemented; no benefit <50 phrases/intent
2. ~~Stacked NanoEncoder layers~~ → 2-layer implemented in MultiNanoEncoder; same result
3. **Cross-lingual pair distillation** — still open; FNV-1a hashing is language-agnostic
4. **Score fusion with BM25 inverted index** — still open; most impactful at production scale
5. ~~Online centroid updates~~ → pair refinement + `rebuild_centroids` covers this
6. ~~Multi-intent detection~~ → 3/5 recall via word-level centroids + gap detection (2026-04-12)

Open:

7. **NanoEncoder transfer + hard neg** — combine both Phase 2 improvements in one model
8. **Pair generation with full sentences** (not 1-4 word snippets) — would unlock attention diversity
9. **HierarchicalEncoder multi-intent** — L1 relative-floor threshold tuning for cross-domain recall

---

---

## Session 2 — Concept System + Hebbian Graph (2026-04-12)

### Why the Semantic Encoders Failed (Root Cause)

All three encoders (Mini, Nano, Hierarchical) tried to **learn** semantic associations from training
phrases via gradient descent. The training corpus (6–20 phrases per intent) is three orders of
magnitude too small to learn meaningful embeddings. This is a data starvation problem, not an
architecture problem. The crossover point for NanoEncoder is ~50 phrases/intent with full-sentence
pairs — far beyond what a bootstrapped namespace has.

**Key realisation:** the LLM already knows the semantic associations. The task is to
**extract** that knowledge into a lightweight runtime structure, not re-learn it from scratch.

---

### Concept-Signal System (Experiment 1)

**Architecture:**
```
LLM (bootstrap, one-time)
  → defines 20-35 named concepts with signal word lists
  → assigns intent profiles: {concept → weight} per intent
  → defines intent_required: conjunction constraints

Runtime (no LLM):
  query → concept activation (signal string matching) → dot product → intent scores
```

**Results (20 intents, 4 domains, gap=2.5):**
- Clean queries: 13/13 = 100%
- OOV informal queries: 4/8 = 50%
- Multi-intent recall (gap=2.5): 78%  
- Multi-intent recall (gap=4.0, raw-score gap filter): **98%**

**Critical fix — gap filter used adjusted scores:** The conjunction multiplier (1.5× when all
required concepts fire) created artificial score gaps, dropping valid second intents. Fixed in
`score_query_multi` by computing gap on **raw scores** (pre-conjunction) and returning
**adjusted scores** (post-conjunction) for confidence tiering.

**Auto-learn for concept system:**
Turn 2 prompt shows concept registry + what fired → LLM identifies missing signal → `add_signal(concept, signal)`.
One signal addition benefits all intents sharing that concept. Tested: "ping" added to `action_send`
fixed "ping the team" routing. Auto-learn only triggers on flagged records (miss or low confidence).
Multi-intent partial misses (one intent routed, second missed) are NOT flagged — fundamental gap.

**Honest assessment of concept system:**
- NOT truly semantic — it is LLM-seeded keyword matching. Signal lists are string-matched at runtime.
  The LLM's intelligence is frozen at bootstrap; inference is deterministic lookup.
- Better than human keyword lists (wider vocabulary from LLM)
- Worse than BM25 for precision (fires on individual words globally, no span awareness)
- False positives: `create_repo` appearing in send_message queries because `action_create` fires
  on words like "project" or "launch" in the same query.

**Concept system vs term-index:**

| Property | Term-index (BM25) | Concept system |
|---|---|---|
| Cold start | Needs phrases | Zero-shot bootstrap |
| Precision | High (exact phrases) | Lower (global word matching) |
| OOV | Hard miss | Catches synonyms in signal list |
| Multi-intent | Span-aware | Gap-filter only, no spans |
| False positives | Rare | Common (word fires globally) |
| Learning unit | Phrase per intent | Signal per concept (shared) |

**Verdict:** Concept system is not a term-index replacement. It is a cold-start tool and a
semantic verification layer. Best role: **concept activations guide auto-learn** rather than
routing directly.

---

### The Right Architecture: Concept as Wrapper

```
Concept (semantic space definition)
  ├── pre:   query expansion / normalisation
  ├── inner: term-index (precision scoring, span tracking, learned weights)
  └── post:  concept profile re-ranking, disambiguation
```

Concept system is the semantic envelope; term-index is the precision engine inside.

**How auto-learn works in this architecture:**
1. Query arrives
2. Term-index routes (or misses)
3. Concept activation runs in parallel — identifies semantic space
4. If term-index is confident and concept agrees → high confidence, no flag
5. If term-index is uncertain and concept fires strongly → auto-learn: LLM sees both signals,
   adds phrase to term-index. Concept provides the oracle (which intent), term-index learns the
   exact phrase.
6. Next query: term-index hits directly. Concept system unchanged.

Single auto-learn pipeline. Concept is the semantic oracle, not the learner.

---

### Span Tracking — Is It Critical?

ASV's `multi.rs` tracks WHERE in the query each matched token cluster appears. Two clusters at
different positions → two intents. The question: is this critical for functional correctness or
just for visual display?

**Functional role of span tracking:**

1. **Multi-intent detection** — NOT strictly required. Gap-based filtering (return all intents
   within N of top score) achieves 98% recall without spans. Two genuine intents score close
   because both concept clusters fire independently.

2. **False positive suppression** — THIS is where spans matter. "I listed all the ways to cancel"
   — "listed" and "cancel" appear but belong to the same intent cluster. Span tracking sees they're
   distant with low token density per cluster → rejects multi-intent. Gap filter alone can't see this.

3. **Intent ordering** — "cancel then refund" vs "refund then cancel". Span gives ordering for
   sequential intent relations. Gap filter has no ordering information.

**Verdict:** Span tracking is NOT required for basic multi-intent detection. It IS useful for
suppressing false positive multi-intents (rare in practice) and for intent ordering (niche use
case). For the current routing use case, gap-based filtering is sufficient. Spans are a precision
improvement, not a prerequisite.

---

### Hebbian Association Graph (Experiment 2)

**Why Hebbian is the right structure for synonym expansion:**

The failed encoders tried to learn a dense weight matrix from training data. Not enough data.
The Hebbian graph instead: ask the LLM to generate association weights directly (LLM already knows
"terminate" and "cancel" are related). Weights are initialised from LLM knowledge, not learned from
scratch. True Hebbian update rule can reinforce edges from routing confirmations later.

**Structure:**

```rust
HebbianGraph {
  edges: HashMap<String, Vec<HebbianEdge>>,  // source → [(target, weight, kind)]
  synonym_threshold: f32,                    // default 0.80
}

EdgeKind { Morphological, Abbreviation, Synonym, Semantic }
```

**Weight tiers and query actions:**

| Weight   | Kind          | Action at query time                      |
|----------|---------------|-------------------------------------------|
| 0.97–1.0 | Morphological | Normalize — substitute in place           |
| 0.97–1.0 | Abbreviation  | Normalize — substitute in place           |
| 0.80–0.96| Synonym       | Expand — append canonical term            |
| 0.60–0.79| Semantic      | Signal only — no query modification       |

**Pipeline:**
```
raw query: "terminate my sub"
  ↓ normalize (morph + abbrev substitution)
"terminate my subscription"
  ↓ expand (synonym injection above threshold)
"terminate my subscription cancel"
  ↓ term-index BM25 on expanded query
cancel_subscription ✓
```

**Test results (21/21 unit tests passing, `src/hebbian.rs`):**
- Morphological: "canceling" → "cancel", "shipped" → "ship", "merged the pr" → "merge the pull request"
- Abbreviation: "cancel my sub" → "cancel my subscription"
- Synonym: "terminate" injects "cancel", "kill" injects "cancel", "ping" injects "send"
- Semantic (no expand): "stop" fires semantic signal to cancel but does NOT expand query
- Reinforcement: `reinforce("terminate", "cancel", 0.05)` strengthens existing edge, creates new edge for unknown words

**Key safety decision — ambiguous words excluded:**
"pull", "open", "get", "end" excluded from synonym edges. "pull" in "pull request" would
incorrectly expand to "list". "open" in "open a file" ≠ "create". Single-word synonyms
must be unambiguous in the domain. Multi-word synonyms ("spin up" → create) require
phrase matching, not yet implemented.

**LLM bootstrap prompt for Hebbian:**
Single call per namespace after intents are defined. LLM generates all four edge types
with weights and types. Stored as `_hebbian.json` per namespace. Never regenerated unless
namespace intents change significantly.

---

### Can Hebbian Replace Term-Index? (Analysis)

**Two-layer Hebbian as full intent detector:**

```
Layer 1 (word → word):    terminate → cancel (0.92, synonym)
Layer 2 (word → intent):  cancel → cancel_subscription (0.85)
                           subscription → cancel_subscription (0.90)

Spreading activation:
  "terminate my subscription"
  L1: terminate → cancel (0.92), subscription stays
  L2: cancel(0.92) × 0.85 + subscription(1.0) × 0.90 = 1.68
  → routes to cancel_subscription ✓
```

This IS routing via Hebbian spreading activation. No term-index. No BM25.

**Why it could beat BM25:**
- BM25 weights by document frequency (statistical). Hebbian weights by semantic discriminativeness (domain-distilled).
- BM25 has no synonym awareness. Hebbian does natively.
- Continuous Hebbian update from routing confirmations — every confirmed routing strengthens relevant edges.

**Current gaps vs term-index:**
1. **No span tracking** — global activation, no position information
2. **No conjunction natively** — pairwise Hebbian can't represent "cancel AND subscription must both fire". Needs AND-gate nodes or Layer 2 uses phrase-level nodes rather than word-level.
3. **Activation calibration** — spreading activation scores are unbounded; BM25 scores are well-studied.

**Verdict:** Full Hebbian intent detection is viable and theoretically cleaner. Not a quick swap — requires building Layer 2 (word-intent edges) and conjunction handling. Should be built alongside term-index and benchmarked.

---

### Hierarchical Multi-Layer Hebbian (Vision)

Each layer handles a distinct level of abstraction. All LLM-bootstrappable. All Hebbian-updatable.

```
Layer 0 — Sub-word (optional)
  Character n-grams, suffix rules
  Handles: typos, unknown morphological forms ("uncanceled" → suppress cancel)
  LLM generates: suffix → base rules

Layer 1 — Lexical (BUILT: src/hebbian.rs)
  Word → word associations
  Handles: morphology, abbreviations, synonyms, semantic neighbours
  LLM generates: edge list with weights and kinds

Layer 2 — Phrase / Pattern
  (word + word) → phrase node, phrase node → intent
  Handles: conjunction ("cancel" + "subscription" together → cancel_subscription node)
  Handles: negation ("not" + intent_word → suppression node)
  Handles: context markers ("I want", "please" → user_requesting boost)
  LLM generates: phrase patterns with conjunction and suppression rules

Layer 3 — Intent
  Phrase nodes → intent activation
  Direct equivalent of term-index word-intent associations
  LLM bootstraps from intent descriptions; auto-learn adds edges from routing confirmations

Layer 4 — Discourse / Session (future)
  Previous intent + current query → informed routing
  "I also want to..." → additive to last intent
  "Actually..." → correction signal
  Built from session logs, not LLM

Layer 5 — Confidence / Meta
  Aggregate activation across layers
  Multi-intent gap detection
  Escalation threshold (when to ask for clarification)
```

**Why this is the right long-term architecture:**

1. **Unified learning mechanism** — same Hebbian update rule at every layer. No seam between BM25 and semantic layers.
2. **LLM-bootstrappable at every layer** — no training data needed at cold start.
3. **Continuously refinable** — routing confirmations flow back as Hebbian updates at the relevant layer.
4. **Interpretable** — every routing decision traceable to specific edges. Explainability is built-in.
5. **Per-namespace** — each namespace gets its own graph, distilled from its own domain context.
6. **No embeddings required** — graph is sparse, structured, human-readable. Not a dense matrix.

**What nano/mini encoder was trying to be:** Layer 1 + Layer 3 collapsed into a single dense matrix, learned from scratch. The matrix couldn't be bootstrapped from LLM and needed data it didn't have.

**What Hebbian graph is:** Same knowledge, stored as a sparse edge list, bootstrapped from LLM. The Hebbian update rule applies when data is available. Starts useful from day one.

---

### Implementation Status

| Component | Status | Location |
|---|---|---|
| HebbianGraph struct + EdgeKind | ✅ Built, 21 tests passing | `src/hebbian.rs` |
| saas_test_graph (hand-crafted) | ✅ Built | `src/hebbian.rs` |
| Demo binary | ✅ Built | `src/bin/test_hebbian.rs` |
| LLM bootstrap endpoint | ⬜ Not started | `routes_concept.rs` or new file |
| Integration with routes_core.rs | ⬜ Not started | pre-processing before term-index |
| Layer 2 (phrase/conjunction nodes) | ⬜ Design only | — |
| Layer 3 (word-intent edges) | ⬜ Design only | — |
| Concept system as post-ranker | ⬜ Design only | `routes_core.rs` |

### Next Steps (Priority Order)

1. **Wire Layer 1 into routing** — call `hebbian.preprocess(query)` before term-index in `routes_core.rs`. Immediate gain for inflected queries and abbreviations.
2. **LLM bootstrap for Hebbian** — `POST /api/hebbian/bootstrap` generates the graph for a namespace.
3. **Debug endpoint** — `GET /api/hebbian/expand?query=...` shows normalization + expansion for a query.
4. **Layer 2 — phrase conjunction nodes** — "cancel" + "subscription" activating a shared node that strongly points to cancel_subscription. This replaces `intent_required` from concept system.
5. **Layer 3 — word-intent edges** — initialize from training phrases (term-index becomes a Hebbian subgraph). Benchmark spreading activation vs BM25.

## Files

- `src/semantic.rs` — all encoders + helpers + tests (139 tests, all passing)
- `src/bin/server/routes_semantic.rs` — HTTP API (build/score/compare/pairs endpoints)
- `src/bin/server/state.rs` — ServerState fields for all three model caches
- `src/hebbian.rs` — Hebbian graph (Layer 1), 21 tests passing
- `src/bin/test_hebbian.rs` — standalone demo binary

## Test Coverage (139 lib tests)

All passing as of 2026-04-12. Additions:
- `test_hard_neg_improvement` — random vs hard negative mining comparison
- `test_nano_transfer_init` — scratch vs transfer-init NanoEncoder
- `test_improved_model_comparison` — all 4 Phase 2 models side-by-side
- `test_multinano_basic/save_load/head_specialization/layer_depth/negation_awareness/multiturn` — MultiNanoEncoder suite
- `test_long_phrase_context_sensitivity` — longer full-sentence phrases, transfer NanoEncoder vs MiniEncoder
- `test_multi_intent_detection` — 5 multi-intent + 3 single-intent queries, recall/precision gate
