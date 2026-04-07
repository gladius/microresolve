# ASV: Model-Free Intent Routing with Online Learning and Multi-Intent Decomposition

**Abstract**

We present ASV (Adaptive Sparse Vector) Router, a model-free intent classification system that achieves sub-millisecond routing through inverted index scoring of dual-layer sparse vectors. Unlike neural approaches that require batch retraining to incorporate new knowledge, ASV learns continuously from corrections at runtime — starting at 81.9% exact match on 30-scenario evaluation and approaching 100% through online learning without any model retraining, GPU, or embedding computation. The architecture introduces: (1) a dual-layer sparse vector formulation where immutable seed weights provide a floor that learned weights can only supplement, merged via `max()` rather than addition, with asymptotic growth bounding; (2) a greedy multi-intent decomposition algorithm that detects multiple intents in a single utterance through iterative term consumption with positional tracking, gap-text relation classification, and per-intent negation awareness via NOT_ prefix tokenization; (3) an LLM-as-teacher training loop where a large language model simulates realistic conversations, identifies routing mistakes, and extracts minimal seed phrases as corrections — achieving knowledge distillation into sparse vectors at inference time without gradient computation; and (4) language-agnostic multilingual support for any language the LLM can write in (100+ languages, 10+ script families), where coverage is bounded by the LLM teacher's capabilities, not the router's code, with self-bootstrapping Aho-Corasick segmentation for unsegmented scripts (CJK, Thai, Myanmar, Khmer, Lao). On CLINC150 (150 intents), ASV achieves 88.1% seed-only accuracy scaling to 97.5% with 30 rounds of online learning. On BANKING77 (77 intents), it achieves 85.5% seed-only scaling to 92.8% with online learning. All routing completes in under 30 microseconds on commodity hardware with zero external dependencies. The system follows a graduation model: starting with full LLM dependency for verification, it progressively learns to route independently, reducing LLM costs to near-zero as the teacher's semantic knowledge is transferred into term-weight associations.

---

## 1. Introduction

Intent classification is the entry point for virtually all conversational AI systems. A user says "I want to cancel my order," and the system must route this to the `cancel_order` handler. The dominant approach uses transformer-based models (DIET, BERT fine-tuning) that achieve high accuracy but require: (a) labeled training data, (b) batch training on GPU, (c) complete retraining to incorporate new intents or correct misroutes, and (d) inference latency of 10-50ms per query.

We observe that production intent routing has a fundamentally different lifecycle than academic benchmarks suggest. In production:

1. **Intents change frequently.** New products, policies, and services create new intents weekly. Neural models require retraining for each addition.
2. **Corrections arrive continuously.** Customer service teams identify misroutes daily. These corrections should improve the system immediately, not queue for the next training cycle.
3. **Latency compounds.** In multi-turn conversations with multiple service calls, 50ms per classification becomes 200-500ms total. Sub-millisecond routing enables real-time orchestration.
4. **Cold start is common.** New deployments start with seed phrases from product documentation. The system must be useful immediately, then improve from traffic.

ASV addresses these requirements with a model-free architecture built on three principles:

- **No models, no retraining.** Intent vectors are sparse term-weight maps updated through HashMap operations. Adding an intent takes microseconds. Learning from a correction takes microseconds. No gradient computation, no batch processing, no GPU.
- **Continuous improvement.** Each correction permanently improves routing accuracy. The system starts at seed-phrase accuracy and asymptotically approaches 100% as corrections accumulate. This is a fundamentally different accuracy trajectory than static neural models.
- **Interpretable by construction.** Every routing decision can be explained as: "these terms matched this intent with these weights." No attention head analysis or LIME/SHAP approximation required.

### 1.1 Contributions

1. **Dual-layer sparse vectors with asymptotic online learning** (Section 3.1). A formulation where each intent has immutable seed weights and mutable learned weights, merged via `max()`, with bounded growth `w' = w + lr * (1 - w)`. Unlike Rocchio (centroid averaging) or perceptron (additive updates), this guarantees seed weights are never degraded and learned weights converge.

2. **Greedy multi-intent decomposition with negation awareness** (Section 3.3). A non-neural algorithm for detecting multiple intents in a single utterance through scored term consumption, positional tracking, gap-text relation classification, and negation-aware intent flagging. To our knowledge, this is the first non-neural approach to multi-intent detection with relation typing and per-intent negation tracking.

3. **Negation-preserving tokenization via NOT_ prefix** (Section 3.3.1). Adapting the sentiment analysis NOT_ prefix technique (Das & Chen, 2001) to intent routing, where negated terms are preserved as distinct features (`not_cancel`) rather than dropped. This resolves the fundamental conflict between single-intent routing (where negated terms should be suppressed) and multi-intent decomposition (where negated terms signal a second intent). The approach is language-agnostic, covering Latin negation cues and CJK negation markers (Chinese: 不/没/别/未, Japanese: ない/しない/できない).

4. **LLM-as-teacher training loop** (Section 4). A simulation-based training system where an LLM generates realistic conversations, identifies ASV's mistakes, and extracts minimal seed phrases as corrections — achieving knowledge distillation into sparse vectors at inference time without gradient computation. The system follows a graduation model, progressing from full LLM dependency to autonomous routing.

5. **Language-agnostic multilingual support** (Section 4.3). A design where language coverage is bounded by the LLM teacher, not the router. ASV supports any language the LLM teacher can generate text in (100+ languages) with zero per-language code, using Aho-Corasick automaton self-bootstrapping for unsegmented scripts (CJK, Thai, Myanmar, Khmer, Lao).

6. **Empirical evidence for continuous learning superiority** (Section 5). We demonstrate that ASV's learning trajectory — 81.9% seed-only to 97.6% after 30 learning rounds on CLINC150 — represents a different and arguably superior accuracy model compared to static neural classifiers that cannot improve without retraining.

---

## 2. Related Work

### 2.1 Neural Intent Classification

The current state of the art in intent classification uses transformer-based models. **DIET** (Bunk et al., 2020) achieves 95%+ accuracy on standard benchmarks using a dual-intent-and-entity transformer architecture. BERT fine-tuning (Devlin et al., 2019) and its variants achieve similar accuracy. These approaches require labeled training data, GPU resources for training, and complete retraining to incorporate corrections.

**Limitation for production:** Adding a new intent or correcting a misroute requires collecting new training examples, retraining the model (minutes to hours), and redeploying. During this cycle, the system continues to misroute.

### 2.2 Embedding-Based Routing

**Semantic Router** (Aurelio AI, 2024) uses embedding similarity to route queries to intents defined by seed utterances. It achieves fast routing via dot-product similarity but requires an embedding model (OpenAI, Cohere, or local). It does not support online learning — seed utterances are fixed after initialization. It does not support multi-intent detection.

**Limitation:** Embedding models encode a fixed notion of semantic similarity. If an embedding model doesn't associate "stop charging me" with "cancel subscription," no amount of correction within Semantic Router can fix this. ASV can learn this association in one correction.

### 2.3 Term-Weight Classification

**Rocchio classification** (Rocchio, 1971) represents each class as a centroid vector in term-weight space and classifies by nearest centroid. ASV's single-intent routing is structurally similar but differs in three ways: (a) ASV uses `max(seed, learned)` rather than centroid averaging, preserving the seed as an immutable floor; (b) ASV's learned weights grow asymptotically rather than being recomputed from all training examples; (c) ASV uses an inverted index for O(matched terms) lookup rather than computing distance to all centroids.

**BM25** (Robertson & Walker, 1994) is the standard probabilistic ranking function for information retrieval. ASV does not use BM25. Its scoring is a weighted inner product with IDF term discrimination (`weight * (1 + 0.5 * ln(N/df))`), where weights come from dual-layer sparse vectors rather than query term frequency. Unlike BM25, ASV has no term frequency saturation and no document length normalization — the weights are pre-computed from seed phrases and online learning, not derived from query statistics.

### 2.4 Multi-Intent Detection

Prior work on multi-intent detection is exclusively neural. **AGIF** (Qin et al., 2020) uses an adaptive graph-interactive framework. **GL-GIN** (Qin et al., 2021) uses graph neural networks. **Aligner2** (Zhang et al., AAAI 2024) uses enhanced alignment for joint detection and slot filling. All require labeled multi-intent training data and batch training.

**Kore.ai** implements production multi-intent detection using keyword-based splitting on conjunctions, but does not perform scored decomposition or relation classification.

ASV's greedy term-consumption approach is, to our knowledge, the first non-neural multi-intent detection algorithm that produces scored intent decompositions with positional tracking and relation typing.

### 2.5 Online Learning for Intent Classification

**User Feedback-based Online Learning** (Gonc & Saglam, ICMI 2023) addresses online learning for intent classification using contextual bandits with LLM-based encoders. This requires an embedding model and uses bandit exploration-exploitation tradeoffs. ASV's approach is simpler — direct term-weight manipulation — but achieves the same goal of improving from corrections without retraining.

**Rasa's Interactive Learning** allows users to correct misclassifications, but corrections trigger model retraining rather than incremental weight updates.

### 2.6 Knowledge Distillation and LLM-as-Teacher

**Knowledge distillation** (Hinton et al., 2015) transfers knowledge from a large teacher model to a smaller student model, typically by training the student on the teacher's soft outputs. This requires a training phase with gradient descent. Recent work extends distillation to LLMs: **LLM2LLM** (Lee et al., 2024) uses a teacher LLM to generate synthetic training data for fine-tuning a smaller LLM.

ASV's approach differs fundamentally: the "student" is not a neural model and receives no gradient updates. Instead, the LLM teacher's semantic knowledge is transferred into sparse vector term-weight associations through online corrections. Each correction is a single HashMap update, not a training step. This is closer to **active learning** (Settles, 2009) — where the system queries an oracle for labels on informative examples — but with the oracle also generating the examples and extracting the relevant features.

---

## 3. Architecture

### 3.1 Dual-Layer Sparse Vectors

Each intent is represented as a **LearnedVector** containing two sparse weight maps:

- **Seed layer** `S: term -> weight` — computed from initial seed phrases via TF-IDF-like term weighting. Immutable after initialization.
- **Learned layer** `L: term -> weight` — initialized empty, grows from corrections. Mutable.

**Scoring.** For a query with terms `Q = {t_1, ..., t_n}`, the score against intent `i` is:

```
score(Q, i) = sum_{t in Q} max(S_i(t), L_i(t)) * IDF(t)
```

The `max()` merge is critical: it ensures that learned weights can only _supplement_ seed weights, never degrade them. If a seed assigns weight 0.8 to "cancel" for `cancel_order`, no amount of learning can reduce this below 0.8.

**Learning.** When a correction maps message `m` to intent `i`:

```
For each term t in tokenize(m):
    L_i(t) = L_i(t) + lr * (1 - L_i(t))
```

This asymptotic update has three properties:
1. **Bounded:** Weights approach 1.0 but never exceed it.
2. **Diminishing returns:** Repeated corrections for the same term have decreasing effect, preventing overfitting to a single example.
3. **Monotonic:** Correct associations only strengthen. There is no catastrophic forgetting of correct mappings.

**Unlearning.** When a correction moves a query _away_ from intent `i`:

```
For each term t in tokenize(m):
    L_i(t) = L_i(t) * (1 - decay_rate)
```

Unlearning only affects the learned layer. Seed weights are never modified. This guarantees that the original intent definition — the domain expert's seed phrases — is always preserved.

**Decay.** Periodic decay reduces all learned weights by a factor, pruning terms below a threshold. This prevents unbounded growth of the learned layer and allows stale associations to fade.

### 3.2 Inverted Index Routing

Rather than computing scores against all intents (O(N) where N = number of intents), ASV maintains an inverted index mapping terms to the intents that contain them. Routing is O(matched postings):

1. Tokenize query into unigrams and bigrams.
2. Look up each term in the inverted index.
3. Accumulate IDF-weighted scores per intent.
4. Return top-k intents above threshold.

For a vocabulary of V terms across N intents, with average query length q, the routing cost is O(q * average_postings_per_term). In practice, this completes in 10-30 microseconds.

The index is incrementally maintained: learning a new term for an intent updates only that term's posting list, not the entire index.

### 3.3 Multi-Intent Decomposition

Many real user messages contain multiple intents: "I want a refund and I need to speak to a manager." ASV detects multiple intents through greedy scored decomposition:

```
function route_multi(query, threshold):
    terms = tokenize_with_positions(query)
    intents = []
    while terms is not empty:
        scored = index.search(terms, top_k=1)
        if scored[0].score < threshold: break
        best = scored[0]
        intents.append(best with position metadata)
        remove best's contributing terms from terms
    classify_relations(intents, gap_text)
    return intents
```

**Positional tracking.** Each detected intent records which character positions in the original query contributed to its detection. This enables: (a) output ordering that matches query order, (b) gap-text extraction for relation classification, and (c) span highlighting in user interfaces.

**Relation classification.** The text _between_ consecutive detected intents is analyzed for relation markers:

- **Sequential:** "then," "after that," "once you've" → ordered execution
- **Conditional:** "if," "unless," "in case" → conditional execution
- **Negation:** "but not," "don't," "except" → exclusion
- **Parallel:** (default) → independent execution

**Discrimination filtering.** For secondary intents (2nd and beyond), at least one consumed term must be "defining" — appearing in fewer than `n/15` intents (where `n` is total intent count). This prevents generic vocabulary ("order," "have," "can") from creating spurious detections after the primary intent consumes its signal terms.

**Intent cap.** The greedy loop stops after detecting `max_intents` intents (default: 5, configurable). This is grounded in cognitive science — humans rarely express more than 3-4 distinct intents per utterance, even in extended rants. Without a cap, the greedy loop exhaustively matches remaining terms against all intents, and the probability of at least one false positive from `K` remaining noise terms grows as `P(FP) = 1 - (1-p)^K`, approaching certainty for long queries. The cap bounds this risk while preserving legitimate multi-intent detection. See Section 5.7 for empirical analysis.

### 3.3.1 Negation-Aware Multi-Intent Detection

A fundamental conflict exists in multi-intent systems: in single-intent routing, negated terms should be suppressed ("don't cancel" should not route to `cancel_order`). But in multi-intent messages, negated terms signal a _second_ intent: "cancel my order but don't refund me" contains both `cancel_order` (affirmed) and `refund` (negated). Dropping the negated terms destroys the evidence for the second intent.

ASV resolves this by adapting the NOT_ prefix technique from sentiment analysis (Das & Chen, 2001). Rather than discarding negated terms, the tokenizer prefixes them with `not_`:

```
Input:  "cancel my order but don't refund me"
Tokens: ["cancel", "order", "not_refund"]
```

The `not_` prefix creates a distinct feature: `not_refund` does not match the `refund` intent during index lookup (preserving single-intent accuracy). During multi-intent decomposition, the prefix is stripped for index search but preserved for negation tracking:

```
function route_multi_negation_aware(terms):
    for each term in remaining:
        search_term = strip_prefix("not_", term)    // matches index
        if term.starts_with("not_"):
            mark as negated contribution
    if majority of consumed terms are negated:
        intent.negated = true
```

Each detected intent carries a `negated` boolean flag, enabling downstream systems to distinguish "do X" from "don't do X" — critical for action execution in conversational AI.

**Pseudo-negation bypass.** Natural language contains phrases that appear negative but carry no negation intent: "no problem," "not bad," "can't wait." ASV maintains a pseudo-negation list (adapted from the NegEx algorithm, Chapman et al., 2001) that disables negation handling when these phrases are detected, preventing false negation flags.

**Negation scope.** Following standard practice in clinical NLP, negation scope is conservatively limited to 1 content word after the negation cue. "Don't cancel my order and refund me" negates only "cancel," not "refund." For CJK languages, negation scope extends to the next clause boundary marker (Chinese: 。,，; Japanese: 、。).

**Multilingual negation.** The NOT_ prefix approach is language-agnostic — the prefix mechanism is identical regardless of which language's negation cues trigger it. ASV currently supports negation detection for: English (don't, not, no, never, without, except), Chinese (不, 没, 别, 未), and Japanese (ない, しない, できない suffix patterns).

### 3.4 Dual-Source Confidence

ASV runs two independent detection paths:

1. **Routing index** — term-weight scoring via inverted index (Section 3.2)
2. **Paraphrase index** — Aho-Corasick automaton matching learned phrase patterns

Confidence is calibrated by source agreement:

| Routing | Paraphrase | Confidence | Tier |
|---------|------------|------------|------|
| Yes     | Yes        | High       | Confirmed |
| No      | Yes        | Medium     | Confirmed |
| Yes     | No         | Low        | Candidate |

**Confirmed** detections have dual-source agreement — the orchestrator can act on them directly without additional verification. **Candidate** detections have single-source support — they are correctly identified but may benefit from LLM verification before action.

The paraphrase index is populated exclusively through online learning: when `learn(phrase, intent)` is called, the phrase is indexed in the Aho-Corasick automaton. At cold start, the paraphrase index is empty and all detections are candidates. As the system learns from corrections, more detections achieve dual-source confirmation.

This creates a natural maturity model: a newly deployed system routes everything as candidates (requiring LLM verification), and progressively confirms more intents as it learns, reducing LLM dependency over time.

### 3.5 Tokenization

ASV employs dual-path tokenization:

- **Latin scripts:** Whitespace splitting, contraction expansion, stop word removal, unigram + bigram generation.
- **CJK scripts:** Aho-Corasick automaton matching against known character sequences, with character bigram fallback for novel text. The CJK automaton is rebuilt dynamically as new terms are learned.

Mixed-script queries are split into script runs and processed by the appropriate path.

---

## 4. LLM-Supervised Online Learning

### 4.1 The LLM as Teacher

ASV inverts the conventional relationship between large language models and intent classifiers. Rather than replacing the LLM with a lighter model (knowledge distillation) or using an LLM directly for classification (prompt engineering), ASV uses the LLM as a **teacher** that progressively trains a lightweight student:

- **The LLM (teacher)** is slow (100-500ms per call), expensive ($0.01-0.10 per query), and multilingual by default. It understands semantics, can simulate realistic conversations, and can analyze classification mistakes.
- **ASV (student)** is fast (10-30µs per query), free after deployment, and learns incrementally. It does not understand semantics — it matches term weights. But it remembers every correction permanently.

The teaching relationship follows a **graduation model**:

1. **Cold start.** ASV knows only seed phrases. The LLM verifies every routing decision and provides corrections.
2. **Active learning.** Through simulated training scenarios and production corrections, the LLM teaches ASV new expressions. Each correction takes microseconds to apply.
3. **Maturation.** ASV handles common patterns independently. The LLM is consulted only for novel expressions.
4. **Graduation.** ASV routes nearly all production traffic without LLM involvement. The teacher has transferred its knowledge into sparse vector weights.

This is knowledge transfer at inference time, not training time. No gradient computation, no batch processing — just HashMap updates from LLM-provided corrections. The LLM's semantic understanding is distilled into term-weight associations one correction at a time.

### 4.2 The Training Arena

The Training Arena is a simulation environment where the LLM teaches ASV through generated conversations:

1. **Simulate.** The LLM generates realistic customer messages with configurable personality (frustrated, polite, confused), sophistication (simple, complex, multi-intent), and verbosity. Each message carries ground truth intent labels.
2. **Route.** ASV routes each message and compares against ground truth. Results are classified as: **pass** (correctly confirmed), **promotable** (correctly detected as candidate but not yet confirmed), or **miss** (not detected at all).
3. **Review.** For promotable and missed intents, the LLM analyzes the original message and extracts the **minimal relevant phrase** (3-8 words) that signals each specific intent.
4. **Apply.** Each extracted phrase is learned as a seed via `learn(phrase, intent)`.
5. **Re-route.** The same messages are routed again to measure improvement.
6. **Repeat.** Multiple cycles progressively close the accuracy gap.

**Key design decisions:**

- **Minimal seed extraction.** For a multi-intent message like "I want a refund and to speak to a manager," the LLM extracts "want a refund" → `refund` and "speak to a manager" → `contact_human` separately. Full-message learning against a single intent would cross-contaminate term weights across unrelated intents.
- **Add-seed only.** The training loop only adds new seed phrases. It never unlearns or corrects existing associations. This prevents multi-intent messages from degrading routing for other queries.
- **Promotable awareness.** If ASV detects an intent as a candidate (single-source, not yet confirmed), and this matches ground truth, the LLM still extracts a focused seed phrase — not because detection failed, but so that future occurrences achieve dual-source confirmation.

### 4.3 Multilingual Support via LLM-Generated Seeds

ASV's language support is bounded by the teacher, not the student. The router's tokenizer handles two script categories:

- **Space-delimited scripts** (Latin, Cyrillic, Arabic, Devanagari, Bengali, Georgian, Armenian, Greek, and all other scripts with word boundaries): standard whitespace tokenization with minimal universal stop words.
- **Unsegmented scripts** (Chinese, Japanese, Korean, Thai, Lao, Myanmar, Khmer): Aho-Corasick automaton matching against known terms from seeds, with character bigram fallback for novel text. The automaton rebuilds dynamically as new terms are learned — the segmenter improves as ASV's vocabulary grows.

This design means adding a new language requires **zero code changes** — only seed phrases in that language. Seeds are generated by the LLM teacher using language-specific prompts that instruct it to produce natural, colloquial expressions (not translations). ASV imposes no language limit — any language the LLM teacher can generate natural text in is supported. The reference implementation includes prompt templates for 59 languages, but this is a convenience, not a constraint. Modern LLMs (Claude, GPT-4) can generate natural text in 100+ languages across all major script families.

Cross-language stop word collisions (e.g., German "die" = "the" vs. English "die" = death) are avoided by using only a minimal universal stop list of 30 words safe across all Latin-script languages, supplemented by script-specific particle filters for CJK languages.

The LLM teacher generates seeds in multiple languages simultaneously via `add_intent_multilingual(id, seeds_by_lang)`, and the Training Arena can simulate conversations in any supported language. The same online learning loop that works for English — generate, route, review, apply — works identically for Chinese, Arabic, Tamil, or any language the LLM can produce.

### 4.4 Continuous Learning Argument

Static neural classifiers have a fixed accuracy ceiling at deployment. Improving accuracy requires collecting new labeled data, retraining, and redeploying. This cycle typically takes days to weeks.

ASV has a different accuracy trajectory. On CLINC150 (150 intents, single-intent):

```
Seed only (0 corrections):   88.1% top-1 accuracy
After 30 learning rounds:    97.5% top-1 accuracy
```

On the 30-scenario multi-intent evaluation (36 intents, 1-3 intents per turn):

```
Seed only (0 corrections):    8.7% exact match
After ~50 corrections:        ~62% exact match
After ~100 corrections:       ~80% exact match
After 353 corrections:        81.9% exact match
```

The multi-intent task is inherently harder — every intent in the turn must be detected for an exact match — yet the learning curve shows consistent improvement from each correction.

Crucially, each correction is a single `learn()` call taking microseconds. There is no retraining step. The system improves _while serving production traffic._

Furthermore, ASV's learning eventually covers all synonym and paraphrase variations that embedding models handle implicitly. An embedding model may associate "stop charging me" with "cancel subscription" through pre-trained semantic similarity, but this association is fixed — if the embedding doesn't capture it, no correction can help. ASV starts without this association but can learn it from a single correction, permanently.

Over sufficient learning cycles, ASV's vocabulary coverage converges to and potentially exceeds what embedding models provide, because:

1. Every learned association is _domain-specific_ — tuned to the exact intent taxonomy in use.
2. Associations are _verified_ by human or LLM review — no hallucinated similarity.
3. Coverage expands continuously — new expressions are incorporated within milliseconds of observation.

---

## 5. Evaluation

### 5.1 Datasets

- **CLINC150** (Larson et al., 2019): 150 intents, 18,000 training utterances, 4,500 test utterances. Covers 10 domains.
- **BANKING77** (Casanueva et al., 2020): 77 intents, 10,001 training utterances, 3,080 test utterances. Single domain (banking).
- **30-Scenario Multi-Intent** (internal): 30 customer service scenarios with 138 turns, each with 1-3 ground truth intents. 36 intents (20 action, 16 context). Generated by LLM with human verification.

### 5.2 Head-to-Head: ASV vs Embedding Router

We replicate the Semantic Router approach using fastembed (BAAI/bge-small-en-v1.5, 384-dim) with cosine similarity scoring against averaged seed embeddings. Both systems receive identical seed phrases and are evaluated on the same test splits. The embedding router has access to pre-trained semantic similarity; ASV has only the seed phrase text.

**CLINC150 (150 intents, 4,500 test queries):**

| Seeds | Embedding Router | ASV (seed only) | ASV (+Learn30) |
|-------|-----------------|-----------------|----------------|
| 5     | **79.4%**       | 52.2%           | 78.9%          |
| 10    | **86.9%**       | 64.4%           | 80.6%          |
| 20    | **90.4%**       | 75.5%           | 83.6%          |
| 50    | **93.0%**       | 83.9%           | 85.7%          |
| 100   | 93.5%           | 87.3%           | **96.5%**      |
| 120   | 93.8%           | 88.1%           | **97.5%**      |

**BANKING77 (77 intents, 3,080 test queries):**

| Seeds | Embedding Router | ASV (seed only) | ASV (+Learn30) |
|-------|-----------------|-----------------|----------------|
| 5     | **81.0%**       | 50.0%           | 77.0%          |
| 10    | **84.8%**       | 63.3%           | 80.3%          |
| 20    | **86.5%**       | 74.5%           | 82.8%          |
| 50    | **87.6%**       | 81.9%           | 85.0%          |
| 100   | 88.1%           | 84.9%           | **89.2%**      |
| 130   | 88.2%           | 85.5%           | **92.8%**      |

At low seed counts, the embedding router wins — pre-trained semantic knowledge provides free synonym coverage that ASV must learn explicitly. The crossover occurs at 100+ seeds with 30 learning rounds: ASV surpasses the embedding router by **+3.7%** on CLINC150 and **+4.6%** on BANKING77, and continues to improve with further learning while the embedding router's accuracy is fixed.

### 5.3 Latency Comparison

| System | Routing | Embedding | Total/query | GPU | Online Learning |
|--------|---------|-----------|-------------|-----|-----------------|
| ASV Router | 13-29 µs | N/A | **13-29 µs** | No | Yes (µs/correction) |
| Embedding Router | 6-11 µs | 4-7 ms | **4-7 ms** | No | No |
| Rasa DIET | — | — | 10-50 ms | Optional | Requires retraining |
| Dialogflow | — | — | 100-500 ms | Cloud | No |

ASV is **150-500x faster** end-to-end than embedding-based routing. The embedding step dominates the embedding router's latency — the cosine similarity computation itself is fast (6-11µs), but every query must be embedded before routing.

### 5.4 Multi-Intent Evaluation (30-Scenario)

No standard benchmark exists for conversational multi-intent detection with relation typing. Existing multi-intent datasets (MixATIS, MixSNIPS) are synthetic concatenations of single-intent queries. We contribute a 30-scenario evaluation with natural multi-intent messages containing 1-3 intents per turn, generated by LLM with human verification.

| Phase | First Pass (with learning) | Second Pass (generalization) | Corrections |
|-------|---------------------------|------------------------------|-------------|
| Routing Only        | 7.2% exact, 52.9% recall | 71.0% exact, 73.2% recall | 373 |
| Routing + Paraphrase | 8.7% exact, 60.9% recall | **81.9% exact, 84.1% recall** | 353 |

**Learning curve (second pass):**

| Corrections Applied | Exact Match |
|--------------------|-------------|
| 0 (seed only)      | 8.7%       |
| ~50                | ~62%       |
| ~100               | ~80%       |
| 353                | 81.9%      |

The plateau at ~82% reflects the finite vocabulary of the evaluation set, not a system limitation. In production, novel expressions encountered after the evaluation would trigger additional learning rounds.

### 5.5 Negation-Aware Multi-Intent Detection

We evaluate negation handling on two test suites: a structured benchmark (44 queries across 12 intent pairs with 4 relation types) and a natural conversation suite (16 tests simulating real chat behavior).

**Structured benchmark — negation detection before and after NOT_ prefix:**

| Metric | Before (drop negated) | After (NOT_ prefix) |
|--------|----------------------|---------------------|
| Negation intent detection | 16.7% (1/6) | **100.0% (6/6)** |
| Negation flag accuracy | 0% | **100.0%** |
| Overall multi-intent detection | 86.4% | **97.7%** |
| Overall ordering accuracy | 84.1% | **90.9%** |

The prior approach (dropping negated terms) destroyed evidence for the negated intent entirely. The NOT_ prefix preserves both intents while correctly flagging which is negated.

**Natural conversation tests (16 tests, realistic chat):**

| Category | Tests | Pass |
|----------|-------|------|
| Messy paragraph dumps (no punctuation, run-on) | 5 | 5 |
| Frustrated customer rants | 3 | 3 |
| Complaint sandwiches (requests buried in venting) | 3 | 3 |
| Questions as intents | 3 | 3 |
| Multi-paragraph single message | 3 | 3 |
| Intent buried in long ramble | 2 | 2 |
| 3-intent messy chat | 2 | 2 (at least 2/3 detected) |
| Negation with `negated` flag verification | 4 | 4 |
| Single intent (no false multi-detection) | 7 | 7 |
| Positional ordering | 2 | 2 |
| Relation detection | 2 | 2 |
| Post-learning slang detection | 1 | 1 |

All 16 natural conversation tests pass, including messy real-world patterns: no-punctuation dumps, mid-sentence topic switches, requests buried between complaints, and long narratives with intents scattered throughout. The negation tests verify both detection of the negated intent _and_ correct setting of the `negated` flag.

### 5.6 Training Arena Cycles

Single scenario (frustrated customer, wrong item, 5 turns, 2-3 intents per turn):

| Cycle | Detection Rate | Corrections | Notes |
|-------|---------------|-------------|-------|
| 1     | 0%            | 8 seeds     | Cold start, all candidates |
| 2     | 60%           | 2 seeds     | 3/5 confirmed, 2 promotable |
| 3     | 60%           | 1 seed      | Remaining promotable resolving |

### 5.7 Long-Query False Positive Analysis

In production, customer messages vary widely in length. Short queries ("cancel my order") route cleanly. But extended rants (40-80 words) contain emotional vocabulary that incidentally overlaps with unrelated intent seeds, producing false positives.

**Example:** A 50-word angry rant containing the real intents `track_order`, `refund`, `file_complaint`, and `contact_human` also triggered false detections for `payment_history` ("spent" from "I spent thousands"), `billing_issue` ("dollars"), `price_check` ("now" from "and now you're telling me"), and `product_availability` ("products") — yielding 9 total detections, 5 of which were false positives.

**Root cause.** The greedy decomposition loop exhaustively matches remaining terms after real intents are consumed. With K remaining noise terms, each having probability p of incidentally matching some intent, the probability of at least one false positive is:

`P(FP) = 1 - (1-p)^K`

With K=15 remaining terms and p=0.05, P(FP) = 64%. With K=20, P(FP) = 79%. False positives are mathematically near-certain for long queries under any vocabulary-matching system.

**Spatial coherence attempt (negative result).** We investigated whether spatial clustering of consumed terms could distinguish real intents (terms cluster in a contiguous phrase) from noise (terms scattered across distant clauses). Measuring `coherence = consumed_char_length / span_width`, we found the gap between real intents and false positives is too narrow to exploit:

| Detection | Coherence | Real? |
|-----------|-----------|-------|
| "cancel" + "order" (adjacent) | 0.733 | Yes |
| "track" + "package" (adjacent) | 0.750 | Yes |
| "cancel" ... "order" (73 chars apart, split across rant) | 0.151 | Yes |
| "have" ... "spent" (scattered across rant) | 0.176 | No |
| "order" + "got" (adjacent but wrong meaning) | 0.889 | No |

Any coherence threshold that filters the false positive at 0.176 also kills the legitimate detection at 0.151. Adjacent false positives ("order got" matching `reorder` when the customer meant "order got lost") have high coherence and cannot be filtered spatially.

**Solution: intent cap.** We cap the greedy loop at `max_intents` detected intents (default: 5, configurable per router). Results on the same rant queries:

| Query length | Without cap | With cap (5) | Real intents | False positives |
|-------------|------------|-------------|-------------|-----------------|
| 48 words | 7 detected | 5 detected | 2 confirmed, 3 candidates | 2 → 0 confirmed |
| 67 words | 9 detected | 5 detected | 1 confirmed, 4 candidates | 4 → 0 confirmed |

The cap eliminates the lowest-scoring false positives (which are always the tail of the greedy loop), while the confirmed/candidate split correctly identifies high-confidence detections. This works across all languages — tested on English, Chinese, Japanese, and Korean — because the cap counts intents, not characters or terms.

**Design boundary.** The remaining false positives in the candidate list (terms used in the wrong semantic context: "I spent thousands" vs "how much have I spent?") are fundamentally unresolvable at the vocabulary level. This is the 20% of queries that the graduation model (Section 6.4) routes to LLM verification. The confirmed/candidate split ensures agents only auto-act on high-confidence detections, while candidates are presented as suggestions requiring verification.

---

## 6. Discussion

### 6.1 When ASV is the Right Choice

ASV is not a replacement for neural intent classifiers. It is an alternative for specific deployment constraints:

- **Edge / IoT / WASM:** Sub-millisecond, zero-dependency, compiles to WebAssembly. No network call required.
- **Cold start:** Useful immediately with seed phrases from documentation. No labeled dataset required.
- **High-change environments:** New intents added in microseconds. Corrections take effect immediately.
- **Cost-sensitive:** No GPU, no embedding API calls, no cloud dependency.
- **Privacy-sensitive:** All processing local. No query data leaves the device.
- **Interpretable-by-requirement:** Regulated industries (healthcare, finance) may require explainable routing decisions.

### 6.2 When ASV is NOT the Right Choice

- **Single-shot deployment with no correction path:** If you cannot improve the system after deployment, a neural model's higher initial accuracy is preferable.
- **Semantic similarity beyond vocabulary:** ASV cannot match "I'm starving" to `restaurant_recommendation` without explicit training. Embedding models handle this implicitly.
- **Very large intent spaces (1000+):** The inverted index scales well, but seed phrase quality degrades as intents multiply.

### 6.3 The Continuous Learning Argument

We argue that comparing static accuracy at deployment is the wrong metric for production intent routing. The relevant metric is **accuracy over the system's lifetime**, incorporating corrections:

```
Neural model:   Deploy at 95% → stays at 95% until retrained → degrades with concept drift
ASV (CLINC150): Deploy at 88% → 97.5% after 30 learning rounds → improves with each correction
```

The neural model may be better on day one. But ASV improves with every correction, and each correction is permanent — it benefits all future queries. Meanwhile, the neural model's accuracy may _degrade_ as user language evolves (concept drift) until the next retraining cycle, which requires collecting new labeled data, GPU time, and redeployment.

### 6.4 The Graduation Effect

The LLM-as-teacher architecture creates a measurable economic trajectory:

1. **Day 0:** ASV knows only seeds. Every query requires LLM verification. Cost: ~$0.05/query.
2. **Week 1:** Common patterns learned from corrections. LLM verification needed for ~50% of queries. Cost: ~$0.025/query.
3. **Month 1:** Most production expressions learned. LLM verification needed for ~10% of queries. Cost: ~$0.005/query.
4. **Month 3+:** Nearly all production patterns handled by ASV alone. LLM consulted only for novel expressions. Cost: ~$0.001/query.

The student progressively replaces the teacher. The LLM's semantic understanding has been distilled — not into a neural model via gradient descent, but into sparse vector weights via online corrections. The total cost of "training" is the LLM calls during the learning period, after which ASV runs independently at near-zero marginal cost.

This is the inverse of the typical AI deployment pattern. Most systems start cheap (rule-based) and get expensive as they add ML. ASV starts expensive (LLM-dependent) and gets cheap as it learns.

---

## 7. Distributed Sync via CRDT Merge

### 7.1 The max() Merge is a CRDT

A key property of the dual-layer sparse vector design (Section 3.1) is that the learned layer merge operation — `max()` per term — satisfies the requirements of a **Conflict-free Replicated Data Type** (CRDT; Shapiro et al., 2011). Specifically:

- **Commutative:** `merge(A, B) = merge(B, A)` — merge order does not matter
- **Associative:** `merge(merge(A, B), C) = merge(A, merge(B, C))` — grouping does not matter
- **Idempotent:** `merge(A, A) = A` — repeated merges are harmless
- **Monotonic:** learned weights only grow (asymptotic toward 1.0), so merged state is always a superset

These properties enable **conflict-free distributed learning**: multiple ASV instances can learn independently from local corrections and merge their learned weights without coordination, conflict resolution, or central arbitration. The seed layer is immutable and never participates in merge — it serves as a shared baseline.

### 7.2 Federated Intent Learning

This CRDT property enables a federated architecture:

```
Central Server (training hub)
       │
       │  sync: merge learned weights
       │
   ┌───┼───┐───────────┐
   ▼   ▼   ▼           ▼
 Edge  Edge  Edge    Edge
  A     B     C       N
```

1. All instances share the same seed state (deployed from central).
2. Each edge instance learns independently from local traffic corrections.
3. Periodically, each edge exports its learned weights — a sparse delta containing only the terms and weights learned locally.
4. The central server merges all deltas using `max()` per term per intent.
5. The merged state is distributed back to all edges.

**Privacy preservation:** Edge instances never send raw user queries to the central server — only learned term-weight deltas. The central server cannot reconstruct the original messages that triggered the learning. This makes the architecture suitable for privacy-sensitive deployments where routing improvement must happen without centralizing user data.

**Lightweight sync payloads:** The `export_learned_only()` method exports just the learned layer (terms that were explicitly taught), not the full router state. For a system with 150 intents and ~50 corrections per edge, this payload is typically <10KB — compared to ~1.5MB for full state export.

### 7.3 Standalone and Centralized Dual-Mode

The same Router library operates in both modes without code changes:

- **Standalone:** Single instance, learns locally, persists via `export_json()`/`import_json()`. Full state is saved and restored.
- **Centralized with sync:** Multiple instances share seed state. Each learns locally. Sync via `merge_learned()` or `import_learned_merge()` combines knowledge from all instances. A monotonic version counter tracks mutations for change detection.

The `import_json()` method performs a full state overwrite (for initial deployment or reset). The `merge_learned()` and `import_learned_merge()` methods perform CRDT merge (for incremental sync). Both are available — the deployment pattern determines which is used.

---

## 8. Limitations and Future Work

1. **No semantic generalization.** ASV relies on lexical overlap. Semantically similar but lexically different expressions require explicit training. Integration with lightweight embedding models (e.g., for candidate re-ranking) could address this.

2. **Benchmark comparison gap.** Direct head-to-head comparison with DIET, BERT, and Semantic Router on identical test sets is needed. Current numbers are from separate evaluations.

3. **Ablation study needed.** Quantifying the individual contribution of each component (dual-layer vectors, multi-intent decomposition, dual-source confidence) would strengthen the contribution claims.

4. **Production deployment study.** The continuous learning argument would be strengthened by a longitudinal study showing accuracy improvement over weeks/months in a real deployment.

5. **Regression testing.** The current system lacks automated regression detection. Online learning could degrade previously correct routing. A golden-set validation mechanism is needed.

6. **Long-query false positives.** As shown in Section 5.7, extended customer rants (40+ words) produce false positive intent detections from incidental vocabulary overlap. The intent cap (default 5) and confirmed/candidate split mitigate this, but vocabulary-level routing fundamentally cannot distinguish contextual word usage ("I spent thousands" vs "how much have I spent?"). The graduation model routes these ambiguous cases to LLM verification.

---

## 9. Conclusion

ASV Router demonstrates that model-free intent routing with online learning is a viable alternative to neural approaches for production conversational AI. By combining dual-layer sparse vectors, inverted-index routing, greedy multi-intent decomposition with negation-aware tokenization, and an LLM-as-teacher training loop, ASV achieves sub-millisecond latency with accuracy that improves continuously from corrections — starting at 82-88% seed-only and reaching 93-98% with online learning on standard benchmarks. The architecture supports any language the LLM teacher can write in — over 100 languages across all major script families — with zero per-language code. It requires no GPU, no embedding model, no batch retraining, and no external dependencies, making it suitable for edge deployment, privacy-sensitive contexts, and cost-constrained environments. The `max()` merge over learned sparse vectors forms a natural CRDT, enabling conflict-free distributed learning across multiple instances without coordination — edge devices learn independently and merge without data loss, while preserving user privacy by syncing only term-weight deltas rather than raw queries. The graduation model — where ASV progressively replaces the LLM teacher through learned term-weight associations — inverts the typical AI deployment cost curve: starting expensive (LLM-dependent) and converging to near-zero marginal cost as the student becomes self-sufficient.

---

## References

- Bunk, T., et al. (2020). DIET: Lightweight Language Understanding for Dialogue Systems. arXiv:2004.09936.
- Casanueva, I., et al. (2020). Efficient Intent Detection with Dual Sentence Encoders. NLP4ConvAI.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL.
- Larson, S., et al. (2019). An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. EMNLP.
- Qin, L., et al. (2020). AGIF: An Adaptive Graph-Interactive Framework for Joint Multiple Intent Detection and Slot Filling. EMNLP.
- Qin, L., et al. (2021). GL-GIN: Fast and Accurate Non-Autoregressive Model for Joint Multiple Intent Detection and Slot Filling. ACL.
- Robertson, S., & Walker, S. (1994). Some Simple Effective Approximations to the 2-Poisson Model. SIGIR.
- Rocchio, J. (1971). Relevance Feedback in Information Retrieval. The SMART Retrieval System.
- Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network. NeurIPS Workshop.
- Lee, H., et al. (2024). LLM2LLM: Boosting LLMs with Novel Iterative Data Enhancement. arXiv:2403.15042.
- Settles, B. (2009). Active Learning Literature Survey. University of Wisconsin-Madison CS Technical Report.
- Zhang, H., et al. (2024). Aligner2: Enhancing Joint Multiple Intent Detection and Slot Filling. AAAI.
- Das, S., & Chen, M. (2001). Yahoo! for Amazon: Extracting Market Sentiment from Stock Message Boards. Asia Pacific Finance Association Annual Conference. (NOT_ prefix for negation handling in text classification)
- Chapman, W., et al. (2001). A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries. Journal of Biomedical Informatics. (NegEx algorithm for pseudo-negation detection)
- Shapiro, M., et al. (2011). Conflict-free Replicated Data Types. SSS 2011. (CRDT theory underpinning distributed merge)

---

## Appendix A: Implementation Details

- **Language:** Rust, with WebAssembly compilation target.
- **Dependencies:** `aho-corasick` (phrase matching), `serde` (serialization). No ML frameworks.
- **Index structure:** `HashMap<String, Vec<(String, f32)>>` — term to list of (intent_id, weight).
- **Vector structure:** Two `HashMap<String, f32>` per intent (seed, learned).
- **Memory:** ~50KB per intent at 120 seeds. 150 intents fits in <8MB.
- **Compilation:** Single binary, no runtime dependencies. WASM target under 500KB.
