# Insertion-Time Discrimination Gating for Vocabulary-Based Intent Routers

## Abstract

Vocabulary-based intent routers classify user queries by matching terms against seed phrase vocabularies per intent. A persistent failure mode is **term pollution**: adding a seed phrase to one intent introduces terms that are primary vocabulary in competing intents, degrading overall system precision. Existing approaches (IDF, c-TF-IDF, BM25) mitigate this at search time but do not prevent it at insertion time. We propose **Seed Guard**, a real-time discrimination gate that evaluates seed phrases before they enter the index, using a discrimination ratio derived from the existing inverted index. This is, to our knowledge, the first system to provide insertion-time term collision detection for vocabulary-based intent routing without requiring neural embeddings or external corpora.

## 1. Introduction

### 1.1 The Problem

Intent routing systems map natural language queries to predefined intent categories. Vocabulary-based routers (as opposed to embedding-based or neural approaches) operate by maintaining per-intent term vocabularies, typically initialized from short seed phrases. When a query is received, it is tokenized and scored against these vocabularies using inverted index lookup with IDF weighting.

The core lifecycle involves iterative vocabulary expansion: seed phrases are added to intents to improve coverage of diverse customer language. This expansion creates a tension:

- **Coverage pressure**: each intent needs enough vocabulary to match the diverse ways customers express that intent
- **Discrimination pressure**: each intent's vocabulary must remain distinct enough to avoid false positives

When a seed phrase is added to intent A that contains terms already primary in intent B, three things happen:

1. The term's document frequency (df) increases, reducing its IDF weight
2. Intent B's precision for queries containing that term decreases
3. The system's overall ability to discriminate between A and B degrades

This is well-documented in text classification literature as "vocabulary overlap degradation" (NLU Platform Comparison, arxiv 2012.02640), but no existing system prevents it at the point of insertion.

### 1.2 Current Mitigation

**Search-time approaches:**
- **IDF weighting** (Sparck Jones, 1972): terms shared across many intents get lower weight. Effective but reactive — damage is already done when the term is indexed.
- **c-TF-IDF** (BERTopic, Grootendorst 2022): class-level IDF that measures distinctiveness per class. Operates on batch-computed topic models, not real-time intent routing.
- **BM25 saturation**: limits the contribution of high-frequency terms. Helps but does not address the discrimination problem.

**Batch feature selection:**
- **Chi-square test**: measures statistical dependence between term and class. Requires a document-term matrix from a labeled corpus.
- **Information Gain** (Forman, 2003): measures entropy reduction. Best for precision-oriented selection. Requires labeled corpus.
- **Bi-Normal Separation** (Forman, 2003): best overall performance across 229 classification tasks. Requires labeled corpus.
- **PMI**: measures co-occurrence above chance. Used in topic models and our own discovery module.

**Seed selection research:**
- **OptimSeed** (Jiang et al., NAACL 2021): automatic seed word selection via interim model training. Requires unlabeled corpus.
- **ConWea** (Mekala & Shang, ACL 2020): context-dependent seed word disambiguation. Requires BERT.
- **LOTClass** (Meng et al., EMNLP 2020): label-name-only expansion via masked language model. Requires BERT.

All existing approaches operate either at search time (reactive) or in batch mode (requires corpus/model). None provide real-time gating at the moment a seed phrase is added.

### 1.3 Our Contribution

We introduce **Seed Guard**, a zero-cost, real-time discrimination gate that:

1. Evaluates each seed phrase at insertion time against the current inverted index
2. Computes a per-term discrimination ratio using only existing index weights
3. Detects term collisions (terms already primary in competing intents)
4. Detects redundancy (all terms already covered by existing seeds)
5. Returns actionable warnings without blocking the insertion

Key properties:
- **Zero additional infrastructure**: uses only the existing inverted index
- **Zero model dependency**: no embeddings, no neural network, no external corpus
- **Constant-time per term**: O(1) lookup in the inverted index per term
- **Non-blocking**: warns but does not prevent insertion (user decides)

## 2. Method

### 2.1 Discrimination Ratio

Given a term `t` and a target intent `C`, the discrimination ratio is:

```
ρ(t, C) = w(t, C) / Σ_i w(t, i)    for all intents i
```

Where `w(t, i)` is the weight of term `t` in intent `i` from the inverted index.

Interpretation:
- ρ = 1.0: term exists only in intent C (fully discriminative)
- ρ ≈ 1/N (N = number of intents): term is uniformly distributed (non-discriminative)
- ρ = 0.0: term does not exist in intent C (collision — term belongs elsewhere)

For a NEW term (not in any intent), ρ is undefined — we treat this as safe, since it introduces fresh vocabulary with no existing associations.

### 2.2 Seed Evaluation

Given a seed phrase `s` to be added to intent `C`:

1. **Tokenize** `s` using the system tokenizer (stop-word removal, CJK handling)
2. **For each content term** `t` in the tokenized seed:
   a. If `t` is not in the index → classify as `new_term` (safe)
   b. If `t` is in the index with ρ(t, C) > 0 → classify as `existing_own` (already in C)
   c. If `t` is in the index with ρ(t, C) = 0 AND max weight in any other intent > threshold → classify as `collision`
3. **Aggregate**:
   - If ALL content terms are `existing_own` → seed is **redundant** (skip)
   - If any terms are `collision` → **warn** with affected intents
   - If zero content terms after tokenization → **skip** (stop-words only)
   - Otherwise → **clean addition**

### 2.3 Collision Severity

Not all collisions are equal. "order" appearing in 14 intents is different from "visa" appearing primarily in one.

We define collision severity based on the concentration of the term in the competing intent:

```
severity(t, competing_intent) = w(t, competing_intent) / Σ_i w(t, i)
```

- severity > 0.7: **high** — this term is a primary identifier for the competing intent
- severity 0.3-0.7: **medium** — shared but somewhat discriminative
- severity < 0.3: **low** — already distributed, minimal additional damage

Only high-severity collisions warrant user warning. Medium and low are informational.

### 2.4 Integration with Learning

The system supports three vocabulary expansion paths:

| Path | Collision Check | Behavior |
|------|----------------|----------|
| `add_seed(intent, phrase)` | Full check | Warn on collision, skip if redundant |
| `learn(query, intent)` | Check but don't block | User correction is ground truth |
| `correct(query, wrong, right)` | Check but don't block | User correction is ground truth |

The rationale: seed phrases are human-authored or LLM-generated patterns that may be imprecise. They should be gated. But `learn()` and `correct()` represent observed user behavior — if a real user's query about "refund to visa" was correctly classified as refund, the system should learn from that even if "visa" collides.

## 3. Relation to Existing Work

### 3.1 TF-IDF-rho (Zhang & Ge, 2019)

Our discrimination ratio is directly inspired by TF-IDF-rho's "class discriminative strength." TF-IDF-rho applies the discrimination factor as a weight multiplier at search time. We apply it as a quality gate at insertion time. The mathematical formulation is the same; the application point differs.

### 3.2 Complement Naive Bayes (Rennie et al., 2003)

CNB estimates P(term|NOT class) from all documents outside the class. This is conceptually similar to our collision detection: we check how much of a term's weight lives in OTHER intents. CNB applies this at classification time; we apply it at training data (seed) insertion time.

### 3.3 Rocchio Negative Centroids

Rocchio classification uses `centroid = β·positive - γ·negative`. Our approach could be extended similarly: when a colliding term is added, its weight in the target intent could be automatically reduced by a factor of the collision severity. We leave this as future work (Section 5).

### 3.4 c-TF-IDF (BERTopic)

c-TF-IDF replaces document-level IDF with class-level IDF, asking "how distinctive is this term for this class?" Our discrimination ratio answers the same question but in the context of a sparse seed-phrase vocabulary rather than a dense topic model.

## 4. System Architecture

### 4.1 Where Seed Guard Fits

```
User/LLM suggests seed phrase
        ↓
    Tokenizer (stop words, CJK, bigrams)
        ↓
    Seed Guard: check_seed()
        ↓
    ┌─ Redundant? → Skip, inform user
    ├─ Collision? → Add, warn user (show affected intents)  
    ├─ Empty? → Skip, inform user
    └─ Clean? → Add, report new terms
        ↓
    Inverted Index rebuilt
        ↓
    IDF recomputed (search-time mitigation still active)
```

### 4.2 Return Structure

```rust
pub struct SeedCheckResult {
    pub added: bool,
    pub new_terms: Vec<String>,
    pub conflicts: Vec<TermConflict>,
    pub redundant: bool,
    pub warning: Option<String>,
}

pub struct TermConflict {
    pub term: String,
    pub competing_intent: String,
    pub severity: f32,       // 0.0-1.0
    pub competing_weight: f32,
}
```

### 4.3 Complexity

- Per-term index lookup: O(1) (HashMap)
- Per-term weight summation: O(k) where k = number of intents containing the term
- Typical seed phrase: 3-8 content terms
- Total per seed: O(8k) ≈ O(1) for practical intent counts (< 100)

No iteration over the full index. No matrix computation. No model inference.

## 5. Future Work

### 5.1 Automatic Weight Adjustment
Instead of just warning, automatically adjust the weight of colliding terms:

```
adjusted_weight(t, C) = base_weight * (1.0 - collision_severity(t))
```

A term with severity 0.9 in a competing intent would get only 10% of its normal weight in the target intent. This implements Rocchio-style negative evidence at insertion time.

### 5.2 Cross-Intent Term Budget
Define a maximum "term spread" — the number of intents a term can belong to before it's considered non-discriminative and excluded from scoring entirely. This is a hard version of IDF that drops terms below a discrimination threshold rather than downweighting them.

### 5.3 Semantic Collision Detection
With an optional embedding model, detect semantic collisions even when terms are different. "reimburse" and "refund" don't share terms but are semantically close — adding "reimburse" to a non-refund intent could cause confusion via the similarity expansion layer. This requires the optional embedding fallback (not in current scope).

### 5.4 Collision-Aware Seed Generation
Feed the collision data back to the LLM seed generation prompt: "avoid these specific terms because they conflict with these intents." This creates a feedback loop where the generation quality improves from the guard's knowledge — but at the cost of larger prompts. Applicable only for initial intent setup, not per-fix review.

## 6. References

- Forman, G. (2003). An Extensive Empirical Study of Feature Selection Metrics for Text Classification. JMLR.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
- Jiang, Y. et al. (2021). OptimSeed: Automatic Seed Selection for Weakly-Supervised Text Classification. NAACL-SRW.
- Mekala, D. & Shang, J. (2020). Contextualized Weak Supervision for Text Classification. ACL.
- Meng, Y. et al. (2020). Text Classification Using Label Names Only. EMNLP.
- Rennie, J. et al. (2003). Tackling the Poor Assumptions of Naive Bayes Text Classifiers. ICML.
- Sparck Jones, K. (1972). A Statistical Interpretation of Term Specificity. Journal of Documentation.
- Zhang, W. & Ge, S. (2019). Class Specific TF-IDF Boosting for Short-text Classification. ACM.
- NLU Platform Comparison (2020). arXiv:2012.02640.
