# ASV Router — Novelty Assessment & Publication Justification

> Compiled 2026-04-02 from exhaustive prior-art search (100+ queries across academic papers, GitHub, npm, PyPI, crates.io, commercial platforms).

---

## 1. Verdict

**Paper-worthy: YES.** No existing system — academic or commercial — combines all of ASV's differentiators. Individual techniques (inverted indexes, IDF, keyword matching) are well-known; the *architecture and combination* is novel.

**This is NOT another case of reimplementing an existing technique.** The n-gram speculative decoding situation was: exact same technique, already shipped as a default feature. ASV's situation is: known building blocks assembled into an architecture nobody has published or deployed.

---

## 2. What We Searched

### Academic Sources
- arXiv (2020–2026): "inverted index intent classification", "sparse vector intent routing", "online learning intent detection", "multi-intent decomposition without neural", "keyword-based intent classification with learning", "model-free NLU", "non-neural multi-intent detection"
- ACL Anthology: multi-intent SLU surveys, online learning for dialogue, keyword-based classification
- IEEE/Springer: lightweight intent routing, edge deployment NLU
- Google Scholar: "dual-layer sparse vector", "term-weight online learning intent", "inverted index routing"

### Code/Library Sources
- GitHub: keyword intent router, sparse vector intent, multi-intent keyword, lightweight NLU
- npm: intent-router, keyword-classifier, nlu-lite
- PyPI: intent-classifier, keyword-router, lightweight-nlu
- crates.io: intent, nlu, router, classifier

### Commercial Platforms
- Kore.ai, Dialogflow, Watson Assistant, LUIS, Rasa, Cognigy, Voiceflow, Wit.ai
- Semantic Router (Aurelio AI), NLP.js, Snips NLU, Mycroft Adapt, natural, brain.js, compromise, wink-nlp

---

## 3. Closest Competitors — Honest Overlap Assessment

### Semantic Router (Aurelio AI, 2024)
- **What it is:** Embedding similarity routing. Embed seed phrases → cosine similarity at query time.
- **GitHub:** ~2,000 stars, actively maintained
- **Overlap with ASV:** ~25% — both route queries to intents using seed phrases
- **Key differences:**
  - Requires embedding model (OpenAI, Cohere, or local fastembed)
  - No online learning — seeds are fixed after init
  - No multi-intent detection
  - No relation classification
  - Latency: 4-7ms (embedding dominates) vs ASV's 13-29µs
- **Why ASV is distinct:** Fundamentally different architecture (sparse vectors vs dense embeddings). ASV learns from corrections; Semantic Router cannot.

### Kore.ai
- **What it is:** Enterprise conversational AI platform with keyword-based multi-intent support
- **Overlap with ASV:** ~40% — keyword matching, multi-intent splitting on conjunctions
- **Key differences:**
  - Proprietary SaaS, no published architecture
  - No online learning (configuration-based, not learning-based)
  - Multi-intent splits on conjunctions only, no scored decomposition
  - No relation detection between intents
  - No paper published
- **Why ASV is distinct:** Open architecture with online learning + scored decomposition + relation typing. Kore.ai's approach is rule-based splitting, not scored term consumption.

### NLP.js (AXA Group, open source)
- **What it is:** Lightweight NLU library with perceptron-based intent classification
- **GitHub:** 6,561 stars, actively maintained
- **Overlap with ASV:** ~30% — lightweight, multi-language, no GPU required
- **Key differences:**
  - Perceptron classifier (neural, though simple)
  - Requires batch retraining to incorporate new data
  - No multi-intent detection
  - No online learning (must retrain)
  - Node.js only
- **Why ASV is distinct:** True online learning (no retrain), multi-intent decomposition, inverted index (not perceptron).

### Mycroft Adapt
- **What it is:** Trie-based keyword intent matching for voice assistants
- **GitHub:** ~500 stars, low maintenance
- **Overlap with ASV:** ~25% — keyword-based, lightweight
- **Key differences:**
  - Trie-based exact matching (no scoring, no ranking)
  - No learning of any kind
  - No multi-intent support
  - Python only, voice-assistant focused
- **Why ASV is distinct:** Scored routing, online learning, multi-intent, relation detection — essentially everything beyond basic keyword lookup.

### Snips NLU
- **What it is:** Lightweight NLU library with logistic regression + CRF
- **GitHub:** ~3,800 stars, dead since Apple acquisition (2020)
- **Overlap with ASV:** ~15% — lightweight intent classification
- **Key differences:**
  - Logistic regression (requires training data and batch training)
  - Dead project (no updates since 2020)
  - No online learning
  - No multi-intent
- **Why ASV is distinct:** Active, online learning, multi-intent, zero-dependency.

### Rocchio Classification (1971)
- **What it is:** Centroid-based text classification in term-weight space
- **Overlap with ASV:** ~35% — term-weight vectors, centroid similarity
- **Key differences:**
  - Centroid averaging (ASV uses `max(seed, learned)`)
  - No inverted index (computes distance to all centroids)
  - No online learning (recomputes centroids from all examples)
  - No multi-intent
  - Academic technique, not a deployed system
- **Why ASV is distinct:** Dual-layer vector design, incremental index updates, online learning with asymptotic growth, multi-intent decomposition.

### Classical BM25 / Information Retrieval
- **Overlap with ASV:** ~20% — inverted index, IDF weighting
- **Key difference:** BM25 ranks documents by relevance. ASV routes queries to intents. The data structures are similar but the application and learning mechanism are entirely different. ASV's weights come from seed phrases and online learning, not term frequency statistics.

### Gonc & Saglam (ICMI 2023) — "User Feedback-based Online Learning"
- **What it is:** Online learning for intent classification using contextual bandits with LLM encoders
- **Overlap with ASV:** Uses online learning for intent classification
- **Key differences:**
  - Requires embedding model (LLM-based encoder)
  - Uses bandit exploration-exploitation (not direct weight updates)
  - No multi-intent detection
  - No inverted index architecture
- **Why ASV is distinct:** Model-free (no embeddings), simpler mechanism (direct weight updates), multi-intent support.

---

## 4. Key Academic Validation

### Multi-Intent SLU Survey (December 2025, ACL)
The most recent survey on multi-intent spoken language understanding explicitly confirms:
- **All existing multi-intent approaches are neural** (AGIF, GL-GIN, Aligner2, etc.)
- **Non-neural multi-intent detection is absent from the literature**
- **Inter-intent relation detection is identified as an open research problem**

ASV directly addresses both gaps.

### Knowledge Distillation Literature
- Hinton et al. (2015) — distillation into neural students via gradient descent
- LLM2LLM (Lee et al., 2024) — LLM generates synthetic data for fine-tuning smaller LLMs
- **Neither distills into non-neural sparse vectors via online corrections.** ASV's approach is novel in the distillation literature.

---

## 5. What IS Well-Known (Don't Oversell)

These components are standard and the paper should NOT claim novelty for them individually:
- Inverted indexes (textbook IR, 1960s)
- IDF weighting (Sparck Jones, 1972)
- Keyword matching for intent detection (basic NLP technique)
- Stop word removal, tokenization (standard preprocessing)
- Aho-Corasick automata (1975 algorithm)
- Asymptotic growth functions (standard math)
- Knowledge distillation concept (Hinton 2015)

**The novelty is the combination and the architecture**, not any single component.

---

## 6. What Makes ASV Genuinely Novel (The Combination)

No existing system combines ALL five:

1. **Inverted-index architecture applied to intent routing** (not document retrieval) with dual-layer sparse vectors (`max(seed, learned)` merge)
2. **True online learning** — single-example weight updates without retraining, rebuilding, or batch processing
3. **Non-neural multi-intent decomposition** with positional tracking and inter-intent relation detection (sequential, conditional, negation, parallel)
4. **LLM-as-teacher knowledge distillation into sparse vectors** — semantic knowledge transferred via online corrections, not gradient descent
5. **Zero external dependencies** — no embeddings, no GPU, no model files, compiles to WASM

---

## 7. Risk Assessment for Publication

| Risk | Level | Mitigation |
|------|-------|------------|
| "Just keyword matching" dismissal | **Medium** | Ablation study showing learning + multi-intent contribute; head-to-head vs embedding router shows competitive accuracy |
| Semantic Router comparison | **Low** | Fundamentally different architecture (sparse vs dense); ASV surpasses at 100+ seeds with learning |
| Someone publishes similar first | **Low** | No indication anyone is working on this exact combination |
| Reviewers demand neural baseline | **Medium** | Already have embedding router comparison; could add BERT/DistilBERT baseline |
| "Not enough datasets" criticism | **Medium** | CLINC150 + BANKING77 are standard; adding ATIS or SNIPS would strengthen |
| Accuracy inferior to SOTA neural | **Expected** | Paper explicitly positions ASV for different use cases (edge, online learning, interpretability) — not competing on raw accuracy |

---

## 8. Recommended Target Venues

| Venue | Fit | Rationale |
|-------|-----|-----------|
| **arXiv + open-source** | Excellent | Immediate impact, no gatekeeping, establishes priority |
| **EMNLP Industry Track** | Strong | Practical system with real benchmarks, deployed architecture |
| **ACL System Demonstrations** | Strong | Working system with UI, multi-intent visualization |
| **NAACL Industry Track** | Strong | Same rationale as EMNLP Industry |
| **ACL/EMNLP Main** | Weak | Lacks the theoretical depth expected for main track |

**Recommendation:** Publish on arXiv immediately + open-source. Then submit to EMNLP Industry Track or ACL System Demonstrations.

---

## 9. Potential Paper Improvements (Not Yet Implemented)

These could strengthen the paper but are not blocking:

### Ablation Study
Quantify individual contribution of each component:
- Dual-layer vectors vs single-layer
- With/without online learning
- With/without multi-intent decomposition
- With/without dual-source confidence

### Additional Baselines
- BERT/DistilBERT fine-tuned on same seed data
- TF-IDF + logistic regression
- Rasa DIET with equivalent training data

### Proposed Novel Extensions
1. **Federated Intent Learning** — max-merge learned layers across instances preserves privacy while sharing knowledge
2. **Agent Tool Routing** — reframe ASV as universal microsecond routing layer for AI agent tool selection
3. **Intent Drift Detection** — monitor learned weight velocity to detect concept drift
4. **Explainable Routing** — every decision decomposable to "term X matched intent Y with weight Z" (already inherent, just needs formalization)

### Third Dataset
Add ATIS or SNIPS benchmark for broader evaluation coverage.

---

## 10. Comparable Published Work (For Related Work Section)

| Paper | Year | Venue | Relevance |
|-------|------|-------|-----------|
| DIET (Bunk et al.) | 2020 | arXiv | Neural intent baseline |
| AGIF (Qin et al.) | 2020 | EMNLP | Neural multi-intent |
| GL-GIN (Qin et al.) | 2021 | ACL | Neural multi-intent |
| Aligner2 (Zhang et al.) | 2024 | AAAI | SOTA neural multi-intent |
| Semantic Router (Aurelio) | 2024 | GitHub | Embedding-based routing |
| BM25 (Robertson & Walker) | 1994 | SIGIR | IR scoring baseline |
| Rocchio | 1971 | Book | Term-weight classification |
| Hinton KD | 2015 | NeurIPS | Knowledge distillation |
| LLM2LLM (Lee et al.) | 2024 | arXiv | LLM-based data augmentation |
| Settles Active Learning | 2009 | Survey | Active learning framework |
| Multi-Intent SLU Survey | 2025 | ACL | Confirms gap ASV fills |
| Gonc & Saglam | 2023 | ICMI | Online learning for intents |

---

## 11. One-Paragraph Elevator Pitch

ASV Router is the first intent routing architecture that combines inverted-index scoring, dual-layer sparse vectors with online learning, non-neural multi-intent decomposition with relation detection, and LLM-as-teacher knowledge distillation — all with zero external dependencies. It starts with LLM-supervised verification and progressively "graduates" to autonomous routing as learned term-weight associations accumulate, inverting the typical AI cost curve. On standard benchmarks (CLINC150, BANKING77), ASV surpasses embedding-based routers after online learning while routing 150-500x faster. The December 2025 Multi-Intent SLU Survey confirms that non-neural multi-intent detection is absent from the literature and inter-intent relation classification is an open problem — ASV addresses both.

---

*This document is for internal reference and paper justification. Not for publication.*
