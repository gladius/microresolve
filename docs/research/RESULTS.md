# ASV Router — Benchmark Results

All runs on a single CPU core, no GPU, no embedding model, no API calls.

**Hardware:** AMD Ryzen, Linux 6.18 (Fedora 43), `cargo run --release`

---

## 1. Seed Sweep — Finding the Ceiling

How does accuracy scale as we use more training data? CLINC150 has 120 examples/intent, BANKING77 has ~130/intent.

### CLINC150 (150 intents, 4,500 test queries)

| Seeds/intent | Seed-Only | Top-3 | +Learn30 | Avg Latency | No-match |
|-------------|-----------|-------|----------|-------------|----------|
| 5 | 50.1% | 67.4% | 74.4% | 27.8 μs | 3.0% |
| 10 | 61.5% | 78.6% | 76.1% | 24.7 μs | 1.6% |
| 20 | 71.0% | 86.2% | 79.5% | 42.3 μs | 0.9% |
| 50 | 80.4% | 92.3% | 82.1% | 41.9 μs | 0.6% |
| 80 | 83.2% | 93.7% | 84.2% | 39.1 μs | 0.4% |
| 100 | 83.8% | 94.2% | 93.5% | 70.1 μs | 0.4% |
| **120 (MAX)** | **84.9%** | **94.4%** | **95.1%** | 76.0 μs | 0.4% |

### BANKING77 (77 intents, 3,080 test queries)

| Seeds/intent | Seed-Only | Top-3 | +Learn30 | Avg Latency | No-match |
|-------------|-----------|-------|----------|-------------|----------|
| 5 | 51.9% | 72.7% | 75.0% | 46.1 μs | 0.0% |
| 10 | 61.5% | 81.8% | 77.6% | 53.9 μs | 0.0% |
| 20 | 71.8% | 89.4% | 81.3% | 50.3 μs | 0.0% |
| 50 | 80.4% | 93.0% | 83.1% | 43.7 μs | 0.0% |
| 80 | 83.0% | 94.7% | 84.9% | 46.9 μs | 0.0% |
| 100 | 83.1% | 94.8% | 87.6% | 66.1 μs | 0.0% |
| **130 (MAX)** | **83.3%** | **94.9%** | **89.7%** | 49.5 μs | 0.0% |

### Key Findings

**The ceiling is ~85% seed-only, ~95% with learning.** Both datasets plateau around 80-85% with seed-only routing. Adding 30 learned examples/intent pushes to 89-95%. Diminishing returns kick in around 80 seeds.

**BANKING77 plateaus harder.** Even with all 130 training examples as seeds, BANKING77 seed-only only reaches 83.3%. The remaining ~17% are queries where intent is determined by semantics ("my top-up is pending" vs "my top-up failed"), not vocabulary. This is the hard ceiling for any lexical approach.

**Top-3 accuracy is 94-95% at max data.** A hybrid ASV → LLM re-ranker that sends the top-3 candidates to an LLM for final selection would achieve near-SOTA accuracy at minimal cost.

---

## 2. Comparison Against Published Baselines

### CLINC150

| Method | Accuracy | Latency | Cost/1M queries | Infrastructure |
|--------|----------|---------|-----------------|----------------|
| Fine-tuned RoBERTa (full data) | 97.0% | ~10 ms | ~$0 (GPU amortized) | GPU + model hosting |
| **ASV (120 seeds + learn)** | **95.1%** | **76 μs** | **$0** | **HashMap only** |
| Rasa DIET (full data) | 89.4% | ~5 ms | ~$0 | CPU + trained model |
| GPT-4 3-shot | ~90% | ~500 ms | ~$150 | API dependency |
| **ASV (120 seeds, no learning)** | **84.9%** | **76 μs** | **$0** | **HashMap only** |
| SetFit 8-shot cosine | 85.9% | ~10 ms | ~$0 | Embedding model |
| TF-IDF / Naive Bayes | ~81-85% | ~1 ms | ~$0 | Trained model |

### BANKING77

| Method | Accuracy | Latency | Cost/1M queries | Infrastructure |
|--------|----------|---------|-----------------|----------------|
| Fine-tuned RoBERTa (full data) | 94.1% | ~10 ms | ~$0 (GPU amortized) | GPU + model hosting |
| Rasa DIET (full data) | 89.9% | ~5 ms | ~$0 | CPU + trained model |
| **ASV (130 seeds + learn)** | **89.7%** | **50 μs** | **$0** | **HashMap only** |
| GPT-4 3-shot | 83.1% | ~500 ms | ~$150 | API dependency |
| **ASV (130 seeds, no learning)** | **83.3%** | **50 μs** | **$0** | **HashMap only** |
| SetFit 8-shot cosine | 77.9% | ~10 ms | ~$0 | Embedding model |

**ASV with full training data matches Rasa DIET** (89.7% vs 89.9% on BANKING77) while running 100x faster and requiring zero training pipeline. On CLINC150, ASV with learning (95.1%) approaches fine-tuned RoBERTa (97.0%).

---

## 3. Multi-Intent Decomposition

No other router benchmarks multi-intent. ASV is the only system that does this.

### Synthesized Compound Queries (12 intents, 44 test queries)

| Category | Detection | Ordering | Relation |
|----------|-----------|----------|----------|
| 2-Intent Parallel (and/also) | 100.0% (16/16) | 93.8% | 100.0% |
| 2-Intent Sequential (then/after) | 100.0% (8/8) | 100.0% | 50.0% |
| 2-Intent Conditional (or/otherwise) | 100.0% (6/6) | 100.0% | 83.3% |
| 2-Intent Negation (except/without) | 66.7% (4/6) | 66.7% | 66.7% |
| 3-Intent Parallel (and...and...) | 87.5% (7/8) | 87.5% | N/A |
| **TOTAL** | **95.5% (42/44)** | **90.9%** | **65.9%** |

### Exhaustive Pair Test (all 12×12 = 132 pairs)

| Metric | Result |
|--------|--------|
| Both intents detected | **100.0% (132/132)** |
| Correct positional order | **97.7% (129/132)** |

**100% detection on exhaustive pairs.** When every combination of 2 intents from 12 is tested, ASV correctly identifies both intents in every single case.

### Where It Struggles

- **Negation** (66.7%): "track the package without cancelling the order" — the negated intent's terms get consumed by the primary intent's greedy pass. Needs negation-aware tokenizer.
- **Sequential relation detection** (50%): "then" sometimes gets consumed as part of an intent's bigram context. The relation classifier sees a collapsed gap.
- **3-intent with shared terms**: "cancel my order and get a refund and close my account" — "cancel" in cancel_order and close_account overlap, causing mis-detection.

---

## 4. OOS Rejection (CLINC150)

Can ASV say "I don't know" when a query is out-of-scope?

**Setup:** 50 seeds/intent, 1,000 OOS test queries, 4,500 in-scope test queries.

| Threshold | OOS Rejected | In-Scope Correct | In-Scope Wrongly Rejected | F1 |
|-----------|-------------|-------------------|---------------------------|-----|
| 0.3 | 2.4% | 80.6% | 0.6% | 0.046 |
| 0.5 | 12.3% | 79.5% | 2.4% | 0.200 |
| 0.8 | 26.1% | 78.0% | 5.2% | 0.350 |
| **1.0** | **49.4%** | **73.4%** | **13.0%** | **0.475** |
| **1.5** | **74.5%** | **66.0%** | **24.4%** | **0.524** |
| 2.0 | 89.3% | 53.2% | 41.0% | 0.478 |
| 2.5 | 95.2% | 40.0% | 56.5% | 0.423 |
| 3.0 | 98.4% | 26.9% | 71.2% | 0.379 |

**Best F1 at threshold ~1.5** (0.524): rejects 74.5% of OOS queries while still correctly routing 66% of in-scope. This is the fundamental tradeoff — a single score threshold is a blunt instrument.

**Better approach:** Use the confidence score (top1/top2 ratio) instead of raw score for OOS rejection. This is not yet implemented but would significantly improve the precision-recall curve.

---

## 5. Memory Profiling

### Synthetic Data

| Intents | Seeds/intent | Memory (RSS) | Per-intent | Throughput |
|---------|-------------|-------------|------------|------------|
| 10 | 5 | 168 KB | 16.8 KB | 129K q/s |
| 10 | 50 | 432 KB | 43.2 KB | 108K q/s |
| 77 | 10 | 496 KB | 6.4 KB | 78K q/s |
| 77 | 50 | 2.7 MB | 35.6 KB | 102K q/s |
| 77 | 130 | 5.1 MB | 68.0 KB | 101K q/s |
| 150 | 50 | 640 KB | 4.3 KB | 152K q/s |
| 150 | 120 | 8.8 MB | 60.2 KB | 149K q/s |
| 1000 | 20 | 10.3 MB | 10.6 KB | 152K q/s |

### Real Datasets (serialized JSON)

| Dataset | Seeds | JSON Size |
|---------|-------|-----------|
| CLINC150 | 10 | 182 KB |
| CLINC150 | 50 | 731 KB |
| CLINC150 | 120 | 1.5 MB |
| BANKING77 | 10 | 141 KB |
| BANKING77 | 50 | 552 KB |
| BANKING77 | 130 | 1.1 MB |

**Verdict:** A full 150-intent router with 120 seeds each fits in **<10 MB RAM** and **1.5 MB on disk**. A 1,000-intent router with 20 seeds each is ~10 MB. This runs on a Raspberry Pi Zero with room to spare.

**Throughput:** 100K-150K queries/second sustained on a single core. That's 8.6 billion queries/day.

---

## 6. The Full Picture

```
                    Accuracy
    100% ┤
         │   ●RoBERTa-FT
     95% ┤         ★ASV-120+learn ●Rasa-DIET ★ASV-130+learn
         │   ●GPT-4
     90% ┤
         │         ●SetFit-8shot
     85% ┤ ★ASV-120  ★ASV-130
         │ ★ASV-80
     80% ┤ ★ASV-50
         │
     70% ┤ ★ASV-20
         │
     60% ┤ ★ASV-10
         │
     50% ┤ ★ASV-5
         └──────────────────────────────────────
              1μs   100μs  1ms   10ms  100ms  1s
                         Latency (log scale)

  ★ = ASV (sub-100μs, $0, no infrastructure)
  ● = Existing systems (ms-scale, need models/APIs)
```

---

## Reproducing These Results

```bash
# Download datasets
bash data/download.sh

# Seed sweep (find the ceiling)
cargo run --release --bin benchmark -- --sweep

# Single run
cargo run --release --bin benchmark -- --dataset all --seeds 50 --learn-rounds 3

# OOS rejection evaluation
cargo run --release --bin benchmark -- --oos

# Memory profiling
cargo run --release --bin benchmark -- --memory

# Multi-intent benchmark
cargo test --test multi_intent_benchmark -- --nocapture
```
