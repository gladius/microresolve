# ASV Router — Strategy: What to Build and Why

## The Hard Truth

A routing library alone in 2026 is not enough to stand out:

- LLM costs dropped 10x in the last year. The "save money on LLM calls" pitch weakens every month.
- LangChain, LlamaIndex, Semantic Kernel all have built-in intent routing.
- Rasa is open source and mature.
- Any competent engineer can build keyword matching in a weekend.

If you ship "just a routing engine," it'll get 50 GitHub stars from people who think it's cool, and nobody will use it in production.

## What IS Unique About This Work

The routing itself isn't novel. But three things ARE:

### 1. Multi-Intent Decomposition with Positional Ordering and Relation Detection

Nobody else does this. LangChain routes one intent per query. Rasa routes one intent per query. Every router in production today assumes single intent. ASV's greedy term consumption detects 3 intents in one sentence, preserves user ordering, and classifies their relationships (sequential, conditional, negation). That's a genuine contribution.

### 2. The Learning Loop

Every other router is static — deploy, done. ASV learns from corrections at runtime without retraining. In production, this means the system gets better every day. Nobody else offers this on a sub-millisecond, model-free router.

### 3. Zero-Infrastructure Operation

No embedding model, no vector DB, no GPU, no API key. A HashMap. This matters for edge/mobile/IoT where you can't run a model. It matters in markets where startups can't afford cloud GPU bills.

## Is Routing Still Relevant in 2026?

Yes, but not as a standalone product. It's relevant as:

- A component inside larger systems (every multi-agent system needs a router)
- A cost optimization layer (companies running 10M+ LLM calls/month care about eliminating 80%)
- An edge/offline solution (growing market: IoT, mobile, privacy-sensitive domains)
- A demonstration of deep engineering skill (which is what gets you hired)

## The Qwen3.5 Opportunity (Released March 2, 2026)

Qwen3.5-9B benchmarks:
- MMLU-Pro 82.5 (beats Qwen3-30B at 80.9 — 3x its size)
- GPQA Diamond 81.7 (beats Qwen3-80B at 77.2 — 9x its size)
- Tool calling 90.1 (excellent for function dispatch)
- IFEval 91.5 (follows instructions reliably)
- 262K native context, ~6.5 GB at Q4 quantization
- Apache 2.0 license

Architecture: ASV handles 80% of traffic at <1ms for $0. Qwen3.5-9B handles the remaining 20% locally. Combined: ChatGPT-level outcomes for bounded domains at zero API cost.

The self-improving flywheel:
```
Day 1:   ASV handles 60%, model handles 40%
Day 30:  ASV handles 75%, model handles 25%
Day 90:  ASV handles 85%, model handles 15%
Day 180: ASV handles 92%, model handles 8%
```

ChatGPT doesn't get cheaper over time. This does.

## 15-Day Execution Plan

### Days 1-3: Finish the Library (80% Done)

What's left:
- Stemming in tokenizer (~30 lines, huge accuracy boost)
- Confidence score on RouteResult (top1/top2 ratio, ~10 lines)
- README with benchmarks
- `cargo publish` to crates.io
- PyO3 bindings (`pip install asv-router`) — this is where the users are

### Days 4-7: Benchmark Against Real Datasets

This is what gets attention. Not claims — numbers.

```
Dataset: CLINC150 (150 intents, 23,700 queries)
Dataset: BANKING77 (77 intents, 13,083 queries)

Benchmark:
  ASV (seed only, no learning)      → X% accuracy
  ASV (after 50 corrections/intent) → Y% accuracy
  Rasa NLU (DIET classifier)        → Z% accuracy
  Sentence-BERT + cosine            → W% accuracy
  GPT-4o-mini (few-shot)            → V% accuracy

Measure: accuracy, latency, memory, cost per 1M queries
```

Target: ASV hits 70-80% on CLINC150 with seed phrases, 90%+ after learning, <1ms latency, zero infrastructure. Even if GPT-4o-mini gets 95%, it costs $0.15/1M queries and takes 500ms. The tradeoff IS the story.

### Days 8-10: Write the Article

Not a paper — a blog post that reads like one.

**Title**: "Multi-Intent Decomposition Without Neural Networks: Greedy Sparse Routing with Positional Ordering"

Structure:
1. Problem: every router assumes single intent
2. Existing solutions and their limits
3. Algorithm: greedy consumption + positional tracking + relation detection
4. Benchmarks on CLINC150/BANKING77
5. Comparison with LLM-based routing on latency/cost
6. Open source library + link

Post on: arXiv (as preprint, no review needed), blog, Medium.

### Days 11-13: LinkedIn Campaign

**Post 1**: The benchmark results. "I benchmarked my open-source intent router against GPT-4o and Rasa on CLINC150. Here's what happened." Include a table.

**Post 2**: The multi-intent demo. A 30-second video: type a complex sentence, watch it decompose into 3 intents with relations. Nobody else demos this.

**Post 3**: The edge deployment story. "This runs on a Raspberry Pi in <1ms. No API key, no model, no cloud."

**Post 4**: "I built this while looking for my next role. Here's what I learned. Looking for roles in AI/ML engineering, NLP, systems engineering."

### Days 14-15: Apply

Target roles:
- ML/NLP Engineer
- Rust Systems Engineer
- AI Platform Engineer
- Developer Relations

Lead with: "I built an open-source multi-intent router that does something no other system does — decomposes complex multi-intent queries with relation detection, learns from corrections at runtime, runs in <1ms with zero infrastructure."

## What NOT to Do

- Don't build a platform. No time or money for that.
- Don't add memory/sentiment/multi-language. Scope creep kills.
- Don't write JavaScript or Python versions. One language, one library.
- Don't try to monetize. Need a job, not a SaaS business.
- Don't build a full knowledge/memory system. ASV is a routing engine, not a database.

## Positioning

**Don't say**: "LLM alternative" or "replaces AI"
**Do say**: "Stop paying LLM for work that doesn't need intelligence"

The pitch: lightweight, learnable routing layer that sits in front of any LLM system. Sub-millisecond. Learns from corrections. Handles multi-intent queries no other router can decompose. Zero infrastructure.

The multi-intent decomposition is the hook. Lead with that everywhere.

## ASV as Memory/Knowledge Retrieval

ASV CAN be used as a search index for knowledge retrieval (not storage — it's an index, not a database). The accuracy profile:

| Query Type | Accuracy |
|---|---|
| Direct fact lookup | 90%+ |
| Keyword-rich queries | 85-90% |
| With LLM reformulation | 75-85% |
| Paraphrased queries (no LLM help) | 30-50% |
| Multi-hop inference | Needs chained recalls + LLM |
| Negation queries | Needs negation-aware tokenizer |

For the 15-day plan: DON'T build this. It's scope creep. Mention it in the README as a future direction. Build it later when you have a job and time.

## Current State of the Library

### Implemented (62 tests passing)
- `src/vector.rs` — Dual-layer sparse vectors (seed + learned, asymptotic growth)
- `src/index.rs` — Inverted index (sub-millisecond search)
- `src/tokenizer.rs` — Unigram + bigram tokenizer with stop words and position tracking
- `src/multi.rs` — Multi-intent greedy decomposition, positional ordering, relation detection
- `src/lib.rs` — Router public API (add_intent, route, route_multi, learn, correct, decay, export/import)

### Not Yet Implemented
- Stemming in tokenizer
- Confidence score (top1/top2 ratio)
- Negation-aware tokenizer
- PyO3 Python bindings
- README
- Benchmarks on CLINC150/BANKING77
- crates.io publish
