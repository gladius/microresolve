# ASV Benchmark Plan — Launch Edition

**Goal**: Produce credible, reproducible benchmark numbers that prove ASV's value across
its real use cases. Not just intent classification — tool routing, guardrails, multi-intent,
multilingual, and latency vs alternatives.

Every dataset runs two passes: **seed-only** then **after learning** (missed queries fed back
via `/api/learn/now`). The delta is the story — a router that improves from its own failures
is a different product category.

---

## Datasets & Coverage

### Track 1 — Intent Classification (foundational credibility)

These are the benchmarks every NLP evaluator knows. They establish ASV as a serious
routing system, not a toy.

| Dataset | Source | Intents | Examples | Domain |
|---|---|---|---|---|
| CLINC150 | HuggingFace `clinc_oos` | 150 | 22,500 | General / multi-domain |
| BANKING77 | HuggingFace `banking77` | 77 | 13,083 | Financial |
| HWU64 | HuggingFace `DH-arc/hwu64` | 64 | 25,716 | Smart home / personal assistant |
| MASSIVE | HuggingFace `AmazonScience/massive` | 18 | 19,521/lang | 51 languages |

**MASSIVE strategy**: run English baseline, then run 3-4 additional languages (Spanish, French,
Japanese, Arabic) to prove multilingual routing with no extra setup.

**Seed strategy**: extract up to 20 phrases per intent from training split, single-intent only.
No LLM, no human curation — purely dataset-derived. This is the honest cold-start number.

**Learning pass**: for every miss in the test split, call `/api/learn/now` with ground truth
intent. Then re-evaluate the same test queries. Report delta.

---

### Track 2 — Tool / API Routing (MCP / OpenAPI use case)

The hottest benchmark category in 2026. Maps directly to ASV's MCP and OpenAPI import.
Shows ASV can pre-select the right tool before the LLM sees the full tool list.

| Dataset | Source | Tools | Task |
|---|---|---|---|
| BFCL v3 | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | 2000+ | Function selection from natural language |
| ToolBench (subset) | `ToolBench/ToolBench` | 500 APIs (curated subset) | API endpoint selection |

**BFCL strategy**: use the "simple" and "multiple" function call subsets. ASV routes to the
correct function name — latency and accuracy vs LLM-based selectors.

**Seed strategy**: each tool's `description` + `name` becomes an intent. Parameters become
metadata. No seed phrases needed — import directly via `/api/import/spec`.

**Key metric**: what % of tool selections does ASV get right before the LLM is invoked?
Even 70% correct at 30µs vs 100% correct at 400ms is a compelling cost/latency trade.

---

### Track 3 — Guardrails & Safety (prompt injection / jailbreak use case)

Proves the guardrail namespace use case with real attack data.

| Dataset | Source | Examples | Task |
|---|---|---|---|
| JailbreakBench | `JailbreakBench/JailbreakBench-Attack-Artifacts` | 100 behaviors × attacks | Detect jailbreak attempts |
| AdvBench | `walledai/AdvBench` | 520 | Detect harmful instruction requests |
| PromptBench | `microsoft/promptbench` | ~2000 | Adversarial prompt detection |

**Strategy**: train a `guardrails-demo` namespace with intent classes:
- `prompt_injection` — instruction override attempts
- `jailbreak` — role-play escapes, DAN variants
- `off_topic` — out-of-scope requests
- `pii_request` — personal data requests
- `legal_threat` — threatening language

Seeds come from known attack patterns in the datasets. Test on held-out attacks.

**Key metric**: detection rate (true positive) and false positive rate on clean queries.
The claim: ASV detects attacks in 30µs before the LLM ever sees the payload.

---

### Track 4 — Multi-Intent Decomposition (unique differentiator)

No other router benchmarks multi-intent natively. This is a clean ASV win.

| Dataset | Source | Intents | Multi-intent % |
|---|---|---|---|
| MixSNIPS | HuggingFace `nahyeon00/mixsnips_clean` | 7 | 80% |
| MixATIS | GitHub `LooperXX/AGIF` | 18 | 83% |

**Already have results** from `tests/data/benchmarks/results.txt`. Rerun with current
pipeline to confirm numbers, then add after-learning pass.

**Key metrics**: exact match, partial match (any intent found), F1, avg recall.

---

### Track 5 — Latency vs Alternatives (cross-cutting)

Not a dataset — a controlled head-to-head latency comparison on the same query set
(CLINC150 test split, 150 intents).

| System | Type | Latency | Cost/1M queries | Accuracy |
|---|---|---|---|---|
| ASV (seed-only) | Deterministic | 30µs | $0 | TBD |
| ASV (+ learning) | Deterministic | 30µs | $0 | TBD |
| sentence-transformers (MiniLM) | Embedding | ~8ms | $0 + GPU | TBD |
| OpenAI text-embedding-3-small | Embedding API | ~80ms | ~$20 | TBD |
| GPT-4o-mini (zero-shot classify) | LLM | ~400ms | ~$100 | TBD |

**Methodology**: same 150 intents, same test queries, measure wall-clock latency
(p50/p95/p99) and accuracy. ASV runs locally. Embedding models via sentence-transformers.
LLM via API. Report honest numbers — ASV wins on latency/cost, LLM wins on accuracy.
The story is the trade-off, not a false claim of superiority.

---

## Output Format

Each track produces a results file in `tests/data/benchmarks/results/`:

```
results/
  clinc150.json          — seed-only + after-learning
  banking77.json
  hwu64.json
  massive_multilingual.json
  bfcl.json
  toolbench.json
  guardrails.json
  mixsnips.json
  mixatis.json
  latency_comparison.json
```

And a single `BENCHMARKS.md` at the root summarising all results in human-readable tables,
ready to paste into README and blog posts.

---

## Implementation Plan

### Phase 1 — Infrastructure (shared runner)

Write `tests/data/benchmarks/run_all.py`:

```
1. Check server is running (GET /api/health)
2. For each dataset:
   a. Download if not cached (HuggingFace datasets / git clone)
   b. Create fresh namespace via POST /api/namespaces
   c. Load seeds via POST /api/intents + POST /api/intents/phrase
   d. Run test queries → record seed-only results
   e. For each miss: POST /api/learn/now with {query, ground_truth}
   f. Run same test queries again → record post-learning results
   g. Compute delta, save JSON
   h. Delete namespace (cleanup)
3. Print summary table
4. Write BENCHMARKS.md
```

Shared utilities in `tests/data/benchmarks/lib.py`:
- `create_namespace(name)` / `delete_namespace(name)`
- `load_seeds(namespace, intent_phrases)`
- `run_queries(namespace, queries)` → results
- `apply_learning(namespace, misses)` → phrases added
- `compute_metrics(results)` → accuracy, F1, latency

### Phase 2 — Dataset Downloaders

Extend `download_datasets.py` with:
- `download_clinc150()` — HuggingFace `clinc_oos`
- `download_banking77()` — HuggingFace `banking77`
- `download_hwu64()` — HuggingFace `DH-arc/hwu64`
- `download_massive(langs)` — HuggingFace `AmazonScience/massive`
- `download_bfcl()` — gorilla-llm BFCL v3 simple + multiple subsets
- `download_guardrails()` — JailbreakBench + AdvBench

### Phase 3 — Track Runners

One runner per track, callable independently:
- `run_track1_intent.py` — CLINC150, BANKING77, HWU64, MASSIVE
- `run_track2_tools.py` — BFCL, ToolBench subset
- `run_track3_guardrails.py` — JailbreakBench, AdvBench
- `run_track4_multiintent.py` — MixSNIPS, MixATIS (rerun existing)
- `run_track5_latency.py` — head-to-head latency comparison

### Phase 4 — Results & README

- Collect all JSON results
- Write `BENCHMARKS.md` with clean tables
- Update `README.md` benchmarks section with real numbers

---

## Seed Strategy (Honest Methodology)

To keep numbers honest and reproducible:

- **Max 20 seed phrases per intent** from training split only
- **Single-intent examples only** for seeding (no leakage from multi-intent test distribution)
- **No LLM-generated seeds** for baseline (LLM seeds tested separately as an ablation)
- **No human curation** — purely automated
- **Fresh namespace per run** — no state leakage between experiments
- **Same random seed** for any sampling

This matches what a real user would do on day 1: import a spec or paste some examples,
no manual work.

---

## Metrics

| Track | Primary | Secondary |
|---|---|---|
| Intent classification | top-1 accuracy | top-3 accuracy, latency |
| Tool routing | top-1 accuracy | top-3 accuracy, latency vs LLM |
| Guardrails | detection rate (TPR) | false positive rate (FPR) |
| Multi-intent | F1 | exact match, partial match, recall |
| Latency | p50 / p95 latency | cost per 1M queries |

---

## What We Can Claim After This

- "Routes 150 intent classes at 30µs, 87% top-1 accuracy out of the box, 94%+ after learning from 1 day of traffic" (CLINC150)
- "Selects the correct API tool in X% of cases at 30µs, before the LLM sees the request" (BFCL)
- "Detects jailbreak attempts in X% of cases with Y% false positive rate" (JailbreakBench)
- "Decomposes multi-intent queries with 90% recall, 83% F1" (MixSNIPS)
- "1000× faster than GPT-4o-mini classification at Z% of the accuracy" (latency comparison)

Each claim maps to a use case. Each use case maps to a blog post.

---

## Dependencies

```
pip install datasets huggingface_hub sentence-transformers openai
```

Server must be running on `localhost:3001` before running benchmarks.
ANTHROPIC_API_KEY or equivalent needed only for Track 5 LLM comparison.

---

## Timeline

| Phase | Work |
|---|---|
| Phase 1 — Infrastructure | Shared runner + lib utilities |
| Phase 2 — Downloaders | All dataset download scripts |
| Phase 3 — Track 1 | Intent benchmarks (CLINC150, BANKING77, HWU64, MASSIVE) |
| Phase 4 — Track 4 | Multi-intent rerun (MixSNIPS, MixATIS) |
| Phase 5 — Track 2 | Tool routing (BFCL) |
| Phase 6 — Track 3 | Guardrails (JailbreakBench, AdvBench) |
| Phase 7 — Track 5 | Latency comparison |
| Phase 8 — Results | BENCHMARKS.md + README update |
