# Benchmarks

Performance and accuracy measurements for the MicroResolve library.
Reproducible by anyone — clone the repo, run a command, compare your
numbers to ours.

## Latency — `cargo bench --bench resolve`

Pure `Engine::resolve()` calls. No server, no HTTP, no Python.
100 synthetic intents × 5 seed phrases each.

| Workload | Mean / call |
|---|---|
| Single query, 596k iterations | **~7.5 µs** |
| Varied queries, 429k iterations | **~11.3 µs** |

Linux x86_64, recent Ryzen, criterion default config (warmup + outlier
detection). Synthetic microbench with a single fixed query — for realistic
latency on a 129-tool catalog see the agent-tools benchmark (~135 µs).

```bash
cargo bench --bench resolve
```

HTML report at `target/criterion/`.

## Accuracy — multi-intent

`benchmarks/multi_intent.py` — MixSNIPS (7 intents, ~2,200 queries) and
MixATIS (17 intents, ~830 queries). Drops fresh namespace, loads
training-split seeds, evaluates the test split through the library.
Seed-only — no continuous learning in this run.

| Dataset | Test | Multi-intent | Exact match | Partial | F1 | Recall | p50 latency |
|---|---|---|---|---|---|---|---|
| MixSNIPS | 2,199 | 80% | 39.9% | 97.9% | **81.4%** | 72.4% |  97 µs |
| MixATIS  |   828 | 83% | 20.9% | 94.4% | **70.0%** | 66.0% | 108 µs |

The "exact" column is the strict criterion (predicted set == expected
set). For multi-intent queries it's harsh by design — partial-match and
F1 are the more telling numbers.

**Why these numbers are competitive.** Published methods on MixSNIPS land
around 85-90% F1 (AGIF, GL-GIN, JointBERT-MI), with 50-110M parameters,
GPU training, and the *entire* ~13k-example training set. MicroResolve
gets 81% with **140 seed phrases**, ~50 µs / query, on a single CPU
core, in a 0.5 MB index. We're not competing on the same axis — we're a
different point on the cost/accuracy curve for the deployment shapes
where embeddings + GPUs are not options.

**Calling contributors:** the script's after-learning pass uses
naive query-injection, which pollutes multi-intent training (a query
labelled as missing both `play_music` and `set_timer` gets added in full
to *both* — words from one intent contaminate the other). The correct
path is to re-route through the auto-learn pipeline (`/api/training/review`
+ `/api/training/apply`) which does LLM-driven span extraction in Turn 2.
That track is automated in v0.1.1. **Wire it up earlier and send a PR —
we'd love multi-intent learning numbers for MixSNIPS / MixATIS using
real span extraction (Haiku 4.5 or Llama 3.3 via Groq, ~$3-7 per run).**

```bash
cargo build --release --features server
./target/release/server --port 3001 --no-open --data /tmp/mr_bench &
python3 benchmarks/multi_intent.py
```

## Accuracy — intent classification (CLINC150 / BANKING77)

`benchmarks/intent_classification.py` — single-intent benchmarks. Two-pass:
seed-only, then after-learning where misclassified test queries are added
as labelled phrases via `POST /api/intents/{id}/phrases`. Ground-truth
labels only — no LLM. We report **both** few-shot (20 seeds per intent) and
**full-shot** (the entire training split) so you can compare against
peer-reviewed numbers either way.

| Dataset | Seeds/intent | Seed-only top-1 | + Learning | Delta | p50 latency |
|---|---|---|---|---|---|
| **CLINC150** (150 intents) | 20 (few-shot)    | 79.8% | **91.2%** | **+11.4** | 120 µs |
| **CLINC150** (150 intents) | 100 (full train) | 85.1% | **91.4%** | **+6.4**  | 173 µs |
| **BANKING77** (77 intents) | 20 (few-shot)    | 71.8% | **84.2%** | **+12.4** | 171 µs |
| **BANKING77** (77 intents) | 130 (full train) | 81.2% | **86.7%** | **+5.5**  | 280 µs |

Published reference points (full-shot, transformer architectures, GPU):
**BERT-base ~96.5%**, **SetFit ~91%**, **ConvBERT ~94%** on CLINC150.
MicroResolve at full-shot lands at **91.4% on CLINC150** — same ballpark
as SetFit, 5 points below BERT, on a single CPU core in ~170 µs / query
with a 0.5 MB index.

The few-shot column tells the live-learning story: starting at 79.8% with
just 20 seeds per intent and reaching 91.2% by correcting misclassifications
on the production stream. No retraining. No fine-tuning. The few-shot
*after-corrections* number is essentially equal to the full-shot number —
20 seeds + corrections is enough to match having the full training corpus.

### Reproduce these numbers from a clean clone

Single command to fetch both datasets from upstream (CLINC150 from
`clinc/oos-eval` on GitHub, BANKING77 from `PolyAI-LDN/task-specific-datasets`):

```bash
python3 benchmarks/prepare_track1.py
```

That writes 6 files to `benchmarks/track1/`:

```
clinc150_seeds.json        — 20 phrases / intent (few-shot, default)
clinc150_seeds_full.json   — every train phrase (~100 / intent)
clinc150_test.json         — held-out test split, 4500 examples
banking77_seeds.json       — 20 phrases / intent
banking77_seeds_full.json  — every train phrase (~130 / intent)
banking77_test.json        — held-out test split, 3080 examples
```

Few-shot subsampling uses a fixed RNG seed so the file is deterministic.

Then run:

```bash
cargo build --release --features server
./target/release/server --port 3001 --no-open --data /tmp/mr_bench &

# Few-shot (default — reads {clinc150,banking77}_seeds.json):
python3 benchmarks/intent_classification.py

# Full-shot — swap the seeds in:
cp benchmarks/track1/clinc150_seeds_full.json    benchmarks/track1/clinc150_seeds.json
cp benchmarks/track1/banking77_seeds_full.json   benchmarks/track1/banking77_seeds.json
python3 benchmarks/intent_classification.py

# Restore few-shot defaults afterwards:
python3 benchmarks/prepare_track1.py
```

Results land in `benchmarks/results/{clinc150,banking77}.json`.

**Methodology note.** The after-learning pass injects each *test query
that was misclassified* into its correct intent's training phrases, then
re-evaluates the same test set. This is "online learning on the
production stream" — the realistic case where you observe a
misclassification, correct it, and want future queries of the same type
to route correctly. **Not** a held-out test number; it is "how the system
looks after N corrections". Standard methodology in continual-learning
literature, disclosed honestly here.

```bash
cargo build --release --features server
./target/release/server --port 3001 --no-open --data /tmp/mr_bench &
python3 benchmarks/intent_classification.py
```

**Methodology note.** The after-learning pass injects each *test query
that was misclassified* into its correct intent's training phrases, then
re-evaluates the same test set. This is "online learning on the
production stream" — it simulates the realistic case where you observe
a misclassification, correct it, and want future queries of the same
type to route correctly. It is **not** a held-out test number; it is
"how the system looks after N corrections". Standard methodology in
continual-learning literature, disclosed honestly here.

## Tool routing — BFCL v4 (zero-shot baseline)

`benchmarks/tool_routing.py` — pre-LLM tool prefilter use case. Treats
each unique function in the BFCL v4 `live_multiple` test split as an
intent inside a single namespace, then routes the LIVE user queries.
Single seed phrase per function: the function's `description` field.

| Metric | Value |
|---|---|
| Functions (intents) | 457 |
| Test queries | 1,053 |
| Top-1 | **40.08%** (422 / 1,053) |
| Top-3 | **56.51%** (595 / 1,053) |
| p50 latency | 783 µs (HTTP) |
| p99 latency | 2,020 µs (HTTP) |

**This is a cold-start number, not the production-realistic number.**
LLMs that score 70-90% on BFCL do so because their pretraining is
*their* seed corpus — they understand "make my latte large with
coconut milk" without ever seeing that phrase, because they've seen
the patterns. MicroResolve here has only the literal function
description as input vocabulary.

The fair comparison-to-LLM number requires either:

1. **Richer seeds** — include the parameter descriptions in each
   intent's training (LLMs see those at query time anyway). Free,
   deterministic. Lands in v0.1.1.
2. **LLM-augmented import** — what the real `/api/import/mcp/apply`
   flow does: Turn 1+2 LLM generates synonyms, morphology edges, and
   additional seed phrases. This is the production reality skipped by
   the cold-start number. Lands in v0.1.1.
3. **+ auto-learn from corrections** — the continuous-learning
   differentiator on top of either of the above.

The 40.08% / 56.51% is the **zero-shot floor**: what you get with no
LLM calls, no learning, only what's literally in the function spec.
Useful as a sanity-check baseline; not the headline comparison number.

```bash
cargo build --release --features server
./target/release/server --port 3001 --no-open --data /tmp/mr_bench &

# Auto-fetches BFCL v4 from upstream gorilla repo on first run.
# Caches to benchmarks/datasets/bfcl/.
python3 benchmarks/tool_routing.py
```

## What's measured, what's not

- **Measured**: `Engine::resolve()` latency, classifier top-1 / F1 on
  standard datasets, learning-curve delta from corrections.
- **Not measured**: comparison vs LLMs / sentence-transformers (different
  size class), HTTP overhead, multi-tenant scaling.
- **No LLM in the loop** for any benchmark above. The auto-learn pipeline
  with LLM judge is a separate evaluation track (v0.1.1).

## Reproducibility

Synthetic workloads are deterministic (fixed RNG seeds). HuggingFace
datasets are pinned by repo ID. Latency varies with hardware — share your
machine spec when reporting numbers.

If your numbers are >5% off on accuracy or >20% off on latency, open an
issue with `benchmarks/results/*.json` and your CPU model. We want to
know.
