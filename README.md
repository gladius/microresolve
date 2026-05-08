# MicroResolve

[![crates.io](https://img.shields.io/crates/v/microresolve.svg)](https://crates.io/crates/microresolve)
[![PyPI](https://img.shields.io/pypi/v/microresolve.svg)](https://pypi.org/project/microresolve/)
[![npm](https://img.shields.io/npm/v/microresolve.svg)](https://www.npmjs.com/package/microresolve)
[![docs.rs](https://docs.rs/microresolve/badge.svg)](https://docs.rs/microresolve)
[![CI](https://github.com/gladius/microresolve/actions/workflows/ci.yml/badge.svg)](https://github.com/gladius/microresolve/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**MicroResolve is the System 1 relay for LLM apps.** Every request runs
through a sub-millisecond reflex layer that picks a candidate intent +
confidence band and hands the result to your **System 2** — your LLM,
or a human reviewer for high-stakes domains (HIPAA, legal, financial).
We never talk to your users; we give your decision-maker a head start.

Tool selection, intent triage, guardrail dispatch, refusal
classification — the routing decisions your LLM keeps making run in
~50 µs here and improve on your traffic via corrections.

**In the box**

- **Studio** — web UI for namespace management, simulation, review, training. Git-backed history + rollback.
- **4 reference packs** — `safety-filter`, `hipaa-triage`, `eu-ai-act-prohibited`, `mcp-tools-generic`. Pre-calibrated thresholds + voting-gate, drop into a data dir and go.
- **Library** — Python / Node / Rust, same Rust core. Embed in prod, or stay live-connected to a Studio.
- **Online learning** — Hebbian + LLM-judged corrections. No fine-tuning, no restart.
- **Native imports** — MCP, OpenAI functions, LangChain tools, OpenAPI specs.
- **Multilingual** — Latin + CJK tokenization; learns whichever language your traffic is in.

> v0.2 — early release; pin exact versions in production.

[**Documentation**](https://gladius.github.io/microresolve/) ·
[**Benchmarks & methodology**](benchmarks/) ·
[**Changelog**](CHANGELOG.md) ·
[**Contributing**](CONTRIBUTING.md)

## Adding System 1 to your LLM app — as simple as this

```python
from microresolve import MicroResolve
from openai import OpenAI                            # or anthropic, ollama, …

mr = MicroResolve()                                  # opens ~/.local/share/microresolve
safety = mr.namespace("safety-filter")               # `microresolve-studio install safety-filter`
llm = OpenAI()

def handle(query: str) -> str:
    # System 1 — deterministic pre-filter, ~50 µs, $0
    if any(i.band == "High" for i in safety.resolve(query).intents):
        return "Blocked by pre-LLM safety filter."

    # System 2 — your LLM (small model viable: catalog already narrowed)
    return llm.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": query}],
    ).choices[0].message.content
```

That's the whole integration. Branch on `result.disposition`
(`Confident` / `LowConfidence` / `NoMatch`) to decide whether to act,
escalate to the LLM with the candidate list, or fall through — see
[Bands & Disposition](https://gladius.github.io/microresolve/concepts-bands/).
Same shape in [Node](https://www.npmjs.com/package/microresolve) and
[Rust](https://docs.rs/microresolve).

Pack not what you need? Swap `safety-filter` for `mcp-tools-generic`
(tool routing), `hipaa-triage` (medical query triage), or build your own
in the [Studio](#studio-single-binary-ui--http-server). Auto-learn from
corrections, multi-intent decomposition, live FP/recall tuning all run
in the same binary.

## Install

### Python

```bash
pip install microresolve
```

### Node.js

```bash
npm install microresolve
```

### Rust

```bash
cargo add microresolve
```

### Studio (single-binary UI + HTTP server)

Pre-built tarballs for Linux (x86_64 / aarch64, glibc + musl), macOS
(x86_64 / aarch64), and Windows ship on every release.

```bash
# Linux x86_64 — adjust for your platform from the releases page
curl -L https://github.com/gladius/microresolve/releases/latest/download/microresolve-studio-x86_64-unknown-linux-gnu.tar.gz \
  | tar xz

# One-time interactive setup: data dir, port, optional LLM key
./microresolve-studio config

# Install a reference pack (see the table below for available packs)
./microresolve-studio install safety-filter
./microresolve-studio install hipaa-triage   # or any of the other 4

# Start the Studio (uses ~/.config/microresolve/config.toml)
./microresolve-studio
# Studio at http://localhost:4000
```

All artifacts come from the same source-of-truth Rust core — same algorithm,
same data files, fully interchangeable.

## Why this lets you use a smaller LLM

200-tool catalogs force the LLM to be a frontier model — small models
drop tools beyond ~50 in catalog and hallucinate calls on the long
tail. MicroResolve narrows to ~3 candidates in 50µs, so the LLM that
follows can be a small one.

```
without:  query → 200 schemas → frontier model     → ~$0.03  · 1.5s
with:     query → 50µs prefilter → 3 → small model → ~$0.0002 · 0.3s
```

| | Today | With MicroResolve |
|---|---|---|
| Prompt | 20K tokens (200 schemas) | 300 tokens (3 candidates) |
| Model | GPT-5 / Sonnet 4.6 / Gemini Pro | GPT-5 nano / Haiku 4.5 / Flash |
| Cost / call | ~$0.03 | ~$0.0002 |
| Latency | 1.5s | 0.3s |

50–200× cheaper, 3–5× faster. When confidence is low, the LLM gets
the full catalog as fallback — see
[Bands & Disposition](https://gladius.github.io/microresolve/concepts-bands/).

## Reference packs

Four pre-curated packs ship as v0.2.1 release tarballs. Install via
`microresolve-studio install <pack>` (CLI fetches the tarball matching
your binary version), or copy from [`packs/`](packs/) into any data dir
manually.

| Pack | Intents | Seeds | Default | What it's for |
|---|---|---|---|---|
| **`safety-filter`** | 5 | 100 | min=3, thr=1.5 | Pre-LLM jailbreak / prompt-injection detection. **98% recall / 8% FP** on 50/50 eval. Pair with a dedicated safety classifier (LlamaGuard / Prompt-Guard) for adversarial coverage. |
| **`eu-ai-act-prohibited`** | 6 | 70 | min=2, thr=1.5 | Article 5 prohibited-practice triage. **85% top-1 / 6% FP**. Pair with lawyer review for final determination. |
| **`hipaa-triage`** | 6 | 743 | min=3, thr=1.5 | Medical query triage (clinical_urgent, clinical_routine, mental_health_crisis, administrative, billing, scheduling). **96.9% top-1 / 36.5% FP** at default; **94.8% / 21.2% at thr=2.0** for stricter precision. Triage filter, not a final decision — pair with LLM judgment or human review. **Not a HIPAA compliance solution.** |
| **`mcp-tools-generic`** | 7 | 70 | min=2, thr=1.5 | Generic MCP-style tool router (web_search, send_message, fetch_url, file_operations, database_query, code_execution, calendar_management). For closed-domain tool dispatch — open-ended chat traffic produces FPs from idiomatic English. |

> Each pack ships with calibrated `default_threshold` + `default_min_voting_tokens`. Tune live in the Studio sidebar (TuningPanel) or via `PATCH /api/namespaces` for your FP/recall trade-off.

## Benchmarks

Headline numbers — full methodology, datasets, and reproduction scripts in
[`benchmarks/`](benchmarks/):

- **Agent tool routing**, 129 real tools across 5 MCP servers (Stripe / Linear / Notion / Slack / Shopify): **76.5% top-1, 88.2% top-3** cold-start; **88.2% / 97.1%** after corrections. p50 **64–87 µs**. No LLM at runtime.
- **CLINC150** (150 intents, 20 seeds/intent): **80.1%** top-1 cold, **97.4%** after-learning (4500 test).
- **BANKING77** (77 intents, 20 seeds/intent): **73.15%** cold, **94.6%** after-learning (3080 test).
- **In-process Rust** (`cargo bench --bench resolve`): mean **~15 µs** / query
  (criterion, 100-intent synthetic namespace). Through the Python wheel:
  ~100 µs / query end-to-end.

## Audit & compliance

Direct continuation of the v0.2.0 compliance packs
(`eu-ai-act-prohibited`, `hipaa-triage`) — those packs shipped first;
this is the tamper-evident chain that makes them deployable in
regulated environments.

Every routing decision and every namespace mutation is recorded in a
per-key SHA-256 hash chain at `{data_dir}/_audit/{kid}.log` — same
pattern Certificate Transparency (RFC 6962) uses for HTTPS. Identity =
one App-scope API key per workload (Deployment / service / tenant), not
per pod; replicas share the key via a Secret. On by default; off via
`[audit].mode = "off"` in `config.toml`. Query content is stored as a
SHA-256 hash (PII-safe).

```bash
microresolve-studio verify-log                          # cron-friendly integrity check
microresolve-studio export-log --since 30d > audit.jsonl  # hand off to SIEM / auditor
```

Maps onto **EU AI Act Art. 13**, **HIPAA §164.312(b)**, **SOC 2 CC7.2**,
**NIST AI RMF Govern**. Suitable for SMB / regulated-but-not-certified
deployments; no SOC 2 attestation, no managed service required.

## Architecture, multi-intent, multilingual, HTTP API

Deeper concept docs live on the [documentation site](https://gladius.github.io/microresolve/concepts/):

- [Concepts](https://gladius.github.io/microresolve/concepts/) — classification pipeline, multi-intent decomposition, projected context (co-occurrence), multilingual / CJK tokenization
- [Bands & Disposition](https://gladius.github.io/microresolve/concepts-bands/) — the System 1 → System 2 confirm-turn pattern, including the `confirm_full_catalog` fallback for tool routing
- [HTTP API reference](https://gladius.github.io/microresolve/server/api/) — namespaces via `X-Namespace-ID`; core endpoints `/api/resolve`, `/api/intents`, `/api/training/*`, `/api/import/*`
- [Threshold tuning](https://gladius.github.io/microresolve/threshold-tuning/) — calibrating threshold + voting-gate per pack

## Commercial support

I help teams ship MicroResolve in regulated environments — HIPAA,
financial, legal, government — where the self-serve path isn't enough.
Custom packs for your domain, threshold/eval calibration on your real
traffic, on-prem deployment review, integration help. Solo author,
project-based engagements, no enterprise SLAs.

Contact: [gladius.thayalarajan@gmail.com](mailto:gladius.thayalarajan@gmail.com)

## License

Dual-licensed under **MIT** or **Apache-2.0** at your option — the
standard Rust ecosystem licensing. Both are fully permissive and allow
commercial use.

- [LICENSE-MIT](LICENSE-MIT)
- [LICENSE-APACHE](LICENSE-APACHE) — adds an explicit patent grant

### Contribution

Unless you state otherwise, any contribution intentionally submitted for
inclusion in this work shall be dual-licensed as above, without any
additional terms or conditions.
