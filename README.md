# MicroResolve

[![crates.io](https://img.shields.io/crates/v/microresolve.svg)](https://crates.io/crates/microresolve)
[![PyPI](https://img.shields.io/pypi/v/microresolve.svg)](https://pypi.org/project/microresolve/)
[![npm](https://img.shields.io/npm/v/microresolve.svg)](https://www.npmjs.com/package/microresolve)
[![docs.rs](https://docs.rs/microresolve/badge.svg)](https://docs.rs/microresolve)
[![CI](https://github.com/gladius/microresolve/actions/workflows/ci.yml/badge.svg)](https://github.com/gladius/microresolve/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A pre-LLM reflex layer for AI agents — intent routing and tool prefilter
in tens of microseconds. Embedded library with optional HTTP server.

> **v0.1 — early release.** API may change before 1.0. Pin exact versions
> in production.

[**Documentation**](https://gladius.github.io/microresolve/) ·
[**Benchmarks & methodology**](benchmarks/) ·
[**Changelog**](CHANGELOG.md) ·
[**Contributing**](CONTRIBUTING.md)

```text
"the dispute on order 1834 came back invalid — refund the customer,
 mark the dispute resolved, then post in #ops that we lost it"

  → create_refund        (confirmed, high)
  → update_dispute       (confirmed, high)
  → slack_send_message   (confirmed, medium)
  → relation: sequential

                          ~92 µs · narrowed 129 tools to top-3
                          for the LLM, with no LLM call yet
```

**Your LLM is System 2** — slow, deliberate, expensive. **MicroResolve is
System 1** — fast, learned, cheap. Pattern-match the easy 80% of queries
in microseconds; let the LLM reason about the hard 20%. MicroResolve
**learns continuously from LLM-judged corrections** — every misroute the
LLM fixes flows back into the index, so the candidate set sharpens from
production traffic alone.

**Use it for:**

- **LLM-agent tool prefiltering** — narrow 100+ tools to the top 3 the
  LLM actually needs to see this turn. Cuts prompt tokens, cuts latency,
  scales to large catalogs.
- **Customer-support triage** — route incoming tickets / chat messages
  to the right queue or workflow before the LLM gets involved (or
  without an LLM at all for the easy cases).
- **Intent classification** — single-utterance bucketing for
  conversational interfaces, IVR-style menus, search-query routing.
- **Slash / chat command routing** — recognize the user's command from
  free-form phrasing without retraining a model every time the catalog
  changes.
- **Workflow / decision routing** — multi-intent decomposition with
  relation detection (sequential / conditional / parallel / negation)
  for steps that need to fan out or chain.
- **Permission and risk gating** — classify a request into a risk tier
  before paying for an LLM round-trip.

|                    | LLM classification | Embedding model | **MicroResolve** |
|--------------------|--------------------|-----------------|------------------|
| Latency            | 200–2000 ms        | 10–50 ms        | **30–90 µs**     |
| Cost / query       | per-token          | GPU / API       | **$0**           |
| Setup              | prompt engineering | training data + GPU | seed phrases or one OpenAPI / MCP import |
| Continuous learning| retrain pipeline   | full retrain    | **incremental, in place** |
| Multi-intent       | prompt-dependent   | separate model  | **native**       |
| Dependencies       | API key            | PyTorch / ONNX  | **none at runtime** — Rust core with Python and Node bindings |

> [!IMPORTANT]
> **MicroResolve is a prefilter, not an absolute classifier.** It returns
> ranked candidates — the LLM is the final picker. Your agent prompt
> **must** include a `confirm_full_catalog` fallback tool so the LLM can
> reach the full catalog when none of the candidates fit. That fallback
> is part of the design, not an optional safety net. See
> [Confirm-turn pattern](#confirm-turn-pattern-system-1--system-2).

## Benchmarks

**Agent tool routing** — 129 real tools imported from 5 production MCP
servers (Stripe / Linear / Notion / Slack / Shopify), single namespace,
scored as a prefilter for an LLM picker:

| Stage                              | Single top-1 | **Single top-3** | Multi F1 | p50  |
|------------------------------------|--------------|------------------|----------|------|
| Cold start (LLM-seeded import)     | 76.5 %       | 88.2 %           | 40.9 %   | 87 µs |
| **+ auto-learn from corrections**  | **88.2 %**   | **97.1 %**       | **76.6 %** | **64 µs** |

`+ auto-learn` = incremental Hebbian + LLM-judged phrase ingestion from
production corrections. No retraining pipeline; the data updates in place.
Reproduce: `python3 benchmarks/agent_tools_bench.py` (~$0.55 / ~10 min on
Haiku 4.5).

> Methodology, datasets, and reproduction scripts live in
> [`benchmarks/`](benchmarks/). What's *not* yet benchmarked: out-of-scope
> rejection (see [Confirm-turn pattern](#confirm-turn-pattern-system-1--system-2)),
> adversarial robustness, drift over multi-week production traffic.

**Single-utterance intent classification** (academic baselines):

| Dataset    | Intents | Seeds | Top-1 (cold) | Top-1 (+ learning) | Top-3 | p50 latency |
|------------|---------|-------|--------------|--------------------|-------|-------------|
| CLINC150   | 150     | 50    | 84.0 %       | 85.8 %             | 94.2 % | 22 µs       |
| CLINC150   | 150     | 100   | 87.5 %       | 96.4 %             | 95.9 % | 24 µs       |
| BANKING77  | 77      | 50    | 81.9 %       | 85.0 %             | 94.1 % | 21 µs       |
| BANKING77  | 77      | 130   | 85.5 %       | 92.8 %             | 96.0 % | 23 µs       |

## Quick start

Embedded (Rust):

```rust
use microresolve::Engine;

let engine = Engine::open("./data")?;
let ns = engine.namespace("agent")?;
ns.add_intent("cancel_subscription", &["cancel my plan", "stop the recurring billing"])?;
let r = ns.resolve("end my subscription right now");
// r.ranked[0].id == "cancel_subscription"
```

HTTP server (with the optional Studio UI):

```bash
cp .env.example .env                        # set LLM_API_KEY
cargo run --release --bin server --features server -- --data ./data
cd ui && npm install && npm run dev         # http://localhost:3000
```

Bootstrap intents from an existing spec:

```bash
curl -X POST http://localhost:3001/api/import/mcp/apply \
  -H "Content-Type: application/json" \
  -H "X-Namespace-ID: agent" \
  -d '{"tools_json": "<MCP tools/list response>", "selected": ["..."], "domain": ""}'
```

Bindings: [`microresolve` on crates.io](https://crates.io/crates/microresolve),
[`microresolve` on PyPI](https://pypi.org/project/microresolve/),
[`microresolve` on npm](https://www.npmjs.com/package/microresolve).

## How it works

A query passes through five layers, all in-process, all in-memory:

```text
raw query
  → L0  typo correction        (character n-gram Jaccard)
  → L1  vocabulary bridging    (morphology / abbreviation, OOV-gated)
  → L2  intent scoring         (IDF-weighted sparse term graph)
  → L3  confidence + disposition
  → L4  cross-provider tiebreak
  → result

failures → review queue → auto-learn worker → L1 + L2 updated
```

L0 fixes typos against the known vocabulary. L1 substitutes morphological
variants (`canceling → cancel`) and abbreviations (`pr → pull request`)
**only when the source word is out-of-vocabulary**, so distinctive user
input is never rewritten. L2 is the core scorer: each intent has a learned
sparse vector of IDF-weighted terms; multi-intent uses a token-consumption
pass to confirm one intent then re-score the rest. L3 classifies the score
distribution as `confident` / `low_confidence` / `escalate`. L4 breaks ties
when the same action exists across multiple providers.

Every confirmed routing reinforces term weights. Every correction
shifts weights away from the wrong intent. No retraining; the data
updates in place.

## Confirm-turn pattern (System 1 → System 2)

MicroResolve returns ranked candidates; the LLM picks one or falls back
to the full catalog when nothing fits. **Always include a
`confirm_full_catalog` fallback tool in your agent prompt** — without it,
out-of-scope queries and novel phrasing route to the wrong tool:

```text
Candidate tools (from the prefilter):
  - tool_a  (...description...)
  - tool_b  (...description...)
  - tool_c  (...description...)

If one of these clearly applies, call it. Otherwise — unsure, out of
scope, or candidates look unrelated — call `confirm_full_catalog` to
receive every tool, then pick.
```

The prefilter shrinks 150 tools to 3 in ~60 µs; the LLM is the final
picker. Every confirmed call flows back as a corrected example, so the
candidate set sharpens over time.

## HTTP API

Send `X-Namespace-ID: my-namespace` to isolate intents per namespace. The
core endpoints are `/api/route_multi`, `/api/intents`, `/api/training/{review,apply}`,
and `/api/import/mcp/{search,fetch,apply}`. Full reference in the
[server API docs](https://gladius.github.io/microresolve/server/api/).

## Multi-intent, projection, and multilingual

**Native multi-intent** — a single query can confirm several intents in
one call, with the relation between them detected:

```text
"cancel my order and update my address"
  → cancel_order   (confirmed, high)
  → update_address (confirmed, medium)
  → relation: parallel
```

Detected relations: `sequential`, `conditional`, `negation`, `parallel`.

**Projected context** — co-occurrence tracking discovers what auxiliary
intents typically fire alongside the primary one, so your orchestrator
can pre-fetch them in parallel without an extra LLM round-trip:

```text
"I want a refund"
  → refund_order  (confirmed, high)
  → projected:    check_balance (21 %), warranty_lookup (13 %)
```

These relationships emerge from accumulated routing — they're not
configured.

**Multilingual** — Latin scripts via whitespace tokenization, CJK
(Chinese / Japanese / Korean) via Aho-Corasick automaton + character
bigrams, all in the same namespace. Per-language seed phrases per intent.
Multi-intent decomposition runs after tokenization, so a Chinese query
like "取消订阅然后退款" decomposes the same way an English query does.

## Import formats

- **MCP tools/list** (and Smithery registry passthrough)
- **OpenAPI / Swagger** — each endpoint becomes an intent
- **OpenAI function-calling schemas**
- **LangChain tools**

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

## Commercial support

Maintained by Gladius Thayalarajan — consulting on custom integrations,
multilingual tuning, and agent-stack tool routing: [gladius.thayalarajan@gmail.com](mailto:gladius.thayalarajan@gmail.com).
