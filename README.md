# ASV Router

Model-free intent routing. Sub-millisecond latency, no embeddings, no GPU.

Route natural language queries to intents in 30µs. Gets smarter from every production query — no retraining, no pipeline, no model.

## Why ASV?

| | ASV | Embedding models | LLM classification |
|---|---|---|---|
| Latency | **30µs** | 10–50ms | 200–2000ms |
| Cost at inference | **$0** | GPU / API | Per-token |
| Setup | Seed phrases | Training data + GPU | Prompt engineering |
| Learning | Real-time, incremental | Full retrain | N/A |
| Multi-intent | Native | Separate model | Prompt-dependent |
| Dependencies | None | PyTorch / ONNX | API key |

## Benchmarks

| Dataset | Intents | Seeds/intent | Seed-only | + Learning | Top-3 | Latency |
|---|---|---|---|---|---|---|
| CLINC150 | 150 | 50 | 84.0% | 85.8% | 94.2% | 22µs avg |
| CLINC150 | 150 | 100 | 87.5% | 96.4% | 95.9% | 24µs avg |
| BANKING77 | 77 | 50 | 81.9% | 85.0% | 94.1% | 21µs avg |
| BANKING77 | 77 | 130 | 85.5% | 92.8% | 96.0% | 23µs avg |

Zero ML training. Zero GPU. **+ Learning** shows accuracy after incremental learning from corrections — no retraining, just routing confirmations feeding back into the weights.

## Quick Start

```bash
# Copy and configure environment
cp .env.example .env
# Set ANTHROPIC_API_KEY (or any OpenAI-compatible endpoint) for LLM features

# Build and run the server
cargo run --release --bin server --features server -- --data ./data

# UI (separate terminal)
cd ui && npm install && npm run dev
# → http://localhost:3000
```

Or import your existing API spec to bootstrap intents instantly:

```bash
curl -X POST http://localhost:3001/api/import/spec \
  -H "Content-Type: application/json" \
  -d '{"spec": "<your OpenAPI JSON here>"}'
```

## How It Works — The Routing Pipeline

A query passes through five layers in sequence, all in-process, all in-memory:

```
raw query
  → L0  typo correction       (character n-gram Jaccard)
  → L1  vocabulary bridging   (Hebbian lexical graph)
  → L2  intent scoring        (IDF weighted term graph)
  → L3  confidence + disposition
  → L4  cross-provider tiebreak
  → result

failures → review queue → auto-learn worker → L1 + L2 updated
```

### L0 — Typo Correction

Character trigram Jaccard similarity against the known vocabulary. Tokens ≥4 chars that don't appear in the vocabulary are corrected to the nearest known term if similarity ≥0.5.

No training required. Language-agnostic. CJK passes through unchanged.

```
"cancl my subscrption" → "cancel my subscription"
```

### L1 — Hebbian Lexical Graph

A weighted term-association graph, seeded from LLM knowledge and reinforced from routing confirmations. Runs in two phases before scoring:

**Phase 1 — Normalize**: morphological variants and abbreviations are substituted in-place.
```
"canceling" → "cancel"
"pr"        → "pull request"
```

**Phase 2 — Expand**: synonyms above the threshold are appended to the query.
```
"terminate my plan" → "terminate my plan cancel"
```

Edge types by weight:

| Weight | Kind | Effect |
|---|---|---|
| 0.97–1.0 | Morphological / Abbreviation | Substitute in place |
| 0.80–0.96 | Synonym | Append canonical term |
| 0.60–0.79 | Semantic | Confidence boost only |

Result: "I want to terminate my plan" and "cancel my subscription" arrive at L2 looking the same.

### L2 — Intent Graph (IDF Scoring)

The core scorer. Each intent has a learned weighted vector of terms. Query tokens are scored against all intent vectors simultaneously.

**Two passes**:
1. **Raw pass** — single-pass IDF scores for top-N ranking
2. **Token-consumption pass** — iterative: confirm top intent, remove its tokens, re-score remaining. This is what enables multi-intent detection.

```
"cancel my order and update my address"
  → pass 1: cancel_order (0.82), update_address (0.61)
  → pass 2: remove cancel_order tokens → update_address confirmed
  → result: [cancel_order, update_address]
```

Every confirmed routing reinforces term weights. Every correction shifts weights away from the wrong intent.

### L3 — Disposition & Confidence

Interprets the score distribution:

| Disposition | Condition |
|---|---|
| `confident` | Clear winner |
| `low_confidence` | Top score barely above threshold → flagged for review |
| `escalate` | 3+ intents at similar scores → genuinely ambiguous |

Per-intent confidence: `high` (≥80% of top score), `medium` (≥50%), `low` (<50%).

Low-confidence and missed queries are queued for review or auto-learn.

### L4 — Cross-Provider Disambiguation

When the same action exists across multiple providers (e.g., `stripe:list_customers` and `shopify:list_customers` both score high), query-word exclusivity breaks the tie — provider-specific terms push the correct one up.

### Auto-Learn Worker

A background worker woken by every flagged query. In auto mode: calls the LLM to analyse the failure, determines correct/wrong intents, adds phrases, and reinforces L1+L2. In manual mode: queues for human review.

This closes the loop — the router gets better from every production failure.

## Multi-Intent

ASV natively detects when a single query contains multiple intents:

```
"cancel my order and track the other package"
→ cancel_order  (confirmed, high)
→ track_order   (confirmed, medium)
→ relation: parallel
```

Detected relations: `sequential`, `conditional`, `negation`, `parallel`.

## Projected Context

ASV tracks which intents fire together and uses co-occurrence patterns to predict what auxiliary data your workflow will need — even when the user doesn't ask for it explicitly.

```
Query: "I want a refund"
→ refund_order    (confirmed, high)
→ projected:      check_balance (21%), warranty_lookup (13%)
```

The user only asked for a refund. But from past routing patterns, ASV knows refund workflows typically also need balance and warranty data. Your orchestrator can pre-fetch both in parallel — eliminating LLM round-trips.

These relationships are not configured. They emerge from accumulated co-occurrence across real queries. ASV discovers your domain's dependency graph automatically.

## Import Formats

Bootstrap intents from your existing specs — no manual phrase writing:

- **OpenAPI / Swagger** — each endpoint becomes an intent
- **MCP tools** — each tool becomes an intent
- **OpenAI function calling** — direct import
- **LangChain tools** — direct import

```bash
# UI import flow: /import → paste spec → review → apply
```

## HTTP API

Send `X-Namespace-ID: my-namespace` to isolate intents per namespace. No header defaults to `default`.

```
POST /api/route_multi          — route a query (multi-intent)
GET  /api/intents              — list intents
POST /api/intents              — add intent
POST /api/intents/phrase       — add training phrase
POST /api/learn                — teach the router
POST /api/learn/now            — synchronous learn (bypasses queue)
POST /api/correct              — fix a misroute
POST /api/report               — flag a query for review
GET  /api/review/queue         — get flagged queue
POST /api/review/analyze       — LLM analysis of a flagged item
POST /api/review/fix           — apply phrases from analysis
GET  /api/review/stats         — pending count
POST /api/import/spec          — import OpenAPI / MCP spec
GET  /api/namespaces           — list namespaces
POST /api/namespaces           — create namespace
POST /api/simulate/turn        — generate a simulated query
POST /api/training/run         — baseline accuracy test
GET  /api/export               — export state
POST /api/import               — import state
GET  /api/events               — SSE stream (live routing events)
```

## Why Server-Side

ASV is designed for server-side deployment. A WASM build exists for demos, but production routing belongs on the server:

- **Seed phrases are your business logic.** Shipping them to the browser exposes your entire intent taxonomy — competitors can inspect it, attackers can game it.
- **Learned weights contain user patterns.** If the router has learned from corrections, those weights reflect real user behaviour. They belong on your server.
- **The hybrid pattern needs a decision point.** Route the easy 80% with ASV ($0, 30µs), send the hard 20% to your LLM. This logic lives on the server where you control the fallback.
- **Namespaces.** The server supports multiple isolated namespaces via `X-Namespace-ID`, each with independent intents and persistence.

WASM is appropriate for: public demos, open-source intent sets, offline tools, and edge devices with no server available.

## Architecture

Five in-memory layers, all in the same process, no network hops:

- **L0** `src/ngram.rs` — character trigram typo correction
- **L1** `src/scoring.rs` (`LexicalGraph`) — Hebbian term normalization + synonym expansion
- **L2** `src/scoring.rs` (`IntentGraph`) — IDF-weighted term scoring + token consumption
- **L3** `src/bin/server/routes_core.rs` — disposition and confidence classification
- **L4** `src/bin/server/routes_core.rs` — cross-provider disambiguation
- **Worker** `src/bin/server/worker.rs` — background auto-learn loop

Intent definitions and training phrases live in `src/lib.rs` (`Router`). The `Router` struct is the training-data store that feeds L1+L2 bootstrap.

## CJK Support

Native Chinese, Japanese, and Korean via Aho-Corasick automaton + character bigrams. Multilingual intents with per-language seeds supported.

## License

MIT OR Apache-2.0
