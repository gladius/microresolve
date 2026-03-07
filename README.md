# ASV Router

Model-free intent routing. Sub-millisecond latency, no embeddings, no GPU.

ASV builds sparse vectors from seed phrases and routes queries using BM25-inspired scoring. It decomposes multi-intent queries, detects relationships between intents, learns incrementally from corrections, and discovers domain structure through co-occurrence patterns.

## Why ASV?

| | ASV | Embedding models | LLM classification |
|---|---|---|---|
| Latency | **12-25μs** | 10-50ms | 200-2000ms |
| Setup | Seed phrases | Training data + GPU | Prompt engineering |
| Cold start | Instant | Hours | Instant but expensive |
| Learning | Incremental, real-time | Full retrain | N/A |
| Multi-intent | Native decomposition | Separate model | Prompt-dependent |
| Dependencies | None | PyTorch/ONNX | API key |

## Benchmarks

Tested on standard intent classification datasets:

| Dataset | Intents | Seed-only (50/intent) | + Learning | Top-3 | Latency |
|---|---|---|---|---|---|
| CLINC150 | 150 | 83.9% | 93.6% | 95.9% | 22μs avg |
| BANKING77 | 77 | 81.9% | 92.8% | 95.9% | 21μs avg |

Zero ML training. Zero GPU. Just seed phrases and incremental learning.

## Quick Start

```rust
use asv_router::Router;

let mut router = Router::new();

// Define intents with seed phrases
router.add_intent("cancel_order", &[
    "cancel my order",
    "I want to cancel my purchase",
    "stop my order from shipping",
]);
router.add_intent("track_order", &[
    "where is my package",
    "track my order",
    "shipping status update",
]);

// Route a query
let results = router.route("I need to cancel something");
assert_eq!(results[0].id, "cancel_order");

// Learn from corrections
router.learn("stop charging me", "cancel_order");
router.correct("cancel my subscription", "cancel_order", "cancel_subscription");
```

## Multi-Intent Decomposition

ASV natively handles queries with multiple intents:

```rust
let output = router.route_multi("cancel my order and track the package", 0.3);

for intent in &output.intents {
    println!("{}: {:.2} ({:?})", intent.id, intent.score, intent.intent_type);
    // cancel_order: 5.46 (Action)
    // track_order: 3.40 (Context)
}

// Relationship detection
for relation in &output.relations {
    println!("{:?}", relation);
    // Parallel — both intents should be handled
}
```

Detected relationships: `Parallel`, `Sequential`, `Conditional`, `Negation`, `Reverse`.

## Projected Context

ASV tracks which intents fire together and uses co-occurrence patterns to predict what auxiliary data your workflow will need — even when the user doesn't ask for it.

```
Query: "I want a refund"
→ refund (score: 4.77, action)
→ projected_context: check_balance (21%), warranty_info (13%)
```

The user only asked for a refund, but ASV knows from past patterns that refund workflows typically need balance and warranty data. Your orchestrator can pre-fetch both in parallel — cutting LLM round-trips.

These relationships are not configured. They emerge from the mathematical structure of sparse vector routing + natural language patterns. ASV discovers your domain's dependency graph automatically.

## Intent Types & Metadata

```rust
use asv_router::IntentType;

// Classify intents as Action (user wants this done) or Context (supporting data)
router.set_intent_type("refund", IntentType::Action);
router.set_intent_type("check_balance", IntentType::Context);

// Attach opaque metadata — ASV stores it, never interprets it
router.set_metadata("refund", "handler", vec!["billing_service".into()]);
router.set_metadata("refund", "requires_auth", vec!["true".into()]);
```

## CJK Support

Native Chinese, Japanese, and Korean routing via Aho-Corasick automaton + character bigrams:

```rust
router.add_intent("cancel", &["注文をキャンセルしたい", "订单取消"]);
router.add_intent("track", &["荷物の追跡", "包裹在哪里"]);

let result = router.route("注文をキャンセルしてください");
assert_eq!(result[0].id, "cancel");
```

Multilingual intents with per-language seeds:

```rust
use std::collections::HashMap;

let seeds = HashMap::from([
    ("en".into(), vec!["cancel my order".into()]),
    ("ja".into(), vec!["注文をキャンセルしたい".into()]),
    ("zh".into(), vec!["取消订单".into()]),
]);
router.add_intent_multilingual("cancel_order", seeds);
```

## State Persistence

```rust
// Export to JSON
let json = router.export_json();

// Import from JSON
let router = Router::import_json(&json).unwrap();
```

## HTTP Server

```bash
# Set API key for LLM features (optional)
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env

# Run server
cargo run --release --bin server --features server
```

Endpoints:
- `POST /api/route_multi` — route with multi-intent decomposition + projected context
- `POST /api/route` — single best-match routing
- `GET /api/intents` — list all intents with seeds, types, metadata
- `POST /api/intents` — add intent
- `POST /api/learn` — teach the router
- `POST /api/correct` — fix misroutes
- `POST /api/review` — LLM-powered routing review (requires API key)
- `GET /api/co_occurrence` — intent co-occurrence matrix
- `GET /api/export` / `POST /api/import` — state persistence

## Interactive UI

```bash
cd ui && npm install && npm run dev
```

Web dashboard at `http://localhost:5173` with:
- Live routing with query highlighting and intent span visualization
- Intent management with type toggles and metadata editing
- Learn mode with LLM-reviewed routing and one-click corrections
- Co-occurrence matrix and query log viewer
- AI-powered seed phrase generation

## WASM

```bash
wasm-pack build --target web --out-dir web/pkg
```

## Architecture

ASV uses an inverted index of weighted terms (inspired by BM25) rather than dense embeddings. Each intent is a sparse vector built from tokenized seed phrases. Routing is a dot product against the query's term vector — O(unique terms in query × avg postings per term).

Key properties:
- **No model**: Pure algorithmic scoring, no neural network weights
- **Incremental learning**: `learn()` and `correct()` update term weights without retraining
- **Multi-intent**: Sliding window decomposition with positional scoring
- **Negation-aware**: Detects "not", "don't", negation markers in 12 languages
- **Emergent structure**: Co-occurrence tracking reveals domain relationships from usage

## License

MIT OR Apache-2.0
