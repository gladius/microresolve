# Show HN: MicroResolve — embedded lexical layer for AI agents (intent routing + PII + safety, microseconds, no LLM at runtime)

A tiny embeddable engine that handles four jobs your AI stack needs but probably has wired up as four separate libraries today:

- **Intent routing** — pick the right tool/function from natural language
- **PII detection / extraction / masking** — find SSNs, cards, emails, custom IDs in user input
- **Prompt-injection / jailbreak first-pass filter** — catch the obvious attack patterns
- **Custom entity detection** — define new entity types in plain English; the system distills patterns

All on the same engine. Microseconds per call. No LLM at runtime. Embeds as a Rust crate, Python package, Node module, or WASM bundle. No service to deploy, no Python runtime requirement when called from Rust/Node.

---

## What this is, plainly

It's an inverted index plus a small morphology graph plus an optional regex+Aho-Corasick entity layer, all in ~10K lines of Rust. The LLM teaches it at training/setup time (generates seed phrases, distills custom entity patterns); at query time, only the lexical engine runs. That's the whole pitch.

The thing this replaces in your stack is the *cheap end* of:
- LLM-based intent classification (the "which tool" call)
- LLM-based or ML-based safety filtering on every request
- Cloud-API PII detection
- Hand-coded regex collections

Not the *whole thing*. For the long tail (semantic edge cases, novel attack framings, named-entity recognition for PERSON/ORG/LOCATION), pair MicroResolve with a small fine-tuned model or LLM verification on the borderline cases. The point is that 80–95% of requests get handled at $0/microsecond, leaving the expensive layer with a much smaller bill.

## Numbers from real benchmarks

All run against published, third-party datasets. Honest, including where we lose.

### Intent routing — MCP tools (50 GitHub + Slack tools from Smithery)

- 5 LLM-generated seed phrases per tool, no hand tuning
- 40-query test set with realistic phrasings
- Median latency: **~60 µs per route**
- Accuracy: **100%** (40/40)
- Cost per route: **$0**

For comparison, LLM function-calling on the same setup runs at 200–500 ms and ~$0.0005 per call. Same accuracy, ~5000× faster, free.

### PII detection — Microsoft Presidio's `synth_dataset_v2.json` (1500 examples)

Patterns enabled: 30+ built-in (PII, credentials, identifiers, financial, web/tech).

| Entity type | Precision | Recall | F1 |
|-------------|-----------|--------|------|
| **SSN** | 1.00 | 1.00 | **1.00** |
| **IBAN** | 1.00 | 1.00 | **1.00** |
| **US Driver's License** | 1.00 | 1.00 | **1.00** |
| EMAIL | 0.88 | 1.00 | 0.93 |
| Credit card | 0.69 | 0.96 | 0.80 |
| URL | 0.56 | 1.00 | 0.72 |
| Phone | 1.00 | 0.55 | 0.71 |
| ZIP code | 1.00 | 0.49 | 0.65 |
| IPv4 | 0.52 | 0.93 | 0.67 |
| Date of birth | 0.80 | 0.40 | 0.54 |
| **Overall** | **0.76** | **0.75** | **0.75** |

Latency: 3 µs median, 9 µs mean across 1500 examples. Total runtime to process the entire 1500-example dataset: **14 ms**.

Coverage gap: Presidio's dataset includes named entities (PERSON 857, STREET_ADDRESS 598, ORGANIZATION 250, GPE 411) that need NER models — out of scope for a pattern-based detector. Pair with spaCy or similar for those.

### Prompt injection — `deepset/prompt-injections` test set (116 examples)

Out-of-the-box `security` namespace, no training on this dataset:

| Metric | Value |
|--------|-------|
| Accuracy | 64.7% |
| Precision | **85.2%** |
| Recall | 38.3% |
| F1 | 0.53 |
| Latency | ~25 µs median, p99 ~250 µs (full pipeline, with entity layer ON) |

Honest read: a fine-tuned classifier (deepset's own DeBERTa baseline at 99.1% accuracy) beats us substantially on accuracy. **We are higher precision but lower recall** — when we flag, we're usually right; we miss a lot.

This is the right outcome for this architecture. Use MicroResolve as a **high-precision first-pass filter**: anything it catches, block or flag with high confidence. Anything it misses, fall through to a fine-tuned model. The recall gap is the price of paying $0/ms instead of $0.0005/200ms.

The recommended deployment is **tag-don't-block**:
- Score above the high threshold → block (we're confident)
- Score in the middle band → tag with metadata, inject into next LLM turn for verification
- Score below the low threshold → forward normally

This makes the modest recall acceptable: the LLM is the safety net for the ~60% of attacks we miss, but it only gets called on the borderline cases.

### Custom entity detection — your own formats

Describe an entity in plain English:

```
Entity name: hospital_patient_id
Description: Hospital patient identifier — 7 to 10 digits,
             prefixed with PT- or PID-.
Examples (optional): ["PT-1234567", "PID-9876543"]
```

One LLM call (~3 seconds, ~$0.0001), auto-validated, returned for review. Save it; the entity is live in the namespace immediately. Detect, extract, and mask all work for your custom entities the same way they work for built-ins.

We tested 5 custom entity types (passport, AWS key, IBAN, Medicare ID, JWT). 4/5 produced usable patterns straight from the LLM (the 5th had a description-vs-examples contradiction). Auto-repair drops bad regexes before they reach production.

## Architecture

```
query
  │
  ▼
L0  typo correction       (character n-gram Jaccard, ~2µs)
  │
  ▼
entity layer (optional)   (regex + Aho-Corasick, ~3µs)
  ├─ detect / extract / mask
  └─ augment query with mr_pii_<label> tokens for downstream routing
  │
  ▼
L1  morphology graph      ("running" → "run", "subscriptions" → "subscription")
  │
  ▼
L2  intent index          (IDF-weighted, multi-intent, multi-round)
  │
  ▼
result: ranked intents, confirmed set, dispositions

failures → review queue → continuous learning → L1 + L2 + entity layer updated
```

Same engine for all four jobs. Same continuous-learning loop. Same threshold cascade (per-request → per-namespace → server default).

## Design principles (from the README)

1. **Microsecond at runtime.** Every routing call resolves in single- to double-digit microseconds. No exceptions.
2. **No LLM at inference.** The LLM teaches at setup time; at query time only the lexical engine runs. Inference is $0 per call and works offline.
3. **Embedded library, not a binary or sidecar.** Linked into your app as a Rust crate, Python/Node binding, or WASM bundle. No external service.
4. **One library, many jobs.** Same engine handles routing, safety, PII, masking. Every new feature reuses existing primitives.
5. **Continuous learning from real traffic.** Static rules and frozen models go stale. Patterns reinforced in production, corrections feed back in place. No retraining pipeline.

## What's in the box

- Rust core: `cargo add microresolve`
- HTTP server with web UI: `cargo run --release --bin server --features server`
- Python bindings: `pip install microresolve` (planned alongside launch)
- Node bindings: `npm install microresolve` (planned alongside launch)
- WASM bundle for browser/edge: in the `wasm` crate
- 30+ built-in entity patterns: PII, credentials (AWS/GCP/Stripe/GitHub/OpenAI/Anthropic/JWT), identifiers (US passport, ZIP, NHS, Aadhaar, PAN), financial (IBAN, Bitcoin, Ethereum), web/tech (IPv4, IPv6, MAC, URL, UUID, MD5, SHA-256)
- LLM distillation pipeline for custom entities
- Per-namespace config UI: select which patterns are active, define custom entities

## Honest scope — what this is NOT

- Not a full ML-based prompt-injection detector. We catch obvious patterns; pair with a fine-tuned model for the long tail.
- Not a NER engine. We detect entities with stable formats (regex-able). For PERSON/ORG/LOCATION, pair with spaCy or similar.
- Not a content-moderation classifier. JailbreakBench-style harmful-content requests require a different layer.
- Not a chat model. We pick intents and detect entities; we don't generate responses.

## Try it

```bash
git clone https://github.com/[org]/microresolve
cd microresolve
cp .env.example .env  # add ANTHROPIC_API_KEY for setup-time LLM features
cargo run --release --bin server --features server -- --data ./data

# Separate terminal:
cd ui && npm install && npm run dev
# → http://localhost:3000
```

Or import an OpenAPI spec to bootstrap intents in one call:

```bash
curl -X POST http://localhost:3001/api/import/spec \
  -H "Content-Type: application/json" \
  -d @your-openapi.json
```

The mcp-demo workspace (50 GitHub + Slack tools, 100% routing accuracy at 60µs) ships pre-loaded so you can see the routing demo immediately.

## Repo / docs / blog posts

- Repo: [link]
- Docs: [link]
- MCP tool routing benchmark write-up
- Lexical jailbreak detection deep dive (the tag-don't-block architecture)
- Entity layer design + LLM distillation results

Discussion welcome. Specifically: where this fits in your stack, what entity types you'd want pre-built, what we got wrong about the recall vs precision tradeoff.
