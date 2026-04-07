# ASV Router — Product Strategy (April 2026)

> Strategic positioning, distribution decisions, and market analysis.

---

## 1. Positioning — What Changed

### The old pitch (2024)
"LLMs are too slow and expensive for routing. Use ASV instead."

### The new pitch (2026)
"Your AI agents make 10-50 tool routing decisions per conversation. At $0.01 and 200ms each, that's $0.50 and 2-10 seconds of routing overhead per session. ASV handles 80% of those decisions at 30us and $0."

### Why the shift
- LLMs are ~5x cheaper and ~2x faster than 2024
- Structured output / tool use is standard
- Edge models (Phi-3, Gemma) run on phones
- Every company is building AI agents
- Cost optimization is #1 enterprise AI concern

### What ASV is NOT
- Not an LLM replacement
- Not competing on single-intent accuracy
- Not trying to understand semantics

### What ASV IS
- The routing layer for AI agent pipelines
- The 80/20 split: handle the easy 80% at $0, send the hard 20% to LLM
- A system that gets cheaper over time (graduation model), not more expensive
- Complementary to LLMs, not competitive

---

## 2. Value Proposition by Audience

### Enterprise (CFO/CTO pitch)
"You're spending $X/month on LLM routing. This makes 80% of it free. ROI in the first week."

### AI Agent Developers (IC pitch)
"pip install asv-router. 5 lines to add intent routing. Learns while it runs. No API key. No model. Works offline."

### Edge/IoT Teams
"30us, 8MB RAM, runs on Raspberry Pi. No internet required. Compiles to WASM for browser."

### Privacy/Compliance Teams
"All processing local. No query data leaves the device. CRDT sync sends only weight deltas, never raw user messages."

---

## 3. What's Uniquely Valuable in 2026

| Capability | Why it matters now |
|---|---|
| CRDT distributed learning | No other routing system has conflict-free multi-instance learning |
| Negation-aware multi-intent | "Do X but not Y" — LLMs do this but not at 30us |
| Online learning without retraining | Still rare even in 2026 |
| Graduation economics | Start with LLM verification, converge to $0 marginal cost |
| Sub-millisecond at zero cost | LLMs are cheaper but not $0 and not 30us |

---

## 4. What's NOT Valuable Anymore

- Competing on single-intent accuracy against LLMs (LLMs win)
- Positioning as "LLM alternative" (wrong framing — it's complementary)
- Cold-start advantage (LLMs can bootstrap faster now, but ASV's teacher model handles this)
- Pure speed argument alone (need the cost + learning story too)

---

## 5. Distribution Priority

| Priority | Target | Why first |
|---|---|---|
| 1 | Server (polish + Docker) | Enterprise, any-language, instant testing, central learning hub |
| 2 | Python (PyO3) | 80% of AI developers, LangChain integration |
| 3 | npm WASM | Browser demo, web apps |
| 4 | Node.js (napi-rs) | Server-side JS |
| 5 | CLI | Developer tooling |

---

## 6. Sync Philosophy

**Don't build sync infrastructure. Provide sync primitives.**

Enterprise customers have Kafka, Redis, S3, PostgreSQL. They don't want ASV's custom sync protocol. They want ASV's state to be "just data" — a JSON blob they can store, version, and distribute through their existing systems.

What ASV provides:
- `export_json()` / `import_json()` — full state as portable JSON
- `export_learned_only()` / `import_learned_merge()` — lightweight deltas for incremental sync
- `version()` — change detection
- Server exposes these as HTTP endpoints

What ASV does NOT provide:
- Custom sync protocol
- WebSocket streaming
- Event replay
- Sync scheduling
- Transport layer

Transport is the user's problem. ASV reads and writes JSON. Everything else is infrastructure.

---

## 7. Launch Plan

1. **arXiv paper + GitHub** — establish priority, open source (week 1)
2. **PyPI package** — `pip install asv-router` gateway (week 2)
3. **Hacker News launch** — lead with "130K routes/sec, no GPU, $0" (week 2)
4. **LangChain/LlamaIndex integration** — drop-in ASVToolRouter (week 3)
5. **Blog post** — "We replaced $4,500/month GPT-4 routing with a HashMap" (week 3)
6. **Docker image** — `docker run asv-router` for enterprise eval (week 2)

---

*This document captures strategic decisions as of April 2026.*
