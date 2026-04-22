# MicroResolve — Blog Posts to Write

## Launch / Positioning Blogs

### 1. "Your LLM Agent Burns 72% of Tokens Before Doing Anything — Here's the Fix"
- Hook: Perplexity dropped MCP because tool descriptions eat context windows
- Show: 50 tools = 100K+ tokens per call, before a single user message
- Solution: MicroResolve pre-selects 2-3 relevant tools at 30µs, 90% token reduction
- Include benchmark: tokens with/without MicroResolve pre-filtering
- End with: pip install, 5 lines of code

### 2. "We Tested Intent Routing at 30 Microseconds vs 500ms — Here's What Changed"
- Direct latency comparison: MicroResolve vs embedding-based vs LLM-based routing
- Real numbers from Bitext 27K benchmark
- Show the cost math at scale: 10K, 50K, 100K queries/day
- The hybrid architecture diagram (Layer 0 → Layer 1 → Layer 2 → LLM)

### 3. "Zero Models, Zero Embeddings, Zero API Keys — Intent Routing That Runs Anywhere"
- The pitch for edge/offline/privacy use cases
- WASM demo in browser
- Raspberry Pi benchmark
- Compare: what you need to deploy MicroResolve vs Aurelio semantic-router vs vLLM-SR

### 4. "The Only Intent Router That Gets Smarter Every Day (Without Retraining)"
- Online learning: learn() and correct() adjust routing from individual corrections
- Show the learning curve: accuracy over time with corrections
- No model retraining, no embedding recomputation, no downtime
- Compare to static systems that need redeployment for every change

### 5. "Predicting What Your Customer Wants Next — Before They Ask"
- Co-occurrence tracking: "cancel_order + refund appear together 85% of the time"
- Temporal flow: "track_order → shipping_complaint → contact_human is a churn path"
- Workflow projections: anticipate the next intent, pre-load the right tools
- No competitor has this. Unique to MicroResolve.

## Technical Deep Dives

### 6. "Multi-Intent Decomposition: When Users Say Three Things in One Message"
- "Cancel my order but don't refund, and let me talk to someone"
- How MicroResolve decomposes this into 3 intents with relations (sequential, conditional, negation)
- Negation-aware routing: "don't refund" preserves refund intent as negated
- Benchmark results on MixSNIPS and MixATIS datasets

### 7. "CJK Intent Routing Without Word Boundaries"
- The dual-path tokenization architecture
- Aho-Corasick automaton for known terms, bigram fallback for novel terms
- CJK negation detection (Chinese 不/没/别/未, Japanese ない/しない)
- Why this matters: $2B+ CJK chatbot market, no model-free CJK router exists

### 8. "CRDT Merge for Distributed Intent Learning"
- max() merge on learned weights is a natural CRDT
- Multiple library instances learn independently, merge without conflicts
- How this enables edge deployment + central aggregation
- Comparison to embedding systems that can't merge learned state

### 9. "How We Detect Prompt Injection at 30 Microseconds"
- Jailbreak/prompt injection as just another intent
- Pattern-based detection + online learning from new attack patterns
- Runs before the query reaches the LLM — zero additional latency
- Compare to ML-based detection (10-50ms) and regex-only (brittle)

### 10. "Building a Semantic Cache Without Embeddings"
- MicroResolve intent + entity pattern = deterministic cache key at 30µs
- Skip the embedding computation for cache lookup entirely
- Cache hit rates vs embedding-based semantic caches
- When to use MicroResolve cache keys vs when you need full embedding similarity

## Use Case / Industry Blogs

### 11. "Reducing Contact Center AI Costs by 80% with Hybrid Routing"
- The $80B Gartner opportunity
- Architecture: MicroResolve handles 80% of IVR routing, LLM handles 20%
- Self-hosted for regulated industries (banking, healthcare, government)
- Case study format with cost math

### 12. "Why Your AI Agent Calls the Wrong Tool (and How to Fix It)"
- Tool call hallucination: $14,200/employee/year in verification costs
- MicroResolve as a validator: check if LLM's tool call matches detected intent
- Pre-filtering: only show relevant tools to reduce hallucination surface
- Real examples from GPT-5 router backlash

### 13. "Intent Routing for n8n, Zapier, and Workflow Automation"
- Event-driven classification at 30µs
- One event → multiple automations (multi-intent)
- Learning from user corrections when automations misfire
- n8n's own "Production AI Playbook" recommends deterministic steps

### 14. "The Developer's Guide to Not Getting a $47K API Bill"
- Real enterprise cost horror stories (Pluralsight, DoorDash, anonymous fintech)
- The 4-layer hybrid architecture
- Where MicroResolve fits: Layer 0
- ROI calculator: queries/day × cost/query × MicroResolve hit rate = monthly savings

## Comparison / Alternative Blogs

### 15. "MicroResolve vs Aurelio Semantic Router vs vLLM Semantic Router"
- Honest feature comparison table
- When to use which: MicroResolve (intent routing, learning, edge), Aurelio (embedding similarity, Python), vLLM-SR (model routing, infrastructure)
- They're different layers, not direct competitors

### 16. "MicroResolve vs Dialogflow vs Rasa in 2026"
- Dialogflow: cloud lock-in, no learning, no multi-intent
- Rasa: heavy, moving to CALM (LLM-augmented), open-source version in maintenance
- MicroResolve: lightweight, self-hosted, learns, multi-intent, runs in WASM
- Migration guide from Dialogflow/Rasa to MicroResolve

## Community / Ecosystem Blogs

### 17. "Using MicroResolve with LangChain / LlamaIndex / CrewAI"
- Integration code examples for each framework
- Before/after: tool descriptions sent, tokens consumed, latency
- The 3-line integration pattern

### 18. "Building an MCP Gateway with MicroResolve"
- MCP proxy that filters tool descriptions before they hit the LLM
- Transparent to the rest of the stack
- Solves the 72% context window waste problem
