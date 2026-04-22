# MicroResolve Use-Case Patterns

A catalog of the patterns MicroResolve fits, with concrete blog ideas inside each. The patterns are organized by what the namespace *is doing*, not what industry it serves — because the same pattern shows up across industries.

Each idea is a candidate blog post. They share a structure: setup, namespace design, cold benchmark, learning loop, threshold sweep, production architecture, cost comparison.

---

## Pattern A — Tool routing for AI agents

**Shape:** many intents, each with distinct vocabulary, low threshold (0.1–0.5).

The "drop in for production" use case. Replaces LLM-based function-calling with a microsecond router that gets cheaper and more accurate as you use it.

### Blog ideas

- **MCP tool routing at scale** — 50 GitHub + Slack MCP tools, 100% accuracy on 40-query benchmark, ~60µs per route, $0/call. *(Demo already run, just needs writing up.)*
- **Function calling without the LLM tax** — show how MicroResolve in front of an OpenAI-compatible function-calling endpoint saves 90%+ of token cost on tool selection.
- **Plugin / extension routing for desktop apps** — VS Code, browser extensions: route natural-language commands to plugin actions.
- **Multi-agent task delegation** — in agent-of-agents systems, route incoming tasks to the right specialist agent without an LLM dispatcher.

---

## Pattern B — Safety / classification first-pass

**Shape:** few intents (3–10), high vocabulary overlap with input, high threshold (1.0–2.0), tag-don't-block architecture.

The lexical pre-filter that makes downstream LLM verification affordable.

### Blog ideas

- **Lexical jailbreak detection** — already in `blogs/jailbreak-detection.md`.
- **PII intent detection** *(complementary to regex/NER, not a replacement)* — detects intent to *request, share, exfiltrate* sensitive data. "send me the API key", "what's the customer's SSN", "export all user emails". Pairs with Presidio/spaCy for entity extraction.
- **Content moderation intent** — hate speech, harassment, NSFW request intent. Routes to deeper LLM check on borderline cases.
- **Compliance triggers** — flags mentions of regulated topics: medical advice request, legal advice request, financial advice request. Useful in chatbots that must hand off to licensed professionals.
- **Brand safety / topic guardrails** — route customer questions about competitors or controversial topics to human review.

---

## Pattern C — Conversational routing

**Shape:** medium intent count (10–50), mixed vocabulary specificity, threshold 0.3–1.0.

Chatbot first-response routing. Currently dominated by LLM classifiers costing $0.001/msg.

### Blog ideas

- **Customer support tier-1 routing** — billing / technical / account / sales / cancellation. Real enterprise market, latency-sensitive. Shows the cost story most clearly.
- **FAQ deflection** — route to canned answer or human in <100µs. Reduces LLM call rate to 10–20%.
- **Onboarding flow stage detection** — "where in the multi-step form is the user stuck?" Routes to the right help content.
- **Slot-filling for forms** — in conversational form filling, route each user utterance to the field it answers.

---

## Pattern D — Workflow / triage

**Shape:** classification with high precision required, high stakes.

Email-shaped problems. Universal pain point.

### Blog ideas

- **Email triage** — inbox routing to teams (sales / support / hr / billing). Training data already exists in your sent folder.
- **Incident classification** — sev-1 vs sev-3, which team, which runbook. High stakes — exactly where lexical + LLM verify shines.
- **Bug report triage** — severity, component, owner. Github issues / Jira / Linear use case.
- **Code review categorization** — security / performance / style / correctness. Routes PRs to right reviewer skill.

---

## Pattern E — Model / route selection ⭐

**Shape:** a few "where does this go" intents, lots of training data possible, direct $$ ROI.

The most exciting unexplored pattern. Plugs directly into the existing AI ecosystem.

### Blog ideas

- **LLM router** ⭐ — route queries to the cheapest capable model. "this needs Opus / this Haiku / this local Llama." Replaces hand-coded if/else with a learned classifier. Direct measurable ROI: typical agent stack saves 60–80% of LLM cost when 90% of queries can go to a cheaper model.
- **RAG document / index routing** — "which document collection should this query hit?" Avoids embedding-search-everything. Cuts vector DB cost.
- **Multi-tenant query routing** — in a SaaS product, route incoming queries to the right tenant-specific knowledge base.
- **Skill-based help desk routing** — same idea applied to humans instead of models.

---

## Pattern F — Embedded / offline assistants

**Shape:** runs on-device. No cloud. No LLM. Privacy-first.

Showcases the WASM target. Different deployment model from everything else.

### Blog ideas

- **Voice command routing on-device** — smart home, IoT, car commands. Runs in the browser via WASM, or natively via the Rust crate. No cloud round-trip.
- **IDE command routing** — VS Code extension that maps natural language to commands without sending text to a server. Privacy-preserving developer assistant.
- **Offline assistant for regulated industries** — medical, legal, defense — places where you can't send queries to OpenAI.

---

## Pattern G — Continuous-learning observation

**Shape:** the system learns *from* live traffic, not just routes it.

The reinforcement-loop story. Less about a single use case, more about how MicroResolve gets better over time.

### Blog ideas

- **Conversation intent drift** — over weeks, watch the distribution of resolved intents shift. Surface emerging topics before they become support tickets.
- **Latent topic clustering** — when the router can't confidently route, what are users asking? Auto-cluster the misses into proposed new intents.
- **A/B testing with intent routing** — use intent distribution as the metric.

---

## Cross-cutting blog ideas (not tied to one pattern)

- **The threshold is per-namespace** — narrative version of `docs/threshold-tuning.mdx`.
- **Why we don't use embeddings** — design defense. Lexical scoring + Hebbian learning vs vector search.
- **Continuous learning vs cold-start benchmarks** — most published numbers are cold-start. We measure improvement over time.
- **Multilingual without separate models** — one router, many languages, via L1 morphology base + CJK Aho-Corasick.
- **What MicroResolve is not** — honest limits doc. Not a chat model, not a regex engine, not an embedding store.
