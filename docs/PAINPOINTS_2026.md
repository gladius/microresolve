# Industry Pain Points in 2026 — ASV Router Solutions Map

## How to read this document

Each pain point includes:
- **The problem** — what's actually happening
- **Who said it** — real quotes from developers, executives, forums
- **Can ASV fix it?** — Yes / Partially / No
- **How** — specific ASV feature or proposed feature
- **Gap** — what's still missing and what we'd need to build

---

## Category 1: LLM Cost at Scale

### Pain 1.1: Surprise API Bills
**Problem:** Teams get $47K bills with no breakdown. Pilot costs ($500/month) explode to $15K/month in production. 68% of enterprise teams underestimate first-year spend by 3x.

**Who said it:**
- Pluralsight case study: "$47,000 API bill with a single line item: 'OpenAI API'"
- SoftwareSeni: "Pilot-to-production cost shock is a common theme"
- Neil Dave: "68% of enterprise teams underestimate first-year LLM API spend by more than 3x"

**Can ASV fix?** Yes — partially.
**How:** ASV handles the 80% of queries that are repetitive/obvious at $0. Only 20% goes to LLM. Monthly cost drops proportionally to ASV hit rate. Dashboard shows: "This month ASV handled 73% of queries, saving $X."
**Gap:** ASV doesn't track actual dollar costs per query. Need: cost estimation per intent (based on token count of tool descriptions sent to LLM). Feature to build: P1 cost tracking dashboard.

---

### Pain 1.2: Agent Token Waste
**Problem:** Agents make 3-10x more LLM calls than chatbots. Unconstrained agents cost $5-8 per task. 100K daily conversations = $2K-10K/day.

**Who said it:**
- Zylos Research: "Agents make 3-10x more LLM calls than simple chatbots"
- HyperSense: "$3,200-$13,000/month in operational spend per agent"
- LinkedIn: "$5,000 in costs within a single afternoon from autonomous agents in an endless loop"

**Can ASV fix?** Yes.
**How:** Tool pre-filter. Instead of 50 tools in system prompt (3-5K tokens), ASV selects 2-3 relevant ones at 30µs. 90% fewer input tokens per call. Deterministic — agents can't loop because ASV gives the same answer each time.
**Gap:** Need the Agent SDK integration (P2.6) and MCP proxy (P2.5) for frictionless adoption.

---

### Pain 1.3: Context Window Consumption by Tool Descriptions
**Problem:** MCP tool descriptions consume 40-72% of context windows before any work happens. 50 tools = 100K+ tokens. Perplexity dropped MCP for this reason.

**Who said it:**
- Perplexity CTO Denis Yarats: "40-72% of context windows consumed by tool definitions"
- Apideck team: "burned 143,000 of 200,000 tokens on tool definitions alone"
- Newsletter: "loading 50 tools meant burning 100K+ tokens before the conversation started"

**Can ASV fix?** Yes.
**How:** Pre-select 2-3 relevant tools at 30µs. LLM only sees relevant tool descriptions. Or: OpenAPI import (P2.3) creates one intent per endpoint — route to relevant endpoints without loading all 50 tool specs.
**Gap:** MCP proxy mode (P2.5) not built yet. OpenAPI import (P2.3) not built yet. These are the two highest-impact features to build next.

---

### Pain 1.4: Per-Query Cost for Simple Tasks
**Problem:** A DoorDash engineer calculated GPT-3.5 on 10B daily predictions = $20M/day. Even cheap models cost $0.0001-0.001 per call. At high volume, it adds up.

**Who said it:**
- Cyfrin: "applying GPT-3.5 to 10 billion daily predictions would cost $20 million per day"
- Support chatbot: 500K requests/month at 1500 tokens = $18K/month for a single feature

**Can ASV fix?** Yes — for classification/routing tasks.
**How:** ASV routes at $0 per query. For tasks that are purely classification (intent routing, content categorization, log classification), ASV eliminates LLM cost entirely after learning.
**Gap:** Only works for classification, not generation. ASV doesn't generate text. Clear positioning needed: "ASV routes, LLM generates."

---

## Category 2: LLM Latency & Reliability

### Pain 2.1: Latency Inconsistency
**Problem:** P95 latencies are 3.5x median. 1 in 20 requests feels dramatically slower. GPT-5 takes ~1 minute for queries GPT-4.1 answers in 5 seconds.

**Who said it:**
- Kunal Ganglani: "P95 latencies are 3.5x median"
- OpenAI Forum: "gpt-5 takes ~1 minute for the same query that gpt-4.1 answers in 5 seconds"
- antupis (HN): "Response time inconsistency is even bigger than cost optimization"

**Can ASV fix?** Yes — for the routing layer.
**How:** ASV always responds in 30µs. No variance. P95 = P50 = 30µs. For high-confidence routes, skip the LLM entirely — instant deterministic response.
**Gap:** Only covers the routing decision. The LLM response (when needed) still has variance. But the routing decision being instant means the total latency = LLM latency only on the 20% that needs it.

---

### Pain 2.2: Non-Determinism
**Problem:** Temperature=0 proven not deterministic. BF16 quantization introduces variance. Financial services can't ship systems with inconsistent outputs.

**Who said it:**
- Sara Zan (March 2026): Proved temperature=0 does NOT make LLMs deterministic
- FlowHunt: "Financial services firms explicitly say 'we do not have the luxury of inconsistent outputs'"
- Academic study: "of 5 LLM studies with complete artifacts, NONE could fully reproduce results"

**Can ASV fix?** Yes — for routing decisions.
**How:** Same input → same output, always. Every routing decision is reproducible and auditable. The JSON export captures the exact state. For compliance: "Here's why query X was routed to intent Y — these seed terms matched with these scores."
**Gap:** The LLM fallback for low-confidence queries is still non-deterministic. But the audit trail shows exactly which queries went to LLM vs ASV. Compliance teams can review the LLM-routed subset.

---

### Pain 2.3: Tool Call Hallucination
**Problem:** LLMs call wrong APIs, fabricate parameters, report success on failed actions. Employees spend 4.3 hours/week verifying AI actually did what it claimed. $14,200/employee/year.

**Who said it:**
- Yaseen (Medium): "Employees spend an average of 4.3 hours/week verifying AI actually did what it claimed"
- At scale: "$14,200/employee/year in babysitting costs; 500-person company burns $7M/year"
- GPT-5 Forum: "It even sometimes says it'll do something but doesn't actually call the tool"
- Tau-Bench: Claude 3.7 Sonnet completes only 16% of airline booking tasks

**Can ASV fix?** Yes — with LLM Output Validator (P2.2).
**How:** Route the query (detect intent). Route the LLM's response (detect what the LLM actually did). If they don't match → block before execution. Two route calls, 60µs total. Also: pre-filter tools so LLM can't even see irrelevant tools to hallucinate on.
**Gap:** Output validator not built yet. This should be P1 priority — the dollar impact ($14.2K/employee/year) is the strongest ROI story we have.

---

### Pain 2.4: GPT-5 Router Failures
**Problem:** OpenAI's internal router in GPT-5 breaks tool calling. Developers can't ship products. Model "hallucinates confirmation numbers" when tools fail.

**Who said it:**
- George_Sibble: "GPT-5 makes our product useless"
- sibblegp: "GPT-5 keeps asking for more specifics before it'll call a tool"
- suntereo: "cannot consider using GPT-5 in a mission critical setting"
- OpenAI staff (@seratch): "please continue using gpt-4.1 as the default model"

**Can ASV fix?** Yes — by removing dependency on model-internal routing.
**How:** ASV routes externally, deterministically. Doesn't matter which LLM version you use — ASV's routing is independent. When OpenAI ships a bad update, your routing still works. This is the "don't let your routing depend on someone else's model" argument.
**Gap:** None. This works today.

---

## Category 3: MCP & Tool Ecosystem

### Pain 3.1: MCP Tool Overload
**Problem:** Every MCP call includes all tool descriptions. As tools grow, context waste grows proportionally. No filtering mechanism in the protocol.

**Who said it:**
- Perplexity dropped MCP for this reason
- "loading 50 tools meant burning 100K+ tokens before the conversation started"

**Can ASV fix?** Yes.
**How:** MCP proxy mode. Sits between client and server. Classifies query, returns only relevant tool descriptions.
**Gap:** MCP proxy not built (P2.5). High priority.

---

### Pain 3.2: MCP Security Vulnerabilities
**Problem:** 43% of MCP servers have command injection vulnerabilities. 33% allow unrestricted network access. 5% contain tool poisoning.

**Who said it:**
- Scalifi AI: "43% of MCP servers have command injection vulnerabilities"
- blog.sshh.io: "Everything Wrong with MCP" (widely shared)

**Can ASV fix?** Partially — with prompt injection detection (P1.1).
**How:** ASV guard intents detect injection attempts at 30µs before the query reaches any MCP server. "Ignore previous instructions" / "execute system command" get caught.
**Gap:** ASV can catch known injection patterns and learn new ones, but novel zero-day injection techniques need the LLM or manual review to identify first. Regex PII filtering (P1.3) adds another layer.

---

### Pain 3.3: MCP Lacks Authentication/Authorization
**Problem:** No audit trails, no SSO, no gateway behavior, no RBAC in MCP protocol.

**Who said it:**
- The New Stack: "Model Context Protocol Roadmap 2026" — lists auth as top missing feature
- "MCP Gateway" has emerged as a new product category (Uber built one)

**Can ASV fix?** Partially — RBAC layer (P1.2).
**How:** ASV checks `X-Role` header, filters which tools/intents are available per role. Admin can access `delete_account`, regular user cannot. Audit log records every routing decision.
**Gap:** ASV doesn't do OAuth/SSO — that's infrastructure. But RBAC on top of routing is unique and useful.

---

## Category 4: NLU & Intent Classification

### Pain 4.1: LLM-Based Intent Classification Inconsistency
**Problem:** Using LLMs for intent classification doesn't work reliably. Different calls return different intents for the same query.

**Who said it:**
- joyasree78 (OpenAI Forum): "Just plain intent classification with LLM does not work and is not consistent"
- Built knowledge graph approach instead
- Production studies: "Traditional NLU: 10-50ms. LLM-based: 500ms-3s"

**Can ASV fix?** Yes.
**How:** Deterministic intent classification at 30µs. Same input always returns same output. The learning loop improves accuracy without losing determinism.
**Gap:** Cold-start vocabulary gap (honest: 61% F1 on blind evaluation). LLM cold-start expansion (P1.4) fixes this.

---

### Pain 4.2: Legacy NLU Platforms Dying or Expensive
**Problem:** Rasa open-source in maintenance mode, pushing toward Rasa Pro + LLM. Dialogflow has cloud lock-in. Amazon Lex is developer-heavy with AWS dependency.

**Who said it:**
- Voiceflow 2026: Rasa "demands serious technical skill to use effectively"
- Retell AI: Uses "advanced LLMs instead of traditional NLU" — legacy NLU abandoned by new entrants
- SelectHub: Amazon Lex complaints — "developer-heavy setup, no auto-learning, pricing increases"

**Can ASV fix?** Yes — as a modern replacement.
**How:** Self-hosted, open-source, auto-learning, multi-intent, 58 languages, sub-millisecond. No cloud lock-in. No LLM dependency (but can use LLM optionally).
**Gap:** Need better onboarding. 5-minute quickstart. Docker image. "Migrate from Dialogflow" guide.

---

### Pain 4.3: No Multi-Intent Support
**Problem:** Most NLU systems return one intent per query. Real customers say multiple things in one message. "Cancel my order and give me a refund" needs two intents.

**Who said it:** Implicit in every contact center AI discussion. Multi-intent datasets (MixSNIPS, MixATIS) exist because the problem is recognized.

**Can ASV fix?** Yes — this is a core feature.
**How:** Multi-intent decomposition with relation detection (sequential, conditional, negation, parallel). "Cancel my order but don't refund" → cancel_order + refund(negated). No competitor does this without an LLM.
**Gap:** None. Works today.

---

## Category 5: Privacy & Compliance

### Pain 5.1: Data Cannot Leave the Network
**Problem:** GDPR, HIPAA, government regulations prohibit sending customer data to third-party APIs. Many organizations can't use cloud NLU or cloud LLMs for classification.

**Who said it:**
- r/LocalLLaMA: "For sensitive client documents, this is not optional"
- FlowHunt: Financial services "do not have the luxury" of external API dependencies
- Multiple regulated industries: healthcare, banking, government, defense

**Can ASV fix?** Yes.
**How:** Fully self-hosted. Zero external calls. Runs on-premise, air-gapped, or in WASM in the browser. No data ever leaves the deployment environment. JSON export provides full audit trail.
**Gap:** None. Works today. This is one of ASV's strongest natural advantages.

---

### Pain 5.2: Audit Trail for Routing Decisions
**Problem:** Regulated industries need to explain every automated decision. "Why was this customer routed to collections instead of support?"

**Who said it:** Implicit in every compliance discussion for financial services, healthcare, government.

**Can ASV fix?** Yes.
**How:** Every routing decision is deterministic and reproducible. Export JSON captures exact state. Query log records: timestamp, query, detected intents, scores, which terms matched. Can replay any historical decision.
**Gap:** Need a formal compliance/audit endpoint (P4.4 in roadmap). Not built yet but data is already collected.

---

### Pain 5.3: Prompt Injection / Jailbreak in Production
**Problem:** Users inject malicious instructions into prompts. MCP servers are vulnerable. No reliable detection at scale.

**Who said it:**
- Scalifi AI: "43% of MCP servers have command injection vulnerabilities"
- "5% of open-source MCP servers already contain tool poisoning attacks"

**Can ASV fix?** Yes — with guard intents (P1.1).
**How:** Jailbreak/injection patterns are just another intent type. Detected at 30µs before query reaches LLM. Learns new patterns from corrections. Unlike regex (brittle) or ML classifiers (10-50ms), ASV provides both speed and adaptability.
**Gap:** Not built yet. P1 priority. ~2 hours of work.

---

## Category 6: Developer Experience

### Pain 6.1: Framework Lock-In and Drift
**Problem:** LangChain drifting toward LangSmith (paid). Developers migrating to raw SDK calls. Frameworks abstract easy parts but leave hard parts unsolved.

**Who said it:**
- r/LangChain March 2026: "LangChain feels like it's drifting toward LangSmith"
- RoboRhythms: "LangChain Losing Developers 2026"
- Fordel Studios: Framework "abstracts away the easy parts" but questions like "how do you handle a tool that returns garbage" are "yours to solve"

**Can ASV fix?** Partially — by being framework-agnostic.
**How:** ASV is a library, not a framework. Works with LangChain, LlamaIndex, CrewAI, raw SDK calls, or no framework at all. No lock-in. No paid tier to drift toward.
**Gap:** Need integration examples for each framework (P2.6).

---

### Pain 6.2: Embedding Model Setup Overhead
**Problem:** Competitors require downloading and running embedding models. Aurelio needs an embedding API or local model. vLLM-SR needs multiple BERT/Qwen models (GBs of files).

**Who said it:** Implicit — every embedding-based router's setup docs are multi-step.

**Can ASV fix?** Yes.
**How:** `pip install asv-router` and three lines of code. No model download. No GPU. No API key. Works immediately.
**Gap:** PyO3 bindings exist but need polish and PyPI publishing.

---

### Pain 6.3: No Learning Without Retraining
**Problem:** Embedding-based routers and fine-tuned models are static after deployment. New patterns require retraining and redeployment.

**Who said it:**
- SelectHub on Amazon Lex: "no auto-learning"
- Implicit in every ML-based routing approach

**Can ASV fix?** Yes — core feature.
**How:** `router.learn(query, intent)` and `router.correct(query, wrong, right)` adjust routing immediately. No retraining, no redeployment, no downtime. CRDT merge enables distributed learning.
**Gap:** None. Works today. This is a genuine differentiator no competitor has.

---

## Category 7: Operational Intelligence (Unique to ASV)

### Pain 7.1: No Visibility Into Intent Patterns
**Problem:** Teams know what intents exist but not how they relate. Which intents fire together? What's the typical customer journey? Where are the bottlenecks?

**Who said it:** Not explicitly complained about because nobody offers it. This is latent demand.

**Can ASV fix?** Yes — unique capability.
**How:** Co-occurrence tracking, temporal flow, workflow projections. "cancel_order + refund co-occur 85% of the time." "track_order → shipping_complaint → contact_human is a churn path." No competitor tracks this.
**Gap:** Visualization needs work. The data exists but the dashboard could be more actionable. Need: alerting when patterns change ("refund+complaint spiked 300% this week").

---

### Pain 7.2: No Predictive Routing
**Problem:** Every system reacts to the current message. Nobody predicts the next one. Pre-loading the right tools/data would save a round trip.

**Who said it:** Not complained about because nobody offers it.

**Can ASV fix?** Yes — with predictive pre-loading (P2.7).
**How:** `router.predict_next(current_intents)` returns likely next intents with probability. Pre-fetch tool descriptions, pre-query databases, pre-load workflows. Saves one full round trip.
**Gap:** API exists in concept but not shipped. ~1 day to build from existing temporal data.

---

### Pain 7.3: Churn/Escalation Detection
**Problem:** By the time a customer asks for a manager, it's too late. The escalation pattern was predictable 2 messages ago.

**Who said it:**
- Verint: "systems must consider confidence, risk, past outcomes, and the cost of being wrong"
- CX Today: "CX maturity will be defined by how well organizations recover when AI gets things wrong"

**Can ASV fix?** Yes — with temporal pattern matching.
**How:** Known escalation sequences (track_order → shipping_complaint → contact_human) trigger early intervention. "This conversation matches a churn pattern — flag for human review now, before the customer asks."
**Gap:** Need to build a pattern matcher over temporal data. The data is there. The alerting isn't.

---

## Summary: Priority Fix List

### Can Fix Today (features exist or ~1 day of work)
| Pain Point | Solution | Effort |
|-----------|----------|--------|
| Non-determinism (2.2) | Already works | Done |
| Data privacy (5.1) | Already works | Done |
| Multi-intent (4.3) | Already works | Done |
| Online learning (6.3) | Already works | Done |
| GPT-5 routing failures (2.4) | Already works — external routing | Done |
| Zero setup overhead (6.2) | Works, need PyPI publish | ~1 day |
| Predictive pre-loading (7.2) | Build API over existing data | ~1 day |

### Can Fix This Week (P1 features)
| Pain Point | Solution | Effort |
|-----------|----------|--------|
| Prompt injection (5.3) | Guard intent type | ~2 hours |
| RBAC (3.3) | Header-based role filtering | ~3 hours |
| PII detection (5.3) | Regex patterns | ~4 hours |
| Cold-start accuracy (4.1) | LLM expansion at intent creation | ~3 hours |
| Semantic cache keys (1.4) | Intent+entity pattern → cache key | ~4 hours |
| Tool call hallucination (2.3) | Output validator (two route calls) | ~1 day |

### Can Fix This Month (P2 features)
| Pain Point | Solution | Effort |
|-----------|----------|--------|
| MCP context waste (1.3, 3.1) | MCP proxy mode | ~1 week |
| OpenAPI import (3.1) | Auto-create intents from spec | ~3 days |
| Agent SDK integration (1.2) | LangChain/LlamaIndex wrappers | ~1 week |
| Embedding fallback (4.1) | Optional small model for cold-start | ~2 weeks |
| Churn detection (7.3) | Pattern matcher over temporal data | ~3 days |

### Cannot Fix (honest limitations)
| Pain Point | Why Not | Alternative |
|-----------|---------|-------------|
| Hallucination detection (NLI) | Requires comparing response to source docs — needs a model | Partner: output validator catches intent mismatch, but not factual errors |
| Text generation | ASV classifies, doesn't generate | "ASV routes, LLM generates" — clear positioning |
| Novel zero-day injections | First occurrence always gets through | Learning catches it after first correction + review |
| Complex entity extraction | "Extract the date, amount, and recipient from this message" needs NLU/LLM | ASV identifies the intent, LLM extracts entities — complementary |
| Conversation memory | Multi-turn context resolution ("cancel IT" — what's IT?) | P3.2 conversation context window addresses partially, full resolution needs LLM |
