# MicroResolve — Use Cases

> MicroResolve is the **reflex layer** for AI applications — the fast,
> learned System 1 that resolves the easy 80% of decisions so your LLM
> only handles what actually needs thought.

## Table of Contents

**Tier 1 — high-impact, unexplored (NEW)**
- [LLM output classification](#llm-output-classification) — filter responses for quality/safety at 30μs
- [Model cascade routing](#model-cascade-routing) — Haiku vs Sonnet vs Opus, learned per query
- [Context relevance scoring for RAG](#context-relevance-scoring-for-rag) — 30μs vs 50ms reranking
- [Multi-agent dispatch](#multi-agent-dispatch) — which specialist agent handles this

**Tier 2 — solid (see below)**
- Prompt template selection
- Semantic query deduplication
- Urgency / priority classification
- Tone / sentiment (pre or post LLM)
- Learned compliance / policy enforcement
- Language detection at microsecond speed

**Tier 3 — niche**
- Intent evolution tracking across multi-turn sessions
- Cross-lingual intent unification
- Per-namespace learned guardrails

---

## Tier 1 — New high-impact use cases

### LLM output classification

**The problem.** Guardrails / safety / relevance checking on LLM responses
currently costs another LLM call — 200ms + $0.001 per generation. For
apps doing 10M generations/day, that's $10k/day and a doubled latency
budget.

**The reflex-layer approach.** Train the reflex on labelled LLM outputs
(good / bad / flagged). Classify each generation in ~30μs before shipping
to the user. Costs drop 100×, latency drops 7,000×.

```python
reflex.add_intent("safe", [...good_examples])
reflex.add_intent("hallucination", [...known_hallucinations])
reflex.add_intent("policy_violation", [...flagged_outputs])
reflex.add_intent("off_topic", [...derails])

# after each LLM call
label = reflex.route(llm_output)
if label != "safe":
    fall_back_to_llm_or_human()
```

Replaces: Guardrails AI, NeMo-Guardrails, secondary LLM judge calls.

---

### Model cascade routing

**The problem.** Most queries don't need Opus. Some don't even need Sonnet.
Teams pay top-tier model prices for queries a cheaper model could handle.

**The reflex-layer approach.** Log historical query → model-needed decisions
(human-curated or LLM-judged). The reflex learns which queries routinely
succeed at Haiku, which need Sonnet, which truly need Opus.

```python
reflex.add_intent("haiku_ok",   [...simple_queries])
reflex.add_intent("need_sonnet",[...moderate_queries])
reflex.add_intent("need_opus",  [...complex_queries])

target = reflex.route(user_query)[0].id
response = anthropic.messages.create(model=target, ...)
```

Typical savings: 70–80% of LLM spend at scale.

---

### Context relevance scoring for RAG

**The problem.** RAG retrieves 50 candidate chunks; the LLM needs 5.
Current solutions: cross-encoder rerankers (10–50ms) or reranking LLMs
(hundreds of ms + cost). Both scale poorly.

**The reflex-layer approach.** Per-namespace relevance reflex trained on
query + chunk → relevance label. Microsecond scoring for all 50 chunks
in parallel, before the LLM sees anything.

```python
# Per-query, score all chunks
scores = [reflex.relevance(query, chunk) for chunk in retrieved_50]
top_5  = sorted(zip(scores, retrieved_50), reverse=True)[:5]
llm.answer(query, context=top_5)
```

30μs × 50 = 1.5ms total — 30× faster than a cross-encoder, 200× faster
than an LLM reranker.

---

### Multi-agent dispatch

**The problem.** Agent swarms (Anthropic SubAgents, LangGraph, CrewAI)
need a dispatcher that decides which specialist handles a request.
Hand-coded rules don't cover edge cases; an LLM dispatcher doubles cost.

**The reflex-layer approach.** The dispatcher IS a reflex — learned from
historical (task → correct agent) pairs.

```python
reflex.add_intent("code_agent",   [...coding_tasks])
reflex.add_intent("data_agent",   [...analytical_tasks])
reflex.add_intent("writer_agent", [...prose_tasks])
reflex.add_intent("web_agent",    [...browsing_tasks])

agent_id = reflex.route(task)[0].id
agents[agent_id].handle(task)
```

Reflex learns new dispatch patterns as supervisors correct mistakes. No
re-training, no redeploy.

---



### Customer Support Ticket Routing

```python
router.add_intent("billing", ["billing issue", "wrong charge", "overcharged"])
router.add_intent("technical", ["app crashed", "not loading", "error message"])
router.add_intent("account", ["reset password", "locked out", "change email"])

# Route incoming tickets to the right team
result = router.route("I keep getting charged twice")
# → billing, score 1.8
```

No ML model to train or host. Learns from agent corrections over time.

### Chatbot Intent Classification

```python
router.add_intent("cancel_order", ["cancel my order", "I want to cancel", "stop my order"])
router.add_intent("track_order", ["where is my package", "track order", "shipping status"])
router.add_intent("refund", ["get a refund", "money back", "return item"])

result = router.route("I need to cancel something")
# → cancel_order
```

## Creative Use Cases

### Multi-Agent Orchestrator

Route user requests to the right AI agent — no LLM router needed.

```python
router.add_intent("vision_agent", ["analyze this image", "what's in this photo", "describe picture"])
router.add_intent("code_agent", ["write a function", "fix this bug", "refactor code"])
router.add_intent("search_agent", ["find information about", "look up", "research"])
router.add_intent("email_agent", ["draft an email", "reply to", "compose message"])

result = router.route("can you check what's in this screenshot?")
# → vision_agent, score 1.8
```

Everyone building multi-agent systems in 2026 needs a router. Most use an LLM to route to other LLMs. MicroResolve replaces that with <1ms dispatch.

### CLI Command Dispatch — Natural Language Shell

```python
router.add_intent("docker_ps", ["show running containers", "list containers", "what's running"])
router.add_intent("git_status", ["what files changed", "show changes", "uncommitted files"])
router.add_intent("git_push", ["push my code", "deploy changes", "send to remote"])
router.add_intent("k8s_pods", ["show pods", "kubernetes status", "what's deployed"])

user_says("what's running on docker?")
# → docker_ps → os.system("docker ps")
```

No LLM needed for known commands. Sub-ms. Works offline.

### Smart Home on a Raspberry Pi

```python
router.add_intent("lights_off", ["turn off lights", "lights out", "dark mode"])
router.add_intent("lights_on", ["turn on lights", "lights please", "brighten up"])
router.add_intent("music_play", ["play music", "put on some songs", "start playlist"])
router.add_intent("music_stop", ["stop music", "pause", "quiet"])
router.add_intent("thermostat", ["set temperature", "make it warmer", "too cold"])

# Runs on Pi Zero — 512MB RAM, no internet needed
# After speech-to-text, route in <1ms
# Learns user's vocabulary: "hit the lights" → lights_off after one correction
```

Edge AI with zero cloud dependency. Privacy-preserving.

### Code Function Dispatch — Natural Language API Gateway

```python
router.add_intent("calculate_shipping", ["shipping cost", "delivery fee", "how much to ship"])
router.add_intent("apply_discount", ["apply coupon", "discount code", "promo", "voucher"])
router.add_intent("check_inventory", ["in stock", "available", "how many left"])
router.add_intent("create_order", ["place order", "buy", "purchase", "checkout"])

# Non-technical staff queries business logic in English
result = router.route("how much to ship to Mumbai?")
# → calculate_shipping → call_function("calculateShipping", params)
```

Natural language API gateway without hosting an LLM.

### Game NPC Dialogue Trees

```python
router.add_intent("quest_accept", ["I'll do it", "sign me up", "let's go", "count me in"])
router.add_intent("quest_decline", ["no thanks", "not interested", "maybe later"])
router.add_intent("ask_about_quest", ["tell me more", "what's the reward", "details"])
router.add_intent("trade", ["buy", "sell", "what do you have", "show me your wares"])
router.add_intent("attack", ["fight", "attack", "draw sword", "battle"])

# Runs in-game, no server round-trip, no LLM cost per player interaction
# Learns from player patterns: "yolo let's do this" → quest_accept
# Zero frame drops — ~30μs routing, CPU-only
```

Replaces rigid keyword matching with fuzzy, learnable routing.

### Log/Alert Classification

```python
router.add_intent("disk_critical", ["disk space", "storage full", "no space left on device"])
router.add_intent("memory_leak", ["OOM", "out of memory", "heap exhausted", "memory pressure"])
router.add_intent("network_issue", ["connection refused", "timeout", "DNS failure", "unreachable"])
router.add_intent("auth_failure", ["unauthorized", "401", "forbidden", "invalid token"])
router.add_intent("deploy_issue", ["deploy failed", "rollback", "build broken", "CI failed"])

# Route 10,000 log lines/second — no ML model needed
# Learns from ops team corrections over time
```

Embedding models choke at volume. MicroResolve handles millions of log lines per second.

### Accessibility — Personal Voice Command Learning

A user with a speech disability says "ughh ligh" for "turn off lights". MicroResolve learns this after one correction:

```python
# User says something the system doesn't understand
result = router.route("ughh ligh")  # → no match

# Caregiver corrects once:
router.learn("ughh ligh", "lights_off")

# From now on, their specific pronunciation routes correctly
result = router.route("ughh ligh")  # → lights_off ✓
```

No retraining, no model update, no cloud upload. The system adapts to THEIR voice in real-time, on-device.

### Email Auto-Labeling

```python
router.add_intent("urgent", ["ASAP", "urgent", "immediately", "critical deadline"])
router.add_intent("meeting", ["meeting invite", "calendar", "schedule", "sync up"])
router.add_intent("newsletter", ["unsubscribe", "weekly digest", "newsletter", "updates"])
router.add_intent("invoice", ["invoice", "payment due", "receipt", "billing statement"])

# Process inbox at startup — classify hundreds of emails in milliseconds
for email in inbox:
    label = router.route_best(email.subject, min_score=1.0)
    if label:
        email.apply_label(label.id)
```

### Content Moderation (First Pass)

```python
router.add_intent("obvious_violation", ["kill", "bomb threat", "explicit slur list..."])
router.add_intent("needs_review", ["hate", "harassment", "threatening"])
router.add_intent("likely_safe", ["hello", "thanks", "good morning"])

# MicroResolve as first pass: <1ms per message
# Only send "needs_review" to expensive LLM for nuanced judgment
# Saves 80% of LLM costs on moderation
```

## Priority for README Examples

1. **Customer support routing** — proves it works, everyone understands
2. **Multi-agent orchestrator** — 2026 hot topic
3. **CLI dispatch** — developers instantly get it
4. **Smart home on Raspberry Pi** — "wow it's that small?" moment
5. **Accessibility** — the one that makes people share your post
