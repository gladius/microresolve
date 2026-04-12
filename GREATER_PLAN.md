# The Greater Plan

> ASV started as an intent router. It may be something much larger.

---

## The Insight

Every AI agent system today has the same architecture:

```
user query → [giant static system prompt + all tools] → LLM → response
```

The system prompt is a blob. The tool list is a dump. The LLM has to read everything on every call, reason over all of it, and pick the right behavior. This works at small scale. At enterprise scale it is expensive, slow, unreliable, and unauditable.

ASV's routing already solves tool selection. But there is a deeper observation:

**What if intent detection returned instructions, not just labels?**

```
user query → ASV (30µs) → [intent_id + prompt_fragment + tool_schema]
           → assembled system prompt → LLM → response
```

The system prompt is no longer static. It is assembled at query time from the detected intents. The LLM receives exactly the instructions, tools, and context relevant to *this specific query* — nothing else.

This is not routing. This is **semantic prompt assembly**. And it changes what ASV is.

---

## What This Makes ASV

A **cortex** — a deterministic, learnable dispatch layer that sits between the user and the LLM. It decides what the LLM should know, what tools it should have, and how it should behave — before the LLM sees a single token of the request.

The intents are no longer just labels. They are **executable units** carrying:
- Routing vocabulary (phrases that activate them)
- Behavioral instructions (what the LLM should do when active)
- Tool schemas (which capabilities become available)
- Guardrails (what the LLM must not do in this context)
- Persona modifiers (tone, formality, technicality)

The user's query is the key. ASV is the lookup. The LLM is the executor.

---

## Use Cases

### 1. Dynamic System Prompt Assembly
Instead of a 10,000-token static system prompt covering all scenarios, you store prompt fragments per intent. A billing question injects billing instructions. A technical question injects technical instructions. A legal question injects legal guardrails. The LLM only ever sees what is relevant.

**Impact:** 60-80% reduction in average system prompt size. Faster responses. Lower cost per query.

### 2. Intent-Conditioned Tool Injection (Solving MCP Dump)
Detected `stripe:create_refund` → inject only the Stripe refund tool schema.
Detected `linear:create_issue` → inject only the Linear create tool schema.

The LLM never sees 100 tool schemas. It sees 2-3, pre-selected by ASV at 30µs.

**Impact:** Eliminates tool selection errors. Reduces input tokens by 90% for tool-heavy agents. Solves the core MCP scaling problem without changing the LLM or the tools.

### 3. Guardrail Injection
Guardrails today are static and always-on. With intent routing, guardrails only activate when the relevant intent fires.

- Detected `update_dispute` → inject "do not admit liability, escalate disputes over $1000"
- Detected `create_refund` → inject "verify original payment ID before processing"
- Detected `legal_question` → inject "do not provide legal advice, redirect to counsel"

Guardrails that do not apply to the current query consume zero tokens.

### 4. Intent-Specific Personas
One LLM, many behaviors — routed deterministically.

- Billing dispute intent → formal, empathetic, no commitments
- Technical debug intent → precise, show code, skip pleasantries
- Sales inquiry intent → warm, highlight value, soft close

No prompt engineering required for each case. Define once per intent, activate automatically.

### 5. Negation-Aware Instruction Flipping
ASV already detects negation. "Do not process a refund" routes to `create_refund` with `negated: true`. Map negated intents to different instruction fragments — the refund denial playbook instead of refund approval. Context-aware behavior with no LLM involvement in the routing decision.

### 6. Multi-Tenant Prompt Differentiation
Different namespaces (enterprise clients) carry different instruction sets for the same intent.

- Client A's `create_refund` → "auto-approve under $50, no questions"
- Client B's `create_refund` → "always escalate to human, never auto-approve"
- Client C's `create_refund` → "check fraud score first, apply 3-day hold"

Same routing logic. Different assembled behavior per tenant. Zero per-tenant prompt engineering at inference time.

### 7. Compliance and Regulatory Routing
Certain intents trigger compliance requirements that must appear in the system prompt.

- GDPR namespace: data deletion requests inject data handling regulations
- Healthcare namespace: medical questions inject HIPAA disclaimers and clinical guardrails
- Financial namespace: investment questions inject SEC disclosure requirements

Compliance instructions are attached to intents, not baked into a monolithic prompt. They can be audited, versioned, and updated independently.

### 8. Observable, Auditable AI
Every routing decision is logged with the query, matched intents, scores, and latency. The assembled system prompt is deterministic and reproducible from the routing output.

This is the opposite of LLM black-box behavior. Every decision has a traceable cause:
- Why did the agent behave this way? → Because `stripe:update_dispute` was detected at score 0.87
- Why did the LLM have refund instructions? → Because `create_refund` was in the confirmed set
- Why was the guardrail active? → Because the dispute intent fired

Auditors get a routing log. Developers get reproducibility. Compliance gets traceability.

### 9. Edge Deployment and Privacy
Because routing is model-free and runs at 30µs, it can run:
- At the edge (CDN nodes, regional endpoints)
- In WASM (browsers, mobile apps)
- On-premise (sensitive data never leaves the network)
- Offline (no LLM call needed for routing decisions)

The decision about *what the LLM should see* can be made locally, before any data is transmitted. Sensitive routing logic (PII detection, compliance flags, customer tier checks) runs without touching external APIs.

### 10. Learned Prompt Improvement
The online learning that improves routing accuracy also improves prompt assembly. When a query is misrouted, the correction updates the routing index — which also corrects which prompt fragment gets assembled. Routing accuracy and prompt assembly quality improve together, with the same learning signal.

### 11. Intent Inheritance and Composition
Intents can form hierarchies:

```
refund (base)
  ├── refund:standard    → basic approval flow
  ├── refund:disputed    → escalation flow + legal guardrail
  └── refund:fraudulent  → fraud investigation flow + hold instructions
```

Multi-intent detection composes behaviors:
- `create_refund` + `update_dispute` detected simultaneously → compose both instruction sets, merge tool schemas, apply intersection of guardrails

### 12. A/B Testing Prompt Strategies
Because prompt fragments are stored per intent, you can run controlled experiments:

- Intent A gets prompt fragment version 1 for 50% of traffic, version 2 for 50%
- Measure downstream LLM output quality, cost, user satisfaction
- Promote the winner without touching routing logic

Prompt experimentation decoupled from routing logic.

---

## The Programming Language Angle

If you extend this far enough, intents start looking like **functions**. The routing system is a **pattern matcher**. The metadata is an **environment**. The learning is **runtime optimization**.

A domain-specific AI behavior could be expressed as:

```
intent create_refund
  matches: ["refund", "money back", "reverse charge", ...]
  type: action
  instructions: |
    Verify payment_intent ID before processing.
    Apply refund within 5-10 business days.
    Never promise instant credit.
  tools: [stripe.refunds.create]
  guardrails: [no_liability_admission, verify_identity_first]
  on_negation: defer_to_human
```

This is declarative AI behavior. Not a prompt. Not a chain. Not a workflow. A typed, learnable, composable unit of domain-specific intelligence.

A collection of these intents is a **program** — one that describes how an AI agent should behave across an entire domain, without LLM calls in the decision path.

---

## The Core Properties That Make This Different

| Property | LLM routing | Vector DB (RAG) | ASV |
|---|---|---|---|
| Routing latency | 500ms-2s | 20-50ms | 30µs |
| Learns from corrections | No | No | Yes |
| Negation aware | Sometimes | No | Yes |
| Multi-intent | Sometimes | Rare | Yes |
| Runs offline / edge | No | No | Yes |
| Auditable decisions | No | Partial | Yes |
| Token cost at scale | High | Medium | Near-zero |
| Prompt assembly | No | Manual | Native |

---

## The Positioning

**Today:** ASV is "80% of routing at $0, 30µs, send 20% to LLM."

**Greater Plan:** ASV is the deterministic cortex of every AI agent system — the layer that decides what the LLM knows, what it can do, and how it should behave, before the LLM sees a single token. The LLM becomes an executor, not a planner.

**The pattern:**
```
Input → ASV (deterministic, learnable, 30µs)
      → assembled context (instructions + tools + guardrails)
      → LLM (executes against focused context)
      → Output
```

This is not a new LLM. It is not a new agent framework. It is the missing layer between user intent and LLM execution — one that is fast enough to run on every request, cheap enough to deploy at any scale, and smart enough to improve over time without retraining.

---

---

## Intent Programming

> Zero code. The chain is written inside the instructions. The LLM is the interpreter.

### The Problem with Every Existing Approach

LangChain, LangGraph, Semantic Kernel, AutoGPT — all of them require a programmer to define the control flow in code:

```python
if refund_detected and dispute_detected:
    chain = dispute_chain
else:
    chain = refund_chain
```

The programmer decides which prompts chain to which. The LLM just executes within those fixed paths. Add a new scenario → modify code.

### The Intent Programming Model

In Intent Programming there is no code. Every intent carries:

- **Phrases** — what user language activates this intent (routing vocabulary, learned over time)
- **Instructions** — what the LLM should do when this intent is active (natural language)
- **Schema** — what structured data is expected or produced (input/output contract)
- **Context** — background information the LLM needs for this intent
- **Guardrails** — what the LLM must not do in this context
- **Decision rules** — written in plain English, evaluated by the LLM ("if payment ID is missing, ask before proceeding")
- **Transitions** — which intents may follow, or let the LLM discover them from the next query

The execution loop:

```
user input
  → ASV routes (30µs, no LLM)
  → loads matched intent's full payload
  → LLM executes against that payload
  → LLM response + conversation history
  → ASV routes next turn
  → loads next intent's payload
  → repeat
```

No programmer writes the chain. The chain emerges from how the LLM follows the instructions embedded in each intent.

### What This Means Concretely

An intent called `create_refund` could carry:

```
instructions: |
  Process a refund for the customer.
  Before proceeding: verify the payment_intent ID is provided.
  If the customer also mentions a dispute, acknowledge both and handle
  the dispute context first — do not process the refund until the
  dispute intent is resolved.
  Never promise a specific timeline unless the payment method confirms it.

schema:
  required: [payment_intent_id]
  optional: [amount, reason]

guardrails:
  - Do not admit fault on behalf of the company
  - Do not process refunds over $500 without escalation

context: |
  Refunds typically take 5-10 business days.
  Fraudulent refund requests should be flagged, not processed.
```

The LLM reads this and knows: ask for missing fields, sequence with dispute if needed, apply the guardrails. The programmer wrote none of that logic in code.

### Self-Improving

Intents improve over time along two axes simultaneously:

1. **Routing improves** — more queries get matched to the right intent as phrases are learned from corrections
2. **Instructions improve** — the LLM's failures (wrong decisions, missing context, guardrail violations) feed back into improving the intent's instruction text

This is a system that gets better with usage without retraining a model or modifying code. The "program" rewrites itself.

### Comparison to Industry

| System | Control flow | Instructions | Routing | Learns |
|---|---|---|---|---|
| LangChain | Python code | Hardcoded prompts | None | No |
| Semantic Kernel | C# code | Skills (code + prompt) | LLM planner (slow) | No |
| AutoGPT | Python code | Fixed system prompt | None | No |
| ReAct | Code loop | Fixed tools | None | No |
| **Intent Programming** | **None** | **Per-intent, natural language** | **30µs keyword router** | **Yes** |

The critical difference: **the routing and the program are the same object.** An intent's phrases determine when it activates. Its instructions determine what happens when it does. No programmer defines the chain separately.

### The Execution Engine Needed

The existing `simulate_turn` + `simulate_respond` loop in Studio is almost this. What closes it:

1. **Intent payload endpoint** — `GET /api/intents/:id/payload` returns instructions + schema + guardrails + context assembled into a ready-to-use system prompt fragment
2. **Route-and-assemble** — `POST /api/route/assemble` routes the query and returns the merged payload from all matched intents
3. **Conversation loop** — each LLM response feeds back into routing, the loop continues until no new intents fire or a terminal intent is reached
4. **Intent metadata schema** — standardize `instructions`, `schema`, `guardrails`, `context`, `transitions` as first-class metadata keys

### Proof of Concept Plan

Before designing the engine, the routing layer must be validated:

1. **Test auto-learn accuracy** — does the router correctly identify the right intent from natural language? (the 20-intent, 25-query test we are building)
2. **Test multi-intent composition** — do multiple simultaneous intents assemble coherently?
3. **Test instruction following** — given a loaded intent payload, does the LLM follow the instructions correctly?
4. **Test chain emergence** — does a multi-turn conversation, routed turn by turn, produce a coherent workflow with no programmer-defined transitions?

The routing accuracy test comes first. Everything else depends on it. A chain built on bad routing is just a faster way to give wrong answers.

---

## What Needs to Be Built

### Near-term (already feasible)
1. **Route-and-assemble endpoint** — `POST /api/route/assemble` returns intent IDs + merged instruction fragments from metadata. One endpoint, ~30 lines of Rust.
2. **Metadata schema conventions** — document `instructions`, `tools`, `guardrails`, `persona` as standard metadata keys.
3. **Studio prompt editor** — edit intent instruction fragments in the UI alongside phrases.

### Medium-term
4. **Intent inheritance** — parent/child intent relationships, instruction merging rules.
5. **Negation routing** — `on_negation` metadata field, different assembled output for negated intents.
6. **Prompt versioning** — version intent instruction fragments, A/B routing built in.

### Long-term
7. **Intent definition language** — a declarative syntax for defining intents, phrases, instructions, tools, and guardrails as a single unit.
8. **Compiler** — transpile intent programs to ASV router state + metadata packs.
9. **Runtime** — serve intent programs as a managed API, multi-tenant, with per-namespace learning.

---

## Intent Programming v2: Closing the Gaps

> These four mechanisms make Intent Programming fully possible without per-intent code.

> **Status: All four mechanisms tested and working — 2026-04-11**
> See `/tmp/ip_v2_test.py` for the test harness and results below each section.

### 1. Reflective Routing — Conversational State Without Code

**Problem**: Follow-up turns ("yes", "the amount is $50") don't carry routing keywords — the intent drops.

**Solution**: Route on `[last 150 chars of LLM's previous response] + [current user message]`.

When the LLM responds "Could you provide the **dispute_id** and the **evidence** you have?" — those words are now in the routing window. The next user turn "dp_12345, I have a signed contract" routes back to `update_dispute` automatically.

**The LLM's output vocabulary closes the routing loop.** No session store. No `current_intent` field. The conversation text is the state.

Implementation: in `route_execute`, build `routing_query = last_assistant_window + " " + user_query` before calling `route_assemble`.

**Test result**: Turn 2 (`"dp_12345, I have a contract"`) routed alone → `confirmed=[]`. With 150-char reflective window → `confirmed=['stripe:update_dispute']`. Proved.

**Open question**: This is vocabulary bridging for a keyword router — not "context" in the LLM sense. The LLM always had full history. The novelty is that the *30µs model-free router* gets contextual vocabulary without LLM involvement. LangChain passes history to an LLM to route; ASV passes a vocabulary window to a keyword index. Routing stays at 30µs with no model call.

**Known gap**: If the LLM's clarify/recovery response contains vocabulary from the wrong intent (e.g., it mentions "vercel" while recovering from an unrelated query), reflective routing may pull the next turn toward vercel. Clarify turns should be stripped from the routing window — see §2 below.

---

### 2. `__clarify__` — The Universal Fallback Intent

**Problem**: When nothing routes above threshold, the whole system fails silently.

**Solution**: A synthetic system intent `__clarify__` activates whenever no real intent matches. Its metadata:

```
ip_instructions: The user's request did not match any known workflow.
                 Greet them and list what you can help with. Ask what they need.
ip_context: Available capabilities: [dynamically injected from router.intent_ids()]
```

The LLM's clarify response will naturally contain vocabulary from whatever domain the user intends — and the *next* turn routes correctly. One turn of friction, then back on track.

**This is also the solution for unknown intents in normal tool flows.** Instead of a null routing result breaking the pipeline, `__clarify__` gracefully recovers. The system always has a response — even for inputs it has never seen.

The fallback is not "no intent" — it is a real intent whose *instruction* is to recover gracefully.

**Test result**: For queries with zero token overlap with any intent, `__clarify__` fires. For queries with partial overlap (e.g., "dentist appointment" matching "vercel" via a shared token), the candidate intent loads instead — but the LLM's instructions still drive a graceful "I can't help with that, here's what I can" response. Both paths recover correctly.

**Universal scope**: `__clarify__` is not just for Intent Programming. Any ASV pipeline — tool routing, RAG, agent orchestration — benefits from this safety net. The system always has a response. The LLM's clarify reply naturally contains capability vocabulary, seeding correct routing on the next turn.

**Problem 1 — Wrong intent triggered**: ASV is not intelligent; it can route to the wrong intent. If `create_refund` fires when the user meant `cancel_subscription`, the LLM gets wrong instructions and asks for wrong fields. Solutions:
- **LLM as verifier**: The system prompt tells the LLM which intent was detected. If the LLM recognizes a mismatch ("Active intents: create_refund, but user said 'cancel'"), it can say so. Add to prompt: "If the detected intent clearly does not match the user's request, say [WRONG_INTENT] and ask for clarification." The loop re-routes on the next turn.
- **Confidence gating**: Below a configurable threshold (e.g., score < 0.5 with no high-confidence match), fall back to `__clarify__` rather than loading potentially wrong instructions.
- **Disambiguation prompt**: When top-2 intents have similar scores (score difference < 0.15), present both to the LLM and ask it to confirm which applies given the user's message.

**Problem 2 — Clarify turn context pollution**: Once a real intent is identified after a `__clarify__` turn, the clarify exchange should be stripped from conversation history. A clarify turn is a routing artifact, not part of the actual workflow. Keeping it pollutes the LLM's context, wastes tokens, and injects wrong vocabulary into the reflective routing window (causing the next turn to route toward whatever the clarify response mentioned). Implementation: mark turns with `"is_clarify": true` in history, strip them when a real intent fires.

---

### 3. `ip_next_if` — Conditional Branching via Natural Language

**Problem**: Workflows branch ("if amount > $1000, escalate instead of refunding") — normally requires code.

**Solution**: A metadata field `ip_next_if` carries natural language branch conditions. The LLM evaluates them and signals the next intent.

```
ip_next_if:
  - "if refund amount exceeds $1000, activate stripe:escalate_to_manager"
  - "if all required fields are collected, activate stripe:confirm_refund"
```

The system prompt includes these as instructions. When the LLM decides a condition applies, it outputs `[ACTIVATE: stripe:escalate_to_manager]`. The execution loop reads this signal, routes to the new intent, loads its metadata, continues.

**The LLM is the `if` statement.** Natural language conditions are more expressive than boolean expressions, and the ceiling rises every model generation automatically.

**Test result**: Refund query for $2000 → LLM emitted `[ACTIVATE: stripe:escalate_to_manager]`, explained escalation, applied the guardrail. Zero branching code.

---

### 4. `ip_execute` — Tool Execution With Result Injection

**Problem**: Actually calling an API (Stripe, Linear, Vercel) requires code per-intent.

**Solution**: One generic HTTP executor in the loop, driven by `ip_execute` metadata:

```
ip_execute: POST https://api.stripe.com/v1/refunds
```

When the LLM has collected all required schema fields, it outputs `[EXECUTE: {"payment_intent_id": "pi_abc", "amount": 50}]`. The loop:

1. Parses the fields from the signal
2. POSTs them to the `ip_execute` URL
3. Injects the result back as a conversation turn: `Tool result: {"refund_id": "re_xyz", "status": "succeeded"}`
4. Re-calls LLM with the result → LLM responds to user

One generic executor, written once. Every intent's tool call is JSON metadata. No per-intent code.

**Test result** (end-to-end):
- Turn 1: LLM asked for payment_intent_id and amount.
- Turn 2: Fields provided, LLM asked for confirmation.
- Turn 3: "yes go ahead" → LLM emitted `[EXECUTE: {"amount": 75, "payment_intent_id": "pi_test123"}]` → server called `https://httpbin.org/post` → got `status=200` → result injected back → LLM confirmed: *"I've successfully processed your refund. pi_test123, $75, 5-10 business days."* No per-intent code. Tool called and result injected.

---

### 5. Conversational Intent Programming Page (Future)

Once the above four mechanisms are proven, the natural product surface is a page where:

1. User describes: "I need a workflow that cancels a subscription and optionally refunds the last payment. Refunds over $500 need manager approval."
2. LLM generates: intent IDs + full `ip_instructions`, `ip_guardrails`, `ip_schema`, `ip_execute`, `ip_next_if` metadata
3. User iterates in natural language: "Add Portuguese support", "Change the refund timeline to 7-14 days"
4. "Apply" → intents created in current namespace via API, live immediately

**The "program" is intent metadata. The "compiler" is the LLM. The "runtime" is the universal execution loop. The "IDE" is a chat window.**

No framework today does this. LangChain generates Python. Semantic Kernel generates C#. They produce code. Intent Programming produces *data* — structured metadata that the loop interprets. Change the data, behavior changes. No deploy. No code review. A product manager can author a new workflow.

---

## Open Questions

- Should instruction fragments be stored in metadata (current) or as a first-class field on intents?
- How do you merge conflicting guardrails from two simultaneously detected intents?
- What is the right granularity for an intent — coarse (one per feature) or fine (one per behavior variant)?
- Can situation patterns (existing ASV feature) serve as the conditional branching mechanism inside intent programs?
- Is there a market for an ASV-native intent definition format, or should it remain API/JSON-driven?

---

*Written 2026-04-11. Starting point for a longer conversation.*
