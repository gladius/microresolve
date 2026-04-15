# Intent Programming

> Natural language is the programming language. Intents are functions.
> ASV is the interpreter. The LLM is the CPU.

---

## The Problem

Every AI agent system today is built in code:

```python
# LangChain — a developer writes this
chain = (
    prompt_template
    | llm
    | output_parser
    | tool_selector
    | execute_tool
)
```

This requires developers. It's brittle. It doesn't learn. And it dumps everything
into one giant system prompt that the LLM reads on every call.

## The Idea

**What if you could build an AI agent by describing your business in a conversation?**

```
You: "I run a pet grooming shop. Customers ask about appointments,
      cancellations, pricing, and complaints."

Builder: Creates 4 intents automatically.

You: "For cancellations, there's a fee if they cancel less than
      24 hours before. Offer to reschedule first. Don't waive
      the fee unless a manager approves."

Builder: Writes the instruction paragraph for the cancel intent.

Done. Working agent. Zero code. The conversation IS the program.
```

No LangChain. No Python. No YAML. No drag-and-drop workflow builder.
A non-technical business owner describes their rules in natural language,
and the system builds a working agent from that description.

---

## How It Works

### Each intent is a function

```
Intent: "customer wants to cancel"
  Phrases:      ["cancel my appointment", "I need to cancel", ...]
  Instructions: "You're handling a cancellation request. Ask the customer
                 when their appointment is. If it's within 24 hours,
                 process the cancellation directly. If it's outside
                 24 hours, explain there's a cancellation fee and offer
                 to reschedule instead. If they insist, charge the fee
                 and confirm. Never waive the fee without manager approval."
  Guardrails:   "Do not offer discounts. Do not admit fault."
  Persona:      "Friendly but professional."
  Tools:        [cancel_booking, reschedule_booking]
```

The instructions contain ALL the branching logic in plain English.
The LLM understands conditionals, exceptions, tone — no state machine needed.

### At query time: snippet assembly

When a user says something, ASV detects the intent (30µs) and assembles
a focused prompt from the matching intent's components:

```
User: "I want to cancel my appointment tomorrow"

ASV detects: booking:cancel (30µs, no LLM)

Assembled system prompt (sent to LLM):
┌─────────────────────────────────────────────┐
│ [Instructions block - from intent]          │
│ You're handling a cancellation request...   │
│                                             │
│ [Guardrails block - appended if present]    │
│ Do not offer discounts. Do not admit fault. │
│                                             │
│ [Persona block - appended if present]       │
│ Be friendly but professional.               │
│                                             │
│ [Tools block - appended if MCP tools exist] │
│ Available tools:                            │
│   cancel_booking(booking_id, reason)        │
│   reschedule_booking(booking_id, new_date)  │
└─────────────────────────────────────────────┘

Each block is optional. Only present blocks are assembled.
The LLM sees ~200 tokens instead of 2000.
```

### The LLM is the CPU

ASV doesn't execute the instructions. The LLM does. ASV is the dispatcher:
- It decides WHICH instructions the LLM sees
- It decides WHICH tools are available
- It decides WHAT guardrails apply
- It decides HOW the LLM should sound

All at 30µs, deterministically, without an LLM call in the routing path.

### It learns

When the agent makes a mistake:
- The correction teaches ASV new routing vocabulary (auto-learn)
- The same intent's instructions get updated (conversational editing)
- Next time: the routing is better AND the instructions are better
- The program improves itself through use

---

## The Conversational Builder

A dedicated UI page where a non-technical user builds their agent:

```
Builder: "What does your business do?"
User:    "We're a veterinary clinic. People call about appointments,
          prescriptions, emergencies, and billing."

Builder: [Creates 4 intents with auto-generated phrases]
         "I've set up 4 categories. Let's define how each should
          be handled. What happens when someone asks about an appointment?"

User:    "Check if they're an existing patient. If yes, offer next
          available slot. If new patient, collect name, pet type,
          and phone number first. We don't do walk-ins."

Builder: [Writes instructions for appointment intent]
         "Got it. What about emergencies?"

User:    "Always take them immediately. Tell them to come in right away.
          If it's after hours, give them the emergency hotline number:
          555-0123. Never tell them to wait."

Builder: [Writes instructions + guardrail for emergency intent]
         "I've added a guardrail: never tell emergency callers to wait.
          What about billing?"
...
```

The builder IS an LLM conversation. It:
1. Asks about the domain → creates intents with phrases
2. Asks about each intent → writes instruction paragraphs
3. Asks about constraints → adds guardrails
4. Asks about tone → adds persona
5. Optionally connects MCP tools → adds tool schemas

The output: a complete namespace with intents, each carrying its instruction
snippet. Ready to serve real users.

---

## Why This Is Different

| | LangChain/CrewAI | Dialogflow/Rasa | Prompt Engineering | **Intent Programming** |
|---|---|---|---|---|
| Who builds | Developers | Developers | Prompt engineers | **Anyone** |
| How | Python code | YAML + UI forms | Trial and error | **Conversation** |
| Flow control | Code (if/else) | State machine | Giant prompt | **Natural language in each intent** |
| Learns | No | No | No | **Yes — from every interaction** |
| Routing cost | LLM per decision | LLM or rules | N/A | **30µs, no LLM** |
| Prompt size | Static, full | Static, full | Static, full | **Assembled per query, minimal** |
| Auditable | Code review | Flow inspection | Read the prompt | **Intent log + scoring trace** |

---

## The Stack

```
User message
  │
  ├─ ASV Intent Detection (30µs, deterministic, learns)
  │   ├─ L0: Typo correction
  │   ├─ L1: Morphology + synonyms
  │   ├─ L2: IDF scoring + token consumption
  │   └─ L3: Inhibition (false positive suppression)
  │
  ├─ Snippet Assembly (microseconds)
  │   ├─ Instructions block (from detected intent)
  │   ├─ Guardrails block (appended if present)
  │   ├─ Persona block (appended if present)
  │   └─ Tools block (MCP schemas, appended if present)
  │
  ├─ LLM Execution (against focused context only)
  │   └─ Sees ~200 tokens, not 2000
  │
  └─ Learning Loop
      ├─ Routing correction → ASV learns new vocabulary
      ├─ Instruction edit → intent updated conversationally
      └─ System improves continuously
```

---

## What's Built vs What's Needed

### Built (this branch)
- Intent detection: 69% exact, 98% recall@3, 100% recall@5
- Multi-intent: 88% exact
- Cross-domain: 62% exact (98% recall@3)
- CJK/multilingual: verified
- Auto-learn: LLM key_word extraction
- Token consumption multi-pass
- Cross-provider disambiguation
- Library-level route API with RouteResult (confirmed, ranked, disposition)
- Sub-millisecond latency

### Needed
- **Intent metadata extension**: instructions, guardrails, persona fields per intent
- **Snippet assembler**: takes RouteResult → builds system prompt from intent metadata
- **Conversational builder UI**: LLM-powered page that creates intents from dialogue
- **MCP tool schema storage**: optional per-intent tool definitions

### Not Needed
- State machines or intent chaining (LLM handles branching in instructions)
- Entity extraction (LLM extracts from focused context)
- Vector stores or embeddings (IDF + learning is sufficient)
- Pre-trained models (everything learned from LLM distillation)

---

## The Vision

**Today:** ASV routes queries to intents at 30µs.

**Next:** ASV assembles focused context for the LLM — instructions, tools,
guardrails — all from natural language definitions built through conversation.

**End state:** Anyone can build an AI agent by describing their business.
The agent learns from every interaction. The programming language is English
(or Chinese, Japanese, Korean — ASV is multilingual).

This is intent programming: declarative, learnable, natural language
programs that define AI agent behavior without code.
