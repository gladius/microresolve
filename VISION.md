# ASV Router — Vision: Conversational Decision Engine

## The Core Insight

ASV is not an LLM alternative. It doesn't generate text. It doesn't reason. It doesn't handle novel situations.

What it IS: **a decision engine that eliminates the LLM from the 80% of conversations that are repetitive.**

## The Economics

A typical customer support operation:
- 10,000 conversations/day
- 80% are the same 50 intents (cancel, refund, track, reset password...)
- 20% are genuinely novel / complex

Today: **every turn** goes through an LLM. At $0.01/turn = $36,500/year.

With ASV: **80% route through ASV** (cost: $0). Only 20% hit the LLM. Cost drops to $7,300/year. **$29,000/year saved per customer.**

## The Full Architecture

```
┌─────────────────────────────────────────────────┐
│            Conversational Decision Engine        │
│                                                  │
│  Input Router (ASV)                              │
│    "what user wants" → intent + confidence       │
│                                                  │
│  Sentiment Layer                                 │
│    "how user feels" → angry/neutral/happy        │
│    (adjust response tone, escalation priority)   │
│                                                  │
│  Response Selector (ASV again, output routing)   │
│    intent + sentiment + context → template       │
│    "cancel_order + angry" → apologetic template  │
│    "cancel_order + neutral" → standard template  │
│                                                  │
│  LLM Fallback (only when ASV confidence < 0.6)  │
│    Novel questions, complex reasoning, edge cases│
│                                                  │
│  Learning Loop                                   │
│    Every LLM fallback → candidate for learning   │
│    Human reviews → ASV learns → fewer fallbacks  │
│    System gets cheaper over time                 │
└─────────────────────────────────────────────────┘
```

## Self-Improving System

The more it runs, the less it needs the LLM. Every LLM fallback is a candidate for ASV to learn. Human reviews the LLM's response, approves it, and ASV learns the mapping. Next time, ASV handles it directly.

This creates a **self-improving conversation handler that asymptotically reduces LLM dependency**.

## What Already Exists (Prior Art)

| System | What it does | What it lacks |
|--------|-------------|---------------|
| Rasa | Intent routing + response selection | Needs ML training, can't learn incrementally |
| Amazon Lex | Intent + slot filling | Cloud-only, not learnable from production |
| Dialogflow | Intent classification | Google Cloud dependency, no on-device |
| Rule engines (Drools) | Decision trees | No natural language, no learning |

**What none of them do**: incremental learning from production without retraining. That's the edge.

## What This is NOT

- Not a chatbot builder (routes, doesn't generate)
- Not an LLM replacement (handles the boring 80%, LLM handles the interesting 20%)
- Not an NLU system (intent routing, not entity extraction or slot filling)

## Future Layers

1. **Simulate enough phrases** → covers the semantic gap
2. **Sentiment system** → understands emotional tone
3. **Multi-language** → works globally
4. **Input routing** → understands what user wants
5. **Output routing** → selects the right response template

Combined: a system that handles 80% of conversations without an LLM, and gets smarter every day.

## Positioning

**Don't say**: "LLM alternative" or "replaces AI"
**Do say**: "Stop paying LLM for work that doesn't need intelligence"

The pitch: lightweight, learnable routing layer that sits in front of any LLM system and saves 60-80% of LLM API costs.
