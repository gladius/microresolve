# Intent Programming — Design Notes

## Core Concept

Intents are functions. Instructions are the function body. ASV is the entry
point detector. The LLM is the CPU that executes the instructions.

ASV fires at **entry points and intent transitions only**. Within an intent's
flow, the LLM drives the conversation using the instructions + history.

## Conversation Flow

```
User: "I need to cancel my appointment"
  → ASV fires: cancel_appointment (entry point)
  → Instructions loaded + assembled as system prompt
  → LLM: "When is your appointment?"

User: "Tomorrow at 2pm"
  → ASV: no new intent detected
  → LLM continues with SAME instructions + history
  → LLM: "That's within 24 hours, $25 fee. Want to reschedule?"

User: "Yeah let's reschedule"
  → ASV fires: book_appointment (intent TRANSITION detected)
  → NEW instructions loaded + assembled
  → Runtime context carries forward: "rescheduling from cancelled appt"
  → LLM: "What day works for you?"

User: "Saturday morning"
  → ASV: no new intent
  → LLM continues with booking instructions + history
  → LLM: "10am or 11am?"
```

## When ASV Fires vs When LLM Drives

| Situation | Who handles | Why |
|-----------|-------------|-----|
| First message | ASV | Entry point — detect what the user wants |
| Follow-up within same topic | LLM | Instructions already loaded, history has context |
| User switches topic | ASV | New intent detected → new instructions |
| Ambiguous follow-up | LLM first, ASV if LLM unsure | LLM has context, ASV is fallback |

## Intent Structure

```
intent: cancel_appointment
  phrases: ["cancel my appointment", "I need to cancel", ...]
  instructions: |
    Handle cancellation. Ask when their appointment is.
    If more than 24 hours away: cancel with no fee.
    If within 24 hours: explain $25 fee.
    Offer to reschedule before confirming cancel.
    If they accept rescheduling: proceed to book_appointment flow.
  guardrails: ["Never waive fee without manager", "No free services"]
  persona: "Professional but understanding"
```

The instructions contain the COMPLETE flow logic including transitions
("proceed to book_appointment flow"). No state machine. No chaining config.

## Runtime Context

Each conversation accumulates context as it moves through intents:

```
Turn 1-3: cancel_appointment
  → Context: {customer wants to cancel, appointment is tomorrow 2pm, 
              offered reschedule, customer accepted}

Turn 4+: book_appointment  
  → System prompt: booking instructions
  → Runtime context from previous intent: "Customer is rescheduling a 
    cancelled appointment for tomorrow 2pm. Previous appointment had 
    a $25 cancellation fee that was waived via reschedule."
```

The runtime context is built from conversation history. The LLM reads
previous turns and understands why the customer is now booking.

## API Design

```
POST /api/execute
{
  "query": "I need to cancel",
  "history": [],                    // empty = new conversation
  "session_id": "abc123"            // optional, for context tracking
}

Response:
{
  "response": "When is your appointment?",
  "routing": {
    "intent": "cancel_appointment",
    "disposition": "confident",
    "is_transition": false           // true when intent changed
  },
  "assembled_prompt": "Handle cancellation..."
}
```

Follow-up:
```
POST /api/execute
{
  "query": "Tomorrow at 2pm",
  "history": [
    {"role": "user", "content": "I need to cancel"},
    {"role": "assistant", "content": "When is your appointment?"}
  ],
  "session_id": "abc123"
}
```

Server logic:
1. Route query → if new intent detected → assemble new prompt
2. If no intent detected + history exists → reuse last intent's prompt
3. Build messages: [system_prompt, ...history, user_query]
4. Call LLM
5. Return response + routing info

## What's Built

- [x] Intent detection with IDF scoring + token consumption
- [x] Metadata storage (instructions, guardrails, persona per intent)
- [x] Snippet assembly (instructions + guardrails + persona → system prompt)
- [x] /api/execute endpoint (single-turn working)
- [x] LLM call with messages array
- [ ] Follow-up handling (reuse last intent when no new intent detected)
- [ ] Intent transition detection (detect when topic changes mid-conversation)
- [ ] Runtime context accumulation
- [ ] Conversational builder UI
