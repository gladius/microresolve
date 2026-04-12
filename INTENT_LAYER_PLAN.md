# Intent Layer — Requirements & Architecture Thinking

*Brainstorm document. Not final. Date: 2026-04-12*

---

# Experiment 1: Concept-Signal Architecture

## How It Works

The system has three layers:

```
Raw query text
      ↓
Layer 1 — Concept Activation
  Scan the query for signals (words/phrases) defined per concept.
  Each concept that fires gets a score.
      ↓
Layer 2 — Intent Scoring
  Each intent has a profile: which concepts matter, and how much.
  Dot product: concept scores × intent profile weights = intent score.
      ↓
Layer 3 — Output
  Return all intents above threshold.
  Multiple intents can score simultaneously → multi-intent is natural.
```

### Concrete walkthrough

```
Setup (LLM does this once):

  Concept: wants_to_stop_service
    signals: [cancel, terminate, discontinue, quit, unsubscribe, close account,
              no longer need, want to leave, stop my, end my]

  Concept: unexpected_charge
    signals: [charged twice, unauthorized charge, overcharged, debited, wrong amount]

  Intent profiles:
    billing:cancel → { wants_to_stop_service: 1.0, unexpected_charge: 0.1 }
    billing:refund → { unexpected_charge: 0.9,     wants_to_stop_service: 0.2 }

Query: "I want to terminate my monthly plan"

  Step 1 — Concept activation:
    scan for signals...
    "terminate" matches wants_to_stop_service → score 1.0
    nothing matches unexpected_charge → score 0.0
    activations = { wants_to_stop_service: 1.0 }

  Step 2 — Intent scoring:
    billing:cancel = 1.0 × 1.0 + 0.0 × 0.1 = 1.0   ← wins
    billing:refund = 1.0 × 0.2 + 0.0 × 0.9 = 0.2

  Output: billing:cancel (score 1.0)
```

### Multi-intent walkthrough

```
Query: "I was charged twice and want to cancel my subscription"

  Concept activation:
    "charged twice" → unexpected_charge: 1.0
    "cancel"        → wants_to_stop_service: 1.0
    activations = { unexpected_charge: 1.0, wants_to_stop_service: 1.0 }

  Intent scoring:
    billing:cancel = 1.0×1.0 + 1.0×0.1 = 1.1
    billing:refund = 1.0×0.2 + 1.0×0.9 = 1.1

  Both score above threshold → return billing:cancel AND billing:refund
  Multi-intent detected correctly.
```

### Continuous learning walkthrough

```
Query: "I need to axe my membership"
  No signals fire → low confidence → sent to LLM

LLM verdict: billing:cancel
LLM says: "axe my" is a signal for wants_to_stop_service

System update:
  wants_to_stop_service.signals.append("axe my")
  Save. Done. Live immediately.

Next query "axe my account" → wants_to_stop_service fires → billing:cancel ✓
```

---

## Is It Multi-Intent?

**Yes, naturally.** Unlike term index (which returns one best match) and unlike
embedding-based systems (which return one centroid winner), this architecture scores
each intent independently based on which concepts fire. Two intents can score high
simultaneously if the query expresses two different semantic clusters.

No special multi-intent logic needed. It falls out of the architecture.

---

## Is This an Existing System?

This is a hybrid of known ideas, but this specific combination is new:

**Similar to:**
- **Amazon Lex / Google Dialogflow**: use intent + slot entities, but activation is
  ML-trained on utterances, not LLM-defined signal lists. Human experts or ML models
  write the rules; here the LLM does it.
- **AIML (old chatbot pattern matching)**: rule-based signal matching, very similar
  mechanics. But static — no LLM distillation, no continuous learning.
- **IBM Watson Assistant**: concept-based dialog management. Similar layer model.
  Requires manual concept authoring by experts.
- **Rasa NLU**: intent + entity extraction via ML. Closest to what this replaces,
  but requires labeled training data (hundreds of examples per intent).

**What is new here:**
1. LLM defines the concept space (no human expert, no labeled data)
2. LLM writes intent profiles (no training)
3. LLM updates signals from production corrections (continuous learning)
4. Signal lists grow organically from real user queries via LLM judgment

**Who uses something like this:**
Enterprise NLU platforms (Nuance, IBM Watson, SAP Conversational AI) use concept-based
routing but require weeks of expert authoring. What makes this different is the LLM
replacing the expert — bootstrap in one API call instead of weeks.

---

## Is It Multilingual?

**Partially yes, with conditions.**

The signal matching is language-agnostic — it just does string matching on whatever
signals the LLM generates. So if the LLM generates multilingual signals:

```
wants_to_stop_service:
  signals: [cancel, terminate,        ← English
            annuler, résilier,         ← French
            kündigen, beenden,         ← German
            取消, 终止,                 ← Chinese
            キャンセル, 解約,           ← Japanese
            إلغاء, إنهاء]              ← Arabic
```

It works for all those languages with zero extra code.

**CJK:** ASV already has character-level tokenization for CJK scripts. Signal matching
on Chinese/Japanese/Korean will work as long as signals are at the character/word level.

**What it does NOT handle automatically:**
- Morphological variants: "kündige" (I cancel) vs "kündigen" (to cancel) in German
- Conjugations: "annulé", "annulons", "annulez" in French (all forms of "annuler")
- Agglutinative languages: Turkish, Finnish — one word can encode a whole phrase

**Fixes:**
1. LLM generates all morphological variants in the signal list
2. Stemming/lemmatization before matching (language-specific, but simple libraries exist)
3. Prefix/suffix matching rules per language

---

## Limitations and How to Address Them

### 1. Signal list is finite — novel expressions are missed
**Problem:** "I wanna peace out of this service" — "peace out" not in signals → miss.
**Fix:** LLM continuous learning. Every miss gets caught, LLM adds the signal.
After first miss, never misses again. Cost: one wrong routing per novel expression.

### 2. Exact string matching is brittle
**Problem:** "cancelling" vs "cancel", "bugs" vs "bug", typos like "cancle".
**Fix A:** Normalize query before matching — lowercase, strip punctuation, basic stemming.
**Fix B:** LLM adds morphological variants to signal list during bootstrap.
**Fix C:** Fuzzy matching with edit distance for short words.

### 3. Negation blindness
**Problem:** "I do NOT want to cancel" → "cancel" fires wants_to_stop_service → wrong.
**Fix:** Negation detection before concept activation (ASV already has this).
Negated signals reduce score instead of increasing it.

### 4. Concept ambiguity
**Problem:** "charge" → could mean billing charge OR "charge forward" (deploy context).
**Fix:** Intent profiles handle this via weights. If "charge" fires unexpected_charge,
billing:refund scores high. deploy:release has zero weight on unexpected_charge.
The profile weights are the disambiguation layer.

### 5. Cold start — first time a domain is set up
**Problem:** Concepts LLM generates may not perfectly capture the domain.
**Fix:** Bootstrap gives a usable starting point. Real queries quickly reveal gaps.
LLM corrections refine it within the first day of production traffic.

### 6. Concept granularity is hard to get right
**Problem:** Too few concepts → intents underdifferentiated. Too many → overlap and noise.
**Guideline:** 20-40 concepts covers most domains. LLM should be asked to stay minimal
and merge concepts that would share most signals.

### 7. No confidence calibration
**Problem:** All signal matches treated equally regardless of signal strength or context.
"I have a cancel button in my UI" → "cancel" fires billing:cancel but shouldn't.
**Fix A:** Longer signal phrases score higher (phrase match > single word match).
**Fix B:** Context window — signal only fires if surrounded by user-intent language,
not technical/UI language.
**Fix C:** LLM adds negative signals ("cancel button", "cancel event") with negative weights.

### 8. Profile weights are static after bootstrap
**Problem:** LLM-written weights may not match real production distribution.
"unexpected_charge: 0.9 for billing:refund" — but maybe your users rarely mention it.
**Fix:** Weight adjustment over time based on which concepts actually predicted correctly.
Simple frequency counting: if wants_to_stop_service fires 95% of the time billing:cancel
is correct, its weight should increase.

---

## Summary

| Dimension | Assessment |
|---|---|
| Multi-intent | ✓ Natural — multiple concepts fire independently |
| Existing precedent | Similar to enterprise NLU but LLM-native. Novel combination. |
| Multilingual | ✓ Yes, if LLM generates multilingual signals. CJK works with existing tokenizer. |
| Novel expressions | △ Misses on first occurrence, learns immediately after |
| Negation | △ Needs explicit negation handling (already in ASV tokenizer) |
| Speed | ✓ String matching + dot product. Microseconds. |
| Interpretable | ✓ You can always see which concepts fired and why |
| No training required | ✓ LLM defines everything. Zero gradient descent. |
| Continuous learning | ✓ Signal additions from LLM corrections, live, no rebuild |

---

---

## What We Are Trying to Build

A system that detects user intent — single or multiple — from natural language queries.
It must be:

- **Semantically meaningful**: "terminate my plan" and "cancel my subscription" should
  route to the same intent, not because they share words, but because they mean the same thing
- **LLM-distilled**: the LLM's understanding of language is the knowledge source.
  We extract that knowledge into a local representation
- **Continuously learning**: as real queries stream in, LLM judgment refines the system.
  It gets smarter from production traffic, not from pre-built datasets
- **CPU-fast at inference**: no LLM call per request. Millisecond routing
- **Multi-intent aware**: one query can express two or more needs simultaneously
- **Not a copy of what exists**: this is not an inverted index with learned weights.
  Not a neural model trained from scratch. Something new

---

## What Is Wrong With What We Have

**Term index (BM25/inverted index):**
- Requires exact word overlap
- "terminate" doesn't match "cancel" unless explicitly added
- No semantic understanding — purely lexical

**Current semantic encoders (MiniEncoder, NanoEncoder):**
- Trained from scratch using word embeddings + triplet loss
- LLM pair distillation bridges vocabulary gaps but requires explicit enumeration
- Every synonym must be explicitly told to the model via a pair
- Fundamentally: the model is still matching word vectors, not meaning
- At this scale (6-8 phrases/intent) the architecture is overkill for what it delivers

**The real gap:**
The LLM *already knows* that terminate = cancel in billing context.
We are not using that knowledge efficiently.
We are making the LLM generate word pairs and then training a model on those pairs —
an indirect, lossy path from LLM knowledge to routing logic.

---

## The Core Requirement

**LLM should be able to say what to learn and how.**
Not generate training data. Not label pairs. Actually define the semantic structure.

The representation we build should be something the LLM can write into directly,
and that a fast local system can use without calling the LLM again.

---

## A New Direction: Concept-Based Intent Representation

Instead of word vectors, represent intents through **semantic concepts**.

### What is a concept?

A concept is a named semantic unit that the LLM defines for a domain.
Examples for a SaaS support domain:

```
wants_to_stop_service
financial_dispute  
unexpected_charge
software_failure
ship_new_code
undo_deployment
request_new_functionality
```

These are not words. They are meanings. The LLM defines them.

### Intent = a profile over concepts

```
billing:cancel  = { wants_to_stop_service: 1.0, financial_dispute: 0.0, ... }
billing:refund  = { unexpected_charge: 0.9, wants_to_stop_service: 0.2, ... }
support:bug     = { software_failure: 1.0, ship_new_code: 0.0, ... }
deploy:rollback = { undo_deployment: 1.0, software_failure: 0.4, ... }
```

The LLM writes these profiles. Not gradient descent. Not training. The LLM knows
what billing:cancel means — it writes the concept weights directly.

### Query → concept activation → intent match

At inference:
1. Query arrives
2. Which concepts does this query activate? (fast local lookup)
3. Cosine/dot product between activated concepts and intent profiles
4. Highest scoring intent(s) = detected intent(s)

Multi-intent is natural: a query activating both `wants_to_stop_service` AND
`software_failure` returns billing:cancel AND support:bug simultaneously.

### How queries map to concepts (the distillation step)

The LLM defines, per concept, a set of signals:
- Words and phrases that activate this concept
- Semantic patterns (not just exact words)

```
wants_to_stop_service:
  signals: [cancel, terminate, discontinue, quit, unsubscribe, close account,
            no longer need, want to leave, stop my, end my, get out of]

software_failure:
  signals: [crash, broken, error, bug, not working, exception, failure,
            keeps failing, stopped working, went down, is down]
```

The LLM writes this once per concept. Matching is simple string/pattern check.
No training. No vectors. Pure LLM knowledge encoded as signal lists.

### Continuous learning: how the LLM updates the system

When a query routes wrong or with low confidence:

1. LLM reviews the query
2. LLM says: correct intent is X
3. LLM also says: which concepts were active in this query, and were any missing?
4. System updates: add new signals to the relevant concept, or adjust concept weights for the intent

Example:
```
Query: "I need to axe my membership"
Routed to: support:bug (wrong)
LLM verdict: billing:cancel
LLM says: "axe my membership" activates wants_to_stop_service

Update: add "axe my" to wants_to_stop_service signals
Done. Immediately. No retraining.
```

Next time "axe my plan" appears: wants_to_stop_service fires → billing:cancel.

---

## Properties of This Architecture

| Property | Status |
|---|---|
| No neural training from scratch | ✓ |
| LLM directly writes the knowledge | ✓ |
| CPU fast at inference | ✓ (string matching + dot product) |
| Multi-intent natural | ✓ (multiple concepts can fire) |
| Continuous learning from production | ✓ (add signals from LLM corrections) |
| Generalizes to unseen vocabulary | ✓ (LLM adds it when first seen) |
| Interpretable | ✓ (you can see which concepts fired) |
| Domain-agnostic | ✓ (LLM defines concepts per domain) |

---

## What Needs to Be Built

### 1. Concept registry
- Store: concept name → list of signals (words/phrases)
- Store: intent → concept weight profile
- LLM populates this via a one-time setup call per namespace

### 2. Concept activation engine
- Query → tokenize → match signals → concept activation scores
- Fast: simple string matching against signal lists
- Returns: `{concept: score}` map for any query

### 3. Intent scoring
- Dot product: concept activation scores × intent concept profile
- Returns ranked intent list
- Multi-intent: return all intents above threshold or within gap

### 4. LLM judgment pipeline
- When routing is low-confidence or wrong
- LLM confirms/corrects + names active concepts
- System appends new signals to concept registry
- No retraining, no rebuilding — live update

### 5. Bootstrap call
- For a new namespace/domain:
  `POST /api/concepts/bootstrap` → sends intent list to LLM
  LLM returns: concept definitions + intent profiles + initial signal lists
  System is immediately usable

---

## Open Questions

1. How many concepts per domain? (hypothesis: 20-50 is enough for most domains)
2. Should concepts be shared across domains or per-namespace?
3. How to handle concept ambiguity? ("charge" activates both unexpected_charge and ship_new_code)
4. Confidence scoring: how to express uncertainty when concepts partially match?
5. Can concept weights be soft (learned from LLM corrections over time)?
6. Should this replace or sit alongside the existing term index?

---

## What We Are NOT Building

- No word embeddings trained from scratch
- No attention mechanism
- No triplet loss, no gradient descent
- No pre-trained embedding model dependency
- No hard-coded synonym lists

The LLM defines the semantics. The system executes them.

---

## Relationship to Existing Work

The term index continues to handle exact-match routing (it's already good at that).
This concept layer sits above it as a semantic understanding layer.
Together:

```
Query
  ↓
Concept activation (semantic, LLM-distilled)  ← new
  + Term index (lexical, fast)                ← existing
  ↓
Combined score → intent
```

Or: replace term index entirely if concept layer proves sufficient.

---

*Next step: validate the concept architecture on a small example before any code.*
