# MicroResolve — Launch Strategy Notes

Working doc for positioning, messaging, and launch sequencing decisions.
Feeds into `LAUNCH_CHECKLIST.md` and all public content.

---

## 1. The Hook (working version — NOT the "HashMap beats GPT-4" line)

**Problem with "A HashMap beats GPT-4":** reads as marketing ploy to any
technical reader. Triggers the "cherry-picked benchmark" pattern-match.
HN regulars have seen 1,000 "X beats ChatGPT" posts.

**Better hook options — factual, origin-story, or question-framed:**

- **(preferred)** *"I didn't want to call GPT-4 for every 'cancel my order' —
  so I built a ~30μs decision layer."* — origin story, honest.
- *"Route 150 intents in 30μs. Learn from every query. No GPU, no embeddings."*
  — pure spec, zero claim.
- *"What if the first 80% of your AI agent's decisions didn't need an LLM?"*
  — question framing, invites engagement.
- *"A ~30μs decision layer that sits in front of your LLM — intent routing,
  PII, safety, tool selection."* — descriptive, no comparison.

**TODO:** pick one, use it as the HN title, Twitter thread opener, LinkedIn
post opener, and blog post lede. **Same line every surface.**

---

## 2. Category Name

### Rejected candidates and why

| Name | Rejected because |
|---|---|
| "pre-LLM decision middleware" | "Middleware" = boring plumbing; too long |
| "Decision cache" | "Cache" evokes Redis / exact-match KV storage. We're LEARNED + GENERALIZING + ADAPTING — mechanically not a cache |
| "LLM prefilter" | Describes, doesn't brand. No category creation |
| "Semantic switch" | "Semantic" is inaccurate — we're lexical |
| "Lexical gateway" | Honest but not sexy |

### Working candidate: **"Reflex layer"** ⭐

Parallel to Kahneman System 1 / System 2:
- **LLM = System 2** — slow, deliberate, expensive reasoning
- **MicroResolve = System 1** — fast, learned, instinctive pattern-matching

Why this works:
- AI crowd recognizes the Kahneman framing immediately
- "Reflex" captures: fast + learned + pre-thought (not cached)
- Doesn't pull anyone toward the wrong mental model (unlike "cache")
- CTO-speakable: *"we put a reflex layer in front of Claude"*
- Creates a new category MicroResolve can own

### Positioning sentence

> *MicroResolve is the **reflex layer** for AI applications. Classify intents,
> detect PII, filter jailbreaks, pre-select tools — all in ~30μs, the way a
> trained reflex fires before your brain thinks. Your LLM only handles what
> actually needs thinking about.*

### Still-open alternatives (revisit if "reflex" doesn't stick in real use)

| Name | Note |
|---|---|
| "Distilled decision layer" | Technically accurate (knowledge distillation is a real term) |
| "Intent reflex" | Punchy but limits us to intent routing |
| "Decision predictor" | CPU branch-predictor metaphor; niche |

---

## 2b. The Deeper Pattern — Reflex Substrate, Not Just Router

**Key insight from Gladius (2026-04-22):** the listed use cases (intent, PII,
safety, MCP prefiltering) are shallow extractions. The actual product is
deeper.

### What MicroResolve really is

> **A substrate for distilling any LLM classification decision into
> microsecond pattern-matching.**

**Anything that has:**
- Bounded output space (not generative)
- Repeated per query
- A learnable pattern
- A ground-truth signal (LLM or human)

…can be compiled into MicroResolve.

### Positioning shift

- "Intent router" → one feature
- "Reflex substrate" → a category every AI team has a reason to care about

### New high-impact use cases (not yet explored)

**Tier 1 — viral potential**

1. **LLM output classification** — classify the response before shipping
   (quality / safety / relevance). $100M+ pain point right now (Guardrails
   AI, NeMo-Guardrails). Currently done with another LLM call (expensive).
   Reflex layer = 100× cost reduction.

2. **Model cascade routing** — "does this query need Opus, or will Haiku
   suffice?" Learn from LLM correctness history. Saves ~80% LLM cost at
   scale.

3. **Context relevance scoring for RAG** — given query + 50 retrieved chunks,
   classify which are actually relevant before stuffing the LLM.
   Currently done with cross-encoder reranking (10-50ms). Reflex = 30μs.

4. **Multi-agent dispatch** — "which specialist agent handles this?"
   Unsolved problem in agent swarms. Reflex-learned routing > hand-coded
   rules.

**Tier 2 — solid but less viral**

5. Prompt template selection
6. Semantic query deduplication (cache key generation)
7. Urgency / priority classification (tickets, emails)
8. Tone / sentiment (before or after LLM call)
9. Learned compliance / policy enforcement
10. Language detection at microsecond speed

**Tier 3 — niche**

11. Intent evolution tracking (multi-turn sessions)
12. Cross-lingual intent unification
13. Per-namespace learned guardrails

### Why this matters for launch

- Agent teams → care about #4 (dispatch)
- RAG teams → care about #3 (context scoring)
- Every API-cost-conscious team → #2 (model cascade)
- Safety teams → #1 (output classification)

**Every one of those audiences = potential stars + consulting leads.**

### Proposed new architecture layers (L4–L7)

All microsecond-budget-preserving (O(1) or O(k) lookups).

- **L4 — Temporal reflex.** Session-aware routing. "Given previous 2 queries
  were billing, this ambiguous query probably continues billing." Learned
  from session flows; session state vector.

- **L5 — Confidence calibration.** Learned per-intent calibration.
  "Intent A's top score needs ≥ 0.8 to be reliable; intent B reliable at
  0.4." Makes 80/20 split automatic and tunable — decides when to fall
  through to LLM.

- **L6 — Meta-learning / drift detection.** Detects when accuracy drops on
  a namespace → flags for re-distillation from LLM. Closes the loop:
  LLM → reflex → user → correction → re-distillation.

- **L7 — Cross-namespace transfer (opt-in).** Learned patterns exportable
  and reusable. Seed packs become bidirectional. Federated-learning
  effect, privacy-preserving.

These aren't required for v0.1.0 launch but represent the **roadmap story**
that moves MicroResolve from "library" to "platform."

---

## 2c. The "What If" — Beyond Classification

**Gladius pushback (2026-04-22):** earlier layers L4–L7 were all analytical
(scoring/calibration/drift). What if MicroResolve does more than classify?
What if it *acts*?

### The framing shift

**Today's story:** MicroResolve is the reflex layer. Fires classifications
at microsecond speed.

**Deeper story:** MicroResolve is the **autonomic nervous system** for AI
agents. Your LLM is the cortex — slow, conscious, deliberate. MicroResolve
is everything the cortex doesn't handle directly:

- **Reflex arcs** — classifications (what we have today)
- **Muscle memory** — learned responses for repeated queries (new)
- **Autonomic guards** — preventing irreversible mistakes (new)
- **Working memory** — learned facts distilled from LLM output (new)
- **Motor prediction** — speculative execution of likely outputs (new)

### Five new capability layers (beyond classification)

**L8 — Response synthesis.** LLM teaches reflex static responses for
repeated queries. Reflex serves "What's your name?" and "What can you do?"
directly at 30μs. Only novel queries fall through to the LLM.

**L9 — Tool-call validator.** Learned veto reflexes. "Never delete
user 0", "Never charge > $10k". Enforced before execution at microsecond
speed.

**L10 — Context compressor.** Per-query relevance reflex on conversation
history. Drop irrelevant messages before LLM invocation. 50–90% token
reduction, 30μs × N messages.

**L11 — Learned memory.** LLM says "I'll remember you prefer dark mode"
→ reflex auto-extracts the fact → future sessions inject it into the
system prompt. Auto-distilled long-term memory, CPU-only, no vector DB.

**L12 — Speculative execution.** Reflex predicts what LLM would say;
when confidence is very high, ship speculatively while LLM runs as
verifier. Same pattern CPUs have used since the 1960s (branch
prediction) applied to LLM calls.

### Why this matters for positioning

- Library competitors: Presidio, Rasa, Semantic Router, LangChain
- Platform competitors (if we take this framing): Guardrails AI,
  Letta / Mem0, LangGraph (partial), custom agent infra

**"Reflex library"** is a $10k consulting engagement.
**"Autonomic nervous system for agents"** is a $100k platform consulting engagement.

Same product, 10× wider audience, 10× higher deal size.

### Day-to-day LLM problems this addresses

| Problem | Capability |
|---|---|
| Hallucinated facts in output | L3 / output classification |
| Tool call irreversibility | L9 (validator) |
| Context window waste (90% irrelevant) | L10 (compressor) |
| Lost session state | L11 (memory) |
| High latency per LLM call | L12 (speculative exec) |
| LLM cost at scale | L8 (response synthesis) |
| Wrong model for cheap query | Tier 1 #2 (cascade routing) |
| Bad RAG chunk ranking | Tier 1 #3 (relevance scoring) |
| Wrong specialist agent chosen | Tier 1 #4 (dispatch) |
| Runaway agent loops | (out of scope — separate project: `loopdetector`) |

### Launch implication

- v0.1.0 ships with L0–L3 (current lexical + intent + entity layers)
- Blog post / roadmap page describes L4–L12 as the trajectory
- This turns MicroResolve from "library" → "platform story in progress"
- CTOs evaluate trajectory as much as current state — the story matters

---

## 2d. Agent Loop Detector — Separated

Originally scoped as an L9 layer in MicroResolve. Extracted to a standalone
project at `~/Workspace/loopdetector` — it's a different algorithm (ring
buffer + cycle detection on tool-call traces) that doesn't share primitives
with the routing engine beyond a rough interest in sparse vectors. Keeping
it separate preserves MicroResolve's "one engine, many lenses" story and
lets the detector evolve on its own cadence.

---

## 3. Use Case Expansion

**Current 4 (keep):**
1. Intent routing
2. PII detection / masking
3. Prompt injection / jailbreak filtering
4. MCP tool prefiltering

**Add 3 high-viral-potential stories (blog posts staggered post-launch):**

| # | Use Case | Why Viral | Who Shares It |
|---|---|---|---|
| 5 | **AI agent tool selection** | Hits 2026 agent hype directly | Every agent-framework author |
| 6 | **RAG retriever routing** | Solves real RAG pain (which index to hit) | RAG builders — huge audience |
| 7 | **Jailbreak detection** (standalone angle) | Safety is a 2026 top concern | Safety researchers, CTOs |

**Rest — list briefly, no dedicated story:**
- Chatbot handoff (LLM → human decision)
- Log / alert classification
- Command palette for apps
- Content moderation prefilter
- Smart home / voice intent
- Customer support ticket routing

---

## 4. Launch Day Mechanics — One Splash, Then Slow Compound

**Mental model:** ~72 hours of HN frontpage + Twitter wave → ~70% of eventual
attention. If splash flops, rarely get a second chance — HN mods suppress
reposts, Twitter fatigue kicks in. **Treat launch as one-shot. Over-prepare.**

### Coordinated multi-channel launch day (Tuesday, 9am ET)

| Channel | Time | Angle |
|---|---|---|
| Show HN | 9:00 ET | Main post, origin story framing |
| Twitter/X thread | 9:05 ET | 10 tweets w/ benchmark chart image |
| LinkedIn post | 9:10 ET | Same story, professional tone + "available for consulting" |
| Reddit r/LocalLLaMA | 10:00 ET | "No-GPU intent router" angle |
| Reddit r/rust | 11:00 ET | Technical deep-dive angle |
| Reddit r/MachineLearning | 12:00 ET | "What if we cached LLM decisions?" angle |
| dev.to cross-post | Same day | SEO surface |

### First-day make-or-break: comment response
- Answer every HN comment within 10 min for the first 6 hours
- Hostile comments especially — concede what's true, correct what's not,
  never get defensive
- Single biggest thing most launches get wrong

### Second-wave artifact (within 72 hours)
- Publish first blog post ("Why I don't use embeddings for intent routing"
  or similar) as HN decays to keep momentum and capture click-throughs

---

## 5. Pre-Launch Runway (2 weeks before)

### Influencer pre-seeding — genuine engagement, NOT pitching
- Simon Willison (@simonw)
- Hamel Husain (@HamelHusain)
- Swyx (@swyx)
- Thorsten Ball (@thorstenball)
- Jeremy Howard (@jeremyphoward)
- People starring `llama.cpp`, `candle`, `outlines`, `guidance`
- Rust-AI crowd (Burn, Candle maintainers)

Reply substantively to their posts about LLM infra. Do not pitch. Be present
so that at launch time, a DM saying "I built this, might interest you" lands
with someone who recognises your handle. **1 influencer retweet = 500–5000
stars.**

### Essential launch assets to build
1. **Live hosted demo** at `<domain>/playground` — single highest-ROI
   launch asset. Server-side (browser → hosted MicroResolve instance); the
   routing runs on your server, not in the browser. 30s of "oh shit that's
   fast" beats 100 lines of README. Without it: ceiling ~2k stars. With
   it: 10–15k range plausible.
2. **Benchmark chart image** (latency + accuracy + cost vs GPT-4 / BERT /
   embedding routers). Share natively on Twitter/LinkedIn (not as link —
   link posts get ~10× less reach).
3. **Honest limitations page** — loud about cold-start accuracy, no OOV
   semantic, etc. HN rewards honesty; pretending kills credibility.

---

## 6. 20k Stars — Honest Assessment

- **Floor (zero effort):** 300–800 stars
- **Good launch:** 2–5k stars
- **Hit territory (20k+):** requires everything to click — hook, category,
  demo, pre-seeding, coordinated multi-channel, fast first-day responses,
  timely second-wave content
- **Realistic target:** aim for 5–10k at launch → compound to 20k over
  6–12 months via newsletter features, YouTube reviews, and incremental
  blog posts

---

## 7. Consulting Conversion

- HN/Twitter/Reddit = stars + credibility
- **LinkedIn = the consulting lead channel**
- Every LinkedIn post linked from README should explicitly say:
  *"Open source. Available for consulting engagements — DM me."*
- Goal: not to be cool on LinkedIn; to be *found by CTOs next month*
  when they need someone.

---

## Open Questions / TODOs

- [ ] Commit to "decision cache" or find something stronger
- [ ] Pick and lock the hook line
- [ ] Stand up the hosted demo playground
- [ ] Draft the 3 use-case blog posts (agent tools / RAG routing / jailbreak)
- [ ] Rewrite README Commercial Support section (current draft is too generic)
- [ ] Buy domain + set up contact email
- [ ] Pre-seed the influencer list 2 weeks before launch
- [ ] Generate the benchmark chart image
