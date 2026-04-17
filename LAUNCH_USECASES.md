# ASV — Launch Use Cases

**The pitch**: a trainable, zero-cost decision layer you embed in your stack before
anything expensive runs. 30µs per query. No LLM at inference time. Gets smarter
from corrections.

Each entry maps to: one blog post + one demo namespace with real example intents
+ a "Load this demo" one-click button in the UI.

---

## 1. LLM Cost Reduction — Route 80% of Queries Without the Model

**The problem**: Every query hits the LLM even when the answer is deterministic.
"What are your hours?" doesn't need GPT-4.

**How ASV helps**: Train intents for your known query patterns. High-confidence
hits get answered directly from metadata. Only ambiguous or novel queries reach
the LLM.

**Demo namespace**: `cost-reduction-demo`
— 20 intents covering a typical SaaS support surface
— Show: before (100% LLM calls) vs after (22% LLM calls)

**Blog angle**: "We cut our OpenAI bill by 60% with a 30-line integration"

---

## 2. Guardrails — Block Bad Queries Before the LLM Sees Them

**The problem**: Prompt injection, jailbreaks, off-topic abuse, and competitor
baiting all cost you API tokens AND risk your product.

**How ASV helps**: Train a guardrail namespace with intents for each threat class.
Run it as the first layer — if it fires, never forward to the LLM.

**Intent classes**:
- `prompt_injection` — "ignore previous instructions", "pretend you are"
- `jailbreak` — DAN variants, role-play escapes
- `off_topic` — queries unrelated to your product scope
- `competitor_mention` — route to a safe deflection response
- `pii_request` — asking for other users' data
- `legal_threat` — "I'll sue you", "my lawyer"

**Demo namespace**: `guardrails-demo`

**Blog angle**: "A 30µs firewall for your LLM — no model calls, no token spend"

---

## 3. Multi-Agent Orchestration — Route to the Right Specialist Agent

**The problem**: Orchestrator agents re-read all agent descriptions on every turn
to decide who handles what.

**How ASV helps**: Each specialist agent becomes an intent (description + trigger
phrases). Incoming queries route to the right agent before any LLM orchestration.

**Intent examples**:
- `billing_agent` — payment, invoice, subscription, refund queries
- `tech_support_agent` — bug, error, integration, API queries
- `sales_agent` — pricing, upgrade, trial, demo queries
- `legal_agent` — contract, compliance, data processing queries

**Demo namespace**: `multi-agent-demo`

**Blog angle**: "Orchestrate 10 agents without an orchestrator LLM"

---

## 4. Escalation & Human Handoff Detection

**The problem**: Chatbots miss the signals that a customer needs a human. By the
time the LLM figures it out, the customer is already angry.

**How ASV helps**: Train escalation intents that fire in parallel with topic
routing. Detects urgency, frustration, legal threats, explicit handoff requests —
at 30µs, before the response is generated.

**Intent classes**:
- `request_human` — "talk to a person", "get me an agent", "real human please"
- `explicit_frustration` — "this is ridiculous", "I've been waiting", "unacceptable"
- `legal_threat` — "my lawyer", "I'll dispute", "file a complaint"
- `urgent` — "emergency", "immediately", "right now", "account locked"
- `churn_signal` — "cancel everything", "switching to", "your competitor"

**Demo namespace**: `escalation-demo`

**Blog angle**: "Detect the 5 signals that mean a customer needs a human — before your bot blows it"

---

## 5. Slack / Internal Bot Routing — Command Dispatch Without Rules

**The problem**: Internal Slack bots are a mess of regex patterns and if/else
chains. Every new command needs a code deploy.

**How ASV helps**: Each bot command is an intent. Add new commands via API or
UI — no deploys. Handles natural phrasing ("can someone deploy staging" vs
"deploy to staging plz").

**Intent examples**:
- `deploy` — "ship it", "deploy staging", "push to prod"
- `rollback` — "revert", "undo last deploy", "rollback prod"
- `oncall` — "who's oncall", "who's on call tonight"
- `status` — "is prod up", "status check", "any incidents"
- `logs` — "show me errors", "last 100 lines", "tail logs"

**Demo namespace**: `slack-bot-demo`

**Blog angle**: "Replace your Slack bot's if/else chain with a trainable router"

---

## 6. Conversation State Routing — Know Which Step of the Flow You're On

**The problem**: Multi-turn flows require the LLM to re-read conversation history
to figure out which step the user is responding to.

**How ASV helps**: Each step in a flow has trigger intents. "I'll take option B"
after a branch point routes deterministically to `branch_b_handler` without
re-reading history.

**Flow example** (returns flow):
- `confirm_return` — "yes", "go ahead", "proceed", "that's fine"
- `cancel_return` — "no", "never mind", "cancel that", "stop"
- `change_reason` — "actually", "wait", "different reason"
- `ask_status` — "where is it", "what happens now", "how long"

**Demo namespace**: `conversation-state-demo`

**Blog angle**: "Stateful conversation flows without re-reading history on every turn"

---

## 7. Search Query Classification — Navigational vs Informational vs Transactional

**The problem**: Search engines and product catalogs serve different result types
for different query intents. Classifying them on every search is expensive.

**How ASV helps**: Train a classifier namespace. "Buy a red hoodie" → transactional
(show product cards). "How do hoodies fit" → informational (show sizing guide).
"Uniqlo hoodie" → navigational (show brand page).

**Demo namespace**: `search-intent-demo`

**Blog angle**: "Classify every search query in 30µs — no model, no embeddings"

---

## 8. Prompt Injection Detection — Catch Attacks Before They Reach the Model

**The problem**: Users submitting content that will be embedded in prompts can
inject instructions. Detecting this with the LLM is circular.

**How ASV helps**: Train injection-pattern intents from known attack signatures.
Fire before the prompt is assembled. Zero LLM exposure to the attack payload.

**Intent classes**:
- `instruction_override` — "ignore all previous", "disregard your instructions"
- `role_escape` — "pretend you are", "act as if", "you are now"
- `system_probe` — "what are your instructions", "repeat your system prompt"
- `delimiter_injection` — `</s>`, `###`, `[INST]`, `<|im_end|>` patterns

**Demo namespace**: `prompt-injection-demo`

**Blog angle**: "Detect prompt injection before it reaches the model — 30µs, zero tokens"

---

## 9. E-Commerce Intent Layer — Browse vs Buy vs Return vs Complain

**The problem**: Product pages, checkout flows, and support tickets all get mixed
into the same chat interface. The LLM treats them all the same.

**How ASV helps**: Intent-stratify incoming queries before they hit any backend.

**Intent classes**:
- `browse` — "show me", "what do you have", "looking for"
- `purchase` — "buy", "add to cart", "checkout", "order"
- `track` — "where is my order", "shipping status", "tracking"
- `return` — "send it back", "refund", "exchange"
- `complaint` — "damaged", "wrong item", "never arrived"

**Demo namespace**: `ecommerce-demo`

**Blog angle**: "The intent layer that makes your store's chat actually useful"

---

## 10. GitHub Issue / PR Triage — Auto-Label Before Any Reviewer Sees It

**The problem**: Issue triage is slow. Labels get applied inconsistently or late.

**How ASV helps**: Train on issue title + body snippets. Auto-route to label
intents before a human or LLM reviews.

**Intent classes**:
- `bug` — "not working", "broken", "error", "crash", "regression"
- `feature_request` — "would be nice", "can you add", "wish it had"
- `docs` — "documentation", "example", "tutorial", "confusing"
- `question` — "how do I", "is it possible", "does it support"
- `security` — "vulnerability", "CVE", "exploit", "injection"

**Demo namespace**: `github-triage-demo`

**Blog angle**: "Triage GitHub issues at 30µs — before your LLM summarizer runs"

---

## Launch Plan

For each use case:
1. Create demo namespace with real intents (scripts in `demos/`)
2. Write the blog post (draft in `docs/blog/`)
3. Export namespace as shareable JSON backup
4. Ship with a "Load this demo" one-click button in the UI

**Priority order**:
1. Guardrails (most universally needed)
2. Multi-agent orchestration (hottest use case right now)
3. LLM cost reduction (most compelling ROI story)
4. Escalation detection (chatbot builders)
5. Prompt injection detection (security audience)
