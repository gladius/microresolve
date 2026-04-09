# Simulation Plan: Real Enterprise Data

## The Problem

Current simulation uses 30 hardcoded queries. That's not realistic. We need:
- Real multi-turn customer sessions (not single queries)
- Enough volume for patterns to emerge (workflows, escalations, temporal flows)
- Multiple enterprise use cases
- The simulation must use the real ASV library to build real data

## Dataset: ABCD (Action-Based Conversations Dataset)

**Source:** github.com/asappresearch/abcd  
**Size:** 10,042 human-to-human customer service dialogues  
**License:** MIT  
**Location:** `/tmp/abcd_v1.1.json` (116MB, downloaded)

### What it contains

**10 flows (high-level intents):**
```
storewide_query:        872 conversations
product_defect:         863
purchase_dispute:       858
account_access:         847
single_item_query:      840
order_issue:            831
troubleshoot_site:      819
shipping_issue:         814
subscription_inquiry:   718
manage_account:         572
```

**96 subflows (specific actions):**
```
account_access.recover_username:           295
account_access.reset_2fa:                  286
account_access.recover_password:           266
troubleshoot_site.credit_card:             219
shipping_issue.status:                     216
troubleshoot_site.shopping_cart:           210
shipping_issue.missing:                    208
product_defect.return_size:                151
product_defect.refund_update:              146
subscription_inquiry.manage_dispute_bill:  145
...
```

**Each conversation is a full session:**
- 10-30 turns per conversation
- Customer and agent messages
- Scenario with customer info, order details, policy constraints
- Real language: frustration, confusion, multi-step requests

**Example conversation:**
```
Flow: product_defect.return_size (29 turns)
  Customer: Hi! I need to return an item, can you help me with that?
  Customer: Crystal Minh
  Customer: I got the wrong size.
  Customer: Username: cminh730
  ...
```

### Why this is perfect for ASV

1. **Session-bound** — each conversation IS a customer session. We can track
   intent sequences across turns within a session.

2. **Real language** — not synthetic. Customers say "I got the wrong size"
   not "initiate return due to size mismatch."

3. **Workflows emerge** — within a single session:
   - product_defect → return → refund (3-step workflow)
   - account_access → recover_password → manage_account (escalation)
   - shipping_issue.status → shipping_issue.missing → contact_human (escalation)

4. **Cross-intent patterns** — some conversations touch multiple flows:
   - Customer starts with shipping_issue, discovers product_defect, wants refund
   - Customer has account_access issue AND order_issue in same session

## Simulation Architecture

### Step 1: Map ABCD flows to ASV intents

ABCD has 10 flows + 96 subflows. We map them to ASV intents:

```
ABCD flow                    → ASV intent(s)
─────────────────────────────────────────────
product_defect.return_*      → return_item, product_issue
product_defect.refund_*      → refund, product_issue
shipping_issue.status        → track_order
shipping_issue.missing       → track_order, contact_human
shipping_issue.cost          → shipping_info
account_access.*             → account_access, password_reset
purchase_dispute.*           → billing_issue, contact_human
order_issue.*                → order_issue, change_order
troubleshoot_site.*          → technical_support
subscription_inquiry.*       → subscription, billing_issue
manage_account.*             → account_settings
storewide_query.*            → product_inquiry, store_info
single_item_query.*          → product_inquiry
```

### Step 2: Create ASV intents from ABCD data

Extract seed phrases from ABCD customer messages per flow:
- For each flow, collect all customer messages from first turns
- Use the most common opening messages as seeds
- Create 15-25 intents covering all 10 flows + key subflows

### Step 3: Run simulation

For each ABCD conversation:
1. Extract customer messages (skip agent messages)
2. Route each customer message through ASV's `route_multi()`
3. Record co-occurrence and intent sequences (this builds the analytics data)
4. Track: which intents fire, in what order, which escalate

This processes 8,000+ real customer sessions through ASV, building:
- Co-occurrence matrix from real usage
- Temporal ordering from real conversation flows
- Workflow clusters from real session patterns
- Escalation patterns from real customer journeys

### Step 4: Dashboard shows real patterns

After simulation, the dashboard displays intelligence discovered from
10,000 real customer conversations — not 30 hardcoded queries.

## Implementation

### Option A: Server-side simulation endpoint

```
POST /api/simulate/abcd
{
  "dataset_path": "/tmp/abcd_v1.1.json",
  "max_conversations": 1000,
  "split": "train"
}
```

Server loads ABCD, creates intents from the data, runs all conversations
through route_multi, returns summary stats. Dashboard refreshes automatically.

### Option B: CLI simulation binary

```bash
cargo run --bin simulate -- --dataset /tmp/abcd_v1.1.json --conversations 1000
```

Standalone binary that loads dataset, creates router, runs simulation,
saves state to JSON. Server loads the JSON and dashboard shows results.

### Option C: Python simulation script

```bash
python examples/python/simulate_abcd.py --dataset /tmp/abcd_v1.1.json --conversations 1000
```

Uses the Python library directly. Most flexible for data manipulation.
Outputs results to server via API calls.

**Recommended: Option A** (server endpoint) — keeps everything in one system,
dashboard shows results in real-time as simulation runs, no external scripts needed.

## Expected Outcomes

After running 8,000+ ABCD conversations through ASV:

**Workflows discovered:**
- product_defect → return_item → refund (most common)
- account_access → password_reset → account_settings
- shipping_issue → track_order → contact_human (escalation)
- purchase_dispute → billing_issue → contact_human (escalation)

**Temporal ordering:**
- return_item → refund (85% of the time)
- billing_issue → contact_human (60%)
- product_inquiry → order (40%)

**Escalation patterns:**
- Any issue lasting >5 turns → contact_human (high frequency)
- billing_issue → purchase_dispute → contact_human (3-step escalation)

**Projected context:**
- When refund fires → return_item co-occurs 82%
- When track_order fires → order_info co-occurs 70%

## Other Datasets for Future Use Cases

| Use Case | Dataset | Size | Available |
|----------|---------|------|-----------|
| Banking | Bitext Banking | 27K queries | HuggingFace |
| Telecom | Bitext Telco | 27K queries | HuggingFace |
| E-commerce | Bitext E-commerce | 27K queries | HuggingFace |
| Insurance | DSTC11 Track 2 | 67K utterances | HuggingFace |
| Travel/Hospitality | MultiWOZ | 10K dialogues | HuggingFace |
| Real messy language | Twitter Support | 3M tweets | Kaggle |

These can be added later for domain-specific simulations.
