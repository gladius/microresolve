# Emergent Context Discovery in Sparse Vector Routing

## Experiment Date: 2026-03-07

## Setup
- 36 intents (20 action, 16 context) with ~7 seed phrases each
- 100 natural multi-intent customer service queries
- Zero hardcoded relationships between intents
- Co-occurrence tracked automatically from route_multi results

## What We Found

When routing multi-intent queries, ASV's sparse vectors create weak secondary
matches through shared vocabulary. Tracking which intents fire together reveals
semantic relationships that were never programmed:

| Action Intent    | Discovered Context Partners              | Semantically Valid? |
|------------------|------------------------------------------|---------------------|
| refund           | check_balance(21%), warranty_info(13%)   | Yes                 |
| cancel_order     | track_order(28%), order_history(21%)     | Yes                 |
| billing_issue    | payment_history(25%), check_balance(17%) | Yes                 |
| apply_coupon     | price_check(33%)                         | Yes                 |
| change_plan      | subscription_status(20%)                 | Yes                 |
| reorder          | product_availability(18%)                | Yes                 |
| report_fraud     | account_status(15%), order_history(15%)  | Yes                 |
| update_address   | track_order(28%)                         | Yes                 |

100% of top-1 projected context relationships are semantically valid.

## Projected Context Output

Single-intent queries that return ONLY the action intent now also return
inferred context based on co-occurrence patterns:

```
Query: "I want a refund"
  → refund (4.77, action)
  ⟐ projected: check_balance(21%), warranty_info(13%)

Query: "cancel my order"
  → cancel_order (5.46, action)
  ⟐ projected: track_order(28%), order_history(21%)

Query: "there's a billing error"
  → billing_issue (6.33, action)
  ⟐ projected: payment_history(25%), check_balance(17%), transaction_details(17%)
```

These context intents do NOT appear in the routing results — they are inferred
purely from historical co-occurrence patterns.

## Mechanism

1. ASV builds sparse vectors from seed phrases (weighted term frequencies)
2. Related intents share vocabulary: "refund" seeds contain "money," "account,"
   "charge" — words that also appear in check_balance and payment_history seeds
3. Multi-intent queries activate the primary intent strongly and related intents weakly
4. Co-occurrence tracking captures these statistical co-activation patterns
5. The co-occurrence matrix forms a weighted graph of intent relationships
6. Projected context uses this graph to infer context for single-intent queries

## Important Caveats

- Experiment used simulated (not real user) queries — patterns need real-world validation
- Term overlap is a known property of bag-of-words models
- What's novel: using routing co-occurrence as a self-organizing context graph
- Strength values will shift as more queries accumulate — needs stability analysis
- Minimum query volume needed before projections are reliable (cold start problem)
