# ASV Experiment Results

Scenarios: 30, Turns: 138

---

## BASELINE (route_multi, threshold=0.3)

### Baseline Results

| Metric | Value |
|---|---|
| Total turns | 138 |
| Pass (exact) | 14 (10.1%) |
| Partial (GT found + extras) | 99 |
| Fail (missed GT) | 25 |
| Pass+Partial | 81.9% |
| Avg false positives/turn | 2.82 |
| Avg intents detected/turn | 3.88 |
| Total missed intents | 67 |

**By category:**

| Category | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| adversarial | 25 | 0 | 21 | 4 | 0.0% |
| confused_elderly | 25 | 2 | 18 | 5 | 8.0% |
| email_longform | 13 | 0 | 13 | 0 | 0.0% |
| frustrated_verbose | 25 | 1 | 22 | 2 | 4.0% |
| terse_impatient | 25 | 10 | 5 | 10 | 40.0% |
| topic_shifting | 25 | 1 | 20 | 4 | 4.0% |

**By word count:**

| Words | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| 1-5 | 15 | 8 | 2 | 5 | 53.3% |
| 6-10 | 11 | 2 | 3 | 6 | 18.2% |
| 11-20 | 25 | 1 | 21 | 3 | 4.0% |
| 21-40 | 70 | 3 | 57 | 10 | 4.3% |
| 41+ | 17 | 0 | 16 | 1 | 0.0% |

**Top false positives:** cancel_order (42x), remove_item (25x), price_check (22x), track_order (22x), product_availability (19x), billing_issue (19x), reorder (17x), contact_human (17x), report_fraud (15x), add_payment_method (15x)

**Top missed:** refund (12x), billing_issue (7x), transaction_details (6x), payment_history (4x), account_status (4x), order_history (3x), request_invoice (3x), subscription_status (3x), delivery_estimate (3x), pause_subscription (3x)

**Sample failures (25 total, showing 5):**

- `frustrated_double_billing` [2] (50 words)
  "Look I already called about this three days ago and the agent said they would..."  
  GT: ["billing_issue"] | DT: ["transaction_details:4.8", "delivery_estimate:1.3", "refund:2.4", "contact_human:7.4", "apply_coupon:3.9", "cancel_order:1.1"]
- `frustrated_hacked_account` [5] (23 words)
  "Send me a full statement of all account activity for the past 30 days. I need..."  
  GT: ["payment_history", "request_invoice"] | DT: ["refund:1.2", "order_history:3.2", "report_fraud:6.5", "delivery_estimate:1.3", "file_complaint:1.3"]
- `confused_cant_login` [1] (26 words)
  "The thing won't let me in anymore. I put in my name and the secret word but i..."  
  GT: ["reset_password"] | DT: ["cancel_order:2.2", "remove_item:4.0", "pause_subscription:1.5", "close_account:1.1"]
- `confused_cant_login` [5] (27 words)
  "Thank you dearie. While I have you on the line, can you check if my account i..."  
  GT: ["account_status"] | DT: ["product_availability:5.8", "report_fraud:3.0"]
- `confused_mystery_order` [3] (31 words)
  "Oh wait my daughter might have ordered this for me as a gift. Can you tell me..."  
  GT: ["order_history", "transaction_details"] | DT: ["gift_card_redeem:4.0", "cancel_order:3.8", "reorder:1.3", "contact_human:2.5", "track_order:1.3", "update_address:2.7"]

### Threshold Sweep

| Threshold | Pass | Partial | Fail | Pass% | P+P% | Avg FP/turn |
|---|---|---|---|---|---|---|
| 0.3 | 14 | 99 | 25 | 10.1% | 81.9% | 2.82 |
| 1.0 | 14 | 99 | 25 | 10.1% | 81.9% | 2.82 |
| 2.0 | 21 | 86 | 31 | 15.2% | 77.5% | 1.41 |
| 3.0 | 32 | 67 | 39 | 23.2% | 71.7% | 0.86 |
| 4.0 | 37 | 47 | 54 | 26.8% | 60.9% | 0.50 |
| 5.0 | 35 | 32 | 71 | 25.4% | 48.6% | 0.23 |
| 7.0 | 26 | 21 | 91 | 18.8% | 34.1% | 0.05 |

---

## TEST A: SymSpell Correction

**Vocabulary size:** 1097 unique terms

**Query term coverage (no typos):** 1577/4690 (33.6%)

**With 30% typo rate (baseline for comparison):**

| Condition | Pass | Partial | Fail | Pass% |
|---|---|---|---|---|
| No typos (baseline) | 14 | 99 | 25 | 10.1% |
| With typos, no correction | 14 | 73 | 51 | 10.1% |
| With typos + SymSpell | 8 | 105 | 25 | 5.8% |

**Corrections applied:** 1176 words across all turns

---

## TEST B: IDF Noise Gate

**Term IDF distribution across all query terms:**

- Unique terms in queries: 2935
- IDF range: 0.00 to 2.79
- Median IDF: 0.00
- 25th percentile: 0.00

**Noisiest terms (highest df):**

| Term | DF (intents) | IDF |
|---|---|---|
| how | 18 | 1.35 |
| what | 15 | 1.44 |
| do | 14 | 1.47 |
| can | 13 | 1.51 |
| need | 13 | 1.51 |
| account | 12 | 1.55 |
| order | 10 | 1.64 |
| want | 10 | 1.64 |
| not | 9 | 1.69 |
| when | 8 | 1.75 |
| have | 8 | 1.75 |
| before | 7 | 1.82 |
| shipping | 7 | 1.82 |
| product | 7 | 1.82 |
| purchase | 7 | 1.82 |
| does | 7 | 1.82 |
| delivery | 6 | 1.90 |
| show | 6 | 1.90 |
| how do | 6 | 1.90 |
| billing | 6 | 1.90 |

**Noise gate results (max_df = exclude terms in > N intents):**

| max_df | Pass | Partial | Fail | Pass% | P+P% | Avg FP | Avg missed |
|---|---|---|---|---|---|---|---|
| 3 | 13 | 97 | 28 | 9.4% | 79.7% | 2.75 | 0.51 |
| 5 | 13 | 98 | 27 | 9.4% | 80.4% | 2.77 | 0.50 |
| 8 | 13 | 97 | 28 | 9.4% | 79.7% | 2.80 | 0.50 |
| 10 | 13 | 97 | 28 | 9.4% | 79.7% | 2.81 | 0.51 |
| 12 | 13 | 98 | 27 | 9.4% | 80.4% | 2.83 | 0.51 |
| 15 | 14 | 100 | 24 | 10.1% | 82.6% | 2.80 | 0.48 |
| 18 | 14 | 99 | 25 | 10.1% | 81.9% | 2.82 | 0.49 |
| 25 | 14 | 99 | 25 | 10.1% | 81.9% | 2.82 | 0.49 |
| 36 | 14 | 99 | 25 | 10.1% | 81.9% | 2.82 | 0.49 |

### Noise Gate (max_df=8) Detail

| Metric | Value |
|---|---|
| Total turns | 138 |
| Pass (exact) | 13 (9.4%) |
| Partial (GT found + extras) | 97 |
| Fail (missed GT) | 28 |
| Pass+Partial | 79.7% |
| Avg false positives/turn | 2.80 |
| Avg intents detected/turn | 3.86 |
| Total missed intents | 69 |

**By category:**

| Category | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| adversarial | 25 | 0 | 20 | 5 | 0.0% |
| confused_elderly | 25 | 1 | 18 | 6 | 4.0% |
| email_longform | 13 | 0 | 13 | 0 | 0.0% |
| frustrated_verbose | 25 | 1 | 21 | 3 | 4.0% |
| terse_impatient | 25 | 10 | 5 | 10 | 40.0% |
| topic_shifting | 25 | 1 | 20 | 4 | 4.0% |

**By word count:**

| Words | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| 1-5 | 15 | 8 | 2 | 5 | 53.3% |
| 6-10 | 11 | 2 | 3 | 6 | 18.2% |
| 11-20 | 25 | 1 | 21 | 3 | 4.0% |
| 21-40 | 70 | 2 | 55 | 13 | 2.9% |
| 41+ | 17 | 0 | 16 | 1 | 0.0% |

**Top false positives:** cancel_order (43x), price_check (24x), remove_item (24x), track_order (23x), product_availability (21x), reorder (18x), contact_human (17x), billing_issue (16x), add_payment_method (15x), report_fraud (14x)

**Top missed:** refund (12x), billing_issue (7x), transaction_details (6x), subscription_status (4x), order_history (4x), payment_history (4x), account_status (4x), pause_subscription (3x), delivery_estimate (3x), file_complaint (3x)

**Sample failures (28 total, showing 5):**

- `frustrated_double_billing` [2] (50 words)
  "Look I already called about this three days ago and the agent said they would..."  
  GT: ["billing_issue"] | DT: ["transaction_details:4.8", "delivery_estimate:1.3", "refund:3.5", "contact_human:5.4", "cancel_order:1.1", "apply_coupon:1.5"]
- `frustrated_wrong_item` [2] (37 words)
  "I already checked my order history and the order clearly shows a wireless key..."  
  GT: ["order_history"] | DT: ["payment_history:3.7", "track_order:1.5"]
- `frustrated_hacked_account` [5] (23 words)
  "Send me a full statement of all account activity for the past 30 days. I need..."  
  GT: ["payment_history", "request_invoice"] | DT: ["refund:1.2", "order_history:3.2", "report_fraud:4.1", "delivery_estimate:1.3", "file_complaint:1.3"]
- `confused_cant_login` [1] (26 words)
  "The thing won't let me in anymore. I put in my name and the secret word but i..."  
  GT: ["reset_password"] | DT: ["cancel_order:2.2", "remove_item:1.3", "pause_subscription:1.5", "close_account:1.1"]
- `confused_cant_login` [5] (27 words)
  "Thank you dearie. While I have you on the line, can you check if my account i..."  
  GT: ["account_status"] | DT: ["product_availability:5.8", "report_fraud:1.5"]

---

## TEST C: Per-Intent Confidence Calibration

**Seed self-test score distributions:**

| Intent | Seeds | Mean | StdDev | Min threshold |
|---|---|---|---|---|
| transfer_funds | 7 | 9.74 | 2.15 | 6.52 |
| account_status | 7 | 5.55 | 1.01 | 4.03 |
| payment_history | 7 | 10.15 | 1.75 | 7.52 |
| account_limits | 7 | 8.60 | 2.16 | 5.37 |
| file_complaint | 7 | 9.37 | 1.58 | 7.00 |
| report_fraud | 7 | 11.93 | 2.38 | 8.36 |
| reorder | 7 | 11.39 | 3.52 | 6.11 |
| gift_card_redeem | 7 | 10.62 | 1.13 | 8.94 |
| order_history | 7 | 10.57 | 2.93 | 6.17 |
| contact_human | 8 | 11.37 | 5.79 | 2.68 |
| delivery_estimate | 7 | 9.51 | 2.51 | 5.75 |
| track_order | 8 | 8.95 | 3.84 | 3.19 |
| loyalty_points | 7 | 10.28 | 4.10 | 4.12 |
| price_check | 7 | 9.39 | 1.37 | 7.33 |
| pause_subscription | 7 | 9.35 | 1.90 | 6.50 |
| reset_password | 8 | 9.57 | 2.84 | 5.31 |
| billing_issue | 8 | 9.59 | 3.58 | 4.23 |
| apply_coupon | 7 | 10.17 | 2.82 | 5.94 |
| update_address | 8 | 8.00 | 2.16 | 4.76 |
| subscription_status | 7 | 9.48 | 1.29 | 7.54 |
| remove_item | 7 | 10.53 | 2.34 | 7.01 |
| transaction_details | 7 | 8.66 | 2.56 | 4.83 |
| return_policy | 7 | 9.34 | 2.44 | 5.68 |
| schedule_callback | 7 | 10.15 | 1.52 | 7.87 |
| shipping_options | 7 | 8.08 | 1.74 | 5.47 |
| cancel_order | 8 | 9.17 | 2.40 | 5.57 |
| eligibility_check | 7 | 9.98 | 2.09 | 6.84 |
| close_account | 7 | 8.02 | 2.76 | 3.89 |
| add_payment_method | 7 | 10.82 | 1.90 | 7.98 |
| change_plan | 7 | 8.09 | 1.66 | 5.59 |
| check_balance | 8 | 8.44 | 3.21 | 3.63 |
| upgrade_shipping | 7 | 8.03 | 2.14 | 4.82 |
| refund | 8 | 9.71 | 4.12 | 3.53 |
| request_invoice | 7 | 7.98 | 1.69 | 5.44 |
| product_availability | 7 | 9.50 | 2.41 | 5.88 |
| warranty_info | 7 | 7.80 | 0.92 | 6.43 |

### Calibrated Thresholds

| Metric | Value |
|---|---|
| Total turns | 138 |
| Pass (exact) | 30 (21.7%) |
| Partial (GT found + extras) | 38 |
| Fail (missed GT) | 70 |
| Pass+Partial | 49.3% |
| Avg false positives/turn | 0.19 |
| Avg intents detected/turn | 0.74 |
| Total missed intents | 138 |

**By category:**

| Category | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| adversarial | 25 | 6 | 6 | 13 | 24.0% |
| confused_elderly | 25 | 0 | 5 | 20 | 0.0% |
| email_longform | 13 | 1 | 11 | 1 | 7.7% |
| frustrated_verbose | 25 | 5 | 13 | 7 | 20.0% |
| terse_impatient | 25 | 7 | 0 | 18 | 28.0% |
| topic_shifting | 25 | 11 | 3 | 11 | 44.0% |

**By word count:**

| Words | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| 1-5 | 15 | 6 | 0 | 9 | 40.0% |
| 6-10 | 11 | 2 | 0 | 9 | 18.2% |
| 11-20 | 25 | 9 | 3 | 13 | 36.0% |
| 21-40 | 70 | 12 | 22 | 36 | 17.1% |
| 41+ | 17 | 1 | 13 | 3 | 5.9% |

**Top false positives:** billing_issue (6x), track_order (5x), contact_human (5x), refund (2x), delivery_estimate (1x), cancel_order (1x), loyalty_points (1x), reorder (1x), change_plan (1x), close_account (1x)

**Top missed:** refund (21x), billing_issue (12x), payment_history (7x), transaction_details (6x), reset_password (6x), request_invoice (6x), add_payment_method (6x), transfer_funds (5x), track_order (5x), reorder (5x)

---

## TEST D: Coverage Ratio Analysis

**Accuracy by query term coverage:**

| Coverage | Total | Pass | Part | Fail | Pass% | P+P% |
|---|---|---|---|---|---|---|
| 0-20% | 9 | 2 | 5 | 2 | 22.2% | 77.8% |
| 20-40% | 65 | 4 | 48 | 13 | 6.2% | 80.0% |
| 40-60% | 50 | 2 | 43 | 5 | 4.0% | 90.0% |
| 60-80% | 8 | 1 | 3 | 4 | 12.5% | 50.0% |
| 80-100% | 6 | 5 | 0 | 1 | 83.3% | 83.3% |

---

## TEST E: Anti-Co-occurrence Filter

**Valid pairs identified:** 74
**Invalid pairs identified:** 357

**Top invalid pairs (most common false co-detections):**

| Pair | False count | Valid count |
|---|---|---|
| billing_issue + refund | 13 | 5 |
| cancel_order + refund | 12 | 1 |
| price_check + refund | 9 | 0 |
| refund + remove_item | 9 | 0 |
| refund + track_order | 7 | 5 |
| order_history + refund | 7 | 0 |
| billing_issue + track_order | 7 | 2 |
| billing_issue + transaction_details | 6 | 2 |
| cancel_order + track_order | 6 | 0 |
| refund + return_policy | 6 | 2 |
| billing_issue + order_history | 6 | 0 |
| refund + reorder | 6 | 3 |
| cancel_order + change_plan | 6 | 0 |
| refund + transfer_funds | 6 | 1 |
| contact_human + refund | 5 | 1 |

### Anti-Co-occurrence Filter

| Metric | Value |
|---|---|
| Total turns | 138 |
| Pass (exact) | 37 (26.8%) |
| Partial (GT found + extras) | 55 |
| Fail (missed GT) | 46 |
| Pass+Partial | 66.7% |
| Avg false positives/turn | 0.82 |
| Avg intents detected/turn | 1.57 |
| Total missed intents | 111 |

**By category:**

| Category | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| adversarial | 25 | 6 | 9 | 10 | 24.0% |
| confused_elderly | 25 | 4 | 10 | 11 | 16.0% |
| email_longform | 13 | 1 | 10 | 2 | 7.7% |
| frustrated_verbose | 25 | 4 | 16 | 5 | 16.0% |
| terse_impatient | 25 | 12 | 2 | 11 | 48.0% |
| topic_shifting | 25 | 10 | 8 | 7 | 40.0% |

**By word count:**

| Words | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| 1-5 | 15 | 8 | 1 | 6 | 53.3% |
| 6-10 | 11 | 4 | 1 | 6 | 36.4% |
| 11-20 | 25 | 9 | 8 | 8 | 36.0% |
| 21-40 | 70 | 15 | 32 | 23 | 21.4% |
| 41+ | 17 | 1 | 13 | 3 | 5.9% |

**Top false positives:** remove_item (12x), product_availability (10x), price_check (6x), billing_issue (6x), order_history (6x), cancel_order (6x), contact_human (6x), report_fraud (5x), pause_subscription (5x), reorder (4x)

**Top missed:** refund (27x), billing_issue (12x), transaction_details (6x), track_order (6x), request_invoice (5x), subscription_status (4x), order_history (4x), account_status (4x), payment_history (4x), change_plan (3x)

**Note:** This test is somewhat circular (trained on test data) but shows the ceiling.

---

## TEST G: Anchor-Based Scoring

**Discrimination threshold:** df <= 3 (N=36, N/15=2)

**Anchor terms per intent (df <= threshold, weight >= 0.5):**

| Intent | Anchor terms (df) |
|---|---|
| transfer_funds | money another (1), another (1), money (3) |
| account_status | account status (1), status (2) |
| payment_history | spent (1), when last (1), transaction log (1), last payment (1), charges account (1) |
| account_limits | limit account (1), restrictions account (1), what spending (1), transfer limit (1), maximum (1) |
| file_complaint | complaint (1), service (3) |
| report_fraud | access account (1), account hacked (1), need report (1), transactions (1), activity card (1) |
| reorder | buy (1), reorder what (1), what got (1), got (1), reorder (1) |
| gift_card_redeem | gift (1), gift card (1) |
| order_history | view order (1), did order (1), past (1), all previous (1), up recent (1) |
| contact_human | customer service (1), get agent (1), someone who (1), speak (1), useless (1) |
| delivery_estimate | arrival (1), days until (1), estimated (1), order arrive (1), delivery date (1) |
| track_order | tracking (1), delivery arrive (1), delivered never (1), package tracking (1), need shipping (1) |
| loyalty_points | points (1), reward (1), points do (1), reward points (1), loyalty (1) |
| price_check | guarantee (1), sale (1), discounts item (1), price item (1), price (1) |
| pause_subscription | temporarily (1), can freeze (1), want temporarily (1), next quarter (1), until next (1) |
| reset_password | reset (1), getting (1), locked (1), password (1), locked out (1) |
| billing_issue | wrong charge (1), bill (1), twice same (1), overcharged (1), bill does (1) |
| apply_coupon | where do (1), coupons one (1), rejected (1), two coupons (1), code rejected (1) |
| update_address | address (1) |
| subscription_status | billing date (1), features included (1), does subscription (1), when next (1), what features (1) |
| remove_item | item order (1), remove (2), one (2), take (3) |
| transaction_details | need details (1), reference number (1), what charge (1), breakdown (1), transaction reference (1) |
| return_policy | returns (1), policy (1), return (2) |
| schedule_callback | call back (1), call (1), phone call (1), phone (1), back (3) |
| shipping_options | (NONE) |
| cancel_order | cancel order (1), ordered (2), cancel (2) |
| eligibility_check | eligible credit (1), meet (1), am eligible (1), apply program (1), can apply (1) |
| close_account | close account (1), close (1) |
| add_payment_method | add (1), new (2), payment (3) |
| change_plan | subscription plan (1), cheaper plan (1), switch premium (1), downgrade account (1), account basic (1) |
| check_balance | do owe (1), need know (1), available funds (1), know how (1), current account (1) |
| upgrade_shipping | faster (1), expedite shipment (1), rush delivery (1), upgrade express (1), extra faster (1) |
| refund | refund (1), money back (1), return (2), money (3), back (3) |
| request_invoice | invoice (1) |
| product_availability | check if (1), available delivery (1), product available (1), size (1), do carry (1) |
| warranty_info | warranty (1) |

**WARNING: 1 intents have NO anchor terms:** ["shipping_options"]

**Anchor-based results by window size:**

| Window | Pass | Partial | Fail | Pass% | P+P% | Avg FP | Avg missed |
|---|---|---|---|---|---|---|---|
| 3 | 6 | 107 | 25 | 4.3% | 81.9% | 4.21 | 0.43 |
| 5 | 6 | 105 | 27 | 4.3% | 80.4% | 4.09 | 0.44 |
| 7 | 7 | 104 | 27 | 5.1% | 80.4% | 3.91 | 0.44 |
| 10 | 7 | 104 | 27 | 5.1% | 80.4% | 3.64 | 0.45 |
| 15 | 7 | 104 | 27 | 5.1% | 80.4% | 3.39 | 0.46 |

### Anchor-Based (window=7) Detail

| Metric | Value |
|---|---|
| Total turns | 138 |
| Pass (exact) | 7 (5.1%) |
| Partial (GT found + extras) | 104 |
| Fail (missed GT) | 27 |
| Pass+Partial | 80.4% |
| Avg false positives/turn | 3.93 |
| Avg intents detected/turn | 5.04 |
| Total missed intents | 61 |

**By category:**

| Category | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| adversarial | 25 | 1 | 19 | 5 | 4.0% |
| confused_elderly | 25 | 0 | 17 | 8 | 0.0% |
| email_longform | 13 | 0 | 13 | 0 | 0.0% |
| frustrated_verbose | 25 | 1 | 24 | 0 | 4.0% |
| terse_impatient | 25 | 4 | 11 | 10 | 16.0% |
| topic_shifting | 25 | 1 | 20 | 4 | 4.0% |

**By word count:**

| Words | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| 1-5 | 15 | 4 | 5 | 6 | 26.7% |
| 6-10 | 11 | 0 | 6 | 5 | 0.0% |
| 11-20 | 25 | 2 | 19 | 4 | 8.0% |
| 21-40 | 70 | 1 | 57 | 12 | 1.4% |
| 41+ | 17 | 0 | 17 | 0 | 0.0% |

**Top false positives:** apply_coupon (40x), billing_issue (39x), order_history (37x), payment_history (34x), track_order (33x), reorder (32x), report_fraud (31x), check_balance (28x), product_availability (27x), price_check (25x)

**Top missed:** refund (12x), request_invoice (6x), transfer_funds (5x), account_status (4x), close_account (4x), account_limits (3x), add_payment_method (3x), file_complaint (3x), transaction_details (3x), subscription_status (2x)

**Sample failures (27 total, showing 5):**

- `confused_cant_login` [1] (26 words)
  "The thing won't let me in anymore. I put in my name and the secret word but i..."  
  GT: ["reset_password"] | DT: ["reorder:1.3", "track_order:2.0", "product_availability:1.4", "delivery_estimate:1.6", "pause_subscription:1.5"]
- `confused_cant_login` [3] (25 words)
  "OK so you sent me an email to change it but I can't find the email. Where wou..."  
  GT: ["reset_password"] | DT: ["track_order:2.0", "apply_coupon:4.0"]
- `confused_cant_login` [5] (27 words)
  "Thank you dearie. While I have you on the line, can you check if my account i..."  
  GT: ["account_status"] | DT: ["product_availability:4.1", "report_fraud:3.0"]
- `confused_mystery_order` [3] (31 words)
  "Oh wait my daughter might have ordered this for me as a gift. Can you tell me..."  
  GT: ["order_history", "transaction_details"] | DT: ["cancel_order:3.8", "billing_issue:2.1", "reorder:3.6", "gift_card_redeem:3.3", "contact_human:3.6", "track_order:3.7", "update_address:3.3"]
- `confused_mystery_order` [4] (28 words)
  "OK she says she didn't order it either. I want to send it back. How do I do t..."  
  GT: ["refund", "return_policy"] | DT: ["track_order:3.9", "order_history:3.1", "report_fraud:2.6"]

---

## TEST H: Session-Based Prior

**Transition pairs observed:** 190


### Session Prior

| Metric | Value |
|---|---|
| Total turns | 138 |
| Pass (exact) | 14 (10.1%) |
| Partial (GT found + extras) | 99 |
| Fail (missed GT) | 25 |
| Pass+Partial | 81.9% |
| Avg false positives/turn | 2.82 |
| Avg intents detected/turn | 3.88 |
| Total missed intents | 67 |

**By category:**

| Category | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| adversarial | 25 | 0 | 21 | 4 | 0.0% |
| confused_elderly | 25 | 2 | 18 | 5 | 8.0% |
| email_longform | 13 | 0 | 13 | 0 | 0.0% |
| frustrated_verbose | 25 | 1 | 22 | 2 | 4.0% |
| terse_impatient | 25 | 10 | 5 | 10 | 40.0% |
| topic_shifting | 25 | 1 | 20 | 4 | 4.0% |

**By word count:**

| Words | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| 1-5 | 15 | 8 | 2 | 5 | 53.3% |
| 6-10 | 11 | 2 | 3 | 6 | 18.2% |
| 11-20 | 25 | 1 | 21 | 3 | 4.0% |
| 21-40 | 70 | 3 | 57 | 10 | 4.3% |
| 41+ | 17 | 0 | 16 | 1 | 0.0% |

**Top false positives:** cancel_order (43x), remove_item (25x), track_order (23x), price_check (22x), product_availability (19x), billing_issue (18x), reorder (17x), contact_human (17x), report_fraud (16x), add_payment_method (14x)

**Top missed:** refund (12x), billing_issue (7x), transaction_details (6x), payment_history (4x), account_status (4x), account_limits (3x), file_complaint (3x), track_order (3x), subscription_status (3x), order_history (3x)

---

## TEST F: Combined Pipeline

**Pipeline:** Noise gate (max_df=8) + Anti-co-occurrence + Calibrated thresholds


### Combined Pipeline

| Metric | Value |
|---|---|
| Total turns | 138 |
| Pass (exact) | 26 (18.8%) |
| Partial (GT found + extras) | 25 |
| Fail (missed GT) | 87 |
| Pass+Partial | 37.0% |
| Avg false positives/turn | 0.09 |
| Avg intents detected/turn | 0.46 |
| Total missed intents | 162 |

**By category:**

| Category | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| adversarial | 25 | 5 | 4 | 16 | 20.0% |
| confused_elderly | 25 | 1 | 2 | 22 | 4.0% |
| email_longform | 13 | 2 | 7 | 4 | 15.4% |
| frustrated_verbose | 25 | 3 | 11 | 11 | 12.0% |
| terse_impatient | 25 | 7 | 0 | 18 | 28.0% |
| topic_shifting | 25 | 8 | 1 | 16 | 32.0% |

**By word count:**

| Words | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| 1-5 | 15 | 6 | 0 | 9 | 40.0% |
| 6-10 | 11 | 2 | 0 | 9 | 18.2% |
| 11-20 | 25 | 6 | 0 | 19 | 24.0% |
| 21-40 | 70 | 9 | 18 | 43 | 12.9% |
| 41+ | 17 | 3 | 7 | 7 | 17.6% |

**Top false positives:** billing_issue (4x), contact_human (3x), upgrade_shipping (1x), order_history (1x), track_order (1x), refund (1x), close_account (1x)

**Top missed:** refund (24x), billing_issue (16x), request_invoice (8x), track_order (7x), payment_history (7x), order_history (7x), file_complaint (6x), reset_password (6x), subscription_status (6x), add_payment_method (6x)

**Sample failures (87 total, showing 5):**

- `frustrated_double_billing` [2] (50 words)
  "Look I already called about this three days ago and the agent said they would..."  
  GT: ["billing_issue"] | DT: ["contact_human:5.4"]
- `frustrated_double_billing` [4] (41 words)
  "You know what, if this isn't resolved by end of day I'm going to dispute both..."  
  GT: ["file_complaint"] | DT: []
- `frustrated_double_billing` [5] (17 words)
  "Just cancel the whole order and refund everything. I'll buy it somewhere else..."  
  GT: ["cancel_order", "refund"] | DT: []
- `frustrated_service_outage` [1] (41 words)
  "My internet has been down for three days now and nobody seems to care. I work..."  
  GT: ["billing_issue", "contact_human"] | DT: []
- `frustrated_wrong_item` [1] (39 words)
  "I ordered a blue wireless keyboard and instead received a pink phone case. Th..."  
  GT: ["refund", "track_order"] | DT: ["billing_issue:5.0"]

**Pipeline variant: Anchor (window=7) + Anti-co-occurrence:**


### Anchor + Anti-Co-occurrence

| Metric | Value |
|---|---|
| Total turns | 138 |
| Pass (exact) | 26 (18.8%) |
| Partial (GT found + extras) | 64 |
| Fail (missed GT) | 48 |
| Pass+Partial | 65.2% |
| Avg false positives/turn | 1.09 |
| Avg intents detected/turn | 1.80 |
| Total missed intents | 115 |

**By category:**

| Category | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| adversarial | 25 | 3 | 10 | 12 | 12.0% |
| confused_elderly | 25 | 3 | 11 | 11 | 12.0% |
| email_longform | 13 | 1 | 11 | 1 | 7.7% |
| frustrated_verbose | 25 | 3 | 17 | 5 | 12.0% |
| terse_impatient | 25 | 9 | 6 | 10 | 36.0% |
| topic_shifting | 25 | 7 | 9 | 9 | 28.0% |

**By word count:**

| Words | Total | Pass | Part | Fail | Pass% |
|---|---|---|---|---|---|
| 1-5 | 15 | 5 | 4 | 6 | 33.3% |
| 6-10 | 11 | 4 | 2 | 5 | 36.4% |
| 11-20 | 25 | 5 | 11 | 9 | 20.0% |
| 21-40 | 70 | 10 | 34 | 26 | 14.3% |
| 41+ | 17 | 2 | 13 | 2 | 11.8% |

**Top false positives:** eligibility_check (19x), product_availability (13x), delivery_estimate (13x), apply_coupon (13x), order_history (10x), billing_issue (8x), payment_history (7x), contact_human (7x), report_fraud (7x), remove_item (6x)

**Top missed:** refund (26x), billing_issue (12x), request_invoice (6x), transaction_details (6x), close_account (5x), transfer_funds (5x), payment_history (5x), account_status (4x), track_order (4x), account_limits (3x)

---

## COMPARISON SUMMARY

| Experiment | Pass | Part | Fail | Pass% | P+P% | Avg FP/turn |
|---|---|---|---|---|---|---|
| Baseline (t=0.3) | 14 | 99 | 25 | 10.1% | 81.9% | 2.82 |
| Noise Gate (max_df=8) | 13 | 97 | 28 | 9.4% | 79.7% | 2.80 |
| Confidence Calibrated | 30 | 38 | 70 | 21.7% | 49.3% | 0.19 |
| Anti-Co-occurrence | 37 | 55 | 46 | 26.8% | 66.7% | 0.82 |
| Anchor (window=7) | 7 | 104 | 27 | 5.1% | 80.4% | 3.93 |
| Session Prior | 14 | 99 | 25 | 10.1% | 81.9% | 2.82 |
| Combined (NG+Cal+Anti) | 26 | 25 | 87 | 18.8% | 37.0% | 0.09 |
| Anchor+Anti | 26 | 64 | 48 | 18.8% | 65.2% | 1.09 |

**SymSpell (Test A) measured separately** — typo recovery: 10.1% → 5.8% (with 30% typo rate)
