#!/bin/bash
# Emergence experiment: simulate natural multi-intent customer queries
# No hardcoded context_intents metadata — all relationships must emerge from co-occurrence

API="http://localhost:3001/api/route_multi"
CT="Content-Type: application/json"

route() {
  curl -s -X POST "$API" -H "$CT" -d "{\"query\":\"$1\",\"threshold\":0.3}" > /dev/null
}

echo "=== Emergence Experiment: 100 multi-intent queries across 36 intents ==="
echo "=== No hardcoded metadata — watching what co-occurs naturally ==="
echo ""

# --- Financial queries (refund + balance/payment context) ---
route "I want a refund, how much was I charged for that order"
route "get my money back and show me the transaction details"
route "I returned the item, when will the refund hit my balance"
route "refund my purchase and show me my payment history"
route "I need a refund, what does my account balance look like"
route "process my return and check if the credit went through"
route "I want my money back, show me the charges on my account"
route "refund the damaged item and tell me my remaining balance"

# --- Order management (cancel + order/shipping context) ---
route "cancel my order and tell me where the package is"
route "I need to cancel, has it shipped yet"
route "cancel the purchase and show me my order history"
route "stop my order from shipping, what's the delivery estimate"
route "cancel order and check the shipping status"
route "I changed my mind, cancel it and show me what I ordered"
route "withdraw my order, when was it supposed to arrive"
route "cancel and show me a tracking update"

# --- Billing disputes (billing + payment/transaction context) ---
route "I was charged twice, show me my payment history"
route "wrong charge on my account, pull up the transaction details"
route "billing error, what is my current balance"
route "dispute this charge, show me the breakdown of payments"
route "unauthorized charge on my card, check my account status"
route "overcharged me, I need to see my recent transactions"
route "billing mistake, how much have I paid this month"
route "the charge is wrong, what was the original price"

# --- Account security (password/fraud + account context) ---
route "my account was hacked, reset my password immediately"
route "someone made unauthorized purchases, check my account status"
route "I think there's fraud on my account, show me recent transactions"
route "reset my password and tell me if my account is compromised"
route "fraudulent charges, what is my account balance now"
route "lock my account and reset the password"
route "unauthorized access, show me the login history and change password"
route "report fraud and check my account limits"

# --- Subscription management (change plan/pause + subscription context) ---
route "upgrade my plan, what am I paying now"
route "I want to downgrade, when does my subscription renew"
route "change my plan and tell me what features I'll lose"
route "pause my subscription, when is my next billing date"
route "switch to annual billing, what's my current plan"
route "cancel my subscription, am I eligible for a refund"
route "freeze my account temporarily, show me my plan details"
route "change plans and check if I qualify for a discount"

# --- Shopping/ordering (reorder + product/price context) ---
route "reorder my last purchase, is the item still in stock"
route "buy the same thing again, what's the current price"
route "repeat my order, check product availability"
route "order this again and tell me the shipping options"
route "reorder but with faster shipping this time"
route "buy this again, any coupons available"
route "same order as before, what's the delivery estimate"
route "reorder and apply my loyalty points"

# --- Shipping issues (update address/upgrade shipping + delivery context) ---
route "change my shipping address, when will it arrive"
route "update delivery address and tell me the tracking status"
route "upgrade to express shipping, what's the estimated delivery"
route "switch to overnight, how long does shipping normally take"
route "change the destination and show me shipping options"
route "expedite my order, what delivery methods are available"
route "rush delivery, when is the estimated arrival"
route "new address, will this delay my package"

# --- Payment management (add payment + balance/account context) ---
route "add a new credit card and check my balance"
route "update my payment method, what do I currently owe"
route "replace my expired card, show me my account status"
route "link my bank account, what are my account limits"
route "add PayPal, when is my next payment due"
route "change payment method and show me payment history"
route "new card on file, check my subscription billing"
route "register payment method, am I eligible for auto-pay discount"

# --- Returns/warranty (refund + return policy/warranty context) ---
route "I want to return this, what's your return policy"
route "is this still under warranty, I need a replacement"
route "return this damaged product, does warranty cover this"
route "how do I get a refund, what's the return window"
route "warranty claim on my purchase, do I qualify"
route "return policy for electronics, I want my money back"
route "does the warranty cover shipping damage, I need a refund"
route "can I return this, check my order history for the date"

# --- Complaints/escalation (complaint + various context) ---
route "file a complaint, I want to talk to a manager"
route "terrible service, I need to speak to a real person"
route "escalate my issue, check my account history"
route "formal complaint about billing, show me the charges"
route "I want to complain and schedule a callback"
route "report this issue, when can someone call me"
route "file complaint, what's been happening with my orders"
route "your service is unacceptable, connect me to a supervisor"

# --- Gift cards/coupons (redeem + price/balance context) ---
route "redeem my gift card and check the remaining balance"
route "apply a coupon code, what's the total price"
route "use my gift certificate, how much is left on it"
route "promo code not working, what's the item price"
route "apply discount and show me the final cost"
route "gift card balance check, can I use it on this order"
route "enter my coupon, does it work with this product"
route "redeem points and apply my gift card together"

# --- Complex multi-step (3+ intents) ---
route "cancel my order, get a refund, and check my balance"
route "close my account, transfer remaining funds, and download invoices"
route "update address, upgrade shipping, and track my order"
route "report fraud, reset password, and check account status"
route "change plan, apply coupon, and check eligibility"
route "file complaint, request callback, and show order history"
route "return item, check warranty, and get refund status"
route "reorder, apply loyalty points, and check delivery estimate"

echo ""
echo "=== Simulation complete: 100 queries sent ==="
echo ""

# Now dump the co-occurrence matrix
echo "=== CO-OCCURRENCE RESULTS ==="
curl -s http://localhost:3001/api/co_occurrence | python3 -c "
import json, sys

pairs = json.load(sys.stdin)
pairs.sort(key=lambda p: -p['count'])

print(f'Total unique pairs: {len(pairs)}')
print(f'Total co-occurrences: {sum(p[\"count\"] for p in pairs)}')
print()

# Show top 30
print('TOP 30 CO-OCCURRING PAIRS:')
print(f'{\"Intent A\":<25} {\"Intent B\":<25} {\"Count\":>6}')
print('-' * 60)
for p in pairs[:30]:
    print(f'{p[\"a\"]:<25} {p[\"b\"]:<25} {p[\"count\"]:>6}')

print()

# Analyze: for each action intent, what context intents co-occur most?
from collections import defaultdict

# Get intent types
intents = json.loads(open('/dev/stdin', 'r').read()) if False else None

# Build adjacency
adj = defaultdict(lambda: defaultdict(int))
for p in pairs:
    adj[p['a']][p['b']] += p['count']
    adj[p['b']][p['a']] += p['count']

# Known types from the experiment
actions = {'cancel_order','refund','contact_human','reset_password','update_address',
           'billing_issue','change_plan','close_account','report_fraud','apply_coupon',
           'schedule_callback','file_complaint','request_invoice','pause_subscription',
           'transfer_funds','add_payment_method','remove_item','reorder',
           'upgrade_shipping','gift_card_redeem'}
contexts = {'track_order','check_balance','account_status','order_history',
            'payment_history','shipping_options','return_policy','product_availability',
            'warranty_info','loyalty_points','subscription_status','delivery_estimate',
            'price_check','account_limits','transaction_details','eligibility_check'}

print('EMERGED CONTEXT RELATIONSHIPS (action → context co-occurrences):')
print(f'{\"Action\":<25} {\"Context Partners (count)\":<60}')
print('-' * 85)

for action in sorted(actions):
    if action not in adj:
        continue
    context_pairs = []
    for neighbor, count in adj[action].items():
        if neighbor in contexts:
            context_pairs.append((neighbor, count))
    context_pairs.sort(key=lambda x: -x[1])
    if context_pairs:
        partners = ', '.join(f'{c}({n})' for c, n in context_pairs[:5])
        print(f'{action:<25} {partners}')

print()
print('EMERGED ACTION-ACTION RELATIONSHIPS:')
print(f'{\"Action A\":<25} {\"Action B\":<25} {\"Count\":>6}')
print('-' * 60)
for p in pairs:
    if p['a'] in actions and p['b'] in actions and p['count'] >= 2:
        print(f'{p[\"a\"]:<25} {p[\"b\"]:<25} {p[\"count\"]:>6}')
"
