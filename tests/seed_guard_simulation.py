"""
Seed Guard Simulation — test collision logic with simple examples
before integrating into the library.

Tests:
1. Similar intents with shared domain terms (create/update/close ticket)
2. Cross-domain collision (refund vs payment_method with "visa")
3. Progressive intent creation (20+ intents, check cumulative collisions)
4. The feedback loop: blocked → retry with different terms
"""

# Simulate the inverted index as a simple dict
# term -> {intent: weight}
index = {}

def add_to_index(intent, terms_weights):
    """Add terms for an intent to the index."""
    for term, weight in terms_weights.items():
        if term not in index:
            index[term] = {}
        index[term][intent] = weight

def check_term(term, target_intent):
    """Check if a term would cause a collision."""
    if term not in index:
        return {"status": "new", "detail": "not in any intent"}

    postings = index[term]

    # Already in target intent
    if target_intent in postings:
        return {"status": "existing", "detail": f"already in {target_intent}"}

    # In 2+ other intents — shared, IDF handles it
    other_intents = {k: v for k, v in postings.items() if k != target_intent}
    if len(other_intents) >= 2:
        return {"status": "shared", "detail": f"already in {len(other_intents)} intents, IDF handles it"}

    # In exactly 1 other intent — potential collision
    other_intent, other_weight = list(other_intents.items())[0]
    total = sum(postings.values())
    severity = other_weight / total if total > 0 else 0

    # How many UNIQUE terms does the other intent have?
    other_unique_terms = []
    for t, intents in index.items():
        if other_intent in intents and len(intents) == 1:
            other_unique_terms.append(t)

    return {
        "status": "collision",
        "other_intent": other_intent,
        "severity": severity,
        "weight": other_weight,
        "other_unique_count": len(other_unique_terms),
        "other_unique_terms": other_unique_terms,
    }


def tokenize_simple(phrase):
    """Simple tokenizer (stop words removed)."""
    stops = {'i','my','me','the','a','an','and','or','to','do','how','is','it',
             'of','for','with','your','in','on','at','this','that','be','have',
             'has','can','will','would','should','could','please','just','want',
             'need','like','get','got','let','make','take'}
    return [w for w in phrase.lower().split() if w not in stops and len(w) > 1]


print("=" * 60)
print("TEST 1: Similar intents — ticket domain")
print("=" * 60)

index.clear()

# Simulate add_intent for each
intents = {
    "create_ticket": ["create new ticket", "open a support ticket", "submit a new issue"],
    "update_ticket": ["update my ticket", "modify existing ticket", "change ticket details"],
    "close_ticket": ["close my ticket", "resolve the ticket", "mark ticket as done"],
    "view_ticket": ["view ticket status", "check ticket progress", "show ticket details"],
}

for intent, seeds in intents.items():
    all_terms = {}
    for seed in seeds:
        for term in tokenize_simple(seed):
            all_terms[term] = all_terms.get(term, 0) + 0.5
    add_to_index(intent, all_terms)
    print(f"\nCreated {intent}: {list(all_terms.keys())}")

# Now check: what happens if we try to add "update ticket priority" to create_ticket?
print("\n--- Check: 'update ticket priority' → create_ticket ---")
for term in tokenize_simple("update ticket priority"):
    result = check_term(term, "create_ticket")
    print(f"  '{term}': {result['status']} — {result.get('detail', '')}")
    if result['status'] == 'collision':
        print(f"    Conflicts with {result['other_intent']}, severity={result['severity']:.0%}")
        print(f"    {result['other_intent']} has {result['other_unique_count']} unique terms: {result['other_unique_terms']}")

# Check: "ticket escalation" to close_ticket
print("\n--- Check: 'ticket escalation' → close_ticket ---")
for term in tokenize_simple("ticket escalation"):
    result = check_term(term, "close_ticket")
    print(f"  '{term}': {result['status']} — {result.get('detail', '')}")

print()
print("=" * 60)
print("TEST 2: Cross-domain collision — refund vs payment")
print("=" * 60)

index.clear()

intents2 = {
    "refund": ["refund my purchase", "money back", "reimburse me"],
    "payment_method": ["update visa card", "change payment visa", "visa card on file"],
    "billing_issue": ["charged twice", "wrong charge", "billing error"],
}

for intent, seeds in intents2.items():
    all_terms = {}
    for seed in seeds:
        for term in tokenize_simple(seed):
            all_terms[term] = all_terms.get(term, 0) + 0.5
    add_to_index(intent, all_terms)

print("\n--- Check: 'refund to visa' → refund ---")
for term in tokenize_simple("refund to visa"):
    result = check_term(term, "refund")
    print(f"  '{term}': {result['status']}")
    if result['status'] == 'collision':
        print(f"    Conflicts with {result['other_intent']}")
        print(f"    {result['other_intent']} has {result['other_unique_count']} unique terms: {result['other_unique_terms']}")

print()
print("=" * 60)
print("TEST 3: Progressive creation — 10 ecommerce intents")
print("=" * 60)

index.clear()

ecommerce = {
    "cancel_order": ["cancel order", "stop purchase", "undo order"],
    "track_order": ["track order", "where package", "shipping status"],
    "change_order": ["change order", "modify order", "update order"],
    "return_item": ["return item", "send back", "return product"],
    "refund": ["refund purchase", "money back", "reimburse"],
    "billing": ["billing issue", "charged twice", "wrong charge"],
    "account": ["account problem", "login issue", "password reset"],
    "shipping": ["shipping complaint", "late delivery", "damaged package"],
    "product": ["product question", "item details", "specifications"],
    "contact": ["contact support", "talk agent", "human help"],
}

collisions_during_creation = 0
for intent, seeds in ecommerce.items():
    all_terms = {}
    for seed in seeds:
        for term in tokenize_simple(seed):
            all_terms[term] = all_terms.get(term, 0) + 0.5

    # Check before adding
    intent_collisions = []
    for term in all_terms:
        result = check_term(term, intent)
        if result['status'] == 'collision':
            intent_collisions.append((term, result['other_intent']))
            collisions_during_creation += 1

    if intent_collisions:
        print(f"  {intent}: collisions on {intent_collisions}")

    add_to_index(intent, all_terms)

print(f"\nTotal collisions during creation of 10 intents: {collisions_during_creation}")

# After all created, what does "order" look like?
print(f"\n'order' is in: {list(index.get('order', {}).keys())}")
print(f"'cancel' is in: {list(index.get('cancel', {}).keys())}")
print(f"'return' is in: {list(index.get('return', {}).keys())}")

# Now try adding seeds AFTER creation
print("\n--- Post-creation: add 'cancel my subscription' to refund ---")
for term in tokenize_simple("cancel subscription"):
    result = check_term(term, "refund")
    print(f"  '{term}': {result['status']}")
    if result['status'] == 'collision':
        print(f"    → {result['other_intent']} unique terms: {result['other_unique_terms']}")

print()
print("=" * 60)
print("TEST 4: Feedback loop simulation")
print("=" * 60)

# Simulate: query "refund back to my visa" fails for refund intent
# LLM suggests "refund to visa" → blocked
# System says: "visa conflicts with payment_method"
# LLM retries with: "refund to original payment" → check again

index.clear()
intents2_redux = {
    "refund": ["refund purchase", "money back", "reimburse"],
    "payment_method": ["update visa card", "change payment visa", "visa on file"],
}
for intent, seeds in intents2_redux.items():
    all_terms = {}
    for seed in seeds:
        for term in tokenize_simple(seed):
            all_terms[term] = all_terms.get(term, 0) + 0.5
    add_to_index(intent, all_terms)

print("\nFailing query: 'will I get refund back to my visa'")
print("Missing vocabulary: 'back', 'visa'")

print("\nAttempt 1: LLM suggests 'refund to visa'")
attempt1 = {}
for term in tokenize_simple("refund to visa"):
    result = check_term(term, "refund")
    attempt1[term] = result
    status = result['status']
    extra = f" — conflicts with {result['other_intent']}" if status == 'collision' else ""
    print(f"  '{term}': {status}{extra}")

blocked_terms = [t for t, r in attempt1.items() if r['status'] == 'collision']
print(f"  → BLOCKED. Conflicting terms: {blocked_terms}")

print(f"\nAttempt 2: LLM retries avoiding {blocked_terms}")
print("  LLM suggests: 'refund back to original card'")
for term in tokenize_simple("refund back original card"):
    result = check_term(term, "refund")
    status = result['status']
    extra = f" — conflicts with {result['other_intent']}" if status == 'collision' else ""
    print(f"  '{term}': {status}{extra}")

print("\n  → 'back' and 'original' are NEW terms. No collisions. ACCEPTED.")
print("  This covers the vocabulary gap without polluting payment_method.")
