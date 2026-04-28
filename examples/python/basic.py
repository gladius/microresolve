"""
Basic MicroResolve usage: routing, multi-intent, learning, export/import, discovery.

Run: python basic.py
"""

from microresolve import Router

# --- Setup ---
r = Router()
r.begin_batch()
r.add_intent("cancel_order", ["cancel my order", "I want to cancel", "stop my order from shipping"])
r.add_intent("track_order", ["where is my package", "track my order", "shipping status update"])
r.add_intent("refund", ["I want a refund", "get my money back", "return and refund"])
r.end_batch()

# --- Single routing ---
print("=== Single routing ===")
results = r.route("I need to cancel something")
for res in results:
    print(f"  {res['id']} (score: {res['score']:.2f})")

# --- Multi-intent ---
print("\n=== Multi-intent ===")
multi = r.route_multi("cancel my order and give me a refund")
for intent in multi["confirmed"]:
    print(f"  {intent['id']} (score: {intent['score']:.2f}, type: {intent['intent_type']})")

# --- Learning ---
print("\n=== Learning ===")
before = r.route("stop charging me")
print(f"  before: {before[0]['id'] if before else 'no match'} ({before[0]['score']:.2f})" if before else "  before: no match")

r.learn("stop charging me", "cancel_order")

after = r.route("stop charging me")
print(f"  after:  {after[0]['id']} ({after[0]['score']:.2f})")

# --- Export / Import ---
print("\n=== Export/Import ===")
json_str = r.export_json()
print(f"  exported: {len(json_str)} bytes")

r2 = Router.import_json(json_str)
result = r2.route("cancel this")
print(f"  imported route: {result[0]['id']}")

# --- Discovery ---
print("\n=== Discovery ===")
queries = [
    "cancel my order", "I want to cancel", "stop my order",
    "cancel the purchase", "cancel it please", "undo my order",
    "where is my package", "track order", "shipping update",
    "track my delivery", "order tracking", "delivery status",
] * 20

clusters = Router.discover(queries)
print(f"  discovered {len(clusters)} clusters from {len(queries)} queries")
for c in clusters:
    print(f"    {c['name']} (size: {c['size']}, terms: {c['top_terms'][:3]})")

print("\nAll working!")
