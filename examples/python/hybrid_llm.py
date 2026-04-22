"""
Hybrid routing: MicroResolve handles the easy 80%, LLM handles the hard 20%.

This is the core cost-reduction pattern. At 10M queries/day:
  - LLM only: $10,000/day
  - Hybrid:    $2,000/day (80% free via MicroResolve, 20% LLM fallback)
"""

import anthropic  # pip install anthropic
from microresolve import Router

# Initialize ASV router
router = Router()
router.add_intent("cancel_order", ["cancel my order", "stop my order", "I want to cancel"])
router.add_intent("track_order", ["where is my package", "track my order", "shipping status"])
router.add_intent("refund", ["I want a refund", "get my money back", "return and refund"])

# LLM client (only used for low-confidence queries)
client = anthropic.Anthropic()

CONFIDENCE_THRESHOLD = 0.8


def route_query(query: str) -> dict:
    """Route a query using ASV first, LLM fallback for low confidence."""

    # Step 1: ASV routing (30μs, $0)
    results = router.route(query)

    if results and results[0]["score"] >= CONFIDENCE_THRESHOLD:
        # High confidence — ASV handles it directly
        return {
            "intent": results[0]["id"],
            "score": results[0]["score"],
            "source": "asv",
            "cost": 0.0,
        }

    # Step 2: Low confidence — ask LLM ($0.01)
    intents = router.intent_ids()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f"Classify this query into one of these intents: {intents}\n\nQuery: {query}\n\nRespond with just the intent name.",
        }],
    )
    llm_intent = response.content[0].text.strip()

    # Teach ASV for next time (so this query won't need LLM again)
    if llm_intent in intents:
        router.learn(query, llm_intent)

    return {
        "intent": llm_intent,
        "score": 0.0,
        "source": "llm",
        "cost": 0.001,
    }


if __name__ == "__main__":
    # Test queries
    queries = [
        "cancel my order please",          # high confidence → ASV
        "I changed my mind about buying",   # low confidence → LLM → learns
        "I changed my mind about buying",   # now ASV handles it (learned)
        "where is my stuff",                # high confidence → ASV
    ]

    for q in queries:
        result = route_query(q)
        print(f"  [{result['source']:3s}] {q:40s} → {result['intent']} (${result['cost']:.3f})")
