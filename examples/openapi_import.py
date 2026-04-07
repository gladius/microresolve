"""
Import intents from an OpenAPI spec — turn API endpoints into routable intents.

Each API operation becomes an intent, with seeds from summary/description
and endpoint metadata stored for direct routing to the API.
"""

import json
from asv_router import Router


def import_openapi(router: Router, spec: dict):
    """Import all operations from an OpenAPI 3.x spec as intents."""
    created = []

    paths = spec.get("paths", {})
    for path, methods in paths.items():
        for method, operation in methods.items():
            if method in ("parameters", "summary", "description"):
                continue  # skip path-level fields

            op_id = operation.get("operationId", "")
            summary = operation.get("summary", "")
            description = operation.get("description", "")

            if not op_id:
                # Generate from path: /orders/{id}/cancel → cancel_order
                parts = path.strip("/").split("/")
                op_id = "_".join(p for p in reversed(parts) if not p.startswith("{"))

            # Build seeds from summary and description
            seeds = []
            if summary:
                seeds.append(summary.lower())
            if description:
                # Split long descriptions into sentences
                for sent in description.split(". "):
                    sent = sent.strip().lower()
                    if len(sent) > 10:
                        seeds.append(sent)

            if not seeds:
                continue

            # Convert operationId to snake_case intent name
            intent_name = op_id.replace("-", "_").lower()

            router.add_intent(intent_name, seeds)
            router.set_metadata(intent_name, "endpoint", [f"{method.upper()} {path}"])
            router.set_metadata(intent_name, "operation_id", [op_id])
            created.append(intent_name)

    return created


# Example OpenAPI spec
SAMPLE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "E-Commerce API", "version": "1.0.0"},
    "paths": {
        "/orders/{id}": {
            "get": {
                "operationId": "getOrder",
                "summary": "Get order details",
                "description": "Retrieve the current status and details of an order. Shows tracking information and estimated delivery date.",
            }
        },
        "/orders/{id}/cancel": {
            "post": {
                "operationId": "cancelOrder",
                "summary": "Cancel an order",
                "description": "Cancel a pending order that hasn't shipped yet. Refund is issued within 3-5 business days.",
            }
        },
        "/orders/{id}/return": {
            "post": {
                "operationId": "returnOrder",
                "summary": "Return an order",
                "description": "Initiate a return for a delivered order. Must be within 30 days of delivery.",
            }
        },
        "/account/address": {
            "put": {
                "operationId": "updateAddress",
                "summary": "Update shipping address",
                "description": "Change the default shipping address on your account.",
            }
        },
        "/support/callback": {
            "post": {
                "operationId": "requestCallback",
                "summary": "Request a callback from support",
                "description": "Schedule a phone callback from our customer support team.",
            }
        },
    },
}

if __name__ == "__main__":
    router = Router()
    created = import_openapi(router, SAMPLE_SPEC)
    print(f"Created {len(created)} intents from OpenAPI spec:")
    for name in created:
        print(f"  {name}")

    print("\nRouting tests:")
    queries = [
        "cancel my order",
        "where is my package",
        "I want to return this item",
        "change my address",
        "can I talk to someone",
    ]
    for q in queries:
        results = router.route(q)
        if results:
            print(f"  \"{q}\" → {results[0]['id']} (score: {results[0]['score']:.2f})")
        else:
            print(f"  \"{q}\" → no match")
