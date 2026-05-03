#!/usr/bin/env python3
"""Smoke test for Option E (OOV-only synonym substitution at query time).

Setup: 10 Stripe-like tools imported via /api/import/mcp/apply (which
triggers the server's LLM Turn 1+2 to populate L1 with morphology +
abbreviation + synonym edges).

Test set: ~20 queries deliberately using vocabulary that mismatches the
function descriptions ("terminate the subscription" vs description's
"cancel"). Half single-word, half multi-word.

Two passes:
  1. grounded_l1 = false  (control — only morphology / abbreviation used)
  2. grounded_l1 = true   (Option E — also substitutes synonyms for OOV
                           tokens)

Reports top-1 hit rate for each pass + a per-query diff so you can see
exactly which queries flipped.
"""

import json
import os
import time
import urllib.request
import urllib.error

BASE = os.environ.get("MR_SERVER_URL", "http://localhost:3001")
NS = "test-e-stripe"

# 10 synthetic Stripe-like tools. Descriptions match what a real MCP server
# would expose. NO synonyms in the descriptions — that's the point.
TOOLS = [
    {"name": "create_payment_intent",   "description": "Create a PaymentIntent to charge a customer."},
    {"name": "cancel_payment_intent",   "description": "Cancel a PaymentIntent that has not yet been captured."},
    {"name": "create_subscription",     "description": "Start a recurring subscription for a customer."},
    {"name": "cancel_subscription",     "description": "Cancel an active subscription."},
    {"name": "update_subscription",     "description": "Modify the plan or quantity on an existing subscription."},
    {"name": "create_refund",           "description": "Issue a refund against a captured charge."},
    {"name": "create_customer",         "description": "Create a new customer record."},
    {"name": "update_customer",         "description": "Update fields on an existing customer."},
    {"name": "create_invoice",          "description": "Create a one-off invoice for a customer."},
    {"name": "list_charges",            "description": "List charges, optionally filtered by customer or date."},
]

# Test queries with vocabulary mismatched to the descriptions.
# (query, expected tool_name)
TEST_QUERIES = [
    # --- synonyms users might say ---
    ("terminate my subscription right now",          "cancel_subscription"),
    ("end the user's plan",                          "cancel_subscription"),
    ("kill the recurring billing",                   "cancel_subscription"),
    ("give back the money to John",                  "create_refund"),
    ("return the payment we received",               "create_refund"),
    ("issue a refund for that order",                "create_refund"),
    ("start charging the customer monthly",          "create_subscription"),
    ("set up recurring billing",                     "create_subscription"),
    ("modify the customer's plan",                   "update_subscription"),
    ("change subscription details",                  "update_subscription"),
    # --- abbreviations and casual phrasing ---
    ("make a new client",                            "create_customer"),
    ("register a new account",                       "create_customer"),
    ("edit customer info",                           "update_customer"),
    ("invoice the customer for $50",                 "create_invoice"),
    # --- vocabulary that actually matches descriptions (sanity) ---
    ("create a payment intent for John",             "create_payment_intent"),
    ("cancel that payment intent",                   "cancel_payment_intent"),
    ("list charges for last week",                   "list_charges"),
    # --- fully out-of-vocab queries (hardest cases) ---
    ("stop the auto-pay arrangement",                "cancel_subscription"),
    ("reverse that transaction",                     "create_refund"),
    ("enroll customer in monthly plan",              "create_subscription"),
]


def _req(method, path, body=None, ns=None, timeout=120):
    headers = {"Content-Type": "application/json"}
    if ns:
        headers["X-Namespace-ID"] = ns
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


def setup():
    """Drop + recreate namespace, MCP-import the tools (triggers L1 LLM aug)."""
    try:
        _req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception:
        pass
    _req("POST", "/api/namespaces", {"namespace_id": NS,
        "description": "Option E grounded_l1 smoke test"})
    tools_mcp = [{"name": t["name"], "description": t["description"],
                  "inputSchema": {"type": "object", "properties": {}}} for t in TOOLS]
    print("→ MCP-importing 10 tools (triggers Turn 1+2 LLM call to populate L1)...")
    t0 = time.perf_counter()
    res = _req("POST", "/api/import/mcp/apply", {
        "tools_json": json.dumps({"tools": tools_mcp}),
        "selected": [t["name"] for t in TOOLS],
        "domain": "",
    }, ns=NS, timeout=180)
    print(f"  imported {len(TOOLS)} tools in {time.perf_counter() - t0:.1f}s")
    return res


def measure(grounded_l1):
    hits = 0
    results = []
    for query, expected in TEST_QUERIES:
        res = _req("POST", "/api/resolve", {
            "query": query, "log": False,
        }, ns=NS)
        intents = res.get("intents") or []
        top = intents[0]["id"] if intents else "(none)"
        ok = top == expected
        hits += 1 if ok else 0
        results.append({
            "query": query,
            "expected": expected,
            "got": top,
            "hit": ok,
        })
    return hits, results


def main():
    setup()

    print("\n→ pass 1: grounded_l1 = false (control — morphology/abbreviation only)")
    h_off, r_off = measure(grounded_l1=False)
    print(f"  top-1: {h_off}/{len(TEST_QUERIES)} = {100 * h_off / len(TEST_QUERIES):.1f}%")

    print("\n→ pass 2: grounded_l1 = true (Option E — synonym sub for OOV tokens)")
    h_on, r_on = measure(grounded_l1=True)
    print(f"  top-1: {h_on}/{len(TEST_QUERIES)} = {100 * h_on / len(TEST_QUERIES):.1f}%")

    print(f"\n→ delta: +{h_on - h_off} ({100 * (h_on - h_off) / len(TEST_QUERIES):+.1f} pp)")

    # Per-query diff: queries where grounded_l1 flipped the result.
    flipped = []
    for off, on in zip(r_off, r_on):
        if off["hit"] != on["hit"] or off["got"] != on["got"]:
            flipped.append((off["query"], off["expected"], off["got"], on["got"]))
    if flipped:
        print(f"\n→ {len(flipped)} queries differed between passes:")
        for q, exp, off, on in flipped:
            mark_off = "✓" if off == exp else "✗"
            mark_on  = "✓" if on  == exp else "✗"
            print(f"  q: {q!r}")
            print(f"    expected: {exp}")
            print(f"    off [grounded=false] {mark_off} {off}")
            print(f"    on  [grounded=true ] {mark_on} {on}")
    else:
        print("\n→ no per-query differences between passes (synonyms not firing)")

    try:
        _req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception:
        pass


if __name__ == "__main__":
    main()
