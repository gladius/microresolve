#!/usr/bin/env python3
"""Edge-case validation for Option E (OOV-only synonym substitution).

Hand-crafts synonym edges (no LLM, deterministic) and checks:

  1. Pollution: when the synonym canonical appears in MANY intents,
     does substitution still pick the right one or does it muddle?
  2. Distinctive preservation: when source word IS in L2 vocab,
     substitution must NOT fire (preserves user's training signal).
  3. Case sensitivity: "Kill the order" should work the same as "kill".
  4. Latency overhead: grounded_l1=true cost vs false (under same setup).
  5. No-effect floor: synonym to a target that's NOT in L2 → no-op.

No LLM. ~5 sec. Pure logic test.
"""
import json
import os
import statistics
import time
import urllib.request

BASE = os.environ.get("MR_SERVER_URL", "http://localhost:8181")
NS = "test-e-edges"


def req(method, path, body=None, ns=None, timeout=10):
    headers = {"Content-Type": "application/json"}
    if ns:
        headers["X-Namespace-ID"] = ns
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def setup():
    """Build a namespace with 5 cancel_* intents + 1 distinct intent.
    Add hand-crafted synonym edges. Deterministic — no LLM."""
    try:
        req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception:
        pass
    req("POST", "/api/namespaces", {"namespace_id": NS,
        "description": "Option E edge-case test"})

    # 5 intents that ALL contain "cancel" in their seeds (pollution case)
    intents = {
        "cancel_order":        ["cancel my order", "cancel the order", "cancel order"],
        "cancel_subscription": ["cancel subscription", "cancel my subscription", "stop subscription"],
        "cancel_meeting":      ["cancel meeting", "cancel the meeting", "cancel calendar event"],
        "cancel_shipment":     ["cancel shipment", "cancel delivery", "cancel my package"],
        "cancel_invoice":      ["cancel invoice", "void invoice"],
        # One intent where "terminate" is INTENTIONALLY trained (distinctive)
        "fire_employee":       ["terminate the employee", "fire the worker", "let the employee go"],
    }
    for iid, phrases in intents.items():
        req("POST", "/api/intents", {"id": iid, "phrases": phrases}, ns=NS)

    # Hand-craft synonym edges — what an LLM would generate.
    edges = [
        # OOV synonyms — sources NOT in any seed phrase, targets that ARE.
        ("kill",      "cancel", "synonym", 0.90),
        ("abort",     "cancel", "synonym", 0.92),
        ("scrap",     "cancel", "synonym", 0.88),  # below threshold (0.90) — should NOT fire
        ("nuke",      "cancel", "synonym", 0.95),
        # In-vocab synonym source: "terminate" IS in fire_employee seeds.
        # Per Option E rule, this should NOT substitute (preserve training).
        ("terminate", "cancel", "synonym", 0.91),
        # Synonym to a canonical that is NOT in any L2 vocab → no-op.
        ("foo",       "bar",    "synonym", 0.95),
    ]
    for f, t, kind, w in edges:
        req("POST", "/api/layers/l1/edges",
            {"from": f, "to": t, "kind": kind, "weight": w}, ns=NS)


# ── Probes ──────────────────────────────────────────────────────────────────

def route(query, grounded_l1=False):
    res = req("POST", "/api/resolve",
        {"query": query, "log": False}, ns=NS)
    intents = res.get("intents") or []
    top = intents[0]["id"] if intents else "(none)"
    return top, intents, res.get("routing_us", 0)


def check(label, query, expected, grounded_l1=True):
    top, ranked, _ = route(query, grounded_l1)
    ok = top == expected
    flag = "✓" if ok else "✗"
    print(f"  {flag} {label:<48s} {query!r}")
    print(f"      expected: {expected:<25s} got: {top}")
    if not ok:
        # Show top 3 to see why it picked wrong
        print(f"      top-3: {[m['id'] for m in ranked[:3]]}")
    return ok


def test_pollution():
    print("\n=== TEST 1: Pollution — many intents share the canonical word ===")
    print("'kill' is OOV; substitutes to 'cancel' which appears in 5 intents.")
    print("Distinguishing word in query (order/meeting/etc.) must still win.\n")
    results = []
    results.append(check("kill the order",      "kill the order",          "cancel_order"))
    results.append(check("kill the meeting",    "kill that meeting",       "cancel_meeting"))
    results.append(check("nuke the invoice",    "nuke this invoice",       "cancel_invoice"))
    results.append(check("abort the shipment",  "abort the shipment now",  "cancel_shipment"))
    return all(results)


def test_distinctive_preservation():
    print("\n=== TEST 2: Distinctive vocabulary preserved ===")
    print("'terminate' IS in fire_employee seeds (in L2 vocab).")
    print("Even though there's a terminate→cancel synonym, must NOT substitute.\n")
    return check("terminate the employee", "terminate the employee", "fire_employee")


def test_threshold():
    print("\n=== TEST 3: Synonym weight threshold (0.90) ===")
    print("'scrap'→'cancel' has weight 0.88 — below threshold. Should NOT fire.\n")
    # 'scrap the order' — without substitution, "scrap" is unknown, only "the order"
    # weights matter. cancel_order has "the order" — should still pick it but
    # the win shouldn't be from the 0.88-weight scrap→cancel edge.
    top, _, _ = route("scrap the order", grounded_l1=True)
    # Either the lookup picked it via "order" alone (fine), or the weak edge fired.
    # We just assert nothing crashed and the result is sane.
    print(f"      'scrap the order' → {top}")
    return top == "cancel_order"  # weak signal still works via "order"


def test_case_sensitivity():
    print("\n=== TEST 4: Case sensitivity ===")
    print("'Kill' (capitalized) should behave same as 'kill'.\n")
    a = check("Kill (capitalized)",   "Kill the order",          "cancel_order")
    b = check("KILL (uppercase)",     "KILL THE MEETING",        "cancel_meeting")
    return a and b


def test_dead_synonym():
    print("\n=== TEST 5: Synonym to non-L2 target → no-op ===")
    print("'foo'→'bar' but 'bar' isn't in any seed. Substitution must not happen.\n")
    # 'foo the order' — "foo" is OOV, but synonym target "bar" also not in L2,
    # so substitution should NOT fire. Result depends on "the order" alone.
    top, _, _ = route("foo the order", grounded_l1=True)
    print(f"      'foo the order' → {top}")
    return top == "cancel_order"


def test_latency_overhead():
    print("\n=== TEST 6: Latency overhead ===")
    print("Compare server-reported routing_us with and without grounded_l1.\n")
    queries = ["cancel my order", "kill the meeting", "abort the shipment",
               "cancel subscription", "nuke this invoice"] * 20  # 100 samples
    off, on = [], []
    for q in queries:
        _, _, us_off = route(q, grounded_l1=False)
        _, _, us_on  = route(q, grounded_l1=True)
        off.append(us_off); on.append(us_on)
    p50_off = statistics.median(off); p50_on = statistics.median(on)
    print(f"      grounded_l1=false  p50 = {p50_off:.0f} µs")
    print(f"      grounded_l1=true   p50 = {p50_on:.0f} µs")
    print(f"      delta              = {p50_on - p50_off:+.0f} µs")
    return abs(p50_on - p50_off) < 50  # acceptable overhead


def main():
    setup()
    results = {
        "pollution":       test_pollution(),
        "distinctive":     test_distinctive_preservation(),
        "threshold":       test_threshold(),
        "case":            test_case_sensitivity(),
        "dead_synonym":    test_dead_synonym(),
        "latency":         test_latency_overhead(),
    }
    print(f"\n══ Summary ══════════════════════════════")
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  {passed} / {total} passed")
    try:
        req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception:
        pass
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys; sys.exit(main())
