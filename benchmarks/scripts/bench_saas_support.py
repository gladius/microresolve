#!/usr/bin/env python3
"""SaaS customer support intent routing benchmark.

10 intents: payment_declined, refund_request, subscription_cancel,
subscription_upgrade, order_status, account_access, billing_dispute,
technical_support, data_export, feature_request.

Tests: cold start + 1 auto-learn pass with real LLM span extraction.

Usage:
  python3 benchmarks/scripts/bench_saas_support.py
"""

import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from lib import check_server, create_namespace, delete_namespace, load_seeds, _req

DATASETS = os.path.join(os.path.dirname(__file__), "../datasets")
NS = f"bench-saas-{int(time.time())}"

def route(ns, text):
    return _req("POST", "/api/route_multi", {"query": text, "log": False}, ns=ns)

def evaluate(ns, test):
    results = []
    for ex in test:
        query, expected = ex["text"], set(ex["intents"])
        t0 = time.perf_counter()
        resp = route(ns, query)
        lat = (time.perf_counter() - t0) * 1_000_000
        confirmed = set(r["id"] for r in resp.get("confirmed", []))
        ranked    = [r["id"] for r in resp.get("ranked", [])]
        k = len(expected)
        inter = expected & confirmed
        results.append({
            "text": query, "expected": expected, "confirmed": confirmed,
            "exact":      confirmed == expected,
            "precision":  len(inter)/len(confirmed) if confirmed else 0,
            "recall":     len(inter)/len(expected)  if expected  else 0,
            "topk":       expected.issubset(set(ranked[:k])),
            "missed":     expected - confirmed,
            "latency_us": lat,
        })
    return results

def metrics(results):
    n = len(results)
    p = sum(r["precision"] for r in results) / n * 100
    r = sum(r["recall"]    for r in results) / n * 100
    f1 = 2*p*r/(p+r) if (p+r) else 0
    exact = sum(1 for r in results if r["exact"]) / n * 100
    topk  = sum(1 for r in results if r["topk"])  / n * 100
    lats  = sorted(r["latency_us"] for r in results)
    p50   = lats[len(lats)//2] if lats else 0
    return {"exact": exact, "precision": p, "recall": r, "f1": f1, "topk": topk, "p50": p50}

def auto_learn(ns, results):
    learned = 0
    for r in results:
        if not r["exact"]:  # includes both misses AND false positives
            try:
                review = _req("POST", "/api/training/review", {
                    "message": r["text"],
                    "detected": list(r["confirmed"]),
                    "ground_truth": list(r["expected"]),
                }, ns=ns)
                _req("POST", "/api/training/apply", {
                    "query": r["text"], "result": review
                }, ns=ns)
                learned += 1
            except Exception as e:
                print(f"  [warn] auto-learn: {e}")
    return learned

def print_row(label, m, corrections=None):
    c = f"{corrections}" if corrections is not None else "-"
    print(f"  {label:<14} {m['exact']:>6.1f}%  {m['recall']:>6.1f}%  {m['precision']:>6.1f}%  {m['f1']:>6.1f}%  {m['topk']:>6.1f}%  {c:>6}  {m['p50']:>6.0f}µs")

def main():
    if not check_server(): sys.exit(1)

    seeds = json.load(open(os.path.join(DATASETS, "saas_support_seeds.json")))
    test  = json.load(open(os.path.join(DATASETS, "saas_support_test.json")))

    n_intents = len(seeds)
    n_seeds   = sum(len(v) for v in seeds.values())

    print("=" * 75)
    print(f"  SaaS Support Benchmark — {n_intents} intents, {len(test)} test queries")
    print(f"  Cold start + auto-learn with LLM span extraction (Haiku)")
    print("=" * 75)
    print(f"\n  {'':14} {'Exact':>6}  {'Recall':>6}  {'Precis':>6}  {'F1':>6}  {'Top-K':>6}  {'Corr':>6}  {'p50µs':>7}")
    print(f"  {'-'*72}")

    create_namespace(NS)
    load_seeds(NS, seeds)

    # Cold start
    before = evaluate(NS, test)
    print_row("cold start", metrics(before), 0)

    # Auto-learn
    corrections = auto_learn(NS, before)

    # After
    after = evaluate(NS, test)
    print_row("after learn", metrics(after), corrections)

    # Show remaining failures
    failures = [r for r in after if not r["exact"]]
    if failures:
        print(f"\n  Remaining failures ({len(failures)}/{len(test)}):")
        for r in failures[:5]:
            print(f"  · {r['text'][:60]}")
            print(f"    expected={sorted(r['expected'])}  missed={sorted(r['missed'])}")

    delete_namespace(NS)
    print(f"\n  Cleaned up {NS}")

if __name__ == "__main__":
    main()
