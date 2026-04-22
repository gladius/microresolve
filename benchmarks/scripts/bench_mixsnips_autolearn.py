#!/usr/bin/env python3
"""MixSNIPS auto-learn benchmark with top-1/3/5 all-intents metric.

Metrics:
  top-N all-found: % of queries where ALL expected intents appear in top-N ranked results
  exact match:     % of queries where confirmed == expected exactly
  partial match:   % where at least one expected intent is confirmed

Learning: uses the real auto-learn pipeline (/api/training/review + /api/training/apply)
with ground truth provided (skips Turn 1, only Turn 2 span extraction runs).

Usage:
  python3 bench_mixsnips_autolearn.py [--rounds 2]
"""

import json, os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import check_server, create_namespace, delete_namespace, load_seeds, _req

BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets")

def route_query(ns, text):
    return _req("POST", "/api/route_multi", {"query": text, "log": False}, ns=ns)

def auto_learn(ns, query, detected, expected):
    """Run real auto-learn pipeline with ground truth. Returns spans learned."""
    try:
        review = _req("POST", "/api/training/review", {
            "message": query,
            "detected": detected,
            "ground_truth": expected,
        }, ns=ns)
        if review:
            _req("POST", "/api/training/apply", {
                "query": query,
                "result": review,
            }, ns=ns)
            return review.get("spans_to_learn", [])
    except Exception as e:
        print(f"  [warn] auto-learn failed: {e}")
    return []

def evaluate(ns, test, label="evaluating"):
    """Route all test queries. Returns per-query results with top-N ranked."""
    results = []
    n = len(test)
    for i, ex in enumerate(test):
        if i % 100 == 0:
            print(f"  [{label}] {i}/{n}...", flush=True)
        query = ex["text"]
        expected = set(ex["intents"])
        t0 = time.perf_counter()
        resp = route_query(ns, query)
        lat = (time.perf_counter() - t0) * 1_000_000

        confirmed = set(r["id"] for r in resp.get("confirmed", []))
        ranked = [r["id"] for r in resp.get("ranked", [])]

        top1 = set(ranked[:1])
        top3 = set(ranked[:3])
        top5 = set(ranked[:5])

        results.append({
            "text": query,
            "expected": expected,
            "confirmed": confirmed,
            "ranked": ranked,
            "top1_all": expected.issubset(top1),
            "top3_all": expected.issubset(top3),
            "top5_all": expected.issubset(top5),
            "exact": confirmed == expected,
            "partial": bool(confirmed & expected),
            "missed": expected - confirmed,
            "latency_us": lat,
        })
    return results

def print_metrics(label, results, corrections=0):
    n = len(results)
    top1  = sum(1 for r in results if r["top1_all"])
    top3  = sum(1 for r in results if r["top3_all"])
    top5  = sum(1 for r in results if r["top5_all"])
    exact = sum(1 for r in results if r["exact"])
    part  = sum(1 for r in results if r["partial"])
    lats  = sorted(r["latency_us"] for r in results if r["latency_us"] > 0)
    p50   = lats[len(lats)//2] if lats else 0

    corr_str = f"{corrections:>6}" if corrections > 0 else "      -"
    print(f"  {label:<12} {top1/n*100:>6.1f}%  {top3/n*100:>6.1f}%  {top5/n*100:>6.1f}%  "
          f"{exact/n*100:>6.1f}%  {part/n*100:>6.1f}%  {corr_str}  {p50:>6.0f}µs")

def main():
    if not check_server():
        sys.exit(1)

    seeds = json.load(open(os.path.join(BENCH_DIR, "mixsnips_seeds.json")))
    test  = json.load(open(os.path.join(BENCH_DIR, "mixsnips_test.json")))

    n_intents = len(seeds)
    n_seeds   = sum(len(v) for v in seeds.values())
    n_test    = len(test)
    n_multi   = sum(1 for ex in test if len(ex["intents"]) > 1)

    NS = f"bench-ms-al-{int(time.time())}"

    print("=" * 80)
    print(f"  MixSNIPS Auto-Learn Benchmark (real LLM pipeline)")
    print(f"  {n_intents} intents | {n_seeds} seeds | {n_test} test ({n_multi} multi-intent)")
    print(f"  cold start + 1 learning pass")
    print("=" * 80)
    print(f"\n  {'':12} {'Top-1':>7}  {'Top-3':>6}  {'Top-5':>6}  {'Exact':>6}  {'Partial':>7}  {'Corrections':>6}  {'p50µs':>7}")
    print(f"  {'-'*76}")

    create_namespace(NS)
    load_seeds(NS, seeds)

    # Cold start
    results = evaluate(NS, test, label="cold start")
    print_metrics("cold start", results, 0)

    # Single learning pass — all non-exact queries: misses get phrases, false positives get suppression
    non_exact = [r for r in results if not r["exact"]]
    missed_count = sum(1 for r in non_exact if r["missed"])
    fp_count = len(non_exact) - missed_count
    print(f"\n  Learning from {len(non_exact)} non-exact queries ({missed_count} misses, {fp_count} false positives)...", flush=True)
    corrections = 0
    for i, r in enumerate(non_exact):
        if i % 25 == 0:
            print(f"  [learning] {i}/{len(non_exact)}...", flush=True)
        spans = auto_learn(NS, r["text"], list(r["confirmed"]), list(r["expected"]))
        if spans:
            corrections += 1
    print(f"  [learning] done — {corrections}/{len(non_exact)} produced spans", flush=True)

    # After learning
    results = evaluate(NS, test, label="after learn")
    print_metrics("after learn", results, corrections)

    delete_namespace(NS)
    print(f"\n  Cleaned up {NS}")

if __name__ == "__main__":
    main()
