#!/usr/bin/env python3
"""MixATIS multi-intent benchmark.

17 intents | 828 multi-intent test queries
Seeds: ~20 phrases per intent from mixatis_seeds.json
Tests: mixatis_test.json

Usage:
  python3 bench_mixatis.py
"""

import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import check_server, create_namespace, delete_namespace, load_seeds, run_queries, compute_multiintent_metrics

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
NS = f"bench-mixatis-{int(time.time())}"

def main():
    if not check_server():
        sys.exit(1)

    seeds = json.load(open(os.path.join(BENCH_DIR, "mixatis_seeds.json")))
    test  = json.load(open(os.path.join(BENCH_DIR, "mixatis_test.json")))

    n_intents = len(seeds)
    n_seeds   = sum(len(v) for v in seeds.values())
    n_test    = len(test)

    print("=" * 60)
    print(f"  MixATIS Multi-Intent Benchmark")
    print(f"  {n_intents} intents | {n_seeds} seeds | {n_test} test queries")
    print("=" * 60)

    print(f"\n  Creating namespace: {NS}")
    create_namespace(NS)
    load_seeds(NS, seeds)
    print(f"  Seeded {n_seeds} phrases across {n_intents} intents")

    print(f"\n  Running {n_test} test queries...")
    results = run_queries(NS, test)

    m = compute_multiintent_metrics(results)

    print(f"\n  {'Metric':<22} {'Value':>8}")
    print(f"  {'-'*32}")
    print(f"  {'Exact match':<22} {m['exact_match']:>7.1f}%")
    print(f"  {'Partial match':<22} {m['partial_match']:>7.1f}%")
    print(f"  {'Precision':<22} {m['precision']:>7.1f}%")
    print(f"  {'Recall':<22} {m['recall']:>7.1f}%")
    print(f"  {'F1':<22} {m['f1']:>7.1f}%")
    print(f"  {'p50 latency':<22} {m.get('latency_p50_us','?'):>7}µs")
    print(f"  {'p95 latency':<22} {m.get('latency_p95_us','?'):>7}µs")
    print(f"  {'queries':<22} {m['n']:>8}")

    delete_namespace(NS)
    print(f"\n  Cleaned up namespace: {NS}")

if __name__ == "__main__":
    main()
