#!/usr/bin/env python3
"""Track 4 — Multi-Intent Benchmark.

Runs MixSNIPS and MixATIS through ASV.
Uses multi-intent metrics: exact match, partial match, F1, recall.

Usage:
  python3 run_track4_multiintent.py [mixsnips|mixatis]
  python3 run_track4_multiintent.py          # runs both
"""

import json
import os
import sys
import time

from lib import (
    check_server, create_namespace, delete_namespace,
    load_seeds, run_queries, apply_learning,
    compute_multiintent_metrics, save_result,
)

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BENCH_DIR, "results")


def run_dataset(name: str, seeds_path: str, test_path: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    with open(seeds_path) as f:
        seeds = json.load(f)
    with open(test_path) as f:
        test = json.load(f)

    ns = f"bench-{name.lower()}-{int(time.time())}"
    multi_count = sum(1 for e in test if len(e["intents"]) > 1)
    print(f"  Namespace: {ns}")
    print(f"  Intents: {len(seeds)}, Seeds: {sum(len(v) for v in seeds.values())}")
    print(f"  Test: {len(test)} ({multi_count} multi-intent, {len(test)-multi_count} single)")

    create_namespace(ns)
    t0 = time.time()
    added = load_seeds(ns, seeds)
    print(f"  Loaded {added} phrases in {time.time()-t0:.1f}s")

    # Pass 1: seed-only
    print(f"  Running seed-only pass…")
    results_seed = run_queries(ns, test)
    metrics_seed = compute_multiintent_metrics(results_seed)
    print(f"  Seed-only: exact={metrics_seed['exact_match']}%  partial={metrics_seed['partial_match']}%  "
          f"F1={metrics_seed['f1']}%  recall={metrics_seed['recall']}%  "
          f"p50={metrics_seed.get('latency_p50_us','?')}µs")

    # Learning pass
    misses = [r for r in results_seed if not r["top1_correct"]]
    print(f"  Misses: {len(misses)} → applying direct learning…")
    learned = apply_learning(ns, results_seed)
    print(f"  Learned: {learned} reinforcements")

    # Pass 2: after learning
    print(f"  Running post-learning pass…")
    results_learned = run_queries(ns, test)
    metrics_learned = compute_multiintent_metrics(results_learned)
    print(f"  + Learning: exact={metrics_learned['exact_match']}%  partial={metrics_learned['partial_match']}%  "
          f"F1={metrics_learned['f1']}%  recall={metrics_learned['recall']}%  "
          f"p50={metrics_learned.get('latency_p50_us','?')}µs")

    delete_namespace(ns)

    return {
        "dataset": name,
        "intents": len(seeds),
        "test_examples": len(test),
        "multi_intent_examples": multi_count,
        "seed_only": metrics_seed,
        "after_learning": metrics_learned,
        "f1_delta": round(metrics_learned["f1"] - metrics_seed["f1"], 2),
    }


def main():
    if not check_server():
        print("Start the ASV server first.")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    all_results = {}

    if target in ("all", "mixsnips"):
        r = run_dataset(
            "MixSNIPS",
            f"{BENCH_DIR}/mixsnips_seeds.json",
            f"{BENCH_DIR}/mixsnips_test.json",
        )
        all_results["mixsnips"] = r
        save_result(f"{RESULTS_DIR}/mixsnips.json", r)

    if target in ("all", "mixatis"):
        r = run_dataset(
            "MixATIS",
            f"{BENCH_DIR}/mixatis_seeds.json",
            f"{BENCH_DIR}/mixatis_test.json",
        )
        all_results["mixatis"] = r
        save_result(f"{RESULTS_DIR}/mixatis.json", r)

    print(f"\n{'='*60}")
    print("  TRACK 4 SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<12} {'Test':>6} {'Multi%':>7} {'Exact':>7} {'Partial':>8} {'F1':>6} {'Recall':>7} {'+F1':>6}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*6} {'-'*7} {'-'*6}")
    for name, r in all_results.items():
        s = r["seed_only"]
        l = r["after_learning"]
        multi_pct = round(r["multi_intent_examples"] / r["test_examples"] * 100, 0)
        print(f"  {r['dataset']:<12} {r['test_examples']:>6} {multi_pct:>6.0f}%"
              f" {l['exact_match']:>6.1f}%"
              f" {l['partial_match']:>7.1f}%"
              f" {l['f1']:>5.1f}%"
              f" {l['recall']:>6.1f}%"
              f" {r['f1_delta']:>+5.1f}%")


if __name__ == "__main__":
    main()
