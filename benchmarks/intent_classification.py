#!/usr/bin/env python3
"""Track 1 — Intent Classification Benchmark.

Runs CLINC150, BANKING77, HWU64, MASSIVE through MicroResolve.
Two passes per dataset: seed-only, then after direct learning from misses.

Usage:
  python3 run_track1_intent.py [clinc150|banking77|hwu64|massive]
  python3 run_track1_intent.py          # runs all
"""

import json
import os
import sys
import time

from lib import (
    check_server, create_namespace, delete_namespace,
    load_seeds, run_queries, apply_learning,
    compute_metrics, save_result, print_metrics,
)

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK1_DIR = os.path.join(BENCH_DIR, "track1")
RESULTS_DIR = os.path.join(BENCH_DIR, "results")


def run_dataset(name: str, seeds_path: str, test_path: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    with open(seeds_path) as f:
        seeds = json.load(f)
    with open(test_path) as f:
        test = json.load(f)

    ns = f"bench-{name.lower().replace(' ', '-')}-{int(time.time())}"

    print(f"  Namespace: {ns}")
    print(f"  Intents: {len(seeds)}, Seeds: {sum(len(v) for v in seeds.values())}, Test: {len(test)}")

    create_namespace(ns)

    # Load seeds
    t0 = time.time()
    added = load_seeds(ns, seeds)
    print(f"  Loaded {added} phrases in {time.time()-t0:.1f}s")

    # Pass 1: seed-only
    print(f"  Running seed-only pass ({len(test)} queries)…")
    results_seed = run_queries(ns, test)
    metrics_seed = compute_metrics(results_seed)
    print_metrics("  Seed-only", metrics_seed)

    misses = [r for r in results_seed if not r["top1_correct"]]
    print(f"  Misses: {len(misses)} → applying direct learning…")

    # Learning pass (no LLM — direct intent_id injection)
    learned = apply_learning(ns, results_seed)
    print(f"  Learned: {learned} reinforcements")

    # Pass 2: after learning
    print(f"  Running post-learning pass ({len(test)} queries)…")
    results_learned = run_queries(ns, test)
    metrics_learned = compute_metrics(results_learned)
    print_metrics("  + Learning", metrics_learned)

    delete_namespace(ns)

    return {
        "dataset": name,
        "intents": len(seeds),
        "seeds_per_intent": round(sum(len(v) for v in seeds.values()) / max(len(seeds), 1), 1),
        "test_examples": len(test),
        "seed_only": metrics_seed,
        "after_learning": metrics_learned,
        "learning_delta_top1": round(
            metrics_learned["top1_accuracy"] - metrics_seed["top1_accuracy"], 2
        ),
    }


def run_clinc150():
    return run_dataset(
        "CLINC150",
        f"{TRACK1_DIR}/clinc150_seeds.json",
        f"{TRACK1_DIR}/clinc150_test.json",
    )


def run_banking77():
    return run_dataset(
        "BANKING77",
        f"{TRACK1_DIR}/banking77_seeds.json",
        f"{TRACK1_DIR}/banking77_test.json",
    )


def run_hwu64():
    seeds_path = f"{TRACK1_DIR}/hwu64_seeds.json"
    test_path = f"{TRACK1_DIR}/hwu64_test.json"
    if not os.path.exists(seeds_path):
        print("\nHWU64: data not available, skipping")
        return None
    return run_dataset("HWU64", seeds_path, test_path)


def run_massive():
    langs = {
        "en_us": "English",
        "es_es": "Spanish",
        "fr_fr": "French",
        "ja_jp": "Japanese",
        "ar_sa": "Arabic",
    }
    results = {}
    for slug, label in langs.items():
        seeds_path = f"{TRACK1_DIR}/massive_{slug}_seeds.json"
        test_path = f"{TRACK1_DIR}/massive_{slug}_test.json"
        if not os.path.exists(seeds_path):
            print(f"\nMASSIVE {label}: data not available, skipping")
            continue
        result = run_dataset(f"MASSIVE ({label})", seeds_path, test_path)
        results[slug] = result
    return results


def main():
    if not check_server():
        print("Start the MicroResolve server first: cargo run --release --bin microresolve-studio --features server -- --data ./data")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    all_results = {}

    if target in ("all", "clinc150"):
        r = run_clinc150()
        all_results["clinc150"] = r
        save_result(f"{RESULTS_DIR}/clinc150.json", r)

    if target in ("all", "banking77"):
        r = run_banking77()
        all_results["banking77"] = r
        save_result(f"{RESULTS_DIR}/banking77.json", r)

    if target in ("all", "hwu64"):
        r = run_hwu64()
        if r:
            all_results["hwu64"] = r
            save_result(f"{RESULTS_DIR}/hwu64.json", r)

    if target in ("all", "massive"):
        r = run_massive()
        all_results["massive"] = r
        save_result(f"{RESULTS_DIR}/massive.json", r)

    # Summary table
    print(f"\n{'='*60}")
    print("  TRACK 1 SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<25} {'Intents':>7} {'Seeds/i':>7} {'Seed-only':>10} {'+Learning':>10} {'Delta':>7} {'p50 µs':>8}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*10} {'-'*10} {'-'*7} {'-'*8}")

    for key, r in all_results.items():
        if not r or isinstance(r, dict) and "seed_only" not in r:
            # massive is nested
            if isinstance(r, dict):
                for slug, sub in r.items():
                    if sub:
                        _print_row(sub)
        else:
            _print_row(r)


def _print_row(r):
    s = r.get("seed_only", {})
    l = r.get("after_learning", {})
    print(
        f"  {r['dataset']:<25} "
        f"{r['intents']:>7} "
        f"{r['seeds_per_intent']:>7.1f} "
        f"{s.get('top1_accuracy', 0):>9.1f}% "
        f"{l.get('top1_accuracy', 0):>9.1f}% "
        f"{r.get('learning_delta_top1', 0):>+6.1f}% "
        f"{s.get('latency_p50_us', 0):>7.0f}µs"
    )


if __name__ == "__main__":
    main()
