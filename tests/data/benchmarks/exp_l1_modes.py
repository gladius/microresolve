#!/usr/bin/env python3
"""Experiment: Compare L1 modes on MixSNIPS and CLINC150.

Modes tested:
  full       — current default (morphology + synonym expansion)
  morph_only — morphological normalization only, no synonyms
  grounded   — synonyms only for tokens not already in L2
  no_l1      — raw tokens, no L1 at all

Usage:
  python3 exp_l1_modes.py
"""

import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import check_server, create_namespace, delete_namespace, load_seeds, _req, compute_metrics, compute_multiintent_metrics

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK1_DIR = os.path.join(BENCH_DIR, "track1")

MODES = [
    ("full",       {}),
    ("morph_only", {"morphology_only": True}),
    ("grounded",   {"grounded_l1": True}),
    ("no_l1",      {"disable_l1": True}),
]


def run_queries_mode(ns, examples, extra_params):
    results = []
    for ex in examples:
        text = ex["text"]
        expected = ex["intents"]
        try:
            body = {"query": text, "log": False, **extra_params}
            resp = _req("POST", "/api/route_multi", body, ns=ns)
            elapsed_us = resp.get("routing_us") or 0
            predicted = [r["id"] for r in resp.get("confirmed", [])]
            top1 = predicted[0] if predicted else None
            results.append({
                "text": text, "expected": expected, "predicted": predicted,
                "top1_correct": top1 in expected,
                "any_correct": any(p in expected for p in predicted),
                "latency_us": elapsed_us,
            })
        except Exception as e:
            results.append({"text": text, "expected": expected, "predicted": [],
                            "top1_correct": False, "any_correct": False,
                            "latency_us": 0, "error": str(e)})
    return results


def run_experiment(name, seeds_path, test_path, metric_fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    with open(seeds_path) as f: seeds = json.load(f)
    with open(test_path) as f: test = json.load(f)

    ns = f"exp-l1-{name.lower().replace(' ','')}-{int(time.time())}"
    print(f"  Intents: {len(seeds)}, Seeds: {sum(len(v) for v in seeds.values())}, Test: {len(test)}")
    create_namespace(ns)
    load_seeds(ns, seeds)

    print(f"\n  {'Mode':<12} {'Top1/F1':>8} {'Partial':>8} {'p50µs':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

    results_by_mode = {}
    for mode_name, params in MODES:
        results = run_queries_mode(ns, test, params)
        metrics = metric_fn(results)
        results_by_mode[mode_name] = metrics

        if "top1_accuracy" in metrics:
            primary = f"{metrics['top1_accuracy']:.1f}%"
            secondary = "-"
        else:
            primary = f"{metrics.get('f1', 0):.1f}%"
            secondary = f"{metrics.get('partial_match', 0):.1f}%"

        p50 = metrics.get("latency_p50_us", 0)
        print(f"  {mode_name:<12} {primary:>8} {secondary:>8} {p50:>7.0f}µs")

    delete_namespace(ns)
    return results_by_mode


if __name__ == "__main__":
    if not check_server():
        print("Start the MicroResolve server first.")
        sys.exit(1)

    # Experiment 1: MixSNIPS (7 intents, heavily overlapping)
    run_experiment(
        "MixSNIPS (7 intents)",
        f"{BENCH_DIR}/mixsnips_seeds.json",
        f"{BENCH_DIR}/mixsnips_test.json",
        compute_multiintent_metrics,
    )

    # Experiment 2: CLINC150 (150 intents, distinct)
    run_experiment(
        "CLINC150 (150 intents)",
        f"{TRACK1_DIR}/clinc150_seeds.json",
        f"{TRACK1_DIR}/clinc150_test.json",
        compute_metrics,
    )

    print("\nDone.")
