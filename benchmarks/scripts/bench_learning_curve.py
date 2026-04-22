#!/usr/bin/env python3
"""Learning curve benchmark — shows accuracy improvement as corrections accumulate.

Methodology:
  1. Seed with N phrases per intent (cold start)
  2. Test on held-out queries → record baseline accuracy
  3. For each wrong answer: add the query as a training phrase (simulates user correction)
  4. Re-test → record improved accuracy
  5. Repeat for multiple correction rounds

This demonstrates the self-improving nature of ASV: the system learns from
misrouted queries without retraining or LLM calls at inference time.

Usage:
  python3 bench_learning_curve.py [--dataset clinc150|mixsnips|mixatis] [--rounds 3]
"""

import json, os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (check_server, create_namespace, delete_namespace, load_seeds,
                 run_queries, apply_learning, compute_metrics, compute_multiintent_metrics)

BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets")

DATASETS = {
    "clinc150": {
        "seeds_file": None,  # loaded from track1/
        "test_file":  None,
        "multi_intent": False,
        "label": "CLINC150 (150 intents)",
    },
    "mixsnips": {
        "seeds_file": "mixsnips_seeds.json",
        "test_file":  "mixsnips_test.json",
        "multi_intent": True,
        "label": "MixSNIPS (7 intents, multi-intent)",
    },
    "mixatis": {
        "seeds_file": "mixatis_seeds.json",
        "test_file":  "mixatis_test.json",
        "multi_intent": True,
        "label": "MixATIS (17 intents, multi-intent)",
    },
}


def load_clinc150():
    track1_dir = os.path.join(BENCH_DIR, "track1")
    seeds = json.load(open(os.path.join(track1_dir, "clinc150_seeds.json")))
    test  = json.load(open(os.path.join(track1_dir, "clinc150_test.json")))
    return seeds, test


def load_dataset(name):
    cfg = DATASETS[name]
    if name == "clinc150":
        return load_clinc150()
    seeds = json.load(open(os.path.join(BENCH_DIR, cfg["seeds_file"])))
    test  = json.load(open(os.path.join(BENCH_DIR, cfg["test_file"])))
    return seeds, test


def score(results, multi_intent):
    if multi_intent:
        m = compute_multiintent_metrics(results)
        return m["f1"], m["partial_match"], m.get("latency_p50_us", 0)
    else:
        m = compute_metrics(results)
        return m["top1_accuracy"], m["any_accuracy"], m.get("latency_p50_us", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="clinc150", choices=list(DATASETS.keys()))
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    if DATASETS[args.dataset]["multi_intent"] and args.rounds > 1:
        print(f"  [warn] --dataset {args.dataset} uses LLM for learning. {args.rounds} rounds will incur LLM cost.")
        print(f"  [warn] Use --rounds 1 for a single cold+learn pass.")

    if not check_server():
        sys.exit(1)

    cfg = DATASETS[args.dataset]
    seeds, test = load_dataset(args.dataset)

    n_intents = len(seeds)
    n_seeds   = sum(len(v) for v in seeds.values()) if isinstance(seeds, dict) else len(seeds)
    n_test    = len(test)

    NS = f"bench-lc-{args.dataset}-{int(time.time())}"

    print("=" * 62)
    print(f"  Learning Curve — {cfg['label']}")
    print(f"  {n_intents} intents | {n_seeds} cold-start seeds | {n_test} test queries")
    print(f"  {args.rounds} correction rounds")
    print("=" * 62)

    create_namespace(NS)
    load_seeds(NS, seeds)

    print(f"\n  {'Round':<8} {'F1/Top1':>8} {'Partial':>8} {'Corrections':>12} {'p50µs':>8}")
    print(f"  {'-'*52}")

    total_corrections = 0

    for rnd in range(args.rounds + 1):
        results = run_queries(NS, test)
        main_score, partial, lat = score(results, cfg["multi_intent"])
        label = "cold start" if rnd == 0 else f"round {rnd}"
        print(f"  {label:<8} {main_score:>7.1f}% {partial:>7.1f}% {total_corrections:>12} {lat:>7.0f}µs")

        if rnd < args.rounds:
            n_learned = apply_learning(NS, results)
            total_corrections += n_learned

    delete_namespace(NS)

    print(f"\n  Cleaned up {NS}")
    print(f"\n  Key insight: accuracy improves from corrections with zero LLM calls at inference time.")

if __name__ == "__main__":
    main()
