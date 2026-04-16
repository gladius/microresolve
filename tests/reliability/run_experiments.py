#!/usr/bin/env python3
"""
Full experiment runner — rigorous version.

For each intervention (equivalence classes, L3 seeding), measure:
  Cold baseline vs Cold + intervention × {mode A, B, C}
  Warm (corrections only) vs Warm + intervention best mode
  On dev set vs held-out validation set

Each configuration runs against a FRESH namespace cloned from the scale-test
snapshot, so experiments don't contaminate each other.

Results saved to results/{experiment_label}_{mode}_{state}_{set}.json
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import namespace_ops as ns_ops
import measure

ROOT = Path(__file__).parent
CORRECTIONS = json.loads((ROOT / "corrections.json").read_text())["corrections"]
EQUIVALENCE = json.loads((ROOT / "equivalence_classes.json").read_text())
SNAPSHOT = json.loads((ROOT / "scale_test_snapshot.json").read_text())

DEV_PATH = ROOT / "dataset.json"
VAL_PATH = ROOT / "validation.json"


def run_measurement(namespace: str, query_set: str, label: str, use_expansion: bool = False):
    """Execute the measure.py logic for the given configuration, save result, return summary."""
    # Reload equivalence in the measure module if needed
    measure.EQUIVALENCE_MAP = EQUIVALENCE if use_expansion else None

    path = DEV_PATH if query_set == "dev" else VAL_PATH
    dataset = json.loads(path.read_text())
    run = measure.evaluate(dataset, namespace)
    summary = measure.summarize(run)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{label}_{query_set}.json"
    out_path.write_text(json.dumps({"summary": summary, "run": run}, indent=2))

    return summary


def configure_namespace(mode: str, namespace: str, corrections: list = None):
    """
    Set up the namespace for a specific mode.
    Modes:
        baseline       — pristine clone, no intervention
        e2_b           — seed-phrase augmentation (variants as extra phrases)
        warm           — baseline + corrections applied
        e2_b_warm      — e2_b + corrections applied
        e4             — L3 cross-provider inhibition seeding (via synthetic corrections)
        e4_e2b         — e4 + seed augmentation
    """
    # Always start with a fresh clone
    ns_ops.clone_from_snapshot(namespace)

    if "e2_b" in mode:
        print(f"  Adding variant phrases to '{namespace}'...")
        ns_ops.add_variant_phrases(namespace, EQUIVALENCE, SNAPSHOT)

    if mode in ("e4", "e4_e2b"):
        l3_seeds = json.loads((ROOT / "l3_seed_corrections.json").read_text())["corrections"]
        print(f"  Applying {len(l3_seeds)} L3-seeding corrections to '{namespace}'...")
        stats = ns_ops.apply_corrections(namespace, l3_seeds)
        print(f"    applied={stats['applied']} skipped={stats['skipped']}")

    if mode.endswith("warm") or mode == "warm":
        corr = corrections or CORRECTIONS
        print(f"  Applying {len(corr)} corrections to '{namespace}'...")
        stats = ns_ops.apply_corrections(namespace, corr)
        print(f"    applied={stats['applied']} skipped={stats['skipped']}")


def format_summary(label: str, summary: dict) -> str:
    parts = [f"[{label:30s}]"]
    if "overall_top_1_pct" in summary:
        parts.append(f"t1={summary['overall_top_1_pct']:>5.1f}%")
    if "overall_top_3_pct" in summary:
        parts.append(f"t3={summary['overall_top_3_pct']:>5.1f}%")
    if "overall_multi_hit3_pct" in summary:
        parts.append(f"multi={summary['overall_multi_hit3_pct']:>5.1f}%")
    if "overall_multi_partial3_pct" in summary:
        parts.append(f"multi_p={summary['overall_multi_partial3_pct']:>5.1f}%")
    if "overall_oos_rejection_pct" in summary:
        parts.append(f"oos={summary['overall_oos_rejection_pct']:>5.1f}%")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+",
                        default=["baseline", "warm", "e2_a", "e2_b", "e2_c", "e2_b_warm"],
                        help="which experiments to run")
    parser.add_argument("--sets", nargs="+", default=["dev", "validation"],
                        help="which query sets")
    parser.add_argument("--skip-snapshot", action="store_true",
                        help="assume snapshot already exists")
    args = parser.parse_args()

    # Ensure snapshot exists
    if not args.skip_snapshot:
        ns_ops.snapshot("scale-test")

    # ── Build configuration table ─────────────────────────────────────────
    # Each entry: (label, namespace, setup_mode, use_expansion_at_query_time)
    configs = {
        "baseline":      ("bench_baseline",   "baseline",  False),
        "warm":          ("bench_warm",       "warm",      False),
        "e2_a":          ("bench_e2a",        "baseline",  True),   # query-time only
        "e2_b":          ("bench_e2b",        "e2_b",      False),  # seed augmentation only
        "e2_c":          ("bench_e2c",        "e2_b",      True),   # both seed + query-time
        "e2_b_warm":     ("bench_e2b_warm",   "e2_b_warm", False),  # seed augmentation + corrections
        "e4":            ("bench_e4",         "e4",        False),  # L3 cross-provider seeding only
        "e4_e2b":        ("bench_e4_e2b",     "e4_e2b",    False),  # L3 seeding + seed augmentation
    }

    results_summary = {}

    for exp in args.experiments:
        if exp not in configs:
            print(f"Unknown experiment: {exp}")
            continue
        namespace, setup_mode, use_expansion = configs[exp]
        print(f"\n{'═' * 78}\nSetting up '{exp}' → namespace '{namespace}', mode '{setup_mode}'\n{'═' * 78}")
        configure_namespace(setup_mode, namespace)

        for query_set in args.sets:
            label = f"{exp}_{query_set}"
            print(f"  Measuring on {query_set} set...")
            summary = run_measurement(namespace, query_set, exp, use_expansion)
            results_summary[label] = summary

    # ── Print final summary table ─────────────────────────────────────────
    print("\n" + "═" * 78)
    print("FINAL SUMMARY")
    print("═" * 78)
    print(f"\n{'Experiment':35s} {'top-1':>8s} {'top-3':>8s} {'multi-h3':>10s} {'multi-p3':>10s} {'oos-rej':>10s}")
    print("─" * 90)
    for label in sorted(results_summary.keys()):
        s = results_summary[label]
        t1 = f"{s.get('overall_top_1_pct', 0):.1f}%" if 'overall_top_1_pct' in s else "—"
        t3 = f"{s.get('overall_top_3_pct', 0):.1f}%" if 'overall_top_3_pct' in s else "—"
        mh = f"{s.get('overall_multi_hit3_pct', 0):.1f}%" if 'overall_multi_hit3_pct' in s else "—"
        mp = f"{s.get('overall_multi_partial3_pct', 0):.1f}%" if 'overall_multi_partial3_pct' in s else "—"
        oos = f"{s.get('overall_oos_rejection_pct', 0):.1f}%" if 'overall_oos_rejection_pct' in s else "—"
        print(f"{label:35s} {t1:>8s} {t3:>8s} {mh:>10s} {mp:>10s} {oos:>10s}")

    # Save aggregated
    agg_path = ROOT / "results" / "aggregated.json"
    agg_path.write_text(json.dumps(results_summary, indent=2))
    print(f"\nAggregated results: {agg_path}")


if __name__ == "__main__":
    main()
