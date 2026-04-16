#!/usr/bin/env python3
"""
Calibration analysis — does confidence_ratio (top1/(top1+top2)) discriminate
OOS queries from in-scope queries better than raw score threshold?

Runs against the saved baseline.json and produces a threshold-sweep table
showing true-OOS-rejection vs false-rejection-of-in-scope at each threshold.

This doesn't require re-running ASV or any LLM calls — just analyses existing data.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
data_path = ROOT / "results" / "baseline.json"
results = json.loads(data_path.read_text())["run"]["results"]

# Split by whether the query is truly OOS or in-scope single-intent
oos = [r for r in results if r.get("expected") is None and r.get("expected_multi") is None]
in_scope = [r for r in results if r.get("expected") is not None]

print(f"\nTotal OOS queries: {len(oos)}")
print(f"Total in-scope single-intent queries: {len(in_scope)}")

# For each threshold type, compute: at threshold T, what fraction of OOS
# gets rejected, vs what fraction of in-scope gets falsely rejected?

def sweep(metric_fn, thresholds, name):
    print(f"\n──── {name} ────")
    print(f"  threshold  |  OOS-rejected  |  in-scope-wrongly-rejected  |  F1(OOS)")
    for t in thresholds:
        oos_rejected = sum(1 for r in oos if metric_fn(r) < t)
        oos_rejection_pct = oos_rejected / len(oos) * 100
        inscope_falsely_rejected = sum(1 for r in in_scope if metric_fn(r) < t)
        inscope_reject_pct = inscope_falsely_rejected / len(in_scope) * 100
        # F1 on OOS class: precision = oos_rejected / (oos_rejected + inscope_wrongly_rejected)
        # recall = oos_rejected / total_oos
        tp = oos_rejected
        fp = inscope_falsely_rejected
        fn = len(oos) - oos_rejected
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        print(f"  {t:>9.2f}  |  {oos_rejection_pct:>5.1f}% ({oos_rejected:>2}/{len(oos)})  |  "
              f"{inscope_reject_pct:>5.1f}% ({inscope_falsely_rejected:>2}/{len(in_scope)})    |  {f1:.3f}")

# Sweep raw score
score_fn = lambda r: r["top_1_score"]
sweep(score_fn, [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0], "Raw top-1 score threshold")

# Sweep confidence ratio
ratio_fn = lambda r: r["confidence_ratio"]
sweep(ratio_fn, [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90], "Confidence ratio threshold")

# Combined: score >= X AND ratio >= Y (reject if either fails)
print("\n──── Combined: reject if score < X OR ratio < Y ────")
print(f"  score_thr  ratio_thr  | OOS-rejected | in-scope-wrongly | F1")
for score_t in [0.8, 1.0, 1.2, 1.5]:
    for ratio_t in [0.55, 0.60, 0.65, 0.70]:
        oos_rejected = sum(1 for r in oos if r["top_1_score"] < score_t or r["confidence_ratio"] < ratio_t)
        inscope_falsely_rejected = sum(1 for r in in_scope if r["top_1_score"] < score_t or r["confidence_ratio"] < ratio_t)
        tp, fp, fn = oos_rejected, inscope_falsely_rejected, len(oos) - oos_rejected
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        oos_pct = oos_rejected / len(oos) * 100
        in_pct = inscope_falsely_rejected / len(in_scope) * 100
        print(f"  {score_t:>6.1f}    {ratio_t:>6.2f}   |  {oos_pct:>5.1f}%      | {in_pct:>5.1f}%          | {f1:.3f}")

# Raw data dump for inspection
print("\n──── OOS queries and their signatures ────")
for r in sorted(oos, key=lambda r: -r["top_1_score"]):
    print(f"  score={r['top_1_score']:.2f}  ratio={r['confidence_ratio']:.2f}  disp={r['disposition']:15s}  top1={r['top_1']}  query={r['query']!r}")

print("\n──── In-scope queries with lowest confidence (potential false rejections) ────")
for r in sorted(in_scope, key=lambda r: r["confidence_ratio"])[:10]:
    print(f"  score={r['top_1_score']:.2f}  ratio={r['confidence_ratio']:.2f}  correct={r.get('hit_1')}  top1={r['top_1']}  expected={r['expected']}  query={r['query']!r}")
