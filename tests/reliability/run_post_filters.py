#!/usr/bin/env python3
"""
Evaluate post-processing filters (FP filter, char-ngram tiebreaker) as
parallel layers on top of ASV's routing.

PRIMARY test: bench_dense (15 intents, 12 phrases each — realistic production)
SECONDARY test: scale-test (98 intents, 2-3 phrases — diagnostic for thin seeds)

For each namespace × each filter config × each query set, record metrics.
Compare filter configs to baseline (no filter) on the same namespace.
"""
import json
from pathlib import Path
import time

import measure
import namespace_ops as ns_ops
from post_filters import NgramFPFilter, CharNgramTiebreaker

ROOT = Path(__file__).parent


def load_dense_snapshot():
    """Build an in-memory snapshot structure for bench_dense from dense_seed_setup.json."""
    dense_setup = json.loads((ROOT / "dense_seed_setup.json").read_text())
    return [
        {"id": iid, "phrases_by_lang": {"en": phrases}, "intent_type": "action"}
        for iid, phrases in dense_setup["intents"].items()
    ]


def load_thin_snapshot():
    return json.loads((ROOT / "scale_test_snapshot.json").read_text())


def ensure_dense_namespace():
    """Ensure bench_dense is populated. Cheap check: do the intents exist?"""
    import urllib.request
    req = urllib.request.Request(
        "http://localhost:3001/api/intents",
        headers={"X-Namespace-ID": "bench_dense"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            intents = json.loads(r.read())
            if len(intents) >= 15:
                return
    except Exception:
        pass
    # Rebuild
    import run_dense
    run_dense.build_dense_namespace()


def run_set(namespace: str, queries: list, label: str) -> dict:
    dataset = {"queries": queries}
    run = measure.evaluate(dataset, namespace)
    summary = measure.summarize(run)
    out = ROOT / "results" / f"filter_{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "run": run}, indent=2))
    return summary


def compact_summary(s: dict) -> str:
    parts = []
    if "overall_top_1_pct" in s:
        parts.append(f"t1={s['overall_top_1_pct']:.1f}%")
    if "overall_top_3_pct" in s:
        parts.append(f"t3={s['overall_top_3_pct']:.1f}%")
    if "overall_multi_hit3_pct" in s:
        parts.append(f"multi-h3={s['overall_multi_hit3_pct']:.1f}%")
    if "overall_multi_partial3_pct" in s:
        parts.append(f"multi-p3={s['overall_multi_partial3_pct']:.1f}%")
    if "overall_oos_rejection_pct" in s:
        parts.append(f"oos={s['overall_oos_rejection_pct']:.1f}%")
    return " | ".join(parts)


def main():
    # ─── PRIMARY: bench_dense (rich data, production regime) ──────────────
    print("═" * 78)
    print("PRIMARY TEST: bench_dense (15 intents × 12 phrases, 91% baseline)")
    print("═" * 78)

    ensure_dense_namespace()
    dense_snap = load_dense_snapshot()
    dense_queries = json.loads((ROOT / "dense_queries.json").read_text())

    fp = NgramFPFilter(dense_snap, idf_min=1.8)
    tb = CharNgramTiebreaker(dense_snap, n=4)

    configs = [
        ("baseline",          False, False),
        ("fp_only",           True,  False),
        ("tiebreaker_only",   False, True),
        ("fp_plus_tiebreaker",True,  True),
    ]

    dense_results = {}
    for qset_name in ("dev", "validation"):
        queries = dense_queries[qset_name]
        for cfg_name, fp_on, tb_on in configs:
            measure.FP_FILTER = fp if fp_on else None
            measure.TIEBREAKER = tb if tb_on else None
            label = f"dense_{cfg_name}_{qset_name}"
            summary = run_set("bench_dense", queries, label)
            dense_results[label] = summary
            print(f"  [{label:45s}] {compact_summary(summary)}")

    # ─── SECONDARY: scale-test (thin seeds, diagnostic) ───────────────────
    print("\n" + "═" * 78)
    print("SECONDARY TEST: scale-test (98 intents × 2-3 phrases, 40% baseline)")
    print("═" * 78)

    thin_snap = load_thin_snapshot()
    # For scale-test, ensure bench_baseline exists (pristine clone)
    ns_ops.clone_from_snapshot("bench_baseline")

    fp_thin = NgramFPFilter(thin_snap, idf_min=2.0)  # higher threshold given more intents
    tb_thin = CharNgramTiebreaker(thin_snap, n=4)

    dev_queries = json.loads((ROOT / "dataset.json").read_text())["queries"]
    val_queries = json.loads((ROOT / "validation.json").read_text())["queries"]

    thin_results = {}
    for qset_name, queries in (("dev", dev_queries), ("validation", val_queries)):
        for cfg_name, fp_on, tb_on in configs:
            measure.FP_FILTER = fp_thin if fp_on else None
            measure.TIEBREAKER = tb_thin if tb_on else None
            label = f"thin_{cfg_name}_{qset_name}"
            summary = run_set("bench_baseline", queries, label)
            thin_results[label] = summary
            print(f"  [{label:45s}] {compact_summary(summary)}")

    # ─── Final combined summary ────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("FINAL SUMMARY — post-filter effects on both regimes")
    print("═" * 78)

    all_results = {**dense_results, **thin_results}
    out = ROOT / "results" / "post_filter_aggregated.json"
    out.write_text(json.dumps({k: v for k, v in all_results.items()}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
