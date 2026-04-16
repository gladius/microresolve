#!/usr/bin/env python3
"""
Scale test: 125 intents across 5 domains, 12 phrases each.
Tests whether ASV maintains 90%+ top-1 / 97%+ top-3 at 8x the bench_dense scale.
"""
import json
import time
from pathlib import Path

import namespace_ops as ns_ops
import measure

ROOT = Path(__file__).parent
SETUP = json.loads((ROOT / "scale_125_setup.json").read_text())
QUERIES = json.loads((ROOT / "scale_125_queries.json").read_text())
NS = SETUP["_meta"]["namespace"]


def build_namespace():
    print(f"Building '{NS}' — {len(SETUP['intents'])} intents, {sum(len(v) for v in SETUP['intents'].values())} phrases...")
    ns_ops.delete_namespace(NS)
    time.sleep(0.3)
    ns_ops.create_namespace(NS)
    time.sleep(0.3)
    added = 0
    for intent_id, phrases in SETUP["intents"].items():
        body = {
            "id": intent_id,
            "phrases_by_lang": {"en": phrases},
            "intent_type": "action",
        }
        code, _ = ns_ops._post("/api/intents/multilingual", body, ns=NS)
        if code in (200, 201):
            added += 1
        else:
            print(f"  WARN: {intent_id} -> {code}")
    print(f"Built '{NS}': {added}/{len(SETUP['intents'])} intents")


def run_set(queries: list, label: str, use_tiebreaker: bool):
    measure.SERVER_TIEBREAKER = use_tiebreaker
    dataset = {"queries": queries}
    run = measure.evaluate(dataset, NS)
    summary = measure.summarize(run)
    out = ROOT / "results" / f"scale125_{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "run": run}, indent=2))
    return summary


def pretty(label: str, s: dict) -> str:
    parts = [f"[{label:38s}]"]
    if "overall_top_1_pct" in s:
        parts.append(f"t1={s['overall_top_1_pct']:>5.1f}%")
    if "overall_top_3_pct" in s:
        parts.append(f"t3={s['overall_top_3_pct']:>5.1f}%")
    if "overall_multi_partial3_pct" in s:
        parts.append(f"multi-p3={s['overall_multi_partial3_pct']:>5.1f}%")
    if "overall_oos_rejection_pct" in s:
        parts.append(f"oos={s['overall_oos_rejection_pct']:>5.1f}%")
    parts.append(f"lat={s.get('avg_latency_ms', 0)}ms")
    return " | ".join(parts)


def main():
    build_namespace()
    time.sleep(0.5)

    queries = QUERIES["queries"]
    print(f"\nTotal queries: {len(queries)}")

    # Run with and without tiebreaker
    for cfg_name, tb in (("baseline", False), ("tiebreaker", True)):
        summary = run_set(queries, cfg_name, tb)
        print(f"\n{pretty(f'scale125_{cfg_name}', summary)}")
        print("  By category (top-3 / multi-p3 / oos):")
        for cat, info in sorted(summary.get("by_category", {}).items()):
            n = info["n"]
            t1 = f"{info.get('top_1_pct', 0):.0f}%" if "top_1_pct" in info else "—"
            t3 = f"{info.get('top_3_pct', 0):.0f}%" if "top_3_pct" in info else "—"
            oos = f"{info.get('oos_rejection_pct', 0):.0f}%" if "oos_rejection_pct" in info else "—"
            multi = f"{info.get('multi_partial3_pct', 0):.0f}%" if "multi_partial3_pct" in info else "—"
            print(f"    {cat:18s} n={n:3d}  t1={t1:>5s}  t3={t3:>5s}  multi_p3={multi:>5s}  oos={oos:>5s}")


if __name__ == "__main__":
    main()
