#!/usr/bin/env python3
"""
Dense-seed experiment: 3 domains × 5 intents × 12 phrases = 15 intents, 180 phrases.
Tests whether MicroResolve's real ceiling is data density (seeds/intent) rather than algorithmic layers.

Runs baseline MicroResolve on this realistically-seeded namespace and reports dev+validation numbers.
"""
import json
import urllib.request
import time
from pathlib import Path

import namespace_ops as ns_ops
import measure

ROOT = Path(__file__).parent
SETUP = json.loads((ROOT / "dense_seed_setup.json").read_text())
QUERIES = json.loads((ROOT / "dense_queries.json").read_text())
NS = SETUP["_meta"]["namespace"]  # "bench_dense"


def build_dense_namespace():
    """Create bench_dense namespace with 15 intents × 12 phrases each."""
    # Delete + recreate
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
    print(f"Built '{NS}': {added}/{len(SETUP['intents'])} intents with {sum(len(v) for v in SETUP['intents'].values())} phrases")


def run_set(queries: list, label: str):
    dataset = {"queries": queries}
    run = measure.evaluate(dataset, NS)
    summary = measure.summarize(run)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"dense_{label}.json"
    out_path.write_text(json.dumps({"summary": summary, "run": run}, indent=2))
    return summary


def pretty(label: str, s: dict) -> str:
    parts = [f"[{label:18s}]"]
    if "overall_top_1_pct" in s:
        parts.append(f"t1={s['overall_top_1_pct']:>5.1f}%")
    if "overall_top_3_pct" in s:
        parts.append(f"t3={s['overall_top_3_pct']:>5.1f}%")
    if "overall_oos_rejection_pct" in s:
        parts.append(f"oos={s['overall_oos_rejection_pct']:>5.1f}%")
    parts.append(f"lat={s.get('avg_latency_ms', 0)}ms")
    return " | ".join(parts)


def main():
    print("Building dense-seed namespace…")
    build_dense_namespace()
    time.sleep(0.5)

    print("\n=== DEV SET ===")
    dev_s = run_set(QUERIES["dev"], "dev")
    print(pretty("dense_dev", dev_s))

    print("\n=== VALIDATION SET ===")
    val_s = run_set(QUERIES["validation"], "validation")
    print(pretty("dense_val", val_s))

    # Per-category breakdown on validation
    print("\n── Validation by category ──")
    for cat, info in sorted(val_s.get("by_category", {}).items()):
        parts = [f"n={info['n']}"]
        for k in ("top_1_pct", "top_3_pct", "top_5_pct", "oos_rejection_pct"):
            if k in info: parts.append(f"{k[:-4]}={info[k]:.0f}%")
        print(f"  {cat:18s}  {'  '.join(parts)}")


if __name__ == "__main__":
    main()
