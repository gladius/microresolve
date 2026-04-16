#!/usr/bin/env python3
"""
ASV Reliability Measurement Harness
====================================

Runs the enterprise query set against a given namespace via /api/route_multi,
scoring top-1, top-3, false-positive, false-negative, and OOS rejection.

No LLM calls. Pure ASV routing. Fast enough to run in under a minute for 105 queries.

Usage:
    python3 tests/reliability/measure.py                     # default: namespace=scale-test, label=baseline
    python3 tests/reliability/measure.py --namespace X --label Y
    python3 tests/reliability/measure.py --save  # writes results/{label}.json
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path

BASE_URL = "http://localhost:3001"
ROOT = Path(__file__).parent

# ── Equivalence-class expansion (Step 2) ──────────────────────────────────
EQUIVALENCE_MAP = None


def load_equivalence(path: Path):
    global EQUIVALENCE_MAP
    if path.exists():
        EQUIVALENCE_MAP = json.loads(path.read_text())
        print(f"Loaded {len(EQUIVALENCE_MAP)} equivalence-class variants from {path}")


def expand_query(q: str) -> str:
    """Expand each token by appending its canonical form(s) from the equivalence map."""
    if not EQUIVALENCE_MAP:
        return q
    tokens = re.findall(r"[a-zA-Z]+|[^\sa-zA-Z]+", q)
    out = []
    for t in tokens:
        out.append(t)
        tl = t.lower()
        if tl in EQUIVALENCE_MAP:
            for canonical in EQUIVALENCE_MAP[tl]:
                if canonical != tl:
                    out.append(canonical)
    return " ".join(out)


# Step 5: bigram re-ranking toggle
BIGRAM_RERANK = False


def route(namespace: str, query: str) -> dict:
    query = expand_query(query)
    body = json.dumps({"query": query, "threshold": 0.3, "gap": 1.5, "log": False}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/route_multi",
        data=body,
        headers={"Content-Type": "application/json", "X-Namespace-ID": namespace},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    data["_latency_ms"] = (time.time() - t0) * 1000
    data["_query"] = query
    return data


def top_ids(routing: dict, k: int = 3) -> list:
    ranked = routing.get("ranked") or []
    # Apply bigram re-ranking if enabled
    if BIGRAM_RERANK and ranked and "_query" in routing:
        import bigram_rerank
        rescored = bigram_rerank.rerank(routing["_query"], ranked, top_k=10, alpha=0.5)
        return [r["id"] for r in rescored[:k]]
    if ranked:
        return [r["id"] for r in ranked[:k]]
    confirmed = routing.get("confirmed") or []
    return [c["id"] for c in confirmed[:k]]


def top_scores(routing: dict) -> tuple:
    ranked = routing.get("ranked") or []
    if len(ranked) >= 2:
        return ranked[0]["score"], ranked[1]["score"]
    if len(ranked) == 1:
        return ranked[0]["score"], 0.0
    return 0.0, 0.0


def evaluate(dataset: dict, namespace: str) -> dict:
    queries = dataset["queries"]
    results = []

    print(f"\nRunning {len(queries)} queries against '{namespace}'…")
    t_start = time.time()

    for i, q in enumerate(queries, 1):
        query_text = q["query"]
        category = q["category"]
        expected = q.get("expected")  # None for OOS or not-set
        expected_multi = q.get("expected_multi")  # list for multi-intent

        try:
            routing = route(namespace, query_text)
        except urllib.error.HTTPError as e:
            print(f"  [{i}/{len(queries)}] HTTP {e.code} on query: {query_text[:50]!r}")
            continue

        top3 = top_ids(routing, 3)
        top5 = top_ids(routing, 5)
        top1_score, top2_score = top_scores(routing)
        disposition = routing.get("disposition", "?")
        latency = routing.get("_latency_ms", 0)

        record = {
            "category": category,
            "query": query_text,
            "expected": expected,
            "expected_multi": expected_multi,
            "top_1": top3[0] if top3 else None,
            "top_3": top3,
            "top_5": top5,
            "top_1_score": top1_score,
            "top_2_score": top2_score,
            "confidence_ratio": top1_score / (top1_score + top2_score) if (top1_score + top2_score) > 0 else 0.0,
            "disposition": disposition,
            "latency_ms": latency,
        }

        # Score correctness
        if expected_multi is not None:
            record["expected_multi_hit3"] = all(e in top5 for e in expected_multi)
            record["expected_multi_partial3"] = any(e in top5 for e in expected_multi)
        elif expected is None:
            # OOS: correct if disposition is no_match OR top_1_score is very low
            record["oos_rejected"] = disposition == "no_match" or top1_score < 0.5
        else:
            record["hit_1"] = record["top_1"] == expected
            record["hit_3"] = expected in top3
            record["hit_5"] = expected in top5

        results.append(record)

    t_total = time.time() - t_start
    print(f"Done in {t_total:.1f}s")

    return {"namespace": namespace, "results": results, "total_time_s": t_total}


def summarize(run: dict) -> dict:
    results = run["results"]
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    summary = {"namespace": run["namespace"], "total_queries": len(results), "by_category": {}}

    # Per-category metrics
    overall = {"hit_1": 0, "hit_3": 0, "hit_5": 0, "single_n": 0,
               "multi_hit3": 0, "multi_partial3": 0, "multi_n": 0,
               "oos_rejected": 0, "oos_n": 0,
               "total_latency_ms": 0.0, "n": 0}

    for cat, items in sorted(by_cat.items()):
        cat_s = {"n": len(items)}
        if items[0].get("expected_multi") is not None:
            cat_s["multi_hit3"] = sum(1 for r in items if r.get("expected_multi_hit3"))
            cat_s["multi_partial3"] = sum(1 for r in items if r.get("expected_multi_partial3"))
            cat_s["multi_hit3_pct"] = round(100 * cat_s["multi_hit3"] / len(items), 1)
            cat_s["multi_partial3_pct"] = round(100 * cat_s["multi_partial3"] / len(items), 1)
            overall["multi_hit3"] += cat_s["multi_hit3"]
            overall["multi_partial3"] += cat_s["multi_partial3"]
            overall["multi_n"] += len(items)
        elif items[0].get("expected") is None:
            cat_s["oos_rejected"] = sum(1 for r in items if r.get("oos_rejected"))
            cat_s["oos_rejection_pct"] = round(100 * cat_s["oos_rejected"] / len(items), 1)
            overall["oos_rejected"] += cat_s["oos_rejected"]
            overall["oos_n"] += len(items)
        else:
            cat_s["hit_1"] = sum(1 for r in items if r.get("hit_1"))
            cat_s["hit_3"] = sum(1 for r in items if r.get("hit_3"))
            cat_s["hit_5"] = sum(1 for r in items if r.get("hit_5"))
            cat_s["top_1_pct"] = round(100 * cat_s["hit_1"] / len(items), 1)
            cat_s["top_3_pct"] = round(100 * cat_s["hit_3"] / len(items), 1)
            cat_s["top_5_pct"] = round(100 * cat_s["hit_5"] / len(items), 1)
            overall["hit_1"] += cat_s["hit_1"]
            overall["hit_3"] += cat_s["hit_3"]
            overall["hit_5"] += cat_s["hit_5"]
            overall["single_n"] += len(items)

        avg_lat = sum(r["latency_ms"] for r in items) / len(items)
        cat_s["avg_latency_ms"] = round(avg_lat, 1)
        overall["total_latency_ms"] += sum(r["latency_ms"] for r in items)
        overall["n"] += len(items)

        summary["by_category"][cat] = cat_s

    # Overall
    if overall["single_n"]:
        summary["overall_top_1_pct"] = round(100 * overall["hit_1"] / overall["single_n"], 1)
        summary["overall_top_3_pct"] = round(100 * overall["hit_3"] / overall["single_n"], 1)
        summary["overall_top_5_pct"] = round(100 * overall["hit_5"] / overall["single_n"], 1)
    if overall["multi_n"]:
        summary["overall_multi_hit3_pct"] = round(100 * overall["multi_hit3"] / overall["multi_n"], 1)
        summary["overall_multi_partial3_pct"] = round(100 * overall["multi_partial3"] / overall["multi_n"], 1)
    if overall["oos_n"]:
        summary["overall_oos_rejection_pct"] = round(100 * overall["oos_rejected"] / overall["oos_n"], 1)
    summary["avg_latency_ms"] = round(overall["total_latency_ms"] / max(overall["n"], 1), 2)

    return summary


def pretty_print(summary: dict):
    print("\n" + "═" * 78)
    print(f"Summary — namespace: {summary['namespace']}  (n={summary['total_queries']})")
    print("═" * 78)

    # Overall line
    overall_parts = []
    if "overall_top_1_pct" in summary:
        overall_parts.append(f"top-1 {summary['overall_top_1_pct']}%")
    if "overall_top_3_pct" in summary:
        overall_parts.append(f"top-3 {summary['overall_top_3_pct']}%")
    if "overall_multi_hit3_pct" in summary:
        overall_parts.append(f"multi-hit3 {summary['overall_multi_hit3_pct']}%")
    if "overall_multi_partial3_pct" in summary:
        overall_parts.append(f"multi-partial3 {summary['overall_multi_partial3_pct']}%")
    if "overall_oos_rejection_pct" in summary:
        overall_parts.append(f"oos-rejection {summary['overall_oos_rejection_pct']}%")
    print(f"Overall: {' | '.join(overall_parts)}  (avg latency: {summary['avg_latency_ms']}ms)")
    print("─" * 78)

    # Per category
    for cat, s in sorted(summary["by_category"].items()):
        parts = [f"n={s['n']}"]
        if "top_1_pct" in s: parts.append(f"t1={s['top_1_pct']}%")
        if "top_3_pct" in s: parts.append(f"t3={s['top_3_pct']}%")
        if "top_5_pct" in s: parts.append(f"t5={s['top_5_pct']}%")
        if "multi_hit3_pct" in s: parts.append(f"multi-hit3={s['multi_hit3_pct']}%")
        if "multi_partial3_pct" in s: parts.append(f"multi-partial3={s['multi_partial3_pct']}%")
        if "oos_rejection_pct" in s: parts.append(f"reject={s['oos_rejection_pct']}%")
        parts.append(f"lat={s['avg_latency_ms']}ms")
        print(f"  {cat:22s}  {'  '.join(parts)}")
    print("═" * 78)


def show_failures(run: dict, limit: int = 20):
    """Print queries that failed for inspection."""
    print("\nFailures for inspection:")
    print("─" * 78)
    n = 0
    for r in run["results"]:
        failed = False
        if r.get("expected_multi") is not None:
            if not r.get("expected_multi_partial3"):
                failed = True
                print(f"  MULTI-MISS [{r['category']}] {r['query']}")
                print(f"      expected: {r['expected_multi']}")
                print(f"      got top5: {r['top_5']}")
        elif r.get("expected") is None:
            if not r.get("oos_rejected"):
                failed = True
                print(f"  OOS-FALSE-ACCEPT [{r['category']}] {r['query']}")
                print(f"      got top3: {r['top_3']}  (top1 score={r['top_1_score']:.2f})")
        else:
            if not r.get("hit_3"):
                failed = True
                print(f"  MISS [{r['category']}] {r['query']}")
                print(f"      expected: {r['expected']}")
                print(f"      got top3: {r['top_3']}")
        if failed:
            n += 1
            if n >= limit:
                print(f"  … (truncated; use --show-all to see all)")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="scale-test")
    parser.add_argument("--label", default="baseline", help="experiment label for save path")
    parser.add_argument("--save", action="store_true", help="save full results to results/{label}.json")
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--only", default=None, help="only run queries in this category")
    parser.add_argument("--expand", action="store_true", help="apply equivalence-class expansion at query time")
    parser.add_argument("--set", default="dev", choices=["dev", "validation"], help="which query set to run")
    args = parser.parse_args()

    if args.expand:
        load_equivalence(ROOT / "equivalence_classes.json")

    dataset_path = ROOT / ("validation.json" if args.set == "validation" else "dataset.json")
    dataset = json.loads(dataset_path.read_text())

    if args.only:
        dataset["queries"] = [q for q in dataset["queries"] if q["category"] == args.only]
        print(f"Filtered to {len(dataset['queries'])} queries in category '{args.only}'")

    run = evaluate(dataset, args.namespace)
    summary = summarize(run)
    pretty_print(summary)

    show_failures(run, limit=999 if args.show_all else 25)

    if args.save:
        out_dir = ROOT / "results"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{args.label}.json"
        out_path.write_text(json.dumps({"summary": summary, "run": run}, indent=2))
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
