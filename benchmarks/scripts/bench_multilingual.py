#!/usr/bin/env python3
"""Multilingual intent routing benchmark.

5 intents seeded in 5 languages (EN, JA, KO, Tamil, ZH).
Tests that MicroResolve routes correctly regardless of query language.

Usage:
  python3 benchmarks/scripts/bench_multilingual.py
"""

import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lib import check_server, create_namespace, delete_namespace, _req

DATASETS = os.path.join(os.path.dirname(__file__), "../datasets")
NS = f"bench-multilingual-{int(time.time())}"

LANGS = {"en": "English", "ja": "Japanese", "ko": "Korean", "ta": "Tamil", "zh": "Chinese"}

def load_seeds(ns, seeds):
    for intent_id, langs in seeds.items():
        all_phrases = {}
        for lang, phrases in langs.items():
            all_phrases[lang] = phrases
        try:
            _req("POST", "/api/intents/multilingual", {
                "id": intent_id,
                "phrases_by_lang": all_phrases,
            }, ns=ns)
        except Exception as e:
            print(f"  [warn] seed {intent_id}: {e}")

def evaluate(ns, test):
    results = []
    for ex in test:
        query, expected, lang = ex["text"], set(ex["intents"]), ex.get("lang", "en")
        t0 = time.perf_counter()
        resp = _req("POST", "/api/route_multi", {"query": query, "log": False}, ns=ns)
        lat = (time.perf_counter() - t0) * 1_000_000
        confirmed = set(r["id"] for r in resp.get("confirmed", []))
        ranked    = [r["id"] for r in resp.get("ranked", [])]
        k = len(expected)
        results.append({
            "text": query, "lang": lang,
            "expected": expected, "confirmed": confirmed,
            "exact": confirmed == expected,
            "recall": len(expected & confirmed) / len(expected) if expected else 0,
            "topk": expected.issubset(set(ranked[:k])),
            "latency_us": lat,
        })
    return results

def print_by_lang(results):
    for lang_code, lang_name in LANGS.items():
        lr = [r for r in results if r["lang"] == lang_code]
        if not lr: continue
        n = len(lr)
        exact  = sum(1 for r in lr if r["exact"]) / n * 100
        recall = sum(r["recall"] for r in lr) / n * 100
        topk   = sum(1 for r in lr if r["topk"]) / n * 100
        lats   = sorted(r["latency_us"] for r in lr)
        p50    = lats[len(lats)//2] if lats else 0
        print(f"  {lang_name:<10} {exact:>6.0f}%  {recall:>6.0f}%  {topk:>6.0f}%  {p50:>6.0f}µs  (n={n})")

def main():
    if not check_server(): sys.exit(1)

    seeds = json.load(open(os.path.join(DATASETS, "multilingual_seeds.json")))
    test  = json.load(open(os.path.join(DATASETS, "multilingual_test.json")))

    print("=" * 65)
    print(f"  Multilingual Benchmark — EN · JA · KO · Tamil · ZH")
    print(f"  {len(seeds)} intents | {len(test)} test queries across 5 languages")
    print("=" * 65)

    create_namespace(NS)
    load_seeds(NS, seeds)
    print(f"  Seeded {len(seeds)} intents in 5 languages\n")

    results = evaluate(NS, test)
    n = len(results)
    exact  = sum(1 for r in results if r["exact"]) / n * 100
    recall = sum(r["recall"] for r in results) / n * 100
    topk   = sum(1 for r in results if r["topk"]) / n * 100

    print(f"  {'Language':<10} {'Exact':>6}  {'Recall':>6}  {'Top-K':>6}  {'p50µs':>6}")
    print(f"  {'-'*55}")
    print_by_lang(results)
    print(f"  {'-'*55}")
    print(f"  {'OVERALL':<10} {exact:>6.0f}%  {recall:>6.0f}%  {topk:>6.0f}%")

    # Failures
    failures = [r for r in results if not r["exact"]]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for r in failures:
            print(f"  [{r['lang']}] {r['text'][:55]}")
            print(f"         expected={sorted(r['expected'])} got={sorted(r['confirmed'])}")

    delete_namespace(NS)
    print(f"\n  Cleaned up {NS}")
    return exact, recall

if __name__ == "__main__":
    main()
