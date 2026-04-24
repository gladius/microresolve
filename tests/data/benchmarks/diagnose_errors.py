#!/usr/bin/env python3
"""Diagnose CLINC150 errors: what kind of failures are they?"""
import json, os, sys, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import check_server, create_namespace, delete_namespace, load_seeds, _req

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK1_DIR = os.path.join(BENCH_DIR, "track1")

def run_diagnostic():
    with open(f"{TRACK1_DIR}/clinc150_seeds.json") as f:
        seeds = json.load(f)
    with open(f"{TRACK1_DIR}/clinc150_test.json") as f:
        test = json.load(f)

    ns = f"diag-clinc150-{int(time.time())}"
    print(f"Creating namespace {ns}...")
    create_namespace(ns)
    load_seeds(ns, seeds)
    print(f"Loaded {sum(len(v) for v in seeds.values())} seeds for {len(seeds)} intents")
    print(f"Running {len(test)} test queries...")

    errors = []
    correct = 0
    no_match = 0

    for ex in test:
        text = ex["text"]
        expected = ex["intents"][0] if ex["intents"] else None
        resp = _req("POST", "/api/route_multi", {"query": text, "log": False, "debug": True}, ns=ns)
        predicted = resp.get("confirmed", [])
        top1 = predicted[0]["id"] if predicted else None

        if top1 == expected:
            correct += 1
        else:
            if top1 is None:
                no_match += 1
            errors.append({
                "text": text,
                "expected": expected,
                "got": top1,
                "debug": resp.get("debug", {}),
                "all_confirmed": [p["id"] for p in predicted],
                "ranked": [r["id"] for r in resp.get("ranked", [])[:3]],
            })

    delete_namespace(ns)

    total = len(test)
    print(f"\nResults: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"Errors: {len(errors)} — no_match: {no_match}, wrong_intent: {len(errors)-no_match}")

    # Categorize errors
    vocab_overlap_misses = []  # expected intent words appear in query but wrong answer
    oov_misses = []            # query words mostly absent from seed vocabulary
    wrong_intent = []          # got a wrong intent (not no_match)

    seed_vocab = {}
    for intent_id, phrases in seeds.items():
        words = set()
        for ph in phrases:
            words.update(ph.lower().split())
        seed_vocab[intent_id] = words

    for err in errors:
        expected = err["expected"]
        text = err["text"]
        query_words = set(text.lower().split())

        if expected in seed_vocab:
            overlap = len(query_words & seed_vocab[expected]) / max(len(query_words), 1)
        else:
            overlap = 0

        err["vocab_overlap"] = overlap

        if err["got"] is None:
            if overlap > 0.3:
                vocab_overlap_misses.append(err)
            else:
                oov_misses.append(err)
        else:
            wrong_intent.append(err)

    print(f"\n--- No-match errors ({no_match}) ---")
    print(f"  High vocab overlap (should have matched): {len(vocab_overlap_misses)}")
    print(f"  Low vocab overlap (OOV / paraphrase):     {len(oov_misses)}")

    print(f"\n--- Wrong intent errors ({len(wrong_intent)}) ---")
    # Find most common wrong-intent pairs
    confusions = collections.Counter()
    for err in wrong_intent:
        confusions[(err["expected"], err["got"])] += 1
    print("  Top confusions (expected → got):")
    for (exp, got), cnt in confusions.most_common(15):
        print(f"    {cnt:3d}x  {exp} → {got}")

    print(f"\n--- Sample OOV misses (vocabulary gap) ---")
    for err in oov_misses[:10]:
        dbg = err["debug"] or {}
        print(f"  EXPECTED: {err['expected']}")
        print(f"  QUERY:    {err['text']}")
        print(f"  L1 norm:  {dbg.get('l1_normalized','?')}")
        print(f"  Overlap:  {err['vocab_overlap']:.2f}")
        print()

    print(f"\n--- Sample vocab-overlap misses (should have matched) ---")
    for err in vocab_overlap_misses[:10]:
        dbg = err["debug"] or {}
        print(f"  EXPECTED: {err['expected']}")
        print(f"  QUERY:    {err['text']}")
        print(f"  L1 norm:  {dbg.get('l1_normalized','?')}")
        print(f"  L1 inj:   {dbg.get('l1_injected','?')}")
        print(f"  Ranked:   {err['ranked']}")
        print(f"  Overlap:  {err['vocab_overlap']:.2f}")
        print()

    print(f"\n--- Sample wrong-intent cases ---")
    for err in wrong_intent[:10]:
        dbg = err["debug"] or {}
        print(f"  EXPECTED: {err['expected']}")
        print(f"  GOT:      {err['got']}")
        print(f"  QUERY:    {err['text']}")
        print(f"  L1 norm:  {dbg.get('l1_normalized','?')}")
        print(f"  L1 inj:   {dbg.get('l1_injected','?')}")
        print()


if __name__ == "__main__":
    if not check_server():
        print("Start the MicroResolve server first.")
        sys.exit(1)
    run_diagnostic()
