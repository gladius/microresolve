#!/usr/bin/env python3
"""Experiment: Vector similarity vs current L2 token scoring on CLINC150.

Tests three modes:
  baseline   — current ASV server (L2 token IDF, L1 morph+synonym)
  vector     — pure GloVe mean-pool: no server, no token index
  hybrid     — server L2 score + vector cosine, combined

Hypothesis: vector mode handles paraphrase queries the server misses.
            hybrid mode catches both vocabulary-exact and paraphrase.

Usage:
  python3 exp_vector_l1.py
"""

import json, os, sys, time, math, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import check_server, create_namespace, delete_namespace, load_seeds, _req

BENCH_DIR  = os.path.dirname(os.path.abspath(__file__))
TRACK1_DIR = os.path.join(BENCH_DIR, "track1")

GLOVE_PATH = "/home/gladius/Workspace/reason-research/brain_v25/data/glove.6B.100d.txt"
ALPHA      = 0.4   # hybrid: weight for vector score (1-ALPHA for token score)
TOPN       = 1     # only top-1 accuracy reported

# ── GloVe loader ─────────────────────────────────────────────────────────────

def load_glove(path):
    print(f"  Loading GloVe from {path} ...", end="", flush=True)
    t0 = time.time()
    vecs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vecs[word] = np.array(parts[1:], dtype=np.float32)
    print(f" {len(vecs):,} words in {time.time()-t0:.1f}s")
    return vecs

def phrase_vector(glove, text):
    """Mean pool of token vectors. Returns None if no tokens found."""
    tokens = text.lower().split()
    vecs = [glove[t] for t in tokens if t in glove]
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else None

def cosine(a, b):
    return float(np.dot(a, b))   # both already unit-normed


# ── Build intent vectors from seeds ──────────────────────────────────────────

def build_intent_vectors(glove, seeds):
    intent_vecs = {}
    missing = 0
    for intent_id, phrases in seeds.items():
        phrase_vecs = [v for ph in phrases if (v := phrase_vector(glove, ph)) is not None]
        if phrase_vecs:
            avg = np.mean(phrase_vecs, axis=0)
            norm = np.linalg.norm(avg)
            intent_vecs[intent_id] = avg / norm if norm > 1e-9 else avg
        else:
            missing += 1
    if missing:
        print(f"  Warning: {missing} intents had no GloVe coverage")
    return intent_vecs


# ── Scoring modes ─────────────────────────────────────────────────────────────

def score_vector(intent_vecs, query_text):
    """Pure cosine: query mean-pool vs each intent vector."""
    qv = phrase_vector(glove_global, query_text)
    if qv is None:
        return None
    scores = [(iid, cosine(qv, ivec)) for iid, ivec in intent_vecs.items()]
    scores.sort(key=lambda x: -x[1])
    return scores[0][0] if scores else None

def score_server(ns, query_text):
    resp = _req("POST", "/api/route_multi", {"query": query_text, "log": False}, ns=ns)
    confirmed = resp.get("confirmed", [])
    return confirmed[0]["id"] if confirmed else None

def score_hybrid(ns, intent_vecs, query_text):
    """Combine server token score + vector cosine."""
    resp = _req("POST", "/api/route_multi", {"query": query_text, "log": False}, ns=ns)

    # Server ranked list (raw IDF scores, normalised 0-1)
    ranked = resp.get("ranked", [])
    if not ranked:
        # fall back to vector only
        return score_vector(intent_vecs, query_text)

    max_server = ranked[0]["score"] if ranked else 1.0
    server_map = {r["id"]: r["score"] / max(max_server, 1e-6) for r in ranked}

    # Vector scores
    qv = phrase_vector(glove_global, query_text)
    if qv is None:
        confirmed = resp.get("confirmed", [])
        return confirmed[0]["id"] if confirmed else None

    combined = {}
    all_ids = set(server_map) | set(intent_vecs)
    for iid in all_ids:
        ts = server_map.get(iid, 0.0)
        vs = cosine(qv, intent_vecs[iid]) if iid in intent_vecs else 0.0
        # vector score can be negative (cosine range -1..1); shift to 0..1
        vs_norm = (vs + 1.0) / 2.0
        combined[iid] = (1 - ALPHA) * ts + ALPHA * vs_norm

    best = max(combined, key=combined.__getitem__)
    # Only return if combined score above a weak threshold
    if combined[best] > 0.15:
        return best
    return None


# ── Run one mode over all test examples ──────────────────────────────────────

def evaluate(name, predict_fn, test):
    correct = 0
    no_match = 0
    errors = []
    t0 = time.time()
    for ex in test:
        expected = ex["intents"][0] if ex["intents"] else None
        predicted = predict_fn(ex["text"])
        if predicted == expected:
            correct += 1
        else:
            if predicted is None:
                no_match += 1
            errors.append({"text": ex["text"], "expected": expected, "got": predicted})
    elapsed = time.time() - t0
    acc = correct / len(test) * 100
    lat = elapsed / len(test) * 1e6   # microseconds per query
    return acc, no_match, lat, errors


# ── Main ──────────────────────────────────────────────────────────────────────

glove_global = None   # set after load so score_vector can access it

def main():
    global glove_global

    with open(f"{TRACK1_DIR}/clinc150_seeds.json") as f: seeds = json.load(f)
    with open(f"{TRACK1_DIR}/clinc150_test.json")  as f: test  = json.load(f)

    print(f"\n{'='*62}")
    print(f"  CLINC150 Vector-L1 Potency Experiment")
    print(f"  {len(seeds)} intents | {sum(len(v) for v in seeds.values())} seeds | {len(test)} test queries")
    print(f"{'='*62}\n")

    # 1. Load GloVe
    glove_global = load_glove(GLOVE_PATH)

    # 2. Build intent vectors
    print("  Building intent vectors ...", end="", flush=True)
    t0 = time.time()
    intent_vecs = build_intent_vectors(glove_global, seeds)
    print(f" done in {time.time()-t0:.2f}s  ({len(intent_vecs)} intents covered)")

    # 3. Check GloVe coverage on test queries
    covered = sum(1 for ex in test if phrase_vector(glove_global, ex["text"]) is not None)
    print(f"  GloVe query coverage: {covered}/{len(test)} ({covered/len(test)*100:.1f}%)\n")

    # 4. Server namespace (for baseline + hybrid)
    if not check_server():
        print("  Server not running — skipping baseline and hybrid modes.")
        server_ok = False
        ns = None
    else:
        server_ok = True
        ns = f"exp-vec-{int(time.time())}"
        print(f"  Creating namespace {ns} ...")
        create_namespace(ns)
        load_seeds(ns, seeds)
        print()

    # 5. Evaluate each mode
    print(f"  {'Mode':<12} {'Top-1':>7} {'No-match':>9} {'µs/q':>8}")
    print(f"  {'-'*12} {'-'*7} {'-'*9} {'-'*8}")

    results = {}

    # Pure vector
    acc, nm, lat, errs = evaluate("vector", lambda q: score_vector(intent_vecs, q), test)
    results["vector"] = (acc, nm, errs)
    print(f"  {'vector':<12} {acc:>6.1f}% {nm:>9} {lat:>7.0f}µs")

    if server_ok:
        # Baseline (server L2 token)
        acc, nm, lat, errs = evaluate("baseline", lambda q: score_server(ns, q), test)
        results["baseline"] = (acc, nm, errs)
        print(f"  {'baseline':<12} {acc:>6.1f}% {nm:>9} {lat:>7.0f}µs")

        # Hybrid
        acc, nm, lat, errs = evaluate("hybrid", lambda q: score_hybrid(ns, intent_vecs, q), test)
        results["hybrid"] = (acc, nm, errs)
        print(f"  {'hybrid':<12} {acc:>6.1f}% {nm:>9} {lat:>7.0f}µs")

        delete_namespace(ns)

    # 6. Error overlap analysis
    if "baseline" in results and "vector" in results:
        base_errors  = {e["text"] for e in results["baseline"][2]}
        vec_errors   = {e["text"] for e in results["vector"][2]}
        base_only    = base_errors - vec_errors
        vec_only     = vec_errors  - base_errors
        both_wrong   = base_errors & vec_errors

        print(f"\n  ── Error overlap ──────────────────────────────────")
        print(f"  Baseline wrong, vector right: {len(base_only):4d}  ← vector adds value here")
        print(f"  Vector wrong, baseline right: {len(vec_only):4d}  ← vector hurts here")
        print(f"  Both wrong:                   {len(both_wrong):4d}  ← neither can fix these")

        # Sample cases where vector helps
        if base_only:
            print(f"\n  ── Vector rescues (sample 8) ──────────────────────")
            base_err_map = {e["text"]: e for e in results["baseline"][2]}
            vec_err_map  = {e["text"]: e for e in results["vector"][2]}
            for text in list(base_only)[:8]:
                be = base_err_map[text]
                print(f"    QUERY:    {text}")
                print(f"    EXPECTED: {be['expected']}  | baseline got: {be['got']}  | vector got: (correct)")
                print()

        # Sample cases where vector hurts
        if vec_only:
            print(f"  ── Vector hurts (sample 5) ────────────────────────")
            base_err_map = {e["text"]: e for e in results["baseline"][2]}
            vec_err_map  = {e["text"]: e for e in results["vector"][2]}
            for text in list(vec_only)[:5]:
                ve = vec_err_map[text]
                print(f"    QUERY:    {text}")
                print(f"    EXPECTED: {ve['expected']}  | vector got: {ve['got']}")
                print()

    print("\nDone.")

if __name__ == "__main__":
    main()
