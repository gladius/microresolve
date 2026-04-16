#!/usr/bin/env python3
"""
Bigram-IDF experimental layer.

Builds per-intent bigram sets from seed phrases, then at query time
re-ranks ASV's top-K results using bigram overlap bonus.

This simulates how a secondary bigram-scoring signal would behave
without modifying ASV's core scoring code. If this shows a lift,
productizing it is straightforward — add bigram scoring alongside
unigram in ASV's index.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
SNAPSHOT = json.loads((ROOT / "scale_test_snapshot.json").read_text())

STOPWORDS = set("a an the is are was were be been being have has had do does did "
                "i you he she it we they me him her us them my your his its our their "
                "this that these those to of in on at for with by from as but "
                "please let can will would should could may might shall and or".split())


def tokenize(text: str) -> list:
    toks = [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


def extract_bigrams(text: str) -> set:
    toks = tokenize(text)
    return set(zip(toks, toks[1:]))


def build_intent_bigrams(snapshot: list) -> dict:
    """
    intent_id -> {bigram: weight}
    Weight is inverse document frequency across intents.
    """
    intent_bigrams = {}
    for intent in snapshot:
        phrases = intent.get("phrases_by_lang", {}).get("en", [])
        bigrams = set()
        for p in phrases:
            bigrams.update(extract_bigrams(p))
        intent_bigrams[intent["id"]] = bigrams

    # IDF: bigram -> how many intents contain it
    bigram_df = defaultdict(int)
    for bigrams in intent_bigrams.values():
        for bg in bigrams:
            bigram_df[bg] += 1

    # Convert to weighted
    N = len(intent_bigrams)
    import math
    weighted = {}
    for intent_id, bigrams in intent_bigrams.items():
        weighted[intent_id] = {
            bg: math.log((N + 1) / (bigram_df[bg] + 1)) + 1.0
            for bg in bigrams
        }
    return weighted


# Precompute on module load
INTENT_BIGRAMS = build_intent_bigrams(SNAPSHOT)


def bigram_score(query: str, intent_id: str) -> float:
    """How well does this query match this intent's seed bigrams?"""
    q_bigrams = extract_bigrams(query)
    if not q_bigrams:
        return 0.0
    intent_bgs = INTENT_BIGRAMS.get(intent_id, {})
    if not intent_bgs:
        return 0.0
    total = 0.0
    for bg in q_bigrams:
        if bg in intent_bgs:
            total += intent_bgs[bg]
    return total


def rerank(query: str, asv_ranked: list, top_k: int = 10, alpha: float = 0.5) -> list:
    """
    Given ASV's ranked list [{id, score}, ...], add bigram bonus and re-sort.
    alpha: weight of bigram score in the combined score (0 = ASV only).
    Only considers top_k candidates from ASV to limit cost.
    """
    candidates = asv_ranked[:top_k]
    rescored = []
    for c in candidates:
        bg = bigram_score(query, c["id"])
        combined = c["score"] + alpha * bg
        rescored.append({"id": c["id"], "score": combined, "asv_score": c["score"], "bigram_bonus": bg})
    rescored.sort(key=lambda x: -x["score"])
    return rescored


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        print(f"Bigrams for intent {sys.argv[1]}:")
        print(INTENT_BIGRAMS.get(sys.argv[1], {}))
