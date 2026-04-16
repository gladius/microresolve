#!/usr/bin/env python3
"""
Post-processing layers applied on top of ASV's ranked output.

Layer 1: N-gram FP filter — demote top-1 if query has zero distinctive bigram match
Layer 2: Char n-gram Jaccard tiebreaker — break close top-1/top-2 using char overlap

Both are stateless functions that compute indices from seed phrases once,
then apply at query time. No training required.
"""

import json
import re
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent

STOPWORDS = set("a an the is are was were be been being have has had do does did "
                "i you he she it we they me him her us them my your his its our their "
                "this that these those to of in on at for with by from as but "
                "please let can will would should could may might shall and or".split())


# ─── Tokenization helpers ────────────────────────────────────────────────

def tokenize(text: str) -> list:
    return [t.lower() for t in re.findall(r"[a-zA-Z]+", text) if t.lower() not in STOPWORDS and len(t) > 1]


def bigrams(tokens: list) -> set:
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()


def char_ngrams(text: str, n: int = 4) -> set:
    """Character n-grams from the full query text, padded."""
    s = "  " + text.lower().strip() + "  "
    s = re.sub(r"\s+", " ", s)
    return set(s[i:i+n] for i in range(len(s) - n + 1))


# ─── FP filter ───────────────────────────────────────────────────────────

class NgramFPFilter:
    """
    Build distinctive bigrams per intent from seed phrases.
    A bigram is 'distinctive' for intent I if it has high IDF across all intents
    in the namespace — i.e., appears in few other intents.
    """

    def __init__(self, snapshot: list, idf_min: float = 1.5):
        """
        snapshot: list of intent dicts with phrases_by_lang.en
        idf_min: minimum IDF score for a bigram to count as distinctive.
                 idf = log((N+1) / (df+1)) + 1
                 N=15 intents, df=1 -> idf = log(16/2)+1 = 3.08  (highly distinctive)
                 N=15 intents, df=5 -> idf = log(16/6)+1 = 1.98
                 N=98 intents, df=1 -> idf = log(99/2)+1 = 4.90
                 N=98 intents, df=10 -> idf = log(99/11)+1 = 3.20
        """
        self.idf_min = idf_min
        intent_bigrams = {}
        bigram_df = defaultdict(int)

        for intent in snapshot:
            iid = intent["id"]
            phrases = intent.get("phrases_by_lang", {}).get("en", [])
            bg_set = set()
            for p in phrases:
                bg_set.update(bigrams(tokenize(p)))
            intent_bigrams[iid] = bg_set
            for bg in bg_set:
                bigram_df[bg] += 1

        N = len(intent_bigrams)
        self.distinctive = {}  # intent_id -> set of distinctive bigrams
        for iid, bgs in intent_bigrams.items():
            self.distinctive[iid] = set()
            for bg in bgs:
                idf = math.log((N + 1) / (bigram_df[bg] + 1)) + 1
                if idf >= idf_min:
                    self.distinctive[iid].add(bg)

        self.intent_bigrams = intent_bigrams  # all bigrams per intent (for fallback check)

    def has_distinctive_match(self, query: str, intent_id: str) -> bool:
        """Does the query share any distinctive bigram with this intent?"""
        q_bigrams = bigrams(tokenize(query))
        if not q_bigrams:
            return True  # single-word queries get a pass (can't judge)
        distinctive = self.distinctive.get(intent_id, set())
        if not distinctive:
            # No distinctive bigrams for this intent — fallback to any bigram overlap
            intent_bgs = self.intent_bigrams.get(intent_id, set())
            return bool(q_bigrams & intent_bgs) or len(q_bigrams) == 0
        return bool(q_bigrams & distinctive)

    def apply(self, query: str, ranked: list, strong_threshold: float = 3.0) -> list:
        """
        Filter the ranked list in place:
        - If top-1 score >= strong_threshold, keep (too confident to second-guess)
        - Else if top-1 has no distinctive bigram match → demote it below top-2
        """
        if not ranked or len(ranked) < 2:
            return ranked
        top_1 = ranked[0]
        if top_1["score"] >= strong_threshold:
            return ranked  # trust confident match
        if not self.has_distinctive_match(query, top_1["id"]):
            # Demote: move top-1 below top-2 (to position 2 or later)
            demoted = ranked[0]
            rest = ranked[1:]
            return rest + [demoted]
        return ranked


# ─── Char-ngram tiebreaker ───────────────────────────────────────────────

class CharNgramTiebreaker:
    """
    When top-1/top-2 are close, use character 4-gram Jaccard between
    query and each candidate's seed phrase set to break the tie.
    """

    def __init__(self, snapshot: list, n: int = 4):
        self.n = n
        self.intent_chars = {}  # intent_id -> set of char n-grams (union across phrases)
        for intent in snapshot:
            iid = intent["id"]
            phrases = intent.get("phrases_by_lang", {}).get("en", [])
            cs = set()
            for p in phrases:
                cs.update(char_ngrams(p, n))
            self.intent_chars[iid] = cs

    def jaccard(self, query: str, intent_id: str) -> float:
        q = char_ngrams(query, self.n)
        i = self.intent_chars.get(intent_id, set())
        if not q or not i:
            return 0.0
        return len(q & i) / len(q | i)

    def apply(self, query: str, ranked: list, ratio_threshold: float = 0.65, alpha: float = 0.5) -> list:
        """
        When top1/(top1+top2) < ratio_threshold, rescore top-K using
        score + alpha * jaccard and re-sort.
        """
        if not ranked or len(ranked) < 2:
            return ranked
        s1 = ranked[0]["score"]
        s2 = ranked[1]["score"]
        if s1 + s2 <= 0:
            return ranked
        ratio = s1 / (s1 + s2)
        if ratio >= ratio_threshold:
            return ranked
        # Re-rank top-5 (cap)
        top_k = ranked[:5]
        rescored = []
        for c in top_k:
            j = self.jaccard(query, c["id"])
            new_score = c["score"] + alpha * j
            rescored.append({**c, "score": new_score, "_jaccard": j})
        rescored.sort(key=lambda x: -x["score"])
        return rescored + ranked[5:]


if __name__ == "__main__":
    # Quick sanity check
    import sys
    snap_path = ROOT / "scale_test_snapshot.json"
    snap = json.loads(snap_path.read_text())
    f = NgramFPFilter(snap, idf_min=2.0)
    t = CharNgramTiebreaker(snap, n=4)

    # Sample
    print(f"Intent count: {len(snap)}")
    sample_intent = "stripe:create_refund"
    print(f"\n{sample_intent} distinctive bigrams ({len(f.distinctive.get(sample_intent, set()))}):")
    for bg in list(f.distinctive.get(sample_intent, set()))[:10]:
        print(f"  {bg}")
    print(f"\n{sample_intent} char-ngram count: {len(t.intent_chars.get(sample_intent, set()))}")
