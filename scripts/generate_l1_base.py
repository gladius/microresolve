#!/usr/bin/env python3
"""Generate L1 base graph (l1_base.json) from WordNet and ConceptNet.

This is a ONE-TIME setup script. Run it before starting the ASV server
to give L1 a rich synonym/normalization foundation across all namespaces.

Output: data/l1_base.json  (loaded by ASV server at startup)

Usage:
  pip install nltk requests
  python3 scripts/generate_l1_base.py

Optional — also extract domain edges from your own query history:
  python3 scripts/generate_l1_base.py --domain path/to/queries.json

Format of queries.json:
  [{"text": "cancel my subscription", "intent_id": "cancel_order"}, ...]
"""

import argparse
import json
import os
import sys
import math
from collections import defaultdict

# ── Output path ───────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "l1_base.json")

# ── Edge weight bands (match ASV EdgeKind thresholds) ────────────────────────
# Morphological: 0.97–1.0  → substitute in place
# Synonym:       0.80–0.96 → append to query
# Semantic:      0.60–0.79 → confidence boost only

MORPH_WEIGHT   = 0.99
SYNONYM_WEIGHT = 0.88
SEMANTIC_WEIGHT = 0.68

# EdgeKind values must match Rust serde(rename_all = "snake_case")
KIND_MORPH    = "morphological"
KIND_SYNONYM  = "synonym"
KIND_SEMANTIC = "semantic"
KIND_ABBREV   = "abbreviation"

# English stop words — skip these as L1 nodes (too generic, create noise)
STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "up", "about", "into", "through",
    "i", "my", "me", "we", "our", "you", "your", "it", "its",
    "this", "that", "these", "those", "and", "or", "but", "not",
    "get", "go", "make", "take", "use", "want", "need", "like",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(term: str) -> str:
    return term.lower().replace("_", " ").strip()

def is_useful(term: str) -> bool:
    t = clean(term)
    return (
        len(t) >= 3
        and t not in STOP
        and not any(c.isdigit() for c in t)
    )

def add_edge(graph, src, tgt, weight, kind):
    src, tgt = clean(src), clean(tgt)
    if src == tgt or not is_useful(src) or not is_useful(tgt):
        return
    if src not in graph:
        graph[src] = []
    # Avoid duplicates — keep highest weight
    for e in graph[src]:
        if e["target"] == tgt:
            if weight > e["weight"]:
                e["weight"] = weight
                e["kind"] = kind
            return
    graph[src].append({"target": tgt, "weight": round(weight, 4), "kind": kind})

# ── WordNet ───────────────────────────────────────────────────────────────────

def load_wordnet(graph):
    print("Loading WordNet…")
    try:
        import nltk
        try:
            from nltk.corpus import wordnet
            wordnet.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet
    except ImportError:
        print("  SKIP: nltk not installed (pip install nltk)")
        return 0

    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    morph_added = 0
    syn_added = 0

    # Morphological normalization — verb + noun inflections
    # For every word in WordNet vocabulary, map inflected forms → lemma
    SUFFIXES = {
        "v": ["ing", "ed", "s", "er", "ers"],
        "n": ["s", "es", "ing"],
    }
    for pos, suffixes in SUFFIXES.items():
        for synset in wordnet.all_synsets(pos=pos):
            for lemma in synset.lemmas():
                base = lemma.name().replace("_", " ")
                if not is_useful(base):
                    continue
                # Add morph variants via lemmatizer
                for suffix in suffixes:
                    variant = base + suffix
                    lemmatized = lemmatizer.lemmatize(variant, pos=pos)
                    if lemmatized == base and variant != base:
                        add_edge(graph, variant, base, MORPH_WEIGHT, KIND_MORPH)
                        morph_added += 1

    print(f"  Morphological edges: {morph_added}")

    # Synonym edges — within each synset, all lemmas are synonyms
    seen_pairs = set()
    for synset in wordnet.all_synsets():
        lemmas = [l.name().replace("_", " ") for l in synset.lemmas()]
        lemmas = [l for l in lemmas if is_useful(l)]
        if len(lemmas) < 2:
            continue
        # Use first lemma as canonical; others are synonyms
        canonical = lemmas[0]
        for alt in lemmas[1:]:
            pair = tuple(sorted([canonical, alt]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            add_edge(graph, alt, canonical, SYNONYM_WEIGHT, KIND_SYNONYM)
            add_edge(graph, canonical, alt, SYNONYM_WEIGHT, KIND_SYNONYM)
            syn_added += 1

    print(f"  Synonym edges: {syn_added}")
    return morph_added + syn_added


# ── ConceptNet (API — free, no download required) ─────────────────────────────

CONCEPTNET_API = "http://api.conceptnet.io"

# Key intent-routing vocabulary to seed from ConceptNet
# We query per-term (API is free, rate-limited at ~3 req/s)
SEED_TERMS = [
    # Actions
    "cancel", "refund", "pay", "charge", "subscribe", "unsubscribe",
    "upgrade", "downgrade", "reset", "update", "change", "delete",
    "create", "add", "remove", "block", "unlock", "activate", "deactivate",
    "send", "receive", "transfer", "withdraw", "deposit", "dispute",
    "track", "ship", "deliver", "return", "exchange", "replace",
    "login", "logout", "signup", "verify", "confirm", "approve", "reject",
    "schedule", "book", "reserve", "buy", "purchase", "order", "invoice",
    "report", "flag", "escalate", "contact", "help", "support",
    # Nouns / objects
    "account", "subscription", "plan", "billing", "payment", "invoice",
    "card", "credit", "debit", "balance", "transaction", "fee",
    "password", "email", "profile", "settings", "notification",
    "order", "shipment", "delivery", "package", "product",
    "ticket", "issue", "request", "complaint", "feedback",
    # Common abbreviations / informal
    "pr", "repo", "msg", "sub", "acc", "txn", "pw", "2fa",
]

# ConceptNet relation types → ASV edge kind + weight
RELATION_MAP = {
    "/r/Synonym":       (KIND_SYNONYM, 0.92),
    "/r/SimilarTo":     (KIND_SYNONYM, 0.85),
    "/r/RelatedTo":     (KIND_SEMANTIC, 0.68),
    "/r/FormOf":        (KIND_MORPH, 0.99),
    "/r/DerivedFrom":   (KIND_MORPH, 0.97),
    "/r/EtymologicallyRelatedTo": (KIND_SEMANTIC, 0.65),
}


def load_conceptnet(graph):
    print("Loading ConceptNet (API)…")
    try:
        import requests
    except ImportError:
        print("  SKIP: requests not installed (pip install requests)")
        return 0

    import time

    added = 0
    for i, term in enumerate(SEED_TERMS):
        try:
            url = f"{CONCEPTNET_API}/c/en/{term}?limit=100"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            for edge in data.get("edges", []):
                rel = edge.get("rel", {}).get("@id", "")
                if rel not in RELATION_MAP:
                    continue
                kind, weight = RELATION_MAP[rel]

                start = edge.get("start", {}).get("label", "")
                end   = edge.get("end",   {}).get("label", "")
                start_lang = edge.get("start", {}).get("language", "en")
                end_lang   = edge.get("end",   {}).get("language", "en")

                # Only English edges for now
                if start_lang != "en" or end_lang != "en":
                    continue

                # Scale by ConceptNet edge weight (0–10 → 0.60–0.96)
                cn_weight = edge.get("weight", 1.0)
                scaled = min(0.96, weight * (cn_weight / 5.0 + 0.5))

                if start.lower() != end.lower():
                    add_edge(graph, start, end, scaled, kind)
                    added += 1

            # Polite rate limiting
            if i % 10 == 9:
                time.sleep(1)
                print(f"  Progress: {i+1}/{len(SEED_TERMS)} terms…")

        except Exception as e:
            print(f"  Warning: ConceptNet failed for '{term}': {e}")
            continue

    print(f"  ConceptNet edges added: {added}")
    return added


# ── Domain co-occurrence ──────────────────────────────────────────────────────

def load_domain(graph, queries_path: str):
    """Extract synonym edges from labeled historical queries via PMI."""
    print(f"Loading domain data from {queries_path}…")
    with open(queries_path) as f:
        queries = json.load(f)

    tokenize = lambda t: [w for w in t.lower().split() if is_useful(w)]

    # term → set of intent_ids it appears in
    term_intents = defaultdict(set)
    total_intents = set()

    for item in queries:
        intent = item.get("intent_id") or item.get("intent") or item.get("label")
        text   = item.get("text") or item.get("query") or ""
        if not intent or not text:
            continue
        total_intents.add(intent)
        for token in tokenize(text):
            term_intents[token].add(intent)

    n_intents = max(len(total_intents), 1)
    terms = list(term_intents.keys())
    added = 0

    for i, t1 in enumerate(terms):
        for t2 in terms[i+1:]:
            shared = len(term_intents[t1] & term_intents[t2])
            if shared < 2:
                continue

            # PMI
            p_t1   = len(term_intents[t1]) / n_intents
            p_t2   = len(term_intents[t2]) / n_intents
            p_both = shared / n_intents
            if p_t1 * p_t2 == 0:
                continue
            pmi = math.log2(p_both / (p_t1 * p_t2))

            # Jaccard
            jaccard = shared / len(term_intents[t1] | term_intents[t2])

            if pmi < 0.5 or jaccard < 0.3:
                continue

            weight = min(0.94, 0.70 + jaccard * 0.24)
            kind = KIND_SYNONYM if jaccard >= 0.5 else KIND_SEMANTIC
            add_edge(graph, t1, t2, weight, kind)
            add_edge(graph, t2, t1, weight, kind)
            added += 1

    print(f"  Domain edges added: {added}")
    return added


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate ASV L1 base graph")
    parser.add_argument("--domain", metavar="FILE",
                        help="Path to labeled queries JSON for domain co-occurrence")
    parser.add_argument("--skip-conceptnet", action="store_true",
                        help="Skip ConceptNet API calls (faster, offline)")
    parser.add_argument("--out", default=OUT, help=f"Output path (default: {OUT})")
    args = parser.parse_args()

    graph = {}  # term → [edge, ...]

    wn_count = load_wordnet(graph)

    if not args.skip_conceptnet:
        cn_count = load_conceptnet(graph)
    else:
        cn_count = 0
        print("ConceptNet: skipped")

    domain_count = 0
    if args.domain:
        domain_count = load_domain(graph, args.domain)

    total_terms = len(graph)
    total_edges = sum(len(v) for v in graph.values())

    output = {
        "edges": graph,
        "synonym_threshold": 0.80,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = os.path.getsize(args.out) / 1_000_000
    print(f"\n=== L1 base graph generated ===")
    print(f"  Terms:        {total_terms:,}")
    print(f"  Total edges:  {total_edges:,}")
    print(f"    WordNet:    {wn_count:,}")
    print(f"    ConceptNet: {cn_count:,}")
    print(f"    Domain:     {domain_count:,}")
    print(f"  File size:    {size_mb:.1f} MB")
    print(f"  Output:       {args.out}")
    print(f"\nNext step: restart ASV server — it will load this file automatically.")


if __name__ == "__main__":
    main()
