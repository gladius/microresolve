"""Per-rule attribution: which of the 8 policy_overrides actually fire,
on which queries, and do they change the outcome vs. lex-only baseline?

For every corpus query, run BOTH lex_only and both configs, compare
the top result. When they differ, report which policy_override rule
made the difference.
"""
import json
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import microresolve

PACK_NAME = "eu-ai-act-prohibited"
PACK_SRC = Path("packs") / PACK_NAME
CORPUS = Path("_internal/EU_AI_ACT_EVAL_CORPUS.json")
TARGET_THRESHOLD = 1.5


def stage(config: str, root: Path) -> Path:
    cfg_root = root / config
    if cfg_root.exists():
        shutil.rmtree(cfg_root)
    cfg_root.mkdir(parents=True)
    dest = cfg_root / PACK_NAME
    shutil.copytree(PACK_SRC, dest)
    ns_path = dest / "_ns.json"
    ns = json.load(open(ns_path))
    if config == "lex_only":
        ns.pop("policy_overrides", None)
    json.dump(ns, open(ns_path, "w"), indent=2)
    return cfg_root


def top_intent_at_threshold(ns, query):
    r = ns.resolve(query)
    return next((i.id for i in r.intents if i.score >= TARGET_THRESHOLD), None)


def main():
    corpus = json.load(open(CORPUS))
    root = Path("/tmp/policy_attribution")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    e_lex = microresolve.MicroResolve(data_dir=str(stage("lex_only", root)))
    e_both = microresolve.MicroResolve(data_dir=str(stage("both", root)))
    ns_lex = e_lex.namespace(PACK_NAME)
    ns_both = e_both.namespace(PACK_NAME)

    # Load the 8 policy rules so we can match
    rules = json.load(open(PACK_SRC / "_ns.json"))["policy_overrides"]
    print(f"Loaded {len(rules)} policy override rules:\n")
    for i, r in enumerate(rules):
        print(f"  [{i}] {r['words']} → {r['intent']} (bonus={r['bonus']})")
    print()

    # Examine every query
    diff_prohibited = []
    diff_benign = []
    rule_fires = defaultdict(list)  # rule_idx -> [(query, lex_top, both_top)]

    def match_rule(query_lower):
        """Find which rule's words ALL appear in lowercased query."""
        hits = []
        for i, r in enumerate(rules):
            if all(w in query_lower for w in r["words"]):
                hits.append(i)
        return hits

    for entry in corpus["prohibited"]:
        q = entry["text"]
        gt = entry["expected_intent"]
        a = top_intent_at_threshold(ns_lex, q)
        b = top_intent_at_threshold(ns_both, q)
        if a != b:
            diff_prohibited.append((q, gt, a, b))
            for ri in match_rule(q.lower()):
                rule_fires[ri].append((q, a, b, "prohibited", gt))

    for entry in corpus["benign"]:
        q = entry["text"]
        a = top_intent_at_threshold(ns_lex, q)
        b = top_intent_at_threshold(ns_both, q)
        if a != b:
            diff_benign.append((q, a, b))
            for ri in match_rule(q.lower()):
                rule_fires[ri].append((q, a, b, "benign", None))

    print("=" * 72)
    print(f"Queries where lex-only vs both DISAGREE:")
    print(f"  prohibited diffs: {len(diff_prohibited)}")
    print(f"  benign diffs:     {len(diff_benign)}")
    print()
    print("Per-rule firing count (rules that flipped an outcome):")
    print()
    for i, r in enumerate(rules):
        n = len(rule_fires[i])
        label = f"[{i}] {' + '.join(r['words'])} → {r['intent']}"
        flag = "" if n > 0 else "  ← DEAD: never fired"
        print(f"  {n:2d}  {label:60s}{flag}")
    print()

    print("=" * 72)
    print("Diff examples (queries where the addition of policy_overrides changed the result):")
    print()
    print("--- benign (policy helps reject false-positives) ---")
    for q, a, b in diff_benign[:10]:
        marker = "✓" if b == "legitimate_use" or b is None else "?"
        print(f"  {marker}  '{q[:80]}'")
        print(f"     lex_only: {a}  →  both: {b}")
    print()
    print("--- prohibited (policy changes which prohibited intent is picked, or routes to legitimate_use) ---")
    for q, gt, a, b in diff_prohibited[:10]:
        wrong = " ⚠ moved away from ground truth" if a == gt and b != gt else ""
        helps = " ✓ moved toward ground truth" if a != gt and b == gt else ""
        print(f"  '{q[:80]}'")
        print(f"     gt={gt}  lex_only: {a}  →  both: {b}{wrong}{helps}")

    print()
    print("=" * 72)
    print("Summary:")
    fired_n = sum(1 for i in range(len(rules)) if len(rule_fires[i]) > 0)
    print(f"  Rules that ever fired on a query that flipped outcome: {fired_n} / {len(rules)}")

    # Net benign FP reduction
    benign_flips_to_legit = sum(1 for q, a, b in diff_benign if b == "legitimate_use" or b is None)
    benign_flips_other = len(diff_benign) - benign_flips_to_legit
    print(f"  Benign queries that flipped:")
    print(f"    to 'legitimate_use' or NoMatch (helpful): {benign_flips_to_legit}")
    print(f"    to a different prohibited intent (concerning): {benign_flips_other}")


if __name__ == "__main__":
    main()
