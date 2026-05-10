"""Phase 2 smoke test: does lexical_groups normalize plurals + verb tenses
the way we expect, and does it pollute anything when applied?

Test setup:
  1. Load eu-ai-act-prohibited pack (v0.2.2 baseline — 6 intents).
  2. Add a small set of hand-authored morph groups to _ns.json.
  3. Run a focused diagnostic:
     a. queries that use SINGULAR form — should match seeds (baseline behavior)
     b. queries that use PLURAL form / verb tenses — should now match where
        previously they wouldn't (the feature working)
     c. queries with random vocabulary that shouldn't change behavior
        (regression check — no pollution)
"""
import json
import shutil
from pathlib import Path

import microresolve

PACK_NAME = "eu-ai-act-prohibited"
PACK_SRC = Path("packs") / PACK_NAME

# Hand-authored morph groups we'll inject for the test.
MORPH_GROUPS = [
    {"kind": "morph", "lang": "en", "canonical": "child",
     "variants": ["child", "children", "child's"]},
    {"kind": "morph", "lang": "en", "canonical": "warrant",
     "variants": ["warrant", "warrants"]},
    {"kind": "morph", "lang": "en", "canonical": "predict",
     "variants": ["predict", "predicts", "predicted", "predicting", "prediction"]},
    {"kind": "morph", "lang": "en", "canonical": "person",
     "variants": ["person", "persons", "people"]},
    {"kind": "morph", "lang": "en", "canonical": "score",
     "variants": ["score", "scores", "scoring", "scored"]},
    {"kind": "morph", "lang": "en", "canonical": "manipulate",
     "variants": ["manipulate", "manipulates", "manipulating", "manipulated", "manipulation"]},
]


def stage(suffix: str, with_groups: bool) -> Path:
    data = Path(f"/tmp/lexical_test_{suffix}")
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    shutil.copytree(PACK_SRC, data / PACK_NAME)
    if with_groups:
        ns_path = data / PACK_NAME / "_ns.json"
        d = json.loads(ns_path.read_text())
        d["lexical_groups"] = MORPH_GROUPS
        ns_path.write_text(json.dumps(d, indent=2))
    return data


def resolve(ns, query: str):
    r = ns.resolve(query)
    if not r.intents:
        return ("(none)", 0.0, "Low")
    top = r.intents[0]
    return (top.id, top.score, top.band)


def main():
    print("=" * 80)
    print("PHASE 2 — lexical_groups behavioral smoke test")
    print("=" * 80)

    # Pairs where the second query is a morph variant the first form's seeds
    # would normally match. We expect:
    #   - baseline: first matches well, second matches poorly (or differently)
    #   - with groups: both match similarly (the feature working)
    PAIRS = [
        ("query about a child manipulation",
         "query about children manipulation",
         "expects: child = children should give same result"),
        ("predict criminal behavior",
         "predicting criminal behavior",
         "expects: predict = predicting"),
        ("subliminal manipulation",
         "subliminal manipulating",
         "expects: manipulation = manipulating"),
        ("biometric scoring of people",
         "biometric scores of persons",
         "expects: score = scores AND person = persons"),
    ]

    # Random benign queries — should NOT change between baseline and with-groups.
    BENIGNS = [
        "weather forecasting model for agriculture",
        "recommend movies based on genre preferences",
        "translate documents between languages",
        "voice transcription for meeting notes",
    ]

    print("\n--- Setup ---")
    base_dir = stage("baseline", with_groups=False)
    morph_dir = stage("with_morph", with_groups=True)
    print(f"Baseline pack:  {base_dir}/{PACK_NAME} (no lexical_groups)")
    print(f"With-morph pack: {morph_dir}/{PACK_NAME} ({len(MORPH_GROUPS)} groups)")

    e_base = microresolve.MicroResolve(data_dir=str(base_dir))
    ns_base = e_base.namespace(PACK_NAME)

    e_morph = microresolve.MicroResolve(data_dir=str(morph_dir))
    ns_morph = e_morph.namespace(PACK_NAME)

    print("\n--- Behavioral pairs (singular vs morph variant) ---")
    print(f"{'Query':<48}  {'Baseline':<32}  {'With morph':<32}")
    print("-" * 116)
    for first, second, _exp in PAIRS:
        r1_base, r2_base = resolve(ns_base, first), resolve(ns_base, second)
        r1_morph, r2_morph = resolve(ns_morph, first), resolve(ns_morph, second)
        # Show the second (variant) query's behavior: does it now match closer
        # to what the first (canonical) query produces?
        print(f"  {first[:46]:<48}  {f'{r1_base[0]} {r1_base[1]:.2f} ({r1_base[2]})':<32}  {f'{r1_morph[0]} {r1_morph[1]:.2f} ({r1_morph[2]})':<32}")
        print(f"  {second[:46]:<48}  {f'{r2_base[0]} {r2_base[1]:.2f} ({r2_base[2]})':<32}  {f'{r2_morph[0]} {r2_morph[1]:.2f} ({r2_morph[2]})':<32}")
        # Improvement check: did the variant's score get closer to the canonical's?
        improved = abs(r2_morph[1] - r1_morph[1]) < abs(r2_base[1] - r1_base[1])
        print(f"  variant gap closed?  baseline_gap={abs(r2_base[1]-r1_base[1]):.2f}  morph_gap={abs(r2_morph[1]-r1_morph[1]):.2f}  → {'YES' if improved else 'no'}")
        print()

    print("\n--- Regression check: benign queries should not change ---")
    print(f"{'Query':<58}  {'Baseline':<24}  {'With morph':<24}  {'changed?'}")
    print("-" * 124)
    regressions = 0
    for q in BENIGNS:
        r_base = resolve(ns_base, q)
        r_morph = resolve(ns_morph, q)
        same = r_base[0] == r_morph[0] and abs(r_base[1] - r_morph[1]) < 0.01
        if not same and (r_base[2] == "High" or r_morph[2] == "High"):
            regressions += 1
        print(f"  {q[:56]:<58}  {f'{r_base[0]} {r_base[1]:.2f}':<24}  {f'{r_morph[0]} {r_morph[1]:.2f}':<24}  {'NO' if same else 'YES'}")

    print(f"\nRegressions on benigns (different High-band routing): {regressions}")
    if regressions == 0:
        print("✓ PASS — no benign pollution from morph groups")
    else:
        print("✗ FAIL — morph groups changed benign routing")

    # Variant index check
    print("\n--- Internal: how many variants did the LexicalIndex load? ---")
    print(f"Baseline namespace (no groups): {ns_base.intent_count()} intents")
    print(f"With-morph namespace (groups loaded): {ns_morph.intent_count()} intents")


if __name__ == "__main__":
    main()
