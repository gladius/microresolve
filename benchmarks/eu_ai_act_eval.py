"""Confusion-matrix evaluation of the eu-ai-act-prohibited pack.

Per-intent TP/FN/FP/TN, macro precision/recall/F1, threshold sweep,
benign aggregate FP rate. Adjacent-legal benigns (looks-like-prohibited
but carved out by Feb 2025 Commission guidelines) are tracked separately.

Run: python benchmarks/eu_ai_act_eval.py
"""
import json
import shutil
from collections import defaultdict
from pathlib import Path

import microresolve

PACK_NAME = "eu-ai-act-prohibited"
PACK_SRC = Path("packs") / PACK_NAME
CORPUS = Path("_internal/EU_AI_ACT_EVAL_CORPUS.json")
THRESHOLDS = [0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5]


def stage_pack():
    p = Path("/tmp/eu_ai_act_eval_data")
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    shutil.copytree(PACK_SRC, p / PACK_NAME)
    return p


def intents_from_pack():
    return sorted(p.stem for p in PACK_SRC.glob("*.json") if p.name != "_ns.json")


def top1(result):
    """Return (intent_id, score, band) of the top High-band hit, else None."""
    if not result.intents:
        return None
    top = result.intents[0]
    return (top.id, top.score, top.band)


def eval_at_threshold(threshold, intents, prohibited, benigns):
    data = stage_pack()
    e = microresolve.MicroResolve(data_dir=str(data))
    ns = e.namespace(PACK_NAME)
    ns.update_namespace({"default_threshold": threshold})

    # Per-intent confusion matrix: intent → {tp, fn, fp, tn}
    cm = {i: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for i in intents}

    fn_examples = []  # missed prohibited
    fp_examples = []  # benign hits and adjacent-legal hits
    routing_misses = []  # prohibited routed to wrong intent

    # Walk prohibited queries: each one is positive for its expected_intent
    # and negative for every other intent.
    for entry in prohibited:
        text = entry["text"]
        expected = entry["expected_intent"]
        r = ns.resolve(text)
        top = top1(r)
        predicted = top[0] if top and top[2] == "High" else None

        for intent in intents:
            is_positive = (intent == expected)
            is_predicted = (predicted == intent)
            if is_positive and is_predicted:
                cm[intent]["tp"] += 1
            elif is_positive and not is_predicted:
                cm[intent]["fn"] += 1
                if predicted is None:
                    fn_examples.append((text, expected, "no_high_band", top[1] if top else 0.0))
                else:
                    routing_misses.append((text, expected, predicted, top[1]))
                    fn_examples.append((text, expected, f"->{predicted}", top[1]))
            elif not is_positive and is_predicted:
                cm[intent]["fp"] += 1
            else:
                cm[intent]["tn"] += 1

    # Walk benigns: a benign hitting any High-band prohibited intent (excluding
    # legitimate_use, which is the negative class) is a false positive.
    benign_high_hits = 0
    benign_legitimate_hits = 0
    adjacent_high_hits = 0
    adjacent_legitimate_hits = 0
    by_category = defaultdict(lambda: {"high_hits": 0, "legit_hits": 0, "total": 0})

    for entry in benigns:
        text = entry["text"]
        cat = entry.get("category", "generic_benign")
        r = ns.resolve(text)
        top = top1(r)
        is_high = top is not None and top[2] == "High"
        is_legit = is_high and top[0] == "legitimate_use"
        is_prohibited_hit = is_high and top[0] != "legitimate_use"

        by_category[cat]["total"] += 1
        if is_prohibited_hit:
            by_category[cat]["high_hits"] += 1
        if is_legit:
            by_category[cat]["legit_hits"] += 1

        is_adjacent = cat.startswith("adjacent_")
        if is_adjacent:
            if is_prohibited_hit:
                adjacent_high_hits += 1
            if is_legit:
                adjacent_legitimate_hits += 1
        else:
            if is_prohibited_hit:
                benign_high_hits += 1
            if is_legit:
                benign_legitimate_hits += 1

        if is_prohibited_hit:
            fp_examples.append((text, top[0], top[1], cat))

        # For confusion matrix: for each non-legitimate intent, a high-band hit
        # on a benign query is an FP. Already counted via prohibited loop only,
        # so add benign FPs here.
        for intent in intents:
            if is_high and intent == top[0] and intent != "legitimate_use":
                cm[intent]["fp"] += 1
            else:
                cm[intent]["tn"] += 1

    # Compute per-intent P/R/F1
    def f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f

    per_intent = {}
    for intent in intents:
        c = cm[intent]
        prec, rec, f = f1(c["tp"], c["fp"], c["fn"])
        per_intent[intent] = {
            **c,
            "precision": prec,
            "recall": rec,
            "f1": f,
        }

    # Macro averages over PROHIBITED intents only (exclude legitimate_use,
    # which is the negative class).
    macro_intents = [i for i in intents if i != "legitimate_use"]
    macro_precision = sum(per_intent[i]["precision"] for i in macro_intents) / len(macro_intents)
    macro_recall = sum(per_intent[i]["recall"] for i in macro_intents) / len(macro_intents)
    macro_f1 = sum(per_intent[i]["f1"] for i in macro_intents) / len(macro_intents)

    n_generic = sum(1 for b in benigns if not b.get("category", "").startswith("adjacent_"))
    n_adjacent = sum(1 for b in benigns if b.get("category", "").startswith("adjacent_"))

    return {
        "threshold": threshold,
        "per_intent": per_intent,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "generic_benign_fp_rate": benign_high_hits / n_generic if n_generic else 0.0,
        "adjacent_benign_fp_rate": adjacent_high_hits / n_adjacent if n_adjacent else 0.0,
        "generic_legitimate_routing_rate": benign_legitimate_hits / n_generic if n_generic else 0.0,
        "adjacent_legitimate_routing_rate": adjacent_legitimate_hits / n_adjacent if n_adjacent else 0.0,
        "by_category": dict(by_category),
        "fn_examples": fn_examples,
        "fp_examples": fp_examples,
        "routing_misses": routing_misses,
    }


def main():
    corpus = json.load(open(CORPUS))
    prohibited = corpus["prohibited"]
    benigns = corpus["benign"]
    intents = intents_from_pack()

    print(f"Pack: {PACK_NAME}")
    print(f"Intents: {len(intents)}: {intents}")
    print(f"Prohibited queries: {len(prohibited)}")
    n_generic = sum(1 for b in benigns if not b.get("category", "").startswith("adjacent_"))
    n_adjacent = sum(1 for b in benigns if b.get("category", "").startswith("adjacent_"))
    print(f"Benigns: {len(benigns)} ({n_generic} generic + {n_adjacent} adjacent-legal)")
    print()

    print(f"{'thr':>5}  {'macroP':>7}  {'macroR':>7}  {'macroF1':>8}  {'genFP%':>7}  {'adjFP%':>7}  {'adjLegit%':>9}")
    print("-" * 72)
    results = []
    for t in THRESHOLDS:
        r = eval_at_threshold(t, intents, prohibited, benigns)
        results.append(r)
        print(
            f"{r['threshold']:>5.2f}  "
            f"{r['macro_precision'] * 100:>6.1f}%  "
            f"{r['macro_recall'] * 100:>6.1f}%  "
            f"{r['macro_f1'] * 100:>7.1f}%  "
            f"{r['generic_benign_fp_rate'] * 100:>6.1f}%  "
            f"{r['adjacent_benign_fp_rate'] * 100:>6.1f}%  "
            f"{r['adjacent_legitimate_routing_rate'] * 100:>8.1f}%"
        )

    print()
    # Detail at default threshold
    default_t = 1.5
    r = next((x for x in results if x["threshold"] == default_t), results[0])
    print(f"=== Per-intent detail at threshold = {default_t} ===\n")
    print(f"{'intent':<35} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4}  {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 80)
    for intent in intents:
        c = r["per_intent"][intent]
        print(
            f"{intent:<35} {c['tp']:>4} {c['fn']:>4} {c['fp']:>4} {c['tn']:>4}  "
            f"{c['precision'] * 100:>5.1f}% {c['recall'] * 100:>5.1f}% {c['f1'] * 100:>5.1f}%"
        )
    print()

    if r["routing_misses"]:
        print(f"=== Routing misses (top intent != expected, but still High band) ===")
        for text, expected, predicted, score in r["routing_misses"][:15]:
            print(f"  expected={expected:<30} got={predicted:<30}  score={score:.2f}  {text[:80]}")
        print()

    if r["fn_examples"]:
        print(f"=== Missed prohibited (FN; top {min(15, len(r['fn_examples']))}) ===")
        for text, expected, reason, score in r["fn_examples"][:15]:
            print(f"  exp={expected:<30} {reason:<22} score={score:.2f}  {text[:80]}")
        print()

    if r["fp_examples"]:
        print(f"=== Benign false positives (top {min(15, len(r['fp_examples']))}) ===")
        for text, intent, score, cat in r["fp_examples"][:15]:
            print(f"  cat={cat:<25} hit={intent:<30} score={score:.2f}  {text[:80]}")
        print()

    print("=== Adjacent-benign performance by sub-category at thr=1.5 ===")
    for cat, stats in sorted(r["by_category"].items()):
        if not cat.startswith("adjacent_"):
            continue
        total = stats["total"]
        bad = stats["high_hits"]
        legit = stats["legit_hits"]
        print(f"  {cat:<28} total={total:>3}  prohibited_hits={bad:>2}  legit_route={legit:>2}")
    print()

    out = {
        "pack": PACK_NAME,
        "intents": intents,
        "n_prohibited": len(prohibited),
        "n_benign_generic": n_generic,
        "n_benign_adjacent": n_adjacent,
        "results": [
            {
                "threshold": r["threshold"],
                "per_intent": r["per_intent"],
                "macro_precision": r["macro_precision"],
                "macro_recall": r["macro_recall"],
                "macro_f1": r["macro_f1"],
                "generic_benign_fp_rate": r["generic_benign_fp_rate"],
                "adjacent_benign_fp_rate": r["adjacent_benign_fp_rate"],
                "by_category": r["by_category"],
            }
            for r in results
        ],
    }
    out_path = Path("benchmarks/results/eu_ai_act_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()
