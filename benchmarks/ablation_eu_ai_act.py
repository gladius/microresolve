"""4-way ablation on the eu-ai-act-prohibited pack.

Compares macro F1 / R / P / FP rate at the pack's default threshold (1.5)
across four configurations:
  A = baseline       (no lexical_groups, no policy_overrides)
  B = +lexical only  (lexical_groups on, policy_overrides off)
  C = +policy only   (lexical_groups off, policy_overrides on)
  D = +both          (lexical_groups on, policy_overrides on)

For each config we mutate a temp copy of the pack's _ns.json, load it as
a microresolve namespace, run the 100 prohibited + 80 benign corpus
through it, and compute per-intent + macro metrics.

Pass criteria (locked before measurement):
  - D ≥ A + 1pp F1
  - D ≥ B + 0.5pp F1 (proves policy adds value over morph)
  - No pack regresses >1pp F1 from B → D
  - Benign FP rate increases ≤2pp from B → D anywhere
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
GAP = 1.5


def stage_pack(config: str, root: Path) -> Path:
    """Stage the pack to <root>/<config>/<pack>, mutating _ns.json per config."""
    cfg_root = root / config
    if cfg_root.exists():
        shutil.rmtree(cfg_root)
    cfg_root.mkdir(parents=True)
    dest = cfg_root / PACK_NAME
    shutil.copytree(PACK_SRC, dest)

    ns_path = dest / "_ns.json"
    ns = json.load(open(ns_path))
    if config in ("baseline", "policy_only"):
        ns.pop("lexical_groups", None)
    if config in ("baseline", "lex_only"):
        ns.pop("policy_overrides", None)
    json.dump(ns, open(ns_path, "w"), indent=2)
    return cfg_root


def run_config(config: str, root: Path, corpus: dict) -> dict:
    cfg_root = stage_pack(config, root)
    engine = microresolve.MicroResolve(data_dir=str(cfg_root))
    ns = engine.namespace(PACK_NAME)

    # Resolve every query, score it against ground truth
    per_intent = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
    intent_ids = ns.intent_ids()
    benign_hits = 0
    benign_total = 0

    for entry in corpus["prohibited"]:
        gt = entry["expected_intent"]
        query = entry["text"]
        result = ns.resolve(query)
        hit_high = any(
            i.band == "High" and i.score >= TARGET_THRESHOLD for i in result.intents
        )
        top = next(
            (i for i in result.intents if i.score >= TARGET_THRESHOLD),
            None,
        )
        for iid in intent_ids:
            is_gt = iid == gt
            is_hit = top is not None and top.id == iid
            if is_gt and is_hit:
                per_intent[iid]["tp"] += 1
            elif is_gt and not is_hit:
                per_intent[iid]["fn"] += 1
            elif not is_gt and is_hit:
                per_intent[iid]["fp"] += 1
            else:
                per_intent[iid]["tn"] += 1

    for entry in corpus["benign"]:
        query = entry["text"]
        result = ns.resolve(query)
        benign_total += 1
        top = next(
            (i for i in result.intents if i.score >= TARGET_THRESHOLD),
            None,
        )
        if top is not None and top.id != "legitimate_use":
            benign_hits += 1
            for iid in intent_ids:
                if iid == top.id and iid != "legitimate_use":
                    per_intent[iid]["fp"] += 1

    # Compute macros
    metrics = {}
    p_sum = r_sum = f_sum = 0.0
    n = 0
    for iid in intent_ids:
        if iid == "legitimate_use":
            continue
        m = per_intent[iid]
        tp, fn, fp = m["tp"], m["fn"], m["fp"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[iid] = {"p": p, "r": r, "f1": f, "tp": tp, "fn": fn, "fp": fp}
        p_sum += p
        r_sum += r
        f_sum += f
        n += 1

    benign_fp_rate = benign_hits / benign_total if benign_total > 0 else 0.0

    return {
        "config": config,
        "macro_p": p_sum / n,
        "macro_r": r_sum / n,
        "macro_f1": f_sum / n,
        "benign_fp_rate": benign_fp_rate,
        "benign_hits": benign_hits,
        "benign_total": benign_total,
        "per_intent": metrics,
    }


def main():
    corpus = json.load(open(CORPUS))
    root = Path("/tmp/ablation")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    configs = ["baseline", "lex_only", "policy_only", "both"]
    results = {}
    for c in configs:
        print(f"--- {c} ---", flush=True)
        results[c] = run_config(c, root, corpus)
        r = results[c]
        print(
            f"  macro: P={r['macro_p']:.3f}  R={r['macro_r']:.3f}  F1={r['macro_f1']:.3f}"
            f"  benign-FP={r['benign_fp_rate']:.3f} ({r['benign_hits']}/{r['benign_total']})"
        )

    out = Path("benchmarks/results/ablation_eu_ai_act.json")
    out.parent.mkdir(exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)

    print()
    print("=" * 72)
    print("Summary  config        F1     ΔF1    R      ΔR     P      benign-FP")
    print("=" * 72)
    base = results["baseline"]
    for c in configs:
        r = results[c]
        d_f1 = (r["macro_f1"] - base["macro_f1"]) * 100
        d_r = (r["macro_r"] - base["macro_r"]) * 100
        print(
            f"         {c:13s} {r['macro_f1']:.3f}  {d_f1:+5.1f}pp {r['macro_r']:.3f}  {d_r:+5.1f}pp {r['macro_p']:.3f}  {r['benign_fp_rate']:.3f}"
        )

    print()
    print("Pass criteria check:")
    b = results["baseline"]
    lex = results["lex_only"]
    both = results["both"]
    crit1 = (both["macro_f1"] - b["macro_f1"]) >= 0.01
    crit2 = (both["macro_f1"] - lex["macro_f1"]) >= 0.005
    crit3 = (lex["macro_f1"] - both["macro_f1"]) <= 0.01
    crit4 = (both["benign_fp_rate"] - lex["benign_fp_rate"]) <= 0.02
    print(f"  D > A + 1pp F1?              {crit1}  ({(both['macro_f1']-b['macro_f1'])*100:+.2f}pp)")
    print(f"  D > B + 0.5pp F1?            {crit2}  ({(both['macro_f1']-lex['macro_f1'])*100:+.2f}pp)")
    print(f"  D - B regression ≤ 1pp F1?   {crit3}")
    print(f"  D - B benign-FP ≤ 2pp?       {crit4}  ({(both['benign_fp_rate']-lex['benign_fp_rate'])*100:+.2f}pp)")
    all_pass = crit1 and crit2 and crit3 and crit4
    print(f"\n  OVERALL: {'PASS — ship combined' if all_pass else 'KILL — strip policy_overrides'}")

    print()
    print(f"Full results written to {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
