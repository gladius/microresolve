"""Empirical proof: can ~4 additional well-engineered legitimate_use
seed phrases replicate the FP reduction that policy_overrides achieves?

Configs:
  baseline      = no lex, no policy, original seeds
  lex_only      = +lexical, no policy, original seeds
  +policy       = +lexical, +policy_overrides (current main)
  +better_seeds = +lexical, no policy, additional seed phrases on
                  legitimate_use covering the same carve-outs

If +better_seeds matches +policy on benign FP rate at comparable F1,
the user is empirically correct: policy_overrides is a hand-authored
shortcut for seed engineering.
"""
import json, shutil, sys
from pathlib import Path
from collections import defaultdict
import microresolve

PACK_NAME = "eu-ai-act-prohibited"
PACK_SRC = Path("packs") / PACK_NAME
CORPUS = Path("_internal/EU_AI_ACT_EVAL_CORPUS.json")
TARGET_THRESHOLD = 1.5

# Extra seed phrases covering the same carve-outs the policy_overrides target.
EXTRA_LEGITIMATE_SEEDS = [
    # Predictive policing carve-out (Feb 2025 Commission): profiling + objective facts
    "risk model using outstanding warrants and witness reports",
    "criminal investigation profile combining suspect description with verifiable facts",
    "investigation analysis using outstanding warrants for active case",
    # CSAM detection (Omnibus 5/2026): moderation ≠ generation
    "CSAM detection model for NCMEC reporting",
    "automated child safety scanner for content moderation",
    "detection tool for flagging illegal child abuse material",
    # Missing child / AMBER alert (Art 5(1)(h) explicit exception)
    "targeted facial search for missing child",
    "missing children identification system at airports",
]


def stage(config: str, root: Path) -> Path:
    cfg_root = root / config
    if cfg_root.exists():
        shutil.rmtree(cfg_root)
    cfg_root.mkdir(parents=True)
    dest = cfg_root / PACK_NAME
    shutil.copytree(PACK_SRC, dest)
    ns_path = dest / "_ns.json"
    ns = json.load(open(ns_path))

    if config in ("baseline", "lex_only", "better_seeds"):
        ns.pop("policy_overrides", None)
    if config == "baseline":
        ns.pop("lexical_groups", None)

    json.dump(ns, open(ns_path, "w"), indent=2)

    if config == "better_seeds":
        lp = dest / "legitimate_use.json"
        intent = json.load(open(lp))
        existing = intent["phrases"].get("en", [])
        intent["phrases"]["en"] = existing + EXTRA_LEGITIMATE_SEEDS
        json.dump(intent, open(lp, "w"), indent=2)

    return cfg_root


def measure(config: str, root: Path, corpus: dict) -> dict:
    cfg_root = stage(config, root)
    engine = microresolve.MicroResolve(data_dir=str(cfg_root))
    ns = engine.namespace(PACK_NAME)

    per_intent = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
    intent_ids = ns.intent_ids()
    benign_hits = 0
    benign_total = 0

    for entry in corpus["prohibited"]:
        gt = entry["expected_intent"]
        q = entry["text"]
        r = ns.resolve(q)
        top = next((i for i in r.intents if i.score >= TARGET_THRESHOLD), None)
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
        q = entry["text"]
        r = ns.resolve(q)
        benign_total += 1
        top = next((i for i in r.intents if i.score >= TARGET_THRESHOLD), None)
        if top is not None and top.id != "legitimate_use":
            benign_hits += 1
            for iid in intent_ids:
                if iid == top.id and iid != "legitimate_use":
                    per_intent[iid]["fp"] += 1

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
        p_sum += p; r_sum += r; f_sum += f; n += 1

    return {
        "P": p_sum / n,
        "R": r_sum / n,
        "F1": f_sum / n,
        "benign_fp": benign_hits / benign_total,
        "benign_hits": benign_hits,
    }


def main():
    corpus = json.load(open(CORPUS))
    root = Path("/tmp/seeds_vs_policy")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    configs = ["baseline", "lex_only", "with_policy", "better_seeds"]
    out = {}
    for c in configs:
        if c == "with_policy":
            cfg_root = root / c
            if cfg_root.exists(): shutil.rmtree(cfg_root)
            cfg_root.mkdir(parents=True)
            dest = cfg_root / PACK_NAME
            shutil.copytree(PACK_SRC, dest)
            ns_path = dest / "_ns.json"
            ns = json.load(open(ns_path))
            json.dump(ns, open(ns_path, "w"), indent=2)
            # measure inline (don't re-stage)
            engine = microresolve.MicroResolve(data_dir=str(cfg_root))
            nsh = engine.namespace(PACK_NAME)
            from copy import deepcopy
            r_data = defaultdict(lambda: {"tp":0,"fn":0,"fp":0,"tn":0})
            ids = nsh.intent_ids()
            bh = bt = 0
            for entry in corpus["prohibited"]:
                gt = entry["expected_intent"]
                rr = nsh.resolve(entry["text"])
                top = next((i for i in rr.intents if i.score >= TARGET_THRESHOLD), None)
                for iid in ids:
                    is_gt = iid==gt; is_hit = top is not None and top.id==iid
                    if is_gt and is_hit: r_data[iid]["tp"]+=1
                    elif is_gt: r_data[iid]["fn"]+=1
                    elif is_hit: r_data[iid]["fp"]+=1
                    else: r_data[iid]["tn"]+=1
            for entry in corpus["benign"]:
                rr = nsh.resolve(entry["text"])
                bt += 1
                top = next((i for i in rr.intents if i.score >= TARGET_THRESHOLD), None)
                if top is not None and top.id != "legitimate_use":
                    bh += 1
                    for iid in ids:
                        if iid==top.id and iid!="legitimate_use": r_data[iid]["fp"]+=1
            p_sum=r_sum=f_sum=0.0; n=0
            for iid in ids:
                if iid=="legitimate_use": continue
                m = r_data[iid]; tp,fn,fp = m["tp"],m["fn"],m["fp"]
                pp = tp/(tp+fp) if tp+fp>0 else 0.0
                rr2 = tp/(tp+fn) if tp+fn>0 else 0.0
                ff = 2*pp*rr2/(pp+rr2) if pp+rr2>0 else 0.0
                p_sum+=pp; r_sum+=rr2; f_sum+=ff; n+=1
            out[c] = {"P":p_sum/n,"R":r_sum/n,"F1":f_sum/n,"benign_fp":bh/bt,"benign_hits":bh}
        else:
            out[c] = measure(c, root, corpus)
        print(f"  {c:14s}  F1={out[c]['F1']:.3f}  R={out[c]['R']:.3f}  P={out[c]['P']:.3f}  benign-FP={out[c]['benign_fp']:.3f} ({out[c]['benign_hits']}/80)")

    print()
    print("=" * 72)
    base = out["baseline"]
    print("Summary  config           F1     ΔF1       benign-FP   ΔFP")
    print("=" * 72)
    for c in configs:
        r = out[c]
        d_f1 = (r["F1"] - base["F1"]) * 100
        d_fp = (r["benign_fp"] - base["benign_fp"]) * 100
        print(f"         {c:15s} {r['F1']:.3f}  {d_f1:+5.1f}pp  {r['benign_fp']:.3f}     {d_fp:+5.1f}pp")

    print()
    print("Verdict:")
    p_vs_seeds_f1 = (out['better_seeds']['F1'] - out['with_policy']['F1']) * 100
    p_vs_seeds_fp = (out['better_seeds']['benign_fp'] - out['with_policy']['benign_fp']) * 100
    print(f"  better_seeds vs with_policy: F1 diff = {p_vs_seeds_f1:+.2f}pp, FP diff = {p_vs_seeds_fp:+.2f}pp")
    if abs(p_vs_seeds_f1) < 1.0 and p_vs_seeds_fp <= 0.5:
        print("  → SEED ENGINEERING ALONE MATCHES OR BEATS POLICY OVERRIDES")
        print("    Policy_overrides is empirically a thumb-on-the-scale for seed work.")
    else:
        print("  → policy_overrides has a genuine effect that seeds don't replicate")


if __name__ == "__main__":
    main()
