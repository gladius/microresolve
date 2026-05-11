"""Smoke-test the 8 ALPHA candidate packs.

For each pack:
  1. Loads it cleanly
  2. Runs each intent's seed phrases back through resolve — does it
     route to its own intent at threshold?
     (This is the WEAKEST sanity test — seed leakage. A pack that
     can't recognize its own seeds is broken.)
  3. Runs 30 random CLINC benigns (off-domain) — should NOT fire High
  4. Reports: self-seed accuracy, OOD FP rate, per-intent coverage
"""
import json, shutil, random
from pathlib import Path
from collections import defaultdict
import microresolve

PACKS = [
    "content-moderation-generic",
    "csam-ncmec",
    "dsr-triage",
    "emotion-detection",
    "eu-ai-act-transparency",
    "language-detect",
    "nist-genai-12-risk",
    "professional-advice-boundary",
]

# Load OOD probes (random CLINC queries from a different domain)
random.seed(42)
CLINC = json.load(open("benchmarks/track1/clinc150_test.json"))
OOD_PROBES = random.sample(CLINC, 30)


def smoke_one(pack_name: str):
    src = Path("packs") / pack_name
    if not src.exists():
        return None
    root = Path("/tmp/smoke") / pack_name
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    shutil.copytree(src, root / pack_name)

    try:
        engine = microresolve.MicroResolve(data_dir=str(root))
        ns = engine.namespace(pack_name)
    except Exception as e:
        return {"pack": pack_name, "error": f"load failed: {e}"}

    ns_meta = json.load(open(src / "_ns.json"))
    threshold = ns_meta.get("default_threshold", 1.5)

    intent_ids = ns.intent_ids()
    n_intents = len(intent_ids)

    # 1. Self-seed test: each seed should resolve to its own intent at High band
    per_intent_self = {}
    total_seeds = 0
    self_correct = 0
    self_in_top_3 = 0
    for iid in intent_ids:
        intent_path = src / f"{iid}.json"
        if not intent_path.exists():
            continue
        intent = json.load(open(intent_path))
        seeds = intent.get("phrases", {}).get("en", [])
        if not seeds:
            continue
        correct = 0
        top3 = 0
        for seed in seeds:
            r = ns.resolve(seed)
            total_seeds += 1
            top = next((i for i in r.intents if i.score >= threshold), None)
            if top is not None and top.id == iid:
                correct += 1
                self_correct += 1
            top3_ids = [i.id for i in r.intents[:3]]
            if iid in top3_ids:
                self_in_top_3 += 1
                top3 += 1
        per_intent_self[iid] = (correct, top3, len(seeds))

    # 2. OOD test: 30 random CLINC queries — should NOT fire High band
    ood_fires = []
    for probe in OOD_PROBES:
        r = ns.resolve(probe["text"])
        top = next((i for i in r.intents if i.score >= threshold), None)
        if top is not None:
            ood_fires.append({"q": probe["text"], "fired": top.id, "score": top.score})

    return {
        "pack": pack_name,
        "n_intents": n_intents,
        "threshold": threshold,
        "self_seed_acc": self_correct / total_seeds if total_seeds else 0.0,
        "self_seed_top3": self_in_top_3 / total_seeds if total_seeds else 0.0,
        "total_seeds": total_seeds,
        "ood_fp_rate": len(ood_fires) / 30,
        "ood_fires": ood_fires[:5],
        "per_intent": per_intent_self,
    }


def main():
    results = []
    for p in PACKS:
        print(f"--- {p} ---", flush=True)
        res = smoke_one(p)
        if res is None:
            print(f"  pack dir missing")
            continue
        if "error" in res:
            print(f"  ERROR: {res['error']}")
            results.append(res)
            continue
        print(f"  intents={res['n_intents']}  threshold={res['threshold']}")
        print(f"  self-seed top-1: {res['self_seed_acc']:.1%}  top-3: {res['self_seed_top3']:.1%}")
        print(f"  OOD FP rate (CLINC, n=30): {res['ood_fp_rate']:.1%}")
        if res['ood_fires']:
            for f in res['ood_fires'][:3]:
                print(f"    fired: '{f['q'][:60]}' → {f['fired']} ({f['score']:.2f})")
        weakest = sorted(res["per_intent"].items(), key=lambda kv: kv[1][0]/max(kv[1][2],1))[:3]
        for iid, (c, t3, n) in weakest:
            if c < n:
                print(f"    weakest: {iid}  top-1: {c}/{n}  top-3: {t3}/{n}")
        results.append(res)
        print()

    # Summary table
    print("=" * 80)
    print(f"{'Pack':32s} self-top1  self-top3  OOD-FP   Verdict")
    print("=" * 80)
    for r in results:
        if "error" in r:
            print(f"  {r['pack']:30s} ERROR: {r['error']}")
            continue
        verdict = "OK" if r["self_seed_acc"] >= 0.7 and r["ood_fp_rate"] <= 0.20 else (
            "WEAK" if r["self_seed_acc"] >= 0.5 else "BROKEN"
        )
        print(f"  {r['pack']:30s} {r['self_seed_acc']:6.1%}  {r['self_seed_top3']:6.1%}    {r['ood_fp_rate']:5.1%}   {verdict}")

    out = Path("benchmarks/results")
    out.mkdir(exist_ok=True)
    json.dump(results, open(out / "alpha_smoke.json", "w"), indent=2, default=str)
    print(f"\nFull results: benchmarks/results/alpha_smoke.json")


if __name__ == "__main__":
    main()
