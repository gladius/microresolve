"""Look at the actual scores for the 2 queries where policy_override fired.
Show what natural index weights look like (lex_only) vs with the bonus.
"""
import json, shutil
from pathlib import Path
import microresolve

PACK_SRC = Path("packs/eu-ai-act-prohibited")
ROOT = Path("/tmp/trace_policy")
QUERIES = [
    "predictive policing combining personality profile with witness reports and warrants",
    "CSAM detection model for content moderation",
]

if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True)

# Stage lex_only
dest = ROOT / "lex_only" / "eu-ai-act-prohibited"
shutil.copytree(PACK_SRC, dest)
ns_p = dest / "_ns.json"
ns_json = json.load(open(ns_p))
ns_json.pop("policy_overrides", None)
json.dump(ns_json, open(ns_p, "w"), indent=2)

# Stage both
dest2 = ROOT / "both" / "eu-ai-act-prohibited"
shutil.copytree(PACK_SRC, dest2)

e1 = microresolve.MicroResolve(data_dir=str(ROOT / "lex_only"))
e2 = microresolve.MicroResolve(data_dir=str(ROOT / "both"))
ns1 = e1.namespace("eu-ai-act-prohibited")
ns2 = e2.namespace("eu-ai-act-prohibited")

for q in QUERIES:
    print("=" * 72)
    print(f"QUERY: {q}")
    print()
    print("--- LEX ONLY (natural index scoring) ---")
    r = ns1.resolve(q)
    for i, m in enumerate(r.intents[:6]):
        print(f"  {i+1}. {m.id:35s} score={m.score:.2f}  conf={m.confidence:.2f}  band={m.band}")
    print()
    print("--- LEX + POLICY OVERRIDE (with bonus) ---")
    r = ns2.resolve(q)
    for i, m in enumerate(r.intents[:6]):
        print(f"  {i+1}. {m.id:35s} score={m.score:.2f}  conf={m.confidence:.2f}  band={m.band}")
    print()

# Now look at what tokens map to what intents in the index
print("=" * 72)
print("Token → intent weight breakdown for KEY tokens:")
print()
print("(Querying the resolver's index would need more API surface;")
print("for now, the score differences above tell the story.)")
