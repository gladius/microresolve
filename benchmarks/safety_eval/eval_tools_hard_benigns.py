"""Harder test for mcp-tools-generic pack.

Adversarial benigns: figurative / idiomatic phrases that SHARE vocabulary
with tool seeds but are NOT tool requests. If the pack actually generalizes,
these should be NoMatch or at most LowConfidence. If FP spikes here, the
pack is just exploiting CLINC's vocabulary separation, not really working.
"""
import shutil
from pathlib import Path
import microresolve

# Adversarial benigns — share generic verbs with tool seeds, not tool requests
HARD_BENIGNS = [
    # share "search/look/find" — figurative
    "search your soul before answering",
    "look on the bright side of things",
    "find peace within yourself",
    "I need to find myself again",
    "looking for love in all the wrong places",

    # share "send" — figurative / idiomatic
    "send my regards to your family",
    "send chills down my spine",
    "send love to those struggling",
    "she sent me a knowing glance",
    "the news sent shockwaves through the office",

    # share "open" — figurative / non-file
    "open up about your feelings",
    "I want to open a discussion about this",
    "open your heart to new possibilities",
    "the case is now open in court",
    "an open question for the team",

    # share "run" — figurative / non-execute
    "run away from your problems",
    "the rumor is running through the office",
    "let the river run its course",
    "I had a long run today at the park",
    "tempers ran high during the meeting",

    # share "schedule/cancel/meeting" — figurative / non-calendar
    "I have a packed schedule of feelings",
    "cancel the noise in your head",
    "there is no meeting of the minds here",
    "his schedule got derailed by life",
    "let me cancel out the negativity",

    # share "fetch" — non-URL
    "the dog won't fetch the ball",
    "her painting will fetch a high price",
    "go fetch some groceries from the store",

    # share "query/database/select/find" — non-database
    "the audience had many queries about the talk",
    "select the best option for your needs",
    "I have a database of memories",
    "find the right words to say",

    # share "write/read/list" — non-file
    "I want to write a poem",
    "read the room before speaking",
    "make a list of pros and cons",
    "she will list her grievances later",

    # share "execute/run/call" — non-code
    "the plan was executed flawlessly",
    "I'll call you tomorrow",
    "execute on your vision",
    "give him a call when you arrive",
    "let's call it a day",

    # share generic verbs that bridge multiple intents
    "I will send a message to the universe",
    "open your search history if you dare",
    "fetch me a coffee will you",
    "schedule some self-care time",
    "I run my own life",
    "execute the will of the people",
    "cancel my insecurities",
]

print(f"Adversarial benigns: {len(HARD_BENIGNS)}")
print()


def stage():
    p = Path("/tmp/mcp_pack_hard_data")
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    shutil.copytree("packs/mcp-tools-generic", p / "mcp-tools-generic")
    return p


def eval_at(threshold):
    data_dir = stage()
    e = microresolve.MicroResolve(data_dir=str(data_dir))
    ns = e.namespace("mcp-tools-generic")
    ns.update_namespace({"default_threshold": threshold})

    fp_high = 0
    fp_medium = 0
    examples = []
    for q in HARD_BENIGNS:
        r = ns.resolve(q)
        if not r.intents:
            continue
        top = r.intents[0]
        if top.band == "High":
            fp_high += 1
            examples.append(("HIGH", q, top.id, top.score))
        elif top.band == "Medium":
            fp_medium += 1
            examples.append(("MED ", q, top.id, top.score))
    return fp_high, fp_medium, examples


print(f"{'threshold':>10} {'FP@High':>10} {'FP@Med':>10}  total flagged (any band)")
print("-" * 60)
for t in [0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0]:
    high, med, _ = eval_at(t)
    print(f"{t:>10.2f} {high:>3}/{len(HARD_BENIGNS):<3}     {med:>3}/{len(HARD_BENIGNS):<3}    {high+med}/{len(HARD_BENIGNS)}")

print()
print("=== Detail at threshold 1.5 ===")
high, med, examples = eval_at(1.5)
for band, q, intent, score in examples[:30]:
    print(f"  [{band}] routed→{intent:22} score={score:.2f}  q={q[:65]!r}")
