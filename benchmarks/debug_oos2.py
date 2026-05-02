#!/usr/bin/env python3
"""OOS leak source diagnosis — same OOS queries, three configs:
   1. grounded_l1 = true   (default — runtime synonym substitution active)
   2. grounded_l1 = false  (L1 morphology/abbreviation only, no synonyms)
   3. raw query, no L1 at all (would need engine flag — skipped)

If (2) rejects significantly more than (1), synonym substitution is the
culprit. If (2) is just as leaky as (1), then morph/abbrv edges (or L2
itself) are noisy.

Reuses the agent-tools namespace already populated with 129 tools.
"""
import json, urllib.request

BASE = "http://localhost:8181"
NS = "agent-tools"

OOS = [
    "what's the weather today",
    "tell me a joke",
    "what's 2 plus 2",
    "who won the world cup",
    "translate hello to spanish",
    "set a timer for 5 minutes",
    "play some music",
    "what time is it in Tokyo",
    "recommend a good book",
    "how do I cook pasta",
]


def req(method, path, body=None, ns=None, t=60):
    h = {"Content-Type": "application/json"}
    if ns: h["X-Namespace-ID"] = ns
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"{BASE}{path}", data=data, headers=h, method=method)
    with urllib.request.urlopen(r, timeout=t) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def measure(grounded_l1, label):
    rejected = 0
    leaks = []
    for q in OOS:
        res = req("POST", "/api/route_multi",
                  {"query": q, "log": False, "grounded_l1": grounded_l1}, ns=NS)
        confirmed = [r["id"] for r in (res.get("confirmed") or [])]
        ranked    = [(r["id"], round(r.get("score", 0), 2))
                     for r in (res.get("ranked") or [])[:3]]
        if not confirmed:
            rejected += 1
        else:
            leaks.append((q, confirmed, ranked))
    print(f"\n[{label}] reject {rejected}/{len(OOS)} = {100*rejected/len(OOS):.0f}%")
    for q, c, r in leaks:
        print(f"  ✗ {q!r}")
        print(f"    confirmed: {c}")
        print(f"    top-3 raw: {r}")


def populate():
    from pathlib import Path
    fixtures = Path(__file__).resolve().parent / "mcp_fixtures"
    all_tools = []
    for f in ["stripe", "linear", "notion", "slack", "shopify"]:
        spec = json.loads((fixtures / f"{f}.json").read_text())
        all_tools.extend(spec.get("tools", []))
    print(f"importing {len(all_tools)} tools (LLM L1 augment, ~90s)...")
    try: req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception: pass
    req("POST", "/api/namespaces", {"namespace_id": NS, "description": "OOS diag"})
    req("POST", "/api/import/mcp/apply", {
        "tools_json": json.dumps({"tools": all_tools}),
        "selected": [t["name"] for t in all_tools],
        "domain": "",
    }, ns=NS, t=600)


def main():
    populate()
    measure(grounded_l1=True,  label="grounded_l1=true  (synonym sub ON, default)")
    measure(grounded_l1=False, label="grounded_l1=false (synonym sub OFF, only morph/abbrv)")


if __name__ == "__main__":
    main()
