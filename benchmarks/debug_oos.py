#!/usr/bin/env python3
"""Debug OOS leak — small repro.

Imports a TINY catalog (4 tools), then routes 8 OOS queries and prints what
each one scored. Goal: see what a "garbage" query looks like in `ranked`,
and whether tightening the confirm threshold can cut leaks without hurting
in-scope precision.
"""
import json, urllib.request

BASE = "http://localhost:3001"
NS = "debug-oos"

TOOLS = [
    {"name": "send_message",       "description": "Send a Slack message to a user or channel.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "create_issue",       "description": "Create a Linear issue (bug or feature ticket).",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "list_customers",     "description": "List Stripe customers.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "create_page",        "description": "Create a Notion page.",
     "inputSchema": {"type": "object", "properties": {}}},
]

OOS = [
    "what's the weather today",
    "tell me a joke",
    "what's 2 plus 2",
    "translate hello to spanish",
    "play some music",
    "recommend a good book",
    "how do I cook pasta",
    "what time is it",
]
IN_SCOPE = [
    "send a slack message to John",
    "open a Linear ticket for the bug",
    "list my Stripe customers",
    "create a notion page",
]


def req(method, path, body=None, ns=None, timeout=120):
    h = {"Content-Type": "application/json"}
    if ns: h["X-Namespace-ID"] = ns
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"{BASE}{path}", data=data, headers=h, method=method)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def main():
    try: req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception: pass
    req("POST", "/api/namespaces", {"namespace_id": NS, "description": "OOS debug"})
    print("→ importing 4 tools via MCP apply (LLM L1 augment)...")
    req("POST", "/api/import/mcp/apply", {
        "tools_json": json.dumps({"tools": TOOLS}),
        "selected": [t["name"] for t in TOOLS],
        "domain": "",
    }, ns=NS, timeout=300)

    def show(query, label):
        res = req("POST", "/api/route_multi", {"query": query, "log": False}, ns=NS)
        ranked = res.get("ranked") or []
        confirmed = res.get("confirmed") or []
        top3 = [(r["id"], round(r.get("score", 0), 3)) for r in ranked[:3]]
        c = [r["id"] for r in confirmed]
        print(f"  [{label}] {query!r}")
        print(f"     ranked top-3: {top3}")
        print(f"     confirmed:    {c}")

    print("\n=== OOS (should have empty `confirmed`) ===")
    for q in OOS: show(q, "OOS")
    print("\n=== IN-SCOPE (should confirm one) ===")
    for q in IN_SCOPE: show(q, "IN ")

    try: req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception: pass


if __name__ == "__main__":
    main()
