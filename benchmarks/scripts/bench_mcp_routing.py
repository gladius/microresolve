#!/usr/bin/env python3
"""MCP routing benchmark — end-to-end import + route + auto-learn.

Flow:
  1. Fetch real tool definitions from Smithery registry via /api/import/mcp/fetch
  2. Import via /api/import/mcp/apply  — LLM generates seed phrases from tool descriptions
  3. Cold test: route queries, measure accuracy
  4. Auto-learn: LLM span extraction on missed queries (ground truth provided → Turn 1 skipped)
  5. Re-test: measure improvement

Usage:
  python3 benchmarks/scripts/bench_mcp_routing.py
"""

import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import check_server, create_namespace, delete_namespace, _req

DATASETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets")
NS = f"bench-mcp-{int(time.time())}"

SERVERS = [
    {"name": "notion", "domain": "notion", "description": "Notion workspace — pages, databases, comments, users"},
    {"name": "stripe", "domain": "stripe", "description": "Stripe payments — customers, invoices, subscriptions, disputes"},
]


def create_domain(domain, description):
    """Create a domain within the namespace."""
    try:
        _req("POST", "/api/domains", {"domain": domain, "description": description}, ns=NS)
        print(f"  Domain '{domain}' created", flush=True)
    except Exception as e:
        print(f"  [warn] domain '{domain}': {e}", flush=True)


def fetch_and_import(server_name, domain):
    """Fetch tools from Smithery and import via mcp_apply. Returns tool count."""
    print(f"  Fetching {server_name} from Smithery...", flush=True)
    data = _req("GET", f"/api/import/mcp/fetch?name={server_name}", ns=NS)
    tools = data.get("tools", [])
    if not tools:
        print(f"  [warn] no tools found for {server_name}")
        return 0

    tool_names = [t["name"] for t in tools if t.get("name")]
    print(f"  Importing {len(tool_names)} tools into '{domain}' domain via LLM...", flush=True)

    _req("POST", "/api/import/mcp/apply", {
        "tools_json": json.dumps(data),
        "selected": tool_names,
        "domain": domain,
    }, ns=NS, timeout=120)

    print(f"  {server_name}: {len(tool_names)} tools seeded", flush=True)
    return len(tool_names)


def evaluate(test):
    results = []
    n = len(test)
    for i, ex in enumerate(test):
        if i % 10 == 0:
            print(f"  [routing] {i}/{n}...", flush=True)
        query, expected = ex["text"], set(ex["intents"])
        t0 = time.perf_counter()
        resp = _req("POST", "/api/route_multi", {"query": query, "log": False}, ns=NS)
        lat = (time.perf_counter() - t0) * 1_000_000
        confirmed = set(r["id"] for r in resp.get("confirmed", []))
        ranked    = [r["id"] for r in resp.get("ranked", [])]
        k = len(expected)
        inter = expected & confirmed
        results.append({
            "text": query,
            "expected": expected,
            "confirmed": confirmed,
            "exact":     confirmed == expected,
            "precision": len(inter) / len(confirmed) if confirmed else 0,
            "recall":    len(inter) / len(expected)  if expected  else 0,
            "topk":      expected.issubset(set(ranked[:k])),
            "missed":    expected - confirmed,
            "latency_us": lat,
        })
    return results


def metrics(results):
    n = len(results)
    p  = sum(r["precision"] for r in results) / n * 100
    r  = sum(r["recall"]    for r in results) / n * 100
    f1 = 2*p*r/(p+r) if (p+r) else 0
    exact = sum(1 for r in results if r["exact"]) / n * 100
    topk  = sum(1 for r in results if r["topk"])  / n * 100
    lats  = sorted(r["latency_us"] for r in results)
    p50   = lats[len(lats)//2] if lats else 0
    return {"exact": exact, "precision": p, "recall": r, "f1": f1, "topk": topk, "p50": p50}


def auto_learn(results):
    # Learn from ALL non-exact queries:
    # - missed intents  → Turn 2 phrase generation
    # - false positives → Turn 3 anti-Hebbian suppression
    non_exact = [r for r in results if not r["exact"]]
    missed_count = sum(1 for r in non_exact if r["missed"])
    fp_only_count = len(non_exact) - missed_count
    print(f"\n  Learning from {len(non_exact)} non-exact queries "
          f"({missed_count} with misses, {fp_only_count} false-positive-only)...", flush=True)
    learned = 0
    for i, r in enumerate(non_exact):
        if i % 10 == 0:
            print(f"  [learning] {i}/{len(non_exact)}...", flush=True)
        try:
            review = _req("POST", "/api/training/review", {
                "message": r["text"],
                "detected": list(r["confirmed"]),
                "ground_truth": list(r["expected"]),
            }, ns=NS)
            _req("POST", "/api/training/apply", {
                "query": r["text"], "result": review
            }, ns=NS)
            learned += 1
        except Exception as e:
            print(f"  [warn] {e}")
    print(f"  [learning] done — {learned}/{len(non_exact)} corrections applied", flush=True)
    return learned


def print_row(label, m, corrections=None):
    c = f"{corrections}" if corrections is not None else "-"
    print(f"  {label:<14} {m['exact']:>6.1f}%  {m['recall']:>6.1f}%  {m['precision']:>6.1f}%  {m['f1']:>6.1f}%  {m['topk']:>6.1f}%  {c:>5}  {m['p50']:>6.0f}µs")


def main():
    if not check_server():
        sys.exit(1)

    test = json.load(open(os.path.join(DATASETS, "mcp_notion_stripe_test.json")))
    n_multi = sum(1 for ex in test if len(ex["intents"]) > 1)

    print("=" * 78)
    print(f"  MCP Routing Benchmark — end-to-end import + route + auto-learn")
    print(f"  Servers: notion (12 tools) + stripe (25 tools) — fetched live from Smithery")
    print(f"  {len(test)} test queries ({n_multi} multi-tool)")
    print("=" * 78)

    # Import phase
    print("\n  [1/4] Creating namespace + domains...", flush=True)
    create_namespace(NS)
    for s in SERVERS:
        create_domain(s["domain"], s["description"])

    print("\n  [2/4] Fetching + importing MCP tools (LLM seed generation)...", flush=True)
    total_tools = 0
    for s in SERVERS:
        total_tools += fetch_and_import(s["name"], s["domain"])
    print(f"\n  Imported {total_tools} tools total\n")

    # Cold test
    print("  [3/4] Cold test (no prior learning)...", flush=True)
    print(f"\n  {'':14} {'Exact':>6}  {'Recall':>6}  {'Precis':>6}  {'F1':>6}  {'Top-K':>6}  {'Corr':>5}  {'p50µs':>7}")
    print(f"  {'-'*74}")
    before = evaluate(test)
    print_row("cold start", metrics(before), 0)

    # Auto-learn
    print("\n  [4/5] Auto-learn pass...", flush=True)
    corrections = auto_learn(before)

    # Re-test
    print("\n  [5/5] Re-test after learning...", flush=True)
    after = evaluate(test)
    print_row("after learn", metrics(after), corrections)

    # Failures
    failures = [r for r in after if not r["exact"]]
    if failures:
        print(f"\n  Remaining failures ({len(failures)}/{len(test)}):")
        for r in failures[:5]:
            print(f"  · {r['text'][:65]}")
            print(f"    expected={sorted(r['expected'])}  missed={sorted(r['missed'])}")

    delete_namespace(NS)
    print(f"\n  Cleaned up {NS}")


if __name__ == "__main__":
    main()
