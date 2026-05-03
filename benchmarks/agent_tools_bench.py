#!/usr/bin/env python3
"""End-to-end MCP tool-routing benchmark.

Drives MicroResolve's existing MCP-import + auto-learn pipeline against 5
production servers from Smithery (the public MCP registry):

  stripe / linear / notion / slack / shopify  →  ~129 tools total

The script only calls public endpoints — anyone can recreate this run in
their own namespace by pointing at a server with `LLM_API_KEY` set:

  GET  /api/import/mcp/fetch       (server's tools/list)
  POST /api/import/mcp/apply       (creates intents + LLM L1 augmentation)
  POST /api/resolve                (the routing call)
  POST /api/training/review        (auto-learn Turn 2 + Turn 3)
  POST /api/training/apply         (commit the correction)

Two stages:

  baseline     — straight after MCP import (LLM seeds + L1 augment)
  + auto-learn — review + apply on every non-exact query, then re-measure

Three test sets, each scored with the right metric:

  single_intent  →  top-1, top-3
  multi_intent   →  exact, recall, precision, F1, top-K
  out_of_scope   →  reject rate (confirmed should be empty)

Latency is the server-reported `routing_us` (core engine), not HTTP
wall-clock.

Specs cached to benchmarks/mcp_fixtures/ on first run; cached runs are
fully offline.

Run:
  cargo build --release --features server
  ./target/release/microresolve-studio --port 3001 --no-open --data /tmp/mr_bench &
  python3 benchmarks/agent_tools_bench.py

Cost: ~$0.55 Haiku 4.5.   Wall-clock: ~10 min.
"""

from __future__ import annotations
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "mcp_fixtures"
RESULTS = ROOT / "results"
BASE = os.environ.get("MR_SERVER_URL", "http://localhost:3001")
NS = "agent-tools"

DOMAINS = [
    ("stripe",  "stripe"),
    ("linear",  "linear"),
    ("notion",  "notion"),
    ("slack",   "node2flow/slack"),
    ("shopify", "shopify"),
]

# ── Test sets ───────────────────────────────────────────────────────────────
#
# `expected` is a tuple of acceptable tool names. Genuinely-ambiguous queries
# list every legitimate answer (e.g. "list my customers" matches Stripe OR
# Shopify). Single-intent: top-1 hits if engine returns ANY of them.

SINGLE_INTENT = [
    # Stripe
    ("list my Stripe customers",                    ("list_customers",)),
    ("show me products in Stripe",                  ("list_products",)),
    ("what's my balance",                           ("retrieve_balance",)),
    ("look up recent payment intents",              ("list_payment_intents",)),
    ("list active subscriptions",                   ("list_subscriptions",)),
    ("search Stripe docs about webhooks",           ("search_stripe_documentation",)),
    ("show recent disputes",                        ("list_disputes",)),
    # Linear
    ("create a bug ticket for the login issue",     ("create_issue",)),
    ("open a Linear issue for that bug",            ("create_issue",)),
    ("close that ticket, it's done",                ("update_issue",)),
    ("add a comment to the bug",                    ("create_comment",)),
    ("what issues are assigned to me",              ("list_issues",)),
    ("show my Linear projects",                     ("list_projects",)),
    ("create a new project for q1",                 ("create_project",)),
    # Notion
    ("search Notion for the deployment runbook",    ("notion-search",)),
    ("find the onboarding guide in our docs",       ("notion-search",)),
    ("create a Notion page for meeting notes",      ("notion-create-pages",)),
    ("update that page's title",                    ("notion-update-page",)),
    ("duplicate that page",                         ("notion-duplicate-page",)),
    ("create a new database for projects",          ("notion-create-database",)),
    # Slack
    ("send a message to John",                      ("slack_send_message",)),
    ("post in the dev channel that the deploy is done", ("slack_send_message",)),
    ("create a channel for the launch team",        ("slack_create_channel",)),
    ("react with thumbs up to that message",        ("slack_add_reaction",)),
    ("search slack for the word deploy",            ("slack_search_messages",)),
    ("join the random channel",                     ("slack_join_channel",)),
    # Shopify
    ("add a new t-shirt to the Shopify catalog",    ("SHOPIFY_CREATE_PRODUCT",)),
    ("list my recent Shopify orders",               ("SHOPIFY_GET_ORDER_LIST",)),
    ("show me order details for #1234",             ("SHOPIFY_GET_ORDERSBY_ID",)),
    ("delete that broken product",                  ("SHOPIFY_DELETE_PRODUCT",)),
    ("update that order's status",                  ("SHOPIFY_UPDATE_ORDER",)),
    # Genuinely ambiguous
    ("list my customers",                           ("list_customers", "SHOPIFY_GET_ALL_CUSTOMERS")),
    ("show me all the users",                       ("list_users", "notion-get-users", "slack_list_users")),
    ("search recent messages for that bug",         ("slack_search_messages", "list_issues")),
]

# Multi-intent: a single user utterance that asks for 2-3 tools at once.
# `expected` is a frozenset of tool names that must ALL be detected.
# Scored on exact / recall / precision / F1 / top-K (k = |expected|).
MULTI_INTENT = [
    ("create a Linear issue for the bug and post it in slack",
        frozenset({"create_issue", "slack_send_message"})),
    ("cancel the subscription and refund the customer",
        frozenset({"cancel_subscription", "create_refund"})),
    ("create a Notion page for the launch and add a Linear ticket to track it",
        frozenset({"notion-create-pages", "create_issue"})),
    ("send a slack message to the team and create an issue for the regression",
        frozenset({"slack_send_message", "create_issue"})),
    ("list my stripe customers and my shopify orders",
        frozenset({"list_customers", "SHOPIFY_GET_ORDER_LIST"})),
    ("update the order status and notify the customer in slack",
        frozenset({"SHOPIFY_UPDATE_ORDER", "slack_send_message"})),
    ("close the ticket and post in slack that it's done",
        frozenset({"update_issue", "slack_send_message"})),
    ("create a new stripe product and add it to the shopify catalog",
        frozenset({"create_product", "SHOPIFY_CREATE_PRODUCT"})),
    ("search notion for the runbook and create a linear issue if outdated",
        frozenset({"notion-search", "create_issue"})),
    ("create a slack channel and a notion page for the launch team",
        frozenset({"slack_create_channel", "notion-create-pages"})),
    ("list disputes and refund the legitimate ones",
        frozenset({"list_disputes", "create_refund"})),
    ("create the issue, assign it to me, and post about it in slack",
        frozenset({"create_issue", "slack_send_message"})),
]

# Out-of-scope: should match nothing (or nothing above the confirm threshold).
# Scored on reject rate — higher is better.
OUT_OF_SCOPE = [
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


# ── HTTP helpers ────────────────────────────────────────────────────────────

def _req(method, path, body=None, ns=None, timeout=300.0):
    headers = {"Content-Type": "application/json"}
    if ns:
        headers["X-Namespace-ID"] = ns
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} {method} {path}: {msg}") from None


def health_check():
    try:
        _req("GET", "/api/namespaces", timeout=5)
        return True
    except Exception as e:
        print(f"server not reachable at {BASE}: {e}")
        return False


# ── Smithery fetch (cached) ─────────────────────────────────────────────────

def fetch_specs():
    FIXTURES.mkdir(exist_ok=True)
    specs = {}
    for domain, qname in DOMAINS:
        cache = FIXTURES / f"{domain}.json"
        if cache.exists():
            specs[domain] = json.loads(cache.read_text())
            print(f"  ✓ {domain} cached ({len(specs[domain].get('tools',[]))} tools)")
            continue
        url_q = urllib.parse.quote(qname, safe="")
        spec = _req("GET", f"/api/import/mcp/fetch?name={url_q}")
        cache.write_text(json.dumps(spec, indent=2))
        n = len(spec.get("tools", []))
        print(f"  ↓ {domain}  ({qname}) → {n} tools")
        specs[domain] = spec
    return specs


# ── Setup ───────────────────────────────────────────────────────────────────

def setup_namespace(all_tools):
    """Drop + recreate the namespace, then import all tools via the MCP
    pipeline. Server's LLM L1 augmentation runs as part of /apply."""
    try:
        _req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception:
        pass
    _req("POST", "/api/namespaces",
         {"namespace_id": NS, "description": "agent-tools benchmark"})
    res = _req("POST", "/api/import/mcp/apply", {
        "tools_json": json.dumps({"tools": all_tools}),
        "selected": [t["name"] for t in all_tools],
        "domain": "",
    }, ns=NS, timeout=600.0)
    return int(res.get("imported") or res.get("created") or len(all_tools))


# ── Routing call (server-reported latency) ──────────────────────────────────

def route(query):
    res = _req("POST", "/api/resolve",
               {"query": query, "log": False}, ns=NS)
    intents    = [r["id"] for r in (res.get("intents") or [])]
    routing_us = float(res.get("routing_us") or 0)
    return intents, intents, routing_us


# ── Metrics ─────────────────────────────────────────────────────────────────

def measure_single(queries):
    hits1 = hits3 = 0
    lats = []
    misses = []
    for query, expected in queries:
        accepted = set(expected)
        ranked, _, us = route(query)
        lats.append(us)
        top1 = ranked[0] if ranked else None
        top3 = set(ranked[:3])
        if top1 in accepted: hits1 += 1
        if accepted & top3:  hits3 += 1
        else: misses.append((query, list(expected), top1 or "(none)"))
    n = len(queries)
    return {
        "n": n,
        "top1": round(100 * hits1 / n, 1) if n else 0,
        "top3": round(100 * hits3 / n, 1) if n else 0,
        "p50_us": round(statistics.median(lats), 1) if lats else 0,
        "miss_sample": misses[:8],
    }


def measure_multi(queries):
    n = len(queries)
    exact = 0
    sum_p = sum_r = 0.0
    topk_hits = 0
    lats = []
    miss_sample = []
    for query, expected in queries:
        ranked, confirmed, us = route(query)
        lats.append(us)
        conf = set(confirmed)
        exp  = set(expected)
        inter = exp & conf
        precision = (len(inter) / len(conf)) if conf else 0.0
        recall    = (len(inter) / len(exp)) if exp else 0.0
        sum_p += precision
        sum_r += recall
        if conf == exp:
            exact += 1
        else:
            miss_sample.append((query, sorted(exp), sorted(conf)))
        if exp.issubset(set(ranked[:len(exp)])):
            topk_hits += 1
    p = sum_p / n * 100 if n else 0
    r = sum_r / n * 100 if n else 0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0
    return {
        "n": n,
        "exact":     round(100 * exact / n, 1) if n else 0,
        "precision": round(p, 1),
        "recall":    round(r, 1),
        "f1":        round(f1, 1),
        "topk":      round(100 * topk_hits / n, 1) if n else 0,
        "p50_us":    round(statistics.median(lats), 1) if lats else 0,
        "miss_sample": miss_sample[:6],
    }


def measure_oos(queries):
    n = len(queries)
    rejected = 0
    lats = []
    leaked = []
    for query in queries:
        _, confirmed, us = route(query)
        lats.append(us)
        if not confirmed:
            rejected += 1
        else:
            leaked.append((query, confirmed))
    return {
        "n": n,
        "reject_rate": round(100 * rejected / n, 1) if n else 0,
        "p50_us": round(statistics.median(lats), 1) if lats else 0,
        "leaked_sample": leaked[:6],
    }


# ── Auto-learn (proper review + apply) ──────────────────────────────────────

def auto_learn():
    """Run review+apply on every non-exact query in the in-scope sets.
    Uses Turn 2 (missed phrase generation) + Turn 3 (false-positive
    suppression) — the same pipeline the server's auto-learn worker runs."""
    work = []
    # Single-intent: missed if top-1 not in accepted
    for query, expected in SINGLE_INTENT:
        ranked, confirmed, _ = route(query)
        accepted = set(expected)
        if ranked and ranked[0] in accepted:
            continue
        # Ground truth is "first acceptable id" for single-intent.
        work.append((query, list(confirmed), [expected[0]]))
    # Multi-intent: anything where confirmed != expected
    for query, expected in MULTI_INTENT:
        _, confirmed, _ = route(query)
        if set(confirmed) == set(expected):
            continue
        work.append((query, list(confirmed), sorted(expected)))

    learned = 0
    for query, detected, ground in work:
        try:
            review = _req("POST", "/api/training/review", {
                "message":      query,
                "detected":     detected,
                "ground_truth": ground,
            }, ns=NS, timeout=120.0)
            _req("POST", "/api/training/apply",
                 {"query": query, "result": review}, ns=NS, timeout=60.0)
            learned += 1
        except Exception as e:
            print(f"  [warn] auto-learn '{query[:50]}…': {e}")
    return len(work), learned


# ── Main ────────────────────────────────────────────────────────────────────

def stage(label):
    return {
        "label": label,
        "single": measure_single(SINGLE_INTENT),
        "multi":  measure_multi(MULTI_INTENT),
        "oos":    measure_oos(OUT_OF_SCOPE),
    }


def print_stage(s):
    si, mi, oo = s["single"], s["multi"], s["oos"]
    print(f"\n  [{s['label']}]")
    print(f"    single-intent ({si['n']}):  top-1 {si['top1']:>5.1f}%   top-3 {si['top3']:>5.1f}%   p50 {si['p50_us']:>6.1f}µs")
    print(f"    multi-intent  ({mi['n']}):  exact {mi['exact']:>5.1f}%   F1  {mi['f1']:>5.1f}%   recall {mi['recall']:>5.1f}%   prec {mi['precision']:>5.1f}%   p50 {mi['p50_us']:>6.1f}µs")
    print(f"    out-of-scope  ({oo['n']}):  reject {oo['reject_rate']:>5.1f}%   p50 {oo['p50_us']:>6.1f}µs")


def main():
    print(f"Agent-tools benchmark → {BASE}\n")
    if not health_check():
        return 1

    print("→ fetching MCP specs from 5 Smithery servers (cached after first run)...")
    specs = fetch_specs()
    all_tools = []
    for spec in specs.values():
        all_tools.extend(spec.get("tools", []))
    print(f"  {len(all_tools)} tools across {len(specs)} domains\n")

    print("→ stage 1: import tools via /api/import/mcp/apply (LLM seeds + L1 augment)")
    print("  may take 60-120s with Haiku...")
    t0 = time.perf_counter()
    n_imported = setup_namespace(all_tools)
    print(f"  imported {n_imported} tools in {time.perf_counter() - t0:.1f}s")

    baseline = stage("baseline (after MCP import)")
    print_stage(baseline)

    print("\n→ stage 2: auto-learn (review + apply on every non-exact query)")
    t0 = time.perf_counter()
    n_work, n_learned = auto_learn()
    print(f"  applied {n_learned}/{n_work} corrections in {time.perf_counter() - t0:.1f}s")

    after = stage("after auto-learn")
    print_stage(after)

    # Save + summary
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "agent_tools.json"
    out.write_text(json.dumps({
        "domains":      [d for d, _ in DOMAINS],
        "tool_count":   len(all_tools),
        "baseline":     baseline,
        "after_learn":  after,
        "corrections_attempted": n_work,
        "corrections_applied":   n_learned,
    }, indent=2, default=str))
    print(f"\n  → wrote {out}")

    print("\n══ Summary (core engine latency, server-reported routing_us) ══")
    print(f"  {'Stage':<28s} | {'Single top-1':>12s} {'Multi F1':>10s} {'OOS reject':>11s} {'p50 µs':>8s}")
    for s in (baseline, after):
        si, mi, oo = s["single"], s["multi"], s["oos"]
        p50 = statistics.median([si["p50_us"], mi["p50_us"], oo["p50_us"]])
        print(f"  {s['label']:<28s} | {si['top1']:>11.1f}% {mi['f1']:>9.1f}% {oo['reject_rate']:>10.1f}% {p50:>8.1f}")

    try:
        _req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
