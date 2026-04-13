#!/usr/bin/env python3
"""
Simulate-driven evaluation: generate realistic queries via LLM, route them,
auto-learn from misses, measure improvement.

Scenarios cover different languages, personalities, and intent domains.
Each scenario produces N queries with LLM-supplied ground truth labels.

Usage:
    python3 tests/simulate_eval.py
    python3 tests/simulate_eval.py --ns baseline-test --queries 5
    python3 tests/simulate_eval.py --no-learn   # baseline only, skip auto-learn
"""

import json, subprocess, sys, argparse, time

BASE_URL = "http://localhost:3001"

# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def curl(method, path, body=None, ns="baseline-test"):
    cmd = ["curl", "-sf", "-w", "\n%{http_code}", "-X", method,
           f"{BASE_URL}/api{path}", "-H", "Content-Type: application/json",
           "-H", f"X-Namespace-ID: {ns}"]
    if body is not None:
        cmd += ["-d", json.dumps(body)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = r.stdout.rsplit("\n", 1)
    raw = lines[0].strip() if len(lines) > 1 else r.stdout.strip()
    status = int(lines[-1].strip()) if lines[-1].strip().isdigit() else 0
    try:
        return status, json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return status, {"_raw": raw}

def post(path, body, ns="baseline-test"): return curl("POST", path, body, ns)
def get(path, ns="baseline-test"):        return curl("GET",  path, None, ns)

# ─── Scenarios ────────────────────────────────────────────────────────────────
# Each scenario: description, language, personality, sophistication, verbosity,
# mode ("normal"/"adversarial"), and optional intent_filter (subset of intents).

SCENARIOS = [
    {
        "name": "EN frustrated billing",
        "language": "English",
        "personality": "frustrated",
        "sophistication": "low",
        "verbosity": "short",
        "mode": "normal",
        "intents": ["billing:charge_card", "billing:refund", "billing:cancel_subscription",
                    "billing:update_payment", "billing:view_invoice"],
    },
    {
        "name": "ES polite billing+shipping",
        "language": "Spanish",
        "personality": "polite",
        "sophistication": "medium",
        "verbosity": "medium",
        "mode": "normal",
        "intents": ["billing:refund", "billing:cancel_subscription",
                    "shipping:track_order", "shipping:cancel_order", "shipping:return_item"],
    },
    {
        "name": "ZH terse account",
        "language": "Chinese",
        "personality": "terse",
        "sophistication": "medium",
        "verbosity": "short",
        "mode": "normal",
        "intents": ["account:reset_password", "account:delete_account",
                    "account:update_profile", "account:verify_email"],
    },
    {
        "name": "FR medium support",
        "language": "French",
        "personality": "polite",
        "sophistication": "medium",
        "verbosity": "medium",
        "mode": "normal",
        "intents": ["support:create_ticket", "support:check_status",
                    "support:escalate", "support:close_ticket", "support:feedback"],
    },
    {
        "name": "DE high sophistication cross-domain",
        "language": "German",
        "personality": "formal",
        "sophistication": "high",
        "verbosity": "long",
        "mode": "normal",
        "intents": ["account:reset_password", "account:update_profile",
                    "billing:update_payment", "billing:view_invoice"],
    },
    {
        "name": "EN adversarial mixed",
        "language": "English",
        "personality": "confused",
        "sophistication": "low",
        "verbosity": "medium",
        "mode": "adversarial",
        "intents": [],  # all intents
    },
]

# ─── Core functions ───────────────────────────────────────────────────────────

def get_all_intents(ns):
    _, data = get("/intents", ns)
    if not data:
        return []
    return [i["id"] for i in data]

def simulate_query(scenario, ns, all_intents):
    """Call simulate_turn to get one LLM-generated query with ground truth."""
    intents = scenario["intents"] if scenario["intents"] else all_intents
    body = {
        "personality":    scenario["personality"],
        "sophistication": scenario["sophistication"],
        "verbosity":      scenario["verbosity"],
        "mode":           scenario["mode"],
        "language":       scenario["language"],
        "history":        [],
        "intents":        intents,
    }
    status, data = post("/simulate/turn", body, ns)
    if status != 200 or not data:
        return None
    return {
        "message":     data.get("message", ""),
        "ground_truth": data.get("ground_truth", []),
        "description": data.get("intent_description", ""),
    }

def route_query(query, ns, threshold=0.25):
    """Route a query and return detected intents."""
    status, data = post("/route_multi", {"query": query, "threshold": threshold}, ns)
    if status != 200 or not data:
        return [], "error"
    intents = [r["intent"] for r in data.get("results", [])]
    disposition = data.get("disposition", "?")
    return intents, disposition

def score(detected, ground_truth):
    """
    Partial credit scoring:
      - full match: all ground truth intents detected (order-independent)
      - partial: at least one correct
      - miss: none correct
    """
    if not ground_truth:
        return "skip", True
    det_set = set(detected)
    gt_set  = set(ground_truth)
    if gt_set.issubset(det_set):
        return "pass", True
    if det_set & gt_set:
        return "partial", False
    return "miss", False

def wait_for_worker(ns, timeout=180):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        _, stats = get("/review/stats", ns)
        pending = stats.get("pending", "?") if stats else "?"
        if pending != last:
            print(f"    pending: {pending}")
            last = pending
        if isinstance(pending, int) and pending == 0:
            return True
        time.sleep(3)
    print(f"    ✗ timeout after {timeout}s")
    return False

# ─── Evaluation run ───────────────────────────────────────────────────────────

def run_scenario(scenario, ns, n_queries, all_intents):
    print(f"\n  [{scenario['name']}]  lang={scenario['language']}  mode={scenario['mode']}")
    queries = []
    for i in range(n_queries):
        q = simulate_query(scenario, ns, all_intents)
        if not q or not q["message"]:
            print(f"    ⚠ simulate failed (query {i+1})")
            continue
        queries.append(q)

    results = []
    for q in queries:
        detected, disp = route_query(q["message"], ns)
        outcome, correct = score(detected, q["ground_truth"])
        results.append({**q, "detected": detected, "disposition": disp,
                        "outcome": outcome, "correct": correct})
        mark = "✓" if outcome == "pass" else ("~" if outcome == "partial" else "✗")
        gt_str   = ", ".join(q["ground_truth"]) or "—"
        det_str  = ", ".join(detected) or "∅"
        print(f"    {mark} [{outcome:7}] {q['message'][:55]:<55} gt={gt_str}")
        if outcome != "pass":
            print(f"              got={det_str}  [{disp}]")
    return results

def print_summary(label, all_results):
    total   = sum(1 for r in all_results if r["outcome"] != "skip")
    passed  = sum(1 for r in all_results if r["outcome"] == "pass")
    partial = sum(1 for r in all_results if r["outcome"] == "partial")
    missed  = sum(1 for r in all_results if r["outcome"] == "miss")
    pct = 100 * passed / total if total else 0
    print(f"\n  {label}: {passed}/{total} pass ({pct:.0f}%)  partial={partial}  miss={missed}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns",       default="baseline-test")
    ap.add_argument("--queries",  type=int, default=5)
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:3001")
    args = ap.parse_args()

    global BASE_URL
    BASE_URL = args.base_url
    ns = args.ns

    # Verify server
    _, health = get("/intents", ns)
    if health is None:
        print(f"✗ Server not reachable at {BASE_URL}")
        sys.exit(1)

    all_intents = get_all_intents(ns)
    print(f"Namespace: {ns}  |  intents: {len(all_intents)}  |  queries/scenario: {args.queries}")
    print(f"Scenarios: {len(SCENARIOS)}")

    # ── BEFORE ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("BEFORE AUTO-LEARN")
    print("="*60)

    all_results_before = []
    query_sets = {}  # save queries to re-route after learning
    for scenario in SCENARIOS:
        results = run_scenario(scenario, ns, args.queries, all_intents)
        all_results_before.extend(results)
        query_sets[scenario["name"]] = results

    print_summary("BEFORE", all_results_before)

    if args.no_learn:
        return

    # ── AUTO-LEARN ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("AUTO-LEARN CYCLE")
    print("="*60)

    # Set auto mode
    post("/review/mode", {"mode": "auto"}, ns)
    print("  Mode → auto")

    # Re-route failing queries to generate flags for the worker
    failing = [r for r in all_results_before if not r["correct"]]
    print(f"  Re-routing {len(failing)} failing queries to generate flags...")
    for r in failing:
        route_query(r["message"], ns)

    _, stats = get("/review/stats", ns)
    print(f"  Flagged: {stats.get('pending', '?') if stats else '?'} entries queued")

    print("  Waiting for worker...")
    wait_for_worker(ns, timeout=300)

    # Reset to manual
    post("/review/mode", {"mode": "manual"}, ns)
    print("  Mode → manual")

    # ── AFTER ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("AFTER AUTO-LEARN")
    print("="*60)

    all_results_after = []
    for scenario in SCENARIOS:
        saved = query_sets[scenario["name"]]
        print(f"\n  [{scenario['name']}]")
        scenario_after = []
        for q in saved:
            detected, disp = route_query(q["message"], ns)
            outcome, correct = score(detected, q["ground_truth"])
            scenario_after.append({**q, "detected": detected, "disposition": disp,
                                    "outcome": outcome, "correct": correct})
            mark = "✓" if outcome == "pass" else ("~" if outcome == "partial" else "✗")
            was = "←was✗" if not q["correct"] and outcome == "pass" else ""
            print(f"    {mark} {q['message'][:55]:<55} {was}")
        all_results_after.extend(scenario_after)

    print_summary("AFTER ", all_results_after)

    # ── DELTA ─────────────────────────────────────────────────────────────────
    before_pass = sum(1 for r in all_results_before if r["outcome"] == "pass")
    after_pass  = sum(1 for r in all_results_after  if r["outcome"] == "pass")
    total       = sum(1 for r in all_results_before if r["outcome"] != "skip")
    recovered   = sum(
        1 for b, a in zip(all_results_before, all_results_after)
        if not b["correct"] and a["outcome"] == "pass"
    )
    print(f"\n  Improvement: {before_pass} → {after_pass} / {total}  (+{after_pass - before_pass})")
    print(f"  Recovered:   {recovered} queries fixed by auto-learn")

if __name__ == "__main__":
    main()
