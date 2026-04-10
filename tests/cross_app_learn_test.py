#!/usr/bin/env python3
"""
Cross-app multi-intent test with auto-learn loop.

Round 1: Route all 20 queries, record what fires.
Learn:   For each query, call POST /api/learn for ALL expected intents
         in their respective apps — regardless of whether they were detected.
Round 2: Route the same queries again. Measure improvement.

This shows two things:
1. Baseline: what the system knows from seeds alone.
2. After learning: how much a single feedback pass improves coverage.

Usage:
    python3 tests/cross_app_learn_test.py [--base-url http://localhost:3001]
    python3 tests/cross_app_learn_test.py --setup-only  # just register intents
"""

import json
import subprocess
import sys
import argparse

BASE_URL = "http://localhost:3001"

# ─── HTTP ────────────────────────────────────────────────────────────────────

def curl(method, path, body=None, app_id=None):
    cmd = ["curl", "-s", "-w", "\n%{http_code}", "-X", method, f"{BASE_URL}{path}",
           "-H", "Content-Type: application/json"]
    if app_id:
        cmd += ["-H", f"X-App-ID: {app_id}"]
    if body is not None:
        cmd += ["-d", json.dumps(body)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = r.stdout.rsplit("\n", 1)
    body_str = lines[0].strip() if len(lines) > 1 else r.stdout.strip()
    status = int(lines[-1].strip()) if len(lines) > 1 and lines[-1].strip().isdigit() else 0
    try:
        return status, json.loads(body_str) if body_str else None
    except json.JSONDecodeError:
        return status, {"_raw": body_str}

def route_app(app_id, query):
    _, data = curl("POST", "/api/route_multi", {"query": query, "threshold": 0.25}, app_id)
    if not data:
        return []
    items = data.get("confirmed", []) + data.get("candidates", [])
    items.sort(key=lambda x: x.get("score", 0), reverse=True)
    return [{"id": i["id"], "confidence": i.get("confidence", "low"), "score": round(i.get("score", 0), 2)}
            for i in items]

def learn(app_id, query, intent_id):
    status, _ = curl("POST", "/api/learn", {"query": query, "intent_id": intent_id}, app_id)
    return status in (200, 201, 204)

def create_app(app_id):
    status, _ = curl("POST", "/api/apps", {"app_id": app_id})
    return status in (200, 201, 409)

def add_intent(app_id, intent_id, label, seeds):
    status, _ = curl("POST", "/api/intents", {"id": intent_id, "label": label, "seeds": seeds}, app_id)
    return status in (200, 201)

def delete_intent(app_id, intent_id):
    curl("POST", "/api/intents/delete", {"intent_id": intent_id}, app_id)

# ─── App definitions — minimal seeds ─────────────────────────────────────────
# Same as main test suite. These are NOT tuned to match test queries.

APPS = {
    "stripe": [
        ("charge_card",         "Charge Card",         ["charge my card", "process payment", "bill the customer", "run a charge"]),
        ("refund_payment",      "Refund Payment",      ["issue a refund", "refund this payment", "give money back", "reverse the charge"]),
        ("create_subscription", "Create Subscription", ["create a subscription", "set up recurring billing", "start a monthly plan"]),
        ("cancel_subscription", "Cancel Subscription", ["cancel the subscription", "stop recurring billing", "end the plan"]),
        ("create_customer",     "Create Customer",     ["create a customer", "add new customer", "register customer"]),
        ("create_invoice",      "Create Invoice",      ["create an invoice", "generate invoice", "send invoice"]),
        ("pay_invoice",         "Pay Invoice",         ["pay this invoice", "mark invoice paid", "settle invoice"]),
        ("retrieve_balance",    "Retrieve Balance",    ["check balance", "get account balance", "show available funds"]),
        ("create_payout",       "Create Payout",       ["create payout", "withdraw to bank", "initiate payout"]),
        ("list_transactions",   "List Transactions",   ["list transactions", "show payment history", "recent charges"]),
    ],
    "github": [
        ("create_issue",     "Create Issue",     ["create an issue", "file a bug report", "open new issue", "report a bug"]),
        ("close_issue",      "Close Issue",      ["close this issue", "resolve the issue", "mark issue done"]),
        ("create_pr",        "Create PR",        ["open a pull request", "create PR", "submit for review"]),
        ("merge_pr",         "Merge PR",         ["merge the PR", "merge pull request", "land this PR"]),
        ("create_repo",      "Create Repo",      ["create a repository", "new repo", "initialize repository"]),
        ("create_release",   "Create Release",   ["create a release", "publish new version", "tag a release"]),
        ("list_commits",     "List Commits",     ["list commits", "show commit history", "view git log"]),
        ("add_collaborator", "Add Collaborator", ["add a collaborator", "invite contributor", "give access to repo"]),
        ("create_webhook",   "Create Webhook",   ["create a webhook", "set up webhook", "add webhook"]),
        ("search_code",      "Search Code",      ["search code", "find in codebase", "grep across repo"]),
    ],
    "slack": [
        ("send_message",     "Send Message",      ["send a message", "post to channel", "message the team"]),
        ("create_channel",   "Create Channel",    ["create a channel", "new slack channel", "set up channel"]),
        ("invite_user",      "Invite User",       ["invite someone to channel", "add user to channel", "add member"]),
        ("set_reminder",     "Set Reminder",      ["set a reminder", "remind me", "create reminder"]),
        ("create_poll",      "Create Poll",       ["create a poll", "start a vote", "poll the team"]),
        ("schedule_message", "Schedule Message",  ["schedule a message", "send message later", "post later"]),
        ("upload_file",      "Upload File",       ["upload a file", "share document", "attach file"]),
        ("pin_message",      "Pin Message",       ["pin this message", "pin to channel"]),
        ("search_messages",  "Search Messages",   ["search messages", "find a message", "look up conversation"]),
        ("set_topic",        "Set Topic",         ["set channel topic", "update topic", "change channel description"]),
    ],
    "shopify": [
        ("create_product",   "Create Product",   ["create a product", "add product to store", "new product listing"]),
        ("cancel_order",     "Cancel Order",     ["cancel order", "void the order", "cancel purchase"]),
        ("refund_order",     "Refund Order",     ["refund this order", "process order refund", "give refund"]),
        ("list_orders",      "List Orders",      ["list orders", "show recent orders", "view order history"]),
        ("update_inventory", "Update Inventory", ["update inventory", "change stock level", "adjust quantity"]),
        ("ship_order",       "Ship Order",       ["ship this order", "mark as shipped", "fulfill order"]),
        ("track_shipment",   "Track Shipment",   ["track shipment", "where is my order", "track delivery"]),
        ("generate_report",  "Generate Report",  ["generate sales report", "store analytics", "revenue report"]),
        ("process_return",   "Process Return",   ["process a return", "handle return", "accept returned item"]),
        ("create_discount",  "Create Discount",  ["create a discount", "add discount", "set up sale"]),
    ],
    "calendar": [
        ("create_event",           "Create Event",           ["create an event", "add to calendar", "schedule a meeting"]),
        ("cancel_event",           "Cancel Event",           ["cancel the event", "delete meeting", "remove from calendar"]),
        ("invite_attendee",        "Invite Attendee",        ["invite someone to meeting", "add to event", "send calendar invite"]),
        ("check_availability",     "Check Availability",     ["check availability", "am I free", "check if I have time"]),
        ("find_meeting_time",      "Find Meeting Time",      ["find a meeting time", "when can we meet", "find common availability"]),
        ("create_recurring_event", "Create Recurring Event", ["create recurring event", "set up weekly meeting", "repeat this event"]),
        ("set_out_of_office",      "Set Out of Office",      ["set out of office", "mark unavailable", "block vacation"]),
        ("book_room",              "Book Room",              ["book a room", "reserve meeting room", "find a room"]),
        ("reschedule_event",       "Reschedule Event",       ["reschedule this meeting", "move the event", "change event time"]),
        ("set_reminder",           "Set Reminder",           ["set event reminder", "remind me before meeting"]),
    ],
}

# ─── 20 Cross-App Multi-Intent Test Cases ────────────────────────────────────
# Each query is a natural human request implying actions in 2 different apps.
# Vocabulary is deliberately different from seeds.
# expected: list of (app_id, intent_id)

CROSS_APP_TESTS = [
    (
        "the auth bug is live in prod, track it and alert the on-call channel",
        [("github", "create_issue"), ("slack", "send_message")],
        "prod bug → github issue + slack alert",
    ),
    (
        "charge the client for this sprint and drop the receipt in the billing channel",
        [("stripe", "charge_card"), ("slack", "send_message")],
        "billing → stripe charge + slack notify",
    ),
    (
        "we're shipping 2.0 today — tag it and blast the announcement to the team",
        [("github", "create_release"), ("slack", "send_message")],
        "release day → github tag + slack announce",
    ),
    (
        "the package went out this morning, let the customer know on slack",
        [("shopify", "ship_order"), ("slack", "send_message")],
        "order shipped → shopify + slack notify",
    ),
    (
        "block my calendar next week and file it in the engineering channel so no one books me",
        [("calendar", "set_out_of_office"), ("slack", "send_message")],
        "vacation → calendar block + slack heads up",
    ),
    (
        "new enterprise client signed — get them in the system and send them a welcome over slack",
        [("stripe", "create_customer"), ("slack", "send_message")],
        "new client → stripe customer + slack welcome",
    ),
    (
        "hotfix is green, land it and ping the team it's live",
        [("github", "merge_pr"), ("slack", "send_message")],
        "hotfix → github merge + slack ping",
    ),
    (
        "pull the Q3 numbers and share the deck in the finance channel",
        [("shopify", "generate_report"), ("slack", "upload_file")],
        "reporting → shopify report + slack share",
    ),
    (
        "get the sprint planning on the books and kick off the project repo",
        [("calendar", "create_event"), ("github", "create_repo")],
        "sprint kickoff → calendar + github repo",
    ),
    (
        "customer wants out and their money back — cancel the order and process the refund",
        [("shopify", "cancel_order"), ("stripe", "refund_payment")],
        "churn → shopify cancel + stripe refund",
    ),
    (
        "bill them for this month and set up a check-in call for next week",
        [("stripe", "create_invoice"), ("calendar", "create_event")],
        "billing cycle → stripe invoice + calendar call",
    ),
    (
        "the design review is tomorrow — get it on the calendar and post the figma link in slack",
        [("calendar", "create_event"), ("slack", "send_message")],
        "design review → calendar + slack link",
    ),
    (
        "the bug's been closed out, retire the hotfix branch and update the team",
        [("github", "close_issue"), ("slack", "send_message")],
        "bug resolved → github close + slack update",
    ),
    (
        "set up the new product in the store and hook up a webhook so our system gets notified",
        [("shopify", "create_product"), ("github", "create_webhook")],
        "product launch → shopify + github webhook",
    ),
    (
        "find a slot everyone's free and spin up the kickoff repo",
        [("calendar", "find_meeting_time"), ("github", "create_repo")],
        "project start → calendar slot + github repo",
    ),
    (
        "the customer returned the item — process it and issue their money back",
        [("shopify", "process_return"), ("stripe", "refund_payment")],
        "return flow → shopify return + stripe refund",
    ),
    (
        "drop the PR in the engineering channel and ask for eyes",
        [("github", "create_pr"), ("slack", "send_message")],
        "code review → github PR + slack request",
    ),
    (
        "onboard the new dev — set them up in the repo and loop them into the eng channel",
        [("github", "add_collaborator"), ("slack", "invite_user")],
        "dev onboarding → github collaborator + slack invite",
    ),
    (
        "create a weekly standup for the team and pin the agenda in slack",
        [("calendar", "create_recurring_event"), ("slack", "pin_message")],
        "recurring standup → calendar + slack pin",
    ),
    (
        "the inventory is running low — restock it and open a tracking issue",
        [("shopify", "update_inventory"), ("github", "create_issue")],
        "inventory alert → shopify stock + github issue",
    ),
]

# ─── Setup ────────────────────────────────────────────────────────────────────

def setup_apps():
    print("\n=== Setting up apps ===")
    for app_id, intents in APPS.items():
        print(f"  [{app_id}]", end=" ")
        create_app(app_id)
        existing_status, existing = curl("GET", "/api/intents", app_id=app_id)
        if isinstance(existing, list):
            for i in existing:
                delete_intent(app_id, i["id"])
        ok = sum(1 for intent_id, label, seeds in intents if add_intent(app_id, intent_id, label, seeds))
        print(f"{ok}/{len(intents)} intents registered")
    print()

# ─── Test runner ──────────────────────────────────────────────────────────────

def evaluate(label: str) -> list[dict]:
    """Route all 20 queries, return per-test result dicts."""
    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"{'='*64}")

    results = []
    for query, expected, desc in CROSS_APP_TESTS:
        per_app = {}
        for app_id in set(a for a, _ in expected):
            per_app[app_id] = route_app(app_id, query)

        found = []
        missed = []
        for app_id, intent_id in expected:
            ids = [i["id"] for i in per_app.get(app_id, [])]
            if intent_id in ids[:4]:
                found.append((app_id, intent_id))
            else:
                missed.append((app_id, intent_id))

        passed = len(found) == len(expected)
        partial = len(found) > 0 and not passed
        status = "PASS" if passed else ("PART" if partial else "FAIL")

        print(f"\n  [{status}] {desc}")
        print(f"    query: {query}")
        for app_id, intent_id in expected:
            top = [f"{i['id']}({i['confidence'][0].upper()}:{i['score']})"
                   for i in per_app.get(app_id, [])[:3]]
            marker = "✓" if (app_id, intent_id) in found else "✗"
            print(f"    {marker} {app_id}.{intent_id}  →  top: {top if top else '(nothing)'}")

        results.append({
            "query": query,
            "expected": expected,
            "found": found,
            "missed": missed,
            "passed": passed,
            "partial": partial,
        })

    passed = sum(1 for r in results if r["passed"])
    partial = sum(1 for r in results if r["partial"])
    failed = sum(1 for r in results if not r["passed"] and not r["partial"])
    total = len(results)

    # Count intent-level hits too
    intent_hits = sum(len(r["found"]) for r in results)
    intent_total = sum(len(r["expected"]) for r in results)

    print(f"\n  --- {label} summary ---")
    print(f"  Queries:  PASS={passed}  PARTIAL={partial}  FAIL={failed}  / {total}")
    print(f"  Intents:  {intent_hits}/{intent_total} individual intents detected")
    return results

def do_learn(results: list[dict]):
    """Call POST /api/learn for every expected intent of every query, regardless of hit/miss."""
    print(f"\n{'='*64}")
    print("  LEARNING PHASE — teaching correct answers for all 20 queries")
    print(f"{'='*64}")
    learn_count = 0
    for r in results:
        query = r["query"]
        for app_id, intent_id in r["expected"]:
            ok = learn(app_id, query, intent_id)
            status = "✓" if ok else "✗"
            print(f"  {status} learn: {app_id}.{intent_id}")
            if ok:
                learn_count += 1
    print(f"\n  Learned {learn_count} (query, intent) associations")

def diff_summary(r1: list[dict], r2: list[dict]):
    print(f"\n{'='*64}")
    print("  BEFORE vs AFTER comparison")
    print(f"{'='*64}")

    improved = []
    regressed = []
    unchanged_pass = []
    unchanged_fail = []

    for a, b, (_, _, desc) in zip(r1, r2, CROSS_APP_TESTS):
        hits_before = len(a["found"])
        hits_after  = len(b["found"])
        if hits_after > hits_before:
            improved.append((desc, hits_before, hits_after, len(a["expected"])))
        elif hits_after < hits_before:
            regressed.append((desc, hits_before, hits_after, len(a["expected"])))
        elif b["passed"]:
            unchanged_pass.append(desc)
        else:
            unchanged_fail.append(desc)

    if improved:
        print(f"\n  IMPROVED ({len(improved)}):")
        for desc, before, after, total in improved:
            print(f"    {desc}: {before}/{total} → {after}/{total}")
    if regressed:
        print(f"\n  REGRESSED ({len(regressed)}):")
        for desc, before, after, total in regressed:
            print(f"    {desc}: {before}/{total} → {after}/{total}")
    if unchanged_pass:
        print(f"\n  ALREADY PASSING ({len(unchanged_pass)}):")
        for desc in unchanged_pass:
            print(f"    {desc}")
    if unchanged_fail:
        print(f"\n  STILL FAILING ({len(unchanged_fail)}):")
        for desc in unchanged_fail:
            print(f"    {desc}")

    i1 = sum(len(r["found"]) for r in r1)
    i2 = sum(len(r["found"]) for r in r2)
    t  = sum(len(r["expected"]) for r in r1)
    p1 = sum(1 for r in r1 if r["passed"])
    p2 = sum(1 for r in r2 if r["passed"])

    print(f"\n  Queries passed:  {p1}/20 → {p2}/20  (+{p2-p1})")
    print(f"  Intent hits:     {i1}/{t} → {i2}/{t}  (+{i2-i1})")

# ─── Hard tier: 10 genuinely difficult queries ────────────────────────────────
#
# These are NOT scored pass/fail. They are observation cases designed to show
# exactly where and how the system breaks. Categories:
#
#   IMPLICIT  — no action verb; situation described, action must be inferred
#   RAMBLING  — long, filler-heavy; intent buried mid-sentence
#   NEGATION  — one intent should fire, another explicitly should NOT
#   JARGON    — domain shorthand that doesn't appear in seeds
#   AMBIGUOUS — query matches multiple intents with no clear winner
#   TYPO      — realistic misspellings and sloppy typing
#   SHORT     — one or two words, no context
#   PRONOUN   — "it", "them", "that" with no referent
#
# expected_fires: what a production system SHOULD detect
# expected_suppressed: intents that should NOT appear (false positives)

HARD_QUERIES = [
    {
        "category": "IMPLICIT",
        "query": "the payment bounced on the acme account",
        "note": "No action verb. Situation → implies retry charge or create issue, but system has no failure→action mapping.",
        "expected_fires":      [("stripe", "charge_card")],
        "expected_suppressed": [],
    },
    {
        "category": "RAMBLING",
        "query": "hey so we had a situation with that enterprise client yesterday, turns out the payment didn't actually go through and now they're pinging us about it, we probably need to sort that out and also loop in the finance team",
        "note": "50 words, two intents buried in filler. Tests whether term density still wins over noise.",
        "expected_fires":      [("stripe", "charge_card"), ("slack", "send_message")],
        "expected_suppressed": [],
    },
    {
        "category": "NEGATION",
        "query": "process the card but hold off on the invoice for now",
        "note": "charge_card should fire. create_invoice should be SUPPRESSED by negation ('hold off on').",
        "expected_fires":      [("stripe", "charge_card")],
        "expected_suppressed": [("stripe", "create_invoice")],
    },
    {
        "category": "NEGATION",
        "query": "ship it but don't bother pinging the customer",
        "note": "ship_order fires. send_message should NOT fire (explicitly suppressed).",
        "expected_fires":      [("shopify", "ship_order")],
        "expected_suppressed": [("slack", "send_message")],
    },
    {
        "category": "IMPLICIT",
        "query": "the customer churned last night",
        "note": "Past tense, situation only. Implies cancel_subscription. No action word at all.",
        "expected_fires":      [("stripe", "cancel_subscription")],
        "expected_suppressed": [],
    },
    {
        "category": "JARGON",
        "query": "cut the release and push to the eng channel",
        "note": "'Cut' = create_release. 'Push to channel' = send_message. Neither word is in seeds.",
        "expected_fires":      [("github", "create_release"), ("slack", "send_message")],
        "expected_suppressed": [],
    },
    {
        "category": "SHORT",
        "query": "ship it",
        "note": "Two words. Could mean shopify:ship_order OR github:create_release. Ambiguous by design.",
        "expected_fires":      [("shopify", "ship_order")],
        "expected_suppressed": [],
    },
    {
        "category": "AMBIGUOUS",
        "query": "cancel it and let them know",
        "note": "'It' is unresolved. 'Cancel' exists in stripe, shopify, calendar. Should the system pick one confidently or stay low?",
        "expected_fires":      [],   # no clear winner — correct answer is low confidence
        "expected_suppressed": [],
    },
    {
        "category": "TYPO",
        "query": "creat an isue for the auth bug nd assign it to the eng team",
        "note": "Typos in the key action words. Tests tokenizer robustness.",
        "expected_fires":      [("github", "create_issue")],
        "expected_suppressed": [],
    },
    {
        "category": "PRONOUN",
        "query": "do the usual for the new client",
        "note": "'The usual' = create_customer + create_subscription. Pronoun + implicit routine. Pure context recall.",
        "expected_fires":      [("stripe", "create_customer"), ("stripe", "create_subscription")],
        "expected_suppressed": [],
    },
]

def run_hard_tier():
    """Observe system behavior on genuinely difficult queries. No pass/fail scoring."""
    print(f"\n{'='*64}")
    print("  HARD TIER — Observation Only (no pass/fail)")
    print("  These expose real system limits. Failures are expected.")
    print(f"{'='*64}")

    all_apps = list(APPS.keys())

    for case in HARD_QUERIES:
        category = case["category"]
        query    = case["query"]
        note     = case["note"]
        expected_fires      = case["expected_fires"]
        expected_suppressed = case["expected_suppressed"]

        # Route against ALL apps so we can see false positives too
        all_results = {app_id: route_app(app_id, query) for app_id in all_apps}

        # What actually fired (high or medium confidence, any app)
        actually_fired = []
        for app_id, results in all_results.items():
            for r in results:
                if r["confidence"] in ("high", "medium"):
                    actually_fired.append((app_id, r["id"], r["confidence"], r["score"]))

        # Check expected fires
        hit = [(a, i) for a, i in expected_fires
               if any(r[0] == a and r[1] == i for r in actually_fired)]
        missed = [(a, i) for a, i in expected_fires if (a, i) not in hit]

        # Check false positives (suppressed intents that actually fired)
        false_positives = [(a, i) for a, i, c, s in actually_fired
                           if (a, i) in expected_suppressed]

        # Unexpected fires (fired but not in expected list — may or may not be wrong)
        unexpected = [(a, i, c, s) for a, i, c, s in actually_fired
                      if (a, i) not in expected_fires and (a, i) not in expected_suppressed]

        print(f"\n  [{category}] {query[:80]}")
        print(f"    note: {note}")

        if hit:
            print(f"    DETECTED (expected):   {[(a+'.'+i) for a,i in hit]}")
        if missed:
            print(f"    MISSED   (expected):   {[(a+'.'+i) for a,i in missed]}")
        if false_positives:
            print(f"    FALSE POSITIVE (fired but should NOT): {[(a+'.'+i) for a,i in false_positives]}")
        if unexpected:
            top_unexpected = [(f"{a}.{i}({c[0].upper()}:{s})") for a,i,c,s in unexpected[:4]]
            print(f"    UNEXPECTED fires:      {top_unexpected}")
        if not actually_fired:
            print(f"    NOTHING fired at high/medium confidence")
            if expected_fires:
                print(f"    (expected {[(a+'.'+i) for a,i in expected_fires]} — system is blind to this phrasing)")
            else:
                print(f"    (correct — low confidence is the right answer for this query)")

    print(f"\n  Hard tier complete. Review failures above to prioritize improvements.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global BASE_URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:3001")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--only", choices=["standard", "hard", "all"], default="all")
    args = parser.parse_args()
    BASE_URL = args.base_url

    print("ASV Cross-App Multi-Intent Test with Auto-Learn")
    print(f"Target: {BASE_URL}  |  20 queries × 2 apps each = 40 intent targets")
    print("Baseline: seeds only (not tuned to test queries)")

    status, _ = curl("GET", "/api/apps")
    if status == 0:
        print(f"\nERROR: Server not reachable at {BASE_URL}")
        sys.exit(1)

    if not args.skip_setup:
        setup_apps()

    if args.setup_only:
        print("Setup complete.")
        return

    if args.only in ("standard", "all"):
        # Round 1 — seeds only
        round1 = evaluate("ROUND 1 — Seeds Only (baseline)")

        # Learn
        do_learn(round1)

        # Round 2 — after one learning pass
        round2 = evaluate("ROUND 2 — After One Learning Pass")

        # Diff
        diff_summary(round1, round2)

    if args.only in ("hard", "all"):
        run_hard_tier()

    if args.only == "standard":
        total_pass_r2 = sum(1 for r in round2 if r["passed"])
        sys.exit(0 if total_pass_r2 == 20 else 1)

if __name__ == "__main__":
    main()
