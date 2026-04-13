"""
Session simulation: replay customer sessions through ASV server API.
Uses tests/data/simulation_sessions.json — real-feeling customer conversations.

Usage: python simulate_sessions.py [--reset] [--sessions path/to/sessions.json]
"""

import json
import time
import argparse
import requests

BASE = "http://localhost:3001/api"
APP_ID = "ecommerce-demo"
HEADERS = {"X-Namespace-ID": APP_ID, "Content-Type": "application/json"}

DEFAULT_INTENTS = {
    "cancel_order": ["cancel my order", "I want to cancel", "stop my order", "cancel purchase", "cancel it"],
    "track_order": ["where is my package", "track my order", "shipping status", "when will it arrive", "delivery update", "order tracking"],
    "refund": ["I want a refund", "get my money back", "refund my purchase", "full refund", "money back", "refund please"],
    "return_item": ["return this item", "send it back", "return my order", "I want to return", "how do I return this"],
    "change_order": ["change my order", "modify order", "update my order", "change shipping address", "change the size"],
    "billing_issue": ["charged twice", "wrong charge", "billing problem", "overcharged", "unauthorized charge", "double charged"],
    "product_inquiry": ["tell me about this product", "is this in stock", "product details", "do you have this", "what colors available"],
    "account_issue": ["can't log in", "account locked", "forgot password", "reset password", "can't access account"],
    "shipping_complaint": ["package late", "delivery delayed", "wrong address delivery", "damaged in shipping", "never received", "shipping too slow"],
    "contact_human": ["talk to a person", "speak to agent", "real person please", "transfer me", "speak to manager", "not a bot"],
    "order_status": ["order status", "what's happening with my order", "order update", "is my order ready", "order confirmation"],
    "payment_method": ["change payment method", "update credit card", "add new card", "do you accept paypal", "payment options"],
    "feedback": ["great service", "terrible experience", "suggestion", "complaint", "you guys need to improve", "amazing help"],
    "subscription": ["cancel subscription", "upgrade plan", "subscription status", "annual plan", "pause subscription"],
}


def setup_intents(reset: bool = False):
    """Ensure intents exist on the server."""
    if reset:
        print("  Resetting server...")
        requests.post(f"{BASE}/reset", headers=HEADERS)

    resp = requests.get(f"{BASE}/intents", headers=HEADERS)
    existing = {i["id"] for i in resp.json()}

    created = 0
    for intent_id, seeds in DEFAULT_INTENTS.items():
        if intent_id not in existing:
            requests.post(f"{BASE}/intents", headers=HEADERS, json={"id": intent_id, "seeds": seeds})
            created += 1

    if created:
        print(f"  Created {created} intents")
    resp = requests.get(f"{BASE}/intents", headers=HEADERS)
    print(f"  Server has {len(resp.json())} intents")


def run_sessions(sessions: list, verbose: bool = True):
    """Replay sessions through the server API."""
    total_turns = 0
    total_correct = 0
    total_partial = 0
    t0 = time.time()

    for session in sessions:
        sid = session["session_id"]
        if verbose:
            print(f"\n  --- {sid}: {session['description']} ---")

        for turn in session["turns"]:
            msg = turn["message"]
            expected = set(turn["intents"])
            emotion = turn.get("emotion", "")

            # Route through server API
            resp = requests.post(f"{BASE}/route_multi", headers=HEADERS, json={"query": msg, "threshold": 0.3})
            result = resp.json()

            detected_confirmed = {r["id"] for r in result.get("confirmed", [])}
            detected_candidates = {r["id"] for r in result.get("candidates", [])}
            detected_all = detected_confirmed | detected_candidates

            matched = expected & detected_all
            missed = expected - detected_all

            total_turns += 1
            if matched == expected:
                total_correct += 1
                status = "PASS"
            elif matched:
                total_partial += 1
                status = "PARTIAL"
            else:
                status = "MISS"

            # Report failures to review queue
            if missed:
                flag = "miss" if not detected_all else "low_confidence"
                requests.post(f"{BASE}/report", headers=HEADERS, json={
                    "query": msg,
                    "detected": list(detected_all),
                    "flag": flag,
                    "session_id": session["session_id"],
                })

            if verbose:
                detected_str = ", ".join(sorted(detected_all)) if detected_all else "(none)"
                expected_str = ", ".join(sorted(expected))
                marker = {"PASS": "+", "PARTIAL": "~", "MISS": "x"}[status]
                print(f"    [{marker}] [{emotion:12s}] \"{msg[:60]}...\"")
                print(f"         expected: [{expected_str}]")
                print(f"         detected: [{detected_str}]")
                if missed:
                    print(f"         MISSED:   [{', '.join(sorted(missed))}]")

    elapsed = time.time() - t0
    total = len([t for s in sessions for t in s["turns"]])

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Sessions: {len(sessions)}")
    print(f"  Turns:    {total}")
    print(f"  Time:     {elapsed:.1f}s")
    print(f"  Exact:    {total_correct}/{total} ({total_correct/max(1,total)*100:.0f}%)")
    print(f"  Partial:  {total_partial}/{total} ({total_partial/max(1,total)*100:.0f}%)")
    print(f"  Miss:     {total - total_correct - total_partial}/{total} ({(total - total_correct - total_partial)/max(1,total)*100:.0f}%)")
    print(f"  Combined: {total_correct + total_partial}/{total} ({(total_correct + total_partial)/max(1,total)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", default="tests/data/simulation_sessions.json")
    parser.add_argument("--reset", action="store_true", help="Reset server before simulation")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    print("Loading sessions...")
    with open(args.sessions) as f:
        sessions = json.load(f)
    print(f"  {len(sessions)} sessions, {sum(len(s['turns']) for s in sessions)} turns")

    print("\nSetting up intents...")
    setup_intents(reset=args.reset)

    print("\nRunning simulation through server API...")
    run_sessions(sessions, verbose=not args.quiet)

    print(f"\nDashboard: http://localhost:5174/dashboard")


if __name__ == "__main__":
    main()
