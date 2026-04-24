"""
ABCD Dataset Simulation: replay real customer conversations through MicroResolve server.

Uses the HTTP API — exactly like a real production client. Every query goes
through the full stack: HTTP → server → router → record co-occurrence → persist.

Usage:
  python simulate_abcd.py --dataset /tmp/abcd_v1.1.json --conversations 500

Requires: server running on localhost:3001 with intents loaded.
Download ABCD: wget -O /tmp/abcd_v1.1.json.gz https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz && gunzip /tmp/abcd_v1.1.json.gz
"""

import json
import time
import argparse
import requests

BASE = "http://localhost:3001/api"


def load_abcd(path: str, split: str = "train", max_convos: int = 500):
    """Load ABCD dataset, extract customer messages per session."""
    with open(path) as f:
        data = json.load(f)

    conversations = data.get(split, [])[:max_convos]
    sessions = []

    for conv in conversations:
        flow = conv["scenario"].get("flow", "unknown")
        subflow = conv["scenario"].get("subflow", "unknown")
        customer_msgs = [
            turn[1] for turn in conv["original"]
            if turn[0] == "customer" and len(turn[1].strip()) > 5
        ]
        if customer_msgs:
            sessions.append({
                "flow": flow,
                "subflow": subflow,
                "label": f"{flow}.{subflow}",
                "messages": customer_msgs,
            })

    return sessions


def ensure_intents(sessions: list[dict]):
    """Create intents from ABCD flows if they don't exist yet."""
    # Check existing intents
    resp = requests.get(f"{BASE}/intents")
    existing = {i["id"] for i in resp.json()}

    # Map ABCD flows to MicroResolve intents with seed phrases
    intent_map = {
        "product_defect": {
            "seeds": ["defective product", "product is broken", "wrong item received",
                       "item is damaged", "product doesn't work", "faulty product"],
            "type": "action"
        },
        "return_item": {
            "seeds": ["I want to return", "return this item", "send it back",
                       "return my purchase", "I need to return", "how do I return"],
            "type": "action"
        },
        "refund": {
            "seeds": ["I want a refund", "get my money back", "refund my purchase",
                       "process a refund", "refund please", "where is my refund"],
            "type": "action"
        },
        "shipping_issue": {
            "seeds": ["shipping problem", "delivery issue", "package not arrived",
                       "where is my order", "shipping status", "tracking number"],
            "type": "action"
        },
        "track_order": {
            "seeds": ["track my order", "where is my package", "order status",
                       "delivery update", "when will it arrive", "shipping update"],
            "type": "action"
        },
        "billing_issue": {
            "seeds": ["charged twice", "wrong amount", "billing problem",
                       "overcharged", "price doesn't match", "dispute charge"],
            "type": "action"
        },
        "purchase_dispute": {
            "seeds": ["dispute my purchase", "price is wrong", "competitor has better price",
                       "I want to dispute", "charge is incorrect", "wrong price"],
            "type": "action"
        },
        "account_access": {
            "seeds": ["can't log in", "locked out of account", "forgot password",
                       "account is locked", "reset password", "can't access my account"],
            "type": "action"
        },
        "password_reset": {
            "seeds": ["reset my password", "forgot my password", "change password",
                       "new password", "password doesn't work", "recover password"],
            "type": "action"
        },
        "manage_account": {
            "seeds": ["change my email", "update phone number", "edit account info",
                       "change my address", "update my profile", "modify account settings"],
            "type": "action"
        },
        "subscription": {
            "seeds": ["cancel subscription", "subscription status", "renew subscription",
                       "subscription billing", "pause subscription", "upgrade subscription"],
            "type": "action"
        },
        "technical_support": {
            "seeds": ["website not working", "page won't load", "error on checkout",
                       "cart not updating", "search not working", "site is slow"],
            "type": "action"
        },
        "product_inquiry": {
            "seeds": ["tell me about this product", "is this in stock", "product information",
                       "what colors available", "product details", "how much does this cost"],
            "type": "context"
        },
        "store_info": {
            "seeds": ["store hours", "return policy", "shipping policy",
                       "what payment methods", "do you price match", "warranty information"],
            "type": "context"
        },
        "order_info": {
            "seeds": ["my order details", "order history", "previous orders",
                       "order summary", "what did I order", "order confirmation"],
            "type": "context"
        },
        "contact_human": {
            "seeds": ["talk to a person", "speak to a manager", "human agent please",
                       "I want to escalate", "this isn't helping", "let me speak to someone"],
            "type": "action"
        },
    }

    created = 0
    for intent_id, config in intent_map.items():
        if intent_id not in existing:
            requests.post(f"{BASE}/intents", json={
                "id": intent_id,
                "seeds": config["seeds"],
                "intent_type": config["type"],
            })
            created += 1

    if created > 0:
        print(f"  Created {created} new intents from ABCD flows")

    return intent_map


def run_simulation(sessions: list[dict], max_sessions: int = 500):
    """Replay customer sessions through MicroResolve server API."""
    total_queries = 0
    total_sessions = 0
    intent_hits = {}
    accuracy_checks = []

    t0 = time.time()

    for i, session in enumerate(sessions[:max_sessions]):
        session_intents = []

        for msg in session["messages"]:
            # Route through server API — exactly like production
            try:
                resp = requests.post(f"{BASE}/route_multi", json={
                    "query": msg,
                    "threshold": 0.3,
                })
                result = resp.json()
                detected = [r["id"] for r in result.get("confirmed", [])]
                candidates = [r["id"] for r in result.get("candidates", [])]
                all_intents = detected + candidates

                for intent_id in all_intents:
                    intent_hits[intent_id] = intent_hits.get(intent_id, 0) + 1

                session_intents.extend(all_intents)
                total_queries += 1

            except Exception as e:
                print(f"  Error routing: {e}")
                continue

        # Check if ABCD flow maps to any detected intent
        flow = session["flow"]
        if session_intents:
            # Simple accuracy: did we detect something related to the flow?
            flow_related = any(
                flow_word in intent_id
                for intent_id in session_intents
                for flow_word in flow.replace("_", " ").split()
            )
            accuracy_checks.append(flow_related)

        total_sessions += 1

        # Progress
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            qps = total_queries / elapsed if elapsed > 0 else 0
            print(f"  Session {i+1}/{min(len(sessions), max_sessions)} | "
                  f"{total_queries} queries | {qps:.0f} q/s | "
                  f"{elapsed:.1f}s")

    elapsed = time.time() - t0

    # Results
    print(f"\n{'='*60}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Sessions: {total_sessions}")
    print(f"  Queries:  {total_queries}")
    print(f"  Time:     {elapsed:.1f}s ({total_queries/elapsed:.0f} queries/sec)")
    print(f"  Accuracy: {sum(accuracy_checks)}/{len(accuracy_checks)} "
          f"({sum(accuracy_checks)/max(1,len(accuracy_checks))*100:.1f}%) flow-related intent detected")

    print(f"\n  Top intents hit:")
    for intent_id, count in sorted(intent_hits.items(), key=lambda x: -x[1])[:15]:
        print(f"    {intent_id:30s} {count:5d}")

    return {
        "sessions": total_sessions,
        "queries": total_queries,
        "time_sec": elapsed,
        "accuracy": sum(accuracy_checks) / max(1, len(accuracy_checks)),
        "intent_hits": intent_hits,
    }


def main():
    parser = argparse.ArgumentParser(description="ABCD simulation through MicroResolve server")
    parser.add_argument("--dataset", default="/tmp/abcd_v1.1.json", help="Path to ABCD JSON")
    parser.add_argument("--conversations", type=int, default=500, help="Max conversations to simulate")
    parser.add_argument("--split", default="train", help="Dataset split: train/dev/test")
    args = parser.parse_args()

    print(f"Loading ABCD dataset from {args.dataset}...")
    sessions = load_abcd(args.dataset, args.split, args.conversations)
    print(f"  {len(sessions)} sessions loaded")
    print(f"  {sum(len(s['messages']) for s in sessions)} total customer messages")

    # Show flow distribution
    from collections import Counter
    flows = Counter(s["flow"] for s in sessions)
    print(f"\n  Flow distribution:")
    for flow, count in flows.most_common(10):
        print(f"    {flow:25s} {count}")

    print(f"\nEnsuring intents exist on server...")
    ensure_intents(sessions)

    # Check server
    resp = requests.get(f"{BASE}/intents")
    print(f"  Server has {len(resp.json())} intents")

    print(f"\nRunning simulation ({args.conversations} conversations)...")
    print(f"  Using server at {BASE}")
    print()

    results = run_simulation(sessions, args.conversations)

    print(f"\nDashboard should now show patterns from {results['sessions']} real sessions.")
    print(f"Open http://localhost:5174/dashboard to view.")


if __name__ == "__main__":
    main()
