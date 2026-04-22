#!/usr/bin/env python3
"""
Seed a 'layers-demo' workspace with intents crafted to light up every Layers card.

After running, open the Layers page, switch to 'layers-demo' workspace, and try:

  L0 (typo):     "pls cancle my subscripton"
  L1 (synonyms): "terminate my account"
  L2 multi:      "cancel my subscription and refund my last order"
  L3 suppress:   (requires a correction — use Review page after misroute)
  Relations:     any multi-intent query above

Run:  python3 scripts/seed_layers_demo.py
"""
import json, urllib.request, urllib.error, sys

BASE = "http://localhost:3001/api"
NS = "layers-demo"

INTENTS = {
    "cancel_subscription": {
        "label": "Cancel Subscription",
        "phrases": [
            "cancel my subscription",
            "stop my plan",
            "end my membership",
            "terminate my account",
            "I want to cancel",
            "close my subscription",
        ],
    },
    "refund_order": {
        "label": "Refund Order",
        "phrases": [
            "refund my last order",
            "give me my money back",
            "I want a refund",
            "issue a refund for order",
            "return my payment",
            "reimburse me",
        ],
    },
    "update_address": {
        "label": "Update Address",
        "phrases": [
            "change my shipping address",
            "update my address",
            "I moved, new address",
            "edit my delivery address",
            "set a new shipping location",
        ],
    },
    "upgrade_plan": {
        "label": "Upgrade Plan",
        "phrases": [
            "upgrade my plan",
            "switch to premium",
            "go to pro tier",
            "upgrade to business",
            "move me to the higher plan",
        ],
    },
    "reset_password": {
        "label": "Reset Password",
        "phrases": [
            "reset my password",
            "I forgot my password",
            "can't login, need password reset",
            "change password",
            "password recovery",
        ],
    },
    "contact_support": {
        "label": "Contact Support",
        "phrases": [
            "talk to a human",
            "contact support",
            "I need help",
            "speak to an agent",
            "get support",
        ],
    },
}


def req(method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Workspace-ID": NS,
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            txt = resp.read().decode()
            return json.loads(txt) if txt.strip().startswith(("{", "[")) else txt
    except urllib.error.HTTPError as e:
        print(f"  ! {method} {path} → {e.code}: {e.read().decode()[:200]}", file=sys.stderr)
        return None


def main():
    # 1. Create workspace
    print(f"→ Creating workspace '{NS}'…")
    req("POST", "/workspaces", {"workspace_id": NS, "description": "Layers page demo (L0/L1/L2/L3)"})

    # 2. Add intents and phrases
    for intent_id, info in INTENTS.items():
        print(f"→ {intent_id} ({len(info['phrases'])} phrases)")
        req("POST", "/intents", {"id": intent_id, "label": info["label"], "metadata": {}})
        for p in info["phrases"]:
            req("POST", "/intents/phrase", {"intent_id": intent_id, "phrase": p, "lang": "en"})

    # 3. Quick sanity probe
    print("\n→ Probing 'cancel my subscription and refund my last order'…")
    probe = req("POST", "/layers/l2/probe", {"query": "cancel my subscription and refund my last order"})
    if probe and probe.get("multi"):
        rounds = probe["multi"]["rounds"]
        print(f"  {len(rounds)} round(s), stop reason: {probe['multi']['stop_reason']}")
        for i, r in enumerate(rounds, 1):
            print(f"  Round {i}: tokens={len(r['tokens_in'])} confirmed={r['confirmed']} consumed={r['consumed']}")

    print(f"\n✓ Done. Switch UI workspace to '{NS}' and open /layers.")


if __name__ == "__main__":
    main()
