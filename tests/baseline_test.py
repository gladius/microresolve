#!/usr/bin/env python3
"""
Baseline test: 4 domains × 5 intents, multi-intent, multilingual, cross-domain.

Tests the Hebbian L1+L2 router end-to-end against a fresh namespace.
After bootstrap this shows exactly what the system can and cannot route.

Usage:
    python3 tests/baseline_test.py
    python3 tests/baseline_test.py --skip-setup    # reuse existing intents/graphs
    python3 tests/baseline_test.py --base-url http://localhost:3001
"""

import json, subprocess, sys, argparse, time, threading

BASE_URL = "http://localhost:3001"
NS = "baseline-test"

# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def curl(method, path, body=None, ns=NS):
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

def post(path, body):  return curl("POST", path, body)
def get(path):         return curl("GET", path)

# ─── Intents: 4 domains × 5 intents ─────────────────────────────────────────
# Each intent has realistic day-1 seed phrases in English + one other language.

INTENTS = {
    # ── billing ──────────────────────────────────────────────────────────────
    "billing:charge_card": {
        "description": "Charge a payment card",
        "en": ["charge my card", "process payment", "run the card", "bill me now", "make a payment"],
        "es": ["cobrar mi tarjeta", "procesar pago"],
    },
    "billing:refund": {
        "description": "Issue a refund",
        "en": ["give me a refund", "I want my money back", "refund my purchase", "process a refund"],
        "es": ["quiero un reembolso", "devolver mi dinero"],
    },
    "billing:cancel_subscription": {
        "description": "Cancel an ongoing subscription",
        "en": ["cancel my subscription", "stop billing me", "end my plan", "unsubscribe", "cancel plan"],
        "es": ["cancelar mi suscripción", "dejar de cobrarme"],
    },
    "billing:update_payment": {
        "description": "Update payment method or billing details",
        "en": ["update my card", "change payment method", "new credit card", "update billing info"],
        "es": ["actualizar mi tarjeta", "cambiar método de pago"],
    },
    "billing:view_invoice": {
        "description": "View or download an invoice",
        "en": ["show my invoice", "download receipt", "get my bill", "view billing history", "past invoices"],
        "es": ["ver mi factura", "descargar recibo"],
    },

    # ── support ───────────────────────────────────────────────────────────────
    "support:create_ticket": {
        "description": "Open a new support ticket",
        "en": ["open a ticket", "create support request", "file a bug report", "report an issue", "submit a ticket"],
        "fr": ["ouvrir un ticket", "signaler un problème"],
    },
    "support:check_status": {
        "description": "Check the status of an existing ticket",
        "en": ["check ticket status", "what is the status of my issue", "any update on my ticket", "ticket update"],
        "fr": ["quel est le statut", "mise à jour du ticket"],
    },
    "support:escalate": {
        "description": "Escalate a ticket to a senior agent or manager",
        "en": ["escalate my issue", "speak to a manager", "this needs urgent attention", "I want to escalate"],
        "fr": ["escalader le problème", "parler à un responsable"],
    },
    "support:close_ticket": {
        "description": "Close or resolve a support ticket",
        "en": ["close my ticket", "mark as resolved", "issue is fixed", "you can close this"],
        "fr": ["fermer le ticket", "marquer comme résolu"],
    },
    "support:feedback": {
        "description": "Submit feedback or rate the support experience",
        "en": ["leave feedback", "rate my experience", "submit a review", "how was the support"],
        "fr": ["laisser un avis", "évaluer le support"],
    },

    # ── shipping ──────────────────────────────────────────────────────────────
    "shipping:track_order": {
        "description": "Track the delivery status of an order",
        "en": ["where is my order", "track my package", "shipping status", "when will it arrive", "delivery update"],
        "zh": ["我的包裹在哪", "追踪订单"],
    },
    "shipping:cancel_order": {
        "description": "Cancel an order before it ships",
        "en": ["cancel my order", "I want to cancel", "stop my order", "don't ship it"],
        "zh": ["取消我的订单", "不要发货"],
    },
    "shipping:return_item": {
        "description": "Return a received item",
        "en": ["return this item", "I want to return", "send it back", "initiate return", "return request"],
        "zh": ["退货", "我想退货"],
    },
    "shipping:change_address": {
        "description": "Change the delivery address for an order",
        "en": ["change delivery address", "update shipping address", "wrong address", "ship to a different address"],
        "zh": ["更改收货地址", "修改地址"],
    },
    "shipping:delivery_problem": {
        "description": "Report a delivery issue — damaged, missing, wrong item",
        "en": ["my package is damaged", "wrong item delivered", "package not arrived", "missing item in order"],
        "zh": ["包裹损坏", "收到错误商品"],
    },

    # ── account ───────────────────────────────────────────────────────────────
    "account:create_account": {
        "description": "Create a new user account",
        "en": ["create an account", "sign up", "register", "new account please", "I want to join"],
        "de": ["Konto erstellen", "anmelden"],
    },
    "account:delete_account": {
        "description": "Permanently delete a user account",
        "en": ["delete my account", "close my account", "remove my data", "I want to leave", "deactivate account"],
        "de": ["Konto löschen", "mein Konto schließen"],
    },
    "account:reset_password": {
        "description": "Reset or recover account password",
        "en": ["reset my password", "forgot password", "can't log in", "password recovery", "I'm locked out"],
        "de": ["Passwort zurücksetzen", "Passwort vergessen"],
    },
    "account:update_profile": {
        "description": "Update profile information like name or email",
        "en": ["update my profile", "change my email", "edit my name", "update account info"],
        "de": ["Profil aktualisieren", "E-Mail ändern"],
    },
    "account:verify_email": {
        "description": "Verify or resend email verification",
        "en": ["verify my email", "resend verification", "email not verified", "confirm my email address"],
        "de": ["E-Mail bestätigen", "Verifizierung erneut senden"],
    },
}

# ─── Test queries ─────────────────────────────────────────────────────────────
# Format: (query, expected_intents, test_type, notes)
# expected_intents: list of intent IDs that should appear (confirmed or candidates)

TESTS = [
    # ── Single intent: direct vocabulary ──────────────────────────────────────
    ("charge my card please",          ["billing:charge_card"],               "single",      "direct seed"),
    ("I need a refund",                ["billing:refund"],                    "single",      "direct seed"),
    ("track my order",                 ["shipping:track_order"],              "single",      "direct seed"),
    ("open a support ticket",          ["support:create_ticket"],             "single",      "direct seed"),
    ("reset my password",              ["account:reset_password"],            "single",      "direct seed"),

    # ── Single intent: paraphrase (different vocabulary from seeds) ────────────
    ("process the payment",            ["billing:charge_card"],               "paraphrase",  "process≈charge"),
    ("where is my package",            ["shipping:track_order"],              "paraphrase",  "package≈order"),
    ("I forgot my password",           ["account:reset_password"],            "paraphrase",  "forgot≈reset"),
    ("I want to close my account",     ["account:delete_account"],            "paraphrase",  "close≈delete"),
    ("file a complaint",               ["support:create_ticket"],             "paraphrase",  "complaint≈ticket"),

    # ── Cross-domain: same verb, different intent ──────────────────────────────
    ("cancel my subscription",         ["billing:cancel_subscription"],       "cross-domain", "not shipping:cancel"),
    ("cancel my order",                ["shipping:cancel_order"],             "cross-domain", "not billing:cancel"),
    ("close my ticket",                ["support:close_ticket"],              "cross-domain", "not account:delete"),
    ("close my account",               ["account:delete_account"],            "cross-domain", "close≠ticket"),

    # ── Multi-intent: compound queries ────────────────────────────────────────
    ("cancel my order and give me a refund",
                                       ["shipping:cancel_order", "billing:refund"],
                                                                              "multi",       "cross-domain compound"),
    ("I want to return this item and open a support ticket",
                                       ["shipping:return_item", "support:create_ticket"],
                                                                              "multi",       "cross-domain compound"),
    ("reset my password and update my email",
                                       ["account:reset_password", "account:update_profile"],
                                                                              "multi",       "same-domain compound"),
    ("track my order and report a delivery issue",
                                       ["shipping:track_order", "shipping:delivery_problem"],
                                                                              "multi",       "same-domain compound"),

    # ── Multilingual ──────────────────────────────────────────────────────────
    ("cancelar mi suscripción",        ["billing:cancel_subscription"],       "multilingual", "es"),
    ("quiero un reembolso",            ["billing:refund"],                    "multilingual", "es"),
    ("我的包裹在哪",                    ["shipping:track_order"],              "multilingual", "zh"),
    ("取消我的订单",                    ["shipping:cancel_order"],             "multilingual", "zh"),
    ("Passwort vergessen",             ["account:reset_password"],            "multilingual", "de"),
    ("ouvrir un ticket",               ["support:create_ticket"],             "multilingual", "fr"),

    # ── Negation ──────────────────────────────────────────────────────────────
    ("I do not want a refund",         [],                                    "negation",    "negated refund"),
    ("don't cancel my subscription",   [],                                    "negation",    "negated cancel"),

    # ── No match (should return empty) ────────────────────────────────────────
    ("the weather is nice today",      [],                                    "no-match",    "unrelated"),
    ("what is 2 + 2",                  [],                                    "no-match",    "unrelated"),
]

# ─── Setup ────────────────────────────────────────────────────────────────────

def setup_intents():
    print(f"\n{'='*60}")
    print(f"SETUP: Creating {len(INTENTS)} intents in namespace '{NS}'")
    print('='*60)

    for intent_id, data in INTENTS.items():
        phrases_by_lang = {lang: phrases for lang, phrases in data.items()
                          if lang not in ("description",)}
        status, resp = post("/intents/multilingual", {
            "id": intent_id,
            "phrases_by_lang": phrases_by_lang,
        })
        if status not in (200, 201):
            print(f"  ERROR creating {intent_id}: {status} {resp}")
        else:
            total = sum(len(p) for p in phrases_by_lang.values())
            print(f"  ✓ {intent_id} ({total} phrases, langs: {list(phrases_by_lang.keys())})")

        # Set description
        post("/intents/description", {"intent_id": intent_id, "description": data["description"]})

def bootstrap():
    print(f"\n{'='*60}")
    print("BOOTSTRAP: Generating Hebbian L1 + L2 graphs via LLM")
    print('='*60)

    print("  Generating L1 (word association / synonyms)...")
    t0 = time.time()
    status, resp = post("/hebbian/bootstrap", {})
    elapsed = time.time() - t0
    if status == 200:
        edges = resp.get("edge_count", "?") if resp else "?"
        print(f"  ✓ L1 done in {elapsed:.1f}s — {edges} edges")
    else:
        print(f"  ✗ L1 failed: {status} {resp}")
        return False

    print("  Generating L2 (intent activation graph)...")
    t0 = time.time()
    status, resp = post("/hebbian/bootstrap_intent", {})
    elapsed = time.time() - t0
    if status == 200:
        intents = resp.get("intent_count", "?") if resp else "?"
        print(f"  ✓ L2 done in {elapsed:.1f}s — {intents} intents")
    else:
        print(f"  ✗ L2 failed: {status} {resp}")
        return False

    return True

# ─── Routing test ─────────────────────────────────────────────────────────────

def route(query, threshold=0.25):
    status, resp = post("/route_multi", {"query": query, "threshold": threshold})
    if status != 200 or not resp:
        return [], [], "no_match"
    confirmed = [r["id"] for r in resp.get("confirmed", [])]
    candidates = [r["id"] for r in resp.get("candidates", [])]
    disposition = resp.get("disposition", "?")
    return confirmed, candidates, disposition

def run_tests():
    print(f"\n{'='*60}")
    print("ROUTING TESTS")
    print('='*60)

    results = {"pass": 0, "fail": 0, "partial": 0}
    by_type = {}

    for query, expected, test_type, notes in TESTS:
        confirmed, candidates, disposition = route(query)
        all_detected = confirmed + candidates

        if not expected:
            # Expect empty
            ok = len(all_detected) == 0
            status_icon = "✓" if ok else "✗"
            if ok: results["pass"] += 1
            else:  results["fail"] += 1
        else:
            hits = [e for e in expected if e in all_detected]
            if len(hits) == len(expected):
                ok = True
                status_icon = "✓"
                results["pass"] += 1
            elif hits:
                ok = None  # partial
                status_icon = "~"
                results["partial"] += 1
            else:
                ok = False
                status_icon = "✗"
                results["fail"] += 1

        # Track by type
        t = by_type.setdefault(test_type, {"pass": 0, "fail": 0, "partial": 0})
        if ok is True:  t["pass"] += 1
        elif ok is None: t["partial"] += 1
        else:            t["fail"] += 1

        # Print result
        disp_tag = f"[{disposition}]"
        query_display = query[:52] + "…" if len(query) > 53 else query
        print(f"  {status_icon} [{test_type:12}] {query_display:<54} ", end="")
        if expected:
            print(f"→ {confirmed[:2]} | cand={candidates[:2]} {disp_tag}")
        else:
            print(f"→ (expect empty) got={all_detected[:2]} {disp_tag}")
        if ok is False and expected:
            print(f"      expected: {expected}")

    # Summary
    total = results["pass"] + results["fail"] + results["partial"]
    print(f"\n{'='*60}")
    print(f"SUMMARY: {results['pass']}/{total} pass, {results['partial']} partial, {results['fail']} fail")
    print(f"{'='*60}")
    print(f"{'Type':<14} {'Pass':>5} {'Partial':>8} {'Fail':>6}")
    print(f"{'-'*35}")
    for t, r in sorted(by_type.items()):
        print(f"  {t:<12} {r['pass']:>5} {r['partial']:>8} {r['fail']:>6}")

    return results

# ─── Graph stats ──────────────────────────────────────────────────────────────

def show_graph_stats():
    _, l1 = get("/hebbian")
    _, l2 = get("/hebbian/intent_graph")
    print(f"\n{'='*60}")
    print("GRAPH STATS")
    print('='*60)
    if l1:
        print(f"  L1 HebbianGraph : {l1.get('node_count','?')} nodes, {l1.get('edge_count','?')} edges")
    else:
        print("  L1 HebbianGraph : not bootstrapped")
    if l2:
        print(f"  L2 IntentGraph  : {l2.get('intent_count','?')} intents, {l2.get('word_count','?')} words, {l2.get('edge_count','?')} edges")
    else:
        print("  L2 IntentGraph  : not bootstrapped")

# ─── Auto-learn cycle ────────────────────────────────────────────────────────

def enable_auto_mode():
    post("/review/mode", {"mode": "auto"})

def wait_for_worker(timeout=120):
    """Poll review stats until pending count drops to 0 or timeout."""
    print(f"\n  Waiting for worker to process flagged entries (timeout={timeout}s)...")
    t0 = time.time()
    last_pending = None
    while time.time() - t0 < timeout:
        _, stats = get("/review/stats")
        pending = stats.get("pending", "?") if stats else "?"
        if pending != last_pending:
            print(f"  → pending: {pending}")
            last_pending = pending
        if isinstance(pending, int) and pending == 0:
            elapsed = time.time() - t0
            print(f"  ✓ Worker done in {elapsed:.1f}s")
            return True
        time.sleep(3)
    print(f"  ✗ Timeout after {timeout}s — some entries may still be pending")
    return False

def run_auto_learn_cycle():
    print(f"\n{'='*60}")
    print("AUTO-LEARN CYCLE")
    print('='*60)

    # Enable auto mode
    enable_auto_mode()
    print("  Mode set to: auto")

    # Route all failing test queries — flags them for the worker
    print(f"\n  Routing {len(TESTS)} test queries to generate flags...")
    flagged = 0
    for query, expected, test_type, _ in TESTS:
        confirmed, candidates, disposition = route(query, threshold=0.25)
        all_detected = confirmed + candidates
        # Check if this is a failure or partial
        if expected:
            hits = [e for e in expected if e in all_detected]
            if len(hits) < len(expected):
                flagged += 1
        else:
            if all_detected:
                flagged += 1
    print(f"  Flagged {flagged} queries for auto-learn")

    # Wait for worker to process all flags
    wait_for_worker(timeout=300)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:3001")
    parser.add_argument("--skip-setup", action="store_true", help="Reuse existing intents")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Reuse existing graphs")
    parser.add_argument("--auto-learn", action="store_true", help="Run auto-learn cycle then re-test")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.base_url.rstrip("/")

    # Check server
    status, _ = curl("GET", "/health", ns="default")
    if status != 200:
        print(f"ERROR: Server not reachable at {BASE_URL}")
        sys.exit(1)
    print(f"Server: {BASE_URL} ✓")

    if not args.skip_setup:
        setup_intents()

    show_graph_stats()

    if not args.skip_bootstrap:
        ok = bootstrap()
        if not ok:
            print("\nWARNING: Bootstrap failed — routing will use existing graphs only")

    show_graph_stats()

    print("\n" + "="*60)
    print("BEFORE AUTO-LEARN")
    print("="*60)
    run_tests()

    if args.auto_learn:
        run_auto_learn_cycle()

        print("\n" + "="*60)
        print("AFTER AUTO-LEARN")
        print("="*60)
        show_graph_stats()
        run_tests()

if __name__ == "__main__":
    main()
