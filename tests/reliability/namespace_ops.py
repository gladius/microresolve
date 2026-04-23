#!/usr/bin/env python3
"""
Namespace lifecycle helpers for reliability experiments.

- snapshot(src): capture src namespace's intents + seeds to JSON
- clone(snapshot, dst): (re)create dst from the snapshot
- add_phrases(namespace, phrase_map): add variant phrases to intents
- apply_corrections(namespace, corrections): feed /api/correct for each
- delete(namespace): remove from server
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE = "http://localhost:3001"
ROOT = Path(__file__).parent
SNAPSHOT = ROOT / "scale_test_snapshot.json"


def _post(path: str, body: dict, ns: str = None, retries: int = 3):
    headers = {"Content-Type": "application/json"}
    if ns: headers["X-Namespace-ID"] = ns
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{BASE}{path}", data=json.dumps(body).encode(),
                headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                txt = r.read()
                return r.status, json.loads(txt) if txt else {}
        except urllib.error.HTTPError as e:
            if e.code == 409 and attempt < retries - 1:
                time.sleep(0.3)
                continue
            return e.code, {"error": e.reason}


def _delete(path: str, body: dict = None, ns: str = None):
    headers = {"Content-Type": "application/json"}
    if ns: headers["X-Namespace-ID"] = ns
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body or {}).encode(),
        headers=headers, method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def _get(path: str, ns: str = None):
    headers = {}
    if ns: headers["X-Namespace-ID"] = ns
    req = urllib.request.Request(f"{BASE}{path}", headers=headers)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def snapshot(src: str = "scale-test", force: bool = False):
    """Capture a namespace's intents + phrases to SNAPSHOT file."""
    if SNAPSHOT.exists() and not force:
        print(f"Snapshot exists at {SNAPSHOT}, skipping (use force=True to override)")
        return
    intents = _get("/api/intents", ns=src)
    snap = [{
        "id": i["id"],
        "description": i.get("description", ""),
        "phrases_by_lang": i.get("phrases_by_lang", {}),
        "intent_type": i.get("intent_type", "action"),
        "metadata": i.get("metadata", {}),
    } for i in intents]
    SNAPSHOT.write_text(json.dumps(snap, indent=2))
    print(f"Snapshot saved: {len(snap)} intents → {SNAPSHOT}")


def delete_namespace(ns: str):
    code = _delete("/api/namespaces", {"namespace_id": ns})
    return code


def create_namespace(ns: str):
    code, _ = _post("/api/namespaces", {"namespace_id": ns})
    return code


def clone_from_snapshot(dst: str):
    """Create dst namespace and populate from the snapshot."""
    if not SNAPSHOT.exists():
        raise RuntimeError(f"No snapshot at {SNAPSHOT}; run snapshot() first")
    data = json.loads(SNAPSHOT.read_text())

    # Delete and recreate
    delete_namespace(dst)  # ignore errors (may not exist)
    time.sleep(0.2)
    create_namespace(dst)
    time.sleep(0.2)

    added = 0
    for intent in data:
        phrases = intent["phrases_by_lang"] or {"en": []}
        # Strip any non-en / _learned keys — use en only for simplicity
        if "en" not in phrases:
            phrases = {"en": phrases.get(list(phrases.keys())[0], []) if phrases else []}
        else:
            phrases = {"en": phrases["en"]}

        body = {
            "id": intent["id"],
            "phrases_by_lang": phrases,
            "intent_type": intent["intent_type"],
            "description": intent.get("description", ""),
        }
        meta = intent.get("metadata") or {}
        if meta:
            body["metadata"] = meta
        code, _ = _post("/api/intents/multilingual", body, ns=dst)
        if code in (200, 201):
            added += 1
        else:
            print(f"  WARN: failed to add {intent['id']}: {code}")
    print(f"Cloned '{dst}' from snapshot: {added}/{len(data)} intents")


def add_variant_phrases(ns: str, variant_map: dict, intent_data: list):
    """
    For Step 2 Mode B: add each equivalence variant as an extra phrase on the
    intent it belongs to (original word from that intent's seeds).

    variant_map: {variant: [canonical_words]} — from equivalence_classes.json
    intent_data: the snapshot list, so we know which intents contain which words
    """
    import re
    WORD = re.compile(r"[a-zA-Z]+")

    # Build: canonical -> [intent_ids that contain canonical in seed]
    canonical_to_intents = {}
    for intent in intent_data:
        intent_id = intent["id"]
        phrases = intent.get("phrases_by_lang", {}).get("en", [])
        words = set()
        for p in phrases:
            for w in WORD.findall(p):
                words.add(w.lower())
        for w in words:
            canonical_to_intents.setdefault(w, set()).add(intent_id)

    added = 0
    failed = 0
    seen = set()  # avoid duplicate adds

    for variant, canonicals in variant_map.items():
        for canonical in canonicals:
            if canonical not in canonical_to_intents:
                continue
            for intent_id in canonical_to_intents[canonical]:
                key = (intent_id, variant)
                if key in seen:
                    continue
                seen.add(key)
                # Add variant as a one-word phrase
                code, _ = _post(f"/api/intents/{intent_id}/phrases", {
                    "phrase": variant,
                    "lang": "en",
                }, ns=ns)
                if code in (200, 201):
                    added += 1
                else:
                    failed += 1

    print(f"  Added {added} variant phrases across intents; {failed} failed")


def apply_corrections(ns: str, corrections: list) -> dict:
    """Feed corrections via /api/correct. Returns stats."""
    stats = {"total": len(corrections), "applied": 0, "skipped": 0}
    for c in corrections:
        # First need to route the query to find what WAS detected (wrong_intent)
        code, routing = _post("/api/route_multi", {
            "query": c["query"], "threshold": 0.3, "log": False,
        }, ns=ns)
        if code != 200:
            stats["skipped"] += 1
            continue
        confirmed = routing.get("confirmed") or []
        # If nothing detected, call /api/learn instead of /api/correct
        if not confirmed:
            code, _ = _post("/api/learn", {
                "query": c["query"], "intent_id": c["correct"],
            }, ns=ns)
            if code in (200, 201):
                stats["applied"] += 1
            continue
        wrong_intent = confirmed[0]["id"]
        if wrong_intent == c["correct"]:
            stats["skipped"] += 1  # Already correct, no correction needed
            continue
        code, _ = _post("/api/correct", {
            "query": c["query"],
            "wrong_intent": wrong_intent,
            "correct_intent": c["correct"],
        }, ns=ns)
        if code in (200, 201):
            stats["applied"] += 1
        else:
            stats["skipped"] += 1
    return stats


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    if cmd == "snapshot":
        snapshot("scale-test", force=True)
    elif cmd == "clone":
        clone_from_snapshot(sys.argv[2])
    elif cmd == "delete":
        print(delete_namespace(sys.argv[2]))
    else:
        print(f"Unknown cmd: {cmd}")
