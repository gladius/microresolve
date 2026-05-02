"""Connected-mode demo: sync a namespace from a running MicroResolve server,
classify locally, push a correction back.

Start the server first:
  ../target/release/microresolve-studio --port 3001 --no-browser --data /tmp/mr_server_data &

Then run:
  python examples/connected.py
  # or override the URL/key via env:
  MR_SERVER_URL=http://localhost:3097 MR_API_KEY=mr_xxx python examples/connected.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

from microresolve import MicroResolve

SERVER_URL = os.environ.get("MR_SERVER_URL", "http://localhost:3001")
API_KEY    = os.environ.get("MR_API_KEY")
NS         = "demo-py-connected"


def _request(method: str, path: str, body: dict | None = None, ns_header: bool = False) -> None:
    """Tiny stdlib-only HTTP helper. Raises urllib.error.HTTPError on non-2xx."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-Api-Key"] = API_KEY
    if ns_header:
        headers["X-Namespace-ID"] = NS
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{SERVER_URL}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10):
        pass


def http_setup() -> None:
    """Create the demo namespace + intents on the server (out-of-band setup)."""
    # best-effort cleanup of prior runs
    try:
        _request("DELETE", "/api/namespaces", {"namespace_id": NS})
    except urllib.error.HTTPError:
        pass
    _request("POST", "/api/namespaces", {"namespace_id": NS, "description": "python connected demo"})
    for intent_id, phrases in [
        ("list_subscriptions", ["list my subscriptions", "show all subscriptions"]),
        ("cancel_subscription", ["cancel subscription", "stop my subscription"]),
        ("greeting", ["hello", "hi there"]),
    ]:
        _request("POST", "/api/intents", {"id": intent_id, "phrases": phrases}, ns_header=True)


def main() -> int:
    print(f"─── 1. Setup namespace + intents on {SERVER_URL} ─────────────")
    try:
        http_setup()
    except Exception as exc:
        print(f"  ✗ server unreachable: {exc}")
        print(f"  Start the server: ../target/release/microresolve-studio --port {SERVER_URL.rsplit(':', 1)[-1]} --no-browser")
        return 1
    print(f"  ✓ namespace '{NS}' seeded with 3 intents")

    print("\n─── 2. Connect library to server ─────────────────────────────")
    engine = MicroResolve(
        server_url=SERVER_URL,
        api_key=API_KEY,
        subscribe=[NS],
        tick_interval_secs=5,
    )
    ns = engine.namespace(NS)
    print(f"  connected. version = {ns.version()}, intents = {ns.intent_count()}")

    query = "drop my subscription right now"

    print("\n─── 3. Resolve a query ───────────────────────────────────────")
    matches = ns.resolve(query)
    initial = matches[0].id if matches else "(none)"
    print(f"  query  : {query!r}")
    print(f"  routed : {initial} (score: {matches[0].score:.2f})" if matches else "  no match")

    print("\n─── 4. Strict mode: library mutations refused ───────────────")
    print("  Connected libraries are READ-ONLY caches. Calling ns.correct(...) raises:")
    wrong = initial if initial != "(none)" else "list_subscriptions"
    try:
        ns.correct(query, wrong, "cancel_subscription")
        print("    UNEXPECTED: correct() succeeded")
    except Exception as e:
        print(f"    {type(e).__name__}: {e}  ← refused, as designed.")

    print("\n─── 5. Apply correction via the server's HTTP API ───────────")
    api_url = os.environ.get("MICRORESOLVE_URL", "http://localhost:4000")
    headers = {"X-Namespace-ID": NS}
    if api_key:
        headers["X-Api-Key"] = api_key
    r = urllib.request.Request(
        f"{api_url}/api/correct",
        data=json.dumps({"query": query, "wrong_intent": wrong, "right_intent": "cancel_subscription"}).encode(),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            print(f"  ✓ POST /api/correct → HTTP {resp.status}")
    except Exception as e:
        print(f"  ✗ HTTP correct failed: {e}")

    print("\n─── 6. Wait for sync tick to pull the change ────────────────")
    v_before = ns.version()
    for _ in range(8):
        time.sleep(1)
        if ns.version() > v_before:
            print(f"  ✓ pulled v{ns.version()} from server (was v{v_before})")
            break

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
