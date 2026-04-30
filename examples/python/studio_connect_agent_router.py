"""Connected MicroResolve client for the agent-tool-router demo.

Subscribes to a running Studio at http://localhost:3001 and keeps a
local copy of the `agent` namespace in ./data-python/.

Usage:
  python agent_router_client.py status                              # version + intent count
  python agent_router_client.py watch                               # live: print version every change
  python agent_router_client.py resolve "cancel my subscription"    # one-shot resolve
"""

import sys
import time

from microresolve import MicroResolve

SERVER = "http://localhost:3001"
NAMESPACE = "agent"
DATA_DIR = "./data-python"
TICK_SECS = 5

mr = MicroResolve(
    server_url=SERVER,
    subscribe=[NAMESPACE],
    tick_interval_secs=TICK_SECS,
    data_dir=DATA_DIR,
)
ns = mr.namespace(NAMESPACE)


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        print(f"version={ns.version()}  intents={ns.intent_count()}")
        return 0

    if cmd == "watch":
        print(f"watching {NAMESPACE!r} on {SERVER} (tick every {TICK_SECS}s)…")
        last = -1
        while True:
            v = ns.version()
            if v != last:
                print(f"v{v}  intents={ns.intent_count()}")
                last = v
            time.sleep(1)

    if cmd == "resolve":
        query = " ".join(sys.argv[2:])
        if not query:
            print("usage: agent_router_client.py resolve <query…>", file=sys.stderr)
            return 2
        matches = ns.resolve(query)
        if matches:
            print(f"{matches[0].id}  score={matches[0].score:.2f}")
        else:
            print("(no match)")
        return 0

    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
