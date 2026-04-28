#!/usr/bin/env python3
"""End-to-end latency benchmark for MicroResolve's HTTP API.

Measures `POST /api/route_multi` latency including HTTP overhead — the
number a real client sees. For pure in-process latency, see the Rust
criterion bench at `benches/resolve.rs` (next benchmark).

Run:
  # 1. Start the server (fresh data dir):
  cargo build --release --features server
  ./target/release/server --port 3001 --no-open --data /tmp/mr_bench &

  # 2. Run this script:
  python3 benchmarks/latency.py

Output:
  - prints p50 / p95 / p99 / mean / stdev to stdout
  - writes JSON to benchmarks/results/latency.json
"""

from __future__ import annotations
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = os.environ.get("MR_SERVER_URL", "http://localhost:3001")
NS   = "bench-latency"

# ── Workload knobs ──────────────────────────────────────────────────────────
N_INTENTS         = 100
SEEDS_PER_INTENT  = 5
WARMUP_QUERIES    = 50
MEASURED_QUERIES  = 1000

# Synthetic vocabulary: intent name templates + verbs/nouns to make
# queries that look like real user input but cover the indexed words.
VERBS = [
    "cancel", "list", "show", "create", "update", "delete", "fetch", "stop",
    "start", "renew", "refund", "track", "send", "reset", "approve", "decline",
]
NOUNS = [
    "order", "subscription", "invoice", "payment", "account", "shipment",
    "ticket", "user", "session", "report", "notification", "transaction",
    "discount", "product", "customer", "review",
]
QUALIFIERS = [
    "my", "the", "this", "that", "current", "recent", "latest", "pending",
    "open", "active", "draft", "all",
]


# ── HTTP helpers ────────────────────────────────────────────────────────────

def _req(method: str, path: str, body=None, ns: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if ns:
        headers["X-Namespace-ID"] = ns
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {method} {path}: {e.read().decode(errors='replace')}") from None


def health_check() -> bool:
    try:
        _req("GET", "/api/namespaces")
        return True
    except Exception as e:
        print(f"server not reachable at {BASE}: {e}")
        return False


# ── Synthetic intent fixture ────────────────────────────────────────────────

def make_intents(n: int) -> dict[str, list[str]]:
    """Generate n synthetic intents, each with 5 distinct seed phrases."""
    rng = random.Random(42)
    intents: dict[str, list[str]] = {}
    for i in range(n):
        verb = VERBS[i % len(VERBS)]
        noun = NOUNS[(i // len(VERBS)) % len(NOUNS)]
        intent_id = f"{verb}_{noun}_{i:03d}"
        # Seeds: varied phrasings of the same intent
        seeds = [
            f"{verb} my {noun}",
            f"{verb} the {noun}",
            f"please {verb} {noun}",
            f"can you {verb} my {noun}",
            f"i want to {verb} {noun}",
        ]
        intents[intent_id] = seeds
    return intents


def make_queries(intents: dict[str, list[str]], n: int) -> list[str]:
    """Build n realistic queries using vocabulary the index has seen,
    with light variation so we exercise the L0 typo + L1 normalize paths."""
    rng = random.Random(7)
    queries = []
    intent_ids = list(intents.keys())
    for _ in range(n):
        # Pick an intent we want to "hit", build a paraphrased query.
        target_id = rng.choice(intent_ids)
        verb, noun = target_id.rsplit("_", 1)[0].split("_", 1)
        qualifier = rng.choice(QUALIFIERS)
        templates = [
            f"{verb} {qualifier} {noun}",
            f"please {verb} {qualifier} {noun}",
            f"i need to {verb} my {noun} now",
            f"{verb} {noun} immediately",
            f"can you help me {verb} the {noun}",
        ]
        queries.append(rng.choice(templates))
    return queries


# ── Setup ───────────────────────────────────────────────────────────────────

def setup_namespace() -> int:
    """Drop + recreate the bench namespace, load synthetic intents.
    Returns total phrases loaded."""
    try:
        _req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception:
        pass
    _req("POST", "/api/namespaces", {
        "namespace_id": NS,
        "description": "synthetic latency bench",
    })
    intents = make_intents(N_INTENTS)
    total = 0
    for intent_id, phrases in intents.items():
        _req("POST", "/api/intents", {
            "id": intent_id,
            "phrases": phrases,
        }, ns=NS)
        total += len(phrases)
    return total


# ── Measurement ─────────────────────────────────────────────────────────────

def measure(queries: list[str]) -> list[float]:
    """Return list of per-query latencies in microseconds."""
    latencies_us: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        _req("POST", "/api/route_multi", {"query": q, "log": False}, ns=NS)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1_000_000.0)
    return latencies_us


def percentile(sorted_xs: list[float], p: float) -> float:
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_xs) - 1)
    if f == c:
        return sorted_xs[f]
    return sorted_xs[f] + (sorted_xs[c] - sorted_xs[f]) * (k - f)


def main() -> int:
    print(f"MicroResolve latency bench → {BASE}")
    print(f"  workload: {N_INTENTS} intents × {SEEDS_PER_INTENT} seeds, "
          f"{WARMUP_QUERIES} warmup, {MEASURED_QUERIES} measured\n")

    if not health_check():
        return 1

    print("→ setup: creating namespace + indexing intents...")
    t0 = time.perf_counter()
    n_phrases = setup_namespace()
    setup_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  loaded {n_phrases} phrases in {setup_ms:.0f}ms\n")

    intents = make_intents(N_INTENTS)
    warmup_qs   = make_queries(intents, WARMUP_QUERIES)
    measured_qs = make_queries(intents, MEASURED_QUERIES)

    print(f"→ warmup: {WARMUP_QUERIES} queries...")
    measure(warmup_qs)

    print(f"→ measuring: {MEASURED_QUERIES} queries...")
    latencies = measure(measured_qs)
    latencies.sort()

    p50 = percentile(latencies, 0.50)
    p95 = percentile(latencies, 0.95)
    p99 = percentile(latencies, 0.99)
    mean = statistics.fmean(latencies)
    stdev = statistics.pstdev(latencies)
    qps = 1_000_000.0 / mean if mean > 0 else 0.0

    print()
    print(f"  Latency (end-to-end, including HTTP):")
    print(f"    p50  : {p50:>8.1f} µs")
    print(f"    p95  : {p95:>8.1f} µs")
    print(f"    p99  : {p99:>8.1f} µs")
    print(f"    mean : {mean:>8.1f} µs  (± {stdev:.1f} std)")
    print(f"    qps  : {qps:>8.0f}  (single connection, sequential)")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "latency.json"
    out_path.write_text(json.dumps({
        "server_url":    BASE,
        "n_intents":     N_INTENTS,
        "seeds_per_intent": SEEDS_PER_INTENT,
        "warmup":        WARMUP_QUERIES,
        "measured":      MEASURED_QUERIES,
        "p50_us":        round(p50, 1),
        "p95_us":        round(p95, 1),
        "p99_us":        round(p99, 1),
        "mean_us":       round(mean, 1),
        "stdev_us":      round(stdev, 1),
        "qps":           round(qps, 0),
    }, indent=2))
    print(f"\n  → wrote {out_path}")

    # Cleanup
    try:
        _req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
