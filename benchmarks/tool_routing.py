#!/usr/bin/env python3
"""Tool routing benchmark using BFCL v4 (Berkeley Function-Calling Leaderboard).

Treats each unique function across the BFCL v4 `live_multiple` test split
as an intent inside a single namespace, then routes the LIVE user queries
through MicroResolve and reports top-1 / top-3 accuracy + latency.

This is the "pre-LLM tool prefilter" use case: given N candidate tools, can
the deterministic engine pick the right one in microseconds before the LLM
ever sees the request? BFCL `live_multiple` contains real user questions,
each labelled with one ground-truth function name from a candidate set.

Run:
  cargo build --release --features server
  ./target/release/microresolve-studio --port 3001 --no-open --data /tmp/mr_bench &
  python3 benchmarks/tool_routing.py

Outputs:
  benchmarks/results/tool_routing_baseline.json   (seed-only metrics)
  benchmarks/datasets/bfcl/                        (cached BFCL data)

No LLM in the loop for the baseline — pure deterministic classifier eval.
A `--with-auto-learn` variant calling the LLM Turn 1+2 pipeline lands in
v0.1.1 once we wire it into a benchmark harness.

Source dataset:
  https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
  Apache 2.0 license. Cited in: Patil, S. G. et al. "BFCL: From Tool Use
  to Agentic Evaluation of Large Language Models." 2024.
"""

from __future__ import annotations
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "datasets" / "bfcl"
RESULTS = ROOT / "results"
BASE = os.environ.get("MR_SERVER_URL", "http://localhost:3001")
NS = "bench-tool-routing"


# ── Dataset preparation ─────────────────────────────────────────────────────

def fetch_bfcl_v4() -> tuple[Path, Path]:
    """Fetch BFCL v4 live_multiple test split + possible_answer file via
    sparse-clone of the upstream repo. Cache under benchmarks/datasets/bfcl/.

    Returns (questions_path, answers_path).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    questions = CACHE / "BFCL_v4_live_multiple.json"
    answers   = CACHE / "BFCL_v4_live_multiple_answer.json"
    if questions.exists() and answers.exists():
        return questions, answers

    print(f"  → cloning gorilla repo (sparse, ~5MB) into {CACHE}/_repo")
    repo_dir = CACHE / "_repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         "https://github.com/ShishirPatil/gorilla", str(repo_dir)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "sparse-checkout", "set",
         "berkeley-function-call-leaderboard/bfcl_eval/data"],
        check=True, capture_output=True,
    )
    src = repo_dir / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    shutil.copy(src / "BFCL_v4_live_multiple.json", questions)
    shutil.copy(src / "possible_answer" / "BFCL_v4_live_multiple.json", answers)
    shutil.rmtree(repo_dir)
    print(f"  → cached: {questions.name}, {answers.name}")
    return questions, answers


def parse_bfcl(questions_path: Path, answers_path: Path) -> tuple[
    dict[str, str],          # function name → description (for plain mode)
    list[dict],              # full function specs (for mcp-import mode)
    list[tuple[str, str]],   # (query_text, expected_function_name) test pairs
]:
    """Parse BFCL JSONL into (description pool, full function specs, test pairs).

    Pools every unique function (name, description, parameters) across all
    test cases. The test set is the user query + the ground-truth function
    name from possible_answer (the dict's outer key is the function the
    query maps to).
    """
    answers: dict[str, str] = {}
    with open(answers_path) as f:
        for line in f:
            row = json.loads(line)
            gt = row["ground_truth"][0]
            answers[row["id"]] = next(iter(gt.keys()))

    desc_pool: dict[str, str] = {}
    full_specs: dict[str, dict] = {}
    pairs: list[tuple[str, str]] = []
    with open(questions_path) as f:
        for line in f:
            row = json.loads(line)
            for fn in row["function"]:
                desc_pool.setdefault(fn["name"], fn.get("description", "") or fn["name"])
                full_specs.setdefault(fn["name"], fn)
            user_msg = row["question"][0][0]["content"]
            expected = answers.get(row["id"])
            if expected:
                pairs.append((user_msg, expected))

    return desc_pool, list(full_specs.values()), pairs


# ── HTTP helpers ────────────────────────────────────────────────────────────

def _req(method: str, path: str, body=None, ns: str | None = None,
         timeout: float = 30.0) -> dict:
    headers = {"Content-Type": "application/json"}
    if ns:
        headers["X-Namespace-ID"] = ns
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
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


# ── Bench phases ────────────────────────────────────────────────────────────

def setup_namespace_plain(function_pool: dict[str, str]) -> int:
    """Plain mode: each function → 1 seed = its description.
    No LLM. No L1 augmentation. Pure cold-start measurement.
    """
    try:
        _req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception:
        pass
    _req("POST", "/api/namespaces", {
        "namespace_id": NS,
        "description": "BFCL v4 live_multiple — plain seed-only baseline",
    })
    n = 0
    for fname, desc in function_pool.items():
        seed = desc.strip() if desc.strip() else fname
        try:
            _req("POST", "/api/intents", {
                "id": fname,
                "phrases": [seed],
            }, ns=NS)
            n += 1
        except RuntimeError as e:
            print(f"    ⚠ skipping {fname}: {e}")
    return n


def setup_namespace_mcp(function_pool_full: list[dict]) -> int:
    """MCP-import mode: route the function list through `/api/import/mcp/apply`,
    which triggers the server's L1 LLM augmentation (Turn 1 generates a
    morphology / abbreviation / synonym graph across all intents; Turn 2
    verifies). This is the realistic flow used by anyone importing an MCP
    server in production.

    Cost: ~1-2 LLM calls TOTAL (not per-function). With 457 BFCL functions
    the prompt is ~50 KB — well within Llama 3.3 / Haiku context.
    """
    try:
        _req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception:
        pass
    _req("POST", "/api/namespaces", {
        "namespace_id": NS,
        "description": "BFCL v4 live_multiple — MCP-import + L1-augmented",
    })
    # Build MCP-shape tools list. BFCL uses `parameters`, MCP wants `inputSchema`.
    tools = [{
        "name":         f["name"],
        "description":  f.get("description", ""),
        "inputSchema":  f.get("parameters", {"type": "object", "properties": {}}),
    } for f in function_pool_full]
    # Issue ONE bulk import call. Server creates intents + triggers L1 LLM aug.
    # The LLM pass can take 30-180s depending on provider + prompt size — give
    # it room. (Default urllib timeout is 30s; set explicit longer one here.)
    res = _req("POST", "/api/import/mcp/apply", {
        "tools_json": json.dumps({"tools": tools}),
        "selected":   [t["name"] for t in tools],
        "domain":     "",
    }, ns=NS, timeout=300.0)
    return int(res.get("imported") or res.get("created") or len(tools))


def measure(pairs: list[tuple[str, str]], grounded_l1: bool = False) -> dict:
    """Run each test query, compare top-1 / top-3 to ground truth.
    Returns metrics + per-query latencies.

    `grounded_l1=True` enables the L1-grounded preprocessing path on the
    server: morphology + abbreviation always, plus synonym substitution
    only when the source token is OOV to the L2 vocabulary (Option E).
    """
    top1_hits = 0
    top3_hits = 0
    latencies_us: list[float] = []
    misclassifications: list[tuple[str, str, str]] = []
    for query, expected in pairs:
        t0 = time.perf_counter()
        res = _req("POST", "/api/route_multi", {
            "query": query,
            "log": False,
            "grounded_l1": grounded_l1,
        }, ns=NS)
        latencies_us.append((time.perf_counter() - t0) * 1_000_000.0)
        # Server returns "ranked" sorted by score; "confirmed" is the post-
        # consume multi-intent set. For a single-tool routing task, top-N
        # of `ranked` is what we want.
        ranked = res.get("ranked") or res.get("confirmed") or []
        top_ids = [m["id"] for m in ranked[:3]]
        if top_ids and top_ids[0] == expected:
            top1_hits += 1
        if expected in top_ids:
            top3_hits += 1
        else:
            misclassifications.append((query, expected, top_ids[0] if top_ids else "(none)"))
    n = len(pairs)
    sorted_us = sorted(latencies_us)
    p = lambda q: sorted_us[min(int((n - 1) * q), n - 1)] if sorted_us else 0
    return {
        "n":             n,
        "top1_hits":     top1_hits,
        "top3_hits":     top3_hits,
        "top1_pct":      round(100 * top1_hits / n, 2) if n else 0,
        "top3_pct":      round(100 * top3_hits / n, 2) if n else 0,
        "latency_p50_us": round(p(0.50), 1),
        "latency_p95_us": round(p(0.95), 1),
        "latency_p99_us": round(p(0.99), 1),
        "latency_mean_us": round(statistics.fmean(latencies_us), 1) if latencies_us else 0,
        "misclassified_sample": misclassifications[:10],
    }


def main() -> int:
    mode = "plain"
    for arg in sys.argv[1:]:
        if arg in ("--mode=plain", "--plain"):       mode = "plain"
        elif arg in ("--mode=mcp", "--mcp"):         mode = "mcp"
        elif arg in ("-h", "--help"):
            print(__doc__); print("\nFlags: --plain (default) | --mcp"); return 0

    print(f"BFCL v4 tool-routing benchmark → {BASE}")
    print(f"  mode: {mode}\n")
    if not health_check():
        return 1

    print("→ fetching BFCL v4 live_multiple (real user queries on real APIs)...")
    q_path, a_path = fetch_bfcl_v4()
    desc_pool, full_specs, test_pairs = parse_bfcl(q_path, a_path)
    print(f"  {len(desc_pool)} unique functions, {len(test_pairs)} labeled queries\n")

    print(f"→ setup [{mode}]: loading {len(desc_pool)} functions as intents into '{NS}'...")
    t0 = time.perf_counter()
    if mode == "mcp":
        n_loaded = setup_namespace_mcp(full_specs)
    else:
        n_loaded = setup_namespace_plain(desc_pool)
    setup_secs = time.perf_counter() - t0
    print(f"  loaded {n_loaded} intents in {setup_secs:.1f}s\n")

    # In MCP mode, the server populated L1 with LLM-generated morphology +
    # abbreviation + synonym edges. Use grounded_l1 so synonyms get
    # substituted only for OOV tokens — preserves distinctive vocabulary.
    grounded = (mode == "mcp")
    print(f"→ measuring: routing {len(test_pairs)} BFCL queries (grounded_l1={grounded})...")
    t0 = time.perf_counter()
    metrics = measure(test_pairs, grounded_l1=grounded)
    print(f"  done in {time.perf_counter() - t0:.1f}s\n")

    print(f"  top-1     {metrics['top1_pct']:>6.2f}%   ({metrics['top1_hits']} / {metrics['n']})")
    print(f"  top-3     {metrics['top3_pct']:>6.2f}%   ({metrics['top3_hits']} / {metrics['n']})")
    print(f"  p50       {metrics['latency_p50_us']:>6.1f} µs")
    print(f"  p95       {metrics['latency_p95_us']:>6.1f} µs")
    print(f"  p99       {metrics['latency_p99_us']:>6.1f} µs")
    print(f"  mean      {metrics['latency_mean_us']:>6.1f} µs")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"tool_routing_{mode}.json"
    out.write_text(json.dumps({
        "dataset":         "BFCL v4 live_multiple",
        "mode":            mode,
        "setup_seconds":   round(setup_secs, 2),
        "intents_loaded":  n_loaded,
        "test_examples":   metrics["n"],
        "seeds_per_intent": "1 (description)" if mode == "plain"
                           else "1 (description) + LLM-augmented L1 graph",
        **{k: v for k, v in metrics.items() if k != "misclassified_sample"},
        "first_misclassifications": metrics["misclassified_sample"],
    }, indent=2))
    print(f"\n  → wrote {out}")

    try:
        _req("DELETE", "/api/namespaces", {"namespace_id": NS})
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
