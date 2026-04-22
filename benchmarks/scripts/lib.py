"""Shared utilities for ASV benchmark runners.

All functions assume the ASV server is running on localhost:3001.
Namespace header: X-Namespace-ID
"""

import json
import time
import urllib.request
import urllib.error
import statistics
from typing import Any

BASE = "http://localhost:3001"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _req(method: str, path: str, body: Any = None, ns: str | None = None, timeout: int = 30) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    if ns:
        req.add_header("X-Namespace-ID", ns)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().strip()
            if not raw:
                return None
            try:
                return json.loads(raw)
            except Exception:
                return raw.decode(errors="replace")
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} {method} {path}: {body_text}") from e


def check_server():
    try:
        req = urllib.request.Request(f"{BASE}/api/health")
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
        return True
    except Exception as e:
        print(f"Server not reachable at {BASE}: {e}")
        return False


# ── Namespace management ──────────────────────────────────────────────────────

def create_namespace(name: str) -> None:
    try:
        _req("POST", "/api/namespaces", {"namespace_id": name})
    except Exception:
        pass  # may already exist


def delete_namespace(name: str) -> None:
    try:
        _req("DELETE", "/api/namespaces", {"namespace_id": name})
    except Exception:
        pass


# ── Intent & phrase loading ───────────────────────────────────────────────────

def load_seeds(ns: str, intent_phrases: dict[str, list[str]]) -> int:
    """Load intents and seed phrases via multilingual endpoint (seeds L2). Returns total phrases added."""
    total = 0
    for intent_id, phrases in intent_phrases.items():
        try:
            # /api/intents/multilingual seeds L2 (IntentGraph) immediately
            _req("POST", "/api/intents/multilingual", {
                "id": intent_id,
                "phrases_by_lang": {"en": phrases},
            }, ns=ns)
            total += len(phrases)
        except Exception as e:
            pass
    return total


# ── Routing ───────────────────────────────────────────────────────────────────

def route_query(ns: str, text: str) -> dict:
    """Route a single query. Returns the full response dict."""
    return _req("POST", "/api/route_multi", {"query": text}, ns=ns)


def run_queries(ns: str, examples: list[dict]) -> list[dict]:
    """Route all examples. Each example must have 'text' and 'intents' keys.

    Returns list of result dicts with keys:
      text, expected, predicted (list), top1_correct, any_correct, latency_us
    """
    results = []
    for ex in examples:
        text = ex["text"]
        expected = ex["intents"]  # list of intent ids
        t0 = time.perf_counter()
        try:
            resp = route_query(ns, text)
            elapsed_us = resp.get("routing_us") or (time.perf_counter() - t0) * 1_000_000
            predicted = [r["id"] for r in resp.get("confirmed", [])]
            top1 = predicted[0] if predicted else None
            results.append({
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "top1_correct": top1 in expected,
                "any_correct": any(p in expected for p in predicted),
                "latency_us": elapsed_us,
            })
        except Exception as e:
            results.append({
                "text": text,
                "expected": expected,
                "predicted": [],
                "top1_correct": False,
                "any_correct": False,
                "latency_us": 0,
                "error": str(e),
            })
    return results


# ── Learning ──────────────────────────────────────────────────────────────────

def apply_learning(ns: str, results: list[dict]) -> int:
    """For each missed intent, add the query as a training phrase. No LLM.
    Works for both single-intent (top1 miss) and multi-intent (per-missed-intent).
    Returns count of phrases added."""
    learned = 0
    for r in results:
        expected = set(r["expected"])
        predicted = set(r["predicted"])
        missed = expected - predicted  # intents we failed to predict
        for intent_id in missed:
            try:
                resp = _req("POST", "/api/intents/phrase", {
                    "intent_id": intent_id,
                    "phrase": r["text"],
                }, ns=ns)
                if resp and resp.get("added"):
                    learned += 1
            except Exception:
                pass
    return learned


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy, top-1, latency stats from run_queries output."""
    n = len(results)
    if n == 0:
        return {}

    top1_correct = sum(1 for r in results if r["top1_correct"])
    any_correct = sum(1 for r in results if r["any_correct"])
    latencies = [r["latency_us"] for r in results if r["latency_us"] > 0]

    metrics = {
        "n": n,
        "top1_accuracy": round(top1_correct / n * 100, 2),
        "any_accuracy": round(any_correct / n * 100, 2),
    }

    if latencies:
        latencies.sort()
        metrics["latency_p50_us"] = round(statistics.median(latencies), 1)
        metrics["latency_p95_us"] = round(latencies[int(len(latencies) * 0.95)], 1)
        metrics["latency_p99_us"] = round(latencies[int(len(latencies) * 0.99)], 1)
        metrics["latency_mean_us"] = round(statistics.mean(latencies), 1)

    return metrics


def compute_multiintent_metrics(results: list[dict]) -> dict:
    """F1, exact match, partial match for multi-intent evaluation."""
    n = len(results)
    if n == 0:
        return {}

    exact = 0
    partial = 0
    precision_sum = 0.0
    recall_sum = 0.0

    for r in results:
        expected = set(r["expected"])
        predicted = set(r["predicted"])
        if expected == predicted:
            exact += 1
        if expected & predicted:
            partial += 1
        if predicted:
            precision_sum += len(expected & predicted) / len(predicted)
        recall_sum += len(expected & predicted) / len(expected) if expected else 0

    precision = precision_sum / n
    recall = recall_sum / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    latencies = [r["latency_us"] for r in results if r["latency_us"] > 0]

    metrics = {
        "n": n,
        "exact_match": round(exact / n * 100, 2),
        "partial_match": round(partial / n * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
    }

    if latencies:
        latencies.sort()
        metrics["latency_p50_us"] = round(statistics.median(latencies), 1)
        metrics["latency_p95_us"] = round(latencies[int(len(latencies) * 0.95)], 1)
        metrics["latency_mean_us"] = round(statistics.mean(latencies), 1)

    return metrics


# ── Results I/O ───────────────────────────────────────────────────────────────

def save_result(path: str, data: dict) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")


def print_metrics(label: str, metrics: dict) -> None:
    top1 = metrics.get("top1_accuracy") or metrics.get("f1")
    key = "top1" if "top1_accuracy" in metrics else "f1"
    print(f"  {label}: {key}={top1}%  n={metrics.get('n')}  "
          f"p50={metrics.get('latency_p50_us', '?')}µs  "
          f"p95={metrics.get('latency_p95_us', '?')}µs")
