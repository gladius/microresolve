#!/usr/bin/env python3
"""
Knowledge-as-intents experiment.

Test the hypothesis: an LLM can distill unstructured FAQ content
into MicroResolve context intents (routable triggers + bound answer), and
MicroResolve can route natural user queries to the correct FAQ at 30µs with
no RAG / no embeddings.

Success = >=80% top-1 on naturally-phrased held-out queries.
"""
import json
import os
import time
import requests
from pathlib import Path

ROOT = Path(__file__).parent
SERVER = "http://localhost:3001"
NS = "faq-stripe"
FAQS = json.loads((ROOT / "faq_source.json").read_text())

# Load .env
ENV = {}
for line in (Path(__file__).parent.parent.parent.parent / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        ENV[k.strip()] = v.strip()

LLM_URL = ENV["LLM_API_URL"]
LLM_KEY = ENV["LLM_API_KEY"]
LLM_MODEL = ENV["LLM_MODEL"]


import hashlib
CACHE_DIR = ROOT / "llm_cache"
CACHE_DIR.mkdir(exist_ok=True)


def call_llm(prompt: str, max_tokens: int = 4000) -> str:
    key = hashlib.sha256((LLM_MODEL + prompt + str(max_tokens)).encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"{key}.txt"
    if cache_path.exists():
        return cache_path.read_text()

    headers = {
        "Authorization": f"Bearer {LLM_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.15.0",
    }
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    for attempt in range(10):
        try:
            r = requests.post(LLM_URL, headers=headers, json=body, timeout=120)
        except requests.exceptions.RequestException as e:
            wait = min(60, 2 ** attempt + 3)
            print(f"    network error ({e.__class__.__name__}), waiting {wait}s...")
            time.sleep(wait)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(60, 2 ** attempt + 3)
            print(f"    {r.status_code} from LLM, waiting {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        cache_path.write_text(content)
        time.sleep(1.2)  # throttle to stay under Groq free-tier RPM
        return content
    raise RuntimeError("LLM call failed after retries")


def distill_seeds(faq: dict) -> list[str]:
    """Ask LLM to generate 8 seed phrases for how users would ask this in chat."""
    prompt = f"""You are building a chat-based FAQ router. Given the FAQ entry below, write 8 short ways a real user would ACTUALLY ask this question in a support chat.

Rules:
- Write how users TYPE, not how they write formal emails. Include casual, abbreviated, typo-prone phrasings.
- Vary length: mix 2-word fragments ("payout schedule") with full questions ("when do i get paid").
- Never include the FAQ's exact title verbatim.
- One per line, no numbering, no quotes.

FAQ id: {faq['id']}
FAQ title: {faq['title']}
FAQ answer: {faq['answer']}

Output exactly 8 lines:"""
    out = call_llm(prompt, max_tokens=500)
    lines = [l.strip().lstrip("-•*0123456789. )") for l in out.strip().split("\n")]
    lines = [l for l in lines if l and len(l) > 2]
    return lines[:8]


def generate_test_queries(faq: dict, n: int = 2) -> list[str]:
    """Held-out natural queries to test routing (not seen as seeds)."""
    prompt = f"""A user is in a support chat and wants to ask about this FAQ topic:

FAQ title: {faq['title']}
FAQ answer: {faq['answer']}

Write {n} DIFFERENT ways a user might ask about this — different from typical phrasings, with:
- Implied questions ("my payment keeps failing...")
- Indirect references ("is there a way to...")
- Colloquial / frustrated tone ("why isn't my money showing up")

One per line. No numbering, no quotes."""
    out = call_llm(prompt, max_tokens=200)
    lines = [l.strip().lstrip("-•*0123456789. )") for l in out.strip().split("\n")]
    lines = [l for l in lines if l and len(l) > 5]
    return lines[:n]


def setup_namespace():
    # Reset namespace
    requests.delete(f"{SERVER}/api/namespaces", json={"namespace_id": NS})
    time.sleep(0.2)
    r = requests.post(f"{SERVER}/api/namespaces",
                      json={"namespace_id": NS, "description": "Knowledge-as-intents FAQ experiment"})
    print(f"Created namespace '{NS}': {r.status_code}")

    headers = {"X-Namespace-ID": NS}

    print(f"\nDistilling {len(FAQS)} FAQs into context intents...")
    for i, faq in enumerate(FAQS, 1):
        seeds = distill_seeds(faq)
        print(f"  [{i:2d}/{len(FAQS)}] {faq['id']:30s} -> {len(seeds)} seeds")
        r = requests.post(f"{SERVER}/api/intents/multilingual",
                          headers=headers,
                          json={
                              "id": faq['id'],
                              "intent_type": "context",
                              "phrases_by_lang": {"en": seeds},
                              "description": faq['title'],
                              "metadata": {"answer": [faq['answer']]}
                          })
        if r.status_code >= 400:
            print(f"       ERROR: {r.text[:200]}")


def generate_test_set():
    """Generate held-out test queries (2 per FAQ)."""
    test_path = ROOT / "test_queries.json"
    if test_path.exists():
        print(f"Reusing cached test queries: {test_path}")
        return json.loads(test_path.read_text())

    print(f"\nGenerating held-out queries (2 per FAQ = {len(FAQS)*2} total)...")
    queries = []
    for i, faq in enumerate(FAQS, 1):
        qs = generate_test_queries(faq, n=2)
        for q in qs:
            queries.append({"query": q, "expected_id": faq['id']})
        print(f"  [{i:2d}/{len(FAQS)}] {faq['id']:30s} -> {len(qs)} queries")
    test_path.write_text(json.dumps(queries, indent=2))
    return queries


def measure(queries: list[dict]) -> dict:
    headers = {"X-Namespace-ID": NS}
    results = []
    lat_us = []
    top1 = top3 = 0
    t0 = time.time()
    for q in queries:
        r = requests.post(f"{SERVER}/api/route_multi",
                          headers=headers,
                          json={"query": q["query"], "top_k": 5})
        data = r.json()
        lat_us.append(data.get("latency_us", 0))
        ranked = [m["id"] for m in data.get("ranked", [])]
        expected = q["expected_id"]
        is_top1 = bool(ranked) and ranked[0] == expected
        is_top3 = expected in ranked[:3]
        top1 += 1 if is_top1 else 0
        top3 += 1 if is_top3 else 0
        results.append({"query": q["query"], "expected": expected, "got": ranked[:3], "top1": is_top1, "top3": is_top3})

    n = len(queries)
    elapsed = time.time() - t0
    avg_lat_us = sum(lat_us) / max(1, len(lat_us))
    return {
        "n": n,
        "top1_pct": top1 / n * 100,
        "top3_pct": top3 / n * 100,
        "avg_latency_us": avg_lat_us,
        "elapsed_s": elapsed,
        "results": results,
    }


def main():
    setup_namespace()
    queries = generate_test_set()
    print(f"\nRouting {len(queries)} held-out queries...")
    report = measure(queries)
    print(f"\n=== RESULTS ===")
    print(f"  n={report['n']}  top-1={report['top1_pct']:.1f}%  top-3={report['top3_pct']:.1f}%")
    print(f"  avg latency: {report['avg_latency_us']:.1f}µs")
    print(f"  elapsed (incl HTTP RTT): {report['elapsed_s']:.1f}s\n")

    # Misses
    misses = [r for r in report['results'] if not r['top1']]
    if misses:
        print(f"--- {len(misses)} top-1 misses ---")
        for m in misses:
            print(f"  Q: {m['query'][:70]}")
            print(f"    expected: {m['expected']}")
            print(f"    got:      {m['got']}")

    # Save
    (ROOT / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nFull results: {ROOT / 'results.json'}")


if __name__ == "__main__":
    main()
