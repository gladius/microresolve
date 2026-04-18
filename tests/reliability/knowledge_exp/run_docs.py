#!/usr/bin/env python3
"""
Document-as-intents experiment.

Test the alternative hypothesis: instead of pre-formatted FAQs,
take a raw documentation page, split on headers/paragraphs, and
generate diverse questions from each *description paragraph*.

Pipeline:
  doc → sections (heuristic split on ## headers) →
  LLM generates 10 questions per section →
  register as context intents →
  measure routing accuracy on the same held-out test set as run_faq.py

Success = accuracy >= FAQ approach (52.5% top-1 baseline).
No new LLM provider needed — uses same cache/throttle infrastructure.
"""
import json
import os
import re
import time
import hashlib
import requests
from pathlib import Path

ROOT = Path(__file__).parent
SERVER = "http://localhost:3001"
NS = "docs-stripe"

# Reuse the same held-out test queries generated for the FAQ experiment
# (expected_id values match the section ids we'll assign below)
TEST_QUERIES_PATH = ROOT / "test_queries.json"

# Load .env
ENV = {}
for line in (Path(__file__).parent.parent.parent.parent / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        ENV[k.strip()] = v.strip()

LLM_URL = ENV["LLM_API_URL"]
LLM_KEY = ENV["LLM_API_KEY"]
LLM_MODEL = ENV["LLM_MODEL"]

CACHE_DIR = ROOT / "llm_cache"
CACHE_DIR.mkdir(exist_ok=True)


def call_llm(prompt: str, max_tokens: int = 1000) -> str:
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
        time.sleep(1.2)
        return content
    raise RuntimeError("LLM call failed after retries")


# ---------------------------------------------------------------------------
# Section IDs must match the expected_id values in test_queries.json
# (same IDs as the FAQ experiment so we can reuse held-out queries)
HEADER_TO_ID = {
    "Payout Schedule": "payout_schedule",
    "Failed Subscription Payments": "failed_payment_retry",
    "Processing Fee Refunds": "refund_fees",
    "Responding to Chargebacks and Disputes": "dispute_response",
    "International Credit Cards": "international_cards",
    "Setting Up Recurring Billing": "recurring_billing_setup",
    "Test Mode API Keys": "test_mode_keys",
    "Webhook Security": "webhook_security",
    "Sales Tax and VAT Collection": "tax_collection",
    "Charging in Local Currency": "multi_currency",
    "Fraud Prevention with Radar": "fraud_radar",
    "Account Verification": "account_verification",
    "Customer Self-Service Portal": "customer_portal",
    "Automatic Invoice Emails": "invoice_email",
    "Marketplace Split Payments with Connect": "connect_split_payments",
    "Refund Time Window": "refund_window",
    "Transaction Fees and Pricing": "pricing_fees",
    "Stripe Checkout vs Elements": "checkout_vs_elements",
    "Saving Cards for Future Charges": "saving_cards",
    "Payout Holds": "payout_holds",
}


def parse_sections(md_path: Path) -> list[dict]:
    """
    Split a markdown document on ## headers.
    Returns list of {id, header, body} dicts.
    """
    text = md_path.read_text()
    sections = []
    current_header = None
    current_lines = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_header is not None:
                body = "\n".join(current_lines).strip()
                if body:
                    sid = HEADER_TO_ID.get(current_header)
                    if sid:
                        sections.append({"id": sid, "header": current_header, "body": body})
            current_header = line[3:].strip()
            current_lines = []
        elif current_header is not None:
            current_lines.append(line)

    # Last section
    if current_header is not None:
        body = "\n".join(current_lines).strip()
        if body:
            sid = HEADER_TO_ID.get(current_header)
            if sid:
                sections.append({"id": sid, "header": current_header, "body": body})

    return sections


def generate_questions(section: dict, n: int = 10) -> list[str]:
    """
    Generate n diverse questions that a user might ask, given only
    the description paragraph — not a pre-formatted Q&A.
    """
    prompt = f"""You are building a chat-based support router. Below is a documentation paragraph.

Your task: write {n} DIFFERENT questions a real user might type in a support chat that this paragraph would answer.

Rules:
- Write how users TYPE: casual, abbreviated, sometimes frustrated or confused.
- Vary phrasing: mix short fragments ("payout delay"), implied questions ("my payment keeps failing"), indirect asks ("is there a way to..."), and full questions.
- Do NOT use the section title verbatim.
- Cover different angles the paragraph addresses — don't just rephrase the same question.
- One per line, no numbering, no bullet points, no quotes.

Section title: {section['header']}
Documentation text:
{section['body']}

Output exactly {n} lines:"""

    out = call_llm(prompt, max_tokens=600)
    lines = [l.strip().lstrip("-•*0123456789. )") for l in out.strip().split("\n")]
    lines = [l for l in lines if l and len(l) > 3]
    return lines[:n]


def setup_namespace(sections: list[dict]):
    requests.delete(f"{SERVER}/api/namespaces", json={"namespace_id": NS})
    time.sleep(0.2)
    r = requests.post(f"{SERVER}/api/namespaces",
                      json={"namespace_id": NS, "description": "Document-as-intents experiment"})
    print(f"Created namespace '{NS}': {r.status_code}")

    headers = {"X-Namespace-ID": NS}

    print(f"\nGenerating questions for {len(sections)} doc sections...")
    for i, section in enumerate(sections, 1):
        questions = generate_questions(section, n=10)
        cached = (CACHE_DIR / f"{hashlib.sha256((LLM_MODEL + section['id']).encode()).hexdigest()[:8]}.txt").exists()
        print(f"  [{i:2d}/{len(sections)}] {section['id']:30s} -> {len(questions)} questions {'(cached)' if cached else ''}")

        r = requests.post(f"{SERVER}/api/intents/multilingual",
                          headers=headers,
                          json={
                              "id": section['id'],
                              "intent_type": "context",
                              "phrases_by_lang": {"en": questions},
                              "description": section['header'],
                              "metadata": {"answer": [section['body']]}
                          })
        if r.status_code >= 400:
            print(f"       ERROR: {r.text[:200]}")


def measure(queries: list[dict]) -> dict:
    headers = {"X-Namespace-ID": NS}
    results = []
    lat_us = []
    top1 = top3 = 0

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
        results.append({"query": q["query"], "expected": expected, "got": ranked[:3],
                         "top1": is_top1, "top3": is_top3})

    n = len(queries)
    return {
        "n": n,
        "top1_pct": top1 / n * 100,
        "top3_pct": top3 / n * 100,
        "avg_latency_us": sum(lat_us) / max(1, len(lat_us)),
        "results": results,
    }


def main():
    doc_path = ROOT / "stripe_docs.md"
    sections = parse_sections(doc_path)
    print(f"Parsed {len(sections)} sections from {doc_path.name}")

    setup_namespace(sections)

    if not TEST_QUERIES_PATH.exists():
        print(f"\nERROR: {TEST_QUERIES_PATH} not found.")
        print("Run run_faq.py first to generate held-out test queries.")
        return

    queries = json.loads(TEST_QUERIES_PATH.read_text())
    print(f"\nRouting {len(queries)} held-out queries (same set as FAQ experiment)...")
    report = measure(queries)

    print(f"\n=== RESULTS (docs approach) ===")
    print(f"  n={report['n']}  top-1={report['top1_pct']:.1f}%  top-3={report['top3_pct']:.1f}%")
    print(f"  avg latency: {report['avg_latency_us']:.1f}µs")
    print(f"\n  FAQ baseline:  top-1=52.5%  top-3=62.5%")
    delta1 = report['top1_pct'] - 52.5
    delta3 = report['top3_pct'] - 62.5
    sign1 = "+" if delta1 >= 0 else ""
    sign3 = "+" if delta3 >= 0 else ""
    print(f"  Delta:         top-1={sign1}{delta1:.1f}%  top-3={sign3}{delta3:.1f}%")

    misses = [r for r in report['results'] if not r['top1']]
    if misses:
        print(f"\n--- {len(misses)} top-1 misses ---")
        for m in misses:
            print(f"  Q: {m['query'][:70]}")
            print(f"    expected: {m['expected']}")
            print(f"    got:      {m['got']}")

    out_path = ROOT / "results_docs.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
