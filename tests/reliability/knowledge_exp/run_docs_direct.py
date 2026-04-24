#!/usr/bin/env python3
"""
Direct content indexing — no LLM calls at index time.

Instead of: section → LLM generates questions → index questions
Do:         section → split into phrases → index phrases directly

Phrases come from:
  1. Full sentences from the section body
  2. Clause fragments (split on comma, dash, semicolon)
  3. Key noun-phrase windows (2-5 word sliding window over content tokens)

Zero LLM calls. Fully deterministic. Tests whether MicroResolve's inverted index
can match user queries against raw content tokens well enough.

Baseline comparison:
  FAQ approach (8 LLM-generated seeds):       top-1=52.5%  top-3=62.5%
  Docs approach (10 LLM-generated questions): top-1=50.0%  top-3=77.5%
"""
import json
import re
import time
import requests
from pathlib import Path

ROOT = Path(__file__).parent
SERVER = "http://localhost:3001"
NS = "docs-direct"
TEST_QUERIES_PATH = ROOT / "test_queries.json"

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
    text = md_path.read_text()
    sections = []
    current_header = None
    current_lines = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_header is not None:
                body = "\n".join(current_lines).strip()
                sid = HEADER_TO_ID.get(current_header)
                if sid and body:
                    sections.append({"id": sid, "header": current_header, "body": body})
            current_header = line[3:].strip()
            current_lines = []
        elif current_header is not None:
            current_lines.append(line)
    if current_header:
        body = "\n".join(current_lines).strip()
        sid = HEADER_TO_ID.get(current_header)
        if sid and body:
            sections.append({"id": sid, "header": current_header, "body": body})
    return sections


def extract_phrases(header: str, body: str) -> list[str]:
    """
    Extract indexable phrases from section content without any LLM.

    Strategy:
    1. The header itself (as a phrase)
    2. Full sentences
    3. Clause fragments (split on , ; — →)
    4. Sliding 3-5 word windows over content words (noun-phrase proxies)
    """
    phrases = []

    # 1. Header
    phrases.append(header.lower())

    # 2. Full sentences
    sentences = re.split(r'(?<=[.!?])\s+', body)
    for s in sentences:
        s = s.strip().rstrip('.')
        if len(s) > 8:
            phrases.append(s)

    # 3. Clause fragments (split on comma, dash, semicolon, arrow)
    fragments = re.split(r'[,;—→]', body)
    for f in fragments:
        f = f.strip().strip('.')
        # Skip navigation paths like "Dashboard → Balance → Payout settings"
        # (already captured above), keep meaningful clauses
        if 4 < len(f) < 120 and f not in phrases:
            phrases.append(f)

    # 4. Sliding word windows (3-5 words) — captures key noun phrases
    # Strip markdown/punctuation, keep meaningful words
    words = re.findall(r"[a-zA-Z0-9_%-]+(?:'[a-z]+)?", body.lower())
    stop = {'the','a','an','is','are','was','were','be','been','being',
            'to','of','in','on','at','for','with','by','from','as','or',
            'and','but','not','it','its','this','that','these','those',
            'you','your','we','our','they','their','i','my','can','will',
            'if','then','after','before','when','where','how','what','which'}
    content_words = [w for w in words if w not in stop and len(w) > 2]
    for size in (3, 4, 5):
        for i in range(len(content_words) - size + 1):
            window = " ".join(content_words[i:i+size])
            phrases.append(window)

    # Deduplicate preserving order, drop very short entries
    seen = set()
    result = []
    for p in phrases:
        p = p.strip()
        if p and len(p) > 3 and p not in seen:
            seen.add(p)
            result.append(p)

    return result


def setup_namespace(sections: list[dict]):
    requests.delete(f"{SERVER}/api/namespaces", json={"namespace_id": NS})
    time.sleep(0.2)
    r = requests.post(f"{SERVER}/api/namespaces",
                      json={"namespace_id": NS, "description": "Direct content indexing — no LLM"})
    print(f"Created namespace '{NS}': {r.status_code}")

    headers = {"X-Namespace-ID": NS}
    print(f"\nIndexing {len(sections)} sections (zero LLM calls)...")

    for i, section in enumerate(sections, 1):
        phrases = extract_phrases(section['header'], section['body'])
        print(f"  [{i:2d}/{len(sections)}] {section['id']:30s} -> {len(phrases)} phrases")

        r = requests.post(f"{SERVER}/api/intents/multilingual",
                          headers=headers,
                          json={
                              "id": section['id'],
                              "intent_type": "context",
                              "phrases_by_lang": {"en": phrases},
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
        results.append({"query": q["query"], "expected": expected,
                         "got": ranked[:3], "top1": is_top1, "top3": is_top3})

    n = len(queries)
    return {
        "n": n,
        "top1_pct": top1 / n * 100,
        "top3_pct": top3 / n * 100,
        "avg_latency_us": sum(lat_us) / max(1, len(lat_us)),
        "results": results,
    }


def main():
    sections = parse_sections(ROOT / "stripe_docs.md")
    print(f"Parsed {len(sections)} sections\n")

    # Show phrase breakdown for first section
    ex = sections[0]
    ex_phrases = extract_phrases(ex['header'], ex['body'])
    print(f"Example — '{ex['header']}':")
    for p in ex_phrases[:8]:
        print(f"  · {p}")
    print(f"  ... ({len(ex_phrases)} total)\n")

    setup_namespace(sections)

    queries = json.loads(TEST_QUERIES_PATH.read_text())
    print(f"\nRouting {len(queries)} held-out queries...")
    report = measure(queries)

    print(f"\n=== RESULTS ===")
    print(f"  approach        top-1    top-3")
    print(f"  FAQ (LLM×8)     52.5%    62.5%")
    print(f"  Docs (LLM×10)   50.0%    77.5%")
    d1, d3 = report['top1_pct'], report['top3_pct']
    print(f"  Direct (no LLM) {d1:.1f}%    {d3:.1f}%")

    misses = [r for r in report['results'] if not r['top1']]
    if misses:
        print(f"\n--- {len(misses)} top-1 misses ---")
        for m in misses:
            print(f"  Q: {m['query'][:70]}")
            print(f"    expected: {m['expected']}")
            print(f"    got:      {m['got']}")

    (ROOT / "results_direct.json").write_text(json.dumps(report, indent=2))
    print(f"\nFull results: {ROOT / 'results_direct.json'}")


if __name__ == "__main__":
    main()
