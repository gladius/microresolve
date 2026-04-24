#!/usr/bin/env python3
"""
Hybrid: LLM question seeds + discriminative n-gram seeds.

Layer 1 (recall):     LLM generates 10 questions per section → bridges paraphrase gap
Layer 2 (precision):  Cross-section TF-IDF on content n-grams → adds discriminative terms

The discriminative layer finds n-grams that appear in FEW sections (unique vocabulary).
MicroResolve's inverted index already IDF-weights rare terms, so these score higher at routing time.

Zero additional LLM calls — reuses cached questions from run_docs.py.

Baselines:
  FAQ (LLM×8 seeds):           top-1=52.5%  top-3=62.5%
  Docs (LLM×10 questions):     top-1=50.0%  top-3=77.5%
  Direct content (no LLM):     top-1=20.0%  top-3=52.5%
"""
import json
import re
import math
import time
import hashlib
import requests
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).parent
SERVER = "http://localhost:3001"
NS = "docs-hybrid"
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

# LLM config (for cached calls only — no new calls needed if run_docs.py ran first)
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
            print(f"    {r.status_code}, waiting {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        cache_path.write_text(content)
        time.sleep(1.2)
        return content
    raise RuntimeError("LLM call failed after retries")


# ---------------------------------------------------------------------------
# Section parsing

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


# ---------------------------------------------------------------------------
# Layer 1: LLM questions (cached from run_docs.py)

def get_llm_questions(section: dict, n: int = 10) -> list[str]:
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


# ---------------------------------------------------------------------------
# Layer 2: Discriminative n-gram extraction (cross-section TF-IDF)

STOP = {
    'the','a','an','is','are','was','were','be','been','being','to','of',
    'in','on','at','for','with','by','from','as','or','and','but','not',
    'it','its','this','that','these','those','you','your','we','our','they',
    'their','i','my','can','will','if','then','after','before','when',
    'where','how','what','which','do','does','did','use','using','via',
    'into','up','out','about','more','also','each','all','both','per',
    'has','have','any','no','than','so',
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9_'-]*[a-z0-9]|[a-z0-9]", text.lower())
    return [w for w in words if w not in STOP and len(w) > 1]


def extract_ngrams(tokens: list[str], sizes=(1, 2, 3)) -> list[str]:
    ngrams = []
    for n in sizes:
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i:i+n]))
    return ngrams


def compute_discriminative_ngrams(sections: list[dict], top_k: int = 20) -> dict[str, list[str]]:
    """
    For each section, find n-grams with highest TF-IDF score across the section corpus.

    TF  = count of ngram in this section (normalized)
    IDF = log(N / df) where df = number of sections containing this ngram

    High TF-IDF = ngram appears often in this section AND rarely in others.
    That's our discriminative signal.
    """
    N = len(sections)

    # Build per-section ngram counts and global document frequency
    section_ngrams: list[Counter] = []
    df: Counter = Counter()  # how many sections contain each ngram

    for section in sections:
        tokens = tokenize(section['header'] + " " + section['body'])
        ngrams = extract_ngrams(tokens, sizes=(1, 2, 3))
        counts = Counter(ngrams)
        section_ngrams.append(counts)
        for ng in set(ngrams):
            df[ng] += 1

    # Score each ngram per section
    result: dict[str, list[str]] = {}
    for section, counts in zip(sections, section_ngrams):
        total = sum(counts.values()) or 1
        scored = []
        for ng, cnt in counts.items():
            tf = cnt / total
            idf = math.log(N / df[ng])
            # Only include ngrams that appear in ≤ 3 sections (discriminative)
            if df[ng] <= 3 and idf > 0:
                scored.append((tf * idf, ng))

        scored.sort(reverse=True)
        top = [ng for _, ng in scored[:top_k]]
        result[section['id']] = top

    return result


# ---------------------------------------------------------------------------

def setup_namespace(sections: list[dict], discriminative: dict[str, list[str]]):
    requests.delete(f"{SERVER}/api/namespaces", json={"namespace_id": NS})
    time.sleep(0.2)
    r = requests.post(f"{SERVER}/api/namespaces",
                      json={"namespace_id": NS, "description": "Hybrid: LLM questions + discriminative n-grams"})
    print(f"Created namespace '{NS}': {r.status_code}")

    headers = {"X-Namespace-ID": NS}
    print(f"\nBuilding hybrid index for {len(sections)} sections...")

    for i, section in enumerate(sections, 1):
        llm_qs = get_llm_questions(section, n=10)
        disc_ngrams = discriminative.get(section['id'], [])

        # Combine: LLM questions first (recall layer), discriminative n-grams appended (precision layer)
        all_phrases = llm_qs + disc_ngrams

        print(f"  [{i:2d}/{len(sections)}] {section['id']:30s} "
              f"LLM={len(llm_qs)} disc={len(disc_ngrams)} total={len(all_phrases)}")
        if disc_ngrams:
            print(f"              top disc: {disc_ngrams[:5]}")

        r = requests.post(f"{SERVER}/api/intents/multilingual",
                          headers=headers,
                          json={
                              "id": section['id'],
                              "intent_type": "context",
                              "phrases_by_lang": {"en": all_phrases},
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
    print(f"Parsed {len(sections)} sections")

    print("\nComputing discriminative n-grams (cross-section TF-IDF)...")
    discriminative = compute_discriminative_ngrams(sections, top_k=20)

    # Preview
    for sid, ngrams in list(discriminative.items())[:3]:
        print(f"  {sid}: {ngrams[:8]}")

    setup_namespace(sections, discriminative)

    queries = json.loads(TEST_QUERIES_PATH.read_text())
    print(f"\nRouting {len(queries)} held-out queries...")
    report = measure(queries)

    d1, d3 = report['top1_pct'], report['top3_pct']
    print(f"\n=== RESULTS ===")
    print(f"  approach                  top-1    top-3")
    print(f"  FAQ (LLM×8)               52.5%    62.5%")
    print(f"  Docs (LLM×10)             50.0%    77.5%")
    print(f"  Direct (no LLM)           20.0%    52.5%")
    print(f"  Hybrid (LLM+disc ngrams)  {d1:.1f}%    {d3:.1f}%  ← this run")

    misses = [r for r in report['results'] if not r['top1']]
    if misses:
        print(f"\n--- {len(misses)} top-1 misses ---")
        for m in misses:
            print(f"  Q: {m['query'][:70]}")
            print(f"    expected: {m['expected']}")
            print(f"    got:      {m['got']}")

    (ROOT / "results_hybrid.json").write_text(json.dumps(report, indent=2))
    print(f"\nFull results: {ROOT / 'results_hybrid.json'}")


if __name__ == "__main__":
    main()
