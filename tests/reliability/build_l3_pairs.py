#!/usr/bin/env python3
"""
Ask LLM to identify confusable intent pairs from the scale-test namespace.
Produces synthetic corrections that can be fed to /api/correct to seed L3 inhibition.

Output: l3_llm_corrections.json with (query, correct) pairs for each confusable pair.
"""

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "l3_llm_corrections.json"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

API_KEY = None
for line in open("/home/gladius/Workspace/reason-research/asv/.env"):
    if line.startswith("LLM_API_KEY="):
        API_KEY = line.strip().split("=", 1)[1]


def llm_confusable_pairs(intent_summaries: list) -> list:
    """Ask LLM which intent pairs look most confusable from vocabulary perspective."""
    # Batch-send all intents as a numbered list
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(intent_summaries))
    prompt = f"""Here is a list of {len(intent_summaries)} API intents from Stripe, Shopify, Linear, and Vercel:

{numbered}

Identify pairs of intents that use HIGHLY OVERLAPPING vocabulary and could be confused by a bag-of-words router. Focus on:
- Same action across providers (e.g., stripe:create_customer vs shopify:create_customer)
- Similar verbs (cancel, list, update) on different objects
- Verbs like "get" and "list" that may conflate retrieve-one vs retrieve-many

Return a JSON object with a single key "pairs", each pair is an array of two intent ids that are confusable:
{{"pairs": [["stripe:list_customers", "shopify:list_customers"], ...]}}

Include 30-60 of the most confusable pairs. Output only the JSON, no explanation."""

    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(GROQ_URL, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": "curl/8.15.0",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    text = resp["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # Find the JSON object
    start = text.find("{")
    end = text.rfind("}")
    data = json.loads(text[start:end+1])
    return data.get("pairs", [])


def main():
    # Fetch intents
    req = urllib.request.Request(
        "http://localhost:3001/api/intents",
        headers={"X-Namespace-ID": "scale-test"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        intents = json.loads(r.read())

    intent_ids = sorted(i["id"] for i in intents)
    summaries = []
    for iid in intent_ids:
        intent = next(i for i in intents if i["id"] == iid)
        phrases = intent.get("phrases", [])
        summary = f"{iid} — phrases: {phrases[:2]}"
        summaries.append(summary)

    print(f"Asking LLM to identify confusable pairs among {len(summaries)} intents...")
    pairs = llm_confusable_pairs(summaries)
    print(f"LLM returned {len(pairs)} pairs")

    # Validate: pairs must reference existing intent ids
    valid_pairs = []
    id_set = set(intent_ids)
    for p in pairs:
        if len(p) == 2 and p[0] in id_set and p[1] in id_set and p[0] != p[1]:
            valid_pairs.append(p)
    print(f"Valid pairs (both ids exist): {len(valid_pairs)}")

    # For each pair, create correction entries in BOTH directions so L3 learns
    # both "when A fires, suppress B" and "when B fires, suppress A".
    # We use a seed phrase from each intent as the query (route-strong).
    corrections = []
    for a, b in valid_pairs:
        a_intent = next((i for i in intents if i["id"] == a), None)
        b_intent = next((i for i in intents if i["id"] == b), None)
        if not a_intent or not b_intent:
            continue
        a_phrases = a_intent.get("phrases", [])
        b_phrases = b_intent.get("phrases", [])
        if a_phrases:
            corrections.append({"query": a_phrases[0], "correct": a})
        if b_phrases:
            corrections.append({"query": b_phrases[0], "correct": b})

    # Dedup on (query, correct)
    seen = set()
    unique = []
    for c in corrections:
        k = (c["query"], c["correct"])
        if k not in seen:
            seen.add(k)
            unique.append(c)

    OUT.write_text(json.dumps({
        "_meta": {"pairs_count": len(valid_pairs), "corrections_count": len(unique)},
        "pairs": valid_pairs,
        "corrections": unique,
    }, indent=2))
    print(f"Saved: {OUT}")
    print(f"Sample pairs: {valid_pairs[:5]}")


if __name__ == "__main__":
    main()
