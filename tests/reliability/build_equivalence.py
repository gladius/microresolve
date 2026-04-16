#!/usr/bin/env python3
"""
Build equivalence classes for each intent in a namespace via LLM distillation.

For each intent's seed phrases, asks the LLM to identify morphological variants
and clear synonyms of the distinctive words. Result is a variant→canonical map
used at query time to expand user vocabulary.

NOT paraphrase generation (that approach failed — LLM invented novel words).
This is word-level equivalence — LLM as dictionary, not generator.

Caches per-intent results to disk so re-runs don't burn Groq budget.

Usage:
    python3 tests/reliability/build_equivalence.py --namespace scale-test
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).parent
CACHE = ROOT / "equivalence_cache.json"
OUT = ROOT / "equivalence_classes.json"

BASE_URL = "http://localhost:3001"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Load Groq API key
API_KEY = None
for line in open("/home/gladius/Workspace/reason-research/asv/.env"):
    if line.startswith("LLM_API_KEY="):
        API_KEY = line.strip().split("=", 1)[1]
        break

STOPWORDS = set("a an and or the is are was were be been being have has had do does did "
                "i you he she it we they me him her us them my your his its our their "
                "this that these those to of in on at for with by from as is but "
                "please let can will would should could may might shall "
                "my me our".split())

SYSTEM_PROMPT = """You expand words into their morphological variants and direct synonyms for intent routing.

Given words from a SaaS API intent (Stripe/Shopify/Linear/Vercel), return a JSON object mapping each input word to a list of variants (morphological forms + near-synonyms).

Rules:
- Return ONLY morphological variants (verb tenses, plurals, gerunds) and CLEAR synonyms.
- Do NOT include unrelated words, no matter how plausible.
- If a word has no useful variants, return an empty list.
- Keep lists short (2-6 items max per word).
- Lowercase everything.
- Output ONLY the JSON, no commentary."""


def llm_call(words: list) -> dict:
    """Ask LLM for variants of these words. Returns word -> [variants]."""
    user_msg = "Expand these words:\n" + "\n".join(f"- {w}" for w in words)
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 800,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        GROQ_URL, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "curl/8.15.0",
        }
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                resp = json.loads(r.read())
            text = resp["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if any
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if "```" in text:
                    text = text.rsplit("```", 1)[0]
            text = text.strip()
            # Find JSON object boundaries
            if text.startswith("{"):
                json_str = text
            else:
                start = text.find("{")
                end = text.rfind("}")
                json_str = text[start:end+1] if start >= 0 and end > start else "{}"
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"    [parse error] {e}. Raw: {text[:200]}")
                return {}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 15 * (attempt + 1)
                print(f"    [429, waiting {wait}s]")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("LLM call failed after retries")


def tokenize(text: str) -> list:
    return [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]


def extract_words_for_intent(phrases: list) -> list:
    words = set()
    for p in phrases:
        for t in tokenize(p):
            if t not in STOPWORDS and len(t) > 2:
                words.add(t)
    return sorted(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="scale-test")
    args = parser.parse_args()

    # Load cache
    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
        print(f"Loaded cache with {len(cache)} intent entries")

    # Fetch intents
    req = urllib.request.Request(
        f"{BASE_URL}/api/intents",
        headers={"X-Namespace-ID": args.namespace},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        intents = json.loads(r.read())
    print(f"Namespace '{args.namespace}': {len(intents)} intents")

    # Count to-do
    todo = [i for i in intents if i["id"] not in cache]
    print(f"Already cached: {len(intents) - len(todo)}; to call LLM: {len(todo)}")
    if todo:
        print(f"Estimated time: ~{len(todo) * 2.8 / 60:.1f} min")

    for idx, intent in enumerate(todo, 1):
        intent_id = intent["id"]
        phrases = intent.get("phrases") or []
        words = extract_words_for_intent(phrases)
        if not words:
            cache[intent_id] = {}
            continue
        print(f"  [{idx}/{len(todo)}] {intent_id}: {len(words)} words → LLM")
        try:
            result = llm_call(words)
            # Sanity filter: keep only string lists with plausible content
            cleaned = {}
            for k, v in result.items():
                if not isinstance(v, list):
                    continue
                vs = [str(x).lower() for x in v if isinstance(x, str) and str(x).strip()]
                if vs:
                    cleaned[k.lower()] = vs[:6]
            cache[intent_id] = cleaned
        except Exception as e:
            print(f"    ERROR: {e}")
            cache[intent_id] = {}

        # Save progress every 10 intents
        if idx % 10 == 0:
            CACHE.write_text(json.dumps(cache, indent=2))
            print(f"    [saved progress: {len(cache)} intents in cache]")

        time.sleep(2.5)  # Groq Scout RPM limit

    # Save final cache
    CACHE.write_text(json.dumps(cache, indent=2))

    # Build flat variant -> [canonical_words] map (union across intents)
    flat: dict = {}
    for intent_id, mapping in cache.items():
        for canonical, variants in mapping.items():
            for v in variants:
                if v == canonical:
                    continue
                flat.setdefault(v, set()).add(canonical)
    flat_serializable = {k: sorted(list(v)) for k, v in flat.items()}
    OUT.write_text(json.dumps(flat_serializable, indent=2))

    print(f"\nDone.")
    print(f"  Intent cache: {CACHE}  ({len(cache)} entries)")
    print(f"  Flat variant->canonical map: {OUT}  ({len(flat_serializable)} variants)")
    # Show a few samples
    print("\nSample variants:")
    for i, (variant, canonicals) in enumerate(sorted(flat_serializable.items())[:20]):
        print(f"  {variant:20s} → {canonicals}")


if __name__ == "__main__":
    main()
