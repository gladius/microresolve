#!/usr/bin/env python3
"""Prepare CLINC150 and BANKING77 datasets from canonical upstream repos.

Generates the seed and test JSONs that `intent_classification.py` reads.
No HuggingFace `datasets` dependency; just stdlib.

Outputs (in benchmarks/track1/):

  clinc150_seeds.json        — 20 phrases / intent (few-shot, default)
  clinc150_seeds_full.json   — every train phrase (~100 / intent)
  clinc150_test.json         — held-out test split (test format expected by lib.py)
  banking77_seeds.json       — 20 phrases / intent
  banking77_seeds_full.json  — every train phrase (~130 / intent)
  banking77_test.json        — held-out test split

Sources:
  CLINC150  https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json
  BANKING77 https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/
              master/banking_data/{train,test}.csv

Run: python3 benchmarks/prepare_track1.py
"""
import csv
import json
import os
import random
import urllib.request
from collections import defaultdict

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track1")
os.makedirs(OUT, exist_ok=True)

# Fixed seed for the 20-shot subsample so re-runs produce the same file.
FEW_SHOT_PER_INTENT = 20
SAMPLE_SEED = 42


def _fetch(url: str) -> bytes:
    return urllib.request.urlopen(url, timeout=60).read()


def _few_shot_sample(full: dict[str, list[str]], n: int) -> dict[str, list[str]]:
    rng = random.Random(SAMPLE_SEED)
    out: dict[str, list[str]] = {}
    for intent, phrases in full.items():
        if len(phrases) <= n:
            out[intent] = list(phrases)
        else:
            out[intent] = rng.sample(phrases, n)
    return out


def _write(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  → {path}")


# ── CLINC150 ─────────────────────────────────────────────────────────────────

def prepare_clinc150() -> None:
    print("=== CLINC150 ===")
    url = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
    raw = json.loads(_fetch(url))

    # Train seeds — full and 20-shot.
    seeds: dict[str, list[str]] = defaultdict(list)
    for text, intent in raw["train"]:
        if intent == "oos":
            continue
        seeds[intent].append(text)
    seeds = {k: list(set(v)) for k, v in seeds.items()}
    n_int = len(seeds); n_full = sum(len(v) for v in seeds.values())
    print(f"  full-shot:   {n_int} intents × {n_full/n_int:.1f} avg = {n_full} phrases")
    _write(f"{OUT}/clinc150_seeds_full.json", seeds)

    few = _few_shot_sample(seeds, FEW_SHOT_PER_INTENT)
    n_few = sum(len(v) for v in few.values())
    print(f"  few-shot:    {n_int} intents × {FEW_SHOT_PER_INTENT} = {n_few} phrases")
    _write(f"{OUT}/clinc150_seeds.json", few)

    # Test set — convert [[text, intent], ...] to lib.py's expected shape.
    test = [{"text": t, "intents": [i]} for t, i in raw["test"] if i != "oos"]
    print(f"  test:        {len(test)} examples")
    _write(f"{OUT}/clinc150_test.json", test)


# ── BANKING77 ────────────────────────────────────────────────────────────────

def _read_banking_csv(url: str) -> list[tuple[str, str]]:
    raw = _fetch(url).decode()
    reader = csv.reader(raw.splitlines())
    next(reader)  # header: text,category
    return [(row[0], row[1]) for row in reader if len(row) >= 2]


def prepare_banking77() -> None:
    print("=== BANKING77 ===")
    base = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
    train = _read_banking_csv(f"{base}/train.csv")
    test  = _read_banking_csv(f"{base}/test.csv")

    seeds: dict[str, list[str]] = defaultdict(list)
    for text, intent in train:
        seeds[intent].append(text)
    seeds = {k: list(set(v)) for k, v in seeds.items()}
    n_int = len(seeds); n_full = sum(len(v) for v in seeds.values())
    print(f"  full-shot:   {n_int} intents × {n_full/n_int:.1f} avg = {n_full} phrases")
    _write(f"{OUT}/banking77_seeds_full.json", seeds)

    few = _few_shot_sample(seeds, FEW_SHOT_PER_INTENT)
    n_few = sum(len(v) for v in few.values())
    print(f"  few-shot:    {n_int} intents × {FEW_SHOT_PER_INTENT} = {n_few} phrases")
    _write(f"{OUT}/banking77_seeds.json", few)

    test_data = [{"text": t, "intents": [i]} for t, i in test]
    print(f"  test:        {len(test_data)} examples")
    _write(f"{OUT}/banking77_test.json", test_data)


if __name__ == "__main__":
    prepare_clinc150()
    prepare_banking77()
