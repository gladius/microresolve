#!/usr/bin/env python3
"""Download Track 1 intent classification datasets.

Downloads:
  - CLINC150  (HuggingFace: clinc_oos)
  - BANKING77 (HuggingFace: banking77)
  - HWU64     (HuggingFace: DH-arc/hwu64)
  - MASSIVE   (HuggingFace: AmazonScience/massive, English + 4 langs)

Outputs to: tests/data/benchmarks/track1/
  clinc150_seeds.json, clinc150_test.json
  banking77_seeds.json, banking77_test.json
  hwu64_seeds.json, hwu64_test.json
  massive_{lang}_seeds.json, massive_{lang}_test.json

Methodology:
  - Max 20 seed phrases per intent, from training split only
  - Single-intent examples only for seeding
  - No LLM, no human curation
"""

import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track1")
os.makedirs(OUT, exist_ok=True)

MAX_SEEDS = 20


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    → {path}")


def extract_seeds_and_test(train_rows, test_rows, text_key, label_key, max_seeds=MAX_SEEDS):
    """Generic extractor. Returns (seeds_dict, test_examples)."""
    seeds = {}
    for row in train_rows:
        intent = str(row[label_key])
        text = row[text_key]
        if intent not in seeds:
            seeds[intent] = []
        if len(seeds[intent]) < max_seeds:
            seeds[intent].append(text)

    test = []
    for row in test_rows:
        test.append({"text": row[text_key], "intents": [str(row[label_key])]})

    return seeds, test


# ── CLINC150 ─────────────────────────────────────────────────────────────────

def download_clinc150():
    from datasets import load_dataset
    print("=== CLINC150 ===")
    ds = load_dataset("clinc_oos", "plus")  # 'plus' includes out-of-scope

    # Map numeric labels to intent names
    label_names = ds["train"].features["intent"].names

    def rows(split):
        for row in ds[split]:
            label_idx = row["intent"]
            label = label_names[label_idx]
            if label == "oos":  # skip out-of-scope for now
                continue
            yield {"text": row["text"], "label": label}

    seeds = {}
    for row in rows("train"):
        intent = row["label"]
        if intent not in seeds:
            seeds[intent] = []
        if len(seeds[intent]) < MAX_SEEDS:
            seeds[intent].append(row["text"])

    test = [{"text": r["text"], "intents": [r["label"]]} for r in rows("test")]

    print(f"  Intents: {len(seeds)}, Seeds: {sum(len(v) for v in seeds.values())}, Test: {len(test)}")
    save(f"{OUT}/clinc150_seeds.json", seeds)
    save(f"{OUT}/clinc150_test.json", test)


# ── BANKING77 ────────────────────────────────────────────────────────────────

def download_banking77():
    from datasets import load_dataset
    print("=== BANKING77 ===")
    ds = load_dataset("banking77")

    label_names = ds["train"].features["label"].names

    def rows(split):
        for row in ds[split]:
            yield {"text": row["text"], "label": label_names[row["label"]]}

    seeds = {}
    for row in rows("train"):
        intent = row["label"]
        if intent not in seeds:
            seeds[intent] = []
        if len(seeds[intent]) < MAX_SEEDS:
            seeds[intent].append(row["text"])

    test = [{"text": r["text"], "intents": [r["label"]]} for r in rows("test")]

    print(f"  Intents: {len(seeds)}, Seeds: {sum(len(v) for v in seeds.values())}, Test: {len(test)}")
    save(f"{OUT}/banking77_seeds.json", seeds)
    save(f"{OUT}/banking77_test.json", test)


# ── HWU64 ────────────────────────────────────────────────────────────────────

def download_hwu64():
    from datasets import load_dataset
    print("=== HWU64 ===")
    # Try multiple known dataset ids
    ds = None
    for ds_id in ["FastFit/hwu_64", "DeepPavlov/hwu64", "DH-arc/hwu64"]:
        try:
            ds = load_dataset(ds_id)
            print(f"  Loaded from: {ds_id}")
            break
        except Exception as e:
            print(f"  {ds_id} failed: {e}")

    if ds is None:
        print("  HWU64 not available, skipping")
        return

    # Inspect columns
    sample = next(iter(ds["train"]))
    print(f"  Columns: {list(sample.keys())}")

    # Try to find text + label columns
    text_key = next((k for k in ["text", "utterance", "sentence"] if k in sample), None)
    label_key = next((k for k in ["label", "intent", "category"] if k in sample), None)

    if not text_key or not label_key:
        print(f"  Could not identify text/label columns: {list(sample.keys())}")
        return

    # Handle numeric labels
    features = ds["train"].features
    if hasattr(features[label_key], "names"):
        label_names = features[label_key].names
        get_label = lambda row: label_names[row[label_key]]
    else:
        get_label = lambda row: str(row[label_key])

    seeds = {}
    for row in ds["train"]:
        intent = get_label(row)
        text = row[text_key]
        if intent not in seeds:
            seeds[intent] = []
        if len(seeds[intent]) < MAX_SEEDS:
            seeds[intent].append(text)

    test_split = "test" if "test" in ds else "validation"
    test = [{"text": row[text_key], "intents": [get_label(row)]} for row in ds[test_split]]

    print(f"  Intents: {len(seeds)}, Seeds: {sum(len(v) for v in seeds.values())}, Test: {len(test)}")
    save(f"{OUT}/hwu64_seeds.json", seeds)
    save(f"{OUT}/hwu64_test.json", test)


# ── MASSIVE ──────────────────────────────────────────────────────────────────

# SetFit parquet mirrors of MASSIVE (no loading script required)
MASSIVE_LANGS = {
    "en_us": "SetFit/amazon_massive_intent_en-US",
    "es_es": "SetFit/amazon_massive_intent_es-ES",
    "fr_fr": "SetFit/amazon_massive_intent_fr-FR",
    "ja_jp": "SetFit/amazon_massive_intent_ja-JP",
    "ar_sa": "SetFit/amazon_massive_intent_ar-SA",
}

def download_massive():
    from datasets import load_dataset
    print("=== MASSIVE ===")

    for slug, ds_id in MASSIVE_LANGS.items():
        print(f"  Language: {slug} ({ds_id})")
        try:
            ds = load_dataset(ds_id)
        except Exception as e:
            print(f"    Failed: {e}")
            continue

        sample = next(iter(ds["train"]))
        # SetFit MASSIVE has 'text' and 'label' (string) or 'label_text'
        text_key = "text"
        label_key = "label_text" if "label_text" in sample else "label"

        features = ds["train"].features
        if hasattr(features.get(label_key), "names"):
            label_names = features[label_key].names
            get_label = lambda row: label_names[row[label_key]]
        else:
            get_label = lambda row: str(row[label_key])

        seeds = {}
        for row in ds["train"]:
            intent = get_label(row)
            text = row[text_key]
            if intent not in seeds:
                seeds[intent] = []
            if len(seeds[intent]) < MAX_SEEDS:
                seeds[intent].append(text)

        test_split = "test" if "test" in ds else "validation"
        test = [
            {"text": row[text_key], "intents": [get_label(row)]}
            for row in ds[test_split]
        ]

        print(f"    Intents: {len(seeds)}, Seeds: {sum(len(v) for v in seeds.values())}, Test: {len(test)}")
        save(f"{OUT}/massive_{slug}_seeds.json", seeds)
        save(f"{OUT}/massive_{slug}_test.json", test)


if __name__ == "__main__":
    download_clinc150()
    download_banking77()
    download_hwu64()
    download_massive()
    print("\n=== Track 1 datasets ready ===")
