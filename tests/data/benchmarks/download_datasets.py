#!/usr/bin/env python3
"""Download and prepare benchmark datasets for MicroResolve evaluation.

Datasets:
  1. MixSNIPS — Multi-intent voice assistant (7 intents, 40K train / 2.2K test)
  2. MixATIS — Multi-intent airline travel (18 intents, 13K train / 828 test)
  3. Bitext — Customer support single-intent (27 intents, 27K examples)
  4. SGD — Schema-Guided Dialogue sequences (34 intents, 4.2K dialogues)

Requirements: pip install datasets

Methodology:
  - Seed phrases are extracted from each dataset's TRAINING split (single-intent examples only)
  - Up to 20 seed phrases per intent
  - Test evaluation uses the TEST split (which includes multi-intent examples)
  - No human curation or LLM-generated seeds — purely dataset-derived
  - For SGD: no routing tested, only workflow/temporal discovery from intent sequences
"""

import json
import os
import glob
import urllib.request

OUT = os.path.dirname(os.path.abspath(__file__))


def download_mixsnips():
    """MixSNIPS: Multi-intent variant of SNIPS (voice assistant domain)."""
    from datasets import load_dataset

    print("=== MixSNIPS ===")
    ds = load_dataset("nahyeon00/mixsnips_clean")

    # Test examples
    examples = []
    for row in ds["test"]:
        text = " ".join(row["token"])
        raw = row["intent"][0]
        intents = raw.split("#")
        examples.append({"text": text, "intents": intents})
    with open(f"{OUT}/mixsnips_test.json", "w") as f:
        json.dump(examples, f, indent=2)
    multi = sum(1 for e in examples if len(e["intents"]) > 1)
    print(f"  Test: {len(examples)} ({multi} multi-intent)")

    # Seeds from single-intent training examples
    all_intents = set()
    intent_phrases = {}
    for row in ds["train"]:
        raw = row["intent"][0]
        labels = raw.split("#")
        for i in labels:
            all_intents.add(i)
        if len(labels) == 1:
            intent = labels[0]
            if intent not in intent_phrases:
                intent_phrases[intent] = []
            if len(intent_phrases[intent]) < 20:
                intent_phrases[intent].append(" ".join(row["token"]))
    with open(f"{OUT}/mixsnips_seeds.json", "w") as f:
        json.dump(intent_phrases, f, indent=2)
    print(f"  Intents ({len(all_intents)}): {sorted(all_intents)}")
    print(f"  Seeds: {sum(len(v) for v in intent_phrases.values())} phrases")


def download_mixatis():
    """MixATIS: Multi-intent variant of ATIS (airline travel domain).

    Source: https://github.com/LooperXX/AGIF (EMNLP 2020)
    Format: word/tag per line, blank-line separated, last line = intent label
    """
    print("\n=== MixATIS ===")
    dest = f"{OUT}/mixatis_raw"
    os.makedirs(dest, exist_ok=True)

    # Clone the AGIF repo to get the data
    repo_dir = "/tmp/AGIF"
    if not os.path.exists(repo_dir):
        os.system(f"git clone --depth 1 https://github.com/LooperXX/AGIF.git {repo_dir}")

    def parse_agif_file(path):
        examples = []
        with open(path) as f:
            lines = f.readlines()
        current = []
        for line in lines:
            line = line.strip()
            if line == "":
                if current:
                    label = current[-1]
                    text = " ".join(t.split()[0] for t in current[:-1] if t.split())
                    intents = label.split("#")
                    examples.append({"text": text, "intents": intents})
                    current = []
            else:
                current.append(line)
        if current:
            label = current[-1]
            text = " ".join(t.split()[0] for t in current[:-1] if t.split())
            intents = label.split("#")
            examples.append({"text": text, "intents": intents})
        return examples

    test_ex = parse_agif_file(f"{repo_dir}/data/MixATIS_clean/test.txt")
    train_ex = parse_agif_file(f"{repo_dir}/data/MixATIS_clean/train.txt")

    with open(f"{OUT}/mixatis_test.json", "w") as f:
        json.dump(test_ex, f, indent=2)
    multi = sum(1 for e in test_ex if len(e["intents"]) > 1)
    all_intents = set()
    for e in test_ex + train_ex:
        for i in e["intents"]:
            all_intents.add(i)
    print(f"  Test: {len(test_ex)} ({multi} multi-intent)")
    print(f"  Intents ({len(all_intents)}): {sorted(all_intents)}")

    # Seeds from single-intent training examples
    intent_phrases = {}
    for e in train_ex:
        if len(e["intents"]) == 1:
            intent = e["intents"][0]
            if intent not in intent_phrases:
                intent_phrases[intent] = []
            if len(intent_phrases[intent]) < 20:
                intent_phrases[intent].append(e["text"])
    with open(f"{OUT}/mixatis_seeds.json", "w") as f:
        json.dump(intent_phrases, f, indent=2)
    print(f"  Seeds: {sum(len(v) for v in intent_phrases.values())} phrases across {len(intent_phrases)} intents")


def download_bitext():
    """Bitext: Customer support intent classification (27 intents)."""
    from datasets import load_dataset

    print("\n=== Bitext ===")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    examples = []
    intent_phrases = {}
    all_intents = set()
    for row in ds["train"]:
        intent = row["intent"]
        text = row["instruction"]
        all_intents.add(intent)
        examples.append({"text": text, "intents": [intent]})
        if intent not in intent_phrases:
            intent_phrases[intent] = []
        if len(intent_phrases[intent]) < 20:
            intent_phrases[intent].append(text)
    with open(f"{OUT}/bitext_all.json", "w") as f:
        json.dump(examples, f, indent=2)
    with open(f"{OUT}/bitext_seeds.json", "w") as f:
        json.dump(intent_phrases, f, indent=2)
    print(f"  Total: {len(examples)}")
    print(f"  Intents ({len(all_intents)}): {sorted(all_intents)}")


def download_sgd():
    """SGD: Schema-Guided Dialogue (Google, 34+ intents, 4.2K dialogues).

    Source: https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
    License: CC BY-SA 4.0
    """
    print("\n=== SGD ===")
    repo_dir = "/tmp/sgd"
    if not os.path.exists(repo_dir):
        os.system(f"git clone --depth 1 https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git {repo_dir}")

    dialogues = []
    all_intents = set()
    for split in ["test"]:
        split_dir = f"{repo_dir}/{split}"
        files = sorted(glob.glob(f"{split_dir}/dialogues_*.json"))
        for fpath in files:
            with open(fpath) as f:
                data = json.load(f)
            for dlg in data:
                seq = []
                for turn in dlg["turns"]:
                    if turn["speaker"] == "USER":
                        for frame in turn["frames"]:
                            active = frame.get("state", {}).get("active_intent", "NONE")
                            if active and active != "NONE":
                                if not seq or seq[-1] != active:
                                    seq.append(active)
                                    all_intents.add(active)
                if seq:
                    dialogues.append({"intent_sequence": seq})

    with open(f"{OUT}/sgd_dialogues.json", "w") as f:
        json.dump(dialogues, f, indent=2)
    multi = sum(1 for d in dialogues if len(d["intent_sequence"]) >= 2)
    print(f"  Dialogues: {len(dialogues)} ({multi} with 2+ intent transitions)")
    print(f"  Intents ({len(all_intents)}): {sorted(all_intents)[:20]}...")


if __name__ == "__main__":
    download_mixsnips()
    download_mixatis()
    download_bitext()
    download_sgd()
    print("\n=== All datasets ready! ===")
