#!/usr/bin/env python3
"""
Semantic Router vs MicroResolve — Head-to-head comparison on CLINC150 and BANKING77.

Replicates the Semantic Router approach (embedding similarity routing) using
fastembed (BAAI/bge-small-en-v1.5) with cosine similarity scoring.

This is functionally identical to Semantic Router's routing mechanism:
1. Embed seed phrases per intent
2. Embed query
3. Route to intent with highest cosine similarity above threshold

Usage:
    python benchmarks/semantic_router_comparison.py
    python benchmarks/semantic_router_comparison.py --dataset clinc150
    python benchmarks/semantic_router_comparison.py --dataset banking77
    python benchmarks/semantic_router_comparison.py --seeds 10
"""

import json
import csv
import time
import sys
import os
import argparse
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Embedding model (same as Semantic Router's default local encoder)
# ---------------------------------------------------------------------------

def load_embedding_model():
    """Load fastembed model — same model Semantic Router uses locally."""
    from fastembed import TextEmbedding
    print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    print("  Model loaded.")
    return model


def embed_batch(model, texts):
    """Embed a batch of texts. Returns numpy array of shape (N, dim)."""
    embeddings = list(model.embed(texts))
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# Dataset loading (matches MicroResolve benchmark format)
# ---------------------------------------------------------------------------

def load_clinc150(data_dir):
    path = os.path.join(data_dir, "clinc150.json")
    with open(path) as f:
        data = json.load(f)

    def parse_split(key):
        return [
            {"text": pair[0], "intent": pair[1]}
            for pair in data.get(key, [])
            if pair[1] != "oos"
        ]

    train = parse_split("train") + parse_split("val")
    test = parse_split("test")
    intents = sorted(set(ex["intent"] for ex in train))
    return {"name": "CLINC150", "train": train, "test": test, "intents": intents}


def load_banking77(data_dir):
    def parse_csv(filename):
        path = os.path.join(data_dir, filename)
        examples = []
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 2:
                    examples.append({"text": row[0], "intent": row[1].strip()})
        return examples

    train = parse_csv("banking77_train.csv")
    test = parse_csv("banking77_test.csv")
    intents = sorted(set(ex["intent"] for ex in train))
    return {"name": "BANKING77", "train": train, "test": test, "intents": intents}


# ---------------------------------------------------------------------------
# Semantic Router-style routing
# ---------------------------------------------------------------------------

class EmbeddingRouter:
    """Replicates Semantic Router's core routing mechanism."""

    def __init__(self, model):
        self.model = model
        self.intent_ids = []       # intent name per centroid
        self.centroids = None      # (num_intents, dim) array

    def add_intents(self, train_by_intent, seeds_per_intent):
        """Build intent centroids from seed phrases (same as Semantic Router)."""
        intent_ids = []
        centroid_list = []

        for intent_id in sorted(train_by_intent.keys()):
            seeds = train_by_intent[intent_id][:seeds_per_intent]
            if not seeds:
                continue

            # Embed seeds and average to get centroid
            seed_embeddings = embed_batch(self.model, seeds)
            centroid = seed_embeddings.mean(axis=0)
            # Normalize
            centroid = centroid / np.linalg.norm(centroid)

            intent_ids.append(intent_id)
            centroid_list.append(centroid)

        self.intent_ids = intent_ids
        self.centroids = np.array(centroid_list)

    def route(self, query_embedding):
        """Route a single query. Returns (intent_id, score) or (None, 0)."""
        # Cosine similarity (embeddings are normalized)
        scores = self.centroids @ query_embedding
        best_idx = np.argmax(scores)
        return self.intent_ids[best_idx], float(scores[best_idx])

    def route_top3(self, query_embedding):
        """Route and return top-3 intents."""
        scores = self.centroids @ query_embedding
        top3_idx = np.argsort(scores)[-3:][::-1]
        return [(self.intent_ids[i], float(scores[i])) for i in top3_idx]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(dataset, model, seeds_per_intent):
    print(f"\n{'='*70}")
    print(f"  {dataset['name']} — {len(dataset['intents'])} intents, "
          f"{len(dataset['test'])} test, {seeds_per_intent} seeds/intent")
    print(f"{'='*70}")

    # Group training data by intent
    train_by_intent = defaultdict(list)
    for ex in dataset["train"]:
        train_by_intent[ex["intent"]].append(ex["text"])

    # Build router
    router = EmbeddingRouter(model)
    print(f"  Building intent centroids ({seeds_per_intent} seeds each)...")
    router.add_intents(train_by_intent, seeds_per_intent)
    print(f"  {len(router.intent_ids)} intents indexed.")

    # Embed all test queries in batch
    print(f"  Embedding {len(dataset['test'])} test queries...")
    test_texts = [ex["text"] for ex in dataset["test"]]
    t0 = time.time()
    test_embeddings = embed_batch(model, test_texts)
    embed_time = time.time() - t0
    print(f"  Embedded in {embed_time:.1f}s ({embed_time/len(test_texts)*1000:.2f}ms/query)")

    # Normalize test embeddings
    norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    test_embeddings = test_embeddings / norms

    # Evaluate
    correct = 0
    top3_correct = 0
    latencies = []

    for i, ex in enumerate(dataset["test"]):
        query_emb = test_embeddings[i]

        t_start = time.perf_counter()
        pred_intent, score = router.route(query_emb)
        t_end = time.perf_counter()

        latencies.append((t_end - t_start) * 1e6)  # microseconds

        if pred_intent == ex["intent"]:
            correct += 1

        top3 = router.route_top3(query_emb)
        if any(intent == ex["intent"] for intent, _ in top3):
            top3_correct += 1

    total = len(dataset["test"])
    accuracy = correct / total
    top3_accuracy = top3_correct / total
    avg_latency = sum(latencies) / len(latencies)
    latencies.sort()
    p99_latency = latencies[int(len(latencies) * 0.99)]

    print(f"\n  Results ({seeds_per_intent} seeds/intent):")
    print(f"    Top-1 Accuracy:  {accuracy*100:.1f}%")
    print(f"    Top-3 Accuracy:  {top3_accuracy*100:.1f}%")
    print(f"    Avg latency:     {avg_latency:.1f}µs (routing only, excludes embedding)")
    print(f"    P99 latency:     {p99_latency:.1f}µs (routing only)")
    print(f"    Embedding time:  {embed_time/len(test_texts)*1000:.2f}ms/query")
    print(f"    Total per query: {embed_time/len(test_texts)*1000:.2f}ms (embedding) + {avg_latency/1000:.3f}ms (routing)")

    return {
        "dataset": dataset["name"],
        "seeds": seeds_per_intent,
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "avg_latency_us": avg_latency,
        "p99_latency_us": p99_latency,
        "embed_time_ms": embed_time / len(test_texts) * 1000,
    }


def run_sweep(dataset, model):
    """Run multiple seed counts for comparison with MicroResolve sweep."""
    seed_counts = [5, 10, 20, 50, 100, 120] if dataset["name"] == "CLINC150" else [5, 10, 20, 50, 100, 130]

    print(f"\n{'='*70}")
    print(f"  Sweep: {dataset['name']}")
    print(f"{'='*70}")

    # Group training data
    train_by_intent = defaultdict(list)
    for ex in dataset["train"]:
        train_by_intent[ex["intent"]].append(ex["text"])

    # Pre-embed all test queries once
    test_texts = [ex["text"] for ex in dataset["test"]]
    print(f"  Embedding {len(test_texts)} test queries (one-time)...")
    test_embeddings = embed_batch(model, test_texts)
    norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    test_embeddings = test_embeddings / norms

    results = []
    print(f"\n  {'Seeds':>6} │ {'Top-1':>8} {'Top-3':>8} {'Embed ms':>10} {'Route µs':>10}")
    print(f"  {'─'*6}─┼{'─'*40}")

    for seed_count in seed_counts:
        router = EmbeddingRouter(model)
        router.add_intents(train_by_intent, seed_count)

        correct = 0
        top3_correct = 0
        latencies = []

        for i, ex in enumerate(dataset["test"]):
            query_emb = test_embeddings[i]

            t_start = time.perf_counter()
            pred_intent, score = router.route(query_emb)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1e6)

            if pred_intent == ex["intent"]:
                correct += 1

            top3 = router.route_top3(query_emb)
            if any(intent == ex["intent"] for intent, _ in top3):
                top3_correct += 1

        total = len(dataset["test"])
        accuracy = correct / total
        top3_accuracy = top3_correct / total
        avg_latency = sum(latencies) / len(latencies)

        # Measure embedding latency (sample)
        sample = test_texts[:100]
        t0 = time.time()
        _ = embed_batch(model, sample)
        embed_ms = (time.time() - t0) / len(sample) * 1000

        print(f"  {seed_count:>6} │ {accuracy*100:>7.1f}% {top3_accuracy*100:>7.1f}% {embed_ms:>9.2f} {avg_latency:>9.1f}")

        results.append({
            "seeds": seed_count,
            "accuracy": accuracy,
            "top3_accuracy": top3_accuracy,
            "embed_ms": embed_ms,
            "route_us": avg_latency,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Semantic Router comparison benchmark")
    parser.add_argument("--dataset", default="all", choices=["clinc150", "banking77", "all"])
    parser.add_argument("--seeds", type=int, default=0, help="Seeds per intent (0=sweep)")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    model = load_embedding_model()

    datasets = []
    if args.dataset in ("clinc150", "all"):
        datasets.append(load_clinc150(args.data_dir))
    if args.dataset in ("banking77", "all"):
        datasets.append(load_banking77(args.data_dir))

    all_results = []

    for dataset in datasets:
        if args.seeds > 0:
            result = run_benchmark(dataset, model, args.seeds)
            all_results.append(result)
        else:
            results = run_sweep(dataset, model)
            all_results.extend(results)

    # Save results
    output_path = os.path.join(args.data_dir, "semantic_router_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
