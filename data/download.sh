#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== MicroResolve Router — Benchmark Dataset Download ==="
echo ""

# --------------------------------------------------------------------------
# CLINC150 (150 intents, ~23,700 queries)
# Source: https://github.com/clinc/oos-eval
# Format: JSON — { "train": [[text, intent], ...], "test": [...], ... }
# --------------------------------------------------------------------------
if [ -f "clinc150.json" ]; then
    echo "[CLINC150] Already downloaded."
else
    echo "[CLINC150] Downloading from GitHub (clinc/oos-eval)..."
    curl -sL -o clinc150.json \
        "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
    echo "[CLINC150] Done. $(wc -c < clinc150.json) bytes."
fi

# --------------------------------------------------------------------------
# BANKING77 (77 intents, 13,083 queries)
# Source: https://github.com/PolyAI-LDN/task-specific-datasets
# Format: CSV — text,category
# --------------------------------------------------------------------------
if [ -f "banking77_train.csv" ]; then
    echo "[BANKING77] Already downloaded."
else
    echo "[BANKING77] Downloading train set..."
    curl -sL -o banking77_train.csv \
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    echo "[BANKING77] Downloading test set..."
    curl -sL -o banking77_test.csv \
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
    echo "[BANKING77] Done. Train: $(wc -l < banking77_train.csv) lines, Test: $(wc -l < banking77_test.csv) lines."
fi

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo ""
echo "=== Dataset Summary ==="
echo ""

if [ -f "clinc150.json" ]; then
    echo "  CLINC150:  clinc150.json          ($(du -h clinc150.json | cut -f1))"
fi

if [ -f "banking77_train.csv" ]; then
    echo "  BANKING77: banking77_train.csv     ($(wc -l < banking77_train.csv) lines)"
    echo "             banking77_test.csv      ($(wc -l < banking77_test.csv) lines)"
fi

echo ""
echo "Run benchmarks with:"
echo "  cargo run --release --bin benchmark -- --dataset clinc150"
echo "  cargo run --release --bin benchmark -- --dataset banking77"
echo "  cargo run --release --bin benchmark -- --dataset all"
