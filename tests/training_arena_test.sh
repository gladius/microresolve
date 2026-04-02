#!/bin/bash
# Training Arena test script — runs one scenario through the full cycle
# Usage: bash tests/training_arena_test.sh

set -e
BASE="http://localhost:3001/api"

echo "=== Training Arena — End-to-End Test ==="
echo ""

# Step 0: Ensure defaults are loaded
echo "Loading defaults..."
curl -s -X POST "$BASE/intents/load_defaults" -H 'Content-Type: application/json' -d '{}' > /dev/null
INTENT_COUNT=$(curl -s "$BASE/intents" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
echo "  Intents loaded: $INTENT_COUNT"
echo ""

# Step 1: Generate a scenario
echo "Step 1: Generating scenario..."
echo "  Persona: frustrated, low sophistication, medium verbosity"
echo "  Scenario: customer received wrong item, wants refund"
SCENARIO=$(curl -s -X POST "$BASE/training/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "personality": "frustrated",
    "sophistication": "low",
    "verbosity": "medium",
    "turns": 5,
    "scenario": "customer received wrong item, wants refund and to speak to a manager"
  }')

# Check if generation worked
TURN_COUNT=$(echo "$SCENARIO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('turns',[])))")
echo "  Generated $TURN_COUNT turns"
echo ""

# Show generated messages
echo "  Generated conversation:"
echo "$SCENARIO" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for i, t in enumerate(data['turns']):
    gt = ', '.join(t['ground_truth'])
    print(f\"    Turn {i+1}: {t['customer_message'][:80]}...\")
    print(f\"      Ground truth: {gt}\")
"
echo ""

# Build the turns array for routing
TURNS=$(echo "$SCENARIO" | python3 -c "
import sys, json
data = json.load(sys.stdin)
turns = [{'message': t['customer_message'], 'ground_truth': t['ground_truth']} for t in data['turns']]
print(json.dumps({'turns': turns}))
")

# Step 2: Route (Cycle 1 — before training)
echo "Step 2: Routing Cycle 1 (before training)..."
CYCLE1=$(curl -s -X POST "$BASE/training/run" \
  -H 'Content-Type: application/json' \
  -d "$TURNS")

echo "$CYCLE1" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Accuracy: {data['pass_count']}/{data['total']} pass ({data['accuracy']*100:.0f}%)\")
for i, r in enumerate(data['results']):
    status = r['status'].upper()
    detected = ', '.join(r['detected']) or 'none'
    gt = ', '.join(r['ground_truth'])
    missed = ', '.join(r.get('missed',[])) or '-'
    extra = ', '.join(r.get('extra',[])) or '-'
    marker = '  ' if status == 'PASS' else '!!'
    print(f\"  {marker} Turn {i+1} [{status}]: detected=[{detected}] ground_truth=[{gt}]\")
    if r.get('missed'):
        print(f\"      Missed: {missed}\")
    if r.get('extra'):
        print(f\"      False pos: {extra}\")
"
echo ""

# Step 3: Review failures and collect corrections
echo "Step 3: LLM reviewing failures..."
ALL_CORRECTIONS="[]"
REVIEW_COUNT=0

# Process each failed turn
echo "$CYCLE1" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for i, r in enumerate(data['results']):
    if r['status'] != 'pass':
        print(json.dumps({'i': i, 'message': r['message'], 'detected': r.get('details',[]), 'ground_truth': r['ground_truth']}))
" | while read -r FAILED_TURN; do
    TURN_IDX=$(echo "$FAILED_TURN" | python3 -c "import sys,json; print(json.load(sys.stdin)['i'])")

    REVIEW=$(curl -s -X POST "$BASE/training/review" \
      -H 'Content-Type: application/json' \
      -d "$FAILED_TURN")

    echo "  Turn $((TURN_IDX + 1)):"
    echo "$REVIEW" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"    Analysis: {data.get('analysis','')[:100]}\")
corrections = data.get('corrections', [])
print(f\"    Corrections: {len(corrections)}\")
for c in corrections:
    action = c.get('action','?')
    if action == 'learn':
        print(f\"      learn: \\\"{c.get('query','')}\\\" -> {c.get('intent','')}\")
    elif action == 'correct':
        print(f\"      correct: \\\"{c.get('query','')}\\\" from {c.get('from','')} -> {c.get('intent','')}\")
    elif action == 'add_seed':
        print(f\"      add_seed: \\\"{c.get('phrase','')}\\\" -> {c.get('intent','')}\")
# Output corrections to file for collection
with open('/tmp/training_corrections.jsonl', 'a') as f:
    for c in corrections:
        f.write(json.dumps(c) + '\n')
"
done

echo ""

# Step 4: Apply corrections
echo "Step 4: Applying corrections..."
# Collect all corrections
CORRECTIONS=$(python3 -c "
import json
corrections = []
try:
    with open('/tmp/training_corrections.jsonl') as f:
        for line in f:
            line = line.strip()
            if line:
                corrections.append(json.loads(line))
except FileNotFoundError:
    pass
print(json.dumps({'corrections': corrections}))
")
CORR_COUNT=$(echo "$CORRECTIONS" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['corrections']))")
echo "  Corrections to apply: $CORR_COUNT"

if [ "$CORR_COUNT" -gt 0 ]; then
    APPLY_RESULT=$(curl -s -X POST "$BASE/training/apply" \
      -H 'Content-Type: application/json' \
      -d "$CORRECTIONS")
    echo "$APPLY_RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Applied: {data['applied']}\")
if data.get('errors'):
    for e in data['errors']:
        print(f\"  Error: {e}\")
"
fi
echo ""

# Step 5: Re-run same scenario (Cycle 2)
echo "Step 5: Routing Cycle 2 (after training)..."
CYCLE2=$(curl -s -X POST "$BASE/training/run" \
  -H 'Content-Type: application/json' \
  -d "$TURNS")

echo "$CYCLE2" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Accuracy: {data['pass_count']}/{data['total']} pass ({data['accuracy']*100:.0f}%)\")
for i, r in enumerate(data['results']):
    status = r['status'].upper()
    detected = ', '.join(r['detected']) or 'none'
    marker = '  ' if status == 'PASS' else '!!'
    print(f\"  {marker} Turn {i+1} [{status}]: detected=[{detected}]\")
"
echo ""

# Step 6: Summary
echo "=== Summary ==="
C1_ACC=$(echo "$CYCLE1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['accuracy']*100:.0f}\")")
C2_ACC=$(echo "$CYCLE2" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['accuracy']*100:.0f}\")")
echo "  Cycle 1: ${C1_ACC}% pass"
echo "  Corrections applied: $CORR_COUNT"
echo "  Cycle 2: ${C2_ACC}% pass"
echo "  Improvement: ${C1_ACC}% -> ${C2_ACC}%"
echo ""

# Cleanup
rm -f /tmp/training_corrections.jsonl
