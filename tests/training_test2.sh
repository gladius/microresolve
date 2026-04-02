#!/bin/bash
set -e
BASE="http://localhost:3001/api"

echo "=== Training Arena — Confirmed/Candidates Test ==="
echo ""

# Load defaults
curl -s -X POST "$BASE/intents/load_defaults" -H 'Content-Type: application/json' -d '{}' > /dev/null

# Generate
echo "Step 1: Generating scenario (polite/medium/short)..."
SCENARIO=$(curl -s -X POST "$BASE/training/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "personality": "polite",
    "sophistication": "medium",
    "verbosity": "short",
    "turns": 5,
    "scenario": "customer received wrong item, wants refund and to speak to a manager"
  }')
echo "$SCENARIO" > /tmp/training_scenario.json

python3 -c "
import json
data = json.load(open('/tmp/training_scenario.json'))
for i, t in enumerate(data['turns']):
    gt = ', '.join(t['ground_truth'])
    print(f'  Turn {i+1}: {t[\"customer_message\"][:90]}')
    print(f'    GT: {gt}')
"
echo ""

# Build turns
python3 -c "
import json
data = json.load(open('/tmp/training_scenario.json'))
turns = [{'message': t['customer_message'], 'ground_truth': t['ground_truth']} for t in data['turns']]
json.dump({'turns': turns}, open('/tmp/training_turns.json', 'w'))
"

# Cycle 1
echo "Step 2: Routing Cycle 1 (before training)..."
CYCLE1=$(curl -s -X POST "$BASE/training/run" -H 'Content-Type: application/json' -d @/tmp/training_turns.json)
echo "$CYCLE1" > /tmp/training_cycle1.json

python3 -c "
import json
data = json.load(open('/tmp/training_cycle1.json'))
print(f'  Pass: {data[\"pass_count\"]}/{data[\"total\"]} ({data[\"accuracy\"]*100:.0f}%)')
for i, r in enumerate(data['results']):
    s = r['status'].upper()
    confirmed = ', '.join(r.get('confirmed',[])) or 'none'
    candidates = ', '.join(r.get('candidates',[])) or 'none'
    missed = ', '.join(r.get('missed',[])) or '-'
    mark = '  ' if s == 'PASS' else '!!'
    print(f'  {mark} Turn {i+1} [{s}] confirmed=[{confirmed}] candidates=[{candidates}]')
    if r.get('missed'): print(f'       missed: {missed}')
"
echo ""

# Review failures
echo "Step 3: Reviewing failed turns..."
python3 -c "
import json, subprocess

cycle1 = json.load(open('/tmp/training_cycle1.json'))
all_corrections = []

for i, r in enumerate(cycle1['results']):
    if r['status'] == 'pass':
        continue
    if not r.get('missed'):
        print(f'  Turn {i+1}: no missed intents (only candidates as FP), skipping')
        continue

    # Only send confirmed detections to review — candidates are not 'detected'
    confirmed_details = [d for d in r.get('details', []) if d.get('confidence') != 'low']
    payload = json.dumps({
        'message': r['message'],
        'detected': confirmed_details,
        'ground_truth': r['ground_truth']
    })
    result = subprocess.run(
        ['curl', '-s', '-X', 'POST', '$BASE/training/review',
         '-H', 'Content-Type: application/json', '-d', payload],
        capture_output=True, text=True
    )
    review = json.loads(result.stdout)
    print(f'  Turn {i+1}: {review.get(\"analysis\", \"\")[:100]}')
    for c in review.get('corrections', []):
        print(f'    seed: \"{c.get(\"phrase\",\"\")}\" -> {c.get(\"intent\",\"\")}')
        all_corrections.append(c)

json.dump({'corrections': all_corrections}, open('/tmp/training_corrections.json', 'w'))
print(f'  Total: {len(all_corrections)} corrections')
"
echo ""

# Apply
echo "Step 4: Applying corrections..."
CORR_COUNT=$(python3 -c "import json; print(len(json.load(open('/tmp/training_corrections.json'))['corrections']))")
if [ "$CORR_COUNT" -gt 0 ]; then
    APPLY=$(curl -s -X POST "$BASE/training/apply" -H 'Content-Type: application/json' -d @/tmp/training_corrections.json)
    echo "  $APPLY"
else
    echo "  No corrections to apply"
fi
echo ""

# Cycle 2 — SAME turns
echo "Step 5: Routing Cycle 2 (same turns, after training)..."
CYCLE2=$(curl -s -X POST "$BASE/training/run" -H 'Content-Type: application/json' -d @/tmp/training_turns.json)
echo "$CYCLE2" > /tmp/training_cycle2.json

python3 -c "
import json
data = json.load(open('/tmp/training_cycle2.json'))
print(f'  Pass: {data[\"pass_count\"]}/{data[\"total\"]} ({data[\"accuracy\"]*100:.0f}%)')
for i, r in enumerate(data['results']):
    s = r['status'].upper()
    confirmed = ', '.join(r.get('confirmed',[])) or 'none'
    candidates = ', '.join(r.get('candidates',[])) or 'none'
    missed = ', '.join(r.get('missed',[])) or '-'
    mark = '  ' if s == 'PASS' else '!!'
    print(f'  {mark} Turn {i+1} [{s}] confirmed=[{confirmed}] candidates=[{candidates}]')
    if r.get('missed'): print(f'       missed: {missed}')
"
echo ""

# Summary
echo "=== Summary ==="
python3 -c "
import json
c1 = json.load(open('/tmp/training_cycle1.json'))
c2 = json.load(open('/tmp/training_cycle2.json'))
print(f'  Cycle 1: {c1[\"accuracy\"]*100:.0f}% ({c1[\"pass_count\"]}/{c1[\"total\"]})')
print(f'  Cycle 2: {c2[\"accuracy\"]*100:.0f}% ({c2[\"pass_count\"]}/{c2[\"total\"]})')
delta = (c2['accuracy'] - c1['accuracy']) * 100
print(f'  Delta: {delta:+.0f}%')
"

rm -f /tmp/training_scenario.json /tmp/training_turns.json /tmp/training_cycle1.json /tmp/training_cycle2.json /tmp/training_corrections.json
