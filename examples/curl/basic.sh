#!/bin/bash
# Basic ASV Router usage via HTTP API.
# Requires: server running on localhost:3001

BASE="http://localhost:3001/api"

echo "=== Add intents ==="
curl -s -X POST "$BASE/intents" -H 'Content-Type: application/json' \
  -d '{"id":"cancel_order","seeds":["cancel my order","I want to cancel","stop my order"]}'
echo
curl -s -X POST "$BASE/intents" -H 'Content-Type: application/json' \
  -d '{"id":"track_order","seeds":["where is my package","track my order","shipping status"]}'
echo
curl -s -X POST "$BASE/intents" -H 'Content-Type: application/json' \
  -d '{"id":"refund","seeds":["I want a refund","get my money back","return and refund"]}'
echo

echo -e "\n=== Single routing ==="
curl -s -X POST "$BASE/route" -H 'Content-Type: application/json' \
  -d '{"query":"I need to cancel something"}'
echo

echo -e "\n=== Multi-intent ==="
curl -s -X POST "$BASE/route_multi" -H 'Content-Type: application/json' \
  -d '{"query":"cancel my order and give me a refund","threshold":0.3}'
echo

echo -e "\n=== Learn ==="
curl -s -X POST "$BASE/learn" -H 'Content-Type: application/json' \
  -d '{"query":"stop charging me","intent_id":"cancel_order"}'
echo
curl -s -X POST "$BASE/route" -H 'Content-Type: application/json' \
  -d '{"query":"stop charging me"}'
echo

echo -e "\n=== Export ==="
curl -s "$BASE/export" | head -c 200
echo "..."

echo -e "\n=== Intents ==="
curl -s "$BASE/intents" | python3 -c "import sys,json; [print(f'  {i[\"id\"]} ({len(i[\"seeds\"])} seeds)') for i in json.load(sys.stdin)]" 2>/dev/null
echo
