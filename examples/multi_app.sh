#!/bin/bash
# Multi-app routing: isolate intents per bot/application.
#
# Each app has its own intent namespace, own learned weights, own persistence.
# Use X-App-ID header to select the app. No header = "default" app.

BASE="http://localhost:3001/api"

echo "=== Create two apps ==="
curl -s -X POST "$BASE/apps" -H 'Content-Type: application/json' -d '{"app_id":"support-bot"}'
echo
curl -s -X POST "$BASE/apps" -H 'Content-Type: application/json' -d '{"app_id":"sales-bot"}'
echo

echo -e "\n=== Add intents to support-bot ==="
curl -s -X POST "$BASE/intents" \
  -H 'Content-Type: application/json' \
  -H 'X-App-ID: support-bot' \
  -d '{"id":"cancel_order","seeds":["cancel my order","stop my order"]}'
echo
curl -s -X POST "$BASE/intents" \
  -H 'Content-Type: application/json' \
  -H 'X-App-ID: support-bot' \
  -d '{"id":"track_order","seeds":["where is my package","track my order"]}'
echo

echo -e "\n=== Add intents to sales-bot ==="
curl -s -X POST "$BASE/intents" \
  -H 'Content-Type: application/json' \
  -H 'X-App-ID: sales-bot' \
  -d '{"id":"pricing","seeds":["how much does it cost","what is the price"]}'
echo
curl -s -X POST "$BASE/intents" \
  -H 'Content-Type: application/json' \
  -H 'X-App-ID: sales-bot' \
  -d '{"id":"demo","seeds":["I want a demo","show me how it works"]}'
echo

echo -e "\n=== Route to support-bot ==="
curl -s -X POST "$BASE/route" \
  -H 'Content-Type: application/json' \
  -H 'X-App-ID: support-bot' \
  -d '{"query":"cancel my order please"}'
echo

echo -e "\n=== Same query to sales-bot (no match — different intents) ==="
curl -s -X POST "$BASE/route" \
  -H 'Content-Type: application/json' \
  -H 'X-App-ID: sales-bot' \
  -d '{"query":"cancel my order please"}'
echo

echo -e "\n=== Route to sales-bot ==="
curl -s -X POST "$BASE/route" \
  -H 'Content-Type: application/json' \
  -H 'X-App-ID: sales-bot' \
  -d '{"query":"how much is the enterprise plan"}'
echo

echo -e "\n=== List all apps ==="
curl -s "$BASE/apps"
echo
