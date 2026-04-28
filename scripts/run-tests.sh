#!/bin/bash
# Run all available test layers in order. Use this before tagging a release.
#
# Usage:
#   bash scripts/run-tests.sh

set -e
cd "$(dirname "$0")/.."

PASS=0; FAIL=0; SKIP=0
TICK="✓"; CROSS="✗"; DASH="—"

run() {
  local name="$1"; shift
  echo ""
  echo "═══ $name ═══"
  if "$@"; then
    echo "$TICK $name"
    PASS=$((PASS+1))
  else
    echo "$CROSS $name FAILED"
    FAIL=$((FAIL+1))
  fi
}

skip() {
  echo ""
  echo "═══ $1 (SKIPPED) ═══"
  echo "$DASH $1: $2"
  SKIP=$((SKIP+1))
}

# ── Rust ──────────────────────────────────────────────────────────────────────
run "lib unit tests (84 expected)" cargo test --lib --quiet 2>&1
run "server release build" cargo build --release --features server --quiet
run "HTTP integration: smoke" cargo test --release --test http_smoke --quiet
run "HTTP integration: full E2E (manual + MCP + auto-learn + auth)" cargo test --release --test http_full_e2e --quiet

# ── Connected example (live-sync loop) ────────────────────────────────────────
echo ""
echo "═══ connected example (live-sync loop) ═══"
rm -rf /tmp/microresolve_runtests
./target/release/server --port 3099 --no-open --data /tmp/microresolve_runtests > /tmp/runtests-server.log 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null" EXIT
sleep 1.5
if cargo run --release --features connect --example connected --quiet > /dev/null 2>&1; then
  echo "$TICK connected example"
  PASS=$((PASS+1))
else
  echo "$CROSS connected example FAILED"
  FAIL=$((FAIL+1))
fi
kill $SERVER_PID 2>/dev/null || true
trap - EXIT

# ── Bindings + UI build sanity ────────────────────────────────────────────────
[ -d ui ] && run "ui build" bash -c "cd ui && npm run build 2>&1 | tail -3" || skip "ui build" "no ui/"
[ -d python ] && run "python binding build" bash -c "cd python && cargo build --release --quiet 2>&1" || skip "python binding" "no python/"
[ -d node ] && run "node binding build" bash -c "cd node && cargo build --release --quiet 2>&1" || skip "node binding" "no node/"

# ── Skipped layers (deferred to v0.2) ─────────────────────────────────────────
skip "ui playwright" "manual run with vite + server (v0.2 wiring)"
skip "wasm binding" "no automated test harness yet (v0.2)"
skip "python e2e" "bindings not pip-installed in test env (v0.2)"
skip "node e2e" "bindings not npm-installed in test env (v0.2)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Result: $PASS passed, $FAIL failed, $SKIP skipped"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Coverage of feature surface: ~60-70%"
echo "  Gaps: bindings e2e, LLM-judge path, more UI pages, perf bench"
echo "  See tests/integration/README.md."
echo ""
[ $FAIL -gt 0 ] && exit 1 || exit 0
