# Integration tests

Black-box HTTP tests are now Rust-native and live in `tests/`:

```bash
cargo build --release --features server   # build server binary first
cargo test --release --test http_smoke    # 1 grouped test, ~5 sec
cargo test --release --test http_full_e2e # 4 grouped tests, ~10 sec
```

Each test spawns a fresh server on a random port with a clean tmp data dir,
runs assertions, then kills + cleans up. No external server needed; runs
side-by-side without port conflicts. Wired into `.github/workflows/ci.yml`.

## What each test covers

### `tests/http_smoke.rs` — `full_smoke` test
Wire-format smoke for the most-used endpoints:
- Namespace CRUD
- Intent CRUD (mono + multilingual via single `/api/intents`)
- Patch via `update_intent` flow
- Routing (English + French)
- Phrase add
- Namespace metadata patch
- `train_negative` + `rebuild`
- Layer info
- Delete

### `tests/http_full_e2e.rs` — 4 tests
- `manual_creation` — multilingual seeds + metadata patch + cross-language routing
- `mcp_import_three_tools` — full MCP import flow (parse + apply + intent_type derivation + schema preservation)
- `auto_learn_deterministic_path` — train_negative + rebuild_index + post-rebuild routing
- `auth_keys_endpoint` — key CRUD + auth enforcement (401 without key, 200 with)

## Other test layers

- `cargo test --lib` — 84 unit tests inside `src/**/*.rs`
- `cargo run --example connected --features connect` — end-to-end live-sync demo (also in CI)
- `ui/e2e/intents.spec.ts` — Playwright browser-driven UI test (manual run)
- `scripts/run-tests.sh` — wraps all of the above + builds in one command
