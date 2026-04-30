# Contributing to MicroResolve

Thank you for considering a contribution. MicroResolve is a solo-maintained
project — PRs, issues, and seed packs from real-world usage are how it
improves.

---

## Ways to contribute

- **Report a bug** — open an issue using the bug template.
- **Suggest a feature** — open an issue using the feature template; explain the
  use case, not just the implementation.
- **Share a seed pack** — domain-specific intent seeds (SaaS, MCP, e-commerce,
  gaming, etc.) that others can reuse. Open a seed-pack issue.
- **Run a benchmark** — results on your own data, public or private-summary.
  Non-English benchmarks especially welcome.
- **Fix a bug / implement a feature** — open an issue first for anything
  non-trivial so we can align on approach before you write code.

---

## Development setup

**Prerequisites:** Rust stable (`rustup default stable`), Python 3.11+, Node 20+.

```bash
git clone https://github.com/gladius/microresolve
cd microresolve

# Run Rust tests
cargo test --all-features

# Build the HTTP server
cargo build --release --bin microresolve-studio --features server

# Build + install the Python bindings locally
cd python
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release
cd ..

# Build the Node bindings
cd node
npm install
npx napi build --release
cd ..
```

---

## Pull request guidelines

1. **Open an issue first** for any change larger than a typo or a small bug
   fix. This saves both of us wasted work.
2. **One concern per PR.** A PR that fixes a bug and adds an unrelated feature
   is two PRs.
3. **Tests required** for new behaviour. Performance claims need a benchmark.
4. **CI must pass** — `cargo fmt --all -- --check`, `cargo clippy`, `cargo test`,
   Python + Node build smoke tests.
5. **Commit messages** — short imperative subject line (`fix: handle empty
   query`), body explains *why* if non-obvious. Conventional commits prefix
   (`feat:`, `fix:`, `perf:`, `docs:`, `refactor:`, `test:`, `chore:`) is
   preferred but not enforced.

---

## AI-assisted contributions

AI-assisted development is allowed and welcome. If an AI agent (Claude,
Copilot, Cursor, etc.) generated part of your contribution:

- **Verify the code runs and tests pass locally before opening the PR.** Treat
  AI output the same as code from a junior developer: review it.
- **Disclose AI assistance in the PR description** if substantial portions
  were generated. A one-line "Drafted with Claude, reviewed by me" is
  sufficient.
- **Do not submit AI-generated issues or PRs in bulk.** Spam PRs from automated
  agents will be closed without review.

---

## Licensing

By submitting a contribution, you agree it is licensed under the project's
dual MIT OR Apache-2.0 license. No CLA is required.

---

## Code of conduct

Be respectful, be specific, assume good faith. This is a side project, not a
full-time support line — tone and scope matter. Personal attacks, harassment,
or derailing issues/PRs will result in a block.

---

## Questions

- Issues and discussions on [GitHub](https://github.com/gladius/microresolve/issues).
- For consulting and custom integration work, email
  [gladius.thayalarajan@gmail.com](mailto:gladius.thayalarajan@gmail.com).
