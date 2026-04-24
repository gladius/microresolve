# MicroResolve — 2-Day Open Source Launch Plan

Target: ship **MicroResolve** as a standalone Rust library with honest positioning and a minimal Python binding (if time permits). This plan covers MicroResolve only. Intent Programming / Prosessor launches separately on its own track.

## Positioning

**Tagline candidates:**
- *A fast, learning intent router in Rust. 30µs per route. Zero LLM calls.*
- *Your classifier layer, built for scale.*
- *The prefilter that makes LLM routing cheap.*

**Who MicroResolve is for:**
- Teams paying per-token for intent classification
- Multi-agent systems needing a fast dispatcher
- Command palettes, in-app help search, voice assistant frontends
- Anyone building MCP tool selection over 100+ tools
- Enterprise classifier pipelines that want sub-10ms routing

**Who MicroResolve is NOT for (say this plainly):**
- People expecting frontier-LLM semantic understanding out of the box
- Cold-start consumer agents with no training data
- Domains with highly abstract or poetic queries

**Honest limitations section in the README:**
- Vocabulary-overlap can produce false positives ("book three bath sessions" may match a `book_appointment` intent when user meant pricing)
- Accuracy scales with phrase coverage — sparse seeds yield ~60-80% accuracy on adversarial queries
- No semantic understanding of out-of-vocabulary terms without training
- Works best as a prefilter (top-K with LLM disambiguation) rather than top-1 autonomous routing

## Day 1 (~10-12 focused hours)

### Morning — Code cleanup for library publish (3-4h)
- [ ] Audit `src/lib.rs` — keep only router, tokenizer, scoring, index, auto-learn exports
- [ ] Move LP-specific code (`execute.rs`, handoff logic) out of the published surface. Either gate behind a feature flag or move to the server binary only.
- [ ] Verify `cargo doc --no-deps --open` produces a clean public API
- [ ] Run `cargo publish --dry-run` and fix all warnings
- [ ] Update `Cargo.toml` metadata: description, repository, license, keywords, categories

### Afternoon — README + positioning (3-4h)
- [ ] Hero section: one-line pitch + 30µs benchmark headline
- [ ] "Who this is for" with 5 concrete use cases
- [ ] Quick start (install → create router → add intents → route)
- [ ] Honest limitations section (see positioning above)
- [ ] Comparison table: MicroResolve vs Rasa vs LLM classification vs embedding search (latency, cost, accuracy, cold-start)
- [ ] Link to benchmarks

### Evening — Reproducible benchmarks (3-4h)
- [ ] Automate CLINC150 and BANKING77 eval (code likely exists in `src/bin/benchmark.rs`)
- [ ] Publish numbers + methodology in `BENCHMARKS.md`
- [ ] Include a clear "bring your own data" eval harness so users can test on their domain

## Day 2 (~8-12 focused hours)

### Morning — Python binding via pyo3 (4-6h) — optional if tight
- [ ] Wrap `Router` with pyo3: `route`, `add_intent_multilingual`, `add_phrase`, `correct`
- [ ] PyPI scaffolding with `maturin`
- [ ] One example notebook on CLINC150
- [ ] **If tight: skip this day-2-morning item; announce "Python binding coming" in README**

### Afternoon — Launch artifacts (2-3h)
- [ ] Blog post draft: "MicroResolve — a fast, learning intent router"
- [ ] Twitter/X thread (8-10 tweets with code snippets)
- [ ] HN submission (title + comment with context)
- [ ] Reddit posts: r/rust, r/MachineLearning, r/LocalLLaMA
- [ ] `CONTRIBUTING.md` + issue templates (bug, feature, seed-pack contribution)

### Evening — Ship (1-2h)
- [ ] `cargo publish` to crates.io
- [ ] GitHub release with v0.1.0 tag + changelog
- [ ] Post launch thread / HN / Reddit
- [ ] Monitor, respond to first wave of questions

## What Claude Code accelerates

- README drafting (high-quality first pass in minutes)
- Benchmark harness code (pattern-match from `tests/`)
- Python binding boilerplate (pyo3 has conventions to follow)
- Launch copy in multiple registers (technical, marketing, HN-style)
- Migration guide snippets (MicroResolve vs Rasa, MicroResolve vs sklearn classifiers)

## Time-boxing discipline

**If tight, drop in order:**
1. Python binding (save for post-launch)
2. Migration guides (write as responses to actual user questions)
3. MCP import walkthrough (it works; document later)
4. Issue templates beyond the basics

**Do NOT drop:**
- Honest limitations section in README
- Reproducible benchmarks
- Clear "who this is for"
- `cargo publish` to crates.io

## Post-launch expectations (realistic)

- **Week 1:** 100-500 crates.io downloads, 20-100 stars, 2-5 issues
- **Week 4:** traction compounds if positioning is right; community contributes seed packs and benchmark submissions
- **Month 3:** adoption by a handful of teams for classifier use cases
- **Month 6+:** decide whether to do a Python package proper, an HTTP server distribution, or a dedicated funding/support model

Community will NOT fix the core algorithm. They WILL:
- Contribute seed packs for specific domains
- Submit benchmark results from their own data
- Write integrations (Langchain adapter, CrewAI router, etc.)
- Open bug reports with clear reproductions
- Improve docs and examples

## Separation from Intent Programming / Prosessor

This launch is MicroResolve ONLY. Language Programming / Prosessor is a separate product on a separate track. Do not bundle. Do not cross-promote heavily. Users who adopt MicroResolve for classifier work become a natural pipeline for Prosessor later — but don't force the connection at launch.
