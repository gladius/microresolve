# Entity Layer Plan

**Status:** PoC starting on `exp-entity-layer` branch.
**Owner:** in active development.
**Strategic stake:** if MicroResolve matches industry-standard PII / jailbreak / NER benchmarks at microsecond latency and zero per-call cost, it becomes the default lexical layer for any throughput- or cost-sensitive deployment — not an alternative, but the floor. This document is the plan to find out whether that's true.

---

## Why this matters

LLM-based and ML-based detection (Lakera Guard, Microsoft Prompt Shields, Microsoft Presidio, OpenAI Moderation) costs $0.0001–$0.001 per call and adds 50–500 ms of latency. They are excellent. They are also priced and scaled for cases where the alternative is "no detection at all."

A microsecond, $0-per-call lexical layer that **matches** those numbers on their own evaluation sets does two things:

1. **Removes the specialized-model tier from most architectures.** The expensive model becomes a verifier on borderline cases, not a first-pass filter. Total LLM cost drops 80–95% for most workloads.
2. **Repositions MicroResolve.** From "intent router with safety bonus" to "the universal lexical pre-filter for any text-classification task." That includes intent routing, but it also includes PII detection, jailbreak detection, content moderation, brand safety, compliance triggers — every problem currently solved by a small fine-tuned model.

The architectural philosophy stays intact: microsecond resolve, multipurpose, continuous learning, no LLM at inference.

---

## Three phases

### Phase 1 — PoC (this branch, ~2 hours)

**Goal:** answer "is the entity-layer concept worth investing in, and which detector implementation has the best signal-to-cost ratio?"

**Scope:**
- Build three entity detectors in `src/entity.rs` (or similar):
  1. **Regex** — handful of standard patterns (SSN, credit card, email, phone, IPv4)
  2. **Aho-Corasick** — fixed-string match for entity context words ("password is", "card number is", "my SSN")
  3. **Char n-gram classifier** — learned per entity type from LLM-generated examples (the "L1-distillation analog")
- Run all three against ~100 hand-crafted queries covering PII (with embedded entities), jailbreak (we have ~20), and benign (including PII-adjacent like "ticket number 4111-2222").
- Measure: detection rate, false positive rate, per-call latency, build cost, memory footprint.

**Success criteria for Phase 1:**
- At least one detector reaches >80% detection on the synthetic test set with <10% false positives.
- Latency budget: total routing call (entity layer + L0 + L1 + L2) stays under 200µs.
- We can articulate which approach to take into Phase 2 with a clear reason.

**Explicit non-goals for Phase 1:**
- Production-grade test coverage (see Phase 2).
- Integration with the routing pipeline by default — entity detection runs as a preprocessing pass we measure separately.
- International entity formats, multi-turn attacks, encoded payloads.

**Deliverables:**
- `src/entity.rs` (or per-implementation files) with the three detectors.
- A benchmark binary or test that runs the bake-off and prints results.
- A short writeup in `blogs/entity-layer-poc.md` documenting findings.
- A clear "go/no-go and which one" recommendation.

### Phase 2 — Industry-standard validation

**Goal:** prove or disprove that the winning detector matches established benchmarks.

**Benchmarks to target (in priority order):**

| Domain | Benchmark | Source | Size | What "winning" looks like |
|--------|-----------|--------|------|---------------------------|
| PII | Microsoft Presidio test suite | open source, in their repo | ~500 examples, 30+ entity types | F1 within 5pp of Presidio's published baseline |
| PII | spaCy NER test sets | spaCy GitHub | varies | competitive with `en_core_web_sm` on PERSON/ORG/GPE |
| Jailbreak | JailbreakBench | open, academic | ~250 attacks | detection rate within 5pp of Lakera Guard's public numbers |
| Jailbreak | HarmBench | open | ~400 attacks | detection rate >85% |
| General NER | CoNLL-2003 | classic benchmark | ~14K sentences | F1 >70% on PERSON, ORG, LOC, MISC |
| General NER | OntoNotes 5.0 | LDC | ~75K sentences | F1 >65% on the four major types |

**Success criteria for Phase 2:**
- On at least one benchmark in each domain, MicroResolve's detection rate is within 5 percentage points of the published baseline of the dominant tool.
- Per-call latency stays under 100 µs even with full entity layer enabled.
- False positive rate is comparable to or better than the baseline tool.

**If we win:** the blog story changes. Instead of "an interesting alternative," the headline becomes "MicroResolve matches Presidio at 1000× the speed." We make explicit, defensible benchmark comparisons in the docs.

**If we lose:** that's also a valid outcome. We document where we fall short, ship the entity layer as "directional signal for cheap" (still useful), and don't claim parity. The architecture and philosophy survive; the marketing scope narrows.

**Estimated effort:** several days of work to integrate the benchmarks, run them, debug discrepancies, and write up.

### Phase 3 — Production claims

**Only after Phase 2 do the public docs and blog posts make benchmark claims.** Until then, all writing about the entity layer is explicitly marked as PoC / directional.

This is the same discipline the jailbreak detection blog used: numbers are reported with the test-set size and source, and the limits are stated honestly.

---

## Open architectural decisions deferred to after Phase 1

These get answered based on what the PoC shows, not in advance:

- **Is the entity layer a real "L2" (renumbering pipeline)?** Decide once we know it ships. Renaming costs JSON migration, binding changes, doc updates. Worth it only if the layer is a permanent tier.
- **Default on or off per namespace?** Lean toward off, opt-in per namespace via a `entity_detection: ["pii", "jailbreak"]` config.
- **Char n-gram classifier as L1-distillation analog?** The PoC will tell us if statistical learning of patterns beats hand-coded regex meaningfully. If yes, this becomes a research thread; if no, regex wins for the entities that have stable formats and we don't bother.

---

## Anti-goals (what we will NOT do)

- We will not embed a 50MB BERT-based NER model. That violates the microsecond-resolve constraint.
- We will not call an LLM during routing. That violates the no-LLM-at-inference constraint.
- We will not build entity-extraction as a standalone product separate from intent routing. The point is that they share an engine.
- We will not claim parity with industry tools until we have benchmark data to back it up.

---

## Tracking

PoC progress and findings live on the `exp-entity-layer` branch. Industry validation results, when run, will be appended to this document under "Phase 2 results."
