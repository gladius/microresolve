# v0.2.1 Naming Audit

Captured post-rename (2026-05-02). Identifies methods/types/fields a *new contributor*
would find confusing. None of these are renamed here — this is a proposal list for the
next review session.

For each entry: current name, why it's confusing, proposed name, blast radius.

---

## 1. **STATUS: LANDED in #48** — `learn_query_words` (on `IntentIndex` and `NamespaceHandle`)

**Current:** `learn_query_words(words, intent_id)`

**Why confusing:** "learn" is overloaded. The method does online Hebbian reinforcement
of query-derived tokens toward an intent — not training from a phrase, not batch learning.
A new contributor reading this next to `add_intent` and `index_phrase` has no clear mental
model of when each applies.

**Proposed:** `reinforce_tokens(tokens, intent_id)` — matches the Hebbian terminology used
throughout the rest of the codebase comments, and the method's docstring.

**Blast radius:** 4 files (`src/scoring.rs`, `src/engine.rs`, `src/resolver_core.rs`,
`src/resolver_learning.rs`). Low risk — purely mechanical.

---

## 2. **STATUS: LANDED in #48** — `apply_review_local` (on `Resolver` and `NamespaceHandle`)

**Current:** `apply_review_local(...)`

**Why confusing:** "local" is opaque. It means "the deterministic, in-memory part of the
review flow that every binding can call, as opposed to the server-side LLM orchestration
in routes_review.rs." But a new reader doesn't know there's a server-side counterpart, so
"local" just sounds redundant.

**Proposed:** `apply_review(...)` — the "local" qualifier becomes unnecessary once the
server-side step is documented as a higher-level concept. If the distinction needs to be
preserved: `apply_review_sync(...)` to contrast with async LLM calls.

**Blast radius:** 3 files (`src/resolver_core.rs`, `src/engine.rs`,
`src/bin/server/routes_review.rs`). Low risk.

---

## 3. **STATUS: LANDED in chore/v0.2.1-audit-3-4-5** — `train_negative` (on `Resolver` and `NamespaceHandle`)

**Current:** `train_negative(queries, intents, alpha)`

**Why confusing:** "train" implies a full training pass. "negative" is ML jargon
(negative examples). Combined, a new reader might guess it blocks/deletes intents
rather than applying a gentle multiplicative decay. The actual behavior is anti-Hebbian
weight shrinkage, not a hard gate.

**Proposed:** `decay_for_intents(queries, intents, alpha)` — explicit that it's a
decay, not a block. Or `shrink_weights(...)` if "decay" sounds too academic.

**Blast radius:** 5 files. Medium risk — server test fixture checks response body
for `"trained":`. The field name in the JSON response would also want updating.

---

## 4. **STATUS: LANDED in chore/v0.2.1-audit-3-4-5** — `score_multi_traced` → `score_multi_with_trace` (on `IntentIndex`); `route_multi` split into `route_multi` / `route_multi_with_trace` (on `NamespaceHandle`)

**Current:** `score_multi_traced(query, threshold, gap, with_trace)`

**Why confusing:** Still uses "score" language even after the v0.2.1 rename of
higher-level methods. New contributors won't know what "traced" means in this context
without reading the body. The boolean `with_trace` parameter is also a code smell —
it changes the semantic content of the return value.

**Proposed:** Keep `score_multi_traced` as-is for now (it's a low-level `IntentIndex`
method, not public API). Consider splitting into `score_multi` (no trace, returns
`(Vec, bool)`) and `score_multi_with_trace` (always returns trace). This is a bigger
refactor — note for v0.3.

**Blast radius:** 3 files. Medium risk — changes return type.

---

## 5. **STATUS: LANDED in chore/v0.2.1-audit-3-4-5** — `effective_threshold` / `effective_languages` / `effective_llm_model` on `MicroResolve`

**Current:** Multiple `effective_*` getters in `NamespaceHandle` / config resolution.

**Why confusing:** "effective" is opaque to a new reader. It implies "after cascade
resolution" (override → namespace default → global default) but that cascade isn't
obvious from the name.

**Proposed:** Rename to `resolve_*` (matching the existing `resolve_threshold` pattern
already on `Resolver`) or `config_*`. Already partially done (`resolve_threshold`
exists). Remaining `effective_*` methods should be audited and renamed to match.

**Blast radius:** Moderate — affects `NamespaceHandle` public API and call sites in
`routes_*`. Worth landing in v0.2.2 as part of a config-API cleanup pass.

---

## 6. `index_phrase` vs `add_phrase` (on `Resolver`)

**Current:** Both `index_phrase(intent_id, phrase)` and `add_phrase(intent_id, phrase, lang)` exist.

**Why confusing:** New contributor sees two ways to add a phrase and doesn't know which
to call. `index_phrase` is lower-level (skips duplicate check, skips language routing)
while `add_phrase` is the checked path. The naming gives no hint of this distinction.

**Proposed:** Rename `index_phrase` to `index_phrase_raw` or `index_phrase_unchecked` to
signal that it bypasses the checked path. `add_phrase` stays as-is. Or elevate
`add_phrase` to be the only public method and make `index_phrase` `pub(crate)`.

**Blast radius:** Low — `index_phrase` is called from `engine.rs` and
`resolver_intents.rs`. The rename is mechanical.

---

## 7. `rebuild_idf` vs `rebuild_index`

**Current:** Both `rebuild_idf` (on `IntentIndex`) and `rebuild_index` (on `Resolver`,
renamed in this PR from `rebuild_l2`) exist at different levels.

**Why confusing:** `rebuild_idf` sounds like it only rebuilds IDF weights, but it also
rebuilds `known_intents`, `known_words`, `intent_to_tokens`, and `intent_count`. It's a
full in-memory cache rebuild. `rebuild_index` on `Resolver` is even broader — it
recreates the whole `IntentIndex` from raw training data.

**Proposed:**
- `rebuild_idf` → `rebuild_caches` or `refresh_index_caches` — reflects what it actually does.
- `rebuild_index` stays (the name is accurate at the `Resolver` level).

**Blast radius:** Low — `rebuild_idf` is called from engine.rs, resolver_persist.rs,
resolver_core.rs.

---

## 8. `disambiguate_cross_provider`

**Current:** `disambiguate_cross_provider(scored, query)`

**Why confusing:** Long, but mostly clear. "cross-provider" assumes the reader knows
about the namespace prefix convention (`stripe:list_customers`, `shopify:list_customers`).
Could be clearer that it only fires when the same action name appears under multiple
namespace prefixes.

**Proposed:** `deduplicate_by_provider(scored, query)` — shorter, clearer that it
removes duplicates rather than producing a new scoring. Or keep current name and just
add a clearer docstring.

**Blast radius:** Low — 2 call sites, 1 implementation.

---

## 9. `score_tokens` (private, on `IntentIndex`)

**Current:** Private helper — not in public API surface.

**Why confusing:** N/A for external users. Internal readability note: if this ever
becomes public or is referenced in docs, `score_token_set` would be clearer (it
operates on a set of tokens, not a single token).

**Proposed:** Defer. Private method, low blast radius if renamed, not user-visible.

---

## 10. `l2_unique_words` (API response key in `routes_import.rs`)

**Current:** JSON response field `l2_unique_words` in import endpoints.

**Why confusing:** After the v0.2.1 rename, this is the last remaining `l2_*` in
the API surface. External callers parsing this field will see "l2" as unexplained.

**Proposed:** Rename to `vocab_size` (matches the new `NamespaceHandle::vocab_size()`
method). This is a minor breaking API change to the import endpoint response — document
in CHANGELOG.

**Blast radius:** 2 occurrences in `routes_import.rs`. Easy rename; check if any client
code reads this field.

---

## Summary table

| # | Current | Proposed | Risk | Priority |
|---|---------|----------|------|----------|
| 1 | `learn_query_words` | `reinforce_tokens` | Low | Medium |
| 2 | `apply_review_local` | `apply_review` | Low | Medium |
| 3 | `train_negative` | `decay_for_intents` ✅ LANDED | Medium | Done |
| 4 | `score_multi_traced` | `score_multi_with_trace` ✅ LANDED | Medium | Done |
| 5 | `effective_*` → `resolve_*` | `resolve_threshold_for` / `languages_for` / `llm_model_for` ✅ LANDED | Medium | Done |
| 6 | `index_phrase` | `index_phrase_unchecked` | Low | Medium |
| 7 | `rebuild_idf` | `rebuild_caches` | Low | Low |
| 8 | `disambiguate_cross_provider` | `deduplicate_by_provider` | Low | Low |
| 9 | `score_tokens` (private) | defer | Low | Backlog |
| 10 | `l2_unique_words` (API) | `vocab_size` | Low | High — last `l2_*` in API |
