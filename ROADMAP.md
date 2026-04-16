# ASV Router — Roadmap

Ideas under consideration for versions after initial public release.
Order is not priority.

## Language Packs — Decoupled Multilingual Expansion

Move language-specific lexical knowledge out of the core crate into a
dedicated companion repository (`asv-language-packs`).

**Shape:**
- One JSON file per language: `en.json`, `de.json`, `ja.json`, etc.
- Each pack contains: `{stops, synonyms, morphology, negations}`
- Generated via LLM distillation once, then community-curated via PR
- Versioned (`en-v1.json`, `en-v2.json`)
- Licensed permissively (e.g., CC0) — stricter than core if needed

**Router integration:**
- Core ships with `en.json` bundled (current hardcoded stopwords, reformatted)
- Load additional packs at runtime from `~/.local/share/asv/packs/` or URL
- Tokenizer takes resolved stop set as argument; caller composes from enabled
  languages per namespace
- No LLM dependency in core crate

**Why separate repo:**
- Core release cycle does not gate language additions
- Community contributions do not require Rust / routing knowledge — pure JSON
- Precedent: spaCy language models, tree-sitter grammars, hunspell dictionaries

**Namespace-level language selection:**
- `namespace.languages: ['en', 'ja']` — which packs to load per namespace
- Avoids bloating all namespaces with every enabled language
- Multi-tenant deployments can mix profiles (internal-EN, customer-facing-ES+EN,
  etc.) without per-app trade-offs

**Status:** designed, not implemented. See session on 2026-04-16.

---

## L1 Split — Shareable Lexical Pack vs Private Routing Weights

Observation: Layer 1 currently mixes two kinds of knowledge that have very
different sharing properties.

| Kind | Scope | Mergeable across namespaces? |
|------|-------|-------------------------------|
| Lexical (synonyms, morphology, stops, negation patterns) | Language-level | Yes — `refund` ≈ `money back` is an English fact |
| Routing weights (token → intent_id, Hebbian) | Namespace-level | No — `cancel` means different intents in different namespaces |

**Proposed split:**
- `LanguagePack` — mergeable, exportable, public artifact. Feeds the language
  packs repo above.
- `NamespaceRouter` — private, namespace-scoped. Not shared.

**Benefit:** community can publish and share lexical learnings (`english-pack-v3`)
with synonyms and morphology distilled from millions of corrections, without
exposing any user's intent data. Real network effect.

**Status:** architecture sketch, not built.

---

## Per-Namespace Learned Stopword Distillation

Instead of (or in addition to) per-language stops, allow an LLM call at
namespace creation to identify function-ish tokens specific to a domain's
seed phrases.

**Deferred because:** empirical test (2026-04-16, 125 intents × 12 seeds)
showed stopwords and common action verbs (`show`, `fetch`, `list`, `view`)
interleave in the frequency distribution — no statistical threshold separates
them. LLM distillation was the credible alternative, but threading namespace
context through the tokenizer is a non-trivial refactor. Shipped with
hardcoded universal list instead. Revisit post-launch.

---

## Theme Selection (UI)

Add light / dark / system theme selection in Settings. Current UI is dark-only
(zinc/violet palette hardcoded across all pages). Needs:
- Theme preference persisted in app settings
- Tailwind dark-mode classes or CSS variables across all pages
- System theme detection via `matchMedia('(prefers-color-scheme: dark)')`

Touches every page component. Ship dark-only for launch; revisit when
non-launch-critical work can resume.

---

## Reproducible Benchmark Suite

Port the ad-hoc reliability tests (`tests/reliability/*.py`) into a public
benchmark binary that:
- Downloads CLINC150 / BANKING77 automatically
- Reports top-1, top-3, multi-intent p@3, OOS rejection
- Runs under 30 seconds end-to-end
- Can be executed by anyone reproducing published numbers

Current state: ad-hoc Python scripts; Rust benchmark binary was a stub and
has been removed.
