---
title: Rust API Reference
description: Complete API reference for the MicroResolve Rust crate.
---

Full rustdoc is published at [docs.rs/microresolve](https://docs.rs/microresolve).

## MicroResolve

The top-level struct. Create one per application; it manages all namespaces.

```rust
use microresolve::{MicroResolve, MicroResolveConfig};

let engine = MicroResolve::new(MicroResolveConfig {
    data_dir: Some("~/.local/share/microresolve".into()),
    default_threshold: 0.3,
    ..Default::default()
})?;
```

### MicroResolve methods

| Method | Description |
|--------|-------------|
| `MicroResolve::new(config)` | Create engine with config |
| `namespace(id)` | Get or create a namespace handle |
| `namespace_with(id, config)` | Get or create a namespace handle with custom config |
| `try_namespace(id)` | Get a handle if the namespace exists, or `None` |
| `has_namespace(id)` | Check whether a namespace exists |
| `namespaces()` | List all namespace IDs |
| `remove_namespace(id)` | Drop a namespace from memory (does not delete data on disk) |
| `reload_namespace(id)` | Reload a namespace from disk (after external changes) |
| `flush()` | Persist all dirty namespaces to disk |
| `resolve_threshold_for(ns_id)` | Effective resolve threshold for a namespace (cascade: namespace → engine) |
| `languages_for(ns_id)` | Effective language list for a namespace |
| `llm_model_for(ns_id)` | Effective LLM model for a namespace, or `None` |
| `config()` | Return a reference to the `MicroResolveConfig` |

## MicroResolveConfig

```rust
pub struct MicroResolveConfig {
    pub data_dir: Option<PathBuf>,      // auto-load / auto-save location
    pub default_threshold: f32,         // default 0.3
    pub llm: Option<LlmConfig>,
    pub server: Option<ServerConfig>,   // for connected mode
}
```

## NamespaceHandle

Returned by `engine.namespace()`. Provides all classification and training operations for one namespace.

### Intent management

| Method | Description |
|--------|-------------|
| `add_intent(id, seeds)` | Create intent with seed phrases |
| `remove_intent(id)` | Delete intent and all its data |
| `intent(id)` | Get intent info (`Option<IntentInfo>`) |
| `update_intent(id, edit)` | Update intent metadata |
| `intent_ids()` | List all intent IDs |
| `intent_count()` | Number of registered intents |

### Phrase management

| Method | Description |
|--------|-------------|
| `add_phrase(intent_id, phrase, lang)` | Add phrase; returns `PhraseCheckResult` |

### Classification

| Method | Description |
|--------|-------------|
| `resolve(query)` | Classify query; returns `ResolveResult` |
| `resolve_with_trace(query)` | Classify and return `(ResolveResult, ResolveTrace)` |

### Learning

| Method | Description |
|--------|-------------|
| `correct(query, wrong, correct)` | Correct a misclassification |

### Namespace info

| Method | Description |
|--------|-------------|
| `id()` | Namespace ID |
| `version()` | Monotonically increasing mutation counter |
| `flush()` | Persist this namespace to disk |

## IntentSeeds

```rust
pub enum IntentSeeds {
    Phrases(Vec<String>),
}
```

## ResolveResult

```rust
pub struct ResolveResult {
    pub intents: Vec<IntentMatch>,
    pub disposition: Disposition,
}
```

Returned by `resolve()`. `intents` is sorted by score descending.

## IntentMatch

```rust
pub struct IntentMatch {
    pub id: String,
    pub score: f32,
    /// Normalized confidence in [0,1]: score / max_score_in_set.
    pub confidence: f32,
    pub band: Band,
}
```

## Band

```rust
pub enum Band { High, Medium, Low }
```

`High` means the score is at or above the namespace threshold. See [Bands and Disposition](/microresolve/concepts-bands/) for decision patterns.

## Disposition

```rust
pub enum Disposition { Confident, LowConfidence, NoMatch }
```

`Confident` if any intent has `Band::High`; `LowConfidence` if intents exist but none is `High`; `NoMatch` if the result is empty.

## ResolveTrace

```rust
pub struct ResolveTrace {
    pub tokens: Vec<String>,
    pub all_scores: Vec<(String, f32)>,
    pub multi_round_trace: MultiIntentTrace,
    pub negated: bool,
    pub threshold_applied: f32,
}
```

Returned alongside `ResolveResult` by `resolve_with_trace()`. Useful for debugging and threshold calibration.

## NamespaceConfig

```rust
pub struct NamespaceConfig {
    pub default_threshold: Option<f32>,
    // ...
}
```

Pass to `engine.namespace_with()` to override the engine-level default threshold for a specific namespace.

## Error

All fallible methods return `Result<_, microresolve::Error>`. The `Error` type implements `std::error::Error` and `Display`.
