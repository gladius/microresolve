---
title: Rust API Reference
description: Complete API reference for the MicroResolve Rust crate.
---

Full rustdoc is published at [docs.rs/microresolve](https://docs.rs/microresolve).

## Engine

The top-level struct. Create one per application; it manages all namespaces.

```rust
use microresolve::{Engine, EngineConfig};

let engine = Engine::new(EngineConfig {
    data_dir: Some("~/.local/share/microresolve".into()),
    default_threshold: 0.3,
    ..Default::default()
})?;
```

### Engine methods

| Method | Description |
|--------|-------------|
| `Engine::new(config)` | Create engine with config |
| `namespace(id)` | Get or create a namespace handle |
| `namespace_with(id, config)` | Get or create a namespace handle with custom config |
| `try_namespace(id)` | Get a handle if the namespace exists, or `None` |
| `has_namespace(id)` | Check whether a namespace exists |
| `namespaces()` | List all namespace IDs |
| `remove_namespace(id)` | Drop a namespace from memory (does not delete data on disk) |
| `reload_namespace(id)` | Reload a namespace from disk (after external changes) |
| `flush()` | Persist all dirty namespaces to disk |
| `effective_threshold(ns_id)` | Effective resolve threshold for a namespace (cascade: namespace → engine) |
| `effective_languages(ns_id)` | Effective language list for a namespace |
| `effective_llm_model(ns_id)` | Effective LLM model for a namespace, or `None` |
| `config()` | Return a reference to the `EngineConfig` |

## EngineConfig

```rust
pub struct EngineConfig {
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
| `resolve(query)` | Classify query; returns `Vec<Match>` |
| `resolve_with(query, opts)` | Classify with custom `ResolveOptions` |

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

## Match

```rust
pub struct Match {
    pub id: String,
    pub score: f32,
}
```

Returned by `resolve()` and `resolve_with()`. Sorted by score descending.

## ResolveOptions

```rust
pub struct ResolveOptions {
    pub threshold: f32,  // minimum score; default 0.3
    pub gap: f32,        // score ratio gap for multi-intent; default 1.5
}
```

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
