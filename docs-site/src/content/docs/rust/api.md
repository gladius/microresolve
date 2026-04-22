---
title: Rust API Reference
description: Complete API reference for the MicroResolve Rust crate.
---

Full rustdoc is published at [docs.rs/microresolve](https://docs.rs/microresolve).

## Router

The main struct. Create one per namespace.

### Constructors

| Method | Description |
|--------|-------------|
| `Router::new()` | Create empty router |
| `Router::load(path)` | Load from JSON file |
| `Router::import_json(json)` | Deserialize from JSON string |

### Intent management

| Method | Description |
|--------|-------------|
| `add_intent(id, phrases)` | Create intent with seed phrases (English) |
| `add_intent_multilingual(id, phrases_by_lang)` | Create intent with phrases per language |
| `remove_intent(id)` | Delete intent and all data |
| `intent_ids()` | List all intent IDs |
| `intent_count()` | Number of registered intents |

### Phrase management

| Method | Description |
|--------|-------------|
| `add_phrase_checked(intent_id, phrase, lang)` | Add phrase with duplicate check |
| `check_phrase(intent_id, phrase)` | Check phrase without adding |
| `remove_phrase(intent_id, phrase)` | Remove a phrase |
| `get_training(intent_id)` | All phrases flat |
| `get_training_by_lang(intent_id)` | Phrases grouped by language |

### Resolution and learning

| Method | Description |
|--------|-------------|
| `resolve(query, threshold, gap)` | Resolve query to intents |
| `correct(query, wrong_intent, correct_intent)` | Move query to correct intent |

### Metadata

| Method | Description |
|--------|-------------|
| `set_description(id, desc)` / `get_description(id)` | Human-readable description |
| `set_intent_type(id, type)` / `get_intent_type(id)` | `Action` or `Context` |
| `set_instructions(id, text)` / `get_instructions(id)` | LLM instructions |
| `set_persona(id, text)` / `get_persona(id)` | LLM persona |
| `set_guardrails(id, rules)` / `get_guardrails(id)` | Guardrail rules |

### Persistence

| Method | Description |
|--------|-------------|
| `save(path)` | Save to JSON file |
| `export_json()` | Serialize to JSON string |
| `load_from_dir(path)` | Load namespace directory |
| `save_to_dir(path)` | Save namespace directory |

### Layer access (advanced)

| Method | Description |
|--------|-------------|
| `morphology()` / `morphology_mut()` | Morphology graph (L1) |
| `scoring()` / `scoring_mut()` | Scoring index (L2) |
| `typo_index()` | Typo corrector (L0) |
| `merge_l1_base(base)` | Merge global morphology edges |
| `rebuild_l0()` | Rebuild typo corrector from current vocab |
| `rebuild_l2()` | Rebuild scoring index from training phrases |
