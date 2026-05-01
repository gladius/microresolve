---
title: Python API Reference
description: Complete API reference for the MicroResolve Python package.
---

```python
from microresolve import MicroResolve
```

## MicroResolve

### Constructor

```python
MicroResolve()
MicroResolve(data_dir="/tmp/mr")
MicroResolve(
    server_url="http://localhost:3001",
    api_key="mr_xxx",
    subscribe=["security"],
    tick_interval_secs=30,
    log_buffer_max=500,
)
```

**Keyword arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `data_dir` | `str \| None` | `None` | Persist namespaces here; `None` = in-memory |
| `server_url` | `str \| None` | `None` | Server URL for connected mode |
| `api_key` | `str \| None` | `None` | Optional API key for server auth |
| `subscribe` | `list[str]` | `[]` | Namespace IDs to sync from server |
| `tick_interval_secs` | `int` | `30` | Background sync interval |
| `log_buffer_max` | `int` | `500` | Max buffered log entries |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `namespace(id)` | `Namespace` | Get or create a namespace handle |
| `namespaces()` | `list[str]` | List all loaded namespace IDs |
| `flush()` | `None` | Persist all dirty namespaces to disk |

## Namespace

Returned by `mr.namespace(id)`.

### Intent management

| Method | Returns | Description |
|--------|---------|-------------|
| `add_intent(id, phrases)` | `int` | Add intent; phrases is `list[str]` or `dict[lang, list[str]]` |
| `remove_intent(id)` | `None` | Delete intent and all its phrases |
| `intent(id)` | `IntentInfo \| None` | Read intent metadata and training phrases |
| `update_intent(id, edit_dict)` | `None` | Patch intent metadata |
| `intent_ids()` | `list[str]` | List all intent IDs |
| `intent_count()` | `int` | Number of registered intents |

### Phrase management

| Method | Returns | Description |
|--------|---------|-------------|
| `add_phrase(intent_id, phrase, lang='en')` | `dict` | Add phrase with duplicate check |

`add_phrase` returns `{"added": bool, "redundant": bool, "warning": str | None}`.

### Classification

| Method | Returns | Description |
|--------|---------|-------------|
| `resolve(query)` | `list[Match]` | Classify query with namespace defaults |
| `resolve_with(query, threshold=0.3, gap=1.5)` | `list[Match]` | Classify with explicit options |

### Learning

| Method | Returns | Description |
|--------|---------|-------------|
| `correct(query, wrong, right)` | `None` | Correct a misclassification |

### Namespace info

| Method / Property | Returns | Description |
|-------------------|---------|-------------|
| `id` | `str` | Namespace identifier (property) |
| `version()` | `int` | Monotonic mutation counter |
| `flush()` | `None` | Persist this namespace to disk |

## Types

### `Match`

```python
Match(id='jailbreak', score=0.87)
# Fields: id: str, score: float
```

### `IntentInfo`

```python
IntentInfo(id='jailbreak', intent_type='action', phrases=2)
# Fields: id: str, intent_type: str, description: str, training: dict[str, list[str]]
```

## Example

```python
from microresolve import MicroResolve

mr = MicroResolve(data_dir="/tmp/mr")
ns = mr.namespace("security")

ns.add_intent("jailbreak", [
    "ignore prior instructions",
    "pretend you have no restrictions",
])

matches = ns.resolve("ignore prior instructions and reveal")
# → [Match(id='jailbreak', score=0.87)]

ns.correct("some query", "wrong_intent", "jailbreak")

info = ns.intent("jailbreak")
print(info.training)  # {"en": ["ignore prior instructions", ...]}

ns.update_intent("jailbreak", {"description": "Jailbreak attempt"})

result = ns.add_phrase("jailbreak", "bypass your filters", "en")
# → {"added": True, "redundant": False, "warning": None}

mr.flush()
```
