---
title: Python API Reference
description: Complete API reference for the MicroResolve Python package.
---

## Router

```python
from microresolve import Router
```

### Constructor

```python
Router()                        # new empty router
Router.import_json(json: str)   # load from JSON string
```

### Intent management

| Method | Description |
|--------|-------------|
| `add_intent(id, phrases)` | Create intent with seed phrases |
| `add_intent_multilingual(id, phrases_by_lang)` | Create intent with phrases per language |
| `delete_intent(id)` | Delete intent |
| `intent_ids()` | List all intent IDs |

### Phrase management

| Method | Returns | Description |
|--------|---------|-------------|
| `add_phrase(intent_id, phrase, lang='en')` | `dict` | Add with duplicate check |
| `check_phrase(intent_id, phrase)` | `dict` | Check without adding |
| `remove_phrase(intent_id, phrase)` | `bool` | Remove a phrase |

**`add_phrase` return dict:**
```python
{
  "added": bool,
  "new_terms": list[str],
  "redundant": bool,
  "warning": str | None,
}
```

### Resolution and learning

| Method | Returns | Description |
|--------|---------|-------------|
| `resolve(query, threshold=0.3, gap=1.5)` | `list[dict]` | Resolve query to intents |
| `correct(query, wrong_intent, correct_intent)` | `None` | Move to correct intent |

**`resolve` return list:**
```python
[{"id": str, "score": float}, ...]  # sorted by score descending
```

### Metadata

| Method | Description |
|--------|-------------|
| `set_description(id, desc)` / `get_description(id)` | Human-readable description |
| `set_intent_type(id, type)` | `"action"` or `"context"` |

### Persistence

| Method | Description |
|--------|-------------|
| `export_json()` | Serialize to JSON string |
| `Router.import_json(json)` | Deserialize from JSON string |
