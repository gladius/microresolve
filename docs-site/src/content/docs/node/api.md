---
title: Node.js API Reference
description: Complete API reference for the MicroResolve Node.js package.
---

```typescript
import { Router, ResolveMatch, PhraseResult } from 'microresolve';
```

## Router

### Constructor

```typescript
new Router()
Router.importJson(json: string): Router
```

### Intent management

| Method | Description |
|--------|-------------|
| `addIntent(id, phrases)` | Create intent with seed phrases |
| `deleteIntent(id)` | Delete intent |
| `intentIds()` | List all intent IDs |

### Phrase management

| Method | Returns | Description |
|--------|---------|-------------|
| `addPhrase(intentId, phrase, lang?)` | `PhraseResult` | Add with duplicate check |
| `removePhrase(intentId, phrase)` | `boolean` | Remove a phrase |

### Resolution and learning

| Method | Returns | Description |
|--------|---------|-------------|
| `resolve(query, threshold?, gap?)` | `ResolveMatch[]` | Resolve query to intents |
| `correct(query, wrongIntent, correctIntent)` | `void` | Move to correct intent |

### Metadata

| Method | Description |
|--------|-------------|
| `setDescription(id, desc)` / `getDescription(id)` | Human-readable description |
| `setIntentType(id, type)` | `"action"` or `"context"` |

### Persistence

| Method | Description |
|--------|-------------|
| `exportJson()` | Serialize to JSON string |
| `Router.importJson(json)` | Deserialize from JSON string (factory) |

## Types

```typescript
interface ResolveMatch {
  id: string;
  score: number;
}

interface PhraseResult {
  added: boolean;
  newTerms: string[];
  redundant: boolean;
  warning?: string | null;
}
```
