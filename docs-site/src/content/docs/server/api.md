---
title: HTTP Server API Reference
description: REST API reference for the MicroResolve server.
---

The MicroResolve server exposes a REST API on port 3001. All endpoints accept and return JSON.

**Namespace selection:** pass the `X-Namespace-ID` header on every request. Omitting it routes to the `default` namespace.

**Auth:** when API key auth is enabled, pass the key in the `X-Api-Key` header. See [Auth](/server/auth/).

## Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/version` | Server version |
| `GET` | `/api/llm/status` | LLM configuration status |

## Classification

### Resolve a query

```
POST /api/route_multi
X-Namespace-ID: support
Content-Type: application/json
```

```json
{ "query": "cancel my order" }
```

**Response:**

```json
{
  "intents": [
    { "id": "cancel_order", "score": 0.91 }
  ],
  "query": "cancel my order",
  "latency_us": 31
}
```

## Intents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/intents` | List all intents |
| `POST` | `/api/intents` | Create intent |
| `PATCH` | `/api/intents/{id}` | Update intent metadata |
| `DELETE` | `/api/intents/{id}` | Delete intent |
| `POST` | `/api/intents/{id}/phrases` | Add phrase to intent |
| `DELETE` | `/api/intents/{id}/phrases` | Remove phrase from intent |

### Create intent

```
POST /api/intents
X-Namespace-ID: support
```

```json
{
  "id": "cancel_order",
  "description": "User wants to cancel an existing order",
  "phrases": ["cancel my order", "I want to cancel"]
}
```

### Add phrase

```
POST /api/intents/cancel_order/phrases
X-Namespace-ID: support
```

```json
{ "phrase": "abort my order", "lang": "en" }
```

## Namespaces

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/namespaces` | List all namespaces |
| `POST` | `/api/namespaces` | Create namespace |
| `PATCH` | `/api/namespaces` | Update namespace settings |
| `DELETE` | `/api/namespaces` | Delete namespace |
| `GET` | `/api/namespaces/{id}/history` | Git commit log for this namespace |
| `POST` | `/api/namespaces/{id}/rollback` | Roll back to a commit |
| `GET` | `/api/namespaces/{id}/diff` | Semantic diff (intents added / removed / changed) |

### Rollback

```
POST /api/namespaces/support/rollback
```

```json
{ "sha": "abc1234..." }
```

## Review / Auto-learn

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/review/queue` | Get the low-confidence review queue |
| `GET` | `/api/review/stats` | Queue depth and approval rate |
| `POST` | `/api/review/fix` | Reassign a queued item to the correct intent |
| `POST` | `/api/review/reject` | Discard a queued item |
| `POST` | `/api/review/analyze` | LLM-assisted batch analysis of the queue |
| `POST` | `/api/review/intent_phrases` | Suggest phrases for an intent via LLM |
| `POST` | `/api/learn/now` | Trigger a synchronous auto-learn pass |
| `POST` | `/api/learn/words` | Extract and index vocabulary terms |
| `POST` | `/api/report` | Report a misclassification from the client |

## Sync (connected mode)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/sync` | Pull latest namespace state |
| `POST` | `/api/ingest` | Push local events to server |
| `POST` | `/api/correct` | Submit a correction |

## Logs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/logs` | Recent classification log entries |
| `GET` | `/api/logs/stats` | Aggregate stats (match rate, latency) |

## Import / Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/export` | Export current namespace as JSON |
| `POST` | `/api/import` | Import namespace from JSON |
| `POST` | `/api/import/parse` | Parse an import payload without applying |
| `POST` | `/api/import/apply` | Apply a parsed import |
| `GET` | `/api/import/params` | Import configuration |
| `GET` | `/api/import/mcp/search` | Search MCP tool registry |
| `GET` | `/api/import/mcp/fetch` | Fetch MCP tool spec |
| `POST` | `/api/import/mcp/parse` | Parse MCP spec into intents |
| `POST` | `/api/import/mcp/apply` | Apply parsed MCP intents |

## Settings

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/settings` | Get server settings |
| `PATCH` | `/api/settings` | Update server settings |
| `GET` | `/api/languages` | List supported languages |
| `DELETE` | `/api/data/all` | Delete all data (irreversible) |

## Git / Remote sync

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/settings/git` | Get git remote config |
| `PUT` | `/api/settings/git` | Set or clear git remote |
| `POST` | `/api/git/push` | Manual push to remote |

See [Git Data Layer](/server/git-data/) for details.

## Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/auth/keys` | List API keys |
| `POST` | `/api/auth/keys` | Create API key |
| `DELETE` | `/api/auth/keys/{name}` | Delete API key |

See [Auth](/server/auth/) for details.

## Events (SSE)

```
GET /api/events
```

Server-sent event stream. Emits events on intent changes, learn completions, and review queue updates. Connect from the browser or any SSE client.

## Layers (advanced)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/layers/info` | Layer summary for current namespace |
| `GET` | `/api/layers/l1/edges` | List L1 (morphology) graph edges |
| `POST` | `/api/layers/l1/edges` | Add an L1 edge |
| `DELETE` | `/api/layers/l1/edges` | Delete an L1 edge |
| `POST` | `/api/layers/l1/distill` | Trigger LLM distillation of L1 edges |
| `POST` | `/api/layers/l2/probe` | Probe L2 scoring layer with a query |
