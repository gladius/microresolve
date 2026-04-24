---
title: HTTP Server API
description: REST API reference for the MicroResolve server.
---

The MicroResolve server exposes a REST API on port 3001. All endpoints accept and return JSON. Namespace is set via the `X-Namespace-ID` header (defaults to `default`).

## Running the server

```bash
# From source
cargo build --release --bin server --features server
./target/release/server

# With custom data directory
./target/release/server --data /path/to/data

# Environment
ANTHROPIC_API_KEY=sk-...  # required for auto-learn
MICRORESOLVE_DATA_DIR=~/.local/share/microresolve  # optional
```

## Resolve

```
POST /api/resolve
X-Namespace-ID: my-namespace
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
| `DELETE` | `/api/intents/:id` | Delete intent |

## Phrases

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/phrases` | Add phrase to intent |
| `DELETE` | `/api/phrases` | Remove phrase |

## Auto-learn

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/review` | Get review queue |
| `POST` | `/api/learn/now` | Trigger synchronous learn |
| `GET` | `/api/review/stats` | Queue stats |

## Namespaces

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/namespaces` | List namespaces |
| `POST` | `/api/namespaces` | Create namespace |
| `PATCH` | `/api/namespaces/:id` | Update namespace |

## Layers (advanced)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/layers/morphology` | Inspect morphology graph |
| `POST` | `/api/layers/morphology/edges` | Add morphology edge |
| `GET` | `/api/layers/scoring/test` | Test query against scoring layer |
