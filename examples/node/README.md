# Node.js Examples

## Setup

```bash
# Build the native Node addon (from project root)
cd node && npm install && npx napi build --release

# Install example dependencies
cd ../examples/node && npm install
```

After npm publish: just `npm install microresolve` — no build step needed.

## Examples

| Example | Description |
|---------|-------------|
| `basic.js` | Routing, multi-intent, learning, export/import, discovery |
| `express_server.js` | Production REST endpoint with Express |
