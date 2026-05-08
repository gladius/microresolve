# Node.js Examples

## Setup

```bash
# Build the native Node addon (from project root)
cd node && npm install && npx napi build --release

# Install example dependencies
cd ../examples/node && npm install
```

After npm publish: just `npm install microresolve` — no build step needed.

All examples require as `const { Router } = require('microresolve')`.

## Examples

| Example | Description |
|---------|-------------|
| `launch_demo.js` | **Three-namespace fan-out + confirm-turn pattern with a live LLM call.** Mirrors the launch-blog demo end-to-end. |
| `basic.js` | Routing, multi-intent, learning, export/import, discovery |
| `express_server.js` | Production REST endpoint with Express |
