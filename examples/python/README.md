# Python Examples

## Setup

```bash
# Build the native Python module (from project root)
cd python && python -m venv .venv && source .venv/bin/activate
maturin develop --release

# Install example dependencies
cd ../examples/python
pip install -r requirements.txt
```

After PyPI publish: just `pip install microresolve` — no build step needed.

All examples import as `from microresolve import Router`.

## Examples

| Example | Description |
|---------|-------------|
| `launch_demo.py` | **Three-namespace fan-out + confirm-turn pattern with a live LLM call.** Mirrors the launch-blog demo end-to-end. |
| `basic.py` | Routing, multi-intent, learning, export/import, discovery |
| `fastapi_server.py` | Production REST endpoint with FastAPI |
| `hybrid_llm.py` | 80/20 pattern: MicroResolve routes cheap, Claude handles the rest |
| `openapi_import.py` | Import intents from an OpenAPI spec |
