# curl / HTTP API Examples

## Setup

```bash
# Start the server (from project root)
cargo run --release --bin microresolve-studio --features server -- --data ./data
```

## Examples

| Example | Description |
|---------|-------------|
| `basic.sh` | Routing, multi-intent, learning, export via HTTP API |
| `multi_app.sh` | App isolation with X-App-ID headers |
