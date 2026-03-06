#!/bin/bash
# Build WASM and copy to web/pkg/
set -e
cd "$(dirname "$0")/.."
wasm-pack build --target web -- --features wasm
rm -rf web/pkg
cp -r pkg web/pkg
rm -rf pkg

# Inject ANTHROPIC_API_KEY into config.js if set
if [ -n "$ANTHROPIC_API_KEY" ]; then
  echo "window.ASV_CONFIG = { apiKey: '$ANTHROPIC_API_KEY' };" > web/config.js
  echo "API key injected into web/config.js"
else
  echo "Note: ANTHROPIC_API_KEY not set. AI seed generation will be disabled."
fi

echo ""
echo "Build complete. Serve with:"
echo "  cd web && python3 -m http.server 8080"
echo "  Then open http://localhost:8080/index.html"
