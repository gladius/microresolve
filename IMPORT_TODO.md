# Import Formats — Implementation TODO

Priority order for the 3-day launch sprint.

---

## ✅ Done

- **OpenAPI / Swagger** — `parse_spec()` + full UI flow (parse → select → apply)
- **MCP Tools** — Smithery search + paste JSON → apply

---

## 🔜 Next (in order)

### ✅ 1. OpenAI Function Calling
**Effort**: ~1 hour
**Value**: very high — largest installed base of any AI format

Format:
```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather for a location",
      "parameters": { "type": "object", "properties": { "location": { "type": "string" } } }
    }
  }
]
```

Implementation: add normalization shim in `mcp_parse` / `mcp_apply` — detect `type: "function"` wrapper and unwrap `.function`. No new route, no new UI page. Shares MCP paste UI.

---

### ✅ 2. LangChain Tools
**Effort**: ~30 minutes (shares code with OpenAI functions)
**Value**: high — most Python AI developers touch LangChain

Format:
```json
[
  {
    "name": "search",
    "description": "Search the web for current information",
    "args_schema": { "type": "object", "properties": { "query": { "type": "string" } } }
  }
]
```

Implementation: same shim — normalize `args_schema` → `inputSchema`. Auto-detected alongside OpenAI functions in the same MCP paste flow.

---

### 3. Chatbot Migration (Dialogflow + Rasa + Botpress + LUIS)
**Effort**: ~5 hours total (all four share one parser)
**Value**: very high — real migration pain, utterances are direct seeds (no LLM needed)

#### Dialogflow (JSON export)
```json
{
  "name": "book_flight",
  "userSays": [
    { "data": [{ "text": "book a flight to NYC" }] },
    { "data": [{ "text": "I need to fly to London" }] }
  ]
}
```
Extract: `userSays[].data[].text` → seeds

#### Rasa NLU (YAML)
```yaml
nlu:
- intent: book_flight
  examples: |
    - book a flight to NYC
    - fly me to London
    - I want to fly to Paris
```
Extract: `nlu[].examples` lines (strip leading `- `) → seeds

#### Botpress (JSON)
```json
{
  "intents": [
    { "name": "book_flight", "utterances": ["book a flight", "fly me to", "I want to fly"] }
  ]
}
```
Extract: `intents[].utterances` → seeds

#### LUIS (JSON)
```json
{
  "intents": [{ "name": "BookFlight" }],
  "utterances": [
    { "text": "book a flight to Seattle", "intent": "BookFlight" },
    { "text": "fly me to London", "intent": "BookFlight" }
  ]
}
```
Extract: group `utterances[]` by `.intent` → seeds per intent

Implementation plan:
- New route: `POST /api/import/chatbot/parse` + `POST /api/import/chatbot/apply`
- New UI page: `/import/chatbot` — paste JSON or YAML, auto-detect format, select intents, apply
- Format detection: check for `userSays` (Dialogflow), `nlu:` key (Rasa), `intents[].utterances` (Botpress), `utterances[].intent` (LUIS)
- New nav card: "Chatbot Migration" on ImportLanding (replaces the split Dialogflow/Botpress cards)

---

## ❌ Removed / Skip

- **GraphQL** — no descriptions on most queries/mutations, vocabulary mismatch, same problem as direct content indexing. Not worth it.
- **Zapier** — requires OAuth per user, dynamic catalog, wrong audience.
- **n8n** — complex node format, not designed for external consumption.
