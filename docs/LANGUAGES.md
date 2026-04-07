# Language Support

ASV Router supports **57 languages** across all major script families. The router is script-aware, not language-aware — it picks the right tokenization path based on Unicode ranges, so any language works without configuration.

## How it works

| Script type | Tokenization | Languages |
|---|---|---|
| **Space-delimited** (Latin, Cyrillic, Devanagari, Arabic, etc.) | Whitespace splitting + bigrams | Most languages |
| **Unsegmented** (CJK, Thai, Lao, Myanmar, Khmer) | Aho-Corasick automaton + character bigram fallback | Chinese, Japanese, Korean, Thai, Lao, Myanmar, Khmer |

No language detection is needed. The router inspects character Unicode ranges at tokenization time and routes through the correct path automatically.

## Supported languages

### Latin script
English, Spanish, French, German, Portuguese, Italian, Dutch, Polish, Romanian, Hungarian, Czech, Swedish, Danish, Norwegian, Finnish, Catalan, Croatian, Turkish, Azerbaijani, Uzbek, Vietnamese, Indonesian, Malay, Filipino, Swahili, Hausa

### Cyrillic script
Russian, Ukrainian, Serbian, Bulgarian

### Greek script
Greek

### Unique scripts (space-delimited)
Georgian, Armenian, Amharic (Ethiopic)

### Indic scripts (space-delimited)
Hindi (Devanagari), Marathi (Devanagari), Nepali (Devanagari), Bengali, Punjabi (Gurmukhi), Gujarati, Tamil, Telugu, Kannada, Malayalam, Odia, Sinhala

### Arabic script (space-delimited, RTL)
Arabic, Hebrew, Persian, Urdu, Pashto

### CJK (unsegmented, automaton path)
Chinese (simplified + traditional), Japanese, Korean

### Southeast Asian (unsegmented, automaton path)
Thai, Lao, Myanmar (Burmese), Khmer

## Stop words

Stop words are minimal and safe by design:

- **Universal** (~30 words): Only English function words that are universally low-signal and do not collide with meaningful words in other Latin-script languages. Words like "die" (German "the" vs English "die") are deliberately excluded.
- **Unsegmented scripts**: Script-specific particles for Chinese, Japanese, Korean, Thai, Lao, Myanmar, and Khmer. Each script occupies its own Unicode range, so there is no cross-script collision risk.
- **All other languages**: No stop words. Term weighting handles frequency-based suppression as the intent count grows.

## AI seed generation

The "Generate Seeds with AI" feature in the web demo calls Claude to produce seed phrases. Prompt construction and response parsing happen in Rust — the UI only handles HTTP transport.

**LLM language proficiency varies.** Claude produces high-quality seeds for widely-spoken languages (English, Spanish, Chinese, Hindi, Arabic, etc.) but may produce lower-quality or romanized output for less-resourced languages (Hausa, Pashto, Odia, etc.). Generated seeds appear in an editable textarea — review and correct before adding to the router.

The seed generator is optional. Seeds can always be added manually in any language.

## Adding a new language

1. Add an entry to `languages/languages.json` with a `name` and `hint` (prompt guidance for the LLM seed generator, or `null`).
2. If the script is unsegmented (no spaces between words), add its Unicode range to `is_cjk()` in `src/tokenizer.rs`.
3. If the script has pure grammatical particles worth filtering, add them under `unsegmented` in `languages/stopwords.json`.
4. Space-delimited scripts need no code changes — they work automatically.
