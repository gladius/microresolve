//! Optional entity-detection layer (PoC).
//!
//! Sits between L0 (typo correction) and L1 (morphology) in the routing pipeline,
//! emitting entity-type tokens like `[CC]`, `[SSN]`, `[EMAIL]` that get appended
//! to the query before normal token scoring. Intents trained on phrases that
//! include these tokens then score for queries with detected entities.
//!
//! Off by default — enabled per-request via the `enable_entity_layer` flag on
//! /api/route_multi, or programmatically via `Router::set_entity_layer`.
//!
//! Hybrid implementation: regex catches entity values (formats like SSN, CC),
//! Aho-Corasick catches entity context words ("my password", "credit card").
//! Bake-off in `src/bin/entity_bench.rs` showed F1 = 0.98 on synthetic test set
//! at 17µs median latency. See `blogs/entity-layer-poc.md` for findings and
//! `ENTITY_LAYER_PLAN.md` for the broader plan.

use aho_corasick::AhoCorasick;
use regex::Regex;

/// Hybrid entity detector — regex for values, Aho-Corasick for context phrases.
///
/// Construct via `default()` for the built-in PII pattern set; replace patterns
/// later via the LLM-distillation pipeline (Phase 3 in ENTITY_LAYER_PLAN.md).
pub struct EntityLayer {
    regex_patterns: Vec<(&'static str, Regex)>,
    ac: AhoCorasick,
    ac_pattern_to_label: Vec<&'static str>,
}

impl EntityLayer {
    /// Built-in default detector covering common US-format PII and credential
    /// context. Designed as a starting point — production deployments would
    /// extend or replace this via LLM-distilled patterns per namespace.
    pub fn default_pii() -> Self {
        let regex_raw: &[(&'static str, &str)] = &[
            ("CC",    r"\b(?:\d[ -]?){12,18}\d\b"),
            ("SSN",   r"\b\d{3}-\d{2}-\d{4}\b"),
            ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            ("PHONE", r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            ("IPV4",  r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        ];
        let regex_patterns: Vec<_> = regex_raw.iter()
            .map(|(label, p)| (*label, Regex::new(p).expect("built-in regex compiles")))
            .collect();

        let ac_raw: &[(&'static str, &str)] = &[
            ("CC", "credit card"), ("CC", "card number"), ("CC", "cc number"),
            ("CC", "visa"), ("CC", "mastercard"), ("CC", "amex"),
            ("SSN", "ssn"), ("SSN", "social security"), ("SSN", "social security number"),
            ("EMAIL", "email"), ("EMAIL", "e-mail"), ("EMAIL", "email address"),
            ("PHONE", "phone"), ("PHONE", "phone number"), ("PHONE", "cell number"),
            ("PHONE", "mobile number"),
            ("SECRET", "password"), ("SECRET", "passcode"), ("SECRET", "api key"),
            ("SECRET", "secret key"), ("SECRET", "access token"), ("SECRET", "auth token"),
            ("ADDRESS", "home address"), ("ADDRESS", "street address"),
            ("ADDRESS", "zip code"), ("ADDRESS", "postal code"),
        ];
        let ac_patterns: Vec<&str> = ac_raw.iter().map(|(_, p)| *p).collect();
        let ac_pattern_to_label: Vec<&'static str> = ac_raw.iter().map(|(l, _)| *l).collect();
        let ac = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(ac_patterns)
            .expect("built-in AC patterns compile");

        Self { regex_patterns, ac, ac_pattern_to_label }
    }

    /// Detect all entity-type labels present in the query.
    /// Returns deduplicated labels in detection order.
    pub fn detect(&self, query: &str) -> Vec<&'static str> {
        let mut hits: Vec<&'static str> = Vec::new();
        for (label, re) in &self.regex_patterns {
            if re.is_match(query) && !hits.contains(label) {
                hits.push(*label);
            }
        }
        for m in self.ac.find_overlapping_iter(query) {
            let label = self.ac_pattern_to_label[m.pattern().as_usize()];
            if !hits.contains(&label) { hits.push(label); }
        }
        hits
    }

    /// Convenience: return the query augmented with detected entity tokens
    /// appended after the original text. The augmented form is what gets fed
    /// to L1/L2 scoring when the layer is enabled.
    ///
    /// Tokens use the `mr_pii_<label>` convention (lowercase, alphanumeric+underscore)
    /// so they:
    ///   1. survive the standard tokenizer (no bracket stripping)
    ///   2. never collide with natural English vocabulary
    ///   3. are unique to the entity layer (intents seeded with these tokens
    ///      score high IDF for them — the score the layer actually adds is
    ///      pure entity signal, not noise duplicated from words already there)
    ///
    /// Example: `"my SSN is 123-45-6789"` → `"my SSN is 123-45-6789 mr_pii_ssn"`
    pub fn augment(&self, query: &str) -> String {
        let labels = self.detect(query);
        if labels.is_empty() { return query.to_string(); }
        let suffix: String = labels.iter()
            .map(|l| format!(" mr_pii_{}", l.to_lowercase()))
            .collect();
        format!("{}{}", query, suffix)
    }

    /// The token that the augment pass emits for a given entity label.
    /// Useful when seeding intents — your training phrase should include
    /// `entity_token("CC")` to make the entity layer's signal score for that intent.
    pub fn entity_token(label: &str) -> String {
        format!("mr_pii_{}", label.to_lowercase())
    }

    /// Find every entity span in the query, returning label + position + value.
    ///
    /// Unlike `detect()` (which returns deduplicated labels), this returns
    /// every concrete match so callers can do extraction or masking. Spans
    /// are reported in the order they're discovered (regex hits first, then
    /// Aho-Corasick context-word hits).
    pub fn detect_with_spans<'a>(&self, query: &'a str) -> Vec<EntitySpan<'a>> {
        let mut spans = Vec::new();
        // Regex matches give value-level spans (the actual entity value).
        for (label, re) in &self.regex_patterns {
            for m in re.find_iter(query) {
                spans.push(EntitySpan {
                    label,
                    value: &query[m.start()..m.end()],
                    start: m.start(),
                    end: m.end(),
                    source: SpanSource::Value,
                });
            }
        }
        // Aho-Corasick matches give context-word spans (the surrounding
        // language, not the entity value itself). They prove an entity type
        // is being discussed even when no value is present.
        for m in self.ac.find_overlapping_iter(query) {
            let label = self.ac_pattern_to_label[m.pattern().as_usize()];
            spans.push(EntitySpan {
                label,
                value: &query[m.start()..m.end()],
                start: m.start(),
                end: m.end(),
                source: SpanSource::Context,
            });
        }
        spans
    }

    /// Extract all entity values from the query, grouped by label.
    /// Only includes value-level matches (regex hits), since context-word
    /// matches don't carry an entity value to extract.
    ///
    /// Example: `"send 4111-1111-1111-1111 to alice@x.com"` →
    ///   `{"CC": ["4111-1111-1111-1111"], "EMAIL": ["alice@x.com"]}`
    pub fn extract<'a>(&self, query: &'a str) -> HashMap<&'static str, Vec<&'a str>> {
        let mut out: HashMap<&'static str, Vec<&'a str>> = HashMap::new();
        for span in self.detect_with_spans(query) {
            if matches!(span.source, SpanSource::Value) {
                out.entry(span.label).or_default().push(span.value);
            }
        }
        out
    }

    /// Replace every detected entity VALUE in the query with a placeholder.
    /// Context-word matches are left intact (they describe the entity, they
    /// aren't the entity itself, so masking them changes meaning).
    ///
    /// `placeholder_for(label)` controls the replacement text — pass a closure
    /// to format placeholders as you like (`<EMAIL>`, `[REDACTED]`, etc.).
    ///
    /// Example with `|l| format!("<{}>", l)`:
    ///   `"my SSN is 123-45-6789"` → `"my SSN is <SSN>"`
    pub fn mask<F>(&self, query: &str, mut placeholder_for: F) -> String
    where F: FnMut(&str) -> String {
        // Collect value spans only and sort by start position so we can
        // splice in placeholders left-to-right.
        let mut value_spans: Vec<EntitySpan> = self.detect_with_spans(query)
            .into_iter()
            .filter(|s| matches!(s.source, SpanSource::Value))
            .collect();
        value_spans.sort_by_key(|s| s.start);

        // Drop overlapping spans — keep the first (which is the longer match
        // when patterns of different lengths cover the same region, since
        // they were inserted in pattern-declaration order).
        let mut deduped: Vec<EntitySpan> = Vec::with_capacity(value_spans.len());
        let mut cursor = 0usize;
        for span in value_spans {
            if span.start >= cursor {
                cursor = span.end;
                deduped.push(span);
            }
        }

        // Splice.
        let mut out = String::with_capacity(query.len());
        let mut pos = 0usize;
        for span in deduped {
            out.push_str(&query[pos..span.start]);
            out.push_str(&placeholder_for(span.label));
            pos = span.end;
        }
        out.push_str(&query[pos..]);
        out
    }
}

/// One detected entity occurrence in the query.
#[derive(Debug, Clone)]
pub struct EntitySpan<'a> {
    /// The entity-type label (e.g., "CC", "EMAIL").
    pub label: &'static str,
    /// The matched substring from the query.
    pub value: &'a str,
    /// Start byte offset in the original query.
    pub start: usize,
    /// End byte offset (exclusive).
    pub end: usize,
    /// Whether this match came from a value pattern (regex) or a context word
    /// pattern (Aho-Corasick). Value matches are extractable/maskable;
    /// context matches only signal that an entity *type* is being discussed.
    pub source: SpanSource,
}

/// Provenance of an entity span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanSource {
    /// Matched a value pattern (e.g., the actual SSN digits).
    Value,
    /// Matched a context phrase (e.g., the words "social security number").
    Context,
}

use std::collections::HashMap;

impl Default for EntityLayer {
    fn default() -> Self {
        Self::default_pii()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_credit_card_value() {
        let e = EntityLayer::default_pii();
        assert!(e.detect("save my card 4111-1111-1111-1111 for next time").contains(&"CC"));
    }

    #[test]
    fn detects_credit_card_context() {
        let e = EntityLayer::default_pii();
        // No actual value — only the context phrase.
        assert!(e.detect("we should never store credit cards").contains(&"CC"));
    }

    #[test]
    fn detects_ssn() {
        let e = EntityLayer::default_pii();
        assert!(e.detect("my SSN is 123-45-6789 please file the taxes").contains(&"SSN"));
    }

    #[test]
    fn detects_email_value_without_context() {
        let e = EntityLayer::default_pii();
        // Just the address, no surrounding "email" word.
        assert!(e.detect("forward to alice@example.com when ready").contains(&"EMAIL"));
    }

    #[test]
    fn detects_password_context_without_value() {
        let e = EntityLayer::default_pii();
        assert!(e.detect("my password is hunter2").contains(&"SECRET"));
    }

    #[test]
    fn rejects_pii_adjacent_negatives() {
        let e = EntityLayer::default_pii();
        // Short IDs that LOOK like PII shapes but aren't valid lengths.
        assert!(!e.detect("ticket number 4111-2222 was closed").contains(&"CC"));
        assert!(!e.detect("issue #1234 is fixed").contains(&"CC"));
    }

    #[test]
    fn rejects_normal_queries() {
        let e = EntityLayer::default_pii();
        assert!(e.detect("what is the weather today").is_empty());
        assert!(e.detect("create a new pull request").is_empty());
    }

    #[test]
    fn detects_multiple_entities_in_one_query() {
        let e = EntityLayer::default_pii();
        let labels = e.detect("send 4111-1111-1111-1111 to alice@example.com tomorrow");
        assert!(labels.contains(&"CC"));
        assert!(labels.contains(&"EMAIL"));
    }

    #[test]
    fn augment_appends_distinctive_tokens() {
        let e = EntityLayer::default_pii();
        let augmented = e.augment("my SSN is 123-45-6789");
        // Must use mr_pii_xxx convention so it survives tokenization and
        // doesn't collide with natural English vocabulary.
        assert!(augmented.contains("mr_pii_ssn"), "got: {}", augmented);
        assert!(augmented.starts_with("my SSN is 123-45-6789"));
    }

    #[test]
    fn entity_token_helper_matches_what_augment_emits() {
        let token = EntityLayer::entity_token("CC");
        assert_eq!(token, "mr_pii_cc");
        let e = EntityLayer::default_pii();
        let aug = e.augment("save 4111-1111-1111-1111");
        assert!(aug.contains(&token));
    }

    #[test]
    fn augment_returns_original_when_no_entities() {
        let e = EntityLayer::default_pii();
        assert_eq!(e.augment("create a pull request"), "create a pull request");
    }

    // ── Extraction ───────────────────────────────────────────────────────────

    #[test]
    fn extract_returns_credit_card_value() {
        let e = EntityLayer::default_pii();
        let extracted = e.extract("save my card 4111-1111-1111-1111 for next time");
        assert_eq!(extracted.get("CC"), Some(&vec!["4111-1111-1111-1111"]));
    }

    #[test]
    fn extract_handles_multiple_entities_in_one_query() {
        let e = EntityLayer::default_pii();
        let extracted = e.extract("send 4111-1111-1111-1111 to alice@example.com tomorrow");
        assert_eq!(extracted.get("CC"), Some(&vec!["4111-1111-1111-1111"]));
        assert_eq!(extracted.get("EMAIL"), Some(&vec!["alice@example.com"]));
    }

    #[test]
    fn extract_skips_context_only_matches() {
        let e = EntityLayer::default_pii();
        // "credit card" is a context phrase — there's no value to extract.
        let extracted = e.extract("we should never store credit cards in plaintext");
        assert!(extracted.get("CC").is_none(),
            "context-only matches must not produce extracted values, got {:?}",
            extracted);
    }

    #[test]
    fn extract_groups_multiple_values_of_same_type() {
        let e = EntityLayer::default_pii();
        let extracted = e.extract("send to alice@x.com and copy bob@y.com please");
        let emails = extracted.get("EMAIL").expect("should find emails");
        assert_eq!(emails.len(), 2);
        assert!(emails.contains(&"alice@x.com"));
        assert!(emails.contains(&"bob@y.com"));
    }

    // ── Masking ──────────────────────────────────────────────────────────────

    #[test]
    fn mask_replaces_credit_card_with_placeholder() {
        let e = EntityLayer::default_pii();
        let masked = e.mask("save my card 4111-1111-1111-1111 for next time",
            |label| format!("<{}>", label));
        assert_eq!(masked, "save my card <CC> for next time");
    }

    #[test]
    fn mask_handles_multiple_entities() {
        let e = EntityLayer::default_pii();
        let masked = e.mask("send 4111-1111-1111-1111 to alice@example.com",
            |label| format!("<{}>", label));
        assert_eq!(masked, "send <CC> to <EMAIL>");
    }

    #[test]
    fn mask_preserves_context_words() {
        let e = EntityLayer::default_pii();
        // "credit card" is a context phrase — should NOT be masked because
        // masking it would change the meaning of the sentence.
        let masked = e.mask("we should never store credit cards in plaintext",
            |label| format!("<{}>", label));
        assert_eq!(masked, "we should never store credit cards in plaintext");
    }

    #[test]
    fn mask_with_redacted_placeholder() {
        let e = EntityLayer::default_pii();
        let masked = e.mask("my SSN is 123-45-6789", |_| "[REDACTED]".to_string());
        assert_eq!(masked, "my SSN is [REDACTED]");
    }

    #[test]
    fn mask_returns_query_unchanged_when_no_entities() {
        let e = EntityLayer::default_pii();
        let masked = e.mask("create a new pull request", |label| format!("<{}>", label));
        assert_eq!(masked, "create a new pull request");
    }

    // ── detect_with_spans (the underlying primitive) ────────────────────────

    #[test]
    fn detect_with_spans_returns_value_and_context_separately() {
        let e = EntityLayer::default_pii();
        let spans = e.detect_with_spans("my credit card 4111-1111-1111-1111 is on file");
        let value_spans: Vec<_> = spans.iter().filter(|s| s.source == SpanSource::Value).collect();
        let context_spans: Vec<_> = spans.iter().filter(|s| s.source == SpanSource::Context).collect();
        assert!(!value_spans.is_empty(), "should have at least one value span");
        assert!(!context_spans.is_empty(), "should have at least one context span");
        assert_eq!(value_spans[0].value, "4111-1111-1111-1111");
        assert_eq!(value_spans[0].label, "CC");
    }

    #[test]
    fn detect_with_spans_positions_are_correct() {
        let e = EntityLayer::default_pii();
        let query = "email: alice@x.com";
        let spans = e.detect_with_spans(query);
        let email = spans.iter().find(|s| s.label == "EMAIL" && s.source == SpanSource::Value)
            .expect("EMAIL span should be present");
        assert_eq!(&query[email.start..email.end], "alice@x.com");
    }
}
