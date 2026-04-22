//! Optional entity-detection layer.
//!
//! Sits between L0 (typo correction) and L1 (morphology) in the routing pipeline.
//! Detects PII, credentials, identifiers, web/tech entities — emits entity-type
//! tokens (`mr_pii_<label>`) into the query for downstream intent matching, and
//! exposes detection / extraction / masking output modes.
//!
//! Architecture:
//!   - A static **registry of built-in patterns** (CC, SSN, AWS keys, JWT, etc.)
//!     organized by category. Apache 2.0-compatible patterns lifted from common
//!     industry sources where applicable.
//!   - `EntityLayer::recommended()` builds a layer from the preset enabled set.
//!   - `EntityLayer::with_labels()` builds a layer from any subset of patterns
//!     plus optional custom patterns (LLM-distilled or hand-written).
//!   - Per-namespace configuration of which patterns are active is stored in
//!     `_entities.json`; hot-reloaded on change.
//!
//! Hybrid implementation: **regex** catches entity VALUES, **Aho-Corasick**
//! catches entity CONTEXT WORDS. Both run; their outputs merge.

use aho_corasick::AhoCorasick;
use regex::Regex;
use std::collections::{HashMap, HashSet};

// ─── Built-in pattern registry ────────────────────────────────────────────────

/// Metadata for a built-in entity pattern. Loaded from `patterns/builtin.json`
/// at compile time via `include_str!`, parsed once on first access.
///
/// Patterns are organized by category (PII, Credentials, Identifiers, etc.).
/// Adding a new built-in pattern is a JSON edit, not a code change — same
/// runtime cost (regex matching), unified storage with custom entities,
/// reviewable via standard PR diff on the JSON file.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct BuiltinPattern {
    /// Stable identifier — used in API responses, persistence, and as the
    /// suffix of the emitted `mr_pii_<label>` token (lowercased).
    pub label: String,
    /// Display category for UI grouping ("PII", "Credentials", etc.).
    pub category: String,
    /// Human-readable name for UI display.
    pub display_name: String,
    /// Short description shown in the UI tooltip.
    pub description: String,
    /// Value-pattern regex strings (Rust syntax). May be empty if the entity
    /// is detected only via context phrases.
    pub regex_patterns: Vec<String>,
    /// Context-phrase strings for Aho-Corasick (lowercase). May be empty.
    pub context_phrases: Vec<String>,
    /// Whether this is enabled in the "recommended" preset for general use.
    /// Customers can override via per-namespace config.
    pub recommended: bool,
}

/// The full registry of built-in entity patterns.
///
/// Loaded once at first access from `patterns/builtin.json`, which is embedded
/// into the binary at compile time via `include_str!`. Same runtime cost as
/// a hardcoded array (parsed once, immutable thereafter), but adding a new
/// built-in pattern is a JSON edit reviewable as a normal PR diff — no Rust
/// source change required. Unifies storage with custom entities (`_entities.json`).
///
/// Categories: PII, Credentials, Identifiers, Web/Tech, Financial.
pub static BUILTIN_PATTERNS: std::sync::LazyLock<Vec<BuiltinPattern>> =
    std::sync::LazyLock::new(|| {
        const RAW: &str = include_str!("../patterns/builtin.json");
        serde_json::from_str(RAW)
            .expect("patterns/builtin.json must be valid JSON matching BuiltinPattern")
    });

/// Look up a built-in pattern by label.
pub fn get_builtin(label: &str) -> Option<&'static BuiltinPattern> {
    // The LazyLock ensures the Vec lives forever after first access, so the
    // 'static lifetime on the returned reference is sound.
    BUILTIN_PATTERNS.iter().find(|p| p.label.as_str() == label)
}

/// All distinct categories in the registry, in declaration order.
pub fn all_categories() -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for p in BUILTIN_PATTERNS.iter() {
        if seen.insert(p.category.clone()) { out.push(p.category.clone()); }
    }
    out
}

// ─── EntityLayer ──────────────────────────────────────────────────────────────

/// Per-namespace entity-detection configuration.
/// Persisted in `_entities.json` next to `_ns.json`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EntityConfig {
    /// Labels of built-in patterns enabled for this namespace.
    /// Empty = no built-ins (custom-only or fully off).
    pub enabled_builtins: Vec<String>,
    /// Custom (user-defined or LLM-distilled) entities for this namespace.
    #[serde(default)]
    pub custom: Vec<CustomEntity>,
}

impl EntityConfig {
    /// Build a config with the "recommended" preset of built-ins enabled.
    pub fn recommended() -> Self {
        Self {
            enabled_builtins: BUILTIN_PATTERNS.iter()
                .filter(|p| p.recommended)
                .map(|p| p.label.to_string())
                .collect(),
            custom: vec![],
        }
    }

    /// Empty config — no built-ins, no custom. Layer effectively disabled.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build the EntityLayer this config describes.
    pub fn build_layer(&self) -> EntityLayer {
        EntityLayer::with_labels_and_custom(&self.enabled_builtins, &self.custom)
    }
}

/// A user-defined custom entity pattern (typically LLM-distilled, sometimes
/// hand-written). Stored per-namespace alongside the selection of built-ins.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomEntity {
    pub label: String,
    pub display_name: String,
    pub description: String,
    pub regex_patterns: Vec<String>,
    pub context_phrases: Vec<String>,
    /// Optional examples for documentation and validation. Never real PII.
    #[serde(default)]
    pub examples: Vec<String>,
    /// "llm_distillation" or "manual".
    #[serde(default = "default_source")]
    pub source: String,
}

fn default_source() -> String { "manual".to_string() }

/// Hybrid entity detector — regex for values, Aho-Corasick for context phrases.
///
/// Construct via:
///   - `default_pii()`  — PII-only set (back-compat with earlier API)
///   - `recommended()`  — patterns marked `recommended: true` in registry
///   - `with_labels(&[...])` — explicit subset of built-ins
///   - `with_labels_and_custom(&[...], &[...])` — built-ins plus custom entities
pub struct EntityLayer {
    regex_patterns: Vec<(String, Regex)>,
    ac: AhoCorasick,
    ac_pattern_to_label: Vec<String>,
}

impl EntityLayer {
    /// PII-only preset (CC, SSN, EMAIL, PHONE, IPV4, SECRET).
    /// Kept for backward compatibility with earlier code paths.
    pub fn default_pii() -> Self {
        Self::with_labels(&[
            "CC".to_string(), "SSN".to_string(), "EMAIL".to_string(),
            "PHONE".to_string(), "IPV4".to_string(), "SECRET".to_string(),
            "ADDRESS".to_string(),
        ])
    }

    /// Build from the "recommended" preset — every BuiltinPattern with
    /// `recommended: true`. Sensible defaults for general PII + secrets coverage.
    pub fn recommended() -> Self {
        let labels: Vec<String> = BUILTIN_PATTERNS.iter()
            .filter(|p| p.recommended)
            .map(|p| p.label.to_string())
            .collect();
        Self::with_labels(&labels)
    }

    /// Build a layer from an explicit set of built-in pattern labels.
    /// Unknown labels are silently skipped.
    pub fn with_labels(labels: &[String]) -> Self {
        Self::with_labels_and_custom(labels, &[])
    }

    /// Build a layer from built-in labels plus user-defined custom entities.
    /// Custom entities with bad regexes (won't compile) are dropped silently —
    /// validation should happen at save time, not at construction time.
    pub fn with_labels_and_custom(labels: &[String], custom: &[CustomEntity]) -> Self {
        let mut regex_patterns: Vec<(String, Regex)> = Vec::new();
        let mut ac_strings: Vec<String> = Vec::new();
        let mut ac_pattern_to_label: Vec<String> = Vec::new();

        for label in labels {
            if let Some(p) = get_builtin(label) {
                for pat in &p.regex_patterns {
                    if let Ok(rx) = Regex::new(pat) {
                        regex_patterns.push((p.label.clone(), rx));
                    }
                }
                for ctx in &p.context_phrases {
                    ac_strings.push(ctx.clone());
                    ac_pattern_to_label.push(p.label.clone());
                }
            }
        }

        for c in custom {
            for pat in &c.regex_patterns {
                if let Ok(rx) = Regex::new(pat) {
                    regex_patterns.push((c.label.clone(), rx));
                }
            }
            for ctx in &c.context_phrases {
                ac_strings.push(ctx.to_lowercase());
                ac_pattern_to_label.push(c.label.clone());
            }
        }

        let ac = if ac_strings.is_empty() {
            // Build with one harmless dummy so the empty-namespace path doesn't panic.
            // The pattern_to_label mapping is never consulted in this case anyway
            // because find_overlapping_iter on a query won't match a placeholder.
            AhoCorasick::new(&["\u{0001}".to_string()]).expect("placeholder AC builds")
        } else {
            AhoCorasick::builder()
                .ascii_case_insensitive(true)
                .build(&ac_strings)
                .expect("AC patterns compile")
        };

        Self { regex_patterns, ac, ac_pattern_to_label }
    }

    /// Detect all entity-type labels present in the query.
    /// Returns deduplicated labels in detection order.
    pub fn detect(&self, query: &str) -> Vec<String> {
        let mut hits: Vec<String> = Vec::new();
        for (label, re) in &self.regex_patterns {
            if re.is_match(query) && !hits.contains(label) {
                hits.push(label.clone());
            }
        }
        for m in self.ac.find_overlapping_iter(query) {
            let label = &self.ac_pattern_to_label[m.pattern().as_usize()];
            if !hits.contains(label) { hits.push(label.clone()); }
        }
        hits
    }

    /// Return the query augmented with detected entity tokens appended after.
    /// Tokens use the `mr_pii_<label>` convention (see comment above).
    pub fn augment(&self, query: &str) -> String {
        let labels = self.detect(query);
        if labels.is_empty() { return query.to_string(); }
        let suffix: String = labels.iter()
            .map(|l| format!(" mr_pii_{}", l.to_lowercase()))
            .collect();
        format!("{}{}", query, suffix)
    }

    /// The token that the augment pass emits for a given entity label.
    /// Use when seeding intents that should match entity-tagged queries.
    pub fn entity_token(label: &str) -> String {
        format!("mr_pii_{}", label.to_lowercase())
    }

    /// Find every entity span in the query — label + position + value.
    pub fn detect_with_spans<'a>(&self, query: &'a str) -> Vec<EntitySpan<'a>> {
        let mut spans = Vec::new();
        for (label, re) in &self.regex_patterns {
            for m in re.find_iter(query) {
                spans.push(EntitySpan {
                    label: label.clone(),
                    value: &query[m.start()..m.end()],
                    start: m.start(),
                    end: m.end(),
                    source: SpanSource::Value,
                });
            }
        }
        for m in self.ac.find_overlapping_iter(query) {
            let label = self.ac_pattern_to_label[m.pattern().as_usize()].clone();
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
    /// Only includes value-level matches (regex hits).
    pub fn extract<'a>(&self, query: &'a str) -> HashMap<String, Vec<&'a str>> {
        let mut out: HashMap<String, Vec<&'a str>> = HashMap::new();
        for span in self.detect_with_spans(query) {
            if matches!(span.source, SpanSource::Value) {
                out.entry(span.label).or_default().push(span.value);
            }
        }
        out
    }

    /// Replace every detected entity VALUE in the query with a placeholder.
    /// Context-word matches are left intact (preserving sentence meaning).
    pub fn mask<F>(&self, query: &str, mut placeholder_for: F) -> String
    where F: FnMut(&str) -> String {
        let mut value_spans: Vec<EntitySpan> = self.detect_with_spans(query)
            .into_iter()
            .filter(|s| matches!(s.source, SpanSource::Value))
            .collect();
        value_spans.sort_by_key(|s| s.start);

        let mut deduped: Vec<EntitySpan> = Vec::with_capacity(value_spans.len());
        let mut cursor = 0usize;
        for span in value_spans {
            if span.start >= cursor {
                cursor = span.end;
                deduped.push(span);
            }
        }

        let mut out = String::with_capacity(query.len());
        let mut pos = 0usize;
        for span in deduped {
            out.push_str(&query[pos..span.start]);
            out.push_str(&placeholder_for(&span.label));
            pos = span.end;
        }
        out.push_str(&query[pos..]);
        out
    }
}

impl Default for EntityLayer {
    fn default() -> Self { Self::recommended() }
}

/// One detected entity occurrence in the query.
#[derive(Debug, Clone)]
pub struct EntitySpan<'a> {
    pub label: String,
    pub value: &'a str,
    pub start: usize,
    pub end: usize,
    pub source: SpanSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanSource {
    Value,
    Context,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_more_than_30_patterns() {
        assert!(BUILTIN_PATTERNS.len() >= 30,
            "expected 30+ builtin patterns, got {}", BUILTIN_PATTERNS.len());
    }

    #[test]
    fn all_registry_regexes_compile() {
        for p in BUILTIN_PATTERNS.iter() {
            for pat in &p.regex_patterns {
                Regex::new(pat).unwrap_or_else(|e|
                    panic!("{} regex {:?} won't compile: {}", p.label, pat, e));
            }
        }
    }

    #[test]
    fn detects_credit_card_value() {
        let e = EntityLayer::default_pii();
        assert!(e.detect("save my card 4111-1111-1111-1111 for next time").contains(&"CC".to_string()));
    }

    #[test]
    fn detects_credit_card_context() {
        let e = EntityLayer::default_pii();
        assert!(e.detect("we should never store credit cards").contains(&"CC".to_string()));
    }

    #[test]
    fn detects_ssn() {
        let e = EntityLayer::default_pii();
        assert!(e.detect("my SSN is 123-45-6789 please file the taxes").contains(&"SSN".to_string()));
    }

    #[test]
    fn detects_email_value_without_context() {
        let e = EntityLayer::default_pii();
        assert!(e.detect("forward to alice@example.com when ready").contains(&"EMAIL".to_string()));
    }

    #[test]
    fn detects_password_context_without_value() {
        let e = EntityLayer::default_pii();
        assert!(e.detect("my password is hunter2").contains(&"SECRET".to_string()));
    }

    #[test]
    fn rejects_pii_adjacent_negatives() {
        let e = EntityLayer::default_pii();
        assert!(!e.detect("ticket number 4111-2222 was closed").contains(&"CC".to_string()));
    }

    #[test]
    fn rejects_normal_queries() {
        let e = EntityLayer::default_pii();
        assert!(e.detect("create a new pull request").is_empty());
    }

    #[test]
    fn detects_multiple_entities_in_one_query() {
        let e = EntityLayer::default_pii();
        let labels = e.detect("send 4111-1111-1111-1111 to alice@example.com tomorrow");
        assert!(labels.contains(&"CC".to_string()));
        assert!(labels.contains(&"EMAIL".to_string()));
    }

    // ── Recommended preset ───────────────────────────────────────────────────

    #[test]
    fn recommended_includes_credentials() {
        let e = EntityLayer::recommended();
        assert!(e.detect("my AWS key is AKIAIOSFODNN7EXAMPLE").contains(&"AWS_ACCESS_KEY".to_string()));
        assert!(e.detect("ghp_1234567890abcdefghijklmnopqrstuvwxyz1234").contains(&"GITHUB_PAT".to_string()));
    }

    #[test]
    fn recommended_detects_jwt() {
        let e = EntityLayer::recommended();
        let q = "token is eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
        assert!(e.detect(q).contains(&"JWT".to_string()));
    }

    // ── with_labels ──────────────────────────────────────────────────────────

    #[test]
    fn with_labels_only_loads_specified_patterns() {
        let e = EntityLayer::with_labels(&["EMAIL".to_string()]);
        assert!(e.detect("alice@example.com").contains(&"EMAIL".to_string()));
        // SSN not enabled — should not match.
        assert!(!e.detect("123-45-6789").contains(&"SSN".to_string()));
    }

    #[test]
    fn with_labels_silently_skips_unknown() {
        let e = EntityLayer::with_labels(&["EMAIL".to_string(), "NONEXISTENT".to_string()]);
        assert!(e.detect("alice@example.com").contains(&"EMAIL".to_string()));
    }

    // ── Custom entities ─────────────────────────────────────────────────────

    #[test]
    fn custom_entity_detects_after_construction() {
        let custom = vec![CustomEntity {
            label: "PATIENT_ID".to_string(),
            display_name: "Hospital patient ID".to_string(),
            description: "PT-NNNNNNN format".to_string(),
            regex_patterns: vec![r"\bPT-\d{7}\b".to_string()],
            context_phrases: vec!["patient id".to_string(), "patient identifier".to_string()],
            examples: vec!["PT-1234567".to_string()],
            source: "manual".to_string(),
        }];
        let e = EntityLayer::with_labels_and_custom(&[], &custom);
        assert!(e.detect("PT-1234567 is the record").contains(&"PATIENT_ID".to_string()));
        assert!(e.detect("patient id is needed").contains(&"PATIENT_ID".to_string()));
    }

    #[test]
    fn custom_entity_with_bad_regex_is_silently_dropped() {
        let custom = vec![CustomEntity {
            label: "BAD".to_string(),
            display_name: "Bad".to_string(),
            description: "".to_string(),
            regex_patterns: vec!["[unclosed".to_string()],
            context_phrases: vec![],
            examples: vec![],
            source: "manual".to_string(),
        }];
        // Should not panic; should just have no pattern for BAD.
        let e = EntityLayer::with_labels_and_custom(&[], &custom);
        assert!(e.detect("anything").is_empty());
    }

    // ── Augment / extract / mask preserved from earlier API ─────────────────

    #[test]
    fn augment_appends_distinctive_tokens() {
        let e = EntityLayer::default_pii();
        let augmented = e.augment("my SSN is 123-45-6789");
        assert!(augmented.contains("mr_pii_ssn"), "got: {}", augmented);
    }

    #[test]
    fn extract_returns_credit_card_value() {
        let e = EntityLayer::default_pii();
        let extracted = e.extract("save my card 4111-1111-1111-1111 for next time");
        assert_eq!(extracted.get("CC"), Some(&vec!["4111-1111-1111-1111"]));
    }

    #[test]
    fn mask_replaces_with_placeholder() {
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
        let masked = e.mask("we should never store credit cards in plaintext",
            |label| format!("<{}>", label));
        assert_eq!(masked, "we should never store credit cards in plaintext");
    }

    #[test]
    fn detect_with_spans_returns_value_and_context_separately() {
        let e = EntityLayer::default_pii();
        let spans = e.detect_with_spans("my credit card 4111-1111-1111-1111 is on file");
        let value_spans: Vec<_> = spans.iter().filter(|s| s.source == SpanSource::Value).collect();
        let context_spans: Vec<_> = spans.iter().filter(|s| s.source == SpanSource::Context).collect();
        assert!(!value_spans.is_empty());
        assert!(!context_spans.is_empty());
    }

    #[test]
    fn entity_token_helper_matches_what_augment_emits() {
        let token = EntityLayer::entity_token("CC");
        assert_eq!(token, "mr_pii_cc");
    }
}
