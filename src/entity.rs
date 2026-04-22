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

/// Metadata for a built-in entity pattern. The static registry below holds
/// one of these per recognized type. Patterns are organized by category
/// (PII, Credentials, Identifiers, Web/Tech, Financial, Misc).
#[derive(Debug, Clone, Copy)]
pub struct BuiltinPattern {
    /// Stable identifier — used in API responses, persistence, and as the
    /// suffix of the emitted `mr_pii_<label>` token (lowercased).
    pub label: &'static str,
    /// Display category for UI grouping ("PII", "Credentials", etc.).
    pub category: &'static str,
    /// Human-readable name for UI display.
    pub display_name: &'static str,
    /// Short description shown in the UI tooltip.
    pub description: &'static str,
    /// Value-pattern regex strings (Rust syntax). May be empty if the entity
    /// is detected only via context phrases.
    pub regex_patterns: &'static [&'static str],
    /// Context-phrase strings for Aho-Corasick (lowercase). May be empty.
    pub context_phrases: &'static [&'static str],
    /// Whether this is enabled in the "recommended" preset for general use.
    /// Customers can override via per-namespace config.
    pub recommended: bool,
}

/// The full registry of built-in entity patterns. Stable ordering for UI.
///
/// Categories: PII, Credentials, Identifiers, Web/Tech, Financial, Misc.
pub const BUILTIN_PATTERNS: &[BuiltinPattern] = &[
    // ── PII ─────────────────────────────────────────────────────────────────
    BuiltinPattern {
        label: "CC", category: "PII",
        display_name: "Credit card",
        description: "Credit card numbers (13-19 digits with optional separators).",
        regex_patterns: &[r"\b(?:\d[ -]?){12,18}\d\b"],
        context_phrases: &["credit card", "card number", "cc number", "visa", "mastercard", "amex", "american express", "discover card"],
        recommended: true,
    },
    BuiltinPattern {
        label: "SSN", category: "PII",
        display_name: "US Social Security Number",
        description: "US SSN in 3-2-4 digit format (e.g., 123-45-6789).",
        regex_patterns: &[r"\b\d{3}-\d{2}-\d{4}\b"],
        context_phrases: &["ssn", "social security", "social security number", "social security #"],
        recommended: true,
    },
    BuiltinPattern {
        label: "EMAIL", category: "PII",
        display_name: "Email address",
        description: "RFC-5322-style email addresses.",
        regex_patterns: &[r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"],
        context_phrases: &["email", "e-mail", "email address", "email id"],
        recommended: true,
    },
    BuiltinPattern {
        label: "PHONE", category: "PII",
        display_name: "Phone number (US)",
        description: "US phone numbers with optional country code and separators.",
        regex_patterns: &[r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"],
        context_phrases: &["phone", "phone number", "cell number", "mobile number", "telephone", "contact number"],
        recommended: true,
    },
    BuiltinPattern {
        label: "PHONE_INTL", category: "PII",
        display_name: "Phone number (international)",
        description: "International phone numbers with country code (E.164-style).",
        regex_patterns: &[r"\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"],
        context_phrases: &["international phone", "country code", "international number"],
        recommended: false,
    },
    BuiltinPattern {
        label: "DOB", category: "PII",
        display_name: "Date of birth",
        description: "Common date-of-birth formats (YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY).",
        regex_patterns: &[
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        ],
        context_phrases: &["date of birth", "dob", "born on", "birthday", "birthdate"],
        recommended: false,
    },

    // ── Credentials & Secrets ───────────────────────────────────────────────
    BuiltinPattern {
        label: "AWS_ACCESS_KEY", category: "Credentials",
        display_name: "AWS access key ID",
        description: "AWS access key IDs (AKIA*, ASIA*, AROA* — 20 chars total).",
        regex_patterns: &[r"\b(?:AKIA|ASIA|AROA)[0-9A-Z]{16}\b"],
        context_phrases: &["aws access key", "access key id", "aws credentials", "aws_access_key_id"],
        recommended: true,
    },
    BuiltinPattern {
        label: "AWS_SECRET_KEY", category: "Credentials",
        display_name: "AWS secret access key (context-only)",
        description: "AWS secret keys are 40-char base64 — too generic to detect by value alone (collides with hashes, tokens). Detect via context only.",
        regex_patterns: &[],
        context_phrases: &["aws secret key", "secret access key", "aws_secret_access_key"],
        recommended: false,
    },
    BuiltinPattern {
        label: "GCP_KEY", category: "Credentials",
        display_name: "GCP API key",
        description: "Google Cloud API keys (39 chars, AIza prefix).",
        regex_patterns: &[r"\bAIza[0-9A-Za-z_-]{35}\b"],
        context_phrases: &["gcp api key", "google cloud key", "google api key", "firebase key"],
        recommended: true,
    },
    BuiltinPattern {
        label: "STRIPE_KEY", category: "Credentials",
        display_name: "Stripe API key",
        description: "Stripe secret keys (sk_live_*, sk_test_*, pk_live_*, etc.).",
        regex_patterns: &[r"\b(?:sk|pk|rk)_(?:live|test)_[0-9A-Za-z]{24,}\b"],
        context_phrases: &["stripe key", "stripe secret", "stripe api key", "stripe_secret_key"],
        recommended: true,
    },
    BuiltinPattern {
        label: "GITHUB_PAT", category: "Credentials",
        display_name: "GitHub personal access token",
        description: "GitHub tokens (ghp_, gho_, ghs_, ghu_, ghr_ prefixes).",
        regex_patterns: &[r"\b(?:ghp|gho|ghs|ghu|ghr)_[0-9A-Za-z]{36,}\b"],
        context_phrases: &["github token", "github pat", "personal access token", "github_token"],
        recommended: true,
    },
    BuiltinPattern {
        label: "SLACK_TOKEN", category: "Credentials",
        display_name: "Slack token",
        description: "Slack API tokens (xoxb-, xoxa-, xoxp-, xoxe-).",
        regex_patterns: &[r"\bxox[abprseou]-[0-9A-Za-z-]{10,}\b"],
        context_phrases: &["slack token", "slack bot token", "slack api token", "xoxb"],
        recommended: true,
    },
    BuiltinPattern {
        label: "OPENAI_KEY", category: "Credentials",
        display_name: "OpenAI API key",
        description: "OpenAI API keys (sk-* with 48+ chars).",
        regex_patterns: &[r"\bsk-[A-Za-z0-9]{20,}T3BlbkFJ[A-Za-z0-9]{20,}\b", r"\bsk-proj-[A-Za-z0-9_-]{40,}\b"],
        context_phrases: &["openai key", "openai api key", "openai_api_key"],
        recommended: true,
    },
    BuiltinPattern {
        label: "ANTHROPIC_KEY", category: "Credentials",
        display_name: "Anthropic API key",
        description: "Anthropic API keys (sk-ant-* prefix).",
        regex_patterns: &[r"\bsk-ant-[A-Za-z0-9_-]{32,}\b"],
        context_phrases: &["anthropic key", "anthropic api key", "claude api key"],
        recommended: true,
    },
    BuiltinPattern {
        label: "JWT", category: "Credentials",
        display_name: "JWT token",
        description: "JSON Web Tokens (three base64url segments, dot-separated).",
        regex_patterns: &[r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"],
        context_phrases: &["jwt", "jwt token", "json web token", "bearer token", "access token", "auth token"],
        recommended: true,
    },
    BuiltinPattern {
        label: "PRIVATE_KEY", category: "Credentials",
        display_name: "Private key (PEM)",
        description: "PEM-encoded private keys (RSA, EC, OpenSSH, etc.).",
        regex_patterns: &[r"-----BEGIN [A-Z ]*PRIVATE KEY-----"],
        context_phrases: &["private key", "rsa private key", "ec private key", "ssh private key"],
        recommended: true,
    },
    BuiltinPattern {
        label: "SECRET", category: "Credentials",
        display_name: "Generic credential mention",
        description: "Mentions of passwords, API keys, secrets — context-only (no value pattern).",
        regex_patterns: &[],
        context_phrases: &["password", "passcode", "api key", "secret key", "access token", "auth token", "client secret", "credentials"],
        recommended: true,
    },

    // ── Identifiers (US) ────────────────────────────────────────────────────
    BuiltinPattern {
        label: "US_PASSPORT", category: "Identifiers",
        display_name: "US passport number",
        description: "9-character US passport numbers, optionally prefixed with 'US'.",
        regex_patterns: &[r"\b(?:US)?[0-9]{9}\b"],
        context_phrases: &["passport number", "passport no", "us passport", "passport id"],
        recommended: false,
    },
    BuiltinPattern {
        label: "ZIP_CODE", category: "Identifiers",
        display_name: "US ZIP code",
        description: "5-digit US ZIP codes (with required ZIP+4 extension OR with surrounding context).",
        // The prior pattern matched ANY 5-digit number — too greedy.
        // Now require ZIP+4 form for confident regex match; bare 5-digit
        // matches must come through context phrases ("zip code 12345").
        regex_patterns: &[r"\b\d{5}-\d{4}\b"],
        context_phrases: &["zip code", "zipcode", "postal code", "zip:"],
        recommended: false,
    },
    BuiltinPattern {
        label: "US_DRIVERS_LICENSE", category: "Identifiers",
        display_name: "US driver's license",
        description: "US driver's license numbers — formats vary by state. Context-driven.",
        regex_patterns: &[r"\b[A-Z]\d{7,8}\b"],
        context_phrases: &["driver's license", "drivers license", "driving license", "dl number", "license number"],
        recommended: false,
    },
    BuiltinPattern {
        label: "US_EIN", category: "Identifiers",
        display_name: "US Employer ID Number",
        description: "EIN in 2-7 digit format (e.g., 12-3456789).",
        regex_patterns: &[r"\b\d{2}-\d{7}\b"],
        context_phrases: &["ein", "employer identification number", "tax id number"],
        recommended: false,
    },
    BuiltinPattern {
        label: "ITIN", category: "Identifiers",
        display_name: "US ITIN",
        description: "Individual Taxpayer Identification Number (9-digit, starts with 9).",
        regex_patterns: &[r"\b9\d{2}-\d{2}-\d{4}\b"],
        context_phrases: &["itin", "individual taxpayer", "tax identification"],
        recommended: false,
    },

    // ── Identifiers (International) ─────────────────────────────────────────
    BuiltinPattern {
        label: "UK_NHS", category: "Identifiers",
        display_name: "UK NHS number",
        description: "10-digit UK National Health Service number, often spaced 3-3-4.",
        regex_patterns: &[r"\b\d{3}\s?\d{3}\s?\d{4}\b"],
        context_phrases: &["nhs number", "nhs no", "national health service number"],
        recommended: false,
    },
    BuiltinPattern {
        label: "UK_NINO", category: "Identifiers",
        display_name: "UK National Insurance Number",
        description: "UK NI number (e.g., AB123456C).",
        regex_patterns: &[r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"],
        context_phrases: &["national insurance number", "ni number", "nino"],
        recommended: false,
    },
    BuiltinPattern {
        label: "IN_PAN", category: "Identifiers",
        display_name: "India PAN",
        description: "Indian PAN: 5 letters + 4 digits + 1 letter.",
        regex_patterns: &[r"\b[A-Z]{5}\d{4}[A-Z]\b"],
        context_phrases: &["pan number", "pan card", "permanent account number"],
        recommended: false,
    },
    BuiltinPattern {
        label: "IN_AADHAAR", category: "Identifiers",
        display_name: "India Aadhaar",
        description: "12-digit Indian Aadhaar number, often spaced 4-4-4.",
        regex_patterns: &[r"\b\d{4}\s?\d{4}\s?\d{4}\b"],
        context_phrases: &["aadhaar number", "aadhar number", "uid", "unique id"],
        recommended: false,
    },

    // ── Financial ───────────────────────────────────────────────────────────
    BuiltinPattern {
        label: "IBAN", category: "Financial",
        display_name: "IBAN",
        description: "International Bank Account Number (15-34 chars, country prefix).",
        regex_patterns: &[r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"],
        context_phrases: &["iban", "iban code", "iban number", "international bank account"],
        recommended: false,
    },
    BuiltinPattern {
        label: "BTC_ADDRESS", category: "Financial",
        display_name: "Bitcoin address",
        description: "Bitcoin addresses (P2PKH, P2SH, Bech32).",
        regex_patterns: &[
            r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",
            r"\bbc1[a-z0-9]{39,59}\b",
        ],
        context_phrases: &["bitcoin address", "btc address", "wallet address"],
        recommended: false,
    },
    BuiltinPattern {
        label: "ETH_ADDRESS", category: "Financial",
        display_name: "Ethereum address",
        description: "Ethereum addresses (0x-prefixed, 40 hex chars).",
        regex_patterns: &[r"\b0x[a-fA-F0-9]{40}\b"],
        context_phrases: &["ethereum address", "eth address", "wallet address", "metamask"],
        recommended: false,
    },

    // ── Web / Tech ──────────────────────────────────────────────────────────
    BuiltinPattern {
        label: "IPV4", category: "Web/Tech",
        display_name: "IPv4 address",
        description: "IPv4 addresses with valid octet ranges (rejects 999.999.999.999 / version strings).",
        // Each octet must be 0-255; the previous \d{1,3} matched version strings
        // like "1.2.3.4" and "192.168.500.1". This rejects out-of-range octets.
        regex_patterns: &[r"\b(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"],
        context_phrases: &["ip address", "ipv4", "ip:"],
        recommended: true,
    },
    BuiltinPattern {
        label: "IPV6", category: "Web/Tech",
        display_name: "IPv6 address",
        description: "IPv6 addresses (full and compressed forms).",
        regex_patterns: &[r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b"],
        context_phrases: &["ipv6", "ip address"],
        recommended: false,
    },
    BuiltinPattern {
        label: "MAC_ADDRESS", category: "Web/Tech",
        display_name: "MAC address",
        description: "Hardware MAC addresses (colon or hyphen separated).",
        regex_patterns: &[r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"],
        context_phrases: &["mac address", "hardware address", "ethernet address"],
        recommended: false,
    },
    BuiltinPattern {
        label: "URL", category: "Web/Tech",
        display_name: "URL",
        description: "HTTP/HTTPS URLs requiring a TLD (rejects 'http://localhost' shorthand).",
        // Prior pattern matched anything starting with http:// — including
        // version-numbered protocol mentions and URL-shaped log entries.
        // Now require: scheme + valid host with at least one dot + TLD ≥2 chars.
        regex_patterns: &[r"\bhttps?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:[/?#][^\s]*)?\b"],
        context_phrases: &["url", "link", "website"],
        recommended: false,
    },
    BuiltinPattern {
        label: "UUID", category: "Web/Tech",
        display_name: "UUID",
        description: "UUIDs (any version, 8-4-4-4-12 hex format).",
        regex_patterns: &[r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"],
        context_phrases: &["uuid", "guid", "unique id"],
        recommended: false,
    },
    BuiltinPattern {
        label: "SHA256", category: "Web/Tech",
        display_name: "SHA-256 hash (context-only)",
        description: "Bare 64-char hex strings collide with too many things — detect via context.",
        regex_patterns: &[],
        context_phrases: &["sha256", "sha-256", "sha256 hash", "sha256:"],
        recommended: false,
    },
    BuiltinPattern {
        label: "MD5", category: "Web/Tech",
        display_name: "MD5 hash (context-only)",
        description: "Bare 32-char hex strings collide with too many things — detect via context.",
        regex_patterns: &[],
        context_phrases: &["md5", "md5 hash", "md5:"],
        recommended: false,
    },

    // ── Misc structured ─────────────────────────────────────────────────────
    BuiltinPattern {
        label: "ADDRESS", category: "PII",
        display_name: "Postal address (context)",
        description: "Mentions of street/home/postal addresses — context only.",
        regex_patterns: &[],
        context_phrases: &["home address", "street address", "postal address", "mailing address", "shipping address"],
        recommended: false,
    },
];

/// Look up a built-in pattern by label.
pub fn get_builtin(label: &str) -> Option<&'static BuiltinPattern> {
    BUILTIN_PATTERNS.iter().find(|p| p.label == label)
}

/// All distinct categories in the registry, in declaration order.
pub fn all_categories() -> Vec<&'static str> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for p in BUILTIN_PATTERNS {
        if seen.insert(p.category) { out.push(p.category); }
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
                for pat in p.regex_patterns {
                    if let Ok(rx) = Regex::new(pat) {
                        regex_patterns.push((p.label.to_string(), rx));
                    }
                }
                for ctx in p.context_phrases {
                    ac_strings.push(ctx.to_string());
                    ac_pattern_to_label.push(p.label.to_string());
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
        for p in BUILTIN_PATTERNS {
            for pat in p.regex_patterns {
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
