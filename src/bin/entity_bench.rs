//! Entity Layer PoC bake-off: regex vs Aho-Corasick vs char n-gram.
//!
//! Compares three approaches to entity detection (PII, jailbreak markers, etc.)
//! that could augment MicroResolve's L2 scoring with entity-typed tokens like
//! [CC], [SSN], [EMAIL] before normal token matching.
//!
//! For each detector:
//!   - Detection rate on positive (entity-bearing) queries
//!   - False positive rate on negative (entity-free) queries
//!   - Per-query latency
//!   - One-time build cost & memory footprint
//!
//! Run: `cargo run --release --bin entity_bench`
//!
//! NOTE: This is a directional PoC, not a production benchmark. See
//! ENTITY_LAYER_PLAN.md for Phase 2 industry-standard validation.

use aho_corasick::AhoCorasick;
use regex::Regex;
use std::time::Instant;

// ─── Common detector interface ────────────────────────────────────────────────

/// One detector emits a set of entity-type labels for a query
/// (e.g., ["CC", "EMAIL"] when both a credit card and email are detected).
trait EntityDetector {
    fn name(&self) -> &str;
    fn detect(&self, query: &str) -> Vec<&'static str>;
    fn build_micros(&self) -> u128;
    fn approx_bytes(&self) -> usize;
}

// ─── 1) Regex detector ────────────────────────────────────────────────────────
// Standard PII/secret patterns. Stable, well-known, zero learning required.

struct RegexDetector {
    patterns: Vec<(&'static str, Regex)>,
    build_micros: u128,
    bytes: usize,
}

impl RegexDetector {
    fn new() -> Self {
        let t0 = Instant::now();
        let raw: &[(&'static str, &str)] = &[
            // Credit card: 13–19 digits with optional separators (basic Luhn would tighten this).
            ("CC", r"\b(?:\d[ -]?){12,18}\d\b"),
            // SSN: 3-2-4 digits.
            ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
            // Email.
            ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            // US phone: optional country code, area code, then 7 digits with separators.
            ("PHONE", r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            // IPv4.
            ("IPV4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            // API-key-ish: long base64-ish or hex-ish runs.
            ("APIKEY", r"\b[A-Za-z0-9_\-]{32,}\b"),
        ];
        let bytes = raw.iter().map(|(_, p)| p.len()).sum::<usize>() * 4; // very rough
        let patterns: Vec<_> = raw.iter()
            .map(|(label, p)| (*label, Regex::new(p).unwrap()))
            .collect();
        Self { patterns, build_micros: t0.elapsed().as_micros(), bytes }
    }
}

impl EntityDetector for RegexDetector {
    fn name(&self) -> &str { "regex" }
    fn detect(&self, query: &str) -> Vec<&'static str> {
        let mut hits = Vec::new();
        for (label, re) in &self.patterns {
            if re.is_match(query) && !hits.contains(label) {
                hits.push(*label);
            }
        }
        hits
    }
    fn build_micros(&self) -> u128 { self.build_micros }
    fn approx_bytes(&self) -> usize { self.bytes }
}

// ─── 2) Aho-Corasick detector (context words) ─────────────────────────────────
// Doesn't match the entity VALUE — matches the surrounding language.
// "my credit card is X" → CC, even if X isn't a digit pattern.
// Catches what regex misses (typo'd values, partial entities) but not values
// presented without context words.

struct AhoCorasickDetector {
    ac: AhoCorasick,
    pattern_to_label: Vec<&'static str>,
    build_micros: u128,
    bytes: usize,
}

impl AhoCorasickDetector {
    fn new() -> Self {
        let t0 = Instant::now();
        let raw: &[(&'static str, &str)] = &[
            // Credit-card context.
            ("CC", "credit card"), ("CC", "card number"), ("CC", "cc number"),
            ("CC", "visa"), ("CC", "mastercard"), ("CC", "amex"),
            // SSN context.
            ("SSN", "ssn"), ("SSN", "social security"), ("SSN", "social security number"),
            // Email context.
            ("EMAIL", "email"), ("EMAIL", "e-mail"), ("EMAIL", "email address"),
            // Phone context.
            ("PHONE", "phone"), ("PHONE", "phone number"), ("PHONE", "cell number"),
            ("PHONE", "mobile number"),
            // Password / secret context.
            ("SECRET", "password"), ("SECRET", "passcode"), ("SECRET", "api key"),
            ("SECRET", "secret key"), ("SECRET", "access token"), ("SECRET", "auth token"),
            // Address context.
            ("ADDRESS", "home address"), ("ADDRESS", "street address"), ("ADDRESS", "zip code"),
            ("ADDRESS", "postal code"),
        ];
        let patterns: Vec<&str> = raw.iter().map(|(_, p)| *p).collect();
        let pattern_to_label: Vec<&'static str> = raw.iter().map(|(l, _)| *l).collect();
        let bytes = patterns.iter().map(|p| p.len()).sum::<usize>() * 4;
        let ac = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(patterns)
            .unwrap();
        Self { ac, pattern_to_label, build_micros: t0.elapsed().as_micros(), bytes }
    }
}

impl EntityDetector for AhoCorasickDetector {
    fn name(&self) -> &str { "aho-corasick" }
    fn detect(&self, query: &str) -> Vec<&'static str> {
        let mut hits = Vec::new();
        for m in self.ac.find_overlapping_iter(query) {
            let label = self.pattern_to_label[m.pattern().as_usize()];
            if !hits.contains(&label) { hits.push(label); }
        }
        hits
    }
    fn build_micros(&self) -> u128 { self.build_micros }
    fn approx_bytes(&self) -> usize { self.bytes }
}

// ─── 3) Char n-gram classifier (learned per entity type) ─────────────────────
// LLM-distillation analog: imagine LLM gave us 100 examples of each entity
// type at training time. We build a char n-gram frequency profile per type,
// then at inference we slide a window and ask "which profile does this look
// most like?" Highest-density window above threshold → emit that label.
//
// For the PoC we use synthetic training data (a handful of examples per type)
// so we can compare structurally without setting up an LLM call.

struct CharNgramDetector {
    /// For each entity type, the SET of trigrams seen in any training example.
    profiles: Vec<(&'static str, std::collections::HashSet<String>)>,
    build_micros: u128,
    bytes: usize,
    /// Minimum fraction of a token's trigrams that must be in the entity
    /// profile for the token to be classified as that entity.
    coverage_threshold: f32,
}

impl CharNgramDetector {
    fn new() -> Self {
        let t0 = Instant::now();

        // Synthetic training — stand-in for what LLM would generate.
        let training: Vec<(&'static str, Vec<&str>)> = vec![
            ("CC", vec![
                "4111-1111-1111-1111", "5500-0000-0000-0004", "5500 0000 0000 0004",
                "340000000000009", "4012888888881881", "3782 822463 10005",
                "6011000000000004", "5105105105105100", "4222222222222",
                "378282246310005", "4111111111111111", "5555555555554444",
            ]),
            ("SSN", vec![
                "123-45-6789", "987-65-4321", "111-22-3333", "555-12-3456",
                "078-05-1120", "219-09-9999", "457-55-5462", "001-01-0001",
                "999-99-9999", "234-56-7890",
            ]),
            ("EMAIL", vec![
                "alice@example.com", "bob.smith@company.org", "user+tag@gmail.com",
                "first.last@subdomain.example.co.uk", "support@some-site.io",
                "dev@localhost.test", "info@business.net", "x@y.io",
            ]),
            ("PHONE", vec![
                "(555) 123-4567", "+1-202-555-0173", "212-555-1234", "(800) 555-0199",
                "+1 415 555 2671", "555.123.4567", "(212)555-1212", "1-800-555-0100",
            ]),
            ("IPV4", vec![
                "192.168.1.1", "10.0.0.1", "172.16.0.1", "8.8.8.8", "127.0.0.1",
                "255.255.255.0", "192.168.100.50", "203.0.113.42",
            ]),
        ];

        let mut profiles = Vec::new();
        for (label, examples) in &training {
            let mut ngs: std::collections::HashSet<String> = std::collections::HashSet::new();
            for ex in examples {
                for ng in char_ngrams(ex, 3) { ngs.insert(ng); }
            }
            profiles.push((*label, ngs));
        }

        let bytes: usize = profiles.iter()
            .map(|(_, s)| s.iter().map(|k| k.len()).sum::<usize>())
            .sum();

        Self {
            profiles,
            build_micros: t0.elapsed().as_micros(),
            bytes,
            // 70% of the token's trigrams must be in the entity profile.
            // Lower → more recall but more FPs (random words start matching).
            coverage_threshold: 0.70,
        }
    }
}

impl EntityDetector for CharNgramDetector {
    fn name(&self) -> &str { "char-ngram" }
    fn detect(&self, query: &str) -> Vec<&'static str> {
        let mut hits = Vec::new();
        for tok in query.split(|c: char| c.is_whitespace() || c == ',') {
            // Strip surrounding punctuation so "(555)" becomes "555" for fair scoring.
            let tok = tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '+' && c != '-' && c != '.');
            if tok.len() < 5 { continue; }
            let tok_ngs = char_ngrams(tok, 3);
            if tok_ngs.is_empty() { continue; }
            let tok_total = tok_ngs.len() as f32;

            for (label, profile) in &self.profiles {
                let matches = tok_ngs.iter().filter(|ng| profile.contains(*ng)).count();
                let coverage = matches as f32 / tok_total;
                if coverage >= self.coverage_threshold && !hits.contains(label) {
                    hits.push(*label);
                }
            }
        }
        hits
    }
    fn build_micros(&self) -> u128 { self.build_micros }
    fn approx_bytes(&self) -> usize { self.bytes }
}

// ─── 4) Hybrid: regex + Aho-Corasick (combined) ──────────────────────────────
// They are complementary: regex catches entity VALUES, AC catches entity
// CONTEXT WORDS. Combining them should pick up both ("send 4111-..." via
// regex AND "my credit card" via AC).

struct HybridDetector {
    regex: RegexDetector,
    ac: AhoCorasickDetector,
}

impl HybridDetector {
    fn new() -> Self {
        Self { regex: RegexDetector::new(), ac: AhoCorasickDetector::new() }
    }
}

impl EntityDetector for HybridDetector {
    fn name(&self) -> &str { "hybrid (re+ac)" }
    fn detect(&self, query: &str) -> Vec<&'static str> {
        let mut hits = self.regex.detect(query);
        for label in self.ac.detect(query) {
            if !hits.contains(&label) { hits.push(label); }
        }
        hits
    }
    fn build_micros(&self) -> u128 { self.regex.build_micros() + self.ac.build_micros() }
    fn approx_bytes(&self) -> usize { self.regex.approx_bytes() + self.ac.approx_bytes() }
}

fn char_ngrams(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < n { return vec![]; }
    chars.windows(n).map(|w| w.iter().collect::<String>()).collect()
}

// ─── Test set ────────────────────────────────────────────────────────────────
// Hand-crafted: positive examples with embedded entities, negative examples
// that look adjacent (numeric IDs, formatted strings) but should NOT trigger.
//
// Each entry: (query, expected_labels_or_empty).
// PoC scope only — see ENTITY_LAYER_PLAN.md for Phase 2 industry validation.

fn test_set() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        // ── Credit-card positives ────────────────────────────────────────────
        ("save my credit card 4111-1111-1111-1111 for next time",       vec!["CC"]),
        ("the card number is 5500 0000 0000 0004",                      vec!["CC"]),
        ("amex 3782 822463 10005 expired last week",                    vec!["CC"]),
        ("visa ending in 4222222222222",                                vec!["CC"]),
        ("can you update my mastercard 5105 1051 0510 5100",            vec!["CC"]),

        // ── SSN positives ────────────────────────────────────────────────────
        ("my SSN is 123-45-6789 please file the taxes",                 vec!["SSN"]),
        ("social security number 987-65-4321 is on the form",           vec!["SSN"]),
        ("send the social security 555-12-3456 over fax",               vec!["SSN"]),

        // ── Email positives ──────────────────────────────────────────────────
        ("forward to alice@example.com when ready",                     vec!["EMAIL"]),
        ("my email is bob.smith@company.org",                           vec!["EMAIL"]),
        ("contact support@some-site.io with questions",                 vec!["EMAIL"]),
        ("user+tag@gmail.com is the right address",                     vec!["EMAIL"]),

        // ── Phone positives ──────────────────────────────────────────────────
        ("call me at (555) 123-4567 tonight",                           vec!["PHONE"]),
        ("phone number +1-202-555-0173 reaches me anytime",             vec!["PHONE"]),
        ("his cell is 212-555-1234",                                    vec!["PHONE"]),

        // ── IPv4 positives ───────────────────────────────────────────────────
        ("server at 192.168.1.1 is unreachable",                        vec!["IPV4"]),
        ("the IP 8.8.8.8 is google DNS",                                vec!["IPV4"]),

        // ── Multi-entity positives ───────────────────────────────────────────
        ("send 4111-1111-1111-1111 to alice@example.com tomorrow",      vec!["CC", "EMAIL"]),
        ("ssn 123-45-6789 phone (555) 123-4567",                        vec!["SSN", "PHONE"]),

        // ── Secret/credential context (no embedded value) ────────────────────
        ("my password is hunter2",                                      vec!["SECRET"]),
        ("the api key for production",                                  vec!["SECRET"]),
        ("share the access token with the team",                        vec!["SECRET"]),

        // ── Negatives: PII-adjacent but not actual PII ──────────────────────
        ("ticket number 4111-2222 was closed",                          vec![]),
        ("issue #1234 is fixed",                                        vec![]),
        ("PR-9876-AB is ready for review",                              vec![]),
        ("order 5500-0000-0000 placed at noon",                         vec![]), // shaped like CC but too short
        ("version 1.2.3.4 was released",                                vec![]),
        ("the date is 2026-04-22",                                      vec![]),

        // ── Negatives: ordinary queries ─────────────────────────────────────
        ("send a message to the team channel",                          vec![]),
        ("what is the weather today",                                   vec![]),
        ("schedule a meeting for tomorrow",                             vec![]),
        ("create a new pull request",                                   vec![]),
        ("merge the feature branch",                                    vec![]),
        ("how do I reverse a linked list",                              vec![]),
        ("translate this paragraph to french",                          vec![]),
        ("can you summarize the document",                              vec![]),
        ("book a flight to new york",                                   vec![]),
        ("explain how machine learning works",                          vec![]),

        // ── Negatives: mention entity nouns without revealing values ─────────
        ("we should never store credit cards",                          vec!["CC"]), // arguable — mention alone is intent-relevant
        ("how do email addresses work in DNS",                          vec!["EMAIL"]), // arguable
        ("what is a social security number",                            vec!["SSN"]), // arguable
    ]
}

// ─── Bench runner ────────────────────────────────────────────────────────────

#[derive(Default, Debug)]
struct Result {
    tp: usize,    // entity correctly detected
    fp: usize,    // entity wrongly emitted
    fn_: usize,   // entity missed
    tn: usize,    // correctly emitted nothing for negative
    total_queries: usize,
    total_micros: u128,
    min_micros: u128,
    max_micros: u128,
    p99_micros: u128,
    extra_labels_per_query: f32, // average # of unexpected labels per query
}

fn evaluate<D: EntityDetector + ?Sized>(detector: &D, set: &[(&str, Vec<&'static str>)]) -> Result {
    let mut r = Result::default();
    r.min_micros = u128::MAX;
    let mut times = Vec::with_capacity(set.len());
    let mut extras = Vec::with_capacity(set.len());

    for (query, expected) in set {
        let t0 = Instant::now();
        let detected = detector.detect(query);
        let elapsed = t0.elapsed().as_micros();
        times.push(elapsed);
        r.total_micros += elapsed;
        r.min_micros = r.min_micros.min(elapsed);
        r.max_micros = r.max_micros.max(elapsed);

        let detected_set: std::collections::HashSet<_> = detected.iter().copied().collect();
        let expected_set: std::collections::HashSet<_> = expected.iter().copied().collect();

        // Per-label confusion accounting against this query.
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_ = 0;
        for label in &expected_set {
            if detected_set.contains(label) { tp += 1; } else { fn_ += 1; }
        }
        for label in &detected_set {
            if !expected_set.contains(label) { fp += 1; }
        }

        r.tp += tp;
        r.fp += fp;
        r.fn_ += fn_;

        // True negative: query expected nothing AND detector emitted nothing.
        if expected_set.is_empty() && detected_set.is_empty() { r.tn += 1; }

        extras.push(fp as f32);
    }

    times.sort();
    if !times.is_empty() {
        let p99_idx = ((times.len() as f64) * 0.99).ceil() as usize - 1;
        r.p99_micros = times[p99_idx.min(times.len() - 1)];
    }
    r.total_queries = set.len();
    r.extra_labels_per_query = extras.iter().sum::<f32>() / extras.len().max(1) as f32;
    r
}

fn precision(r: &Result) -> f32 {
    let denom = r.tp + r.fp;
    if denom == 0 { 0.0 } else { r.tp as f32 / denom as f32 }
}
fn recall(r: &Result) -> f32 {
    let denom = r.tp + r.fn_;
    if denom == 0 { 0.0 } else { r.tp as f32 / denom as f32 }
}
fn f1(r: &Result) -> f32 {
    let (p, rc) = (precision(r), recall(r));
    if p + rc == 0.0 { 0.0 } else { 2.0 * p * rc / (p + rc) }
}

fn main() {
    println!("\n=== Entity Detector Bake-Off (PoC) ===");
    println!("See ENTITY_LAYER_PLAN.md for Phase 2 industry validation plan.\n");

    let set = test_set();
    println!("Test set: {} queries\n", set.len());

    let regex = RegexDetector::new();
    let ac = AhoCorasickDetector::new();
    let ng = CharNgramDetector::new();
    let hy = HybridDetector::new();

    let detectors: Vec<&dyn EntityDetector> = vec![&regex, &ac, &ng, &hy];

    println!("{:<12} {:>6} {:>6} {:>6} {:>9} {:>9} {:>8} {:>9} {:>9} {:>9}",
        "detector", "TP", "FP", "FN", "precision", "recall", "F1",
        "median µs", "p99 µs", "max µs");
    println!("{}", "─".repeat(95));

    let mut results = Vec::new();
    for d in &detectors {
        let r = evaluate(*d, set.as_slice());
        let median = if r.total_queries > 0 { r.total_micros / r.total_queries as u128 } else { 0 };
        println!("{:<12} {:>6} {:>6} {:>6} {:>9.2} {:>9.2} {:>8.2} {:>9} {:>9} {:>9}",
            d.name(), r.tp, r.fp, r.fn_,
            precision(&r), recall(&r), f1(&r),
            median, r.p99_micros, r.max_micros);
        results.push((d.name().to_string(), r));
    }

    println!();
    println!("Build cost & memory:");
    println!("{:<12} {:>14} {:>16}", "detector", "build µs", "approx bytes");
    println!("{}", "─".repeat(46));
    for d in &detectors {
        println!("{:<12} {:>14} {:>16}", d.name(), d.build_micros(), d.approx_bytes());
    }

    println!("\nNotes:");
    println!("  TP: detector emitted an expected label");
    println!("  FP: detector emitted a label not in the expected set");
    println!("  FN: expected label was missed");
    println!("  Precision and recall are computed over (query, label) pairs, not queries.");
    println!("  Char-ngram trained on synthetic data (~5–9 examples per entity type)");
    println!("  for structural comparison; production would use LLM-generated examples.");

    // ── Per-detector confusion against expected entity types ──────────────────
    println!("\nPer-query agreement (positive queries only — what each found vs expected):");
    let positives: Vec<_> = set.iter().filter(|(_, e)| !e.is_empty()).collect();
    println!("{:<60} {:<12} {:<14} {:<14} {:<14}",
        "query", "expected", "regex", "aho-corasick", "char-ngram");
    println!("{}", "─".repeat(120));
    for (q, expected) in positives.iter().take(15) {
        let exp_str = expected.join(",");
        let r1 = regex.detect(q).join(",");
        let r2 = ac.detect(q).join(",");
        let r3 = ng.detect(q).join(",");
        let qq: String = if q.len() > 58 { format!("{}…", &q[..57]) } else { q.to_string() };
        println!("{:<60} {:<12} {:<14} {:<14} {:<14}", qq, exp_str, r1, r2, r3);
    }
}
