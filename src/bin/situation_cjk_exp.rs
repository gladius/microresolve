//! CJK + Mixed-Script Situation→Action Inference — Option B Experiment
//!
//! Option B: Direct substring matching on raw text (no tokenization).
//! Seeds are key compound patterns (2-4 chars CJK, or English tech terms).
//! Multiple pattern hits per query accumulate votes per intent.
//!
//! Why this works where bigram overlap fails:
//!   "付款" IS a substring of "付款一直失败" ✓
//!   "款失" is NOT a substring of "付款一直失败" ✓ (no false bridge bigram)
//!   "build" IS a substring of "commit之后build挂了" ✓
//!
//! Seeds and tests are loaded from external JSON files — no recompilation needed.
//!
//! Usage:
//!   cargo run --bin situation_cjk_exp
//!     (defaults to data/situation/cjk)
//!   cargo run --bin situation_cjk_exp -- --data data/situation/mixed
//!     (mixed CJK + English tech terms)

use std::collections::HashMap;
use serde::Deserialize;
use serde_json::Value;

// ─── Seed storage ─────────────────────────────────────────────────────────────

struct CjkStore {
    /// (app_id, intent_id) → [(pattern_string, weight)]
    patterns: HashMap<(String, String), Vec<(String, f32)>>,
}

impl CjkStore {
    fn new() -> Self {
        Self { patterns: HashMap::new() }
    }

    fn add(&mut self, app: &str, intent: &str, pattern: &str, weight: f32) {
        if pattern.is_empty() { return; }
        let entry = self.patterns
            .entry((app.to_string(), intent.to_string()))
            .or_default();
        // avoid duplicates
        if !entry.iter().any(|(p, _)| p == pattern) {
            entry.push((pattern.to_string(), weight));
        }
    }

    /// Option B scoring: sum of (weight × sqrt(char_count)) for each pattern
    /// found as a direct substring of the query (case-insensitive for Latin parts).
    fn score_query(&self, query: &str) -> Vec<ScoredIntent> {
        let query_lower = query.to_lowercase();
        let mut results: Vec<ScoredIntent> = Vec::new();

        for ((app, intent), patterns) in &self.patterns {
            let mut score = 0.0f32;
            let mut best_match = String::new();
            let mut best_contrib = 0.0f32;

            for (pattern, weight) in patterns {
                // Try exact match first, then case-insensitive for Latin patterns
                let matched = query.contains(pattern.as_str())
                    || query_lower.contains(pattern.to_lowercase().as_str());

                if matched {
                    let char_len = pattern.chars().count() as f32;
                    let contribution = weight * char_len.sqrt();
                    score += contribution;
                    if contribution > best_contrib {
                        best_contrib = contribution;
                        best_match = pattern.clone();
                    }
                }
            }

            if score > 0.0 {
                results.push(ScoredIntent {
                    app_id: app.clone(),
                    intent_id: intent.clone(),
                    score,
                    best_match,
                });
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    /// Aggregate intent scores per app (sum across all intents in an app).
    fn detect_app(&self, query: &str) -> Vec<(String, f32)> {
        let scored = self.score_query(query);
        let mut app_scores: HashMap<String, f32> = HashMap::new();
        for s in &scored {
            *app_scores.entry(s.app_id.clone()).or_insert(0.0) += s.score;
        }
        let mut apps: Vec<(String, f32)> = app_scores.into_iter().collect();
        apps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        apps
    }

    /// Learn from a failed query: extract char 2-grams and 3-grams, add top N as patterns.
    fn learn(&mut self, query: &str, app: &str, intent: &str) -> usize {
        let existing: std::collections::HashSet<String> = self.patterns
            .get(&(app.to_string(), intent.to_string()))
            .map(|v| v.iter().map(|(p, _)| p.clone()).collect())
            .unwrap_or_default();

        // Extract meaningful characters (CJK + alphanumeric), skip punctuation/spaces
        let meaningful: Vec<char> = query.chars()
            .filter(|c| is_meaningful_char(*c))
            .collect();

        let mut candidates: Vec<String> = Vec::new();
        // 3-char substrings first (more specific)
        for n in [3usize, 2usize] {
            if meaningful.len() >= n {
                for i in 0..=(meaningful.len() - n) {
                    let s: String = meaningful[i..i+n].iter().collect();
                    if !existing.contains(&s) {
                        candidates.push(s);
                    }
                }
            }
        }
        candidates.dedup();

        let mut added = 0;
        for candidate in &candidates {
            if added >= 4 { break; }
            self.add(app, intent, candidate, 0.4);
            added += 1;
        }
        added
    }

    fn pattern_count(&self) -> usize {
        self.patterns.values().map(|v| v.len()).sum()
    }
}

fn is_meaningful_char(c: char) -> bool {
    (c >= '\u{4E00}' && c <= '\u{9FFF}')   // CJK Unified Ideographs
    || (c >= '\u{3040}' && c <= '\u{30FF}') // Hiragana + Katakana
    || (c >= '\u{AC00}' && c <= '\u{D7AF}') // Korean Hangul
    || c.is_ascii_alphanumeric()             // Latin (for mixed-script)
}

// ─── Types ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ScoredIntent {
    app_id: String,
    intent_id: String,
    score: f32,
    best_match: String,
}

#[derive(Deserialize, Clone)]
struct TestCase {
    query: String,
    app: String,
    intent: String,
    category: String,
    #[serde(default)]
    note: String,
}

struct EvalResult {
    app_correct: bool,
    intent_correct: bool,
    correctly_silent: bool,
    top_app: String,
    top_intent: String,
    top_score: f32,
    best_match: String,
    all_firing: Vec<String>,  // all (app.intent score) above threshold
}

// ─── JSON loading ─────────────────────────────────────────────────────────────

fn load_seeds(path: &str) -> CjkStore {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Cannot read seeds file: {}", path));
    let v: Value = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("Bad seeds JSON: {}", e));

    let mut store = CjkStore::new();
    let obj = v.as_object().expect("seeds JSON must be an object");

    for (key, _) in obj {
        if key.starts_with('_') { continue; }  // skip _comment fields
    }

    for (app, intents_val) in obj {
        if app.starts_with('_') { continue; }
        let intents = intents_val.as_object().expect("intents must be object");
        for (intent, patterns_val) in intents {
            let patterns = patterns_val.as_array().expect("patterns must be array");
            for p in patterns {
                let arr = p.as_array().expect("each pattern is [string, weight]");
                let pattern = arr[0].as_str().expect("pattern string");
                let weight = arr[1].as_f64().expect("weight float") as f32;
                store.add(app, intent, pattern, weight);
            }
        }
    }
    store
}

fn load_tests(path: &str) -> Vec<TestCase> {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Cannot read tests file: {}", path));
    let v: Value = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("Bad tests JSON: {}", e));

    // Filter out _comment entries (objects with only _comment keys)
    v.as_array().expect("tests JSON must be array")
        .iter()
        .filter_map(|item| {
            let obj = item.as_object()?;
            if obj.keys().all(|k| k.starts_with('_')) { return None; }
            serde_json::from_value(item.clone()).ok()
        })
        .collect()
}

// ─── Evaluation ───────────────────────────────────────────────────────────────

const SCORE_THRESHOLD: f32 = 0.8;

fn evaluate_one(store: &CjkStore, tc: &TestCase) -> EvalResult {
    let scored = store.score_query(&tc.query);
    let top = scored.first();

    let all_firing: Vec<String> = scored.iter()
        .filter(|s| s.score >= SCORE_THRESHOLD)
        .map(|s| format!("{}.{} ({:.2})", s.app_id, s.intent_id, s.score))
        .collect();

    if tc.category == "negative" {
        let silent = top.map(|s| s.score < SCORE_THRESHOLD).unwrap_or(true);
        return EvalResult {
            app_correct: silent,
            intent_correct: silent,
            correctly_silent: silent,
            top_app: top.map(|s| s.app_id.clone()).unwrap_or_default(),
            top_intent: top.map(|s| s.intent_id.clone()).unwrap_or_default(),
            top_score: top.map(|s| s.score).unwrap_or(0.0),
            best_match: top.map(|s| s.best_match.clone()).unwrap_or_default(),
            all_firing,
        };
    }

    let top_app = store.detect_app(&tc.query).into_iter().next();
    let top_app_name = top_app.as_ref().map(|(a, _)| a.clone()).unwrap_or_default();
    let top_app_score = top_app.as_ref().map(|(_, s)| *s).unwrap_or(0.0);

    let top_intent_name = top.map(|s| s.intent_id.clone()).unwrap_or_default();
    let top_score = top.map(|s| s.score).unwrap_or(0.0);
    let best_match = top.map(|s| s.best_match.clone()).unwrap_or_default();

    let app_correct  = top_app_score >= SCORE_THRESHOLD && top_app_name == tc.app;
    let intent_correct = top_score >= SCORE_THRESHOLD && top_intent_name == tc.intent;

    EvalResult {
        app_correct,
        intent_correct,
        correctly_silent: false,
        top_app: top_app_name,
        top_intent: top_intent_name,
        top_score,
        best_match,
        all_firing,
    }
}

fn run_eval(store: &CjkStore, cases: &[TestCase], label: &str) -> Vec<EvalResult> {
    println!("\n{}", "=".repeat(70));
    println!("  {}", label);
    println!("{}", "=".repeat(70));

    let mut results = Vec::new();
    let mut cat_stats: HashMap<&str, (usize, usize, usize)> = HashMap::new();

    for tc in cases {
        let r = evaluate_one(store, tc);
        let (total, app_ok, intent_ok) = cat_stats
            .entry(tc.category.as_str())
            .or_insert((0, 0, 0));
        *total += 1;
        if r.app_correct    { *app_ok += 1; }
        if r.intent_correct { *intent_ok += 1; }

        if tc.category == "negative" {
            if r.correctly_silent {
                println!("  ✓ SILENT  {}", truncate(&tc.query, 55));
            } else {
                println!("  ✗ FALSE+  {} | fired: {}.{} score={:.2} via '{}'",
                    truncate(&tc.query, 35),
                    r.top_app, r.top_intent, r.top_score, r.best_match);
            }
        } else if tc.category == "cross_app" {
            // Cross-app: show everything that fired
            let status = if r.intent_correct { "✓" } else { "~" };
            println!("  {} [cross] {}", status, truncate(&tc.query, 50));
            println!("    firing: {}", if r.all_firing.is_empty() { "nothing".to_string() }
                                      else { r.all_firing.join(", ") });
            println!("    expected primary: {}.{}", tc.app, tc.intent);
        } else {
            let ok = match (r.app_correct, r.intent_correct) {
                (true, true)   => "PASS",
                (true, false)  => "PART",
                (false, true)  => "PART",
                (false, false) => "FAIL",
            };
            let a = if r.app_correct    { "✓" } else { "✗" };
            let i = if r.intent_correct { "✓" } else { "✗" };
            println!("  [{}] app{} intent{} {}", ok, a, i, truncate(&tc.query, 50));
            if !r.intent_correct {
                println!("    expected: {}.{}", tc.app, tc.intent);
                println!("    got:      {}.{} score={:.2} via '{}'",
                    r.top_app, r.top_intent, r.top_score, r.best_match);
                if !tc.note.is_empty() {
                    println!("    note:     {}", tc.note);
                }
            }
        }

        results.push(r);
    }

    // Summary
    println!("\n  ── {} ──", label);
    let categories = ["situation", "cross_app", "mixed_script", "negative"];
    let mut total_app = 0;
    let mut total_intent = 0;
    let mut total_n = 0;

    for cat in &categories {
        if let Some((n, app_ok, intent_ok)) = cat_stats.get(cat) {
            if *n == 0 { continue; }
            total_n      += n;
            total_app    += app_ok;
            total_intent += intent_ok;
            println!("  {:12} {:2} cases | app {:2}/{} ({:3}%) | intent {:2}/{} ({:3}%)",
                cat, n,
                app_ok, n, 100 * app_ok / n,
                intent_ok, n, 100 * intent_ok / n);
        }
    }
    if total_n > 0 {
        println!("  {:12} {:2} cases | app {:2}/{} ({:3}%) | intent {:2}/{} ({:3}%)",
            "TOTAL", total_n,
            total_app, total_n, 100 * total_app / total_n,
            total_intent, total_n, 100 * total_intent / total_n);
    }

    results
}

// ─── Learning pass ────────────────────────────────────────────────────────────

fn learn_from_failures(store: &mut CjkStore, cases: &[TestCase], results: &[EvalResult]) {
    println!("\n{}", "=".repeat(70));
    println!("  LEARNING — extracting char n-grams from failed queries");
    println!("{}", "=".repeat(70));

    let mut total_added = 0;
    for (tc, r) in cases.iter().zip(results.iter()) {
        if tc.category == "negative" || tc.category == "cross_app" { continue; }
        if !r.intent_correct {
            let before = store.patterns
                .get(&(tc.app.clone(), tc.intent.clone()))
                .map(|v| v.len()).unwrap_or(0);
            let added = store.learn(&tc.query, &tc.app, &tc.intent);
            let after = store.patterns
                .get(&(tc.app.clone(), tc.intent.clone()))
                .map(|v| v.len()).unwrap_or(0);
            if after > before {
                println!("  + {}.{}: +{} patterns from '{}'",
                    tc.app, tc.intent, added, truncate(&tc.query, 45));
                total_added += added;
            }
        }
    }
    println!("  Total new patterns: {}  |  Store size: {}", total_added, store.pattern_count());
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn truncate(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        s.to_string()
    } else {
        format!("{}…", chars[..max_chars].iter().collect::<String>())
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_dir = args.iter()
        .position(|a| a == "--data")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("data/situation/cjk");

    let seeds_path = format!("{}/seeds.json", data_dir);
    let tests_path = format!("{}/tests.json", data_dir);

    println!("CJK Situation→Action Inference — Option B Experiment");
    println!("Data dir: {}  |  Threshold: {:.2}", data_dir, SCORE_THRESHOLD);
    println!("Matching: direct substring search on raw text (no tokenization)");

    let mut store = load_seeds(&seeds_path);
    println!("\nSeed patterns loaded: {}", store.pattern_count());

    let cases = load_tests(&tests_path);
    let sit  = cases.iter().filter(|t| t.category == "situation").count();
    let mix  = cases.iter().filter(|t| t.category == "mixed_script").count();
    let ca   = cases.iter().filter(|t| t.category == "cross_app").count();
    let neg  = cases.iter().filter(|t| t.category == "negative").count();
    println!("Test cases: {} total  ({} situation, {} mixed_script, {} cross_app, {} negative)",
        cases.len(), sit, mix, ca, neg);

    // Round 1 — seeds only
    let r1 = run_eval(&store, &cases, "ROUND 1 — Seed patterns only (baseline)");

    // Learn from failures
    learn_from_failures(&mut store, &cases, &r1);

    // Round 2 — after learning
    let r2 = run_eval(&store, &cases, "ROUND 2 — After one learning pass");

    // Improvement summary
    println!("\n{}", "=".repeat(70));
    println!("  IMPROVEMENT");
    println!("{}", "=".repeat(70));

    let n = cases.len();
    let app_r1    = r1.iter().filter(|r| r.app_correct).count();
    let intent_r1 = r1.iter().filter(|r| r.intent_correct).count();
    let app_r2    = r2.iter().filter(|r| r.app_correct).count();
    let intent_r2 = r2.iter().filter(|r| r.intent_correct).count();
    println!("  App detection:    {}/{} → {}/{} ({:+})",
        app_r1, n, app_r2, n, app_r2 as i32 - app_r1 as i32);
    println!("  Intent detection: {}/{} → {}/{} ({:+})",
        intent_r1, n, intent_r2, n, intent_r2 as i32 - intent_r1 as i32);

    let mut improved: Vec<String> = Vec::new();
    let mut still_failing: Vec<String> = Vec::new();
    for ((tc, a), b) in cases.iter().zip(r1.iter()).zip(r2.iter()) {
        if tc.category == "negative" || tc.category == "cross_app" { continue; }
        if !a.intent_correct && b.intent_correct {
            improved.push(format!("{}.{}", tc.app, tc.intent));
        }
        if !b.intent_correct {
            still_failing.push(format!(
                "{}.{} (got: {}.{} score={:.2}  query: '{}')",
                tc.app, tc.intent, b.top_app, b.top_intent, b.top_score,
                truncate(&tc.query, 30)
            ));
        }
    }

    if !improved.is_empty() {
        println!("\n  Newly passing ({}):", improved.len());
        for s in &improved { println!("    + {}", s); }
    }
    if !still_failing.is_empty() {
        println!("\n  Still failing after learning ({}):", still_failing.len());
        for s in &still_failing { println!("    - {}", s); }
        println!("\n  These require more seed variants or a different vocabulary approach.");
    }
}
