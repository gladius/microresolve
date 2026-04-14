/// Experiment: Coordinator-Aware Query Segmentation.
///
/// Hypothesis: splitting a query on explicit coordinators ("and", "also", "I also need",
/// etc.) and routing each fragment independently yields better multi-intent recall
/// without false positives, because each fragment is a clean single-intent query.
///
/// Run: cargo run --bin experiment_coord_split --features server
use asv_router::{
    Router,
    hebbian::IntentGraph,
    tokenizer::tokenize,
};
use std::collections::HashMap;

// ── Corpus (same as OMP experiment) ──────────────────────────────────────────

fn build_corpus() -> (Router, IntentGraph) {
    let mut router = Router::new();
    let mut ig = IntentGraph::new();

    let intents: &[(&str, &[&str])] = &[
        ("network:vpn", &[
            "can't connect to VPN",
            "VPN is not working",
            "VPN keeps disconnecting",
            "remote VPN connection failed",
        ]),
        ("network:wifi", &[
            "wifi is not connecting",
            "no internet connection",
            "wifi signal is very weak",
            "internet keeps dropping",
        ]),
        ("account:reset_password", &[
            "reset my password",
            "forgot my password",
            "password expired",
            "locked out because of wrong password",
        ]),
        ("account:setup_mfa", &[
            "set up two-factor authentication",
            "configure my authenticator app",
            "enable MFA on my account",
        ]),
        ("hardware:request_loaner", &[
            "need a loaner laptop",
            "borrow a temporary device",
            "replacement laptop while mine is repaired",
        ]),
        ("hardware:setup_device", &[
            "set up my new laptop",
            "configure my new workstation",
            "initial setup for new computer",
        ]),
        ("tickets:escalate_ticket", &[
            "please escalate my ticket",
            "need a faster response on my case",
            "mark my ticket as high priority",
        ]),
        ("tickets:create_ticket", &[
            "open a new support ticket",
            "submit a help desk request",
            "log this issue with IT",
        ]),
    ];

    for (intent_id, phrases) in intents {
        let mut by_lang: HashMap<String, Vec<String>> = Default::default();
        by_lang.insert("en".to_string(), phrases.iter().map(|s| s.to_string()).collect());
        router.add_intent_multilingual(intent_id, by_lang);

        for phrase in *phrases {
            let tokens: Vec<String> = tokenize(phrase).into_iter()
                .map(|t| t.to_string())
                .collect();
            let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
            ig.learn_phrase(&refs, intent_id);
        }
    }

    (router, ig)
}

// ── Coordinator segmenter ─────────────────────────────────────────────────────

/// Returns split points (byte positions) in the query where a new intent segment begins.
/// A split point is AFTER a coordinator phrase and its surrounding whitespace.
fn find_coordinator_splits(query: &str) -> Vec<usize> {
    let lower = query.to_lowercase();

    // Ordered from longest to shortest to avoid partial matches
    let coordinators = [
        "and also i need to",
        "and also i want to",
        "and also i need",
        "and also i want",
        "i also need to",
        "i also want to",
        "i also need",
        "i also want",
        "one more thing",
        "another thing",
        "in addition to",
        "in addition",
        "additionally",
        "furthermore",
        "as well as",
        "on top of that",
        "also please",
        "and also",
        "but also",
        "plus also",
        "plus i",
        "oh and",
        ", and ",
        " and ",
        " also ",
        " plus ",
    ];

    let mut splits: Vec<usize> = Vec::new();

    for coord in &coordinators {
        let mut search_from = 0;
        while let Some(pos) = lower[search_from..].find(coord) {
            let abs_pos = search_from + pos;
            let split_after = abs_pos + coord.len();

            // Avoid splits that are already covered by a longer coordinator
            // (i.e., if this position is already a split point, skip)
            let already_covered = splits.iter().any(|&s| {
                let min = if s < split_after { s } else { split_after };
                let max = if s > split_after { s } else { split_after };
                max - min < 20
            });

            if !already_covered && split_after < query.len() {
                splits.push(split_after);
            }

            search_from = abs_pos + 1;
        }
    }

    splits.sort_unstable();
    splits.dedup();
    splits
}

/// Split a query into segments at coordinator positions.
/// Returns non-empty trimmed segments.
fn split_on_coordinators(query: &str) -> Vec<String> {
    let splits = find_coordinator_splits(query);

    if splits.is_empty() {
        return vec![query.trim().to_string()];
    }

    let mut segments = Vec::new();
    let mut prev = 0usize;

    for &split in &splits {
        let seg = query[prev..split].trim().trim_end_matches(',').trim().to_string();
        if !seg.is_empty() && tokenize(&seg).len() >= 2 {
            segments.push(seg);
        }
        prev = split;
    }

    // Last segment
    let last = query[prev..].trim().to_string();
    if !last.is_empty() && tokenize(&last).len() >= 2 {
        segments.push(last);
    }

    if segments.is_empty() {
        vec![query.trim().to_string()]
    } else {
        segments
    }
}

// ── Routing helpers ───────────────────────────────────────────────────────────

fn route_segment(ig: &IntentGraph, text: &str, threshold: f32, gap: f32) -> Vec<(String, f32)> {
    let (results, _neg) = ig.score_multi_normalized(text, threshold, gap);
    results
}

fn route_split(ig: &IntentGraph, query: &str, threshold: f32, gap_per_segment: f32) -> Vec<(String, f32)> {
    let segments = split_on_coordinators(query);
    let mut seen: HashMap<String, f32> = HashMap::new();

    for seg in &segments {
        let results = route_segment(ig, seg, threshold, gap_per_segment);
        for (id, score) in results {
            // Keep highest score per intent across segments
            let entry = seen.entry(id).or_insert(0.0);
            if score > *entry { *entry = score; }
        }
    }

    let mut merged: Vec<(String, f32)> = seen.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    merged
}

// ── Test cases ────────────────────────────────────────────────────────────────

struct TestCase {
    query:       &'static str,
    expected:    Vec<&'static str>,
    label:       &'static str,
    expect_segs: usize,  // expected number of segments
}

fn test_cases() -> Vec<TestCase> {
    vec![
        // --- Should NOT split (single intent) ---
        TestCase {
            query:       "my VPN keeps disconnecting",
            expected:    vec!["network:vpn"],
            label:       "single: vpn (no split expected)",
            expect_segs: 1,
        },
        TestCase {
            query:       "login and password are not working",
            expected:    vec!["account:reset_password"],
            label:       "edge: 'and' inside phrase, must NOT split",
            expect_segs: 1,
        },
        // --- Should split on explicit coordinators ---
        TestCase {
            query:       "reset my password and also set up my MFA",
            expected:    vec!["account:reset_password", "account:setup_mfa"],
            label:       "two: password + mfa (and also)",
            expect_segs: 2,
        },
        TestCase {
            query:       "I need a loaner laptop and I also need to open a support ticket",
            expected:    vec!["hardware:request_loaner", "tickets:create_ticket"],
            label:       "two: loaner + ticket (I also need)",
            expect_segs: 2,
        },
        TestCase {
            query:       "my password expired, and also I need a loaner, plus please escalate my case",
            expected:    vec!["account:reset_password", "hardware:request_loaner", "tickets:escalate_ticket"],
            label:       "three: password + loaner + escalate",
            expect_segs: 3,
        },
        TestCase {
            query:       "configure my authenticator app and also help me set up my new laptop",
            expected:    vec!["account:setup_mfa", "hardware:setup_device"],
            label:       "two: mfa + setup_device",
            expect_segs: 2,
        },
        TestCase {
            query:       "VPN not working. Also my wifi signal is weak",
            expected:    vec!["network:vpn", "network:wifi"],
            label:       "two: vpn + wifi (Also at start of sentence)",
            expect_segs: 2,
        },
        // --- Harder: implicit coordinator ---
        TestCase {
            query:       "I need to reset my password plus open a new ticket",
            expected:    vec!["account:reset_password", "tickets:create_ticket"],
            label:       "two: password + ticket (plus)",
            expect_segs: 2,
        },
        // --- Must not over-detect on split ---
        TestCase {
            query:       "set up my new laptop additionally configure the workstation",
            expected:    vec!["hardware:setup_device"],
            label:       "edge: same domain split, should merge to one intent",
            expect_segs: 2, // splits are fine, result should deduplicate
        },
    ]
}

// ── Scoring ───────────────────────────────────────────────────────────────────

fn score_result(got: &[(String, f32)], expected: &[&str]) -> (&'static str, String) {
    let got_ids: std::collections::HashSet<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
    let exp_ids: std::collections::HashSet<&str> = expected.iter().copied().collect();

    if got_ids == exp_ids { return ("PASS", String::new()); }
    let missed: Vec<&str> = exp_ids.difference(&got_ids).copied().collect();
    let extra:  Vec<&str> = got_ids.difference(&exp_ids).copied().collect();
    let mut detail = String::new();
    if !missed.is_empty() { detail.push_str(&format!("missed={:?} ", missed)); }
    if !extra.is_empty()  { detail.push_str(&format!("extra={:?} ", extra)); }
    if got_ids.intersection(&exp_ids).count() > 0 { ("PARTIAL", detail) }
    else { ("FAIL", detail) }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let (_router, ig) = build_corpus();

    let THRESHOLD: f32     = 0.3;
    let GAP_GLOBAL: f32    = 1.5;  // current gap for holistic routing
    let GAP_SEGMENT: f32   = 0.8;  // tighter gap per fragment (single-intent expected)

    let cases = test_cases();

    println!("\n{:=<70}", "");
    println!("  Coordinator Segmentation Experiment");
    println!("  Corpus: 8 intents, {} test cases", cases.len());
    println!("{:=<70}\n", "");

    let mut baseline_pass = 0;
    let mut split_pass = 0;
    let total = cases.len();

    for case in &cases {
        // Baseline: holistic routing (no split)
        let (baseline, _) = ig.score_multi_normalized(case.query, THRESHOLD, GAP_GLOBAL);

        // Split routing
        let split_result = route_split(&ig, case.query, THRESHOLD, GAP_SEGMENT);

        // Show segments
        let segments = split_on_coordinators(case.query);

        let (base_status, base_detail) = score_result(&baseline, &case.expected);
        let (split_status, split_detail) = score_result(&split_result, &case.expected);

        if base_status == "PASS"  { baseline_pass += 1; }
        if split_status == "PASS" { split_pass += 1; }

        let improved  = split_status == "PASS" && base_status != "PASS";
        let regressed = base_status == "PASS" && split_status != "PASS";
        let marker = if improved { "⬆ " } else if regressed { "⬇ " } else { "  " };

        let seg_ok = if segments.len() == case.expect_segs { "✓" } else { "✗" };

        println!("{}[{}]  {}", marker, case.label, case.query);
        println!("  Segments [{}]: {:?} (expected {})", seg_ok, segments, case.expect_segs);
        println!("  Baseline  [{:<7}] {:?}  {}",
            base_status,
            baseline.iter().map(|(id,_)| id.split(':').last().unwrap_or(id)).collect::<Vec<_>>(),
            base_detail);
        println!("  Split     [{:<7}] {:?}  {}",
            split_status,
            split_result.iter().map(|(id,_)| id.split(':').last().unwrap_or(id)).collect::<Vec<_>>(),
            split_detail);
        println!("  Expected:          {:?}", case.expected.iter().map(|id| id.split(':').last().unwrap_or(id)).collect::<Vec<_>>());
        println!();
    }

    println!("{:=<70}", "");
    println!("  RESULTS");
    println!("{}", "=".repeat(70));
    println!("  Baseline:  {}/{} ({:.1}%)", baseline_pass, total,
        100.0 * baseline_pass as f32 / total as f32);
    println!("  Split:     {}/{} ({:.1}%)", split_pass, total,
        100.0 * split_pass as f32 / total as f32);
    let delta = split_pass as i32 - baseline_pass as i32;
    println!("  Delta:     {:+} cases", delta);
    println!();
    if delta > 0 {
        println!("  ✓ COORDINATOR SPLIT IMPROVES over baseline — worth integrating");
    } else if delta < 0 {
        println!("  ✗ SPLIT REGRESSES — check coordinator patterns");
    } else {
        println!("  ~ No change — review segment quality");
    }
}
