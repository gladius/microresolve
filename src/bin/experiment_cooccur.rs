/// Experiment: Intent Co-occurrence PMI Boosting.
///
/// Hypothesis: if we track which intents co-occur in real conversations, we can
/// boost the score of intents that frequently appear alongside already-confirmed
/// intents. This helps weak-but-real secondary intents cross the threshold.
///
/// PMI(I1, I2) = log[ P(I1 ∩ I2) / (P(I1) × P(I2)) ]
/// Positive PMI → these intents tend to co-occur.
///
/// Run: cargo run --bin experiment_cooccur --features server
use asv_router::{
    Router,
    hebbian::IntentGraph,
    tokenizer::tokenize,
};
use std::collections::HashMap;

// ── PMI table ─────────────────────────────────────────────────────────────────

struct CooccurTable {
    /// How many times (I1, I2) co-occurred (unordered pair, stored both ways)
    pair_counts: HashMap<(String, String), u32>,
    /// How many times each intent appeared in any routing
    marginal:    HashMap<String, u32>,
    /// Total number of routing sessions observed
    total_sessions: u32,
}

impl CooccurTable {
    fn new() -> Self {
        Self {
            pair_counts:    HashMap::new(),
            marginal:       HashMap::new(),
            total_sessions: 0,
        }
    }

    /// Record a routing outcome (list of intents confirmed in one query).
    fn record(&mut self, intents: &[&str]) {
        self.total_sessions += 1;
        for &i in intents {
            *self.marginal.entry(i.to_string()).or_default() += 1;
        }
        // Record all unordered pairs
        for (a, rest) in intents.iter().enumerate() {
            for b in &intents[a + 1..] {
                let (k1, k2) = if rest <= b { (rest, b) } else { (b, rest) };
                *self.pair_counts
                    .entry((k1.to_string(), k2.to_string()))
                    .or_default() += 1;
            }
        }
    }

    /// Compute PMI(I1, I2). Returns 0.0 if not enough data.
    fn pmi(&self, i1: &str, i2: &str) -> f32 {
        let n = self.total_sessions as f32;
        if n < 5.0 { return 0.0; }

        let count_i1 = *self.marginal.get(i1).unwrap_or(&0) as f32;
        let count_i2 = *self.marginal.get(i2).unwrap_or(&0) as f32;

        if count_i1 == 0.0 || count_i2 == 0.0 { return 0.0; }

        let (k1, k2) = if i1 <= i2 { (i1, i2) } else { (i2, i1) };
        let count_pair = *self.pair_counts
            .get(&(k1.to_string(), k2.to_string()))
            .unwrap_or(&0) as f32;

        if count_pair == 0.0 { return -1.0; }

        // PMI = log( P(I1,I2) / (P(I1) * P(I2)) )
        //     = log( count_pair/n / (count_i1/n * count_i2/n) )
        //     = log( count_pair * n / (count_i1 * count_i2) )
        ((count_pair * n) / (count_i1 * count_i2)).ln()
    }

    /// Boost candidate scores using PMI from already-confirmed intents.
    /// Returns new scores with adjustments applied.
    fn boost(
        &self,
        confirmed: &[(String, f32)],
        candidates: &[(String, f32)],
        beta: f32,
        threshold: f32,
    ) -> Vec<(String, f32)> {
        let confirmed_ids: Vec<&str> = confirmed.iter().map(|(id, _)| id.as_str()).collect();
        let mut boosted = candidates.to_vec();

        for (cand_id, cand_score) in &mut boosted {
            let mut boost_total = 0.0f32;
            for &conf_id in &confirmed_ids {
                let p = self.pmi(conf_id, cand_id);
                if p > 0.0 {
                    boost_total += p;
                }
            }
            // Apply boost: raise score by beta * total_positive_PMI
            *cand_score += beta * boost_total;
        }

        // Re-sort and filter by threshold
        boosted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        boosted.retain(|(_, s)| *s >= threshold);
        boosted
    }

    fn print_top_pairs(&self, n: usize) {
        let mut pmis: Vec<((String, String), f32)> = self.pair_counts.keys()
            .map(|(i1, i2)| ((i1.clone(), i2.clone()), self.pmi(i1, i2)))
            .collect();
        pmis.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("  Top {} intent co-occurrence pairs by PMI:", n);
        for ((i1, i2), p) in pmis.iter().take(n) {
            let c = self.pair_counts.get(&(i1.clone(), i2.clone())).unwrap_or(&0);
            println!("    PMI={:+.3}  count={}  {} + {}",
                p, c,
                i1.split(':').last().unwrap_or(i1),
                i2.split(':').last().unwrap_or(i2));
        }
    }
}

// ── Corpus ────────────────────────────────────────────────────────────────────

fn build_corpus() -> (Router, IntentGraph) {
    let mut router = Router::new();
    let mut ig = IntentGraph::new();

    let intents: &[(&str, &[&str])] = &[
        ("hardware:request_loaner", &[
            "need a loaner laptop",
            "borrow a temporary device",
            "replacement laptop while mine is repaired",
        ]),
        ("tickets:escalate_ticket", &[
            "please escalate my ticket",
            "need a faster response",
            "mark my ticket as high priority",
            "this is urgent",
        ]),
        ("account:reset_password", &[
            "reset my password",
            "forgot my password",
            "password expired",
        ]),
        ("account:setup_mfa", &[
            "set up two-factor authentication",
            "configure my authenticator app",
            "enable MFA on my account",
        ]),
        ("hardware:report_broken", &[
            "my laptop is broken",
            "screen is cracked",
            "computer won't turn on",
            "device is completely dead",
        ]),
        ("network:vpn", &[
            "can't connect to VPN",
            "VPN is not working",
            "VPN keeps disconnecting",
        ]),
        ("network:wifi", &[
            "wifi is not connecting",
            "no internet connection",
            "internet keeps dropping",
        ]),
    ];

    for (intent_id, phrases) in intents {
        let mut by_lang: HashMap<String, Vec<String>> = Default::default();
        by_lang.insert("en".to_string(), phrases.iter().map(|s| s.to_string()).collect());
        router.add_intent_multilingual(intent_id, by_lang);

        for phrase in *phrases {
            let tokens: Vec<String> = tokenize(phrase).into_iter().map(|t| t.to_string()).collect();
            let refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
            ig.learn_phrase(&refs, intent_id);
        }
    }

    (router, ig)
}

// ── Simulated history ─────────────────────────────────────────────────────────

/// Simulate a realistic conversation history to build up the PMI table.
/// In production this comes from real routing logs.
fn simulate_history(table: &mut CooccurTable) {
    // Loaner + escalate: people who need loaner are usually frustrated → escalate
    for _ in 0..18 {
        table.record(&["hardware:request_loaner", "tickets:escalate_ticket"]);
    }
    for _ in 0..4 {
        table.record(&["hardware:request_loaner"]);  // loaner alone (rarer)
    }
    for _ in 0..3 {
        table.record(&["tickets:escalate_ticket"]);  // escalate alone

    }

    // Password + MFA: often set up together
    for _ in 0..12 {
        table.record(&["account:reset_password", "account:setup_mfa"]);
    }
    for _ in 0..8 {
        table.record(&["account:reset_password"]);   // password alone
    }
    for _ in 0..3 {
        table.record(&["account:setup_mfa"]);        // mfa alone
    }

    // Broken laptop + loaner: broken device → need temporary replacement
    for _ in 0..14 {
        table.record(&["hardware:report_broken", "hardware:request_loaner"]);
    }
    for _ in 0..5 {
        table.record(&["hardware:report_broken"]);   // broken alone
    }

    // VPN + wifi: both network issues often reported together when working remote
    for _ in 0..10 {
        table.record(&["network:vpn", "network:wifi"]);
    }
    for _ in 0..8 {
        table.record(&["network:vpn"]);              // vpn alone
    }
    for _ in 0..4 {
        table.record(&["network:wifi"]);             // wifi alone
    }

    // Random unrelated singles (noise)
    for _ in 0..5 {
        table.record(&["account:reset_password"]);
    }
    for _ in 0..3 {
        table.record(&["tickets:escalate_ticket"]);
    }
}

// ── Test cases ────────────────────────────────────────────────────────────────

struct TestCase {
    query:           &'static str,
    expected:        Vec<&'static str>,
    label:           &'static str,
    /// The intent we expect to be boosted into confirmation (was below threshold/gap before)
    expected_boost:  Option<&'static str>,
}

fn test_cases() -> Vec<TestCase> {
    vec![
        TestCase {
            query:          "I urgently need a loaner laptop",
            expected:       vec!["hardware:request_loaner", "tickets:escalate_ticket"],
            label:          "loaner → escalate should be boosted by PMI",
            expected_boost: Some("tickets:escalate_ticket"),
        },
        TestCase {
            query:          "my laptop is broken and I need a spare",
            expected:       vec!["hardware:report_broken", "hardware:request_loaner"],
            label:          "broken + loaner (both should appear)",
            expected_boost: None,
        },
        TestCase {
            query:          "I need to reset my login credentials",
            expected:       vec!["account:reset_password", "account:setup_mfa"],
            label:          "password → mfa should be boosted by PMI",
            expected_boost: Some("account:setup_mfa"),
        },
        TestCase {
            query:          "VPN won't connect from home",
            expected:       vec!["network:vpn"],
            label:          "vpn alone (wifi should NOT be boosted too much)",
            expected_boost: None,
        },
        TestCase {
            query:          "my VPN is down and internet is slow",
            expected:       vec!["network:vpn", "network:wifi"],
            label:          "vpn + wifi (PMI boost should confirm wifi)",
            expected_boost: Some("network:wifi"),
        },
    ]
}

// ── Routing helpers ───────────────────────────────────────────────────────────

/// Route with gap filter → returns (confirmed, candidates_below_gap).
fn route_with_candidates(
    ig: &IntentGraph,
    query: &str,
    threshold: f32,
    gap: f32,
) -> (Vec<(String, f32)>, Vec<(String, f32)>) {
    let (all_raw, _neg) = ig.score_multi_normalized(query, threshold, 999.0); // get all above threshold
    let (confirmed, below_gap): (Vec<_>, Vec<_>) = all_raw.into_iter()
        .partition(|(_, s)| {
            // "would this be confirmed by normal gap filter?"
            // We need the top score for this, so compute separately
            true // placeholder, fix below
        });
    let _ = below_gap;

    // Actually: get normal confirmed via proper gap
    let (normal_confirmed, _) = ig.score_multi_normalized(query, threshold, gap);

    // Candidates: above threshold but below gap
    let (all_above_threshold, _) = ig.score_multi_normalized(query, threshold, 999.0);
    let confirmed_ids: std::collections::HashSet<&str> =
        normal_confirmed.iter().map(|(id, _)| id.as_str()).collect();
    let candidates_below: Vec<(String, f32)> = all_above_threshold
        .into_iter()
        .filter(|(id, _)| !confirmed_ids.contains(id.as_str()))
        .collect();

    (normal_confirmed, candidates_below)
}

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

    // Build PMI table from simulated history
    let mut table = CooccurTable::new();
    simulate_history(&mut table);

    let THRESHOLD: f32 = 0.3;
    let GAP: f32       = 1.5;
    let PMI_BETA: f32  = 0.8;  // how strongly to apply PMI boost

    println!("\n{:=<70}", "");
    println!("  Co-occurrence PMI Boosting Experiment");
    println!("  {} total sessions simulated", table.total_sessions);
    println!("{:=<70}\n", "");

    table.print_top_pairs(6);
    println!();

    let cases = test_cases();
    let mut baseline_pass = 0;
    let mut boosted_pass = 0;
    let total = cases.len();

    for case in &cases {
        // Baseline routing
        let (baseline, candidates) = route_with_candidates(&ig, case.query, THRESHOLD, GAP);

        // PMI-boosted routing: take candidates below gap and boost them
        let boosted_candidates = table.boost(&baseline, &candidates, PMI_BETA, THRESHOLD);

        // Merge confirmed + newly-boosted-above-gap
        let top_score = baseline.first().map(|(_, s)| *s).unwrap_or(0.0);
        let newly_confirmed: Vec<(String, f32)> = boosted_candidates
            .into_iter()
            .filter(|(_, s)| top_score - s <= GAP)
            .collect();

        let mut final_result = baseline.clone();
        for item in newly_confirmed {
            if !final_result.iter().any(|(id, _)| id == &item.0) {
                final_result.push(item);
            }
        }
        final_result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let (base_status, base_detail) = score_result(&baseline, &case.expected);
        let (boost_status, boost_detail) = score_result(&final_result, &case.expected);

        if base_status == "PASS"  { baseline_pass += 1; }
        if boost_status == "PASS" { boosted_pass += 1; }

        let improved  = boost_status == "PASS" && base_status != "PASS";
        let regressed = base_status == "PASS" && boost_status != "PASS";
        let marker = if improved { "⬆ " } else if regressed { "⬇ " } else { "  " };

        // Show PMI values for expected boost intent
        let pmi_note = if let Some(boost_intent) = case.expected_boost {
            let confirmed_id = baseline.first().map(|(id, _)| id.as_str()).unwrap_or("");
            format!("  PMI({}, {}) = {:.3}",
                confirmed_id.split(':').last().unwrap_or(confirmed_id),
                boost_intent.split(':').last().unwrap_or(boost_intent),
                table.pmi(confirmed_id, boost_intent))
        } else { String::new() };

        println!("{}[{}]", marker, case.label);
        println!("  Query: \"{}\"", case.query);
        if !pmi_note.is_empty() { println!("{}", pmi_note); }
        println!("  Baseline [{:<7}] {:?}  {}",
            base_status,
            baseline.iter().map(|(id,_)| id.split(':').last().unwrap_or(id)).collect::<Vec<_>>(),
            base_detail);
        println!("  Boosted  [{:<7}] {:?}  {}",
            boost_status,
            final_result.iter().map(|(id,_)| id.split(':').last().unwrap_or(id)).collect::<Vec<_>>(),
            boost_detail);
        println!("  Expected:         {:?}", case.expected.iter().map(|id| id.split(':').last().unwrap_or(id)).collect::<Vec<_>>());
        println!();
    }

    println!("{:=<70}", "");
    println!("  RESULTS");
    println!("{}", "=".repeat(70));
    println!("  Baseline: {}/{} ({:.1}%)", baseline_pass, total,
        100.0 * baseline_pass as f32 / total as f32);
    println!("  PMI:      {}/{} ({:.1}%)", boosted_pass, total,
        100.0 * boosted_pass as f32 / total as f32);
    let delta = boosted_pass as i32 - baseline_pass as i32;
    println!("  Delta:    {:+} cases", delta);
    println!();
    if delta > 0 {
        println!("  ✓ PMI BOOSTING IMPROVES over baseline — worth integrating");
    } else if delta < 0 {
        println!("  ✗ PMI BOOSTING REGRESSES — check beta value or history quality");
    } else {
        println!("  ~ No change — may need more training data or higher beta");
    }
}
