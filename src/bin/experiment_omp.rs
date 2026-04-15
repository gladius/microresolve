/// Experiment: Orthogonal Matching Pursuit (OMP) for multi-intent detection.
///
/// Hypothesis: after detecting the top intent, soft-subtracting its token weights
/// from the query allows secondary intents to rise to detectability — without
/// changing the gap parameter.
///
/// Mathematical basis: OMP is a provably correct greedy sparse decomposition
/// algorithm. Here: query = signal, intent vectors = basis, we seek the minimal
/// set of intents whose combined score explains the query.
///
/// Run: cargo run --bin experiment_omp
use asv_router::{
    Router,
    scoring::IntentGraph,
    tokenizer::tokenize,
};
use std::collections::{HashMap, HashSet};

// ── Corpus ────────────────────────────────────────────────────────────────────

fn build_corpus() -> (Router, IntentGraph) {
    let mut router = Router::new();
    let mut ig = IntentGraph::new();

    let intents: &[(&str, &[&str])] = &[
        ("network:vpn", &[
            "can't connect to VPN",
            "VPN is not working",
            "VPN keeps disconnecting",
            "remote VPN connection failed",
            "need VPN access to work from home",
        ]),
        ("network:wifi", &[
            "wifi is not connecting",
            "no internet connection at home",
            "wifi signal is very weak",
            "dropped from the wireless network",
            "internet keeps dropping",
        ]),
        ("account:reset_password", &[
            "reset my password",
            "forgot my password",
            "password expired and I can't log in",
            "need to change my password",
            "locked out because of wrong password",
        ]),
        ("account:setup_mfa", &[
            "set up two-factor authentication",
            "configure my authenticator app",
            "enable MFA on my account",
            "two-step verification is not working",
        ]),
        ("hardware:request_loaner", &[
            "need a loaner laptop",
            "borrow a temporary device",
            "replacement laptop while mine is being repaired",
            "need a spare computer for now",
        ]),
        ("tickets:escalate_ticket", &[
            "this is urgent please escalate my ticket",
            "need a faster response on my case",
            "escalate this to a senior technician",
            "mark my ticket as high priority",
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

// ── IDF helper ────────────────────────────────────────────────────────────────

/// Compute IDF for a token from the intent graph.
/// IDF = ln(N / df) where N = total distinct intents, df = intents containing token.
fn compute_idf(ig: &IntentGraph, token: &str) -> f32 {
    // Count total distinct intents in the graph
    let total_intents: usize = {
        let mut all: HashSet<&str> = HashSet::new();
        for entries in ig.word_intent.values() {
            for (id, _) in entries { all.insert(id.as_str()); }
        }
        all.len().max(1)
    };

    match ig.word_intent.get(token) {
        None => 0.0,
        Some(entries) => {
            let df = entries.len();
            (total_intents as f32 / df as f32).ln().max(0.0)
        }
    }
}

// ── Raw scoring (without gap filter) ─────────────────────────────────────────

/// Score all intents given token weights. Returns sorted (intent, score) pairs.
/// token_weights: token → multiplier (starts at 1.0, reduced by OMP subtraction)
fn score_with_weights(ig: &IntentGraph, token_weights: &HashMap<String, f32>) -> Vec<(String, f32)> {
    // Pre-compute N once
    let total_intents: usize = {
        let mut all: HashSet<&str> = HashSet::new();
        for entries in ig.word_intent.values() {
            for (id, _) in entries { all.insert(id.as_str()); }
        }
        all.len().max(1)
    };

    let mut scores: HashMap<String, f32> = HashMap::new();

    for (token, &tw) in token_weights {
        if tw <= 0.0 { continue; }
        if let Some(entries) = ig.word_intent.get(token.as_str()) {
            let idf = (total_intents as f32 / entries.len() as f32).ln().max(0.0);
            for (intent_id, intent_weight) in entries {
                let intent_weight = *intent_weight;
                *scores.entry(intent_id.clone()).or_default() += intent_weight * idf * tw;
            }
        }
    }

    let mut sorted: Vec<(String, f32)> = scores.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

// ── OMP multi-intent detection ────────────────────────────────────────────────

/// OMP-style greedy multi-intent detection.
///
/// - alpha: soft-subtraction strength [0,1]. 1.0 = fully remove detected tokens.
/// - threshold: minimum score for any round to confirm an intent.
/// - max_rounds: maximum intents to detect (prevents runaway).
fn omp_route(
    ig: &IntentGraph,
    query: &str,
    threshold: f32,
    alpha: f32,
    max_rounds: usize,
) -> Vec<(String, f32)> {
    // Build initial per-token weight vector (starts at 1.0 for each token)
    let tokens = tokenize(query);
    let mut token_weights: HashMap<String, f32> = HashMap::new();
    for t in &tokens {
        *token_weights.entry(t.to_string()).or_default() += 1.0;
    }

    let mut confirmed: Vec<(String, f32)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for _round in 0..max_rounds {
        let scores = score_with_weights(ig, &token_weights);

        let (top_intent, top_score) = match scores.first() {
            Some(s) => (s.0.clone(), s.1),
            None => break,
        };

        if top_score < threshold { break; }
        if seen.contains(&top_intent) { break; }

        confirmed.push((top_intent.clone(), top_score));
        seen.insert(top_intent.clone());

        // Soft-subtract: for each token, reduce its weight proportional to
        // how strongly it's associated with the just-detected intent.
        if let Some(entries) = ig.word_intent.get(&top_intent) {
            // Build a lookup: token → weight for this intent
            // (word_intent maps token→[(intent,weight)], we need the reverse)
            let _ = entries; // unused
        }

        // Walk through all tokens in query and reduce weight based on their
        // association with the detected intent.
        for (token, weight) in &mut token_weights {
            if let Some(entries) = ig.word_intent.get(token.as_str()) {
                for (intent_id, intent_weight) in entries {
                let intent_weight = *intent_weight;
                    if intent_id == &top_intent {
                        // Soft subtraction: scale down this token's contribution
                        *weight *= 1.0 - alpha * intent_weight;
                        break;
                    }
                }
            }
        }
    }

    confirmed
}

// ── Baseline: current gap filter ──────────────────────────────────────────────

fn baseline_route(ig: &IntentGraph, query: &str, threshold: f32, gap: f32) -> Vec<(String, f32)> {
    let (results, _neg) = ig.score_multi_normalized(query, threshold, gap);
    results
}

// ── Test cases ────────────────────────────────────────────────────────────────

struct TestCase {
    query:    &'static str,
    expected: &'static [&'static str],
    label:    &'static str,
}

const CASES: &[TestCase] = &[
    // Single intent (must still work, must not over-detect)
    TestCase {
        query:    "my VPN keeps disconnecting",
        expected: &["network:vpn"],
        label:    "single: vpn only",
    },
    TestCase {
        query:    "I forgot my password",
        expected: &["account:reset_password"],
        label:    "single: password only",
    },
    TestCase {
        query:    "the VPN is slow today",
        expected: &["network:vpn"],
        label:    "edge: vpn only — 'slow' must not trigger wifi",
    },
    // Two intents — the core problem OMP should solve
    TestCase {
        query:    "VPN is not working and my wifi keeps dropping",
        expected: &["network:vpn", "network:wifi"],
        label:    "two: vpn + wifi (explicit and)",
    },
    TestCase {
        query:    "reset my password and configure my authenticator app",
        expected: &["account:reset_password", "account:setup_mfa"],
        label:    "two: password + mfa (explicit and)",
    },
    TestCase {
        query:    "internet is down and VPN won't connect either",
        expected: &["network:vpn", "network:wifi"],
        label:    "two: vpn + wifi (implicit, harder — wifi via 'internet')",
    },
    TestCase {
        query:    "I need a loaner laptop urgently this is critical please escalate",
        expected: &["hardware:request_loaner", "tickets:escalate_ticket"],
        label:    "two: loaner + escalate (urgency implicit)",
    },
    // Three intents
    TestCase {
        query:    "my password expired wifi is not working and I need a loaner laptop",
        expected: &["account:reset_password", "network:wifi", "hardware:request_loaner"],
        label:    "three: password + wifi + loaner",
    },
    // Must NOT over-detect
    TestCase {
        query:    "I cannot log into my account",
        expected: &["account:reset_password"],
        label:    "edge: single — must not over-detect",
    },
];

// ── Scoring ───────────────────────────────────────────────────────────────────

fn score_result(got: &[(String, f32)], expected: &[&str]) -> (&'static str, String) {
    let got_ids: HashSet<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
    let exp_ids: HashSet<&str> = expected.iter().copied().collect();
    if got_ids == exp_ids { return ("PASS", String::new()); }
    let missed: Vec<&str> = exp_ids.difference(&got_ids).copied().collect();
    let extra:  Vec<&str> = got_ids.difference(&exp_ids).copied().collect();
    let mut d = String::new();
    if !missed.is_empty() { d.push_str(&format!("missed={:?} ", missed)); }
    if !extra.is_empty()  { d.push_str(&format!("extra={:?} ", extra)); }
    if got_ids.intersection(&exp_ids).count() > 0 { ("PARTIAL", d) } else { ("FAIL", d) }
}

fn short(id: &str) -> &str { id.split(':').last().unwrap_or(id) }

fn main() {
    let (_router, ig) = build_corpus();

    let THRESHOLD:  f32 = 0.3;
    let GAP:        f32 = 1.5;
    let OMP_ALPHA:  f32 = 0.8;
    let OMP_ROUNDS: usize = 4;

    println!("\n{:=<70}", "");
    println!("  OMP Residual Detection Experiment");
    println!("  Corpus: 6 intents | {} cases | alpha={} rounds={}", CASES.len(), OMP_ALPHA, OMP_ROUNDS);
    println!("{:=<70}\n", "");

    let mut base_pass = 0;
    let mut omp_pass  = 0;

    for case in CASES {
        let baseline = baseline_route(&ig, case.query, THRESHOLD, GAP);
        let omp      = omp_route(&ig, case.query, THRESHOLD, OMP_ALPHA, OMP_ROUNDS);

        let (bs, bd) = score_result(&baseline, case.expected);
        let (os, od) = score_result(&omp,      case.expected);

        if bs == "PASS" { base_pass += 1; }
        if os == "PASS" { omp_pass  += 1; }

        let marker = match (bs, os) {
            ("PASS", "PASS") | ("PARTIAL"|"FAIL", "PARTIAL"|"FAIL") => "  ",
            (_,      "PASS") => "⬆ ",
            ("PASS", _     ) => "⬇ ",
            _                => "  ",
        };

        println!("{}[{}]", marker, case.label);
        println!("  Query:    \"{}\"", case.query);
        println!("  Baseline [{:<7}] {:?}  {}",
            bs, baseline.iter().map(|(id,_)| short(id)).collect::<Vec<_>>(), bd);
        println!("  OMP      [{:<7}] {:?}  {}",
            os, omp.iter().map(|(id,_)| short(id)).collect::<Vec<_>>(), od);
        println!("  Expected:         {:?}", case.expected.iter().map(|id| short(id)).collect::<Vec<_>>());
        println!();
    }

    let total = CASES.len();
    println!("{:=<70}", "");
    println!("  Baseline: {}/{} ({:.0}%)", base_pass, total, 100.0 * base_pass as f32 / total as f32);
    println!("  OMP:      {}/{} ({:.0}%)", omp_pass,  total, 100.0 * omp_pass  as f32 / total as f32);
    let delta = omp_pass as i32 - base_pass as i32;
    println!("  Delta:    {:+} cases", delta);
    if delta > 0      { println!("\n  ✓ OMP IMPROVES — worth integrating into scoring.rs"); }
    else if delta < 0 { println!("\n  ✗ OMP REGRESSES — revisit alpha ({}) or threshold ({})", OMP_ALPHA, THRESHOLD); }
    else              { println!("\n  ~ No change — try alpha=0.9 or max_rounds=5"); }
}
