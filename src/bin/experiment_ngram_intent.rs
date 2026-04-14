/// Experiment: Word N-gram Intent Learning
///
/// THE critical question: can learned word n-grams (bigrams, trigrams, 4-grams)
/// capture intent patterns that unigrams miss, AND are they resilient to:
///   - Slight word changes ("waiting" vs "waited")
///   - Extra words inserted ("been really waiting")
///   - Partial phrase matches
///   - Word order changes
///
/// Run: cargo run --bin experiment_ngram_intent
use asv_router::{
    hebbian::IntentGraph,
    tokenizer::tokenize,
    Router,
};
use std::collections::{HashMap, HashSet};

// ── N-gram IntentGraph extension ──────────────────────────────────────────────

struct NgramIntentGraph {
    /// The base IntentGraph (unigram scoring)
    ig: IntentGraph,
    /// n-gram → intent associations. Key = "token1_token2_token3"
    phrase_intent: HashMap<String, Vec<(String, f32)>>,
}

impl NgramIntentGraph {
    fn new(ig: IntentGraph) -> Self {
        Self { ig, phrase_intent: HashMap::new() }
    }

    /// Learn a phrase as n-grams at ALL sub-lengths (2..=max_n).
    /// "been waiting all morning" with max_n=4 learns:
    ///   bigrams:  been_waiting, waiting_all, all_morning
    ///   trigrams: been_waiting_all, waiting_all_morning
    ///   4-grams:  been_waiting_all_morning
    fn learn_phrase_ngrams(&mut self, phrase: &str, intent: &str, max_n: usize) {
        let tokens = tokenize(phrase);
        if tokens.len() < 2 { return; }

        const RATE: f32 = 0.4;

        for n in 2..=max_n.min(tokens.len()) {
            for window in tokens.windows(n) {
                let key = window.join("_");
                let entries = self.phrase_intent.entry(key).or_default();
                if let Some(e) = entries.iter_mut().find(|(id, _)| id == intent) {
                    e.1 = (e.1 + RATE * (1.0 - e.1)).min(1.0);
                } else {
                    entries.push((intent.to_string(), RATE));
                }
            }
        }
    }

    /// Score a query using BOTH unigram (existing) and n-gram scoring.
    fn score_combined(
        &self,
        query: &str,
        threshold: f32,
        gap: f32,
        ngram_bonus: f32,  // multiplier for n-gram contributions
    ) -> Vec<(String, f32)> {
        let tokens = tokenize(query);

        // --- Unigram scoring (existing L2 logic) ---
        let total_intents: usize = {
            let mut all: HashSet<&str> = HashSet::new();
            for entries in self.ig.word_intent.values() {
                for (id, _) in entries { all.insert(id.as_str()); }
            }
            // Also count intents from phrase_intent
            for entries in self.phrase_intent.values() {
                for (id, _) in entries { all.insert(id.as_str()); }
            }
            all.len().max(1)
        };

        let mut scores: HashMap<String, f32> = HashMap::new();

        // Unigram contribution
        for token in &tokens {
            if let Some(entries) = self.ig.word_intent.get(token.as_str()) {
                let idf = (total_intents as f32 / entries.len() as f32).ln().max(0.0);
                for (intent, weight) in entries {
                    *scores.entry(intent.clone()).or_default() += weight * idf;
                }
            }
        }

        // --- N-gram scoring (NEW) ---
        // Count total "documents" (intents) that have any n-gram association
        let ngram_doc_counts: HashMap<&str, usize> = {
            let mut counts: HashMap<&str, HashSet<&str>> = HashMap::new();
            for (ngram_key, entries) in &self.phrase_intent {
                for (intent, _) in entries {
                    counts.entry(ngram_key.as_str()).or_default().insert(intent.as_str());
                }
            }
            counts.into_iter().map(|(k, v)| (k, v.len())).collect()
        };

        for n in 2..=5 {
            if tokens.len() < n { break; }
            for window in tokens.windows(n) {
                let key = window.join("_");
                if let Some(entries) = self.phrase_intent.get(&key) {
                    let df = ngram_doc_counts.get(key.as_str()).copied().unwrap_or(1);
                    let idf = (total_intents as f32 / df as f32).ln().max(0.1);
                    // Length bonus: longer n-grams are more discriminative
                    let length_bonus = 1.0 + 0.5 * (n as f32 - 1.0);
                    for (intent, weight) in entries {
                        *scores.entry(intent.clone()).or_default() +=
                            weight * idf * ngram_bonus * length_bonus;
                    }
                }
            }
        }

        // Apply threshold + gap filter
        let mut sorted: Vec<(String, f32)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.is_empty() { return sorted; }
        let top = sorted[0].1;
        if top < threshold { return vec![]; }

        sorted.into_iter()
            .filter(|(_, s)| *s >= threshold && top - *s <= gap)
            .collect()
    }

    /// Baseline: unigram-only scoring (existing behavior)
    fn score_unigram_only(&self, query: &str, threshold: f32, gap: f32) -> Vec<(String, f32)> {
        let (results, _) = self.ig.score_multi_normalized(query, threshold, gap);
        results
    }
}

// ── Build corpus ──────────────────────────────────────────────────────────────

fn build() -> NgramIntentGraph {
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
            "can't log in",
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
        ("tickets:escalate_ticket", &[
            "please escalate my ticket",
            "need a faster response",
            "mark my ticket as high priority",
            "this issue is urgent",
        ]),
        ("tickets:check_ticket_status", &[
            "what is the status of my ticket",
            "any update on my request",
            "check on my open IT request",
        ]),
        ("hardware:report_broken", &[
            "my laptop is broken",
            "screen is cracked",
            "computer won't turn on",
        ]),
    ];

    // Seed unigrams (L2 standard)
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

    let mut nig = NgramIntentGraph::new(ig);

    // ── Simulate LLM distillation: learn n-gram patterns ──────────────────
    // These represent what the LLM would extract as intent-bearing spans
    // from real user queries.

    // escalate_ticket: frustration/urgency patterns
    let escalation_patterns = &[
        "been waiting all morning",
        "been waiting for hours",
        "been waiting so long",
        "this is ridiculous",
        "this is unacceptable",
        "this is absurd",
        "done with this",
        "done waiting",
        "going to cry",
        "at my wits end",
        "wits end",
        "getting nowhere",
        "not getting anywhere",
        "need faster response",
        "need this resolved now",
        "extremely frustrated",
        "so frustrated",
        "very frustrated",
        "lost patience",
        "literally impossible",
    ];
    for p in escalation_patterns {
        nig.learn_phrase_ngrams(p, "tickets:escalate_ticket", 5);
    }

    // request_loaner: need a temporary/spare device
    let loaner_patterns = &[
        "need a spare",
        "need something temporary",
        "borrow a machine",
        "use in the meantime",
        "while mine is being fixed",
        "temporary replacement",
        "need to work somehow",
    ];
    for p in loaner_patterns {
        nig.learn_phrase_ngrams(p, "hardware:request_loaner", 5);
    }

    // reset_password: can't get in, locked out
    let password_patterns = &[
        "can't get in",
        "can't get into",
        "locked out of",
        "won't let me in",
        "keeps saying wrong password",
        "forgot what my password",
    ];
    for p in password_patterns {
        nig.learn_phrase_ngrams(p, "account:reset_password", 5);
    }

    // wifi: connection issues at home/office
    let wifi_patterns = &[
        "internet is down",
        "internet not working",
        "connection keeps dropping",
        "connection is terrible",
        "can't get online",
        "no connectivity",
        "keeps disconnecting from network",
    ];
    for p in wifi_patterns {
        nig.learn_phrase_ngrams(p, "network:wifi", 5);
    }

    // report_broken: device damage language
    let broken_patterns = &[
        "completely dead",
        "won't turn on anymore",
        "screen is shattered",
        "fell and broke",
        "stopped working completely",
    ];
    for p in broken_patterns {
        nig.learn_phrase_ngrams(p, "hardware:report_broken", 5);
    }

    nig
}

// ── Test cases ────────────────────────────────────────────────────────────────

struct TestCase {
    query:    &'static str,
    expected: &'static [&'static str],
    label:    &'static str,
    category: &'static str,  // for grouping results
}

fn test_cases() -> Vec<TestCase> {
    vec![
        // ═══════ CATEGORY: Exact n-gram match (should definitely work) ═══════
        TestCase {
            query: "I've been waiting all morning for a response",
            expected: &["tickets:escalate_ticket"],
            label: "exact bigram: 'been waiting'",
            category: "exact",
        },
        TestCase {
            query: "this is ridiculous I need help now",
            expected: &["tickets:escalate_ticket"],
            label: "exact trigram: 'this is ridiculous'",
            category: "exact",
        },
        TestCase {
            query: "my internet is down and VPN won't connect",
            expected: &["network:wifi", "network:vpn"],
            label: "exact bigram: 'internet is down' + vpn",
            category: "exact",
        },
        TestCase {
            query: "I can't get into my account",
            expected: &["account:reset_password"],
            label: "exact trigram: 'can't get into'",
            category: "exact",
        },

        // ═══════ CATEGORY: Slight word variation (the critical test) ═══════
        TestCase {
            query: "I waited all morning and nobody responded",
            expected: &["tickets:escalate_ticket"],
            label: "variation: 'waited' vs learned 'waiting'",
            category: "variation",
        },
        TestCase {
            query: "this is absolutely ridiculous you keep ignoring me",
            expected: &["tickets:escalate_ticket"],
            label: "variation: extra word 'absolutely' inserted",
            category: "variation",
        },
        TestCase {
            query: "I have been waiting way too long",
            expected: &["tickets:escalate_ticket"],
            label: "variation: 'have been waiting' vs 'been waiting'",
            category: "variation",
        },
        TestCase {
            query: "been sitting here waiting since 9am",
            expected: &["tickets:escalate_ticket"],
            label: "variation: 'sitting here waiting' — different word order",
            category: "variation",
        },
        TestCase {
            query: "the internet connection keeps going in and out",
            expected: &["network:wifi"],
            label: "variation: 'connection keeps' — partial match of learned patterns",
            category: "variation",
        },
        TestCase {
            query: "my internet has been terrible all day",
            expected: &["network:wifi"],
            label: "variation: 'internet' unigram + no exact n-gram",
            category: "variation",
        },
        TestCase {
            query: "it won't let me log in no matter what I try",
            expected: &["account:reset_password"],
            label: "variation: 'won't let me' — close to 'won't let me in'",
            category: "variation",
        },

        // ═══════ CATEGORY: Realistic multi-intent (benchmark-style) ═══════
        TestCase {
            query: "I've been waiting all morning my internet is down and VPN won't connect either",
            expected: &["tickets:escalate_ticket", "network:wifi", "network:vpn"],
            label: "multi: escalate + wifi + vpn (3 intents)",
            category: "multi",
        },
        TestCase {
            query: "this is ridiculous I need a spare laptop mine is completely dead",
            expected: &["tickets:escalate_ticket", "hardware:request_loaner", "hardware:report_broken"],
            label: "multi: escalate + loaner + broken (3 intents)",
            category: "multi",
        },
        TestCase {
            query: "can't get into my account and I also need to set up my authenticator app",
            expected: &["account:reset_password", "account:setup_mfa"],
            label: "multi: password + mfa (2 intents)",
            category: "multi",
        },
        TestCase {
            query: "I'm so frustrated the VPN keeps dropping and I need a loaner while my laptop is being repaired",
            expected: &["tickets:escalate_ticket", "network:vpn", "hardware:request_loaner"],
            label: "multi: escalate + vpn + loaner (frustration + explicit)",
            category: "multi",
        },

        // ═══════ CATEGORY: Must NOT over-detect (precision test) ═══════
        TestCase {
            query: "my VPN is slow today",
            expected: &["network:vpn"],
            label: "precision: vpn only, must not trigger escalate or wifi",
            category: "precision",
        },
        TestCase {
            query: "what is the status of my support request",
            expected: &["tickets:check_ticket_status"],
            label: "precision: ticket status only, must not trigger escalate",
            category: "precision",
        },
        TestCase {
            query: "I need to set up MFA",
            expected: &["account:setup_mfa"],
            label: "precision: mfa only, simple",
            category: "precision",
        },
        TestCase {
            query: "can you install Slack on my computer",
            expected: &[],
            label: "precision: no matching intent (install not in corpus)",
            category: "precision",
        },

        // ═══════ CATEGORY: Chatty/indirect (hardest — new-hire style) ═══════
        TestCase {
            query: "um so like I've been here for three days and nothing is working and I'm just so frustrated I don't know what to do",
            expected: &["tickets:escalate_ticket"],
            label: "chatty: frustrated new hire — 'so frustrated' bigram",
            category: "chatty",
        },
        TestCase {
            query: "look I'm done waiting around you keep saying you'll look into it but nothing happens can I just get a loaner or something",
            expected: &["tickets:escalate_ticket", "hardware:request_loaner"],
            label: "chatty: frustrated + loaner request — 'done waiting' + 'get a loaner'",
            category: "chatty",
        },
        TestCase {
            query: "honestly at this point I'm at my wits end the internet has been garbage all week and I can't connect to anything",
            expected: &["tickets:escalate_ticket", "network:wifi"],
            label: "chatty: wits end + internet down — bigrams should fire",
            category: "chatty",
        },
    ]
}

// ── Scoring ───────────────────────────────────────────────────────────────────

fn result_ids(got: &[(String, f32)]) -> HashSet<String> {
    got.iter().map(|(id, _)| id.clone()).collect()
}

fn score_result(got: &[(String, f32)], expected: &[&str]) -> (&'static str, f32, f32) {
    let got_ids: HashSet<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
    let exp_ids: HashSet<&str> = expected.iter().copied().collect();

    if exp_ids.is_empty() && got_ids.is_empty() { return ("PASS", 1.0, 1.0); }
    if exp_ids.is_empty() && !got_ids.is_empty() { return ("FAIL", 0.0, 0.0); }
    if got_ids.is_empty() && !exp_ids.is_empty() { return ("FAIL", 0.0, 0.0); }

    let tp = got_ids.intersection(&exp_ids).count() as f32;
    let recall = tp / exp_ids.len() as f32;
    let precision = tp / got_ids.len() as f32;

    if got_ids == exp_ids {
        ("PASS", recall, precision)
    } else if tp > 0.0 {
        ("PARTIAL", recall, precision)
    } else {
        ("FAIL", recall, precision)
    }
}

fn short(id: &str) -> &str { id.split(':').last().unwrap_or(id) }

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let nig = build();

    let THRESHOLD: f32   = 0.3;
    let GAP: f32         = 1.5;
    let NGRAM_BONUS: f32 = 1.5;  // n-gram contributions weighted 1.5x vs unigrams

    let cases = test_cases();

    println!("\n{:=<78}", "");
    println!("  N-gram Intent Learning Experiment");
    println!("  8 intents | {} test cases | ngram_bonus={} gap={}", cases.len(), NGRAM_BONUS, GAP);
    println!("  {} learned n-gram patterns", nig.phrase_intent.len());
    println!("{:=<78}\n", "");

    let mut by_category: HashMap<&str, (usize, usize, usize, f32, f32)> = HashMap::new();
    let mut total_base_pass = 0usize;
    let mut total_ngram_pass = 0usize;
    let total = cases.len();

    let mut current_cat = "";

    for case in &cases {
        if case.category != current_cat {
            current_cat = case.category;
            println!("  ─── {} ───", current_cat.to_uppercase());
        }

        let baseline = nig.score_unigram_only(case.query, THRESHOLD, GAP);
        let ngram    = nig.score_combined(case.query, THRESHOLD, GAP, NGRAM_BONUS);

        let (bs, br, bp) = score_result(&baseline, case.expected);
        let (ns, nr, np) = score_result(&ngram,    case.expected);

        if bs == "PASS" { total_base_pass += 1; }
        if ns == "PASS" { total_ngram_pass += 1; }

        let cat = by_category.entry(case.category).or_insert((0, 0, 0, 0.0, 0.0));
        cat.0 += 1; // total
        if bs == "PASS" { cat.1 += 1; }
        if ns == "PASS" { cat.2 += 1; }
        cat.3 += nr;
        cat.4 += np;

        let improved  = ns == "PASS" && bs != "PASS";
        let regressed = bs == "PASS" && ns != "PASS";
        let marker = if improved { "⬆" } else if regressed { "⬇" } else { " " };

        // Compact output
        let base_ids: Vec<&str> = baseline.iter().map(|(id, _)| short(id)).collect();
        let ngram_ids: Vec<&str> = ngram.iter().map(|(id, _)| short(id)).collect();
        let exp_ids: Vec<&str> = case.expected.iter().map(|id| short(id)).collect();

        println!("{} [{}]  \"{}\"",
            marker, case.label,
            if case.query.len() > 70 { &case.query[..70] } else { case.query });

        println!("    Baseline [{:<7}] {:?}", bs, base_ids);
        println!("    +Ngram   [{:<7}] {:?}", ns, ngram_ids);
        if ns != "PASS" || bs != "PASS" {
            println!("    Expected          {:?}", exp_ids);
        }
        println!();
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("{:=<78}", "");
    println!("  RESULTS BY CATEGORY");
    println!("{:=<78}", "");

    let categories = ["exact", "variation", "multi", "precision", "chatty"];
    for cat in categories {
        if let Some(&(n, base_p, ngram_p, recall_sum, prec_sum)) = by_category.get(cat) {
            let avg_recall = if n > 0 { recall_sum / n as f32 } else { 0.0 };
            let avg_prec = if n > 0 { prec_sum / n as f32 } else { 0.0 };
            println!("  {:<12}  n={:<3}  base={}/{} ({:.0}%)  ngram={}/{} ({:.0}%)  Δ={:+}  avg_recall={:.0}% avg_prec={:.0}%",
                cat, n,
                base_p, n, 100.0 * base_p as f32 / n as f32,
                ngram_p, n, 100.0 * ngram_p as f32 / n as f32,
                ngram_p as i32 - base_p as i32,
                100.0 * avg_recall, 100.0 * avg_prec);
        }
    }

    println!("\n  {}", "─".repeat(70));
    println!("  TOTAL:  base={}/{} ({:.0}%)  ngram={}/{} ({:.0}%)  Δ={:+}",
        total_base_pass, total, 100.0 * total_base_pass as f32 / total as f32,
        total_ngram_pass, total, 100.0 * total_ngram_pass as f32 / total as f32,
        total_ngram_pass as i32 - total_base_pass as i32);

    let delta = total_ngram_pass as i32 - total_base_pass as i32;
    println!();
    if delta > 3 {
        println!("  ✓ N-GRAM LEARNING IS THE PATH — significant improvement across categories");
    } else if delta > 0 {
        println!("  ~ Modest improvement — n-grams help but may need tuning or more patterns");
    } else if delta == 0 {
        println!("  ✗ No improvement — check n-gram generation, bonus weight, or pattern coverage");
    } else {
        println!("  ✗ REGRESSION — n-grams causing false positives, reduce bonus or tighten gap");
    }

    // Show which n-gram patterns fired for a sample query
    println!("\n{:=<78}", "");
    println!("  DIAGNOSTIC: N-gram matches for sample queries");
    println!("{:=<78}\n", "");

    let diagnostic_queries = [
        "I've been waiting all morning for a response",
        "I waited all morning and nobody responded",
        "this is absolutely ridiculous you keep ignoring me",
        "um so like I've been here for three days and nothing is working and I'm just so frustrated",
        "can't get into my account",
    ];

    for q in diagnostic_queries {
        let tokens = tokenize(q);
        println!("  Query: \"{}\"", if q.len() > 65 { &q[..65] } else { q });
        println!("  Tokens: {:?}", tokens);
        let mut hits: Vec<(String, &str, f32)> = Vec::new();
        for n in 2..=5 {
            if tokens.len() < n { break; }
            for window in tokens.windows(n) {
                let key = window.join("_");
                if let Some(entries) = nig.phrase_intent.get(&key) {
                    for (intent, weight) in entries {
                        hits.push((key.clone(), short(intent), *weight));
                    }
                }
            }
        }
        if hits.is_empty() {
            println!("  N-gram hits: NONE");
        } else {
            for (ngram, intent, w) in &hits {
                println!("    → \"{}\" → {} (w={:.2})", ngram, intent, w);
            }
        }
        println!();
    }
}
