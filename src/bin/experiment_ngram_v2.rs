/// Experiment v2: N-gram intent learning WITH stop words preserved.
///
/// Key change from v1: stop words ("been", "this", "is", etc.) are KEPT
/// for n-gram generation. IDF handles weighting — stop words contribute
/// near-zero unigram score but provide essential n-gram structure.
///
/// Also tests: does the same tokenization pipeline handle morphological
/// variants ("waited" vs "waiting") via L1 normalization?
///
/// Run: cargo run --bin experiment_ngram_v2
use asv_router::tokenizer;
use std::collections::{HashMap, HashSet};

// ── Tokenizer that preserves stop words ───────────────────────────────────────

/// Simple tokenizer: lowercase, split on whitespace/punctuation, expand contractions.
/// NO stop word removal. This preserves "been", "this", "is", etc. for n-gram matching.
fn tokenize_full(text: &str) -> Vec<String> {
    // Expand contractions first (same as asv tokenizer)
    let text = text.replace("n't", " not");
    let text = text.replace("'ve", " have");
    let text = text.replace("'re", " are");
    let text = text.replace("'m", " am");
    let text = text.replace("'ll", " will");
    let text = text.replace("'s", "");  // possessive/is — drop
    let text = text.replace("'d", " would");

    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && w.len() >= 2)  // drop single chars but keep "is", "be", etc.
        .map(|w| w.to_string())
        .collect()
}

/// Standard ASV tokenizer (for comparison — removes stop words)
fn tokenize_asv(text: &str) -> Vec<String> {
    tokenizer::tokenize(text).into_iter().map(|s| s.to_string()).collect()
}

// ── N-gram scoring engine ─────────────────────────────────────────────────────

struct NgramEngine {
    /// unigram → (intent, weight)
    word_intent: HashMap<String, Vec<(String, f32)>>,
    /// ngram_key → (intent, weight)
    phrase_intent: HashMap<String, Vec<(String, f32)>>,
}

impl NgramEngine {
    fn new() -> Self {
        Self { word_intent: HashMap::new(), phrase_intent: HashMap::new() }
    }

    /// Learn a seed phrase as unigrams (using ASV tokenizer — stop words removed)
    fn learn_unigrams(&mut self, phrase: &str, intent: &str) {
        const RATE: f32 = 0.4;
        for token in tokenize_asv(phrase) {
            let entries = self.word_intent.entry(token).or_default();
            if let Some(e) = entries.iter_mut().find(|(id, _)| id == intent) {
                e.1 = (e.1 + RATE * (1.0 - e.1)).min(1.0);
            } else {
                entries.push((intent.to_string(), RATE));
            }
        }
    }

    /// Learn a phrase as n-grams at lengths 2..=max_n.
    /// Uses FULL tokenizer (stop words preserved) to capture structural patterns.
    fn learn_ngrams(&mut self, phrase: &str, intent: &str, max_n: usize) {
        const RATE: f32 = 0.4;
        let tokens = tokenize_full(phrase);
        if tokens.len() < 2 { return; }

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

    fn total_intents(&self) -> f32 {
        let mut all: HashSet<&str> = HashSet::new();
        for entries in self.word_intent.values() {
            for (id, _) in entries { all.insert(id.as_str()); }
        }
        for entries in self.phrase_intent.values() {
            for (id, _) in entries { all.insert(id.as_str()); }
        }
        all.len().max(1) as f32
    }

    /// Score using unigrams only (ASV tokenizer, stop words removed)
    fn score_unigram(&self, query: &str, threshold: f32, gap: f32) -> Vec<(String, f32)> {
        let tokens = tokenize_asv(query);
        let n = self.total_intents();
        let mut scores: HashMap<String, f32> = HashMap::new();

        for token in &tokens {
            if let Some(entries) = self.word_intent.get(token.as_str()) {
                let idf = (n / entries.len() as f32).ln().max(0.0);
                for (intent, weight) in entries {
                    *scores.entry(intent.clone()).or_default() += weight * idf;
                }
            }
        }

        Self::apply_filter(scores, threshold, gap)
    }

    /// Score using unigrams + n-grams (n-grams use FULL tokenizer)
    fn score_combined(&self, query: &str, threshold: f32, gap: f32, ngram_bonus: f32) -> Vec<(String, f32)> {
        let n = self.total_intents();
        let mut scores: HashMap<String, f32> = HashMap::new();

        // Unigram pass (ASV tokenizer — stop words removed, IDF-weighted)
        let uni_tokens = tokenize_asv(query);
        for token in &uni_tokens {
            if let Some(entries) = self.word_intent.get(token.as_str()) {
                let idf = (n / entries.len() as f32).ln().max(0.0);
                for (intent, weight) in entries {
                    *scores.entry(intent.clone()).or_default() += weight * idf;
                }
            }
        }

        // N-gram pass (FULL tokenizer — stop words preserved)
        let full_tokens = tokenize_full(query);
        for ng_len in 2..=5 {
            if full_tokens.len() < ng_len { break; }
            // Length bonus: longer matches = more discriminative
            let length_bonus = 1.0 + 0.5 * (ng_len as f32 - 1.0);
            for window in full_tokens.windows(ng_len) {
                let key = window.join("_");
                if let Some(entries) = self.phrase_intent.get(&key) {
                    let idf = (n / entries.len() as f32).ln().max(0.0);
                    for (intent, weight) in entries {
                        *scores.entry(intent.clone()).or_default() +=
                            weight * idf * ngram_bonus * length_bonus;
                    }
                }
            }
        }

        Self::apply_filter(scores, threshold, gap)
    }

    fn apply_filter(scores: HashMap<String, f32>, threshold: f32, gap: f32) -> Vec<(String, f32)> {
        let mut sorted: Vec<(String, f32)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if sorted.is_empty() { return sorted; }
        let top = sorted[0].1;
        if top < threshold { return vec![]; }
        sorted.into_iter().filter(|(_, s)| *s >= threshold && top - *s <= gap).collect()
    }

    /// Diagnostic: show which n-grams fire for a query
    fn diagnose(&self, query: &str) {
        let full = tokenize_full(query);
        let asv = tokenize_asv(query);
        println!("    ASV tokens:  {:?}", asv);
        println!("    Full tokens: {:?}", full);
        let mut hits = Vec::new();
        for ng_len in 2..=5 {
            if full.len() < ng_len { break; }
            for window in full.windows(ng_len) {
                let key = window.join("_");
                if let Some(entries) = self.phrase_intent.get(&key) {
                    for (intent, weight) in entries {
                        hits.push((key.clone(), intent.split(':').last().unwrap_or(intent).to_string(), *weight));
                    }
                }
            }
        }
        if hits.is_empty() {
            println!("    N-gram hits:  NONE");
        } else {
            for (ng, intent, w) in &hits {
                println!("    N-gram hit:   \"{}\" → {} (w={:.2})", ng, intent, w);
            }
        }
    }
}

// ── Build corpus ──────────────────────────────────────────────────────────────

fn build() -> NgramEngine {
    let mut e = NgramEngine::new();

    // Seed intents with unigrams (standard phrases)
    let intents: &[(&str, &[&str])] = &[
        ("network:vpn", &[
            "can't connect to VPN", "VPN is not working",
            "VPN keeps disconnecting", "remote VPN connection failed",
        ]),
        ("network:wifi", &[
            "wifi is not connecting", "no internet connection",
            "wifi signal is very weak", "internet keeps dropping",
        ]),
        ("account:reset_password", &[
            "reset my password", "forgot my password",
            "password expired", "can't log in",
        ]),
        ("account:setup_mfa", &[
            "set up two-factor authentication", "configure my authenticator app",
            "enable MFA on my account",
        ]),
        ("hardware:request_loaner", &[
            "need a loaner laptop", "borrow a temporary device",
            "replacement laptop while mine is repaired",
        ]),
        ("tickets:escalate_ticket", &[
            "please escalate my ticket", "need a faster response",
            "mark my ticket as high priority", "this issue is urgent",
        ]),
        ("tickets:check_ticket_status", &[
            "what is the status of my ticket", "any update on my request",
        ]),
        ("hardware:report_broken", &[
            "my laptop is broken", "screen is cracked", "computer won't turn on",
        ]),
    ];

    for (intent, phrases) in intents {
        for phrase in *phrases {
            e.learn_unigrams(phrase, intent);
        }
    }

    // ── N-gram patterns (simulating LLM span extraction) ──────────────────
    // These use tokenize_full (stop words preserved) so "been_waiting" works.

    // escalate_ticket: frustration/urgency as multi-word patterns
    let escalation = &[
        "been waiting", "been waiting all morning", "been waiting for hours",
        "been waiting so long", "have been waiting",
        "this is ridiculous", "this is unacceptable", "this is absurd",
        "so frustrated", "extremely frustrated", "very frustrated",
        "done with this", "done waiting", "I am done",
        "at my wits end", "wits end",
        "getting nowhere", "not getting anywhere",
        "going to cry", "literally going to cry",
        "lost patience", "lost my patience",
        "need this resolved now", "need faster response",
    ];
    for p in escalation { e.learn_ngrams(p, "tickets:escalate_ticket", 5); }

    // reset_password: "can't get in" patterns
    let password = &[
        "can not get in", "can not get into", "can not log in",
        "won not let me in", "locked out of", "locked me out",
        "keeps saying wrong password", "forgot what my password is",
        "not let me log in", "not let me sign in",
    ];
    for p in password { e.learn_ngrams(p, "account:reset_password", 5); }

    // wifi: connectivity patterns
    let wifi = &[
        "internet is down", "internet not working", "internet is not working",
        "connection keeps dropping", "connection is terrible",
        "can not get online", "no connectivity",
        "keeps disconnecting from the network",
    ];
    for p in wifi { e.learn_ngrams(p, "network:wifi", 5); }

    // report_broken: device damage
    let broken = &[
        "completely dead", "is completely dead",
        "won not turn on", "stopped working completely",
        "screen is shattered", "fell and broke",
    ];
    for p in broken { e.learn_ngrams(p, "hardware:report_broken", 5); }

    // request_loaner: temporary device
    let loaner = &[
        "need a spare", "need something temporary",
        "in the meantime", "while mine is being fixed",
        "temporary replacement",
    ];
    for p in loaner { e.learn_ngrams(p, "hardware:request_loaner", 5); }

    println!("  Corpus: {} unigram entries, {} n-gram patterns",
        e.word_intent.len(), e.phrase_intent.len());
    e
}

// ── Test harness ──────────────────────────────────────────────────────────────

struct TC { q: &'static str, exp: &'static [&'static str], label: &'static str, cat: &'static str }

fn cases() -> Vec<TC> {
    vec![
        // ─── EXACT N-GRAM ───
        TC { q: "I've been waiting all morning", exp: &["tickets:escalate_ticket"],
             label: "exact: 'been waiting all morning'", cat: "exact" },
        TC { q: "this is ridiculous", exp: &["tickets:escalate_ticket"],
             label: "exact: 'this is ridiculous'", cat: "exact" },
        TC { q: "my internet is down", exp: &["network:wifi"],
             label: "exact: 'internet is down'", cat: "exact" },
        TC { q: "I can't get into my account", exp: &["account:reset_password"],
             label: "exact: 'can not get into'", cat: "exact" },
        TC { q: "my laptop is completely dead", exp: &["hardware:report_broken"],
             label: "exact: 'completely dead'", cat: "exact" },

        // ─── VARIATION: slight wording changes ───
        TC { q: "I've been waiting for hours and no one is helping",
             exp: &["tickets:escalate_ticket"],
             label: "var: 'been waiting for hours'", cat: "variation" },
        TC { q: "I waited all morning and nobody responded",
             exp: &["tickets:escalate_ticket"],
             label: "var: 'waited' not 'waiting' — tests morphology", cat: "variation" },
        TC { q: "this is absolutely ridiculous you keep ignoring me",
             exp: &["tickets:escalate_ticket"],
             label: "var: extra word 'absolutely' between 'is' and 'ridiculous'", cat: "variation" },
        TC { q: "I have been waiting way too long for this",
             exp: &["tickets:escalate_ticket"],
             label: "var: 'have been waiting' — extra 'have'", cat: "variation" },
        TC { q: "been sitting here waiting since 9am",
             exp: &["tickets:escalate_ticket"],
             label: "var: reordered — 'sitting here waiting'", cat: "variation" },
        TC { q: "the internet connection keeps going in and out",
             exp: &["network:wifi"],
             label: "var: 'connection keeps' — partial match", cat: "variation" },
        TC { q: "it won't let me log in no matter what I try",
             exp: &["account:reset_password"],
             label: "var: 'not let me log in'", cat: "variation" },
        TC { q: "I'm so incredibly frustrated right now",
             exp: &["tickets:escalate_ticket"],
             label: "var: 'so frustrated' with 'incredibly' inserted", cat: "variation" },
        TC { q: "this is completely unacceptable and I want it fixed",
             exp: &["tickets:escalate_ticket"],
             label: "var: 'this is unacceptable' with 'completely' inserted", cat: "variation" },

        // ─── MULTI-INTENT ───
        TC { q: "I've been waiting all morning my internet is down and VPN won't connect",
             exp: &["tickets:escalate_ticket", "network:wifi", "network:vpn"],
             label: "multi: escalate + wifi + vpn", cat: "multi" },
        TC { q: "this is ridiculous I need a spare laptop mine is completely dead",
             exp: &["tickets:escalate_ticket", "hardware:request_loaner", "hardware:report_broken"],
             label: "multi: escalate + loaner + broken", cat: "multi" },
        TC { q: "can't get into my account and I also need to set up my authenticator app",
             exp: &["account:reset_password", "account:setup_mfa"],
             label: "multi: password + mfa", cat: "multi" },

        // ─── PRECISION ───
        TC { q: "my VPN is slow today", exp: &["network:vpn"],
             label: "prec: vpn only", cat: "precision" },
        TC { q: "what is the status of my support request",
             exp: &["tickets:check_ticket_status"],
             label: "prec: ticket status only — must NOT trigger escalate", cat: "precision" },
        TC { q: "I need to set up MFA", exp: &["account:setup_mfa"],
             label: "prec: mfa only", cat: "precision" },

        // ─── CHATTY (hardest) ───
        TC { q: "um so like I've been here for three days and nothing is working and I'm just so frustrated I don't know what to do",
             exp: &["tickets:escalate_ticket"],
             label: "chatty: new hire frustration — 'so frustrated' bigram", cat: "chatty" },
        TC { q: "look I'm done waiting around you keep saying you'll look into it but nothing happens can I just get a loaner or something",
             exp: &["tickets:escalate_ticket", "hardware:request_loaner"],
             label: "chatty: 'done waiting' + loaner", cat: "chatty" },
        TC { q: "honestly at this point I'm at my wits end the internet has been garbage all week",
             exp: &["tickets:escalate_ticket", "network:wifi"],
             label: "chatty: 'wits end' + internet", cat: "chatty" },
    ]
}

fn short(id: &str) -> &str { id.split(':').last().unwrap_or(id) }

fn eval(got: &[(String, f32)], exp: &[&str]) -> &'static str {
    let g: HashSet<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
    let e: HashSet<&str> = exp.iter().copied().collect();
    if g == e { "PASS" } else if !g.is_disjoint(&e) { "PARTIAL" } else { "FAIL" }
}

fn main() {
    println!("\n{:=<78}", "");
    println!("  N-gram v2: Stop Words PRESERVED for n-gram matching");
    println!("{:=<78}\n", "");

    let eng = build();
    let cases = cases();

    let THRESHOLD: f32   = 0.3;
    let GAP: f32         = 1.5;
    let NGRAM_BONUS: f32 = 1.5;

    let mut base_pass = 0usize;
    let mut ngram_pass = 0usize;
    let mut by_cat: HashMap<&str, (usize, usize, usize)> = HashMap::new();
    let mut cur_cat = "";

    for tc in &cases {
        if tc.cat != cur_cat {
            cur_cat = tc.cat;
            println!("  ─── {} ───", cur_cat.to_uppercase());
        }

        let base  = eng.score_unigram(tc.q, THRESHOLD, GAP);
        let ngram = eng.score_combined(tc.q, THRESHOLD, GAP, NGRAM_BONUS);

        let bs = eval(&base, tc.exp);
        let ns = eval(&ngram, tc.exp);
        if bs == "PASS" { base_pass += 1; }
        if ns == "PASS" { ngram_pass += 1; }

        let cat = by_cat.entry(tc.cat).or_insert((0, 0, 0));
        cat.0 += 1;
        if bs == "PASS" { cat.1 += 1; }
        if ns == "PASS" { cat.2 += 1; }

        let marker = match (bs, ns) {
            (_, "PASS") if bs != "PASS" => "⬆",
            ("PASS", _) if ns != "PASS" => "⬇",
            _ => " ",
        };

        let bi: Vec<&str> = base.iter().map(|(id, _)| short(id)).collect();
        let ni: Vec<&str> = ngram.iter().map(|(id, _)| short(id)).collect();
        let ei: Vec<&str> = tc.exp.iter().map(|id| short(id)).collect();

        println!("{} [{}]", marker, tc.label);
        println!("    Base  [{:<7}] {:?}", bs, bi);
        println!("    Ngram [{:<7}] {:?}", ns, ni);
        if bs != "PASS" || ns != "PASS" {
            println!("    Exp            {:?}", ei);
            eng.diagnose(tc.q);
        }
        println!();
    }

    // ── Summary ───────────────────────────────────────────────────────────
    let total = cases.len();
    println!("{:=<78}", "");
    println!("  BY CATEGORY:");
    for cat in &["exact", "variation", "precision", "multi", "chatty"] {
        if let Some(&(n, bp, np)) = by_cat.get(cat) {
            println!("    {:<12} base={}/{} ({:>3.0}%)   ngram={}/{} ({:>3.0}%)   Δ={:+}",
                cat, bp, n, 100.0 * bp as f32 / n as f32,
                np, n, 100.0 * np as f32 / n as f32,
                np as i32 - bp as i32);
        }
    }
    println!("\n  TOTAL:     base={}/{} ({:.0}%)   ngram={}/{} ({:.0}%)   Δ={:+}",
        base_pass, total, 100.0 * base_pass as f32 / total as f32,
        ngram_pass, total, 100.0 * ngram_pass as f32 / total as f32,
        ngram_pass as i32 - base_pass as i32);

    let delta = ngram_pass as i32 - base_pass as i32;
    if delta >= 5 {
        println!("\n  ✓ STRONG SIGNAL — n-grams with preserved stop words work");
    } else if delta > 0 {
        println!("\n  ~ Improvement but needs more — check variation failures");
    } else {
        println!("\n  ✗ Not working — examine diagnostic output above");
    }
}
