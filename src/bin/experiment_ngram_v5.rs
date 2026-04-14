/// Experiment v5: Re-pass architecture (user's idea)
///
/// Instead of OMP token subtraction, simply re-run the same scoring pipeline
/// with already-detected intents excluded from candidates. The gap filter
/// naturally recalculates from the new top score.
///
/// Gate: only proceed to round N+1 if new top score ≥ original_top * gate_ratio.
/// This prevents noise from being picked up as intents.
///
/// Run: cargo run --bin experiment_ngram_v5
use asv_router::tokenizer;
use std::collections::{HashMap, HashSet};

// ── Tokenizers ────────────────────────────────────────────────────────────────

fn tokenize_full(text: &str) -> Vec<String> {
    let text = text.replace("n't", " not");
    let text = text.replace("'ve", " have");
    let text = text.replace("'re", " are");
    let text = text.replace("'m", " am");
    let text = text.replace("'ll", " will");
    let text = text.replace("'s", "");
    let text = text.replace("'d", " would");
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && w.len() >= 2)
        .map(|w| w.to_string())
        .collect()
}

fn tokenize_asv(text: &str) -> Vec<String> {
    tokenizer::tokenize(text).into_iter().map(|s| s.to_string()).collect()
}

fn cjk_chars(text: &str) -> Vec<char> {
    text.chars().filter(|c| tokenizer::is_cjk(*c)).collect()
}

fn generate_skip_bigrams(tokens: &[String], max_gap: usize) -> Vec<String> {
    let mut result = Vec::new();
    for i in 0..tokens.len() {
        for j in (i + 2)..=(i + 1 + max_gap).min(tokens.len() - 1) {
            result.push(format!("{}~{}", tokens[i], tokens[j]));
        }
    }
    result
}

// ── Engine ────────────────────────────────────────────────────────────────────

struct Engine {
    word_intent: HashMap<String, Vec<(String, f32)>>,
    phrase_intent: HashMap<String, Vec<(String, f32)>>,
}

impl Engine {
    fn new() -> Self { Self { word_intent: HashMap::new(), phrase_intent: HashMap::new() } }

    fn learn_uni(&mut self, phrase: &str, intent: &str) {
        const R: f32 = 0.4;
        for t in tokenize_asv(phrase) {
            let e = self.word_intent.entry(t).or_default();
            if let Some(x) = e.iter_mut().find(|(id, _)| id == intent) {
                x.1 = (x.1 + R * (1.0 - x.1)).min(1.0);
            } else { e.push((intent.to_string(), R)); }
        }
    }

    fn learn_pattern(&mut self, key: &str, intent: &str) {
        const R: f32 = 0.4;
        let e = self.phrase_intent.entry(key.to_string()).or_default();
        if let Some(x) = e.iter_mut().find(|(id, _)| id == intent) {
            x.1 = (x.1 + R * (1.0 - x.1)).min(1.0);
        } else { e.push((intent.to_string(), R)); }
    }

    fn learn_ngrams(&mut self, phrase: &str, intent: &str, max_n: usize, max_gap: usize) {
        let has_cjk = phrase.chars().any(tokenizer::is_cjk);
        if has_cjk {
            let chars = cjk_chars(phrase);
            for n in 2..=max_n.min(chars.len()) {
                for w in chars.windows(n) {
                    self.learn_pattern(&w.iter().collect::<String>(), intent);
                }
            }
            let strs: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
            for sg in generate_skip_bigrams(&strs, max_gap) {
                self.learn_pattern(&sg, intent);
            }
        } else {
            let tokens = tokenize_full(phrase);
            for n in 2..=max_n.min(tokens.len()) {
                for w in tokens.windows(n) { self.learn_pattern(&w.join("_"), intent); }
            }
            for sg in generate_skip_bigrams(&tokens, max_gap) {
                self.learn_pattern(&sg, intent);
            }
        }
    }

    fn total_intents(&self) -> f32 {
        let mut all: HashSet<&str> = HashSet::new();
        for v in self.word_intent.values() { for (id, _) in v { all.insert(id); } }
        for v in self.phrase_intent.values() { for (id, _) in v { all.insert(id); } }
        all.len().max(1) as f32
    }

    /// Core scoring: returns ALL intents above threshold, sorted by score.
    /// `exclude`: intents to skip (already detected in previous round).
    fn score_all(&self, query: &str, threshold: f32, ngram_bonus: f32,
                 exclude: &HashSet<String>) -> Vec<(String, f32)> {
        let n = self.total_intents();
        let mut scores: HashMap<String, f32> = HashMap::new();

        // Unigrams (ASV tokenizer)
        for token in tokenize_asv(query) {
            if let Some(entries) = self.word_intent.get(token.as_str()) {
                let idf = (n / entries.len() as f32).ln().max(0.0);
                for (intent, weight) in entries {
                    if exclude.contains(intent) { continue; }
                    *scores.entry(intent.clone()).or_default() += weight * idf;
                }
            }
        }

        // N-grams + skip-grams (full tokenizer for Latin, char n-grams for CJK)
        let has_cjk = query.chars().any(tokenizer::is_cjk);
        let mut ngram_keys: Vec<(String, f32)> = Vec::new();

        if has_cjk {
            let chars = cjk_chars(query);
            for ng_n in 2..=5.min(chars.len()) {
                let lb = 1.0 + 0.5 * (ng_n as f32 - 1.0);
                for w in chars.windows(ng_n) {
                    ngram_keys.push((w.iter().collect(), lb));
                }
            }
            let strs: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
            for sg in generate_skip_bigrams(&strs, 2) {
                ngram_keys.push((sg, 1.1));
            }
        } else {
            let full = tokenize_full(query);
            for ng_n in 2..=4.min(full.len()) {
                let lb = 1.0 + 0.5 * (ng_n as f32 - 1.0);
                for w in full.windows(ng_n) {
                    ngram_keys.push((w.join("_"), lb));
                }
            }
            for sg in generate_skip_bigrams(&full, 2) {
                ngram_keys.push((sg, 1.2));
            }
        }

        for (key, len_bonus) in &ngram_keys {
            if let Some(entries) = self.phrase_intent.get(key.as_str()) {
                let idf = (n / entries.len() as f32).ln().max(0.0);
                for (intent, weight) in entries {
                    if exclude.contains(intent) { continue; }
                    *scores.entry(intent.clone()).or_default() +=
                        weight * idf * ngram_bonus * len_bonus;
                }
            }
        }

        let mut sorted: Vec<(String, f32)> = scores.into_iter()
            .filter(|(_, s)| *s >= threshold)
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// THE RE-PASS ARCHITECTURE:
    /// 1. Score all intents → apply gap filter → confirm top group
    /// 2. If strong residual signal: exclude confirmed, re-score, re-apply gap
    /// 3. Repeat until gate fails or max rounds
    fn route_repass(&self, query: &str, threshold: f32, gap: f32,
                     ngram_bonus: f32, gate_ratio: f32, max_rounds: usize) -> Vec<(String, f32)> {
        let mut confirmed: Vec<(String, f32)> = Vec::new();
        let mut excluded: HashSet<String> = HashSet::new();
        let mut original_top: f32 = 0.0;

        for round in 0..max_rounds {
            let all = self.score_all(query, threshold, ngram_bonus, &excluded);
            if all.is_empty() { break; }

            let round_top = all[0].1;

            // Gate: is this round's signal strong enough relative to original?
            if round == 0 {
                original_top = round_top;
            } else if round_top < original_top * gate_ratio {
                break;  // residual is noise, stop
            }

            // Apply gap filter from THIS round's top
            let passed: Vec<(String, f32)> = all.into_iter()
                .filter(|(_, s)| round_top - *s <= gap)
                .collect();

            if passed.is_empty() { break; }

            for (id, score) in &passed {
                confirmed.push((id.clone(), *score));
                excluded.insert(id.clone());
            }
        }

        confirmed
    }

    /// Baseline: unigram only, single pass with gap
    fn route_baseline(&self, query: &str, threshold: f32, gap: f32) -> Vec<(String, f32)> {
        let all = self.score_all(query, threshold, 0.0, &HashSet::new());
        if all.is_empty() { return all; }
        let top = all[0].1;
        all.into_iter().filter(|(_, s)| top - *s <= gap).collect()
    }

    /// N-gram single pass with gap (no re-pass)
    fn route_ngram_single(&self, query: &str, threshold: f32, gap: f32, bonus: f32) -> Vec<(String, f32)> {
        let all = self.score_all(query, threshold, bonus, &HashSet::new());
        if all.is_empty() { return all; }
        let top = all[0].1;
        all.into_iter().filter(|(_, s)| top - *s <= gap).collect()
    }
}

// ── Corpus ────────────────────────────────────────────────────────────────────

fn build() -> Engine {
    let mut e = Engine::new();

    let intents: &[(&str, &[&str])] = &[
        ("network:vpn", &["can't connect to VPN", "VPN is not working", "VPN keeps disconnecting"]),
        ("network:wifi", &["wifi is not connecting", "no internet connection", "internet keeps dropping"]),
        ("account:reset_password", &["reset my password", "forgot my password", "can't log in"]),
        ("account:setup_mfa", &["set up two-factor authentication", "configure my authenticator app"]),
        ("hardware:request_loaner", &["need a loaner laptop", "borrow a temporary device"]),
        ("hardware:report_broken", &["my laptop is broken", "computer won't turn on"]),
        ("tickets:escalate_ticket", &["please escalate my ticket", "mark my ticket as high priority"]),
        ("tickets:check_ticket_status", &["what is the status of my ticket"]),
    ];
    for (i, ps) in intents { for p in *ps { e.learn_uni(p, i); } }

    // CJK seeds
    let cjk: &[(&str, &[&str])] = &[
        ("network:vpn", &["VPN连不上", "连接VPN失败"]),
        ("network:wifi", &["网络连不上", "上不了网", "WiFi断了"]),
        ("account:reset_password", &["重置密码", "忘记密码", "登录不了"]),
        ("tickets:escalate_ticket", &["加急处理", "太荒谬了"]),
        ("hardware:report_broken", &["电脑坏了", "开不了机"]),
        ("hardware:request_loaner", &["借一台临时电脑", "需要备用机"]),
    ];
    for (i, ps) in cjk { for p in *ps { e.learn_uni(p, i); } }

    // Latin n-gram + skip-gram patterns
    let ng: &[(&str, &[&str])] = &[
        ("tickets:escalate_ticket", &[
            "been waiting", "been waiting all morning", "been waiting for hours",
            "have been waiting", "this is ridiculous", "this is unacceptable",
            "so frustrated", "extremely frustrated",
            "done with this", "done waiting", "at my wits end", "wits end",
        ]),
        ("account:reset_password", &[
            "can not get in", "can not get into", "can not log in",
            "not let me in", "not let me log in",
        ]),
        ("network:wifi", &[
            "internet is down", "internet not working",
            "connection keeps dropping", "can not get online",
        ]),
        ("hardware:report_broken", &["completely dead", "is completely dead", "won not turn on"]),
        ("hardware:request_loaner", &["need a spare", "while mine is being fixed"]),
    ];
    for (i, ps) in ng { for p in *ps { e.learn_ngrams(p, i, 4, 2); } }

    // CJK n-gram patterns
    let cjk_ng: &[(&str, &[&str])] = &[
        ("tickets:escalate_ticket", &["等了一上午", "一直在等", "太荒谬了", "受不了了", "到底什么时候", "等了很久", "非常着急"]),
        ("account:reset_password", &["登不进去", "进不去", "登录失败"]),
        ("network:wifi", &["连不上网", "上不了网", "网络一直断"]),
    ];
    for (i, ps) in cjk_ng { for p in *ps { e.learn_ngrams(p, i, 4, 2); } }

    println!("  {} unigram, {} n-gram/skip patterns", e.word_intent.len(), e.phrase_intent.len());
    e
}

// ── Tests ─────────────────────────────────────────────────────────────────────

struct TC { q: &'static str, exp: &'static [&'static str], label: &'static str, cat: &'static str }

fn cases() -> Vec<TC> {
    vec![
        // ── MULTI-INTENT ──
        TC { q: "I've been waiting all morning my internet is down and VPN won't connect",
             exp: &["tickets:escalate_ticket", "network:wifi", "network:vpn"],
             label: "multi: escalate + wifi + vpn (3)", cat: "multi" },
        TC { q: "this is ridiculous I need a spare laptop mine is completely dead",
             exp: &["tickets:escalate_ticket", "hardware:request_loaner", "hardware:report_broken"],
             label: "multi: escalate + loaner + broken (3)", cat: "multi" },
        TC { q: "can't get into my account and also need to set up my authenticator",
             exp: &["account:reset_password", "account:setup_mfa"],
             label: "multi: password + mfa (2)", cat: "multi" },
        TC { q: "the VPN keeps dropping and my wifi is terrible I've been waiting for hours",
             exp: &["network:vpn", "network:wifi", "tickets:escalate_ticket"],
             label: "multi: vpn + wifi + escalate (3)", cat: "multi" },
        TC { q: "I'm so frustrated my laptop is broken and I need a loaner while it's fixed",
             exp: &["tickets:escalate_ticket", "hardware:report_broken", "hardware:request_loaner"],
             label: "multi: frustrated + broken + loaner (3)", cat: "multi" },

        // ── SINGLE + VARIATION ──
        TC { q: "I've been waiting all morning", exp: &["tickets:escalate_ticket"],
             label: "single: escalate", cat: "single" },
        TC { q: "this is absolutely ridiculous", exp: &["tickets:escalate_ticket"],
             label: "var: skip-gram across 'absolutely'", cat: "single" },
        TC { q: "been sitting here waiting since morning",
             exp: &["tickets:escalate_ticket"],
             label: "var: skip 'been~waiting' across 'sitting here'", cat: "single" },
        TC { q: "my VPN is slow today", exp: &["network:vpn"],
             label: "single: vpn only", cat: "single" },
        TC { q: "reset my password", exp: &["account:reset_password"],
             label: "single: password", cat: "single" },

        // ── PRECISION ──
        TC { q: "what is the status of my ticket", exp: &["tickets:check_ticket_status"],
             label: "prec: status — NOT escalate", cat: "precision" },
        TC { q: "I need to set up MFA", exp: &["account:setup_mfa"],
             label: "prec: mfa only", cat: "precision" },
        TC { q: "need a loaner laptop", exp: &["hardware:request_loaner"],
             label: "prec: loaner only", cat: "precision" },
        TC { q: "my laptop is broken", exp: &["hardware:report_broken"],
             label: "prec: broken only", cat: "precision" },

        // ── CJK SINGLE ──
        TC { q: "我等了一上午了这太荒谬了", exp: &["tickets:escalate_ticket"],
             label: "CJK: waited morning + ridiculous → escalate", cat: "cjk" },
        TC { q: "忘记密码登录不了", exp: &["account:reset_password"],
             label: "CJK: forgot password", cat: "cjk" },
        TC { q: "连不上网WiFi一直断", exp: &["network:wifi"],
             label: "CJK: can't connect + wifi dropping", cat: "cjk" },

        // ── CJK MULTI ──
        TC { q: "VPN连不上网络也很慢", exp: &["network:vpn", "network:wifi"],
             label: "CJK multi: vpn + wifi", cat: "cjk_multi" },
        TC { q: "电脑坏了开不了机非常着急",
             exp: &["hardware:report_broken", "tickets:escalate_ticket"],
             label: "CJK multi: broken + urgent", cat: "cjk_multi" },
        TC { q: "连不上网一直在等到底什么时候能好",
             exp: &["network:wifi", "tickets:escalate_ticket"],
             label: "CJK multi: wifi + waiting", cat: "cjk_multi" },
    ]
}

fn short(id: &str) -> &str { id.split(':').last().unwrap_or(id) }

fn eval(got: &[(String, f32)], exp: &[&str]) -> &'static str {
    let g: HashSet<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
    let e: HashSet<&str> = exp.iter().copied().collect();
    if e.is_empty() && g.is_empty() { "PASS" }
    else if g == e { "PASS" }
    else if !g.is_disjoint(&e) { "PARTIAL" }
    else { "FAIL" }
}

fn main() {
    println!("\n{:=<78}", "");
    println!("  N-gram v5: RE-PASS architecture (exclude + re-score)");
    println!("{:=<78}\n", "");

    let eng = build();

    let T: f32 = 0.3;
    let GAP: f32 = 1.5;
    let BONUS: f32 = 1.5;
    let GATE: f32 = 0.35;  // round N+1 top must be ≥ 35% of original top
    let ROUNDS: usize = 4;

    println!("  gate_ratio={} (round N+1 top ≥ {}% of round 1 top)", GATE, (GATE*100.0) as u32);
    println!();

    let cases = cases();
    let mut stats: HashMap<&str, [usize; 4]> = HashMap::new();  // [total, base, ngram, repass]
    let mut cur = "";

    for tc in &cases {
        if tc.cat != cur { cur = tc.cat; println!("  ─── {} ───", cur.to_uppercase()); }

        let base   = eng.route_baseline(tc.q, T, GAP);
        let ngram  = eng.route_ngram_single(tc.q, T, GAP, BONUS);
        let repass = eng.route_repass(tc.q, T, GAP, BONUS, GATE, ROUNDS);

        let bs = eval(&base, tc.exp);
        let ns = eval(&ngram, tc.exp);
        let rs = eval(&repass, tc.exp);

        let s = stats.entry(tc.cat).or_insert([0; 4]);
        s[0] += 1;
        if bs == "PASS" { s[1] += 1; }
        if ns == "PASS" { s[2] += 1; }
        if rs == "PASS" { s[3] += 1; }

        let marker = if rs == "PASS" && bs != "PASS" { "⬆" }
                     else if bs == "PASS" && rs != "PASS" { "⬇" }
                     else { " " };

        let bi: Vec<&str> = base.iter().map(|(id, _)| short(id)).collect();
        let ni: Vec<&str> = ngram.iter().map(|(id, _)| short(id)).collect();
        let ri: Vec<&str> = repass.iter().map(|(id, _)| short(id)).collect();
        let ei: Vec<&str> = tc.exp.iter().map(|id| short(id)).collect();

        println!("{} [{}]", marker, tc.label);
        println!("    Baseline  [{:<7}] {:?}", bs, bi);
        println!("    Ngram     [{:<7}] {:?}", ns, ni);
        println!("    Re-pass   [{:<7}] {:?}", rs, ri);
        if rs != "PASS" { println!("    Expected           {:?}", ei); }
        println!();
    }

    let total = cases.len();
    let (mut tb, mut tn, mut tr, mut tt) = (0usize, 0, 0, 0);
    println!("{:=<78}", "");
    println!("  {:12} {:>3}  {:>12}  {:>12}  {:>12}", "Category", "N", "Baseline", "Ngram", "Re-pass");
    println!("  {}", "-".repeat(60));
    for cat in &["multi", "single", "precision", "cjk", "cjk_multi"] {
        if let Some(s) = stats.get(cat) {
            let [n, b, g, r] = *s;
            println!("  {:12} {:>3}  {:>3}/{} ({:>3.0}%)  {:>3}/{} ({:>3.0}%)  {:>3}/{} ({:>3.0}%)",
                cat, n, b, n, 100.0*b as f32/n as f32, g, n, 100.0*g as f32/n as f32,
                r, n, 100.0*r as f32/n as f32);
            tt += n; tb += b; tn += g; tr += r;
        }
    }
    println!("  {}", "-".repeat(60));
    println!("  {:12} {:>3}  {:>3}/{} ({:>3.0}%)  {:>3}/{} ({:>3.0}%)  {:>3}/{} ({:>3.0}%)",
        "TOTAL", tt, tb, tt, 100.0*tb as f32/tt as f32, tn, tt, 100.0*tn as f32/tt as f32,
        tr, tt, 100.0*tr as f32/tt as f32);
    println!("\n  Baseline → Re-pass: {:+} cases", tr as i32 - tb as i32);
}
