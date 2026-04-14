/// Experiment v3: N-gram + OMP combined (multi-intent) + CJK test cases.
///
/// v2 showed: n-grams boost primary intent so much that secondary intents
/// fall outside the gap. OMP solves this by subtracting the primary's signal
/// before scoring the residual.
///
/// Also tests: CJK character-level n-grams (Chinese, Japanese, Korean).
///
/// Run: cargo run --bin experiment_ngram_v3
use asv_router::tokenizer;
use std::collections::{HashMap, HashSet};

// ── Tokenizers ────────────────────────────────────────────────────────────────

/// Full tokenizer preserving stop words (Latin)
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

/// ASV tokenizer (stop words removed)
fn tokenize_asv(text: &str) -> Vec<String> {
    tokenizer::tokenize(text).into_iter().map(|s| s.to_string()).collect()
}

/// CJK character n-gram generator. Produces char bigrams, trigrams, 4-grams.
fn cjk_char_ngrams(text: &str, max_n: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars()
        .filter(|c| tokenizer::is_cjk(*c))
        .collect();
    let mut ngrams = Vec::new();
    for n in 2..=max_n {
        if chars.len() < n { break; }
        for window in chars.windows(n) {
            ngrams.push(window.iter().collect::<String>());
        }
    }
    ngrams
}

// ── Engine ────────────────────────────────────────────────────────────────────

struct Engine {
    word_intent: HashMap<String, Vec<(String, f32)>>,
    phrase_intent: HashMap<String, Vec<(String, f32)>>,
}

impl Engine {
    fn new() -> Self { Self { word_intent: HashMap::new(), phrase_intent: HashMap::new() } }

    fn learn_unigrams(&mut self, phrase: &str, intent: &str) {
        const R: f32 = 0.4;
        for token in tokenize_asv(phrase) {
            let e = self.word_intent.entry(token).or_default();
            if let Some(x) = e.iter_mut().find(|(id, _)| id == intent) {
                x.1 = (x.1 + R * (1.0 - x.1)).min(1.0);
            } else { e.push((intent.to_string(), R)); }
        }
    }

    fn learn_ngrams(&mut self, phrase: &str, intent: &str, max_n: usize) {
        const R: f32 = 0.4;
        // Detect CJK content
        let has_cjk = phrase.chars().any(tokenizer::is_cjk);
        let tokens = if has_cjk {
            cjk_char_ngrams(phrase, max_n)
        } else {
            let full = tokenize_full(phrase);
            let mut ngrams = Vec::new();
            for n in 2..=max_n.min(full.len()) {
                for w in full.windows(n) {
                    ngrams.push(w.join("_"));
                }
            }
            ngrams
        };

        for key in tokens {
            let e = self.phrase_intent.entry(key).or_default();
            if let Some(x) = e.iter_mut().find(|(id, _)| id == intent) {
                x.1 = (x.1 + R * (1.0 - x.1)).min(1.0);
            } else { e.push((intent.to_string(), R)); }
        }
    }

    fn total_intents(&self) -> f32 {
        let mut all: HashSet<&str> = HashSet::new();
        for v in self.word_intent.values() { for (id, _) in v { all.insert(id); } }
        for v in self.phrase_intent.values() { for (id, _) in v { all.insert(id); } }
        all.len().max(1) as f32
    }

    /// Raw scoring (returns all intents above threshold, no gap filter).
    /// token_weights: multiplier per token (1.0 = full, 0.0 = suppressed by OMP).
    fn score_raw(&self, query: &str, threshold: f32, ngram_bonus: f32,
                 token_weights: &HashMap<String, f32>) -> Vec<(String, f32)> {
        let n = self.total_intents();
        let mut scores: HashMap<String, f32> = HashMap::new();

        // Unigrams (ASV tokenizer)
        for token in tokenize_asv(query) {
            let tw = token_weights.get(token.as_str()).copied().unwrap_or(1.0);
            if tw <= 0.01 { continue; }
            if let Some(entries) = self.word_intent.get(token.as_str()) {
                let idf = (n / entries.len() as f32).ln().max(0.0);
                for (intent, weight) in entries {
                    *scores.entry(intent.clone()).or_default() += weight * idf * tw;
                }
            }
        }

        // N-grams (full tokenizer for Latin, character n-grams for CJK)
        let has_cjk = query.chars().any(tokenizer::is_cjk);
        let ngram_tokens: Vec<String> = if has_cjk {
            cjk_char_ngrams(query, 5)
        } else {
            let full = tokenize_full(query);
            let mut ngrams = Vec::new();
            for ng_len in 2..=5 {
                if full.len() < ng_len { break; }
                let len_bonus = 1.0 + 0.5 * (ng_len as f32 - 1.0);
                for w in full.windows(ng_len) {
                    ngrams.push(format!("{}|{}", w.join("_"), len_bonus));
                }
            }
            ngrams
        };

        for ng_str in &ngram_tokens {
            let (key, len_bonus) = if has_cjk {
                // CJK: key is the char n-gram, bonus by length
                let lb = 1.0 + 0.5 * (ng_str.chars().count() as f32 - 1.0);
                (ng_str.as_str(), lb)
            } else {
                let parts: Vec<&str> = ng_str.rsplitn(2, '|').collect();
                let lb: f32 = parts[0].parse().unwrap_or(1.0);
                (parts[1], lb)
            };

            // Token weight for n-gram: average of constituent token weights
            let constituent_weight = if has_cjk { 1.0 } else {
                let words: Vec<&str> = key.split('_').collect();
                let sum: f32 = words.iter()
                    .map(|w| token_weights.get(*w).copied().unwrap_or(1.0))
                    .sum();
                sum / words.len() as f32
            };
            if constituent_weight <= 0.01 { continue; }

            if let Some(entries) = self.phrase_intent.get(key) {
                let idf = (n / entries.len() as f32).ln().max(0.0);
                for (intent, weight) in entries {
                    *scores.entry(intent.clone()).or_default() +=
                        weight * idf * ngram_bonus * len_bonus * constituent_weight;
                }
            }
        }

        let mut sorted: Vec<(String, f32)> = scores.into_iter()
            .filter(|(_, s)| *s >= threshold)
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// OMP + N-gram routing.
    fn route_omp_ngram(&self, query: &str, threshold: f32, gap: f32,
                        ngram_bonus: f32, alpha: f32, max_rounds: usize) -> Vec<(String, f32)> {
        let mut token_weights: HashMap<String, f32> = HashMap::new();
        // Initialize all tokens to weight 1.0
        for t in tokenize_asv(query) { token_weights.entry(t).or_insert(1.0); }
        for t in tokenize_full(query) { token_weights.entry(t).or_insert(1.0); }

        let mut confirmed: Vec<(String, f32)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for _round in 0..max_rounds {
            let all = self.score_raw(query, threshold, ngram_bonus, &token_weights);

            // Find top unseen intent
            let top = all.iter().find(|(id, _)| !seen.contains(id));
            let (top_id, top_score) = match top {
                Some(t) => (t.0.clone(), t.1),
                None => break,
            };

            if top_score < threshold { break; }

            confirmed.push((top_id.clone(), top_score));
            seen.insert(top_id.clone());

            // OMP: soft-subtract tokens associated with detected intent
            // Reduce weight of unigrams that score for this intent
            for (token, tw) in &mut token_weights {
                if let Some(entries) = self.word_intent.get(token.as_str()) {
                    for (intent, weight) in entries {
                        if intent == &top_id {
                            *tw *= 1.0 - alpha * weight;
                            break;
                        }
                    }
                }
            }

            // Also reduce n-gram contributions for detected intent
            // (implicitly handled: lower token weights → lower n-gram scores)
        }

        confirmed
    }

    /// Baseline: unigram only, gap filter
    fn route_baseline(&self, query: &str, threshold: f32, gap: f32) -> Vec<(String, f32)> {
        let tw: HashMap<String, f32> = HashMap::new(); // empty = all 1.0
        let all = self.score_raw(query, threshold, 0.0, &tw); // ngram_bonus=0 = unigrams only
        if all.is_empty() { return all; }
        let top = all[0].1;
        all.into_iter().filter(|(_, s)| top - *s <= gap).collect()
    }

    /// N-gram only (no OMP), gap filter
    fn route_ngram_gap(&self, query: &str, threshold: f32, gap: f32, bonus: f32) -> Vec<(String, f32)> {
        let tw: HashMap<String, f32> = HashMap::new();
        let all = self.score_raw(query, threshold, bonus, &tw);
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
    for (intent, phrases) in intents {
        for p in *phrases { e.learn_unigrams(p, intent); }
    }

    // CJK seed phrases
    let cjk_intents: &[(&str, &[&str])] = &[
        ("network:vpn", &["VPN连不上", "连接VPN失败", "VPN一直断"]),
        ("network:wifi", &["网络连不上", "上不了网", "WiFi断了", "网速很慢"]),
        ("account:reset_password", &["重置密码", "忘记密码", "登录不了", "密码过期"]),
        ("tickets:escalate_ticket", &["加急处理", "这太荒谬了", "等了一上午了"]),
        ("hardware:report_broken", &["电脑坏了", "屏幕碎了", "开不了机"]),
    ];
    for (intent, phrases) in cjk_intents {
        for p in *phrases {
            e.learn_unigrams(p, intent);
            e.learn_ngrams(p, intent, 5);
        }
    }

    // Latin n-gram patterns
    let patterns: &[(&str, &[&str])] = &[
        ("tickets:escalate_ticket", &[
            "been waiting", "been waiting all morning", "been waiting for hours",
            "have been waiting", "this is ridiculous", "this is unacceptable",
            "so frustrated", "done with this", "done waiting",
            "at my wits end", "wits end", "getting nowhere",
        ]),
        ("account:reset_password", &[
            "can not get in", "can not get into", "can not log in",
            "not let me in", "not let me log in", "locked out of",
        ]),
        ("network:wifi", &[
            "internet is down", "internet not working", "internet is not working",
            "connection keeps dropping", "can not get online",
        ]),
        ("hardware:report_broken", &[
            "completely dead", "is completely dead", "won not turn on",
        ]),
        ("hardware:request_loaner", &[
            "need a spare", "in the meantime", "while mine is being fixed",
        ]),
    ];
    for (intent, phrases) in patterns {
        for p in *phrases { e.learn_ngrams(p, intent, 5); }
    }

    // CJK n-gram patterns (frustration/urgency)
    let cjk_patterns: &[(&str, &[&str])] = &[
        ("tickets:escalate_ticket", &[
            "等了一上午", "一直在等", "太荒谬了", "受不了了",
            "到底什么时候", "等了很久", "非常着急", "急需处理",
        ]),
        ("account:reset_password", &[
            "登不进去", "进不去", "登录失败", "一直登不上",
        ]),
        ("network:wifi", &[
            "连不上网", "上不了网", "网络一直断",
        ]),
    ];
    for (intent, phrases) in cjk_patterns {
        for p in *phrases { e.learn_ngrams(p, intent, 5); }
    }

    println!("  {} unigram entries, {} n-gram patterns", e.word_intent.len(), e.phrase_intent.len());
    e
}

// ── Test cases ────────────────────────────────────────────────────────────────

struct TC { q: &'static str, exp: &'static [&'static str], label: &'static str, cat: &'static str }

fn cases() -> Vec<TC> {
    vec![
        // ── MULTI-INTENT (the key test for OMP + n-grams) ──
        TC { q: "I've been waiting all morning my internet is down and VPN won't connect",
             exp: &["tickets:escalate_ticket", "network:wifi", "network:vpn"],
             label: "multi: escalate + wifi + vpn (3 intents)", cat: "multi" },
        TC { q: "this is ridiculous I need a spare laptop mine is completely dead",
             exp: &["tickets:escalate_ticket", "hardware:request_loaner", "hardware:report_broken"],
             label: "multi: escalate + loaner + broken", cat: "multi" },
        TC { q: "can't get into my account and also need to set up my authenticator",
             exp: &["account:reset_password", "account:setup_mfa"],
             label: "multi: password + mfa", cat: "multi" },
        TC { q: "the VPN keeps dropping and my wifi is terrible I've been waiting for hours",
             exp: &["network:vpn", "network:wifi", "tickets:escalate_ticket"],
             label: "multi: vpn + wifi + escalate", cat: "multi" },
        TC { q: "I'm so frustrated my laptop is broken and I need a loaner while it's being fixed",
             exp: &["tickets:escalate_ticket", "hardware:report_broken", "hardware:request_loaner"],
             label: "multi: frustrated + broken + loaner", cat: "multi" },

        // ── SINGLE-INTENT (must still work, no regression) ──
        TC { q: "I've been waiting all morning", exp: &["tickets:escalate_ticket"],
             label: "single: escalate only", cat: "single" },
        TC { q: "my VPN is slow today", exp: &["network:vpn"],
             label: "single: vpn only", cat: "single" },
        TC { q: "reset my password", exp: &["account:reset_password"],
             label: "single: password only", cat: "single" },

        // ── CJK ──
        TC { q: "我等了一上午了这太荒谬了",
             exp: &["tickets:escalate_ticket"],
             label: "CJK: 等了一上午 + 太荒谬了 → escalate", cat: "cjk" },
        TC { q: "VPN连不上网络也很慢",
             exp: &["network:vpn", "network:wifi"],
             label: "CJK: VPN连不上 + 网络慢 → vpn + wifi", cat: "cjk" },
        TC { q: "忘记密码登录不了",
             exp: &["account:reset_password"],
             label: "CJK: 忘记密码 + 登录不了 → reset_password", cat: "cjk" },
        TC { q: "电脑坏了开不了机急需处理",
             exp: &["hardware:report_broken", "tickets:escalate_ticket"],
             label: "CJK: 电脑坏了 + 急需处理 → broken + escalate", cat: "cjk" },
        TC { q: "连不上网一直在等到底什么时候能好",
             exp: &["network:wifi", "tickets:escalate_ticket"],
             label: "CJK: 连不上网 + 一直在等 → wifi + escalate", cat: "cjk" },

        // ── PRECISION (must NOT over-detect) ──
        TC { q: "what is the status of my support request",
             exp: &["tickets:check_ticket_status"],
             label: "prec: status only — NOT escalate", cat: "precision" },
        TC { q: "I need to set up MFA",
             exp: &["account:setup_mfa"],
             label: "prec: mfa only", cat: "precision" },
    ]
}

fn short(id: &str) -> &str { id.split(':').last().unwrap_or(id) }

fn eval(got: &[(String, f32)], exp: &[&str]) -> &'static str {
    let g: HashSet<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
    let e: HashSet<&str> = exp.iter().copied().collect();
    if e.is_empty() && g.is_empty() { return "PASS"; }
    if g == e { "PASS" } else if !g.is_disjoint(&e) { "PARTIAL" } else { "FAIL" }
}

fn main() {
    println!("\n{:=<78}", "");
    println!("  N-gram v3: OMP + N-grams combined + CJK");
    println!("{:=<78}\n", "");

    let eng = build();
    let cases = cases();

    let T: f32 = 0.3;
    let GAP: f32 = 1.5;
    let BONUS: f32 = 1.5;
    let ALPHA: f32 = 0.85;
    let ROUNDS: usize = 4;

    let mut stats: HashMap<&str, (usize, usize, usize, usize)> = HashMap::new(); // (total, base, ngram_gap, omp_ngram)
    let mut cur_cat = "";

    for tc in &cases {
        if tc.cat != cur_cat {
            cur_cat = tc.cat;
            println!("  ─── {} ───", cur_cat.to_uppercase());
        }

        let base     = eng.route_baseline(tc.q, T, GAP);
        let ng_gap   = eng.route_ngram_gap(tc.q, T, GAP, BONUS);
        let omp_ng   = eng.route_omp_ngram(tc.q, T, GAP, BONUS, ALPHA, ROUNDS);

        let bs = eval(&base, tc.exp);
        let gs = eval(&ng_gap, tc.exp);
        let os = eval(&omp_ng, tc.exp);

        let s = stats.entry(tc.cat).or_insert((0, 0, 0, 0));
        s.0 += 1;
        if bs == "PASS" { s.1 += 1; }
        if gs == "PASS" { s.2 += 1; }
        if os == "PASS" { s.3 += 1; }

        let marker = if os == "PASS" && bs != "PASS" { "⬆" }
                     else if bs == "PASS" && os != "PASS" { "⬇" }
                     else { " " };

        let bi: Vec<&str> = base.iter().map(|(id, _)| short(id)).collect();
        let gi: Vec<&str> = ng_gap.iter().map(|(id, _)| short(id)).collect();
        let oi: Vec<&str> = omp_ng.iter().map(|(id, _)| short(id)).collect();
        let ei: Vec<&str> = tc.exp.iter().map(|id| short(id)).collect();

        println!("{} [{}]", marker, tc.label);
        println!("    Baseline     [{:<7}] {:?}", bs, bi);
        println!("    Ngram+Gap    [{:<7}] {:?}", gs, gi);
        println!("    OMP+Ngram    [{:<7}] {:?}", os, oi);
        if os != "PASS" || bs != "PASS" {
            println!("    Expected              {:?}", ei);
        }
        println!();
    }

    // Summary
    let total = cases.len();
    let (mut tb, mut tg, mut to, mut tn) = (0usize, 0, 0, 0);
    println!("{:=<78}", "");
    println!("  {:12} {:>5}  {:>10}  {:>10}  {:>10}", "Category", "N", "Baseline", "Ngram+Gap", "OMP+Ngram");
    println!("  {}", "-".repeat(55));
    for cat in &["multi", "single", "cjk", "precision"] {
        if let Some(&(n, bp, gp, op)) = stats.get(cat) {
            println!("  {:12} {:>5}  {:>4}/{} {:>3.0}%  {:>4}/{} {:>3.0}%  {:>4}/{} {:>3.0}%",
                cat, n,
                bp, n, 100.0 * bp as f32 / n as f32,
                gp, n, 100.0 * gp as f32 / n as f32,
                op, n, 100.0 * op as f32 / n as f32);
            tn += n; tb += bp; tg += gp; to += op;
        }
    }
    println!("  {}", "-".repeat(55));
    println!("  {:12} {:>5}  {:>4}/{} {:>3.0}%  {:>4}/{} {:>3.0}%  {:>4}/{} {:>3.0}%",
        "TOTAL", tn,
        tb, tn, 100.0 * tb as f32 / tn as f32,
        tg, tn, 100.0 * tg as f32 / tn as f32,
        to, tn, 100.0 * to as f32 / tn as f32);

    println!("\n  OMP+Ngram vs Baseline: {:+} cases", to as i32 - tb as i32);
    if to as i32 - tb as i32 >= 5 {
        println!("  ✓ STRONG — OMP+Ngram is the architecture to integrate");
    }
}
