/// Experiment v4: Skip-grams + Decaying OMP + unified Latin/CJK
///
/// Fixes from v3:
/// 1. Skip-grams: "this_*_ridiculous" matches "this is absolutely ridiculous"
///    - Anchor pairs within a window, gaps allowed
///    - Works for both Latin words and CJK characters
/// 2. Decaying OMP threshold: round N uses threshold * (1 + N * decay_rate)
///    - Stops OMP from greedily over-detecting on residual noise
/// 3. CJK: minimum 3-character n-grams (bigrams too generic)
///
/// Run: cargo run --bin experiment_ngram_v4
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

/// Extract CJK characters from text.
fn cjk_chars(text: &str) -> Vec<char> {
    text.chars().filter(|c| tokenizer::is_cjk(*c)).collect()
}

// ── Skip-gram generation ─────────────────────────────────────────────────────

/// Generate skip-grams: pairs of tokens within a window, allowing gaps.
/// For tokens [A, B, C, D] with window=3:
///   Pairs: (A,B), (A,C), (B,C), (B,D), (C,D)
///   Triples: (A,B,C), (A,B,D), (A,C,D), (B,C,D)
///
/// But for efficiency, we generate:
///   - Contiguous n-grams (2,3,4) — exact matches, high confidence
///   - Skip-bigrams: (token_i, token_j) where j-i <= window — gap-tolerant
fn generate_skip_bigrams(tokens: &[String], max_gap: usize) -> Vec<String> {
    let mut result = Vec::new();
    for i in 0..tokens.len() {
        for j in (i + 1)..=(i + 1 + max_gap).min(tokens.len() - 1) {
            if j > i + 1 {
                // Skip-bigram (gap of j-i-1 words)
                result.push(format!("{}~{}", tokens[i], tokens[j]));
            }
        }
    }
    result
}

/// Generate all n-grams for scoring: contiguous + skip-bigrams
fn generate_all_ngrams(tokens: &[String], max_n: usize, max_gap: usize) -> Vec<(String, f32)> {
    let mut result: Vec<(String, f32)> = Vec::new();

    // Contiguous n-grams (high confidence)
    for n in 2..=max_n.min(tokens.len()) {
        let len_bonus = 1.0 + 0.5 * (n as f32 - 1.0);
        for w in tokens.windows(n) {
            result.push((w.join("_"), len_bonus));
        }
    }

    // Skip-bigrams (moderate confidence, gap-tolerant)
    // Use ~ separator to distinguish from contiguous
    for sg in generate_skip_bigrams(tokens, max_gap) {
        result.push((sg, 1.2));  // lower bonus than contiguous bigrams (1.5)
    }

    result
}

/// CJK: character n-grams (contiguous only, min 3 chars) + skip-bigrams of chars
fn generate_cjk_ngrams(text: &str, max_n: usize, max_gap: usize) -> Vec<(String, f32)> {
    let chars = cjk_chars(text);
    let mut result: Vec<(String, f32)> = Vec::new();

    // Contiguous character n-grams (3+ chars for CJK — bigrams too generic)
    for n in 3..=max_n.min(chars.len()) {
        let len_bonus = 1.0 + 0.5 * (n as f32 - 1.0);
        for w in chars.windows(n) {
            result.push((w.iter().collect::<String>(), len_bonus));
        }
    }

    // Also keep character bigrams for exact learned patterns
    for w in chars.windows(2) {
        result.push((w.iter().collect::<String>(), 1.3));
    }

    // Skip-bigrams of CJK characters
    let char_strs: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
    for sg in generate_skip_bigrams(&char_strs, max_gap) {
        result.push((sg, 1.1));
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

    /// Learn a phrase as contiguous n-grams + skip-bigrams.
    fn learn_ngrams(&mut self, phrase: &str, intent: &str, max_n: usize, max_gap: usize) {
        let has_cjk = phrase.chars().any(tokenizer::is_cjk);
        if has_cjk {
            let chars = cjk_chars(phrase);
            // Contiguous 3+ char n-grams
            for n in 2..=max_n.min(chars.len()) {
                for w in chars.windows(n) {
                    self.learn_pattern(&w.iter().collect::<String>(), intent);
                }
            }
            // Skip-bigrams of characters
            let char_strs: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
            for sg in generate_skip_bigrams(&char_strs, max_gap) {
                self.learn_pattern(&sg, intent);
            }
        } else {
            let tokens = tokenize_full(phrase);
            // Contiguous n-grams
            for n in 2..=max_n.min(tokens.len()) {
                for w in tokens.windows(n) {
                    self.learn_pattern(&w.join("_"), intent);
                }
            }
            // Skip-bigrams
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

    /// Score with token weights (for OMP subtraction)
    fn score_raw(&self, query: &str, threshold: f32, ngram_bonus: f32,
                 token_weights: &HashMap<String, f32>) -> Vec<(String, f32)> {
        let n = self.total_intents();
        let mut scores: HashMap<String, f32> = HashMap::new();

        // Unigrams
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

        // N-grams + skip-grams
        let has_cjk = query.chars().any(tokenizer::is_cjk);
        let ngrams = if has_cjk {
            generate_cjk_ngrams(query, 5, 2)
        } else {
            let full = tokenize_full(query);
            generate_all_ngrams(&full, 4, 2)  // max_n=4, max_gap=2
        };

        for (key, len_bonus) in &ngrams {
            // Average token weight for constituent words
            let avg_tw = if has_cjk { 1.0 } else {
                let words: Vec<&str> = key.split(|c| c == '_' || c == '~').collect();
                let sum: f32 = words.iter()
                    .map(|w| token_weights.get(*w).copied().unwrap_or(1.0))
                    .sum();
                sum / words.len().max(1) as f32
            };
            if avg_tw <= 0.01 { continue; }

            if let Some(entries) = self.phrase_intent.get(key.as_str()) {
                let idf = (n / entries.len() as f32).ln().max(0.0);
                for (intent, weight) in entries {
                    *scores.entry(intent.clone()).or_default() +=
                        weight * idf * ngram_bonus * len_bonus * avg_tw;
                }
            }
        }

        let mut sorted: Vec<(String, f32)> = scores.into_iter()
            .filter(|(_, s)| *s >= threshold)
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// OMP + N-gram with DECAYING threshold per round.
    fn route_omp(&self, query: &str, base_threshold: f32, _gap: f32,
                  ngram_bonus: f32, alpha: f32, max_rounds: usize,
                  decay: f32) -> Vec<(String, f32)> {
        let mut token_weights: HashMap<String, f32> = HashMap::new();
        for t in tokenize_asv(query) { token_weights.entry(t).or_insert(1.0); }
        for t in tokenize_full(query) { token_weights.entry(t).or_insert(1.0); }

        let mut confirmed: Vec<(String, f32)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for round in 0..max_rounds {
            // Threshold increases each round — later intents need stronger evidence
            let round_threshold = base_threshold * (1.0 + round as f32 * decay);

            let all = self.score_raw(query, round_threshold, ngram_bonus, &token_weights);

            let top = all.iter().find(|(id, _)| !seen.contains(id));
            let (top_id, top_score) = match top {
                Some(t) => (t.0.clone(), t.1),
                None => break,
            };

            if top_score < round_threshold { break; }

            confirmed.push((top_id.clone(), top_score));
            seen.insert(top_id.clone());

            // OMP subtraction
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
        }

        confirmed
    }

    fn route_baseline(&self, query: &str, threshold: f32, gap: f32) -> Vec<(String, f32)> {
        let tw = HashMap::new();
        let all = self.score_raw(query, threshold, 0.0, &tw);
        if all.is_empty() { return all; }
        let top = all[0].1;
        all.into_iter().filter(|(_, s)| top - *s <= gap).collect()
    }
}

// ── Corpus ────────────────────────────────────────────────────────────────────

fn build() -> Engine {
    let mut e = Engine::new();

    // Latin seed phrases (unigrams)
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
        for p in *phrases { e.learn_uni(p, intent); }
    }

    // CJK seed (unigrams)
    let cjk_seeds: &[(&str, &[&str])] = &[
        ("network:vpn", &["VPN连不上", "连接VPN失败"]),
        ("network:wifi", &["网络连不上", "上不了网", "WiFi断了"]),
        ("account:reset_password", &["重置密码", "忘记密码", "登录不了"]),
        ("tickets:escalate_ticket", &["加急处理", "太荒谬了"]),
        ("hardware:report_broken", &["电脑坏了", "开不了机"]),
        ("hardware:request_loaner", &["借一台临时电脑", "需要备用机"]),
    ];
    for (intent, phrases) in cjk_seeds {
        for p in *phrases { e.learn_uni(p, intent); }
    }

    // Latin n-gram + skip-gram patterns (max_gap=2)
    let ngrams: &[(&str, &[&str])] = &[
        ("tickets:escalate_ticket", &[
            "been waiting", "been waiting all morning", "been waiting for hours",
            "have been waiting", "this is ridiculous", "this is unacceptable",
            "so frustrated", "extremely frustrated", "very frustrated",
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
        ("hardware:report_broken", &[
            "completely dead", "is completely dead", "won not turn on",
        ]),
        ("hardware:request_loaner", &[
            "need a spare", "while mine is being fixed",
        ]),
    ];
    for (intent, phrases) in ngrams {
        for p in *phrases { e.learn_ngrams(p, intent, 4, 2); }
    }

    // CJK n-gram patterns
    let cjk_ngrams: &[(&str, &[&str])] = &[
        ("tickets:escalate_ticket", &[
            "等了一上午", "一直在等", "太荒谬了", "受不了了",
            "到底什么时候", "等了很久", "非常着急",
        ]),
        ("account:reset_password", &[
            "登不进去", "进不去", "登录失败",
        ]),
        ("network:wifi", &[
            "连不上网", "上不了网", "网络一直断",
        ]),
    ];
    for (intent, phrases) in cjk_ngrams {
        for p in *phrases { e.learn_ngrams(p, intent, 4, 2); }
    }

    println!("  {} unigram, {} n-gram/skip-gram patterns", e.word_intent.len(), e.phrase_intent.len());
    e
}

// ── Test cases ────────────────────────────────────────────────────────────────

struct TC { q: &'static str, exp: &'static [&'static str], label: &'static str, cat: &'static str }

fn cases() -> Vec<TC> {
    vec![
        // ── SKIP-GRAM VARIATION (the critical fix test) ──
        TC { q: "this is absolutely ridiculous",
             exp: &["tickets:escalate_ticket"],
             label: "skip: 'this~ridiculous' fires across 'absolutely'", cat: "skipgram" },
        TC { q: "I'm so incredibly frustrated right now",
             exp: &["tickets:escalate_ticket"],
             label: "skip: 'so~frustrated' fires across 'incredibly'", cat: "skipgram" },
        TC { q: "this is completely unacceptable service",
             exp: &["tickets:escalate_ticket"],
             label: "skip: 'this~unacceptable' fires across 'completely'", cat: "skipgram" },
        TC { q: "I have been waiting way too long",
             exp: &["tickets:escalate_ticket"],
             label: "skip: 'have~waiting' or 'been_waiting' contiguous", cat: "skipgram" },
        TC { q: "been sitting here waiting since morning",
             exp: &["tickets:escalate_ticket"],
             label: "skip: 'been~waiting' fires across 'sitting here'", cat: "skipgram" },

        // ── MULTI-INTENT (OMP with decay) ──
        TC { q: "I've been waiting all morning my internet is down and VPN won't connect",
             exp: &["tickets:escalate_ticket", "network:wifi", "network:vpn"],
             label: "multi: escalate + wifi + vpn", cat: "multi" },
        TC { q: "this is ridiculous I need a spare laptop mine is completely dead",
             exp: &["tickets:escalate_ticket", "hardware:request_loaner", "hardware:report_broken"],
             label: "multi: escalate + loaner + broken", cat: "multi" },
        TC { q: "I'm so frustrated the VPN keeps dropping and I need a loaner",
             exp: &["tickets:escalate_ticket", "network:vpn", "hardware:request_loaner"],
             label: "multi: frustrated + vpn + loaner", cat: "multi" },

        // ── SINGLE + PRECISION ──
        TC { q: "my VPN is slow today", exp: &["network:vpn"],
             label: "prec: vpn only", cat: "precision" },
        TC { q: "what is the status of my ticket",
             exp: &["tickets:check_ticket_status"],
             label: "prec: status — NOT escalate", cat: "precision" },
        TC { q: "reset my password", exp: &["account:reset_password"],
             label: "prec: password only", cat: "precision" },
        TC { q: "I need to set up MFA", exp: &["account:setup_mfa"],
             label: "prec: mfa only", cat: "precision" },

        // ── CJK SINGLE ──
        TC { q: "我等了一上午了这太荒谬了",
             exp: &["tickets:escalate_ticket"],
             label: "CJK single: waited all morning + ridiculous → escalate", cat: "cjk" },
        TC { q: "忘记密码登录不了",
             exp: &["account:reset_password"],
             label: "CJK single: forgot password + can't login", cat: "cjk" },
        TC { q: "连不上网WiFi一直断",
             exp: &["network:wifi"],
             label: "CJK single: can't connect + wifi keeps dropping", cat: "cjk" },

        // ── CJK MULTI ──
        TC { q: "VPN连不上网络也很慢",
             exp: &["network:vpn", "network:wifi"],
             label: "CJK multi: vpn + wifi", cat: "cjk_multi" },
        TC { q: "电脑坏了开不了机非常着急",
             exp: &["hardware:report_broken", "tickets:escalate_ticket"],
             label: "CJK multi: broken + urgent → escalate", cat: "cjk_multi" },
        TC { q: "连不上网一直在等到底什么时候能好",
             exp: &["network:wifi", "tickets:escalate_ticket"],
             label: "CJK multi: wifi + been waiting → escalate", cat: "cjk_multi" },

        // ── CJK VARIATION ──
        TC { q: "网上不去了怎么办",
             exp: &["network:wifi"],
             label: "CJK var: 上不去 (slightly different from 上不了网)", cat: "cjk_var" },
        TC { q: "都等了快两个小时了",
             exp: &["tickets:escalate_ticket"],
             label: "CJK var: waited almost 2 hours (新表达)", cat: "cjk_var" },
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
    println!("  N-gram v4: Skip-grams + Decaying OMP + CJK trigrams");
    println!("{:=<78}\n", "");

    let eng = build();
    let cases = cases();

    let T: f32 = 0.3;
    let GAP: f32 = 1.5;
    let BONUS: f32 = 1.5;
    let ALPHA: f32 = 0.85;
    let ROUNDS: usize = 4;
    let DECAY: f32 = 0.6;  // round 1: 0.3, round 2: 0.48, round 3: 0.66, round 4: 0.84

    println!("  OMP: alpha={} decay={} rounds={}", ALPHA, DECAY, ROUNDS);
    println!("  Thresholds by round: {:.2}, {:.2}, {:.2}, {:.2}\n",
        T, T*(1.0+DECAY), T*(1.0+2.0*DECAY), T*(1.0+3.0*DECAY));

    let mut stats: HashMap<&str, (usize, usize, usize)> = HashMap::new();
    let mut cur_cat = "";

    for tc in &cases {
        if tc.cat != cur_cat {
            cur_cat = tc.cat;
            println!("  ─── {} ───", cur_cat.to_uppercase());
        }

        let base = eng.route_baseline(tc.q, T, GAP);
        let omp  = eng.route_omp(tc.q, T, GAP, BONUS, ALPHA, ROUNDS, DECAY);

        let bs = eval(&base, tc.exp);
        let os = eval(&omp, tc.exp);

        let s = stats.entry(tc.cat).or_insert((0, 0, 0));
        s.0 += 1;
        if bs == "PASS" { s.1 += 1; }
        if os == "PASS" { s.2 += 1; }

        let marker = if os == "PASS" && bs != "PASS" { "⬆" }
                     else if bs == "PASS" && os != "PASS" { "⬇" }
                     else { " " };

        let bi: Vec<&str> = base.iter().map(|(id, _)| short(id)).collect();
        let oi: Vec<&str> = omp.iter().map(|(id, _)| short(id)).collect();
        let ei: Vec<&str> = tc.exp.iter().map(|id| short(id)).collect();

        println!("{} [{}]", marker, tc.label);
        println!("    Baseline  [{:<7}] {:?}", bs, bi);
        println!("    OMP+Skip  [{:<7}] {:?}", os, oi);
        if os != "PASS" || bs != "PASS" {
            println!("    Expected           {:?}", ei);
        }
        println!();
    }

    // Summary
    let total = cases.len();
    let (mut tb, mut to, mut tn) = (0usize, 0, 0);
    println!("{:=<78}", "");
    println!("  {:12} {:>4}  {:>12}  {:>12}", "Category", "N", "Baseline", "OMP+Skip");
    println!("  {}", "-".repeat(50));
    for cat in &["skipgram", "multi", "precision", "cjk", "cjk_multi", "cjk_var"] {
        if let Some(&(n, bp, op)) = stats.get(cat) {
            println!("  {:12} {:>4}  {:>3}/{} ({:>3.0}%)  {:>3}/{} ({:>3.0}%)",
                cat, n, bp, n, 100.0*bp as f32/n as f32, op, n, 100.0*op as f32/n as f32);
            tn += n; tb += bp; to += op;
        }
    }
    println!("  {}", "-".repeat(50));
    println!("  {:12} {:>4}  {:>3}/{} ({:>3.0}%)  {:>3}/{} ({:>3.0}%)",
        "TOTAL", tn, tb, tn, 100.0*tb as f32/tn as f32, to, tn, 100.0*to as f32/tn as f32);
    println!("\n  Δ = {:+} cases", to as i32 - tb as i32);
}
