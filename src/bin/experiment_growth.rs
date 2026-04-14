/// Experiment: Can the system grow to 100% with proper learning?
///
/// The RIGHT learning loop: when a query fails, learn its ACTUAL WORDS
/// for the correct intent (simulating LLM telling us "these query words
/// belong to this intent"). Not generate new phrases — learn the QUERY ITSELF.
///
/// Tests: CA3 (no IDF) vs IDF, across multiple learning cycles,
/// with new unseen queries each cycle.
///
/// Run: cargo run --release --bin experiment_growth
use std::collections::{HashMap, HashSet};

struct Engine {
    weights: HashMap<String, HashMap<String, f32>>,
    use_idf: bool,
}

impl Engine {
    fn new(use_idf: bool) -> Self { Self { weights: HashMap::new(), use_idf } }

    fn learn(&mut self, phrase: &str, intent: &str, rate: f32) {
        for word in tokenize(phrase) {
            let w = self.weights.entry(word).or_default()
                .entry(intent.to_string()).or_insert(0.0);
            *w = (*w + rate * (1.0 - *w)).min(1.0);
        }
    }

    fn score_top(&self, query: &str, ratio: f32) -> Vec<(String, f32)> {
        let n_intents = {
            let mut all: HashSet<&str> = HashSet::new();
            for m in self.weights.values() { for k in m.keys() { all.insert(k); } }
            all.len().max(1) as f32
        };

        let mut scores: HashMap<String, f32> = HashMap::new();
        for word in tokenize(query) {
            if let Some(intents) = self.weights.get(&word) {
                let idf = if self.use_idf {
                    (n_intents / intents.len() as f32).ln().max(0.01)
                } else {
                    1.0
                };
                for (intent, &weight) in intents {
                    *scores.entry(intent.clone()).or_default() += weight * idf;
                }
            }
        }

        let mut sorted: Vec<(String, f32)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if sorted.is_empty() { return sorted; }
        let top = sorted[0].1;
        sorted.into_iter().filter(|(_, s)| *s >= top * ratio).collect()
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let stop: HashSet<&str> = ["a","an","the","is","are","was","were","be","been","being",
        "in","on","at","to","of","for","with","from","by","and","or","but",
        "i","me","my","we","our","you","your","it","its","he","she","they","them",
        "this","that","do","does","did","has","have","had","so","if","can","will",
        "just","also","really","like","very","please","would","could","should"].into_iter().collect();
    text.to_lowercase()
        .replace("n't", " not").replace("'ve", " have").replace("'re", " are")
        .replace("'m", " am").replace("'s", "").replace("'d", " would")
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2 && !stop.contains(w))
        .map(|s| s.to_string())
        .collect()
}

fn short(id: &str) -> &str { id.split(':').last().unwrap_or(id) }

fn eval(engine: &Engine, queries: &[(&str, &str)], ratio: f32) -> (usize, usize, usize, Vec<usize>) {
    let (mut e, mut p, mut f) = (0, 0, 0);
    let mut failed_indices = Vec::new();
    for (i, (query, expected)) in queries.iter().enumerate() {
        let got: HashSet<String> = engine.score_top(query, ratio).iter().map(|(id,_)| id.clone()).collect();
        let exp: HashSet<String> = [expected.to_string()].into_iter().collect();
        if got == exp { e += 1; }
        else if got.contains(&expected.to_string()) { p += 1; failed_indices.push(i); }
        else { f += 1; failed_indices.push(i); }
    }
    (e, p, f, failed_indices)
}

fn main() {
    println!("\n{:=<75}", "");
    println!("  GROWTH TEST: Can the system reach 100% with iterative learning?");
    println!("{:=<75}\n", "");

    // Intents with 5 seeds each
    let seeds: &[(&str, &[&str])] = &[
        ("escalate", &["please escalate my ticket", "this is urgent", "mark as high priority", "need faster response", "critical issue"]),
        ("reset_pw", &["reset my password", "forgot my password", "password expired", "can't log in", "locked out wrong password"]),
        ("wifi", &["wifi not connecting", "no internet connection", "wifi signal weak", "internet keeps dropping", "can't connect to wifi"]),
        ("vpn", &["VPN not working", "can't connect to VPN", "VPN keeps disconnecting", "remote VPN failed", "VPN access from home"]),
        ("broken", &["laptop is broken", "computer won't turn on", "screen cracked", "keyboard stopped working", "device completely dead"]),
        ("setup", &["set up new laptop", "configure new workstation", "initialize new machine", "setup for new computer", "help setting up device"]),
        ("loaner", &["need loaner laptop", "borrow temporary device", "spare computer while repaired", "temporary device", "replacement while mine fixed"]),
        ("mfa", &["set up two-factor authentication", "configure authenticator app", "enable MFA", "two-step verification", "Google Authenticator setup"]),
    ];

    // 40 test queries — progressively harder, simulating different users over time
    // Each wave represents a new batch of users with different vocabulary
    let wave1: Vec<(&str, &str)> = vec![
        ("I forgot my login credentials", "reset_pw"),
        ("my password isn't working", "reset_pw"),
        ("wifi keeps cutting out at my desk", "wifi"),
        ("can't get online today", "wifi"),
        ("VPN drops every few minutes from home", "vpn"),
        ("I'm so frustrated nobody is helping", "escalate"),
        ("been waiting all morning", "escalate"),
        ("my machine is completely dead", "broken"),
        ("screen went black won't come back", "broken"),
        ("just started and my computer has nothing on it", "setup"),
    ];

    let wave2: Vec<(&str, &str)> = vec![
        ("system keeps rejecting my credentials", "reset_pw"),
        ("authentication keeps failing", "reset_pw"),
        ("wireless signal terrible in this building", "wifi"),
        ("no connectivity from the meeting room", "wifi"),
        ("home office connection keeps timing out", "vpn"),
        ("this is completely unacceptable service", "escalate"),
        ("I want to speak to a manager about this", "escalate"),
        ("laptop fell off my desk and shattered", "broken"),
        ("new hire needs their workstation ready", "setup"),
        ("how do I add the security code to my phone", "mfa"),
    ];

    let wave3: Vec<(&str, &str)> = vec![
        ("my login expired overnight", "reset_pw"),
        ("the network is impossibly slow", "wifi"),
        ("remote access tunnel won't establish", "vpn"),
        ("I've lost all patience with this process", "escalate"),
        ("getting absolutely nowhere with support", "escalate"),
        ("hardware making grinding noises then died", "broken"),
        ("onboarding tech setup for the new guy", "setup"),
        ("can I borrow a machine while mine is in the shop", "loaner"),
        ("the extra verification step on my phone isn't working", "mfa"),
        ("need something to work on temporarily", "loaner"),
    ];

    let all_waves = [&wave1[..], &wave2[..], &wave3[..]];
    let all_queries: Vec<(&str, &str)> = wave1.iter().chain(wave2.iter()).chain(wave3.iter()).copied().collect();

    for (label, use_idf) in [("CA3 (no IDF)", false), ("IDF", true)] {
        println!("  ── {} ──\n", label);
        let mut engine = Engine::new(use_idf);

        // Seed
        for (intent, phrases) in seeds {
            for p in *phrases { engine.learn(p, intent, 0.4); }
        }

        let (e, p, f, _) = eval(&engine, &all_queries, 0.5);
        let t = all_queries.len();
        println!("    Seed only:          {}/{} ({:>3.0}%) exact | {} partial | {} fail", e, t, 100.0*e as f32/t as f32, p, f);

        // Iterative learning: test each wave, learn from failures, retest ALL
        let mut total_learned = 0;
        for (wave_num, wave) in all_waves.iter().enumerate() {
            // Test this wave
            let (we, wp, wf, failed) = eval(&engine, wave, 0.5);

            // Learn from failures: teach the EXACT query words for the correct intent
            let mut learned_this_wave = 0;
            for &idx in &failed {
                let (query, intent) = wave[idx];
                // Simulate LLM: "these query words belong to this intent"
                engine.learn(query, intent, 0.3);
                learned_this_wave += 1;
            }
            total_learned += learned_this_wave;

            // Retest ALL queries (not just this wave)
            let (ae, ap, af, _) = eval(&engine, &all_queries, 0.5);
            println!("    Wave {} (+{} learned): {}/{} ({:>3.0}%) exact | {} partial | {} fail  (wave: {}/{} before learn)",
                wave_num + 1, learned_this_wave, ae, t, 100.0*ae as f32/t as f32, ap, af, we, wave.len());
        }

        // Final: learn from ALL remaining failures
        let (_, _, _, remaining) = eval(&engine, &all_queries, 0.5);
        for &idx in &remaining {
            let (query, intent) = all_queries[idx];
            engine.learn(query, intent, 0.3);
        }
        let (fe, fp, ff, still_failing) = eval(&engine, &all_queries, 0.5);
        println!("    Final cleanup:      {}/{} ({:>3.0}%) exact | {} partial | {} fail",
            fe, t, 100.0*fe as f32/t as f32, fp, ff);

        // Show remaining failures
        if !still_failing.is_empty() {
            println!("    STILL FAILING:");
            for &idx in &still_failing {
                let (query, expected) = all_queries[idx];
                let got: Vec<String> = engine.score_top(query, 0.5).iter()
                    .map(|(id, s)| format!("{}={:.2}", short(id), s)).collect();
                println!("      \"{}\" → exp={} got=[{}]", &query[..query.len().min(50)], expected, got.join(", "));
            }
        }
        println!("    Total phrases learned: {}\n", total_learned + remaining.len());
    }
}
