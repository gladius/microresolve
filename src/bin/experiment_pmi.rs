/// Experiment: PMI-based word similarity for intent scoring.
///
/// Hypothesis: If we generate enough diverse phrases per intent (via LLM simulation),
/// words that co-occur in the same intent's phrases have high PMI.
/// At query time, UNSEEN words can be "expanded" via PMI to known words.
///
/// Example: "frustrated" co-occurs with "waiting" in escalation phrases.
/// PMI("frustrated", "waiting") is high. When query has "frustrated",
/// we boost intents that "waiting" scores for.
///
/// This is LLM distillation: LLM knowledge → diverse phrases → PMI → scoring.
/// No pre-trained model. Gets BETTER with more phrases (unlike IDF).
///
/// Run: cargo run --bin experiment_pmi
use std::collections::{HashMap, HashSet};

// ── PMI Engine ────────────────────────────────────────────────────────────────

struct PmiEngine {
    /// word → intent → weight (for direct IDF scoring)
    word_intent: HashMap<String, HashMap<String, f32>>,
    /// (word1, word2) → PMI score (symmetric)
    pmi: HashMap<(String, String), f32>,
    /// word → total count across all phrases
    word_count: HashMap<String, f32>,
    /// total phrase count
    total_phrases: f32,
    /// word pair → co-occurrence count
    pair_count: HashMap<(String, String), f32>,
}

impl PmiEngine {
    fn new() -> Self {
        Self {
            word_intent: HashMap::new(),
            pmi: HashMap::new(),
            word_count: HashMap::new(),
            total_phrases: 0.0,
            pair_count: HashMap::new(),
        }
    }

    /// Learn a phrase: updates word→intent weights AND co-occurrence counts.
    fn learn(&mut self, phrase: &str, intent: &str) {
        let words: Vec<String> = tokenize(phrase);
        self.total_phrases += 1.0;

        // Word→intent IDF weight
        for w in &words {
            *self.word_intent.entry(w.clone()).or_default()
                .entry(intent.to_string()).or_insert(0.0) += 1.0;
            *self.word_count.entry(w.clone()).or_insert(0.0) += 1.0;
        }

        // Co-occurrence: all word pairs within the phrase
        let unique: Vec<&String> = words.iter().collect::<HashSet<_>>()
            .into_iter().collect();
        for i in 0..unique.len() {
            for j in (i+1)..unique.len() {
                let (a, b) = if unique[i] < unique[j] {
                    (unique[i].clone(), unique[j].clone())
                } else {
                    (unique[j].clone(), unique[i].clone())
                };
                *self.pair_count.entry((a, b)).or_insert(0.0) += 1.0;
            }
        }
    }

    /// Compute PMI for all word pairs after all phrases are learned.
    fn build_pmi(&mut self) {
        let n = self.total_phrases;
        if n < 2.0 { return; }

        self.pmi.clear();
        for ((w1, w2), &count) in &self.pair_count {
            let p_w1 = self.word_count.get(w1).copied().unwrap_or(0.0) / n;
            let p_w2 = self.word_count.get(w2).copied().unwrap_or(0.0) / n;
            let p_joint = count / n;
            if p_w1 > 0.0 && p_w2 > 0.0 && p_joint > 0.0 {
                let pmi_val = (p_joint / (p_w1 * p_w2)).ln();
                if pmi_val > 0.5 {  // only keep positive PMI (co-occur more than chance)
                    self.pmi.insert((w1.clone(), w2.clone()), pmi_val);
                }
            }
        }
    }

    /// Find words with high PMI to a given word.
    fn similar_words(&self, word: &str, top_k: usize) -> Vec<(String, f32)> {
        let mut results: Vec<(String, f32)> = Vec::new();
        for ((w1, w2), &score) in &self.pmi {
            if w1 == word { results.push((w2.clone(), score)); }
            else if w2 == word { results.push((w1.clone(), score)); }
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Score: IDF + PMI expansion for unknown words.
    fn score(&self, query: &str, threshold: f32) -> Vec<(String, f32)> {
        let words = tokenize(query);
        let total_intents: f32 = {
            let mut all: HashSet<&str> = HashSet::new();
            for intents in self.word_intent.values() {
                for id in intents.keys() { all.insert(id.as_str()); }
            }
            all.len().max(1) as f32
        };

        let mut scores: HashMap<String, f32> = HashMap::new();

        for word in &words {
            if let Some(intents) = self.word_intent.get(word.as_str()) {
                // Direct match: IDF-weighted scoring
                let df = intents.len() as f32;
                let idf = (total_intents / df).ln().max(0.0);
                for (intent, &weight) in intents {
                    let w = (weight / self.word_count.get(word).copied().unwrap_or(1.0)).min(1.0);
                    *scores.entry(intent.clone()).or_default() += w * idf;
                }
            } else {
                // UNKNOWN word: find PMI-similar known words and use their scores
                let similar = self.similar_words(word, 3);
                for (sim_word, pmi_score) in &similar {
                    if let Some(intents) = self.word_intent.get(sim_word.as_str()) {
                        let df = intents.len() as f32;
                        let idf = (total_intents / df).ln().max(0.0);
                        // PMI-weighted contribution (weaker than direct match)
                        let pmi_factor = (pmi_score / 3.0).min(0.8);
                        for (intent, &weight) in intents {
                            let w = (weight / self.word_count.get(sim_word).copied().unwrap_or(1.0)).min(1.0);
                            *scores.entry(intent.clone()).or_default() += w * idf * pmi_factor;
                        }
                    }
                }
            }
        }

        let mut sorted: Vec<(String, f32)> = scores.into_iter()
            .filter(|(_, s)| *s > 0.0)
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.is_empty() { return sorted; }
        let top = sorted[0].1;
        if top < threshold { return vec![]; }
        // Keep intents scoring ≥ 40% of top
        sorted.into_iter().filter(|(_, s)| *s >= top * 0.4).collect()
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let text = text.replace("n't", " not").replace("'ve", " have")
        .replace("'re", " are").replace("'m", " am")
        .replace("'ll", " will").replace("'s", "").replace("'d", " would");
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && w.len() >= 2)
        .map(|s| s.to_string())
        .collect()
}

fn short(id: &str) -> &str { id.split(':').last().unwrap_or(id) }

fn main() {
    // Load actual benchmark
    let bm_json = std::fs::read_to_string("/tmp/asv_benchmark.json")
        .expect("Need /tmp/asv_benchmark.json");
    let bm: serde_json::Value = serde_json::from_str(&bm_json).unwrap();
    let mut all_queries: Vec<(String, Vec<String>)> = Vec::new();
    for batch in bm["batches"].as_array().unwrap() {
        for turn in batch["queries"].as_array().unwrap() {
            let msg = turn["message"].as_str().unwrap().to_string();
            let gt: Vec<String> = turn["ground_truth"].as_array().unwrap()
                .iter().filter_map(|v| v.as_str().map(String::from)).collect();
            all_queries.push((msg, gt));
        }
    }

    let mut engine = PmiEngine::new();

    // 5 original seeds per intent (same as benchmark)
    let seeds: &[(&str, &[&str])] = &[
        ("account:reset_password", &["reset my password", "forgot my password", "password expired and I can't log in", "need to change my password", "locked out because I entered the wrong password"]),
        ("account:unlock_account", &["my account is locked", "locked out of my account", "too many failed login attempts", "account has been blocked", "system won't let me in after failed attempts"]),
        ("account:request_access", &["need access to the shared drive", "request permission to view the database", "grant me access to this application", "I need access to the project folder", "access request for the internal portal"]),
        ("account:setup_mfa", &["set up two-factor authentication", "configure my authenticator app", "enable MFA on my account", "two-step verification is not working", "need to set up Google Authenticator"]),
        ("account:update_profile", &["update my email address", "change my phone number on file", "update my profile information", "edit my display name", "change my department"]),
        ("hardware:report_broken", &["my laptop is broken", "the screen is cracked", "computer won't turn on", "keyboard stopped working", "device is completely dead"]),
        ("hardware:request_equipment", &["need a new laptop", "request a second monitor", "can I get a docking station", "need a wireless keyboard and mouse", "requesting new hardware"]),
        ("hardware:setup_device", &["set up my new laptop", "configure my new workstation", "help setting up the new device", "initial setup for new computer", "need someone to initialize my new machine"]),
        ("hardware:request_loaner", &["need a loaner laptop", "borrow a temporary device", "replacement laptop while mine is being repaired", "need a spare computer", "temporary device while I wait for repair"]),
        ("software:install", &["install Microsoft Office", "need this software installed", "download and install the application", "can you install Slack", "need the program set up"]),
        ("software:license", &["my software license has expired", "need a license for Adobe", "software activation failed", "license key is not working", "need to purchase a new software license"]),
        ("software:troubleshoot_app", &["the application keeps crashing", "software not working properly", "getting an error message", "program freezes constantly", "can't open the application"]),
        ("software:update", &["update my software to the latest version", "need to install Windows updates", "my operating system is outdated", "run system updates", "need the latest version"]),
        ("software:uninstall", &["uninstall this program", "remove the application", "delete this software", "how do I uninstall this", "need old software removed"]),
        ("network:vpn", &["can't connect to VPN", "VPN is not working", "need VPN access to work from home", "VPN keeps disconnecting", "remote VPN connection failed"]),
        ("network:wifi", &["wifi is not connecting", "no internet connection", "dropped from the wireless network", "wifi signal is very weak", "can't connect to the office wifi"]),
        ("network:network_access", &["can't access the company network drive", "need permission to reach the internal server", "network share is not accessible", "can't see the shared folders", "need network access for a new project"]),
        ("network:remote_desktop", &["remote desktop is not working", "RDP connection failed", "can't connect to my office computer remotely", "need remote access to my work PC", "remote desktop keeps disconnecting"]),
        ("network:email", &["email is not syncing", "Outlook is not working", "can't send or receive emails", "email client stopped working", "mail server connection failed"]),
        ("tickets:create_ticket", &["open a new support ticket", "submit a help desk request", "log this issue with IT", "create a ticket for my problem", "report this to the IT team"]),
        ("tickets:check_ticket_status", &["what is the status of my ticket", "any update on my support request", "where is my case in the queue", "ticket status update please", "check on my open IT request"]),
        ("tickets:escalate_ticket", &["this is urgent please escalate my ticket", "need a faster response on my case", "escalate this to a senior technician", "this issue is critical for the business", "mark my ticket as high priority"]),
        ("tickets:close_ticket", &["my issue is fixed please close the ticket", "mark my request as resolved", "close my support case", "ticket resolved you can close it", "please mark this as done"]),
    ];

    // Simulated LLM output: diverse ways users EXPRESS each intent
    // This is what the LLM would generate when asked "how would users say this?"
    let llm_diverse: &[(&str, &[&str])] = &[
        ("tickets:escalate_ticket", &[
            "I'm so frustrated right now", "been waiting all morning for help",
            "this is completely unacceptable", "nobody seems to care about my problem",
            "I've lost patience with this process", "how much longer do I have to wait",
            "I want to speak with a manager", "this is the worst support experience",
            "I'm fed up with being ignored", "absolutely ridiculous response time",
            "getting nowhere with your team", "I'm at my wits end here",
            "this has been going on for days now", "can someone competent handle this",
            "I need this fixed right now not tomorrow",
        ]),
        ("hardware:setup_device", &[
            "just started today and nothing works", "new employee everything needs configuring",
            "they handed me a blank laptop", "I don't have any programs on my machine",
            "my workstation isn't ready yet", "when will my computer be set up",
            "first day and I can't do anything", "who helps new people get started",
            "all my colleagues have working machines except me",
        ]),
        ("account:request_access", &[
            "I'm not able to open the shared folder", "it says permission denied",
            "my manager says I should have access to this", "I can see the folder but can't open it",
            "getting access denied error", "who can give me permission to view these files",
            "I need to be added to the team's shared space",
        ]),
        ("account:reset_password", &[
            "I can't get in", "my login isn't working", "it keeps rejecting my credentials",
            "I've been locked out since this morning", "the system won't accept my password",
            "how do I get back into my account", "my password must have expired overnight",
        ]),
        ("network:wifi", &[
            "the internet is so slow I can't work", "keeps dropping every few minutes",
            "I have no connectivity at my desk", "the wireless signal is terrible here",
            "can't load any web pages", "my connection is extremely unstable",
        ]),
        ("network:vpn", &[
            "can't access anything from home", "the remote connection keeps failing",
            "working remotely and nothing loads", "home office setup isn't connecting",
        ]),
        ("hardware:report_broken", &[
            "my machine is completely dead", "screen went black and won't come back",
            "something is physically broken on my device", "it fell and now it won't start",
            "making strange noises and shutting down",
        ]),
        ("hardware:request_loaner", &[
            "I need something to work on while mine is fixed",
            "can I borrow a machine temporarily", "is there a spare I can use",
            "I'll be without a computer for a week otherwise",
        ]),
        ("account:setup_mfa", &[
            "how do I add the extra security to my login",
            "the two step thing for signing in", "my security code isn't working",
            "I need the authenticator thing on my phone",
        ]),
        ("tickets:check_ticket_status", &[
            "I submitted something last week and haven't heard back",
            "is anyone working on my issue", "what's happening with my request",
            "any progress on the thing I reported",
        ]),
    ];

    // ── TEST 1: Seeds only (baseline) ───
    let mut eng_seeds = PmiEngine::new();
    for (intent, phrases) in seeds {
        for p in *phrases { eng_seeds.learn(p, intent); }
    }
    eng_seeds.build_pmi();

    let (s_pass, s_partial, s_fail) = eval(&eng_seeds, &all_queries);
    let t = all_queries.len();
    println!("\n{:=<70}", "");
    println!("  PMI-Expanded Scoring — ACTUAL BENCHMARK ({} queries)", t);
    println!("{:=<70}\n", "");
    println!("  5 SEEDS ONLY:");
    println!("    Exact: {}/{} ({:.0}%) | Partial: {} | Fail: {} | PMI pairs: {}",
        s_pass, t, 100.0*s_pass as f32/t as f32, s_partial, s_fail, eng_seeds.pmi.len());

    // ── TEST 2: Seeds + LLM diverse phrases ───
    let mut eng_rich = PmiEngine::new();
    for (intent, phrases) in seeds {
        for p in *phrases { eng_rich.learn(p, intent); }
    }
    let mut extra_count = 0;
    for (intent, phrases) in llm_diverse {
        for p in *phrases { eng_rich.learn(p, intent); extra_count += 1; }
    }
    eng_rich.build_pmi();

    let (r_pass, r_partial, r_fail) = eval(&eng_rich, &all_queries);
    println!("\n  5 SEEDS + {} LLM PHRASES:", extra_count);
    println!("    Exact: {}/{} ({:.0}%) | Partial: {} | Fail: {} | PMI pairs: {}",
        r_pass, t, 100.0*r_pass as f32/t as f32, r_partial, r_fail, eng_rich.pmi.len());
    println!("\n    Δ exact: {:+}", r_pass as i32 - s_pass as i32);

    // Show how PMI expansion works on sample queries
    println!("\n{:=<70}", "");
    println!("  PMI EXPANSION EXAMPLES");
    println!("{:=<70}\n", "");

    let examples = [
        ("frustrated", "Is 'frustrated' expanded to known words?"),
        ("waiting", "Does 'waiting' have PMI neighbors?"),
        ("ignored", "Does 'ignored' expand to escalation words?"),
        ("dead", "Does 'dead' expand to broken device words?"),
        ("connectivity", "Does 'connectivity' expand to wifi words?"),
    ];

    for (word, question) in &examples {
        let similar = eng_rich.similar_words(word, 5);
        let known = eng_rich.word_intent.contains_key(*word);
        println!("  \"{}\" — {}", word, question);
        println!("    In vocabulary: {}", if known { "YES (direct match)" } else { "NO (needs PMI expansion)" });
        if similar.is_empty() {
            println!("    PMI neighbors: NONE");
        } else {
            for (w, score) in &similar {
                let intents: Vec<&str> = eng_rich.word_intent.get(w.as_str())
                    .map(|m| m.keys().map(|k| short(k)).collect())
                    .unwrap_or_default();
                println!("    → \"{}\" (PMI={:.2}) maps to {:?}", w, score, intents);
            }
        }
        println!();
    }

    // Show scoring on key queries
    println!("{:=<70}", "");
    println!("  QUERY SCORING COMPARISON");
    println!("{:=<70}\n", "");

    let test_queries = [
        "I'm so frustrated nobody is helping me",
        "been waiting all morning and nothing happened",
        "my machine is completely dead won't start",
        "the internet keeps cutting out at my desk",
        "just started today and my computer has nothing on it",
    ];

    for q in &test_queries {
        let scores = eng_rich.score(q, 0.01);
        let top3: Vec<String> = scores.iter().take(3)
            .map(|(id, s)| format!("{}={:.2}", short(id), s)).collect();
        println!("  \"{}\"", &q[..q.len().min(55)]);
        println!("    Top 3: {}", top3.join(", "));
        println!();
    }
}

fn eval(engine: &PmiEngine, queries: &[(String, Vec<String>)]) -> (usize, usize, usize) {
    let (mut p, mut pa, mut f) = (0, 0, 0);
    for (msg, gt) in queries {
        let got: HashSet<String> = engine.score(msg, 0.1)
            .iter().map(|(id, _)| id.clone()).collect();
        let gt_set: HashSet<String> = gt.iter().cloned().collect();
        if got == gt_set { p += 1; }
        else if !got.is_disjoint(&gt_set) { pa += 1; }
        else { f += 1; }
    }
    (p, pa, f)
}
