/// Experiment: Vector-based intent scoring vs HashMap 1-gram IDF.
///
/// Tests whether cosine similarity on word-intent vectors generalizes better
/// than discrete HashMap lookup. Critical test: does adding MORE phrases
/// help (vectors) or hurt (HashMap)?
///
/// Uses the ACTUAL frozen benchmark queries — no synthetic test cases.
///
/// Run: cargo run --bin experiment_vector
use std::collections::{HashMap, HashSet};

// ── Vector scoring engine ─────────────────────────────────────────────────────

struct VectorEngine {
    /// word → intent_index → weight (the raw co-occurrence data)
    word_weights: HashMap<String, Vec<f32>>,
    /// intent_id → index mapping
    intent_index: HashMap<String, usize>,
    /// index → intent_id
    index_to_intent: Vec<String>,
    /// Number of intents
    n_intents: usize,
    /// Intent centroids (precomputed, normalized)
    centroids: Vec<Vec<f32>>,
}

impl VectorEngine {
    fn new(intent_ids: &[String]) -> Self {
        let n = intent_ids.len();
        let intent_index: HashMap<String, usize> = intent_ids.iter()
            .enumerate().map(|(i, id)| (id.clone(), i)).collect();
        Self {
            word_weights: HashMap::new(),
            intent_index,
            index_to_intent: intent_ids.to_vec(),
            n_intents: n,
            centroids: vec![vec![0.0; n]; n],
        }
    }

    /// Learn a phrase for an intent. Each word gets weight added to its intent dimension.
    fn learn_phrase(&mut self, phrase: &str, intent_id: &str) {
        let idx = match self.intent_index.get(intent_id) {
            Some(&i) => i,
            None => return,
        };
        let words = tokenize_simple(phrase);
        for word in words {
            let vec = self.word_weights.entry(word)
                .or_insert_with(|| vec![0.0; self.n_intents]);
            vec[idx] += 1.0;
        }
    }

    /// Precompute intent centroids from word vectors.
    /// Centroid[intent] = average of all word vectors that have weight for this intent.
    fn build_centroids(&mut self) {
        self.centroids = vec![vec![0.0; self.n_intents]; self.n_intents];
        let mut counts = vec![0usize; self.n_intents];

        for (_, word_vec) in &self.word_weights {
            for intent_idx in 0..self.n_intents {
                if word_vec[intent_idx] > 0.0 {
                    for dim in 0..self.n_intents {
                        self.centroids[intent_idx][dim] += word_vec[dim];
                    }
                    counts[intent_idx] += 1;
                }
            }
        }

        // Normalize centroids
        for i in 0..self.n_intents {
            if counts[i] > 0 {
                let c = counts[i] as f32;
                for dim in 0..self.n_intents {
                    self.centroids[i][dim] /= c;
                }
            }
        }
    }

    /// Score a query against all intents using cosine similarity.
    fn score(&self, query: &str) -> Vec<(String, f32)> {
        let words = tokenize_simple(query);

        // Build query vector: sum of word vectors
        let mut query_vec = vec![0.0f32; self.n_intents];
        let mut found_words = 0;
        for word in &words {
            if let Some(wv) = self.word_weights.get(word.as_str()) {
                for dim in 0..self.n_intents {
                    query_vec[dim] += wv[dim];
                }
                found_words += 1;
            }
        }

        if found_words == 0 {
            return vec![];
        }

        // Cosine similarity with each intent centroid
        let mut scores: Vec<(String, f32)> = Vec::new();
        let q_mag = magnitude(&query_vec);
        if q_mag < 0.001 { return vec![]; }

        for i in 0..self.n_intents {
            let c_mag = magnitude(&self.centroids[i]);
            if c_mag < 0.001 { continue; }
            let dot: f32 = query_vec.iter().zip(&self.centroids[i])
                .map(|(a, b)| a * b).sum();
            let cosine = dot / (q_mag * c_mag);
            if cosine > 0.01 {
                scores.push((self.index_to_intent[i].clone(), cosine));
            }
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Multi-intent: return intents above threshold with re-pass gating.
    fn score_multi(&self, query: &str, threshold: f32, gate_ratio: f32) -> Vec<(String, f32)> {
        let all = self.score(query);
        if all.is_empty() { return all; }

        let top = all[0].1;
        if top < threshold { return vec![]; }

        // Include intents that score above threshold AND above gate_ratio of top
        all.into_iter()
            .filter(|(_, s)| *s >= threshold && *s >= top * gate_ratio)
            .collect()
    }
}

// ── 1-gram IDF engine (current system baseline) ──────────────────────────────

struct IdfEngine {
    word_intent: HashMap<String, Vec<(String, f32)>>,
}

impl IdfEngine {
    fn new() -> Self { Self { word_intent: HashMap::new() } }

    fn learn_phrase(&mut self, phrase: &str, intent: &str) {
        let words = tokenize_simple(phrase);
        for word in words {
            let entries = self.word_intent.entry(word).or_default();
            if let Some(e) = entries.iter_mut().find(|(id, _)| id == intent) {
                e.1 = (e.1 + 0.4 * (1.0 - e.1)).min(1.0);
            } else {
                entries.push((intent.to_string(), 0.4));
            }
        }
    }

    fn score(&self, query: &str, threshold: f32, gap: f32) -> Vec<(String, f32)> {
        let total_intents: f32 = {
            let mut all: HashSet<&str> = HashSet::new();
            for entries in self.word_intent.values() {
                for (id, _) in entries { all.insert(id.as_str()); }
            }
            all.len().max(1) as f32
        };

        let words = tokenize_simple(query);
        let mut scores: HashMap<String, f32> = HashMap::new();

        for word in &words {
            if let Some(entries) = self.word_intent.get(word.as_str()) {
                let idf = (total_intents / entries.len() as f32).ln().max(0.0);
                for (intent, weight) in entries {
                    *scores.entry(intent.clone()).or_default() += weight * idf;
                }
            }
        }

        let mut sorted: Vec<(String, f32)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.is_empty() { return sorted; }
        let top = sorted[0].1;
        if top < threshold { return vec![]; }
        sorted.into_iter().filter(|(_, s)| *s >= threshold && top - *s <= gap).collect()
    }
}

// ── Simple tokenizer (stop words preserved, lowercase) ────────────────────────

fn tokenize_simple(text: &str) -> Vec<String> {
    let text = text.replace("n't", " not").replace("'ve", " have")
        .replace("'re", " are").replace("'m", " am")
        .replace("'ll", " will").replace("'s", "").replace("'d", " would");
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && w.len() >= 2)
        .map(|s| s.to_string())
        .collect()
}

fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn short(id: &str) -> &str { id.split(':').last().unwrap_or(id) }

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    // Load actual frozen benchmark
    let benchmark_json = std::fs::read_to_string("/tmp/asv_benchmark.json")
        .expect("Cannot read /tmp/asv_benchmark.json — run benchmark_setup.py first");
    let benchmark: serde_json::Value = serde_json::from_str(&benchmark_json).unwrap();

    let mut all_queries: Vec<(String, Vec<String>)> = Vec::new();
    for batch in benchmark["batches"].as_array().unwrap() {
        for turn in batch["queries"].as_array().unwrap() {
            let msg = turn["message"].as_str().unwrap().to_string();
            let gt: Vec<String> = turn["ground_truth"].as_array().unwrap()
                .iter().filter_map(|v| v.as_str().map(String::from)).collect();
            all_queries.push((msg, gt));
        }
    }

    // Intent seed phrases (same as benchmark setup)
    let seeds: Vec<(&str, Vec<&str>)> = vec![
        ("account:reset_password", vec!["reset my password", "forgot my password", "password expired and I can't log in", "need to change my password", "locked out because I entered the wrong password"]),
        ("account:unlock_account", vec!["my account is locked", "locked out of my account", "too many failed login attempts", "account has been blocked", "system won't let me in after failed attempts"]),
        ("account:request_access", vec!["need access to the shared drive", "request permission to view the database", "grant me access to this application", "I need access to the project folder", "access request for the internal portal"]),
        ("account:setup_mfa", vec!["set up two-factor authentication", "configure my authenticator app", "enable MFA on my account", "two-step verification is not working", "need to set up Google Authenticator"]),
        ("account:update_profile", vec!["update my email address in the system", "change my phone number on file", "update my profile information", "edit my display name", "change my department in the directory"]),
        ("hardware:report_broken", vec!["my laptop is broken", "the screen is cracked", "computer won't turn on", "keyboard stopped working", "device is completely dead"]),
        ("hardware:request_equipment", vec!["need a new laptop", "request a second monitor", "can I get a docking station", "need a wireless keyboard and mouse", "requesting new hardware for my workstation"]),
        ("hardware:setup_device", vec!["set up my new laptop", "configure my new workstation", "help setting up the new device", "initial setup for new computer", "need someone to initialize my new machine"]),
        ("hardware:request_loaner", vec!["need a loaner laptop", "borrow a temporary device", "replacement laptop while mine is being repaired", "need a spare computer for now", "temporary device while I wait for repair"]),
        ("software:install", vec!["install Microsoft Office on my laptop", "need this software installed", "download and install the application", "can you install Slack for me", "need the program set up on my computer"]),
        ("software:license", vec!["my software license has expired", "need a license for Adobe Acrobat", "software activation failed", "license key is not working", "need to purchase a new software license"]),
        ("software:troubleshoot_app", vec!["the application keeps crashing", "software not working properly", "getting an error message when I open the app", "program freezes constantly", "can't open the application at all"]),
        ("software:update", vec!["update my software to the latest version", "need to install Windows updates", "my operating system is outdated", "run system updates on my computer", "need the latest version of the software"]),
        ("software:uninstall", vec!["uninstall this program from my laptop", "remove the application", "delete this software", "how do I uninstall this", "need old software removed from my machine"]),
        ("network:vpn", vec!["can't connect to VPN", "VPN is not working", "need VPN access to work from home", "VPN keeps disconnecting", "remote VPN connection failed"]),
        ("network:wifi", vec!["wifi is not connecting", "no internet connection", "dropped from the wireless network", "wifi signal is very weak", "can't connect to the office wifi"]),
        ("network:network_access", vec!["can't access the company network drive", "need permission to reach the internal server", "network share is not accessible", "can't see the shared folders on the network", "need network access for a new project"]),
        ("network:remote_desktop", vec!["remote desktop is not working", "RDP connection failed", "can't connect to my office computer remotely", "need remote access to my work PC", "remote desktop keeps disconnecting"]),
        ("network:email", vec!["email is not syncing", "Outlook is not working", "can't send or receive emails", "email client stopped working", "mail server connection failed"]),
        ("tickets:create_ticket", vec!["open a new support ticket", "submit a help desk request", "log this issue with IT", "create a ticket for my problem", "report this to the IT team formally"]),
        ("tickets:check_ticket_status", vec!["what is the status of my ticket", "any update on my support request", "where is my case in the queue", "ticket status update please", "check on my open IT request"]),
        ("tickets:escalate_ticket", vec!["this is urgent please escalate my ticket", "need a faster response on my case", "escalate this to a senior technician", "this issue is critical for the business", "mark my ticket as high priority"]),
        ("tickets:close_ticket", vec!["my issue is fixed please close the ticket", "mark my request as resolved", "close my support case", "ticket resolved you can close it", "please mark this as done"]),
    ];

    // Extra diverse phrases per intent (simulating what LLM would generate)
    let extra: Vec<(&str, Vec<&str>)> = vec![
        ("tickets:escalate_ticket", vec![
            "been waiting all morning", "been waiting for hours with no response",
            "this is ridiculous nobody is helping", "I am so frustrated right now",
            "this is completely unacceptable", "I am done waiting",
            "how much longer do I have to wait", "need this resolved immediately",
            "this has been going on for days", "can someone actually help me",
            "I want to speak to a manager", "third time I am asking",
            "nobody cares about my issue", "lost patience with this",
            "at my wits end", "getting nowhere with this support",
            "this is the worst support experience", "I am fed up",
            "absolutely terrible service", "why is this taking so long",
        ]),
        ("hardware:setup_device", vec![
            "help me get my new computer working", "new laptop need it configured",
            "new hire need workstation ready", "just started machine not set up",
            "initialize my new device", "new equipment needs to be configured",
            "get started with my new laptop", "everything installed on new PC",
            "nobody set up my computer before I started", "new employee need machine ready",
        ]),
        ("account:request_access", vec![
            "cannot access shared folder", "need permission to view project files",
            "do not have access to the system", "grant me access to database",
            "permissions not set up yet", "cannot see any shared files",
            "new employee cannot see files", "need access to internal tools",
            "how do I get into the team folder", "account does not have permissions",
        ]),
        ("account:reset_password", vec![
            "forgot my login credentials", "cannot get into my account",
            "login not working anymore", "change my password it expired",
            "system keeps saying invalid password", "locked out wrong password",
            "how do I get a new password", "reset my login please",
        ]),
        ("network:wifi", vec![
            "internet keeps cutting out", "cannot get online at all",
            "connection drops every few minutes", "wifi signal terrible",
            "network super slow today", "no internet access from desk",
            "wireless keeps disconnecting", "the internet is down again",
        ]),
        ("network:vpn", vec![
            "cannot connect remotely from home", "remote connection keeps failing",
            "VPN drops every few minutes", "working from home cannot access anything",
            "secure connection not working",
        ]),
        ("hardware:report_broken", vec![
            "laptop screen shattered", "computer will not power on",
            "device is completely dead", "spilled coffee keyboard stopped",
            "machine keeps crashing and shutting down", "laptop fell screen went black",
        ]),
        ("hardware:request_loaner", vec![
            "temporary laptop while mine being fixed", "borrow a spare machine",
            "something to work on while waiting for repair", "loaner in the meantime",
            "laptop in repair need replacement to keep working",
        ]),
        ("account:setup_mfa", vec![
            "set up security code on my phone", "configure two step login",
            "authenticator app not working", "enable two factor on account",
            "add verification code thing", "extra login security step",
        ]),
        ("tickets:check_ticket_status", vec![
            "where is my support request", "any update on issue I reported",
            "what happened to my IT request", "submitted ticket last week no response",
        ]),
    ];

    let intent_ids: Vec<String> = seeds.iter().map(|(id, _)| id.to_string()).collect();

    // ═══════════════════════════════════════════════════════════════════════════
    //  TEST 1: 5 seeds only
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n{:=<70}", "");
    println!("  Vector vs IDF — ACTUAL BENCHMARK ({} queries)", all_queries.len());
    println!("{:=<70}\n", "");

    // Build engines with 5 seeds
    let mut vec_5 = VectorEngine::new(&intent_ids);
    let mut idf_5 = IdfEngine::new();
    for (intent, phrases) in &seeds {
        for phrase in phrases {
            vec_5.learn_phrase(phrase, intent);
            idf_5.learn_phrase(phrase, intent);
        }
    }
    vec_5.build_centroids();

    let (v5_pass, v5_partial, v5_fail) = eval(&vec_5, &all_queries, 0.05, 0.4);
    let (i5_pass, i5_partial, i5_fail) = eval_idf(&idf_5, &all_queries, 0.3, 1.5);
    let t = all_queries.len();

    println!("  ── 5 SEEDS PER INTENT ──");
    println!("  Vector:  {}/{} ({:.0}%) exact | {} partial | {} fail",
        v5_pass, t, 100.0*v5_pass as f32/t as f32, v5_partial, v5_fail);
    println!("  IDF:     {}/{} ({:.0}%) exact | {} partial | {} fail",
        i5_pass, t, 100.0*i5_pass as f32/t as f32, i5_partial, i5_fail);

    // ═══════════════════════════════════════════════════════════════════════════
    //  TEST 2: 5 seeds + extra phrases (does more data help or hurt?)
    // ═══════════════════════════════════════════════════════════════════════════
    let mut vec_50 = VectorEngine::new(&intent_ids);
    let mut idf_50 = IdfEngine::new();
    for (intent, phrases) in &seeds {
        for phrase in phrases {
            vec_50.learn_phrase(phrase, intent);
            idf_50.learn_phrase(phrase, intent);
        }
    }
    let mut extra_count = 0;
    for (intent, phrases) in &extra {
        for phrase in phrases {
            vec_50.learn_phrase(phrase, intent);
            idf_50.learn_phrase(phrase, intent);
            extra_count += 1;
        }
    }
    vec_50.build_centroids();

    let (v50_pass, v50_partial, v50_fail) = eval(&vec_50, &all_queries, 0.05, 0.4);
    let (i50_pass, i50_partial, i50_fail) = eval_idf(&idf_50, &all_queries, 0.3, 1.5);

    println!("\n  ── 5 SEEDS + {} EXTRA PHRASES ──", extra_count);
    println!("  Vector:  {}/{} ({:.0}%) exact | {} partial | {} fail",
        v50_pass, t, 100.0*v50_pass as f32/t as f32, v50_partial, v50_fail);
    println!("  IDF:     {}/{} ({:.0}%) exact | {} partial | {} fail",
        i50_pass, t, 100.0*i50_pass as f32/t as f32, i50_partial, i50_fail);

    // ═══════════════════════════════════════════════════════════════════════════
    //  COMPARISON
    // ═══════════════════════════════════════════════════════════════════════════
    println!("\n{:=<70}", "");
    println!("  CRITICAL TEST: Does more data help or hurt?");
    println!("{:=<70}", "");
    println!("  Vector: {} → {} ({:+}) with +{} phrases",
        v5_pass, v50_pass, v50_pass as i32 - v5_pass as i32, extra_count);
    println!("  IDF:    {} → {} ({:+}) with +{} phrases",
        i5_pass, i50_pass, i50_pass as i32 - i5_pass as i32, extra_count);

    if v50_pass > v5_pass && i50_pass <= i5_pass {
        println!("\n  ✓ VECTOR SCALES — more data helps vectors but hurts IDF");
    } else if v50_pass > v5_pass {
        println!("\n  ✓ VECTOR IMPROVES with more data");
    } else if v50_pass <= v5_pass {
        println!("\n  ✗ VECTOR ALSO DEGRADES with more data — same problem as IDF");
    }

    // Show sample queries with score breakdown
    println!("\n{:=<70}", "");
    println!("  SAMPLE: Queries that should benefit from extra phrases");
    println!("{:=<70}\n", "");

    let samples = [
        "I've been waiting all morning and nobody is helping me",
        "I'm so frustrated with this terrible service",
        "help me get my new computer working I just started",
        "internet keeps cutting out and VPN drops",
    ];

    for q in &samples {
        let v_scores = vec_50.score_multi(q, 0.05, 0.4);
        let i_scores = idf_50.score(q, 0.3, 1.5);
        let vs: Vec<(&str, String)> = v_scores.iter().map(|(id, s)| (short(id), format!("{:.3}", s))).collect();
        let is: Vec<(&str, String)> = i_scores.iter().map(|(id, s)| (short(id), format!("{:.2}", s))).collect();
        println!("  \"{}\"", &q[..q.len().min(60)]);
        println!("    Vector: {:?}", vs);
        println!("    IDF:    {:?}", is);
        println!();
    }
}

fn eval(engine: &VectorEngine, queries: &[(String, Vec<String>)],
        threshold: f32, gate_ratio: f32) -> (usize, usize, usize) {
    let (mut p, mut pa, mut f) = (0, 0, 0);
    for (msg, gt) in queries {
        let got: HashSet<String> = engine.score_multi(msg, threshold, gate_ratio)
            .iter().map(|(id, _)| id.clone()).collect();
        let gt_set: HashSet<String> = gt.iter().cloned().collect();
        if got == gt_set { p += 1; }
        else if !got.is_disjoint(&gt_set) { pa += 1; }
        else { f += 1; }
    }
    (p, pa, f)
}

fn eval_idf(engine: &IdfEngine, queries: &[(String, Vec<String>)],
            threshold: f32, gap: f32) -> (usize, usize, usize) {
    let (mut p, mut pa, mut f) = (0, 0, 0);
    for (msg, gt) in queries {
        let got: HashSet<String> = engine.score(msg, threshold, gap)
            .iter().map(|(id, _)| id.clone()).collect();
        let gt_set: HashSet<String> = gt.iter().cloned().collect();
        if got == gt_set { p += 1; }
        else if !got.is_disjoint(&gt_set) { pa += 1; }
        else { f += 1; }
    }
    (p, pa, f)
}
