/// Experiment: Word vectors from LLM phrase co-occurrence.
///
/// KEY QUESTION: With only 20 phrases per intent, does a word×phrase
/// co-occurrence matrix produce useful word similarity?
///
/// If "waiting" and "frustrated" end up CLOSE in this space
/// (because they appear in phrases of the same intent), then
/// LLM distillation into vectors WORKS with minimal data.
///
/// If not, this approach needs more data than is realistic.
///
/// No external dependencies. Pure matrix math.
///
/// Run: cargo run --bin experiment_svd
use std::collections::{HashMap, HashSet};

// ── Word-Phrase Matrix ────────────────────────────────────────────────────────

struct PhraseMatrix {
    /// word → index
    vocab: HashMap<String, usize>,
    /// index → word
    words: Vec<String>,
    /// phrase → intent
    phrase_intents: Vec<String>,
    /// Matrix: rows = words, cols = phrases. Value = 1.0 if word in phrase.
    matrix: Vec<Vec<f32>>,
    /// intent → centroid vector (in word-space)
    centroids: HashMap<String, Vec<f32>>,
}

impl PhraseMatrix {
    fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            words: Vec::new(),
            phrase_intents: Vec::new(),
            matrix: Vec::new(),
            centroids: HashMap::new(),
        }
    }

    fn add_phrase(&mut self, phrase: &str, intent: &str) {
        let tokens = tokenize(phrase);
        let phrase_idx = self.phrase_intents.len();
        self.phrase_intents.push(intent.to_string());

        // Ensure all words have indices
        for t in &tokens {
            if !self.vocab.contains_key(t.as_str()) {
                let idx = self.words.len();
                self.vocab.insert(t.clone(), idx);
                self.words.push(t.clone());
                self.matrix.push(Vec::new());
            }
        }

        // Pad all rows to current phrase count, then fill
        let n_phrases = phrase_idx + 1;
        for row in &mut self.matrix {
            while row.len() < n_phrases { row.push(0.0); }
        }
        for t in &tokens {
            let word_idx = self.vocab[t.as_str()];
            while self.matrix[word_idx].len() < n_phrases { self.matrix[word_idx].push(0.0); }
            self.matrix[word_idx][phrase_idx] = 1.0;
        }
    }

    /// Word similarity: cosine of their phrase-occurrence vectors.
    /// Two words are similar if they appear in similar sets of phrases.
    fn word_similarity(&self, w1: &str, w2: &str) -> f32 {
        let i1 = match self.vocab.get(w1) { Some(&i) => i, None => return 0.0 };
        let i2 = match self.vocab.get(w2) { Some(&i) => i, None => return 0.0 };
        cosine(&self.matrix[i1], &self.matrix[i2])
    }

    /// Build intent centroids: average of all phrase vectors for each intent.
    /// A phrase vector is: for each word dimension, 1.0 if word is in phrase.
    /// So centroid = for each word, fraction of intent's phrases containing that word.
    fn build_centroids(&mut self) {
        let n_words = self.words.len();
        let mut intent_phrase_count: HashMap<String, f32> = HashMap::new();
        self.centroids.clear();

        for (phrase_idx, intent) in self.phrase_intents.iter().enumerate() {
            let centroid = self.centroids.entry(intent.clone())
                .or_insert_with(|| vec![0.0; n_words]);
            *intent_phrase_count.entry(intent.clone()).or_insert(0.0) += 1.0;

            for word_idx in 0..n_words {
                centroid[word_idx] += self.matrix[word_idx][phrase_idx];
            }
        }

        // Normalize by phrase count
        for (intent, centroid) in &mut self.centroids {
            let count = intent_phrase_count[intent];
            for v in centroid.iter_mut() {
                *v /= count;
            }
        }
    }

    /// Score a query: build query vector, cosine with intent centroids.
    fn score(&self, query: &str) -> Vec<(String, f32)> {
        let tokens = tokenize(query);
        let n_words = self.words.len();
        let mut query_vec = vec![0.0f32; n_words];

        let mut found = 0;
        for t in &tokens {
            if let Some(&idx) = self.vocab.get(t.as_str()) {
                query_vec[idx] = 1.0;
                found += 1;
            }
        }

        if found == 0 {
            // No known words: try to find similar words for each unknown token
            for t in &tokens {
                if self.vocab.contains_key(t.as_str()) { continue; }
                // Find most similar known word
                let best = self.find_similar_word(t, 1);
                for (sim_word, sim_score) in &best {
                    if let Some(&idx) = self.vocab.get(sim_word.as_str()) {
                        query_vec[idx] += sim_score * 0.5; // dampened
                    }
                }
            }
        }

        let mut scores: Vec<(String, f32)> = Vec::new();
        for (intent, centroid) in &self.centroids {
            let sim = cosine(&query_vec, centroid);
            if sim > 0.01 {
                scores.push((intent.clone(), sim));
            }
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Find known words most similar to an unknown word.
    /// Uses character overlap as a fallback (not semantic — just morphological).
    fn find_similar_word(&self, unknown: &str, top_k: usize) -> Vec<(String, f32)> {
        // For now: no way to find similar words without pre-trained vectors
        // This is the gap that needs filling
        let _ = (unknown, top_k);
        vec![]
    }

    fn score_multi(&self, query: &str, threshold: f32) -> Vec<(String, f32)> {
        let all = self.score(query);
        if all.is_empty() { return all; }
        let top = all[0].1;
        if top < threshold { return vec![]; }
        all.into_iter().filter(|(_, s)| *s >= top * 0.5).collect()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let ma: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if ma < 0.001 || mb < 0.001 { 0.0 } else { dot / (ma * mb) }
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

// ── Simple truncated SVD via power iteration ──────────────────────────────────

/// Compute top-K left singular vectors of matrix (rows × cols).
/// Returns word vectors: Vec<Vec<f32>> of shape [n_words][k].
fn truncated_svd(matrix: &[Vec<f32>], k: usize, iterations: usize) -> Vec<Vec<f32>> {
    let n_rows = matrix.len();
    if n_rows == 0 { return vec![]; }
    let n_cols = matrix[0].len();

    let mut result: Vec<Vec<f32>> = vec![vec![0.0; k]; n_rows];

    for component in 0..k {
        // Random initial vector (deterministic seed)
        let mut v: Vec<f32> = (0..n_cols).map(|i| ((i * 7 + component * 13 + 37) % 100) as f32 / 100.0).collect();
        normalize_vec(&mut v);

        for _ in 0..iterations {
            // u = M * v (multiply matrix by right vector → left vector)
            let mut u: Vec<f32> = vec![0.0; n_rows];
            for (i, row) in matrix.iter().enumerate() {
                u[i] = dot_product(row, &v);
            }

            // Deflate: remove contribution of previous components
            for prev in 0..component {
                let prev_u: Vec<f32> = result.iter().map(|r| r[prev]).collect();
                let proj = dot_product_vecs(&u, &prev_u);
                for i in 0..n_rows {
                    u[i] -= proj * prev_u[i];
                }
            }

            normalize_vec(&mut u);

            // v = M^T * u (multiply transpose by left vector → right vector)
            v = vec![0.0; n_cols];
            for (i, row) in matrix.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    v[j] += val * u[i];
                }
            }
            normalize_vec(&mut v);
        }

        // Final u = M * v
        let mut u: Vec<f32> = vec![0.0; n_rows];
        for (i, row) in matrix.iter().enumerate() {
            u[i] = dot_product(row, &v);
        }
        // Deflate
        for prev in 0..component {
            let prev_u: Vec<f32> = result.iter().map(|r| r[prev]).collect();
            let proj = dot_product_vecs(&u, &prev_u);
            for i in 0..n_rows {
                u[i] -= proj * prev_u[i];
            }
        }
        // Singular value
        let sigma = magnitude_vec(&u);
        if sigma > 0.001 {
            for i in 0..n_rows {
                result[i][component] = u[i] / sigma;
            }
        }
    }

    result
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn dot_product_vecs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize_vec(v: &mut [f32]) {
    let mag = magnitude_vec(v);
    if mag > 0.001 { for x in v.iter_mut() { *x /= mag; } }
}

fn magnitude_vec(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn main() {
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

    // 20 diverse phrases per intent (simulating LLM output — realistic variety)
    let phrases: &[(&str, &[&str])] = &[
        ("account:reset_password", &[
            "reset my password", "forgot my password", "password expired", "can't log in", "locked out wrong password",
            "I forgot my login credentials", "my password isn't working", "need a new password",
            "the system won't accept my password", "how do I change my password",
            "I can't get into my account", "my login expired", "password reset please",
            "authentication failed when logging in", "I need to recover my account access",
            "keeps saying wrong credentials", "haven't been able to sign in all day",
            "my account password needs changing", "login screen rejects everything I type",
            "been locked out since this morning",
        ]),
        ("account:unlock_account", &[
            "my account is locked", "locked out of my account", "too many failed attempts", "account blocked", "system won't let me in",
            "I got locked out after entering wrong password too many times",
            "my account has been disabled", "account frozen need help",
            "security lockout on my profile", "temporarily suspended account",
            "I can't access anything my account is frozen",
            "the system blocked me out", "need my account unlocked",
            "got a lockout notification", "account access suspended",
            "too many bad passwords now I'm locked", "security block on my login",
            "please remove the lock from my account", "been blocked since yesterday",
            "account temporarily unavailable",
        ]),
        ("account:request_access", &[
            "need access to shared drive", "request permission to view database", "grant me access", "I need access to project folder", "access request for portal",
            "I can't see the team files", "permission denied on the shared folder",
            "my manager says I should have access", "who gives permissions around here",
            "I need to be added to the project workspace", "access denied error",
            "how do I get into the team drive", "can't open any shared documents",
            "need read write access to the repository", "permission missing for internal tool",
            "new here and I can't access anything", "my colleague shared a folder but I can't open it",
            "requesting access to the staging environment", "need credentials for the admin panel",
            "I don't have the right permissions yet",
        ]),
        ("account:setup_mfa", &[
            "set up two-factor authentication", "configure authenticator app", "enable MFA", "two-step verification not working", "need to set up Google Authenticator",
            "how do I add the security code to my phone", "the two step login thing",
            "need help with the authenticator setup", "MFA enrollment required",
            "setting up the verification code app", "my security token isn't working",
            "I want extra protection on my login", "how do I enable the second factor",
            "authenticator keeps giving wrong codes", "where do I scan the QR code",
            "two factor authentication enrollment", "setting up login verification",
            "the extra security step during sign in", "need the code generator configured",
            "multi factor authentication setup help",
        ]),
        ("hardware:report_broken", &[
            "my laptop is broken", "screen is cracked", "computer won't turn on", "keyboard stopped working", "device completely dead",
            "my machine died this morning", "the screen went black and won't come back",
            "something is physically wrong with my device", "laptop fell off desk and broke",
            "making loud clicking noises then shut down", "screen has lines running through it",
            "the trackpad doesn't respond anymore", "my computer keeps randomly shutting off",
            "blue screen of death every time I boot", "hardware failure I think",
            "the power button does nothing", "smoke came out of my laptop",
            "battery swollen and device won't charge", "display is flickering badly",
            "dropped my laptop and now it won't start",
        ]),
        ("hardware:setup_device", &[
            "set up my new laptop", "configure new workstation", "help setting up new device", "initial setup for new computer", "need someone to initialize my new machine",
            "just got a new laptop need everything installed", "new hire workstation setup",
            "my new computer has nothing on it", "who configures machines for new employees",
            "first day and my device isn't ready", "need all the standard software on my new PC",
            "how do I get started with this new laptop", "new equipment arrived need it configured",
            "blank machine needs company setup", "onboarding tech setup",
            "everything needs to be installed from scratch", "fresh device needs configuration",
            "setting up for a new team member", "new hardware needs to be provisioned",
            "just received replacement machine need it set up",
        ]),
        ("hardware:request_loaner", &[
            "need a loaner laptop", "borrow a temporary device", "replacement while mine is repaired", "need a spare computer", "temporary device while I wait",
            "can I get a backup machine to use", "need something to work on meanwhile",
            "is there a spare laptop available", "loaner device request",
            "I'll be without a computer for a week need a temp one",
            "what do I use while my laptop is in the shop",
            "need a temporary workstation", "borrow equipment while mine is being fixed",
            "interim device needed urgently", "can't work without a machine need a loaner",
            "my device is in repair I need a substitute", "temporary replacement laptop please",
            "working from a borrowed device in the meantime",
            "need access to a pool laptop", "short term equipment loan request",
        ]),
        ("hardware:request_equipment", &[
            "need a new laptop", "request a second monitor", "can I get a docking station", "need wireless keyboard and mouse", "requesting new hardware",
            "I need a bigger monitor for design work", "my current laptop is too slow",
            "requesting an ergonomic keyboard", "need headphones for video calls",
            "can I get a webcam for my desk", "my monitor is too small",
            "need a USB hub for all my peripherals", "requesting a standing desk converter",
            "laptop bag request", "need a display adapter",
            "can I get upgraded RAM for my machine", "external hard drive request",
            "need a new charger mine is frayed", "requesting dual monitor setup",
            "need better equipment for my role",
        ]),
        ("tickets:escalate_ticket", &[
            "this is urgent please escalate my ticket", "need a faster response", "escalate to senior technician", "this issue is critical", "mark as high priority",
            "I'm so frustrated nobody is helping me", "been waiting all morning for a response",
            "this is completely unacceptable service", "I want to speak with a manager",
            "how much longer do I have to wait for this", "I've lost patience with this process",
            "this has been going on for days with no resolution", "getting absolutely nowhere",
            "I am at my wits end with your support", "third time I'm reaching out about this",
            "can someone competent please handle my case", "this is beyond ridiculous",
            "I need this resolved immediately not next week", "worst IT support experience ever",
            "I'm fed up and want to file a formal complaint",
        ]),
        ("tickets:check_ticket_status", &[
            "what is the status of my ticket", "any update on my support request", "where is my case in the queue", "ticket status update please", "check on my open IT request",
            "I submitted a request last week any news", "has anyone looked at my ticket yet",
            "my issue was reported three days ago still waiting", "tracking number update",
            "is my case being worked on", "when will someone get to my request",
            "any progress on the problem I reported", "just checking in on my open case",
            "what's the ETA on my support ticket", "I want to know where things stand",
            "following up on ticket number from last week",
            "has my request been assigned to anyone", "waiting for an update on my issue",
            "my case seems stuck can you check", "status inquiry for submitted request",
        ]),
        ("tickets:create_ticket", &[
            "open a new support ticket", "submit a help desk request", "log this issue with IT", "create a ticket for my problem", "report this to IT team",
            "I want to formally report this issue", "please document this problem",
            "how do I file a support request", "need to create a new case",
            "logging a new IT issue", "I'd like to open a trouble ticket",
            "where do I submit my IT problem", "need to register a complaint with IT",
            "starting a new support case", "filing a request for technical help",
            "opening a new service request", "documenting this for the IT team",
            "I need a ticket created for this", "putting in a new work order",
            "initiating a support case",
        ]),
        ("tickets:close_ticket", &[
            "my issue is fixed please close the ticket", "mark as resolved", "close my support case", "ticket resolved you can close it", "please mark this as done",
            "everything is working now you can close this", "problem solved thanks",
            "the fix worked go ahead and close it", "issue has been resolved",
            "no longer need help this is fixed", "all good now please close",
            "the solution worked case can be closed", "my problem was addressed thanks",
            "resolved on my end", "this ticket can be closed now",
            "thank you it's working marking as complete", "please archive this case",
            "done with this issue", "support was helpful problem gone",
            "confirming this is resolved and can be closed",
        ]),
        ("network:vpn", &[
            "can't connect to VPN", "VPN is not working", "need VPN access from home", "VPN keeps disconnecting", "remote VPN connection failed",
            "the VPN client won't connect", "VPN drops every few minutes",
            "working from home and can't access company resources",
            "my secure connection keeps timing out", "VPN authentication error",
            "the remote access tunnel won't establish", "split tunneling not working",
            "VPN connected but can't reach internal sites", "need to reconfigure VPN client",
            "corporate VPN extremely slow today", "can't VPN in from this network",
            "VPN certificate expired", "two factor for VPN not working",
            "home internet fine but VPN won't connect", "need VPN credentials reset",
        ]),
        ("network:wifi", &[
            "wifi is not connecting", "no internet connection", "dropped from wireless network", "wifi signal very weak", "can't connect to office wifi",
            "the internet keeps cutting out", "no connectivity at my desk",
            "wifi drops every few minutes", "wireless signal terrible in this building",
            "can't get online at all", "internet is extremely slow today",
            "my connection is completely unstable", "network keeps disconnecting",
            "wifi password isn't working", "no signal in the conference room",
            "internet outage at the office", "can't load any web pages",
            "ethernet works but wifi doesn't", "the wireless is down again",
            "struggling with internet connectivity all day",
        ]),
        ("network:network_access", &[
            "can't access company network drive", "need permission for internal server", "network share not accessible", "can't see shared folders", "need network access for project",
            "mapped drive disappeared", "file server access denied",
            "can't connect to the NAS", "shared network path not found",
            "need to mount the department share", "network printer not showing up",
            "internal web server unreachable", "can't browse network resources",
            "DNS not resolving internal hostnames", "need to be added to network group",
            "my network drive mapping broke", "can't access intranet sites",
            "permission denied on network share", "need VPN split tunnel for internal access",
            "corporate network resources unavailable",
        ]),
        ("network:remote_desktop", &[
            "remote desktop not working", "RDP connection failed", "can't connect to office computer remotely", "need remote access to work PC", "remote desktop keeps disconnecting",
            "RDP session freezes", "can't log into my desktop from home",
            "remote connection extremely laggy", "need to set up remote desktop",
            "my remote session keeps timing out", "RDP credentials not accepted",
            "screen goes black during remote session", "need to restart my office PC remotely",
            "remote desktop resolution is wrong", "can't copy paste in RDP session",
            "multiple monitor support broken in remote", "remote desktop audio not working",
            "need to connect to a different machine remotely", "RDP license error",
            "can't find my computer on the remote desktop gateway",
        ]),
        ("network:email", &[
            "email is not syncing", "Outlook is not working", "can't send or receive emails", "email client stopped working", "mail server connection failed",
            "my inbox isn't updating", "emails are stuck in outbox",
            "Outlook keeps crashing when I open it", "calendar invites not showing up",
            "email attachment size limit issue", "can't add my email account",
            "getting bounce back on every email I send", "spam filter blocking legitimate emails",
            "email signature not displaying correctly", "shared mailbox not accessible",
            "out of office auto reply not working", "email search not returning results",
            "Outlook running extremely slow", "email rules stopped working",
            "can't connect to Exchange server",
        ]),
        ("software:install", &[
            "install Microsoft Office", "need this software installed", "download and install the application", "can you install Slack", "need the program set up",
            "I need Zoom installed for meetings", "requesting software installation",
            "how do I get this app on my computer", "need development tools installed",
            "install the latest version of Chrome", "requesting Adobe Creative Suite",
            "need Python installed for my work", "can someone install this for me",
            "software request for my project", "need a specific program added",
            "install the company standard toolkit", "requesting licensed software",
            "need Visual Studio on my machine", "please install the VPN client",
            "requesting approval to install an application",
        ]),
        ("software:license", &[
            "my software license expired", "need a license for Adobe", "software activation failed", "license key not working", "need to purchase a new license",
            "license renewal required", "trial period ended need full version",
            "not enough seats on our team license", "software says unlicensed",
            "need to transfer my license to new machine", "activation code rejected",
            "license server unreachable", "compliance audit found unlicensed software",
            "requesting additional user licenses", "my license was deactivated",
            "need to upgrade from individual to team license", "license expired notification",
            "can't activate after OS reinstall", "floating license not available",
            "need perpetual license not subscription",
        ]),
        ("software:troubleshoot_app", &[
            "application keeps crashing", "software not working properly", "getting an error message", "program freezes constantly", "can't open the application",
            "app crashes immediately on launch", "blue screen when running this program",
            "error code popping up every time", "the software hangs and becomes unresponsive",
            "unexpected behavior in the application", "program works on colleague's machine but not mine",
            "compatibility issue after update", "the application is extremely slow",
            "getting a runtime error", "software conflict with another program",
            "the app worked yesterday but not today", "crashes when I try to save",
            "graphical glitches in the software", "out of memory error in application",
            "plugin causing the host app to crash",
        ]),
        ("software:update", &[
            "update my software", "need to install Windows updates", "operating system outdated", "run system updates", "need latest version",
            "security patch needs to be applied", "my software version is too old",
            "automatic updates seem to be stuck", "how do I update this application",
            "update failed with error code", "need to upgrade to newest release",
            "system is prompting me to update but it fails", "mandatory update notification",
            "haven't updated in months need to catch up", "update broke something",
            "rolling back a bad update", "firmware update required",
            "browser needs updating", "Java version outdated causing issues",
            "need to schedule maintenance window for updates",
        ]),
        ("software:uninstall", &[
            "uninstall this program", "remove the application", "delete this software", "how do I uninstall this", "need old software removed",
            "leftover software from previous user", "unwanted application on my machine",
            "how do I completely remove this", "uninstall isn't working properly",
            "need to clean up unused programs", "software removal request",
            "this bloatware is slowing my computer", "can't find the uninstaller",
            "program won't let me remove it", "need admin rights to uninstall",
            "want this trial software gone", "removing deprecated software",
            "clean uninstall needed for reinstall", "leftover files after uninstall",
            "need IT to remove this application",
        ]),
        ("account:update_profile", &[
            "update my email address", "change my phone number", "update my profile information", "edit my display name", "change my department",
            "my name changed need to update it everywhere", "wrong title in the directory",
            "update my emergency contact info", "profile photo needs changing",
            "my office location changed in the system", "need to update my job title",
            "wrong manager listed on my profile", "update my team assignment",
            "my extension number changed", "preferred name update request",
            "change my listed pronouns", "update my mailing address",
            "profile says wrong building", "need to add my certification to profile",
            "update my skills in the HR system",
        ]),
    ];

    let mut matrix = PhraseMatrix::new();
    let mut total_phrases = 0;
    for (intent, intent_phrases) in phrases {
        for p in *intent_phrases {
            matrix.add_phrase(p, intent);
            total_phrases += 1;
        }
    }
    matrix.build_centroids();

    println!("\n{:=<70}", "");
    println!("  SVD-less Vector Test: {} phrases, {} words, {} intents",
        total_phrases, matrix.words.len(), matrix.centroids.len());
    println!("{:=<70}\n", "");

    // ── TEST 1: Word similarity (the critical question) ──
    println!("  WORD SIMILARITY (do semantically related words end up close?):\n");
    let pairs = [
        ("frustrated", "waiting", "Both → escalate_ticket?"),
        ("frustrated", "angry", "Both emotional → escalate?"),
        ("frustrated", "password", "Should be FAR apart"),
        ("broken", "dead", "Both → report_broken?"),
        ("broken", "wifi", "Should be FAR apart"),
        ("laptop", "computer", "Synonyms → same intents?"),
        ("install", "uninstall", "Related but different intents"),
        ("slow", "disconnecting", "Both network issues?"),
        ("access", "permission", "Both → request_access?"),
        ("waiting", "status", "Both → ticket related?"),
    ];

    for (w1, w2, note) in &pairs {
        let sim = matrix.word_similarity(w1, w2);
        let bar = "█".repeat((sim * 20.0) as usize);
        println!("    {:<14} ~ {:<14}  sim={:.3}  {}  {}", w1, w2, sim, bar, note);
    }

    // ── TEST 2: Query scoring on benchmark ──
    println!("\n{:=<70}", "");
    println!("  BENCHMARK SCORING ({} queries)", all_queries.len());
    println!("{:=<70}\n", "");

    let mut exact = 0;
    let mut partial = 0;
    let mut fail = 0;
    for (msg, gt) in &all_queries {
        let got: HashSet<String> = matrix.score_multi(msg, 0.1)
            .iter().map(|(id, _)| id.clone()).collect();
        let gt_set: HashSet<String> = gt.iter().cloned().collect();
        if got == gt_set { exact += 1; }
        else if !got.is_disjoint(&gt_set) { partial += 1; }
        else { fail += 1; }
    }
    let t = all_queries.len();
    println!("  Exact: {}/{} ({:.0}%) | Partial: {} | Fail: {}", exact, t, 100.0*exact as f32/t as f32, partial, fail);

    // ── TEST 3: Sample queries ──
    println!("\n  SAMPLE QUERIES:\n");
    let samples = [
        "I'm so frustrated nobody is helping me",
        "been waiting all morning and nothing happened",
        "my machine is completely dead won't start",
        "the internet keeps cutting out at my desk",
        "just started today and my computer has nothing on it",
        "can't get into my account the password is wrong",
        "I need the authenticator thing on my phone",
        "VPN drops every few minutes from home",
    ];
    for q in &samples {
        let scores = matrix.score_multi(q, 0.1);
        let top: Vec<String> = scores.iter().take(3)
            .map(|(id, s)| format!("{}={:.3}", short(id), s)).collect();
        println!("    \"{}\"", &q[..q.len().min(55)]);
        println!("      → {}", if top.is_empty() { "NOTHING".to_string() } else { top.join(", ") });
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  PHASE 2: SVD — compress word vectors, test if similarity emerges
    // ══════════════════════════════════════════════════════════════════════════
    println!("\n{:=<70}", "");
    println!("  PHASE 2: SVD on word×phrase matrix (709 words × 460 phrases)");
    println!("{:=<70}\n", "");

    // Pad all matrix rows to same length
    let n_phrases = matrix.phrase_intents.len();
    let mut mat: Vec<Vec<f32>> = matrix.matrix.clone();
    for row in &mut mat {
        while row.len() < n_phrases { row.push(0.0); }
    }

    let k = 50; // compress to 50 dimensions
    println!("  Computing truncated SVD (k={}, 30 iterations)...", k);
    let word_vectors = truncated_svd(&mat, k, 30);
    println!("  Done. {} word vectors of {} dimensions.\n", word_vectors.len(), k);

    // Word similarity in SVD space
    println!("  WORD SIMILARITY (SVD-compressed):\n");
    let pairs = [
        ("frustrated", "waiting", "Both → escalate_ticket?"),
        ("frustrated", "angry", "Both emotional → escalate?"),
        ("frustrated", "password", "Should be FAR apart"),
        ("broken", "dead", "Both → report_broken?"),
        ("broken", "wifi", "Should be FAR apart"),
        ("laptop", "computer", "Synonyms → same intents?"),
        ("install", "uninstall", "Related but different intents"),
        ("slow", "disconnecting", "Both network issues?"),
        ("access", "permission", "Both → request_access?"),
        ("waiting", "status", "Both → ticket related?"),
    ];
    for (w1, w2, note) in &pairs {
        let i1 = matrix.vocab.get(*w1);
        let i2 = matrix.vocab.get(*w2);
        let sim = match (i1, i2) {
            (Some(&a), Some(&b)) => cosine(&word_vectors[a], &word_vectors[b]),
            _ => -1.0,
        };
        let bar = if sim >= 0.0 { "█".repeat((sim * 20.0) as usize) } else { "?".to_string() };
        println!("    {:<14} ~ {:<14}  sim={:+.3}  {}  {}", w1, w2, sim, bar, note);
    }

    // Build SVD-based intent centroids
    let mut svd_centroids: HashMap<String, Vec<f32>> = HashMap::new();
    let mut intent_counts: HashMap<String, f32> = HashMap::new();
    for (phrase_idx, intent) in matrix.phrase_intents.iter().enumerate() {
        let centroid = svd_centroids.entry(intent.clone())
            .or_insert_with(|| vec![0.0; k]);
        *intent_counts.entry(intent.clone()).or_insert(0.0) += 1.0;
        // Sum word vectors for all words in this phrase
        for (word_idx, row) in mat.iter().enumerate() {
            if row[phrase_idx] > 0.0 {
                for d in 0..k {
                    centroid[d] += word_vectors[word_idx][d];
                }
            }
        }
    }
    // Normalize
    for (intent, centroid) in &mut svd_centroids {
        let count = intent_counts[intent];
        for v in centroid.iter_mut() { *v /= count; }
    }

    // Score queries using SVD vectors
    println!("\n  SVD QUERY SCORING:\n");
    for q in &samples {
        let tokens = tokenize(q);
        let mut qvec = vec![0.0f32; k];
        let mut found = 0;
        for t in &tokens {
            if let Some(&idx) = matrix.vocab.get(t.as_str()) {
                for d in 0..k { qvec[d] += word_vectors[idx][d]; }
                found += 1;
            }
        }

        let mut scores: Vec<(String, f32)> = svd_centroids.iter()
            .map(|(intent, centroid)| (intent.clone(), cosine(&qvec, centroid)))
            .filter(|(_, s)| *s > 0.01)
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top: Vec<String> = scores.iter().take(3)
            .map(|(id, s)| format!("{}={:.3}", short(id), s)).collect();
        let words_found = tokens.iter().filter(|t| matrix.vocab.contains_key(t.as_str())).count();
        println!("    \"{}\"  [{}/{}w]", &q[..q.len().min(55)], words_found, tokens.len());
        println!("      → {}", if top.is_empty() { "NOTHING".to_string() } else { top.join(", ") });
    }

    // Full benchmark with SVD scoring
    println!("\n  SVD BENCHMARK:");
    let mut svd_exact = 0;
    let mut svd_partial = 0;
    let mut svd_fail = 0;
    for (msg, gt) in &all_queries {
        let tokens = tokenize(msg);
        let mut qvec = vec![0.0f32; k];
        for t in &tokens {
            if let Some(&idx) = matrix.vocab.get(t.as_str()) {
                for d in 0..k { qvec[d] += word_vectors[idx][d]; }
            }
        }
        let mut scores: Vec<(String, f32)> = svd_centroids.iter()
            .map(|(intent, centroid)| (intent.clone(), cosine(&qvec, centroid)))
            .filter(|(_, s)| *s > 0.01)
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Multi-intent: keep ≥ 50% of top
        let got: HashSet<String> = if scores.is_empty() { HashSet::new() } else {
            let top = scores[0].1;
            scores.iter().filter(|(_, s)| *s >= top * 0.5)
                .map(|(id, _)| id.clone()).collect()
        };
        let gt_set: HashSet<String> = gt.iter().cloned().collect();
        if got == gt_set { svd_exact += 1; }
        else if !got.is_disjoint(&gt_set) { svd_partial += 1; }
        else { svd_fail += 1; }
    }
    let t = all_queries.len();
    println!("    Exact: {}/{} ({:.0}%) | Partial: {} | Fail: {}\n",
        svd_exact, t, 100.0*svd_exact as f32/t as f32, svd_partial, svd_fail);
}
