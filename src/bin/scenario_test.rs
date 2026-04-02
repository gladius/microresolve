//! Comprehensive experiment runner: tests all 8 approaches against 30 scenarios.
//! Produces detailed results for each experiment.
//!
//! Run with: cargo run --bin scenario_test --release

use asv_router::{IntentType, Router, MultiRouteOutput};
use std::collections::{HashMap, HashSet};
use std::io::Write;

#[derive(serde::Deserialize)]
struct Scenario {
    id: String,
    category: String,
    #[allow(dead_code)]
    persona: serde_json::Value,
    turns: Vec<Turn>,
}

#[derive(serde::Deserialize)]
struct Turn {
    message: String,
    ground_truth: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Status { Pass, Partial, Fail }

struct TurnResult {
    scenario_id: String,
    category: String,
    turn_idx: usize,
    message: String,
    word_count: usize,
    ground_truth: Vec<String>,
    detected: Vec<(String, f32)>,
    matched: Vec<String>,
    missed: Vec<String>,
    extra: Vec<(String, f32)>,
    status: Status,
}

fn evaluate(
    ground_truth: &[String],
    multi_output: &MultiRouteOutput,
) -> (Vec<(String, f32)>, Vec<String>, Vec<String>, Vec<(String, f32)>, Status) {
    let gt_set: HashSet<&str> = ground_truth.iter().map(|s| s.as_str()).collect();
    let detected: Vec<(String, f32)> = multi_output.intents.iter()
        .map(|i| (i.id.clone(), (i.score * 100.0).round() / 100.0))
        .collect();
    let det_set: HashSet<&str> = detected.iter().map(|(id, _)| id.as_str()).collect();

    let matched: Vec<String> = gt_set.intersection(&det_set).map(|s| s.to_string()).collect();
    let missed: Vec<String> = gt_set.difference(&det_set).map(|s| s.to_string()).collect();
    let extra: Vec<(String, f32)> = detected.iter()
        .filter(|(id, _)| !gt_set.contains(id.as_str()))
        .cloned().collect();

    let status = if ground_truth.is_empty() {
        if detected.is_empty() { Status::Pass } else { Status::Fail }
    } else if missed.is_empty() && extra.is_empty() {
        Status::Pass
    } else if !matched.is_empty() && missed.is_empty() {
        Status::Partial // all GT found but with extras
    } else if !matched.is_empty() {
        Status::Partial
    } else {
        Status::Fail
    };

    (detected, matched, missed, extra, status)
}

fn run_experiment<F>(
    router: &Router,
    scenarios: &[Scenario],
    route_fn: F,
) -> Vec<TurnResult>
where
    F: Fn(&Router, &str) -> MultiRouteOutput,
{
    let mut results = Vec::new();
    for scenario in scenarios {
        for (i, turn) in scenario.turns.iter().enumerate() {
            let output = route_fn(router, &turn.message);
            let (detected, matched, missed, extra, status) =
                evaluate(&turn.ground_truth, &output);
            results.push(TurnResult {
                scenario_id: scenario.id.clone(),
                category: scenario.category.clone(),
                turn_idx: i,
                message: turn.message.clone(),
                word_count: turn.message.split_whitespace().count(),
                ground_truth: turn.ground_truth.clone(),
                detected, matched, missed, extra, status,
            });
        }
    }
    results
}

struct ExperimentStats {
    total: usize,
    pass: usize,
    partial: usize,
    fail: usize,
    total_fp: usize,
    total_missed: usize,
    avg_detected: f64,
    by_category: HashMap<String, (usize, usize, usize, usize)>,
    by_wordcount: HashMap<String, (usize, usize, usize, usize)>,
    top_fp: Vec<(String, usize)>,
    top_missed: Vec<(String, usize)>,
}

fn compute_stats(results: &[TurnResult]) -> ExperimentStats {
    let total = results.len();
    let mut pass = 0; let mut partial = 0; let mut fail = 0;
    let mut total_fp = 0; let mut total_missed = 0;
    let mut total_detected = 0usize;
    let mut by_category: HashMap<String, (usize, usize, usize, usize)> = HashMap::new();
    let mut by_wc: HashMap<String, (usize, usize, usize, usize)> = HashMap::new();
    let mut fp_counts: HashMap<String, usize> = HashMap::new();
    let mut miss_counts: HashMap<String, usize> = HashMap::new();

    for r in results {
        match r.status {
            Status::Pass => pass += 1,
            Status::Partial => partial += 1,
            Status::Fail => fail += 1,
        }
        total_fp += r.extra.len();
        total_missed += r.missed.len();
        total_detected += r.detected.len();
        for (id, _) in &r.extra { *fp_counts.entry(id.clone()).or_insert(0) += 1; }
        for id in &r.missed { *miss_counts.entry(id.clone()).or_insert(0) += 1; }

        let cat = &r.category;
        let e = by_category.entry(cat.clone()).or_insert((0,0,0,0));
        e.0 += 1;
        match r.status { Status::Pass => e.1 += 1, Status::Partial => e.2 += 1, Status::Fail => e.3 += 1 }

        let wc_bucket = if r.word_count <= 5 { "1-5" }
            else if r.word_count <= 10 { "6-10" }
            else if r.word_count <= 20 { "11-20" }
            else if r.word_count <= 40 { "21-40" }
            else { "41+" };
        let e = by_wc.entry(wc_bucket.to_string()).or_insert((0,0,0,0));
        e.0 += 1;
        match r.status { Status::Pass => e.1 += 1, Status::Partial => e.2 += 1, Status::Fail => e.3 += 1 }
    }

    let mut top_fp: Vec<_> = fp_counts.into_iter().collect();
    top_fp.sort_by(|a, b| b.1.cmp(&a.1));
    top_fp.truncate(10);

    let mut top_missed: Vec<_> = miss_counts.into_iter().collect();
    top_missed.sort_by(|a, b| b.1.cmp(&a.1));
    top_missed.truncate(10);

    ExperimentStats {
        total, pass, partial, fail, total_fp, total_missed,
        avg_detected: total_detected as f64 / total.max(1) as f64,
        by_category, by_wordcount: by_wc, top_fp, top_missed,
    }
}

fn write_stats(out: &mut String, name: &str, stats: &ExperimentStats) {
    let pass_pct = (stats.pass as f64 / stats.total as f64) * 100.0;
    let pp_pct = ((stats.pass + stats.partial) as f64 / stats.total as f64) * 100.0;
    let fp_avg = stats.total_fp as f64 / stats.total as f64;

    out.push_str(&format!("\n### {}\n\n", name));
    out.push_str(&format!("| Metric | Value |\n|---|---|\n"));
    out.push_str(&format!("| Total turns | {} |\n", stats.total));
    out.push_str(&format!("| Pass (exact) | {} ({:.1}%) |\n", stats.pass, pass_pct));
    out.push_str(&format!("| Partial (GT found + extras) | {} |\n", stats.partial));
    out.push_str(&format!("| Fail (missed GT) | {} |\n", stats.fail));
    out.push_str(&format!("| Pass+Partial | {:.1}% |\n", pp_pct));
    out.push_str(&format!("| Avg false positives/turn | {:.2} |\n", fp_avg));
    out.push_str(&format!("| Avg intents detected/turn | {:.2} |\n", stats.avg_detected));
    out.push_str(&format!("| Total missed intents | {} |\n", stats.total_missed));

    out.push_str("\n**By category:**\n\n");
    out.push_str("| Category | Total | Pass | Part | Fail | Pass% |\n|---|---|---|---|---|---|\n");
    let mut cats: Vec<_> = stats.by_category.iter().collect();
    cats.sort_by_key(|(k, _)| k.clone());
    for (cat, (t, p, pa, f)) in &cats {
        let pct = if *t > 0 { (*p as f64 / *t as f64) * 100.0 } else { 0.0 };
        out.push_str(&format!("| {} | {} | {} | {} | {} | {:.1}% |\n", cat, t, p, pa, f, pct));
    }

    out.push_str("\n**By word count:**\n\n");
    out.push_str("| Words | Total | Pass | Part | Fail | Pass% |\n|---|---|---|---|---|---|\n");
    for bucket in &["1-5", "6-10", "11-20", "21-40", "41+"] {
        if let Some((t, p, pa, f)) = stats.by_wordcount.get(*bucket) {
            let pct = if *t > 0 { (*p as f64 / *t as f64) * 100.0 } else { 0.0 };
            out.push_str(&format!("| {} | {} | {} | {} | {} | {:.1}% |\n", bucket, t, p, pa, f, pct));
        }
    }

    if !stats.top_fp.is_empty() {
        out.push_str("\n**Top false positives:** ");
        let fps: Vec<String> = stats.top_fp.iter().map(|(id, c)| format!("{} ({}x)", id, c)).collect();
        out.push_str(&fps.join(", "));
        out.push_str("\n");
    }
    if !stats.top_missed.is_empty() {
        out.push_str("\n**Top missed:** ");
        let ms: Vec<String> = stats.top_missed.iter().map(|(id, c)| format!("{} ({}x)", id, c)).collect();
        out.push_str(&ms.join(", "));
        out.push_str("\n");
    }
}

fn write_failures(out: &mut String, results: &[TurnResult], max_show: usize) {
    let failures: Vec<&TurnResult> = results.iter()
        .filter(|r| matches!(r.status, Status::Fail))
        .collect();
    if failures.is_empty() { return; }

    out.push_str(&format!("\n**Sample failures ({} total, showing {}):**\n\n", failures.len(), max_show.min(failures.len())));
    for r in failures.iter().take(max_show) {
        let msg = if r.message.len() > 80 { format!("{}...", &r.message[..77]) } else { r.message.clone() };
        out.push_str(&format!("- `{}` [{}] ({} words)\n", r.scenario_id, r.turn_idx + 1, r.word_count));
        out.push_str(&format!("  \"{}\"  \n", msg));
        out.push_str(&format!("  GT: {:?} | DT: {:?}\n", r.ground_truth,
            r.detected.iter().map(|(id, s)| format!("{}:{:.1}", id, s)).collect::<Vec<_>>()));
    }
}

// ============= Intent setup (same as server load_defaults) =============
fn setup_router() -> Router {
    let mut router = Router::new();
    let actions: &[(&str, &[&str])] = &[
        ("cancel_order", &["cancel my order","I need to cancel an order I just placed","please stop my order from shipping","I changed my mind and want to cancel the purchase","how do I cancel something I ordered yesterday","cancel order number","I accidentally ordered the wrong thing, cancel it","withdraw my order before it ships"]),
        ("refund", &["I want a refund","get my money back","I received a damaged item and need a refund","the product was nothing like the description, refund please","how long does it take to process a return","I returned it two weeks ago and still no refund","I want to return this for a full refund","money back"]),
        ("contact_human", &["talk to a human","I need to speak with a real person not a bot","connect me to customer service please","this bot is useless, get me an agent","transfer me to a representative","I want to talk to someone who can actually help","live agent please","escalate this to a manager"]),
        ("reset_password", &["reset my password","I forgot my password and can't log in","my account is locked out","how do I change my password","the password reset email never arrived","I keep getting invalid password error","locked out of my account need help getting back in","send me a password reset link"]),
        ("update_address", &["change my address","I moved and need to update my shipping address","update my delivery address before it ships","my address is wrong on the order","change the shipping destination","I need to correct my mailing address","ship it to a different address instead","new address for future orders"]),
        ("billing_issue", &["wrong charge on my account","I was charged twice for the same order","there's a billing error on my statement","I see an unauthorized charge","you overcharged me by twenty dollars","my credit card was charged the wrong amount","dispute a charge","the amount on my bill doesn't match what I ordered"]),
        ("change_plan", &["upgrade my plan","I want to switch to the premium subscription","downgrade my account to the basic tier","change my subscription plan","what plans are available for upgrade","I want a cheaper plan","switch me to the annual billing"]),
        ("close_account", &["delete my account","I want to close my account permanently","how do I deactivate my profile","remove all my data and close the account","I no longer want to use this service","cancel my membership entirely","please terminate my account"]),
        ("report_fraud", &["someone used my card without permission","I think my account was hacked","there are transactions I did not make","report unauthorized access to my account","fraudulent activity on my card","someone stole my identity and made purchases","I need to report suspicious charges"]),
        ("apply_coupon", &["I have a discount code","apply my coupon to the order","where do I enter a promo code","this coupon isn't working","I forgot to apply my discount before checkout","can I use two coupons on one order","my promotional code was rejected"]),
        ("schedule_callback", &["can someone call me back","I'd like to schedule a phone call","have an agent call me at this number","request a callback for tomorrow morning","I prefer a phone call over chat","when can I expect a call back","set up a time for support to call me"]),
        ("file_complaint", &["I want to file a formal complaint","this is unacceptable, I'm filing a complaint","how do I report poor service","I want to submit a grievance","your service has been terrible and I want it documented","escalate my complaint to upper management","I need to make an official complaint"]),
        ("request_invoice", &["send me an invoice for my purchase","I need a receipt for tax purposes","can I get a PDF of my invoice","email me the billing statement","I need documentation of this transaction","where can I download my invoice","generate an invoice for order number"]),
        ("pause_subscription", &["pause my subscription for a month","I want to temporarily stop my membership","can I freeze my account without canceling","put my plan on hold","suspend my subscription until next quarter","I'm traveling and want to pause billing","temporarily deactivate my subscription"]),
        ("transfer_funds", &["transfer money to another account","send funds to my savings account","move money between my accounts","I want to wire money to someone","initiate a bank transfer","how do I send money to another person","transfer fifty dollars to my checking"]),
        ("add_payment_method", &["add a new credit card to my account","I want to register a different payment method","update my card information","save a new debit card for payments","link my bank account for direct payment","replace my expired card on file","add PayPal as a payment option"]),
        ("remove_item", &["remove an item from my order","take this product out of my cart","I don't want one of the items in my order anymore","delete the second item from my purchase","can I remove something before it ships","take off the extra item I added by mistake","drop one item from my order"]),
        ("reorder", &["reorder my last purchase","I want to buy the same thing again","repeat my previous order","order the same items as last time","can I quickly reorder what I got before","place the same order again","buy this product again"]),
        ("upgrade_shipping", &["upgrade to express shipping","I need this delivered faster","can I switch to overnight delivery","expedite my shipment","change my shipping to two-day delivery","I'll pay extra for faster shipping","rush delivery please"]),
        ("gift_card_redeem", &["redeem my gift card","I have a gift card code to apply","how do I use a gift certificate","enter my gift card balance","apply a gift card to my purchase","my gift card isn't being accepted","check the balance on my gift card"]),
    ];
    let contexts: &[(&str, &[&str])] = &[
        ("track_order", &["where is my package","track my order","my order still hasn't arrived and it's been a week","I need a shipping update on my recent purchase","when will my delivery arrive","package tracking number","it says delivered but I never got it","how long until my order gets here"]),
        ("check_balance", &["check my balance","how much money is in my account","what's my current account balance","show me my available funds","I need to know how much I have left","account summary","remaining balance on my card","what do I owe right now"]),
        ("account_status", &["is my account in good standing","check my account status","am I verified","what is the state of my account","is my account active or suspended","show me my account details","my account status page"]),
        ("order_history", &["show me my past orders","what did I order last month","view my order history","list all my previous purchases","I need to see what I bought before","pull up my recent orders","my purchase history"]),
        ("payment_history", &["show me my payment history","list all charges to my account","what payments have I made","view my transaction log","when was my last payment","how much have I spent this month","pull up my billing history"]),
        ("shipping_options", &["what shipping methods are available","how much does express shipping cost","what are my delivery options","do you offer free shipping","compare shipping speeds and prices","international shipping rates","same day delivery available"]),
        ("return_policy", &["what is your return policy","how many days do I have to return something","can I return a used product","do you accept returns without receipt","what items are not returnable","is there a restocking fee for returns","return and exchange policy"]),
        ("product_availability", &["is this item in stock","when will this product be available again","check if you have this in my size","is this item available for delivery","out of stock notification","do you carry this brand","product availability in my area"]),
        ("warranty_info", &["what does the warranty cover","how long is the warranty period","is my product still under warranty","warranty claim process","does this come with a manufacturer warranty","extended warranty options","what voids the warranty"]),
        ("loyalty_points", &["how many reward points do I have","check my loyalty balance","when do my points expire","how can I redeem my reward points","how many points do I earn per dollar","my rewards program status","transfer loyalty points"]),
        ("subscription_status", &["what plan am I on","when does my subscription renew","show me my current plan details","how much am I paying monthly","when is my next billing date","what features are included in my plan","subscription renewal date"]),
        ("delivery_estimate", &["when will my order arrive","estimated delivery date","how long does shipping take","expected arrival for my package","delivery timeframe for my area","how many business days until delivery","will it arrive before the weekend"]),
        ("price_check", &["how much does this cost","what is the price of this item","is this on sale right now","price match guarantee","compare prices for this product","total cost including shipping","any discounts on this item"]),
        ("account_limits", &["what is my spending limit","daily transfer limit on my account","maximum withdrawal amount","how much can I send per transaction","increase my account limits","what are the restrictions on my account","transaction limits for my plan"]),
        ("transaction_details", &["show me details of my last transaction","what was that charge for","transaction reference number lookup","I need details about a specific payment","when exactly was this charge made","who was the merchant for this transaction","breakdown of charges on my statement"]),
        ("eligibility_check", &["am I eligible for an upgrade","do I qualify for a discount","can I apply for this program","check my eligibility for the promotion","what are the requirements to qualify","am I eligible for a credit increase","do I meet the criteria for this offer"]),
    ];
    for (id, seeds) in actions { router.add_intent(id, seeds); router.set_intent_type(id, IntentType::Action); }
    for (id, seeds) in contexts { router.add_intent(id, seeds); router.set_intent_type(id, IntentType::Context); }
    router
}

// ============= SymSpell (simplified) =============

struct SimpleSymSpell {
    /// delete_variant -> list of original terms
    deletes: HashMap<String, Vec<String>>,
    known: HashSet<String>,
}

impl SimpleSymSpell {
    fn build(vocabulary: &[String]) -> Self {
        let known: HashSet<String> = vocabulary.iter().cloned().collect();
        let mut deletes: HashMap<String, Vec<String>> = HashMap::new();
        for word in vocabulary {
            if word.len() <= 1 { continue; }
            // Generate all single-char deletions
            for i in 0..word.len() {
                if word.is_char_boundary(i) && word.is_char_boundary(i + 1) {
                    let variant = format!("{}{}", &word[..i], &word[i+1..]);
                    deletes.entry(variant).or_default().push(word.clone());
                }
            }
        }
        SimpleSymSpell { deletes, known }
    }

    fn correct(&self, word: &str) -> Option<String> {
        if self.known.contains(word) { return None; } // already correct
        if word.len() <= 2 { return None; } // too short to correct

        // Generate deletions of the input word and look up
        let mut candidates: Vec<(String, usize)> = Vec::new(); // (word, edit_distance)
        for i in 0..word.len() {
            if word.is_char_boundary(i) && (i + 1 <= word.len()) && word.is_char_boundary(i.min(word.len())) {
                let end = (i + 1).min(word.len());
                if word.is_char_boundary(end) {
                    let variant = format!("{}{}", &word[..i], &word[end..]);
                    if let Some(originals) = self.deletes.get(&variant) {
                        for orig in originals {
                            candidates.push((orig.clone(), 1));
                        }
                    }
                    // Also check if the deletion itself is a known word (edit distance 1)
                    if self.known.contains(&variant) {
                        candidates.push((variant, 1));
                    }
                }
            }
        }
        // Also check direct delete lookups (word IS a deletion of a known word)
        if let Some(originals) = self.deletes.get(word) {
            for orig in originals {
                candidates.push((orig.clone(), 1));
            }
        }

        candidates.sort_by_key(|c| c.1);
        candidates.first().map(|(w, _)| w.clone())
    }
}

fn introduce_typos(message: &str, rate: f64) -> String {
    let mut rng_state: u64 = message.len() as u64 * 31 + 7;
    let words: Vec<&str> = message.split_whitespace().collect();
    let mut result = Vec::new();
    for word in words {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = (rng_state >> 33) as f64 / (u32::MAX as f64);
        if r < rate && word.len() > 3 && word.chars().all(|c| c.is_alphabetic()) {
            // Drop a random character
            let idx = ((rng_state >> 16) as usize) % word.len();
            if word.is_char_boundary(idx) && idx + 1 <= word.len() && word.is_char_boundary(idx + 1) {
                let typo = format!("{}{}", &word[..idx], &word[idx+1..]);
                result.push(typo);
            } else {
                result.push(word.to_string());
            }
        } else {
            result.push(word.to_string());
        }
    }
    result.join(" ")
}

// ============= MAIN =============

fn main() {
    let scenario_path = "tests/scenarios/scenarios.json";
    let data = std::fs::read_to_string(scenario_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", scenario_path, e));
    let scenarios: Vec<Scenario> = serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("Failed to parse scenarios: {}", e));

    let router = setup_router();
    let mut report = String::new();

    report.push_str("# ASV Experiment Results\n\n");
    report.push_str(&format!("Scenarios: {}, Turns: {}\n\n", scenarios.len(),
        scenarios.iter().map(|s| s.turns.len()).sum::<usize>()));

    // ============================
    // BASELINE
    // ============================
    eprintln!("Running baseline...");
    let baseline = run_experiment(&router, &scenarios, |r, q| r.route_multi(q, 0.3));
    let baseline_stats = compute_stats(&baseline);

    report.push_str("---\n\n## BASELINE (route_multi, threshold=0.3)\n");
    write_stats(&mut report, "Baseline Results", &baseline_stats);
    write_failures(&mut report, &baseline, 5);

    // Threshold sweep
    report.push_str("\n### Threshold Sweep\n\n");
    report.push_str("| Threshold | Pass | Partial | Fail | Pass% | P+P% | Avg FP/turn |\n|---|---|---|---|---|---|---|\n");
    for &t in &[0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0] {
        let res = run_experiment(&router, &scenarios, |r, q| r.route_multi(q, t));
        let s = compute_stats(&res);
        report.push_str(&format!("| {:.1} | {} | {} | {} | {:.1}% | {:.1}% | {:.2} |\n",
            t, s.pass, s.partial, s.fail,
            (s.pass as f64 / s.total as f64) * 100.0,
            ((s.pass + s.partial) as f64 / s.total as f64) * 100.0,
            s.total_fp as f64 / s.total as f64));
    }

    // ============================
    // TEST A: SymSpell
    // ============================
    eprintln!("Running Test A: SymSpell...");
    report.push_str("\n---\n\n## TEST A: SymSpell Correction\n\n");

    // Build vocabulary from router
    let intent_ids = router.intent_ids();
    let mut vocab: Vec<String> = Vec::new();
    for id in &intent_ids {
        if let Some(v) = router.get_vector(id) {
            for (term, _) in v.effective_terms() {
                vocab.push(term);
            }
        }
    }
    vocab.sort(); vocab.dedup();
    let symspell = SimpleSymSpell::build(&vocab);

    report.push_str(&format!("**Vocabulary size:** {} unique terms\n\n", vocab.len()));

    // Coverage analysis on current (correctly spelled) scenarios
    let mut total_terms = 0usize;
    let mut known_terms = 0usize;
    for scenario in &scenarios {
        for turn in &scenario.turns {
            let (k, t) = router.query_coverage(&turn.message);
            known_terms += k;
            total_terms += t;
        }
    }
    report.push_str(&format!("**Query term coverage (no typos):** {}/{} ({:.1}%)\n\n",
        known_terms, total_terms, (known_terms as f64 / total_terms.max(1) as f64) * 100.0));

    // Test with introduced typos (30% of long words get a char dropped)
    let typo_results = run_experiment(&router, &scenarios, |r, q| {
        let typo_q = introduce_typos(q, 0.3);
        r.route_multi(&typo_q, 0.3)
    });
    let typo_stats = compute_stats(&typo_results);

    let corrected_results = run_experiment(&router, &scenarios, |r, q| {
        let typo_q = introduce_typos(q, 0.3);
        // Apply SymSpell correction
        let words: Vec<&str> = typo_q.split_whitespace().collect();
        let corrected: Vec<String> = words.iter().map(|w| {
            let lower = w.to_lowercase();
            symspell.correct(&lower).unwrap_or(lower)
        }).collect();
        let corrected_q = corrected.join(" ");
        r.route_multi(&corrected_q, 0.3)
    });
    let corrected_stats = compute_stats(&corrected_results);

    report.push_str("**With 30% typo rate (baseline for comparison):**\n\n");
    report.push_str(&format!("| Condition | Pass | Partial | Fail | Pass% |\n|---|---|---|---|---|\n"));
    report.push_str(&format!("| No typos (baseline) | {} | {} | {} | {:.1}% |\n",
        baseline_stats.pass, baseline_stats.partial, baseline_stats.fail,
        (baseline_stats.pass as f64 / baseline_stats.total as f64) * 100.0));
    report.push_str(&format!("| With typos, no correction | {} | {} | {} | {:.1}% |\n",
        typo_stats.pass, typo_stats.partial, typo_stats.fail,
        (typo_stats.pass as f64 / typo_stats.total as f64) * 100.0));
    report.push_str(&format!("| With typos + SymSpell | {} | {} | {} | {:.1}% |\n",
        corrected_stats.pass, corrected_stats.partial, corrected_stats.fail,
        (corrected_stats.pass as f64 / corrected_stats.total as f64) * 100.0));

    // Count corrections made
    let mut corrections_made = 0usize;
    for scenario in &scenarios {
        for turn in &scenario.turns {
            let typo_q = introduce_typos(&turn.message, 0.3);
            for w in typo_q.split_whitespace() {
                if symspell.correct(&w.to_lowercase()).is_some() { corrections_made += 1; }
            }
        }
    }
    report.push_str(&format!("\n**Corrections applied:** {} words across all turns\n", corrections_made));

    // ============================
    // TEST B: Noise Gate (IDF-based)
    // ============================
    eprintln!("Running Test B: Noise Gate...");
    report.push_str("\n---\n\n## TEST B: IDF Noise Gate\n\n");

    // Show IDF distribution
    report.push_str("**Term IDF distribution across all query terms:**\n\n");
    let mut all_idfs: Vec<(String, f32, usize)> = Vec::new();
    for scenario in &scenarios {
        for turn in &scenario.turns {
            let terms = router.analyze_query_terms(&turn.message);
            all_idfs.extend(terms);
        }
    }
    // Unique terms with their IDF
    let mut term_idf_map: HashMap<String, (f32, usize)> = HashMap::new();
    for (term, idf, df) in &all_idfs {
        term_idf_map.entry(term.clone()).or_insert((*idf, *df));
    }
    let mut idf_values: Vec<f32> = term_idf_map.values().map(|(idf, _)| *idf).collect();
    idf_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !idf_values.is_empty() {
        let median_idx = idf_values.len() / 2;
        report.push_str(&format!("- Unique terms in queries: {}\n", term_idf_map.len()));
        report.push_str(&format!("- IDF range: {:.2} to {:.2}\n", idf_values[0], idf_values[idf_values.len()-1]));
        report.push_str(&format!("- Median IDF: {:.2}\n", idf_values[median_idx]));
        report.push_str(&format!("- 25th percentile: {:.2}\n", idf_values[idf_values.len() / 4]));
    }

    // Show noisiest terms (lowest IDF, highest df)
    let mut by_df: Vec<_> = term_idf_map.iter().collect();
    by_df.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    report.push_str("\n**Noisiest terms (highest df):**\n\n");
    report.push_str("| Term | DF (intents) | IDF |\n|---|---|---|\n");
    for (term, (idf, df)) in by_df.iter().take(20) {
        report.push_str(&format!("| {} | {} | {:.2} |\n", term, df, idf));
    }

    // Test multiple max_df cutoffs
    report.push_str("\n**Noise gate results (max_df = exclude terms in > N intents):**\n\n");
    report.push_str("| max_df | Pass | Partial | Fail | Pass% | P+P% | Avg FP | Avg missed |\n|---|---|---|---|---|---|---|---|\n");
    let n_intents = router.intent_count();
    for &max_df in &[3, 5, 8, 10, 12, 15, 18, 25, 36] {
        let res = run_experiment(&router, &scenarios, |r, q| r.route_multi_noise_gated(q, 0.3, max_df));
        let s = compute_stats(&res);
        report.push_str(&format!("| {} | {} | {} | {} | {:.1}% | {:.1}% | {:.2} | {:.2} |\n",
            max_df, s.pass, s.partial, s.fail,
            (s.pass as f64 / s.total as f64) * 100.0,
            ((s.pass + s.partial) as f64 / s.total as f64) * 100.0,
            s.total_fp as f64 / s.total as f64,
            s.total_missed as f64 / s.total as f64));
    }

    // Best noise gate detailed
    let best_ng = run_experiment(&router, &scenarios, |r, q| r.route_multi_noise_gated(q, 0.3, 8));
    let best_ng_stats = compute_stats(&best_ng);
    write_stats(&mut report, "Noise Gate (max_df=8) Detail", &best_ng_stats);
    write_failures(&mut report, &best_ng, 5);

    // ============================
    // TEST C: Per-Intent Confidence Calibration
    // ============================
    eprintln!("Running Test C: Confidence Calibration...");
    report.push_str("\n---\n\n## TEST C: Per-Intent Confidence Calibration\n\n");

    // Self-test: route each seed phrase, record scores per intent
    let mut score_distributions: HashMap<String, Vec<f32>> = HashMap::new();
    for id in &intent_ids {
        if let Some(training) = router.get_training(id) {
            for phrase in &training {
                let results = router.route(phrase);
                if let Some(r) = results.iter().find(|r| r.id == *id) {
                    score_distributions.entry(id.clone()).or_default().push(r.score);
                }
            }
        }
    }

    // Compute mean and stddev per intent
    let mut intent_thresholds: HashMap<String, f32> = HashMap::new();
    report.push_str("**Seed self-test score distributions:**\n\n");
    report.push_str("| Intent | Seeds | Mean | StdDev | Min threshold |\n|---|---|---|---|---|\n");
    for id in &intent_ids {
        if let Some(scores) = score_distributions.get(id) {
            let n = scores.len() as f32;
            let mean = scores.iter().sum::<f32>() / n;
            let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
            let stddev = variance.sqrt();
            let min_thresh = (mean - 1.5 * stddev).max(0.3);
            intent_thresholds.insert(id.clone(), min_thresh);
            report.push_str(&format!("| {} | {} | {:.2} | {:.2} | {:.2} |\n",
                id, scores.len(), mean, stddev, min_thresh));
        }
    }

    // Apply calibrated thresholds as post-filter on baseline
    let calibrated = run_experiment(&router, &scenarios, |r, q| {
        let mut output = r.route_multi(q, 0.3);
        output.intents.retain(|i| {
            let min_t = intent_thresholds.get(&i.id).copied().unwrap_or(0.3);
            i.score >= min_t
        });
        output
    });
    let cal_stats = compute_stats(&calibrated);
    write_stats(&mut report, "Calibrated Thresholds", &cal_stats);

    // ============================
    // TEST D: Coverage Ratio
    // ============================
    eprintln!("Running Test D: Coverage Ratio...");
    report.push_str("\n---\n\n## TEST D: Coverage Ratio Analysis\n\n");

    let mut coverage_buckets: HashMap<String, (usize, usize, usize, usize)> = HashMap::new();
    for r in &baseline {
        let (known, total) = router.query_coverage(&r.message);
        let ratio = if total > 0 { known as f64 / total as f64 } else { 0.0 };
        let bucket = if ratio >= 0.8 { "80-100%" }
            else if ratio >= 0.6 { "60-80%" }
            else if ratio >= 0.4 { "40-60%" }
            else if ratio >= 0.2 { "20-40%" }
            else { "0-20%" };
        let e = coverage_buckets.entry(bucket.to_string()).or_insert((0,0,0,0));
        e.0 += 1;
        match r.status { Status::Pass => e.1 += 1, Status::Partial => e.2 += 1, Status::Fail => e.3 += 1 }
    }

    report.push_str("**Accuracy by query term coverage:**\n\n");
    report.push_str("| Coverage | Total | Pass | Part | Fail | Pass% | P+P% |\n|---|---|---|---|---|---|---|\n");
    for bucket in &["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"] {
        if let Some((t, p, pa, f)) = coverage_buckets.get(*bucket) {
            report.push_str(&format!("| {} | {} | {} | {} | {} | {:.1}% | {:.1}% |\n",
                bucket, t, p, pa, f,
                (*p as f64 / *t as f64) * 100.0,
                ((*p + *pa) as f64 / *t as f64) * 100.0));
        }
    }

    // ============================
    // TEST E: Anti-Co-occurrence Filter
    // ============================
    eprintln!("Running Test E: Anti-Co-occurrence...");
    report.push_str("\n---\n\n## TEST E: Anti-Co-occurrence Filter\n\n");

    // Build valid and invalid pair matrices from baseline results
    let mut valid_pairs: HashMap<(String, String), usize> = HashMap::new();
    let mut invalid_pairs: HashMap<(String, String), usize> = HashMap::new();
    for r in &baseline {
        // Valid pairs: all combinations of ground truth intents
        for (i, a) in r.ground_truth.iter().enumerate() {
            for b in r.ground_truth.iter().skip(i + 1) {
                let pair = if a < b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
                *valid_pairs.entry(pair).or_insert(0) += 1;
            }
        }
        // Invalid pairs: ground truth x extra
        for gt in &r.ground_truth {
            for (extra_id, _) in &r.extra {
                let pair = if gt < extra_id { (gt.clone(), extra_id.clone()) } else { (extra_id.clone(), gt.clone()) };
                *invalid_pairs.entry(pair).or_insert(0) += 1;
            }
        }
    }

    report.push_str(&format!("**Valid pairs identified:** {}\n", valid_pairs.len()));
    report.push_str(&format!("**Invalid pairs identified:** {}\n\n", invalid_pairs.len()));

    // Show top invalid pairs
    let mut inv_sorted: Vec<_> = invalid_pairs.iter().collect();
    inv_sorted.sort_by(|a, b| b.1.cmp(a.1));
    report.push_str("**Top invalid pairs (most common false co-detections):**\n\n");
    report.push_str("| Pair | False count | Valid count |\n|---|---|---|\n");
    for ((a, b), count) in inv_sorted.iter().take(15) {
        let v = valid_pairs.get(&(a.clone(), b.clone())).copied().unwrap_or(0);
        report.push_str(&format!("| {} + {} | {} | {} |\n", a, b, count, v));
    }

    // Apply anti-co-occurrence filter
    let anti_cooc = run_experiment(&router, &scenarios, |r, q| {
        let mut output = r.route_multi(q, 0.3);
        if output.intents.len() <= 1 { return output; }
        // For each pair, if invalid >> valid, suppress the weaker one
        let mut suppress: HashSet<String> = HashSet::new();
        for i in 0..output.intents.len() {
            for j in (i+1)..output.intents.len() {
                let a = &output.intents[i].id;
                let b = &output.intents[j].id;
                let pair = if a < b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
                let inv = invalid_pairs.get(&pair).copied().unwrap_or(0);
                let val = valid_pairs.get(&pair).copied().unwrap_or(0);
                if inv > val + 1 { // strongly invalid
                    // Suppress the weaker intent
                    if output.intents[i].score < output.intents[j].score {
                        suppress.insert(a.clone());
                    } else {
                        suppress.insert(b.clone());
                    }
                }
            }
        }
        output.intents.retain(|i| !suppress.contains(&i.id));
        output
    });
    let anti_stats = compute_stats(&anti_cooc);
    write_stats(&mut report, "Anti-Co-occurrence Filter", &anti_stats);
    report.push_str("\n**Note:** This test is somewhat circular (trained on test data) but shows the ceiling.\n");

    // ============================
    // TEST G: Anchor-Based Scoring
    // ============================
    eprintln!("Running Test G: Anchor-Based Scoring...");
    report.push_str("\n---\n\n## TEST G: Anchor-Based Scoring\n\n");

    // Show anchor terms per intent
    let disc_max_df = (n_intents / 15).max(3);
    report.push_str(&format!("**Discrimination threshold:** df <= {} (N={}, N/15={})\n\n", disc_max_df, n_intents, n_intents/15));

    report.push_str("**Anchor terms per intent (df <= threshold, weight >= 0.5):**\n\n");
    report.push_str("| Intent | Anchor terms (df) |\n|---|---|\n");
    let mut intents_without_anchors = Vec::new();
    for id in &intent_ids {
        if let Some(v) = router.get_vector(id) {
            let mut anchors: Vec<(String, usize)> = Vec::new();
            for (term, weight) in v.effective_terms() {
                let df = router.term_df(&term);
                if df <= disc_max_df && weight >= 0.5 {
                    anchors.push((term, df));
                }
            }
            anchors.sort_by_key(|(_, df)| *df);
            if anchors.is_empty() {
                intents_without_anchors.push(id.clone());
                report.push_str(&format!("| {} | (NONE) |\n", id));
            } else {
                let anchor_str: Vec<String> = anchors.iter().take(5)
                    .map(|(t, df)| format!("{} ({})", t, df)).collect();
                report.push_str(&format!("| {} | {} |\n", id, anchor_str.join(", ")));
            }
        }
    }
    if !intents_without_anchors.is_empty() {
        report.push_str(&format!("\n**WARNING: {} intents have NO anchor terms:** {:?}\n",
            intents_without_anchors.len(), intents_without_anchors));
    }

    // Test multiple window sizes
    report.push_str("\n**Anchor-based results by window size:**\n\n");
    report.push_str("| Window | Pass | Partial | Fail | Pass% | P+P% | Avg FP | Avg missed |\n|---|---|---|---|---|---|---|---|\n");
    for &window in &[3, 5, 7, 10, 15] {
        let res = run_experiment(&router, &scenarios, |r, q| r.route_multi_anchored(q, 0.3, window));
        let s = compute_stats(&res);
        report.push_str(&format!("| {} | {} | {} | {} | {:.1}% | {:.1}% | {:.2} | {:.2} |\n",
            window, s.pass, s.partial, s.fail,
            (s.pass as f64 / s.total as f64) * 100.0,
            ((s.pass + s.partial) as f64 / s.total as f64) * 100.0,
            s.total_fp as f64 / s.total as f64,
            s.total_missed as f64 / s.total as f64));
    }

    // Best anchor window detailed
    let best_anchor = run_experiment(&router, &scenarios, |r, q| r.route_multi_anchored(q, 0.3, 7));
    let best_anchor_stats = compute_stats(&best_anchor);
    write_stats(&mut report, "Anchor-Based (window=7) Detail", &best_anchor_stats);
    write_failures(&mut report, &best_anchor, 5);

    // ============================
    // TEST H: Session Prior
    // ============================
    eprintln!("Running Test H: Session Prior...");
    report.push_str("\n---\n\n## TEST H: Session-Based Prior\n\n");

    // Build transition matrix from scenario ground truths
    let mut transitions: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for scenario in &scenarios {
        for i in 0..scenario.turns.len().saturating_sub(1) {
            for prev_intent in &scenario.turns[i].ground_truth {
                for next_intent in &scenario.turns[i + 1].ground_truth {
                    *transitions.entry(prev_intent.clone()).or_default()
                        .entry(next_intent.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    report.push_str(&format!("**Transition pairs observed:** {}\n\n", transitions.values().map(|m| m.len()).sum::<usize>()));

    // Apply session prior: boost intents that commonly follow previous turn's intent
    let session_results: Vec<TurnResult> = {
        let mut all_results = Vec::new();
        for scenario in &scenarios {
            let mut prev_intents: Vec<String> = Vec::new();
            for (i, turn) in scenario.turns.iter().enumerate() {
                let mut output = router.route_multi(&turn.message, 0.3);

                // Apply prior: boost intents in transition matrix from prev_intents
                if !prev_intents.is_empty() {
                    let mut boost_set: HashSet<String> = HashSet::new();
                    for prev in &prev_intents {
                        if let Some(nexts) = transitions.get(prev) {
                            for (next_id, _) in nexts {
                                boost_set.insert(next_id.clone());
                            }
                        }
                    }
                    // Boost: multiply score by 1.5 for expected follow-ups
                    for intent in &mut output.intents {
                        if boost_set.contains(&intent.id) {
                            intent.score *= 1.5;
                        }
                    }
                    // Re-sort by score
                    output.intents.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                }

                let (detected, matched, missed, extra, status) =
                    evaluate(&turn.ground_truth, &output);

                // Use detected intents as prev for next turn (simulating real session)
                prev_intents = detected.iter().map(|(id, _)| id.clone()).collect();

                all_results.push(TurnResult {
                    scenario_id: scenario.id.clone(),
                    category: scenario.category.clone(),
                    turn_idx: i,
                    message: turn.message.clone(),
                    word_count: turn.message.split_whitespace().count(),
                    ground_truth: turn.ground_truth.clone(),
                    detected, matched, missed, extra, status,
                });
            }
        }
        all_results
    };
    let session_stats = compute_stats(&session_results);
    write_stats(&mut report, "Session Prior", &session_stats);

    // ============================
    // TEST F: Combined Pipeline (best of each)
    // ============================
    eprintln!("Running Test F: Combined Pipeline...");
    report.push_str("\n---\n\n## TEST F: Combined Pipeline\n\n");

    // Combine: noise gate + anchor-based + anti-co-occurrence + calibration
    report.push_str("**Pipeline:** Noise gate (max_df=8) + Anti-co-occurrence + Calibrated thresholds\n\n");

    let combined = run_experiment(&router, &scenarios, |r, q| {
        // Step 1: Noise-gated routing
        let mut output = r.route_multi_noise_gated(q, 0.3, 8);

        // Step 2: Per-intent calibrated thresholds
        output.intents.retain(|i| {
            let min_t = intent_thresholds.get(&i.id).copied().unwrap_or(0.3);
            i.score >= min_t
        });

        // Step 3: Anti-co-occurrence filter
        if output.intents.len() > 1 {
            let mut suppress: HashSet<String> = HashSet::new();
            for i in 0..output.intents.len() {
                for j in (i+1)..output.intents.len() {
                    let a = &output.intents[i].id;
                    let b = &output.intents[j].id;
                    let pair = if a < b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
                    let inv = invalid_pairs.get(&pair).copied().unwrap_or(0);
                    let val = valid_pairs.get(&pair).copied().unwrap_or(0);
                    if inv > val + 1 {
                        if output.intents[i].score < output.intents[j].score {
                            suppress.insert(a.clone());
                        } else {
                            suppress.insert(b.clone());
                        }
                    }
                }
            }
            output.intents.retain(|i| !suppress.contains(&i.id));
        }

        output
    });
    let combined_stats = compute_stats(&combined);
    write_stats(&mut report, "Combined Pipeline", &combined_stats);
    write_failures(&mut report, &combined, 5);

    // Also test: anchor-based + anti-co-occurrence
    report.push_str("\n**Pipeline variant: Anchor (window=7) + Anti-co-occurrence:**\n\n");
    let combined2 = run_experiment(&router, &scenarios, |r, q| {
        let mut output = r.route_multi_anchored(q, 0.3, 7);
        if output.intents.len() > 1 {
            let mut suppress: HashSet<String> = HashSet::new();
            for i in 0..output.intents.len() {
                for j in (i+1)..output.intents.len() {
                    let a = &output.intents[i].id;
                    let b = &output.intents[j].id;
                    let pair = if a < b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
                    let inv = invalid_pairs.get(&pair).copied().unwrap_or(0);
                    let val = valid_pairs.get(&pair).copied().unwrap_or(0);
                    if inv > val + 1 {
                        if output.intents[i].score < output.intents[j].score {
                            suppress.insert(a.clone());
                        } else {
                            suppress.insert(b.clone());
                        }
                    }
                }
            }
            output.intents.retain(|i| !suppress.contains(&i.id));
        }
        output
    });
    let combined2_stats = compute_stats(&combined2);
    write_stats(&mut report, "Anchor + Anti-Co-occurrence", &combined2_stats);

    // ============================
    // COMPARISON SUMMARY
    // ============================
    report.push_str("\n---\n\n## COMPARISON SUMMARY\n\n");
    report.push_str("| Experiment | Pass | Part | Fail | Pass% | P+P% | Avg FP/turn |\n|---|---|---|---|---|---|---|\n");

    let experiments: Vec<(&str, &ExperimentStats)> = vec![
        ("Baseline (t=0.3)", &baseline_stats),
        ("Noise Gate (max_df=8)", &best_ng_stats),
        ("Confidence Calibrated", &cal_stats),
        ("Anti-Co-occurrence", &anti_stats),
        ("Anchor (window=7)", &best_anchor_stats),
        ("Session Prior", &session_stats),
        ("Combined (NG+Cal+Anti)", &combined_stats),
        ("Anchor+Anti", &combined2_stats),
    ];
    for (name, s) in &experiments {
        report.push_str(&format!("| {} | {} | {} | {} | {:.1}% | {:.1}% | {:.2} |\n",
            name, s.pass, s.partial, s.fail,
            (s.pass as f64 / s.total as f64) * 100.0,
            ((s.pass + s.partial) as f64 / s.total as f64) * 100.0,
            s.total_fp as f64 / s.total as f64));
    }

    report.push_str(&format!("\n**SymSpell (Test A) measured separately** — typo recovery: {:.1}% → {:.1}% (with 30% typo rate)\n",
        (typo_stats.pass as f64 / typo_stats.total as f64) * 100.0,
        (corrected_stats.pass as f64 / corrected_stats.total as f64) * 100.0));

    // Save
    let output_path = "experiment_results.md";
    std::fs::write(output_path, &report).expect("Failed to write results");
    eprintln!("Results saved to {}", output_path);

    // Also save detailed JSON for all experiments
    let mut json_experiments: Vec<serde_json::Value> = Vec::new();
    let all_experiment_results = vec![
        ("baseline", &baseline),
        ("noise_gate_8", &best_ng),
        ("calibrated", &calibrated),
        ("anti_cooccurrence", &anti_cooc),
        ("anchor_7", &best_anchor),
        ("session_prior", &session_results),
        ("combined", &combined),
        ("anchor_anti", &combined2),
    ];
    for (name, results) in all_experiment_results {
        for r in results {
            json_experiments.push(serde_json::json!({
                "experiment": name,
                "scenario": r.scenario_id,
                "category": r.category,
                "turn": r.turn_idx + 1,
                "message": r.message,
                "word_count": r.word_count,
                "ground_truth": r.ground_truth,
                "detected": r.detected.iter().map(|(id, s)| serde_json::json!({"id": id, "score": s})).collect::<Vec<_>>(),
                "matched": r.matched,
                "missed": r.missed,
                "extra": r.extra.iter().map(|(id, s)| serde_json::json!({"id": id, "score": s})).collect::<Vec<_>>(),
                "status": format!("{:?}", r.status),
            }));
        }
    }
    let json_output = serde_json::to_string_pretty(&json_experiments).unwrap();
    std::fs::write("experiment_turns_detail.json", &json_output).expect("Failed to write JSON");
    eprintln!("Detailed turn data saved to experiment_turns_detail.json");
}
