//! Final Validation: Integrated Router with Paraphrase Index
//!
//! Tests the production-integrated pipeline (Router with built-in paraphrase matching)
//! against the 30-scenario evaluation suite. Compares with the Phase 2 baseline.
//!
//! Run with: cargo run --release --bin validate_integrated

use asv_router::{IntentType, Router};
use std::collections::{HashMap, HashSet};

#[derive(serde::Deserialize, Clone)]
struct Scenario {
    id: String,
    #[allow(dead_code)]
    category: String,
    #[allow(dead_code)]
    persona: serde_json::Value,
    turns: Vec<Turn>,
}

#[derive(serde::Deserialize, Clone)]
struct Turn {
    message: String,
    ground_truth: Vec<String>,
}

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

struct Stats {
    total: usize,
    exact: usize,
    recall: usize,
    total_fp: usize,
    corrections: usize,
    by_word_count: Vec<(String, usize, usize, usize)>, // (range, total, exact, recall)
    by_confidence: HashMap<String, (usize, usize)>, // confidence -> (total, correct)
}

fn word_count_bucket(wc: usize) -> &'static str {
    match wc {
        0..=5 => "1-5",
        6..=10 => "6-10",
        11..=20 => "11-20",
        21..=40 => "21-40",
        _ => "41+",
    }
}

fn run_pass(
    router: &mut Router,
    scenarios: &[Scenario],
    do_learn: bool,
    threshold: f32,
) -> Stats {
    let mut total = 0;
    let mut exact_count = 0;
    let mut recall_count = 0;
    let mut total_fp = 0;
    let mut corrections = 0;

    let mut bucket_stats: HashMap<&str, (usize, usize, usize)> = HashMap::new();
    let mut confidence_stats: HashMap<String, (usize, usize)> = HashMap::new();

    // Batch mode: defer automaton rebuilds to end of pass (matches clean experiment timing)
    if do_learn {
        router.begin_batch();
    }

    for scenario in scenarios {
        for turn in &scenario.turns {
            total += 1;
            let wc = turn.message.split_whitespace().count();
            let bucket = word_count_bucket(wc);

            let output = router.route_multi(&turn.message, threshold);
            let detected_ids: Vec<&str> = output.intents.iter().take(5).map(|i| i.id.as_str()).collect();
            let gt_set: HashSet<&str> = turn.ground_truth.iter().map(|s| s.as_str()).collect();
            let det_set: HashSet<&str> = detected_ids.iter().copied().collect();

            // Track confidence stats
            for intent in output.intents.iter().take(5) {
                let entry = confidence_stats.entry(intent.confidence.clone()).or_insert((0, 0));
                entry.0 += 1;
                if gt_set.contains(intent.id.as_str()) {
                    entry.1 += 1;
                }
            }

            // Exact match: detected == ground truth exactly
            let exact = gt_set == det_set;
            if exact { exact_count += 1; }

            // Top-5 recall: all GT intents in detected
            let all_found = gt_set.iter().all(|g| det_set.contains(g));
            if all_found { recall_count += 1; }

            // False positives
            let fp = det_set.iter().filter(|d| !gt_set.contains(*d)).count();
            total_fp += fp;

            // Per-bucket stats
            let entry = bucket_stats.entry(bucket).or_insert((0, 0, 0));
            entry.0 += 1;
            if exact { entry.1 += 1; }
            if all_found { entry.2 += 1; }

            // Learning — matches clean experiment's three paths:
            // 1. Missed intents: learn routing + paraphrases
            // 2. False positives: correct (unlearn wrong, learn right)
            // 3. Correct detections: reinforce routing + paraphrases
            if do_learn {
                // Path 1: Learn missed intents
                for gt_intent in &gt_set {
                    if !det_set.contains(gt_intent) {
                        router.learn(&turn.message, gt_intent);
                        // Extra paraphrase reinforcement (clean experiment called both
                        // router.learn AND paraphrase_index.learn_from_message)
                        router.reinforce(&turn.message, gt_intent);
                        corrections += 1;
                    }
                }
                // Path 2: Correct false positives
                for det_intent in &det_set {
                    if !gt_set.contains(det_intent) {
                        if let Some(correct) = gt_set.iter().next() {
                            router.correct(&turn.message, det_intent, correct);
                            corrections += 1;
                        }
                    }
                }
                // Path 3: Reinforce correct detections
                for gt_intent in gt_set.intersection(&det_set) {
                    router.learn(&turn.message, gt_intent);
                    router.reinforce(&turn.message, gt_intent);
                }
            }
        }
    }

    // End batch: rebuild automatons once for the entire pass
    if do_learn {
        router.end_batch();
    }

    let buckets = ["1-5", "6-10", "11-20", "21-40", "41+"];
    let by_word_count = buckets.iter().map(|b| {
        let (t, e, r) = bucket_stats.get(b).copied().unwrap_or((0, 0, 0));
        (b.to_string(), t, e, r)
    }).collect();

    Stats {
        total,
        exact: exact_count,
        recall: recall_count,
        total_fp,
        corrections,
        by_word_count,
        by_confidence: confidence_stats,
    }
}

fn print_stats(label: &str, stats: &Stats) {
    let exact_pct = (stats.exact as f64 / stats.total as f64) * 100.0;
    let recall_pct = (stats.recall as f64 / stats.total as f64) * 100.0;
    let fp_avg = stats.total_fp as f64 / stats.total as f64;

    println!("\n## {}", label);
    println!("- Exact match: {:.1}% ({}/{})", exact_pct, stats.exact, stats.total);
    println!("- Top-5 recall: {:.1}% ({}/{})", recall_pct, stats.recall, stats.total);
    println!("- Avg FP/turn: {:.2}", fp_avg);
    println!("- Corrections applied: {}", stats.corrections);

    println!("\n| Words | Total | Exact% | Recall% |");
    println!("|---|---|---|---|");
    for (range, total, exact, recall) in &stats.by_word_count {
        if *total == 0 { continue; }
        println!("| {} | {} | {:.1}% | {:.1}% |",
            range, total,
            (*exact as f64 / *total as f64) * 100.0,
            (*recall as f64 / *total as f64) * 100.0);
    }

    if !stats.by_confidence.is_empty() {
        println!("\n**Confidence tier accuracy:**");
        println!("| Tier | Total | Correct | TPR |");
        println!("|---|---|---|---|");
        for tier in &["high", "medium", "low"] {
            if let Some((total, correct)) = stats.by_confidence.get(*tier) {
                println!("| {} | {} | {} | {:.1}% |",
                    tier, total, correct,
                    (*correct as f64 / *total as f64) * 100.0);
            }
        }
    }
}

fn main() {
    let scenario_path = "tests/scenarios/scenarios.json";
    let new_scenario_path = "tests/scenarios/new_scenarios.json";
    let paraphrase_path = "tests/data/paraphrases.json";

    let scenarios: Vec<Scenario> = serde_json::from_str(
        &std::fs::read_to_string(scenario_path).expect("scenarios.json")
    ).expect("parse scenarios");

    let new_scenarios: Vec<Scenario> = serde_json::from_str(
        &std::fs::read_to_string(new_scenario_path).expect("new_scenarios.json")
    ).expect("parse new scenarios");

    let paraphrases: HashMap<String, Vec<String>> = serde_json::from_str(
        &std::fs::read_to_string(paraphrase_path).expect("paraphrases.json")
    ).expect("parse paraphrases");

    let total_turns: usize = scenarios.iter().map(|s| s.turns.len()).sum();
    let new_turns: usize = new_scenarios.iter().map(|s| s.turns.len()).sum();

    println!("# ASV Integrated Pipeline — Final Validation");
    println!("\nScenarios: {} ({} turns) + {} new ({} turns)",
        scenarios.len(), total_turns, new_scenarios.len(), new_turns);
    println!("Paraphrase phrases: {}", paraphrases.values().map(|v| v.len()).sum::<usize>());

    // Setup router with paraphrases loaded into integrated index
    let mut router = setup_router();

    // Load paraphrases into the integrated paraphrase index
    router.begin_batch();
    router.add_paraphrases_bulk(&paraphrases);
    router.end_batch();

    println!("\nRouter: {} intents, {} paraphrase phrases",
        router.intent_count(), router.paraphrase_count());

    // ========================================
    // PASS 1: With learning (30 scenarios)
    // ========================================
    eprintln!("Pass 1: Learning pass (30 scenarios)...");
    let pass1 = run_pass(&mut router, &scenarios, true, 0.3);
    print_stats("Pass 1: Learning (30 scenarios)", &pass1);

    let corrections_after_p1 = pass1.corrections;

    // ========================================
    // PASS 2: Generalization (30 scenarios, no learning)
    // ========================================
    eprintln!("Pass 2: Generalization (30 scenarios, no learning)...");
    let pass2 = run_pass(&mut router, &scenarios, false, 0.3);
    print_stats("Pass 2: Generalization (30 scenarios)", &pass2);

    // ========================================
    // NEW SCENARIOS: Unseen data (no learning)
    // ========================================
    eprintln!("New scenarios (10 unseen, no learning)...");
    let new_pass = run_pass(&mut router, &new_scenarios, false, 0.3);
    print_stats("New Scenarios (10 unseen)", &new_pass);

    // ========================================
    // PERSISTENCE TEST
    // ========================================
    eprintln!("Persistence test...");
    let json = router.export_json();
    let restored = Router::import_json(&json).expect("import_json");
    let persist_pass = run_pass(&mut router.clone_for_test(), &scenarios, false, 0.3);
    let restored_pass = run_pass(&mut { restored }, &scenarios, false, 0.3);

    println!("\n## Persistence Roundtrip");
    println!("- Original: {:.1}% exact", (persist_pass.exact as f64 / persist_pass.total as f64) * 100.0);
    println!("- Restored: {:.1}% exact", (restored_pass.exact as f64 / restored_pass.total as f64) * 100.0);
    println!("- JSON size: {} bytes", json.len());

    // ========================================
    // COMPARISON WITH PHASE 2 BASELINE
    // ========================================
    println!("\n## Comparison with Phase 2 Baseline (clean_experiment)");
    println!("| Metric | Phase 2 Baseline | Integrated Pipeline | Delta |");
    println!("|---|---|---|---|");

    let p2_exact = 73.2;
    let p2_recall = 80.4;
    let p2_fp = 0.12;

    let int_exact = (pass2.exact as f64 / pass2.total as f64) * 100.0;
    let int_recall = (pass2.recall as f64 / pass2.total as f64) * 100.0;
    let int_fp = pass2.total_fp as f64 / pass2.total as f64;

    println!("| Exact match | {:.1}% | {:.1}% | {:+.1}% |", p2_exact, int_exact, int_exact - p2_exact);
    println!("| Top-5 recall | {:.1}% | {:.1}% | {:+.1}% |", p2_recall, int_recall, int_recall - p2_recall);
    println!("| Avg FP/turn | {:.2} | {:.2} | {:+.2} |", p2_fp, int_fp, int_fp - p2_fp);
    println!("| Corrections | 372 | {} | {} |", corrections_after_p1, corrections_after_p1 as i64 - 372);

    println!("\n---\nDone.");
}

// Minimal Clone for Router — we just export/import for the persistence test
trait CloneForTest {
    fn clone_for_test(&self) -> Self;
}

impl CloneForTest for Router {
    fn clone_for_test(&self) -> Self {
        let json = self.export_json();
        Router::import_json(&json).expect("clone via export/import")
    }
}
