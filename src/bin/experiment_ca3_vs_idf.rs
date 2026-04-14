/// Experiment: CA3-style scoring (no IDF) vs ASV-style scoring (IDF)
///
/// Tests across 3 domains: IT Helpdesk, Stripe, Shopify
/// Simulates auto-learn: seed → test → learn from failures → test again
/// KEY QUESTION: Does more learning help (CA3) or hurt (ASV/IDF)?
///
/// Run: cargo run --release --bin experiment_ca3_vs_idf
use std::collections::{HashMap, HashSet};

// ── CA3-style engine: NO IDF, just term weight sum ────────────────────────────

struct Ca3Engine {
    /// term → intent → weight
    weights: HashMap<String, HashMap<String, f32>>,
}

impl Ca3Engine {
    fn new() -> Self { Self { weights: HashMap::new() } }

    fn learn(&mut self, phrase: &str, intent: &str, rate: f32) {
        for word in tokenize(phrase) {
            let w = self.weights.entry(word).or_default()
                .entry(intent.to_string()).or_insert(0.0);
            *w = (*w + rate * (1.0 - *w)).min(1.0);  // asymptotic
        }
    }

    fn score(&self, query: &str) -> Vec<(String, f32)> {
        let mut scores: HashMap<String, f32> = HashMap::new();
        for word in tokenize(query) {
            if let Some(intents) = self.weights.get(&word) {
                for (intent, &weight) in intents {
                    *scores.entry(intent.clone()).or_default() += weight;
                }
            }
        }
        let mut sorted: Vec<(String, f32)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    fn score_top(&self, query: &str, threshold_ratio: f32) -> Vec<(String, f32)> {
        let all = self.score(query);
        if all.is_empty() { return all; }
        let top = all[0].1;
        if top < 0.1 { return vec![]; }
        all.into_iter().filter(|(_, s)| *s >= top * threshold_ratio).collect()
    }
}

// ── ASV-style engine: IDF-weighted ────────────────────────────────────────────

struct IdfEngine {
    /// term → intent → weight (same as Ca3)
    weights: HashMap<String, HashMap<String, f32>>,
}

impl IdfEngine {
    fn new() -> Self { Self { weights: HashMap::new() } }

    fn learn(&mut self, phrase: &str, intent: &str, rate: f32) {
        for word in tokenize(phrase) {
            let w = self.weights.entry(word).or_default()
                .entry(intent.to_string()).or_insert(0.0);
            *w = (*w + rate * (1.0 - *w)).min(1.0);
        }
    }

    fn total_intents(&self) -> f32 {
        let mut all: HashSet<&str> = HashSet::new();
        for intents in self.weights.values() {
            for id in intents.keys() { all.insert(id.as_str()); }
        }
        all.len().max(1) as f32
    }

    fn score(&self, query: &str) -> Vec<(String, f32)> {
        let n = self.total_intents();
        let mut scores: HashMap<String, f32> = HashMap::new();
        for word in tokenize(query) {
            if let Some(intents) = self.weights.get(&word) {
                let idf = (n / intents.len() as f32).ln().max(0.0);
                for (intent, &weight) in intents {
                    *scores.entry(intent.clone()).or_default() += weight * idf;
                }
            }
        }
        let mut sorted: Vec<(String, f32)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    fn score_top(&self, query: &str, threshold_ratio: f32) -> Vec<(String, f32)> {
        let all = self.score(query);
        if all.is_empty() { return all; }
        let top = all[0].1;
        if top < 0.1 { return vec![]; }
        all.into_iter().filter(|(_, s)| *s >= top * threshold_ratio).collect()
    }
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

fn tokenize(text: &str) -> Vec<String> {
    let stop: HashSet<&str> = ["a","an","the","is","are","was","were","be","been","being",
        "in","on","at","to","of","for","with","from","by","and","or","but",
        "i","me","my","we","our","you","your","it","its","he","she","they","them",
        "this","that","do","does","did","has","have","had","so","if","can","will",
        "just","also","really","like","very","please","would","could","should",
        "not","no","don","want","need","get","got","going","been"].into_iter().collect();

    text.to_lowercase()
        .replace("n't", " not").replace("'ve", " have").replace("'re", " are")
        .replace("'m", " am").replace("'s", "").replace("'d", " would")
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2 && !stop.contains(w))
        .map(|s| s.to_string())
        .collect()
}

fn short(id: &str) -> &str { id.split(':').last().unwrap_or(id) }

// ── Test data: 3 domains ──────────────────────────────────────────────────────

struct Domain {
    name: &'static str,
    intents: Vec<(&'static str, Vec<&'static str>)>,  // (intent_id, seed_phrases)
    test_queries: Vec<(&'static str, Vec<&'static str>)>,  // (query, expected_intents)
    learn_queries: Vec<(&'static str, &'static str)>,  // (phrase, intent) — simulated auto-learn
}

fn build_domains() -> Vec<Domain> {
    vec![
        Domain {
            name: "IT Helpdesk",
            intents: vec![
                ("help:reset_password", vec!["reset my password", "forgot my password", "password expired", "can't log in", "locked out wrong password"]),
                ("help:setup_mfa", vec!["set up two-factor authentication", "configure authenticator app", "enable MFA on my account"]),
                ("help:escalate", vec!["please escalate my ticket", "this is urgent", "mark as high priority", "need faster response"]),
                ("help:wifi", vec!["wifi is not connecting", "no internet connection", "wifi signal very weak", "internet keeps dropping"]),
                ("help:vpn", vec!["VPN is not working", "can't connect to VPN", "VPN keeps disconnecting"]),
                ("help:broken", vec!["my laptop is broken", "computer won't turn on", "screen is cracked", "device completely dead"]),
                ("help:loaner", vec!["need a loaner laptop", "borrow temporary device", "spare computer while mine is repaired"]),
                ("help:setup_device", vec!["set up my new laptop", "configure new workstation", "initialize my new machine"]),
            ],
            test_queries: vec![
                ("I forgot my login credentials", vec!["help:reset_password"]),
                ("my password isn't working anymore", vec!["help:reset_password"]),
                ("I'm so frustrated nobody is helping me", vec!["help:escalate"]),
                ("been waiting all morning for a response", vec!["help:escalate"]),
                ("the internet keeps cutting out at my desk", vec!["help:wifi"]),
                ("my connection drops every few minutes", vec!["help:wifi"]),
                ("laptop fell and the screen went black", vec!["help:broken"]),
                ("my machine is completely dead", vec!["help:broken"]),
                ("working from home and can't access anything", vec!["help:vpn"]),
                ("just started and my computer has nothing on it", vec!["help:setup_device"]),
                ("how do I add the security code to my phone", vec!["help:setup_mfa"]),
                ("need something to work on while mine is being fixed", vec!["help:loaner"]),
            ],
            learn_queries: vec![
                ("my login credentials aren't working", "help:reset_password"),
                ("system keeps rejecting my password", "help:reset_password"),
                ("I'm fed up with this terrible support", "help:escalate"),
                ("this is completely unacceptable service", "help:escalate"),
                ("at my wits end with waiting", "help:escalate"),
                ("wireless keeps disconnecting in the office", "help:wifi"),
                ("can't get online from my desk", "help:wifi"),
                ("remote connection keeps failing from home", "help:vpn"),
                ("secure tunnel won't establish", "help:vpn"),
                ("device won't power on anymore", "help:broken"),
                ("new hire workstation needs configuring", "help:setup_device"),
                ("extra login security step on my phone", "help:setup_mfa"),
                ("temporary replacement while mine is in the shop", "help:loaner"),
            ],
        },
        Domain {
            name: "Stripe",
            intents: vec![
                ("stripe:create_payment", vec!["create a payment", "charge the customer", "process a payment", "make a new charge"]),
                ("stripe:refund", vec!["refund the payment", "issue a refund", "return the money", "reverse the charge"]),
                ("stripe:list_payments", vec!["list all payments", "show payment history", "get recent transactions"]),
                ("stripe:create_customer", vec!["create a new customer", "add customer record", "register a new client"]),
                ("stripe:create_subscription", vec!["create a subscription", "start a recurring plan", "subscribe the customer"]),
                ("stripe:cancel_subscription", vec!["cancel the subscription", "end the recurring plan", "stop the subscription"]),
                ("stripe:create_invoice", vec!["create an invoice", "generate a bill", "send an invoice"]),
                ("stripe:get_balance", vec!["check the balance", "show account balance", "what is the current balance"]),
            ],
            test_queries: vec![
                ("charge this customer's card", vec!["stripe:create_payment"]),
                ("process a $50 transaction", vec!["stripe:create_payment"]),
                ("give the customer their money back", vec!["stripe:refund"]),
                ("reverse the last transaction", vec!["stripe:refund"]),
                ("show me all recent charges", vec!["stripe:list_payments"]),
                ("add a new client to the system", vec!["stripe:create_customer"]),
                ("set up monthly billing for this customer", vec!["stripe:create_subscription"]),
                ("the customer wants to stop paying monthly", vec!["stripe:cancel_subscription"]),
                ("they don't want the plan anymore", vec!["stripe:cancel_subscription"]),
                ("bill them for the consulting work", vec!["stripe:create_invoice"]),
                ("how much money is in the account", vec!["stripe:get_balance"]),
            ],
            learn_queries: vec![
                ("run a $100 charge on their visa", "stripe:create_payment"),
                ("take payment from this account", "stripe:create_payment"),
                ("customer wants their money back", "stripe:refund"),
                ("credit back the failed order", "stripe:refund"),
                ("pull up the transaction log", "stripe:list_payments"),
                ("enroll in the premium tier", "stripe:create_subscription"),
                ("discontinue the recurring billing", "stripe:cancel_subscription"),
                ("stop automatic payments", "stripe:cancel_subscription"),
                ("send them a bill for services", "stripe:create_invoice"),
            ],
        },
        Domain {
            name: "Shopify",
            intents: vec![
                ("shop:get_orders", vec!["list orders", "show recent orders", "get all orders", "find orders"]),
                ("shop:create_order", vec!["create a new order", "place an order", "make a new order"]),
                ("shop:update_order", vec!["update the order", "modify the order", "change order details"]),
                ("shop:get_products", vec!["list products", "show all products", "get product catalog"]),
                ("shop:create_product", vec!["add a new product", "create product listing", "add item to store"]),
                ("shop:get_customers", vec!["list customers", "show all customers", "get customer list"]),
                ("shop:create_discount", vec!["create a discount code", "set up a promotion", "make a coupon"]),
            ],
            test_queries: vec![
                ("what orders came in today", vec!["shop:get_orders"]),
                ("show me everything that was ordered this week", vec!["shop:get_orders"]),
                ("put in a new order for this customer", vec!["shop:create_order"]),
                ("change the shipping address on that order", vec!["shop:update_order"]),
                ("what do we have in the catalog", vec!["shop:get_products"]),
                ("add this new item to the online store", vec!["shop:create_product"]),
                ("who are our top buyers", vec!["shop:get_customers"]),
                ("set up a 20% off code for the holiday sale", vec!["shop:create_discount"]),
            ],
            learn_queries: vec![
                ("pull up today's sales", "shop:get_orders"),
                ("any new purchases since yesterday", "shop:get_orders"),
                ("submit an order on behalf of this client", "shop:create_order"),
                ("fix the quantity on order #1234", "shop:update_order"),
                ("what's in our inventory right now", "shop:get_products"),
                ("list a new t-shirt design", "shop:create_product"),
                ("create a promo code for Black Friday", "shop:create_discount"),
            ],
        },
    ]
}

// ── Evaluation ────────────────────────────────────────────────────────────────

fn eval_engine<F>(score_fn: F, queries: &[(&str, Vec<&str>)]) -> (usize, usize, usize)
where F: Fn(&str) -> Vec<(String, f32)>
{
    let (mut exact, mut partial, mut fail) = (0, 0, 0);
    for (query, expected) in queries {
        let got: HashSet<String> = score_fn(query).iter().map(|(id, _)| id.clone()).collect();
        let exp: HashSet<String> = expected.iter().map(|s| s.to_string()).collect();
        if got == exp { exact += 1; }
        else if !got.is_disjoint(&exp) { partial += 1; }
        else { fail += 1; }
    }
    (exact, partial, fail)
}

fn main() {
    let domains = build_domains();

    println!("\n{:=<75}", "");
    println!("  CA3 (no IDF) vs ASV (IDF) — 3 domains, seed + auto-learn");
    println!("{:=<75}\n", "");

    let mut total_ca3_seed = (0, 0, 0);
    let mut total_idf_seed = (0, 0, 0);
    let mut total_ca3_learn = (0, 0, 0);
    let mut total_idf_learn = (0, 0, 0);
    let mut total_queries = 0;

    for domain in &domains {
        let mut ca3 = Ca3Engine::new();
        let mut idf = IdfEngine::new();

        // Seed both engines identically
        for (intent, phrases) in &domain.intents {
            for phrase in phrases {
                ca3.learn(phrase, intent, 0.4);
                idf.learn(phrase, intent, 0.4);
            }
        }

        let n = domain.test_queries.len();
        total_queries += n;

        // ── PHASE 1: Seed only ──
        let (ce, cp, cf) = eval_engine(|q| ca3.score_top(q, 0.5), &domain.test_queries);
        let (ie, ip, iff) = eval_engine(|q| idf.score_top(q, 0.5), &domain.test_queries);

        println!("  ── {} ({} intents, {} queries) ──", domain.name, domain.intents.len(), n);
        println!("    SEED ONLY:");
        println!("      CA3: {}/{} ({:>3.0}%) exact | {} partial | {} fail",
            ce, n, 100.0*ce as f32/n as f32, cp, cf);
        println!("      IDF: {}/{} ({:>3.0}%) exact | {} partial | {} fail",
            ie, n, 100.0*ie as f32/n as f32, ip, iff);

        total_ca3_seed.0 += ce; total_ca3_seed.1 += cp; total_ca3_seed.2 += cf;
        total_idf_seed.0 += ie; total_idf_seed.1 += ip; total_idf_seed.2 += iff;

        // ── PHASE 2: Simulate auto-learn (add diverse phrases) ──
        for (phrase, intent) in &domain.learn_queries {
            ca3.learn(phrase, intent, 0.3);  // lower rate for learned
            idf.learn(phrase, intent, 0.3);
        }

        let (ce2, cp2, cf2) = eval_engine(|q| ca3.score_top(q, 0.5), &domain.test_queries);
        let (ie2, ip2, if2) = eval_engine(|q| idf.score_top(q, 0.5), &domain.test_queries);

        println!("    AFTER +{} LEARNED:", domain.learn_queries.len());
        println!("      CA3: {}/{} ({:>3.0}%) exact | {} partial | {} fail  Δ={:+}",
            ce2, n, 100.0*ce2 as f32/n as f32, cp2, cf2, ce2 as i32 - ce as i32);
        println!("      IDF: {}/{} ({:>3.0}%) exact | {} partial | {} fail  Δ={:+}",
            ie2, n, 100.0*ie2 as f32/n as f32, ip2, if2, ie2 as i32 - ie as i32);
        println!();

        total_ca3_learn.0 += ce2; total_ca3_learn.1 += cp2; total_ca3_learn.2 += cf2;
        total_idf_learn.0 += ie2; total_idf_learn.1 += ip2; total_idf_learn.2 += if2;

        // Show specific queries where they differ
        for (query, expected) in &domain.test_queries {
            let ca3_got: HashSet<String> = ca3.score_top(query, 0.5).iter().map(|(id,_)| id.clone()).collect();
            let idf_got: HashSet<String> = idf.score_top(query, 0.5).iter().map(|(id,_)| id.clone()).collect();
            let exp: HashSet<String> = expected.iter().map(|s| s.to_string()).collect();
            let ca3_ok = ca3_got == exp;
            let idf_ok = idf_got == exp;
            if ca3_ok != idf_ok {
                let winner = if ca3_ok { "CA3 ✓" } else { "IDF ✓" };
                let ca3_s: Vec<&str> = ca3_got.iter().map(|s| short(s)).collect();
                let idf_s: Vec<&str> = idf_got.iter().map(|s| short(s)).collect();
                println!("    {} \"{}\"", winner, &query[..query.len().min(50)]);
                println!("           CA3={:?} IDF={:?} exp={:?}", ca3_s, idf_s,
                    expected.iter().map(|s| short(s)).collect::<Vec<_>>());
            }
        }
        println!();
    }

    // ── TOTALS ──
    println!("{:=<75}", "");
    println!("  TOTALS ({} queries across 3 domains)", total_queries);
    println!("{:=<75}", "");
    println!("  SEED ONLY:");
    println!("    CA3: {}/{} ({:.0}%) exact",
        total_ca3_seed.0, total_queries, 100.0*total_ca3_seed.0 as f32/total_queries as f32);
    println!("    IDF: {}/{} ({:.0}%) exact",
        total_idf_seed.0, total_queries, 100.0*total_idf_seed.0 as f32/total_queries as f32);
    println!("  AFTER LEARNING:");
    println!("    CA3: {}/{} ({:.0}%) exact  Δ={:+}",
        total_ca3_learn.0, total_queries, 100.0*total_ca3_learn.0 as f32/total_queries as f32,
        total_ca3_learn.0 as i32 - total_ca3_seed.0 as i32);
    println!("    IDF: {}/{} ({:.0}%) exact  Δ={:+}",
        total_idf_learn.0, total_queries, 100.0*total_idf_learn.0 as f32/total_queries as f32,
        total_idf_learn.0 as i32 - total_idf_seed.0 as i32);

    let ca3_better = total_ca3_learn.0 > total_idf_learn.0;
    let idf_better = total_idf_learn.0 > total_ca3_learn.0;
    println!();
    if ca3_better {
        println!("  ✓ CA3 (no IDF) WINS — learning helps, no dilution");
    } else if idf_better {
        println!("  ✓ IDF WINS — discrimination matters more than coverage");
    } else {
        println!("  ~ TIE — same performance");
    }
}
