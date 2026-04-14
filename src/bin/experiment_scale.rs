/// Scale test: 98 intents (Stripe + Shopify + Linear + Vercel), 500 queries
/// Simulates enterprise MCP routing with iterative learning.
/// Tests if IDF holds at scale or degrades.
///
/// Run: cargo run --release --bin experiment_scale
use std::collections::{HashMap, HashSet};

struct Engine {
    weights: HashMap<String, HashMap<String, f32>>,
}

impl Engine {
    fn new() -> Self { Self { weights: HashMap::new() } }

    fn learn(&mut self, phrase: &str, intent: &str, rate: f32) {
        for word in tokenize(phrase) {
            let w = self.weights.entry(word).or_default()
                .entry(intent.to_string()).or_insert(0.0);
            *w = (*w + rate * (1.0 - *w)).min(1.0);
        }
    }

    fn n_intents(&self) -> f32 {
        let mut all: HashSet<&str> = HashSet::new();
        for m in self.weights.values() { for k in m.keys() { all.insert(k); } }
        all.len().max(1) as f32
    }

    fn vocab_size(&self) -> usize { self.weights.len() }

    fn score_top(&self, query: &str, ratio: f32) -> Vec<(String, f32)> {
        let n = self.n_intents();
        let mut scores: HashMap<String, f32> = HashMap::new();
        for word in tokenize(query) {
            if let Some(intents) = self.weights.get(&word) {
                let idf = (n / intents.len() as f32).ln().max(0.01);
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
        "just","also","really","like","very","please","would","could","should",
        "not","no","don","want","need","get","got","going","been","up","about",
        "all","what","how","when","where","which","who","why","some","any","there",
        "here","than","then","now","out","into","over","after","before",
        "am","being","each","few","more","most","other","such","only","same"].into_iter().collect();
    text.to_lowercase()
        .replace("n't", " not").replace("'ve", " have").replace("'re", " are")
        .replace("'m", " am").replace("'s", "").replace("'d", " would")
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2 && !stop.contains(w))
        .map(|s| s.to_string())
        .collect()
}

fn short(id: &str) -> &str {
    let parts: Vec<&str> = id.split(':').collect();
    if parts.len() >= 2 { parts[1] } else { id }
}

// ── Enterprise MCP intents: 4 providers × ~25 each ────────────────────────────

fn build_intents() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        // ── STRIPE (25 intents) ──
        ("stripe:create_payment_intent", vec!["create a payment intent", "start a new payment", "initiate a charge"]),
        ("stripe:confirm_payment", vec!["confirm the payment", "finalize the charge", "complete the transaction"]),
        ("stripe:capture_payment", vec!["capture the authorized payment", "settle the charge", "capture funds"]),
        ("stripe:cancel_payment", vec!["cancel the payment intent", "void the pending charge", "abort the transaction"]),
        ("stripe:create_refund", vec!["refund the payment", "issue a refund", "return the money to customer"]),
        ("stripe:list_refunds", vec!["list all refunds", "show refund history", "get refund records"]),
        ("stripe:create_customer", vec!["create a new customer", "add customer record", "register new client"]),
        ("stripe:update_customer", vec!["update customer details", "modify customer info", "change customer email"]),
        ("stripe:delete_customer", vec!["delete the customer", "remove customer record", "deactivate client"]),
        ("stripe:list_customers", vec!["list all customers", "show customer directory", "get all clients"]),
        ("stripe:create_subscription", vec!["create a subscription", "start recurring billing", "enroll in plan"]),
        ("stripe:update_subscription", vec!["update the subscription", "change subscription plan", "modify recurring billing"]),
        ("stripe:cancel_subscription", vec!["cancel the subscription", "end recurring billing", "stop the plan"]),
        ("stripe:list_subscriptions", vec!["list subscriptions", "show active plans", "get subscription status"]),
        ("stripe:create_invoice", vec!["create an invoice", "generate a bill", "prepare billing statement"]),
        ("stripe:pay_invoice", vec!["pay the invoice", "settle the bill", "process invoice payment"]),
        ("stripe:list_invoices", vec!["list all invoices", "show billing history", "get invoice records"]),
        ("stripe:create_price", vec!["create a price", "set product pricing", "define price point"]),
        ("stripe:create_product", vec!["create a product in stripe", "add product to catalog", "register new product"]),
        ("stripe:get_balance", vec!["check stripe balance", "show account balance", "get available funds"]),
        ("stripe:list_charges", vec!["list all charges", "show charge history", "get transaction log"]),
        ("stripe:create_coupon", vec!["create a coupon code", "set up a discount", "make a promotional code"]),
        ("stripe:list_disputes", vec!["list disputes", "show chargebacks", "get disputed charges"]),
        ("stripe:create_payout", vec!["create a payout", "transfer to bank", "withdraw funds"]),
        ("stripe:list_payouts", vec!["list payouts", "show withdrawal history", "get transfer records"]),

        // ── SHOPIFY (25 intents) ──
        ("shopify:list_orders", vec!["list shopify orders", "show recent orders", "get order history"]),
        ("shopify:get_order", vec!["get order details", "look up specific order", "find order by ID"]),
        ("shopify:create_order", vec!["create a shopify order", "place a new order", "submit order manually"]),
        ("shopify:update_order", vec!["update order details", "modify the order", "change order shipping"]),
        ("shopify:cancel_order", vec!["cancel the order", "void this order", "abort the purchase"]),
        ("shopify:close_order", vec!["close the order", "archive completed order", "mark order finished"]),
        ("shopify:list_products", vec!["list shopify products", "show product catalog", "get all merchandise"]),
        ("shopify:get_product", vec!["get product details", "look up product info", "find product by name"]),
        ("shopify:create_product", vec!["create shopify product", "add new merchandise", "list new item in store"]),
        ("shopify:update_product", vec!["update product details", "modify product listing", "change product price"]),
        ("shopify:delete_product", vec!["delete the product", "remove from catalog", "take item off store"]),
        ("shopify:list_customers", vec!["list shopify customers", "show buyer directory", "get shopper list"]),
        ("shopify:create_customer", vec!["create shopify customer", "add new shopper", "register store buyer"]),
        ("shopify:update_customer", vec!["update shopify customer", "modify buyer details", "change shopper info"]),
        ("shopify:list_collections", vec!["list collections", "show product categories", "get collection list"]),
        ("shopify:create_collection", vec!["create a collection", "add product category", "organize products into group"]),
        ("shopify:list_inventory", vec!["check inventory levels", "show stock counts", "get warehouse quantities"]),
        ("shopify:adjust_inventory", vec!["adjust inventory count", "update stock level", "correct quantity on hand"]),
        ("shopify:create_fulfillment", vec!["create fulfillment", "ship the order", "mark as shipped"]),
        ("shopify:list_draft_orders", vec!["list draft orders", "show pending drafts", "get unsubmitted orders"]),
        ("shopify:create_draft_order", vec!["create draft order", "start a quote", "prepare order draft"]),
        ("shopify:create_discount", vec!["create shopify discount", "set up store promotion", "make sale code"]),
        ("shopify:list_webhooks", vec!["list webhooks", "show notification hooks", "get webhook subscriptions"]),
        ("shopify:get_shop", vec!["get shop info", "show store details", "get store configuration"]),
        ("shopify:list_themes", vec!["list store themes", "show available designs", "get theme options"]),

        // ── LINEAR (24 intents) ──
        ("linear:create_issue", vec!["create a linear issue", "file a bug report", "open a new task"]),
        ("linear:update_issue", vec!["update the issue", "modify task details", "change issue status"]),
        ("linear:list_issues", vec!["list linear issues", "show all tasks", "get backlog items"]),
        ("linear:get_issue", vec!["get issue details", "look up specific task", "find issue by ID"]),
        ("linear:delete_issue", vec!["delete the issue", "remove the task", "discard this ticket"]),
        ("linear:create_project", vec!["create a linear project", "start new project", "set up project workspace"]),
        ("linear:list_projects", vec!["list all projects", "show project directory", "get active projects"]),
        ("linear:update_project", vec!["update project details", "modify project settings", "change project name"]),
        ("linear:create_comment", vec!["add a comment", "post a note on issue", "write comment on task"]),
        ("linear:list_comments", vec!["list comments on issue", "show discussion thread", "get all notes"]),
        ("linear:create_label", vec!["create a label", "add a tag", "make a new category"]),
        ("linear:list_labels", vec!["list all labels", "show available tags", "get label options"]),
        ("linear:assign_issue", vec!["assign issue to someone", "delegate this task", "set assignee"]),
        ("linear:create_cycle", vec!["create a sprint cycle", "start new iteration", "plan next sprint"]),
        ("linear:list_cycles", vec!["list sprint cycles", "show iterations", "get cycle history"]),
        ("linear:add_to_cycle", vec!["add issue to sprint", "include in current cycle", "schedule for iteration"]),
        ("linear:create_team", vec!["create a team", "set up new team", "add team to workspace"]),
        ("linear:list_teams", vec!["list all teams", "show team directory", "get workspace teams"]),
        ("linear:list_members", vec!["list team members", "show who is on team", "get member list"]),
        ("linear:create_milestone", vec!["create a milestone", "set a project goal", "define milestone target"]),
        ("linear:archive_issue", vec!["archive the issue", "shelve this task", "put issue on hold"]),
        ("linear:create_view", vec!["create a custom view", "save a filter", "make saved search"]),
        ("linear:list_workflows", vec!["list workflow states", "show status options", "get workflow steps"]),
        ("linear:update_workflow", vec!["update workflow state", "change status flow", "modify workflow step"]),

        // ── VERCEL (24 intents) ──
        ("vercel:list_deployments", vec!["list deployments", "show deploy history", "get recent deploys"]),
        ("vercel:create_deployment", vec!["deploy the project", "trigger a new deployment", "push to production"]),
        ("vercel:get_deployment", vec!["get deployment details", "check deploy status", "look up specific deploy"]),
        ("vercel:cancel_deployment", vec!["cancel the deployment", "abort the deploy", "stop deployment in progress"]),
        ("vercel:list_projects", vec!["list vercel projects", "show all projects", "get project directory"]),
        ("vercel:create_project", vec!["create vercel project", "set up new site", "initialize project"]),
        ("vercel:update_project", vec!["update project settings", "modify project config", "change project domain"]),
        ("vercel:delete_project", vec!["delete the project", "remove vercel project", "tear down the site"]),
        ("vercel:list_domains", vec!["list domains", "show configured domains", "get domain list"]),
        ("vercel:add_domain", vec!["add a domain", "configure custom domain", "set up domain mapping"]),
        ("vercel:remove_domain", vec!["remove the domain", "delete domain mapping", "unconfigure domain"]),
        ("vercel:list_env_vars", vec!["list environment variables", "show env config", "get secret values"]),
        ("vercel:add_env_var", vec!["add environment variable", "set a secret", "configure env value"]),
        ("vercel:remove_env_var", vec!["remove environment variable", "delete the secret", "clear env value"]),
        ("vercel:get_logs", vec!["get deployment logs", "show runtime logs", "fetch build output"]),
        ("vercel:list_aliases", vec!["list deployment aliases", "show URL mappings", "get alias config"]),
        ("vercel:create_alias", vec!["create an alias", "add URL mapping", "set custom URL"]),
        ("vercel:list_certs", vec!["list SSL certificates", "show cert status", "get certificate info"]),
        ("vercel:check_dns", vec!["check DNS configuration", "verify domain DNS", "test DNS records"]),
        ("vercel:get_usage", vec!["get usage stats", "show bandwidth consumption", "check resource usage"]),
        ("vercel:list_teams", vec!["list vercel teams", "show team accounts", "get organization list"]),
        ("vercel:rollback", vec!["rollback deployment", "revert to previous version", "undo last deploy"]),
        ("vercel:promote", vec!["promote to production", "make this the live version", "set as active deploy"]),
        ("vercel:list_edge_config", vec!["list edge configs", "show edge configuration", "get edge settings"]),
    ]
}

// ── Generate realistic test queries (enterprise users) ────────────────────────

fn build_queries() -> Vec<(&'static str, &'static str)> {
    vec![
        // Stripe — direct
        ("charge this customer $50", "stripe:create_payment_intent"),
        ("process a payment for order 1234", "stripe:create_payment_intent"),
        ("give the customer their money back", "stripe:create_refund"),
        ("reverse the last transaction", "stripe:create_refund"),
        ("show me all the refunds from this month", "stripe:list_refunds"),
        ("add a new client to our billing system", "stripe:create_customer"),
        ("who are all our paying customers", "stripe:list_customers"),
        ("set up monthly billing for this account", "stripe:create_subscription"),
        ("the customer wants to stop their plan", "stripe:cancel_subscription"),
        ("which subscriptions are currently active", "stripe:list_subscriptions"),
        ("send them a bill for consulting", "stripe:create_invoice"),
        ("how much money do we have available", "stripe:get_balance"),
        ("pull up the transaction log", "stripe:list_charges"),
        ("make a 20% off code for the sale", "stripe:create_coupon"),
        ("any chargebacks we need to deal with", "stripe:list_disputes"),
        ("transfer the balance to our bank", "stripe:create_payout"),

        // Stripe — indirect/natural
        ("they overpaid take care of it", "stripe:create_refund"),
        ("bill is overdue collect payment now", "stripe:create_payment_intent"),
        ("downgrade them from premium to basic", "stripe:update_subscription"),
        ("fire off the weekly payroll batch", "stripe:create_payout"),

        // Shopify — direct
        ("what orders came in today", "shopify:list_orders"),
        ("look up order number 5678", "shopify:get_order"),
        ("manually enter an order for a phone customer", "shopify:create_order"),
        ("change the shipping address on that order", "shopify:update_order"),
        ("customer changed their mind cancel it", "shopify:cancel_order"),
        ("what do we have in stock", "shopify:list_inventory"),
        ("we received 50 more units update the count", "shopify:adjust_inventory"),
        ("ship order 5678", "shopify:create_fulfillment"),
        ("add this new t-shirt to the store", "shopify:create_product"),
        ("take the discontinued item off the site", "shopify:delete_product"),
        ("set up a Black Friday promotion", "shopify:create_discount"),
        ("what themes are available for our store", "shopify:list_themes"),

        // Shopify — indirect/natural
        ("someone called in wanting to buy let me enter it", "shopify:create_order"),
        ("the red dress is sold out everywhere", "shopify:adjust_inventory"),
        ("holiday season is coming prepare some deals", "shopify:create_discount"),

        // Linear — direct
        ("file a bug the login page is broken", "linear:create_issue"),
        ("mark that task as done", "linear:update_issue"),
        ("what's in the backlog", "linear:list_issues"),
        ("assign this to Sarah", "linear:assign_issue"),
        ("start planning the next sprint", "linear:create_cycle"),
        ("add these tickets to the current sprint", "linear:add_to_cycle"),
        ("create a frontend team workspace", "linear:create_team"),
        ("who is on the platform team", "linear:list_members"),
        ("post an update on the auth issue", "linear:create_comment"),
        ("tag this as high priority", "linear:create_label"),
        ("set a Q3 launch milestone", "linear:create_milestone"),
        ("put that on the back burner for now", "linear:archive_issue"),

        // Linear — indirect/natural
        ("the checkout flow is completely broken users are complaining", "linear:create_issue"),
        ("hand this off to the backend team", "linear:assign_issue"),
        ("we need to scope out the next two weeks", "linear:create_cycle"),

        // Vercel — direct
        ("deploy the latest commit", "vercel:create_deployment"),
        ("what's the status of the last deploy", "vercel:get_deployment"),
        ("abort that deploy something is wrong", "vercel:cancel_deployment"),
        ("show me all our vercel projects", "vercel:list_projects"),
        ("set up the custom domain for our marketing site", "vercel:add_domain"),
        ("add the database URL as an env variable", "vercel:add_env_var"),
        ("check why the build failed", "vercel:get_logs"),
        ("are our SSL certs valid", "vercel:list_certs"),
        ("go back to the previous working version", "vercel:rollback"),
        ("make the staging deploy the new production", "vercel:promote"),
        ("how much bandwidth did we use this month", "vercel:get_usage"),

        // Vercel — indirect/natural
        ("push it live", "vercel:create_deployment"),
        ("something broke revert everything", "vercel:rollback"),
        ("the site is down check the logs", "vercel:get_logs"),

        // Cross-domain confusion tests (similar words, different providers)
        ("create a new product listing", "shopify:create_product"),
        ("create a product in our billing system", "stripe:create_product"),
        ("list our customers from the store", "shopify:list_customers"),
        ("list customers in our payment system", "stripe:list_customers"),
        ("delete the project from hosting", "vercel:delete_project"),
        ("show all projects in the tracker", "linear:list_projects"),
    ]
}

fn main() {
    let intents = build_intents();
    let queries = build_queries();

    println!("\n{:=<75}", "");
    println!("  SCALE TEST: {} intents, {} queries (Stripe+Shopify+Linear+Vercel)",
        intents.len(), queries.len());
    println!("{:=<75}\n", "");

    let mut engine = Engine::new();

    // Seed
    for (intent, phrases) in &intents {
        for p in phrases { engine.learn(p, intent, 0.4); }
    }
    println!("  Seeded: {} intents, {} vocab words\n", intents.len(), engine.vocab_size());

    // Baseline
    let (e0, p0, f0, failed0) = eval(&engine, &queries);
    let t = queries.len();
    println!("  SEED ONLY: {}/{} ({:.0}%) exact | {} partial | {} fail | vocab={}",
        e0, t, 100.0*e0 as f32/t as f32, p0, f0, engine.vocab_size());

    // Iterative learning: 5 waves
    let mut total_learned = 0;
    let chunk = queries.len() / 5;
    for wave in 0..5 {
        let start = wave * chunk;
        let end = ((wave + 1) * chunk).min(queries.len());
        let wave_queries = &queries[start..end];

        // Test this wave, learn from failures
        let (_, _, _, failed) = eval(&engine, wave_queries);
        let mut learned = 0;
        for &idx in &failed {
            let (query, intent) = wave_queries[idx];
            engine.learn(query, intent, 0.3);
            learned += 1;
        }
        total_learned += learned;

        // Evaluate ALL
        let (e, p, f, _) = eval(&engine, &queries);
        println!("  Wave {} (+{:>2} learned): {}/{} ({:>3.0}%) exact | {} partial | {} fail | vocab={}",
            wave + 1, learned, e, t, 100.0*e as f32/t as f32, p, f, engine.vocab_size());
    }

    // Final cleanup: learn from ALL remaining failures
    let (_, _, _, remaining) = eval(&engine, &queries);
    for &idx in &remaining {
        let (query, intent) = queries[idx];
        engine.learn(query, intent, 0.3);
    }
    total_learned += remaining.len();

    let (ef, pf, ff, still_fail) = eval(&engine, &queries);
    println!("  Final:     {}/{} ({:.0}%) exact | {} partial | {} fail | vocab={} | total_learned={}",
        ef, t, 100.0*ef as f32/t as f32, pf, ff, engine.vocab_size(), total_learned);

    // Show failures
    if !still_fail.is_empty() {
        println!("\n  REMAINING FAILURES ({}):", still_fail.len());
        for &idx in &still_fail {
            let (query, expected) = queries[idx];
            let got: Vec<String> = engine.score_top(query, 0.5).iter().take(3)
                .map(|(id, s)| format!("{}={:.1}", short(id), s)).collect();
            println!("    exp={:<30} got=[{}]", short(expected), got.join(", "));
            println!("      \"{}\"", &query[..query.len().min(55)]);
        }
    }

    // IDF health check: how many words are in 5+ intents?
    println!("\n  IDF HEALTH:");
    let mut shared_counts = [0usize; 6]; // 1, 2, 3, 4, 5, 6+ intents
    for (_, intents_map) in &engine.weights {
        let n = intents_map.len();
        if n >= 6 { shared_counts[5] += 1; }
        else { shared_counts[n - 1] += 1; }
    }
    println!("    Words in 1 intent:  {} (highly discriminative)", shared_counts[0]);
    println!("    Words in 2 intents: {} (good)", shared_counts[1]);
    println!("    Words in 3 intents: {} (moderate)", shared_counts[2]);
    println!("    Words in 4 intents: {} (weak)", shared_counts[3]);
    println!("    Words in 5 intents: {} (noisy)", shared_counts[4]);
    println!("    Words in 6+ intents:{} (very noisy)", shared_counts[5]);
}

fn eval(engine: &Engine, queries: &[(&str, &str)]) -> (usize, usize, usize, Vec<usize>) {
    let (mut e, mut p, mut f) = (0, 0, 0);
    let mut failed = Vec::new();
    for (i, (query, expected)) in queries.iter().enumerate() {
        let got: HashSet<String> = engine.score_top(query, 0.5).iter()
            .map(|(id, _)| id.clone()).collect();
        let exp: HashSet<String> = [expected.to_string()].into_iter().collect();
        if got == exp { e += 1; }
        else if got.contains(&expected.to_string()) { p += 1; failed.push(i); }
        else { f += 1; failed.push(i); }
    }
    (e, p, f, failed)
}
