//! Quick interactive demo of the integrated pipeline.
//! cargo run --release --bin demo

use asv_router::{IntentType, Router};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

fn main() {
    let mut router = setup_router();

    // Load paraphrases
    let paraphrases: HashMap<String, Vec<String>> = serde_json::from_str(
        &std::fs::read_to_string("tests/data/paraphrases.json").expect("paraphrases.json")
    ).expect("parse");
    router.begin_batch();
    router.add_paraphrases_bulk(&paraphrases);
    router.end_batch();

    // Simulate some learning (a few corrections to warm up)
    let warmup = [
        ("I'm being charged twice and I want my money back", "billing_issue"),
        ("stop charging me every month", "cancel_order"),
        ("where the heck is my stuff", "track_order"),
        ("just give me my money back already", "refund"),
        ("I need to talk to someone real", "contact_human"),
        ("can you check how many points I have", "loyalty_points"),
        ("what plans do you have", "shipping_options"),
        ("is my subscription still active", "subscription_status"),
        ("I want to add my visa card", "add_payment_method"),
        ("take that item off my order", "remove_item"),
    ];
    router.begin_batch();
    for (msg, intent) in &warmup {
        router.learn(msg, intent);
    }
    router.end_batch();

    println!("ASV Router — Interactive Multi-Intent Demo");
    println!("Router: {} intents, {} paraphrase phrases", router.intent_count(), router.paraphrase_count());
    println!("Type a customer message (or 'quit' to exit):\n");

    // Demo queries first
    let demos = [
        "cancel my order and give me a refund",
        "I was charged twice, I want my money back and to talk to a manager",
        "where is my package? also can I check my loyalty points",
        "I need to reset my password and update my address before you ship it",
        "this is ridiculous, close my account and refund everything",
        "how much does express shipping cost? and do you have this in stock?",
        "can someone call me back about the fraudulent charges on my card",
        "just give me a refund or let me speak to a human",
    ];

    for query in &demos {
        print_result(&router, query);
    }

    println!("\n--- Your turn! Type a message: ---\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
        print!("> ");
        stdout.flush().unwrap();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 { break; }
        let line = line.trim();
        if line.is_empty() { continue; }
        if line == "quit" || line == "exit" { break; }
        print_result(&router, line);
    }
}

fn print_result(router: &Router, query: &str) {
    let output = router.route_multi(query, 0.3);
    println!("  Query: \"{}\"", query);
    if output.intents.is_empty() {
        println!("  → No intents detected\n");
        return;
    }
    for (i, intent) in output.intents.iter().enumerate() {
        let conf_icon = match intent.confidence.as_str() {
            "high" => "[HIGH]",
            "medium" => "[MED] ",
            "low" => "[LOW] ",
            _ => "[???] ",
        };
        let itype = match intent.intent_type {
            IntentType::Action => "action",
            IntentType::Context => "context",
        };
        println!("  {}  {}: {} score={:.2} src={} type={}",
            i + 1, conf_icon, intent.id, intent.score, intent.source, itype);
    }
    if !output.relations.is_empty() {
        for rel in &output.relations {
            println!("  relation: {:?}", rel);
        }
    }
    println!();
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
