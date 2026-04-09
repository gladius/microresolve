//! Seed quality test: compare hand-written (poor) seeds vs prompt-guided (diverse) seeds
//! against the simulation dataset.
//!
//! Run: cargo run --release --bin seed_quality_test

use asv_router::Router;
use std::collections::{HashMap, HashSet};

#[derive(serde::Deserialize)]
struct Session {
    turns: Vec<Turn>,
}

#[derive(serde::Deserialize)]
struct Turn {
    message: String,
    intents: Vec<String>,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║        Seed Quality Impact Test (Simulation Data)       ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let sessions: Vec<Session> = serde_json::from_str(
        &std::fs::read_to_string("tests/data/simulation_sessions.json").unwrap()
    ).unwrap();

    let turns: Vec<&Turn> = sessions.iter().flat_map(|s| s.turns.iter()).collect();
    println!("  Test queries: {} turns from {} sessions\n", turns.len(), sessions.len());

    // === OLD SEEDS: tight vocabulary, no diversity ===
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  OLD SEEDS: Hand-written, tight vocabulary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let old_router = build_old_seeds();
    let (old_recall, old_precision, old_f1) = evaluate(&old_router, &turns);

    // === NEW SEEDS: following seed prompt guidelines ===
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  NEW SEEDS: Prompt-guided, vocabulary-diverse");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let new_router = build_diverse_seeds();
    let (new_recall, new_precision, new_f1) = evaluate(&new_router, &turns);

    // === COMPARISON ===
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  Recall:    {:.1}% → {:.1}% ({:+.1}%)", old_recall, new_recall, new_recall - old_recall);
    println!("  Precision: {:.1}% → {:.1}% ({:+.1}%)", old_precision, new_precision, new_precision - old_precision);
    println!("  F1:        {:.1}% → {:.1}% ({:+.1}%)", old_f1, new_f1, new_f1 - old_f1);
}

fn build_old_seeds() -> Router {
    let mut r = Router::new();

    // These are the current load_defaults style — tight vocabulary clusters
    r.add_intent("cancel_order", &[
        "cancel my order",
        "I need to cancel an order I just placed",
        "please stop my order from shipping",
        "cancel order number",
    ]);
    r.add_intent("refund", &[
        "I want a refund",
        "get my money back",
        "I received a damaged item and need a refund",
        "I want to return this for a full refund",
    ]);
    r.add_intent("contact_human", &[
        "talk to a human",
        "I need to speak with a real person not a bot",
        "connect me to customer service please",
        "transfer me to a representative",
    ]);
    r.add_intent("track_order", &[
        "where is my package",
        "track my order",
        "when will my delivery arrive",
        "package tracking number",
    ]);
    r.add_intent("billing_issue", &[
        "wrong charge on my account",
        "I was charged twice for the same order",
        "there's a billing error on my statement",
        "dispute a charge",
    ]);
    r.add_intent("return_item", &[
        "return this item",
        "I want to return my purchase",
        "send this product back",
        "initiate a return",
    ]);
    r.add_intent("shipping_complaint", &[
        "shipping is too slow",
        "my package is late",
        "delivery was supposed to arrive yesterday",
        "shipping complaint",
    ]);
    r.add_intent("product_inquiry", &[
        "product details",
        "tell me about this product",
        "what features does this have",
        "product specifications",
    ]);
    r.add_intent("order_status", &[
        "check my order status",
        "what is the status of my order",
        "order update",
        "has my order been processed",
    ]);
    r.add_intent("change_order", &[
        "change my order",
        "modify my order",
        "update my order details",
        "change the items in my order",
    ]);
    r.add_intent("account_issue", &[
        "account locked",
        "I can't log in",
        "my account has a problem",
        "account access issue",
    ]);
    r.add_intent("feedback", &[
        "I have feedback",
        "suggestion for improvement",
        "customer feedback",
        "I want to give feedback",
    ]);
    r.add_intent("payment_method", &[
        "update payment method",
        "change my credit card",
        "add a new payment option",
        "payment information",
    ]);
    r.add_intent("subscription", &[
        "cancel my subscription",
        "manage my subscription",
        "subscription settings",
        "unsubscribe",
    ]);

    r
}

fn build_diverse_seeds() -> Router {
    let mut r = Router::new();

    // Following the seed prompt guidelines: vocabulary diversity, situation descriptions,
    // emotional variants, formal/casual/frustrated, varied length

    r.add_intent("cancel_order", &[
        // Short
        "cancel order",
        "stop my purchase",
        // Medium
        "I changed my mind about this order",
        "found it cheaper somewhere else, undo this",
        "how do I cancel something I just bought",
        // Long/conversational
        "hey i accidentally ordered the wrong thing can you help me cancel it",
        "i dont want this anymore please withdraw my order before it ships",
        // Situational (no cancel keyword!)
        "I made a mistake with my purchase",
        "I no longer need this item please don't ship it",
    ]);

    r.add_intent("refund", &[
        // Short
        "money back",
        "refund please",
        // Medium
        "get my money back for this order",
        "how long until I get reimbursed",
        "the product was garbage I want compensation",
        // Long/conversational
        "i returned it two weeks ago and still haven't gotten my refund whats going on",
        "can I get my money back if I already opened the package",
        // Situational
        "this isn't what I paid for",
        "I was promised a full refund when will it arrive",
        "how long does the refund take to process",
    ]);

    r.add_intent("contact_human", &[
        // Short
        "real person",
        "live agent",
        // Medium
        "get me a human not a bot",
        "I want to talk to someone who can actually help",
        "transfer me to a manager",
        // Long/conversational
        "this bot isn't helping at all can you connect me to a real person please",
        "I've been going in circles, I need to speak with an actual human being",
        // Emotional/frustrated
        "THIS IS UNACCEPTABLE get me someone NOW",
        "escalate this immediately",
    ]);

    r.add_intent("track_order", &[
        // Short
        "where is my package",
        "track order",
        // Medium
        "when will my stuff arrive",
        "how long until I get my product",
        "can you check on my delivery",
        // Long/conversational
        "i ordered something last week and it still hasnt shown up, whats going on",
        "my order was supposed to arrive 3 days ago can you look into it",
        // Situational (no track/order keywords!)
        "it was a birthday gift and its already late",
        "when am I getting my purchase delivered",
        "estimated arrival for my shipment",
        "how many days until it gets here",
    ]);

    r.add_intent("billing_issue", &[
        // Short
        "wrong charge",
        "charged twice",
        // Medium
        "I see an incorrect amount on my statement",
        "you guys charged me double for one order",
        "there's a billing error on my credit card",
        // Long/conversational
        "i just checked my amex and there are two charges for the same thing",
        "I was charged $299 for something I never ordered this looks like fraud",
        // Situational
        "the amount doesn't match what I agreed to pay",
        "my card was hit with a charge I don't recognize",
    ]);

    r.add_intent("return_item", &[
        // Short
        "return this",
        "send it back",
        // Medium
        "I want to return the item I received",
        "this product is defective I need to send it back",
        "you sent me the wrong thing I need a return label",
        // Long/conversational
        "I ordered a blue jacket size M and you sent me a red hoodie size XL this is ridiculous",
        "how do I return something I bought online",
        // Situational (no return keyword!)
        "this isn't what I ordered",
        "I got the wrong item and want to send it back",
        "the product doesn't match the description at all",
    ]);

    r.add_intent("shipping_complaint", &[
        // Short
        "late delivery",
        "shipping sucks",
        // Medium
        "my package is way overdue",
        "delivery was supposed to come yesterday",
        "this is taking way too long to arrive",
        // Long/conversational
        "its been 10 days and my order still hasnt arrived this is unacceptable",
        "the tracking says delivered but I never got anything",
        // Situational/emotional
        "i paid extra for express and its still not here",
        "this is ridiculous how slow shipping is",
        "worst delivery experience ever",
    ]);

    r.add_intent("product_inquiry", &[
        // Short
        "product question",
        "tell me about this",
        // Medium
        "does this come with a carrying case",
        "what are the specs on this item",
        "is this compatible with my device",
        // Long/conversational
        "hey quick question does this product come in other colors or just the ones shown",
        "I'm thinking about buying this but wanted to know more about the features",
        // Situational
        "do I need to buy accessories separately",
        "what's the difference between the standard and premium version",
    ]);

    r.add_intent("order_status", &[
        // Short
        "order update",
        "status check",
        // Medium
        "what's happening with my order",
        "has my purchase shipped yet",
        "can you tell me when my order will be ready",
        // Long/conversational
        "I placed an order 3 days ago and havent gotten any update or confirmation email",
        "just want to know if my order has been processed and when itll ship",
        // Situational
        "did my payment go through for that order",
        "any news on my recent purchase",
    ]);

    r.add_intent("change_order", &[
        // Short
        "modify order",
        "change my order",
        // Medium
        "can I switch the size on my order",
        "I want to change the color I selected",
        "update the shipping address on my purchase",
        // Long/conversational
        "wait actually can I just change the size instead I ordered large but need medium",
        "i need to swap one of the items before it ships can you help",
        // Situational
        "I picked the wrong option during checkout",
        "add another item to my existing order",
    ]);

    r.add_intent("account_issue", &[
        // Short
        "locked out",
        "cant login",
        // Medium
        "my account is locked and I cant access it",
        "I keep getting an error when trying to sign in",
        "something is wrong with my account",
        // Long/conversational
        "i think i put the wrong password too many times and now my account is locked can you help",
        "I was just trying to check my order and now I cant get into my account at all",
        // Situational
        "the login page keeps saying invalid credentials",
        "my account got suspended for no reason",
    ]);

    r.add_intent("feedback", &[
        // Short
        "just wanted to say thanks",
        "great service",
        // Medium
        "overall I'm happy with the experience",
        "I have a suggestion for your website",
        "the support was really helpful today",
        // Long/conversational
        "wanted to let you know that last time the agent who helped me was amazing, total five stars",
        "your app could really use a dark mode honestly",
        // Negative feedback
        "your service has gone downhill",
        "this used to be better",
    ]);

    r.add_intent("payment_method", &[
        // Short
        "update card",
        "change payment",
        // Medium
        "I need to use a different credit card",
        "my card on file is expired can I update it",
        "switch to paying with PayPal instead",
        // Long/conversational
        "yeah its showing on my amex ending in 4421 can you switch it to my visa",
        "I want to add my new debit card and remove the old one",
        // Situational
        "the card you have is no longer valid",
        "I got a new bank card with different numbers",
    ]);

    r.add_intent("subscription", &[
        // Short
        "cancel subscription",
        "unsubscribe",
        // Medium
        "I want to cancel my premium membership",
        "stop charging me monthly",
        "I'm not using the service enough to keep paying",
        // Long/conversational
        "hi I'd like to cancel my premium subscription I'm not using it enough to justify the cost",
        "no thanks I don't want a discount just cancel it effective immediately please",
        // Situational
        "I don't want to renew next month",
        "just end my membership",
    ]);

    r
}

fn evaluate(router: &Router, turns: &[&Turn]) -> (f32, f32, f32) {
    let mut total_recall = 0.0f64;
    let mut total_precision = 0.0f64;
    let mut total_turns = 0;
    let mut exact_match = 0;

    let mut per_intent_hit: HashMap<String, usize> = HashMap::new();
    let mut per_intent_total: HashMap<String, usize> = HashMap::new();

    for turn in turns {
        let output = router.route_multi(&turn.message, 0.3);
        let detected: HashSet<String> = output.intents.iter().map(|i| i.id.clone()).collect();
        let expected: HashSet<String> = turn.intents.iter().cloned().collect();

        let overlap = detected.intersection(&expected).count();

        let recall = if expected.is_empty() { 1.0 } else { overlap as f64 / expected.len() as f64 };
        let precision = if detected.is_empty() { 0.0 } else { overlap as f64 / detected.len() as f64 };

        total_recall += recall;
        total_precision += precision;
        total_turns += 1;

        if detected == expected { exact_match += 1; }

        for intent in &turn.intents {
            *per_intent_total.entry(intent.clone()).or_insert(0) += 1;
            if detected.contains(intent) {
                *per_intent_hit.entry(intent.clone()).or_insert(0) += 1;
            }
        }
    }

    let avg_recall = total_recall / total_turns as f64 * 100.0;
    let avg_precision = total_precision / total_turns as f64 * 100.0;
    let f1 = if avg_recall + avg_precision > 0.0 { 2.0 * avg_recall * avg_precision / (avg_recall + avg_precision) } else { 0.0 };
    let exact_pct = exact_match as f32 / total_turns as f32 * 100.0;

    println!("  Exact match: {}/{} = {:.1}%", exact_match, total_turns, exact_pct);
    println!("  Avg recall:    {:.1}%", avg_recall);
    println!("  Avg precision: {:.1}%", avg_precision);
    println!("  F1:            {:.1}%", f1);

    println!("\n  Per-intent recall:");
    let mut stats: Vec<_> = per_intent_total.iter().collect();
    stats.sort_by(|a, b| b.1.cmp(a.1));
    for (intent, total) in &stats {
        let hit = per_intent_hit.get(intent.as_str()).copied().unwrap_or(0);
        let pct = if **total > 0 { hit as f32 / **total as f32 * 100.0 } else { 0.0 };
        println!("    {:<25} {}/{} = {:.0}%", intent, hit, total, pct);
    }

    (avg_recall as f32, avg_precision as f32, f1 as f32)
}
