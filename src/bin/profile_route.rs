//! Timing breakdown benchmark — measures each phase of route_multi.
//!
//! Usage:
//!   cargo run --release --bin profile_route

use asv_router::Router;
use std::time::{Duration, Instant};

fn main() {
    let mut router = Router::new();

    let intents: &[(&str, &[&str])] = &[
        ("billing_issue",        &["billing issue", "problem with my bill", "charge error"]),
        ("payment_history",      &["payment history", "billing history", "pull up my billing history"]),
        ("add_payment_method",   &["add payment method", "update payment", "change my card", "how do i update my payment method"]),
        ("cancel_subscription",  &["cancel my subscription", "cancel plan", "stop subscription"]),
        ("refund_request",       &["request a refund", "get my money back", "refund please"]),
        ("track_order",          &["where is my order", "track my package", "shipping status"]),
        ("return_item",          &["return this item", "send it back", "how to return"]),
        ("order_status",         &["order status", "check my order", "is my order ready"]),
        ("change_address",       &["change my address", "update delivery address", "new shipping address"]),
        ("apply_coupon",         &["apply coupon", "use promo code", "discount code"]),
        ("check_balance",        &["check my balance", "account balance", "how much do i have"]),
        ("account_login",        &["can't login", "forgot password", "reset my password"]),
        ("update_profile",       &["update my profile", "change my name", "edit account"]),
        ("product_question",     &["does this work with", "product details", "tell me about this product"]),
        ("product_availability", &["is this in stock", "when will it be available", "availability"]),
        ("subscription_status",  &["subscription status", "when does my plan expire", "active subscription"]),
        ("change_plan",          &["change my plan", "upgrade plan", "downgrade subscription"]),
        ("pause_subscription",   &["pause my subscription", "put on hold", "suspend account"]),
        ("request_invoice",      &["request invoice", "send me a receipt", "need a receipt"]),
        ("return_policy",        &["return policy", "how long to return", "can i return this"]),
        ("shipping_options",     &["shipping options", "how fast can i get it", "express delivery"]),
        ("cancel_order",         &["cancel my order", "cancel this purchase", "stop my order"]),
        ("exchange_item",        &["exchange this item", "swap for different size", "replace my order"]),
        ("loyalty_points",       &["loyalty points", "rewards balance", "how many points do i have"]),
        ("gift_card",            &["gift card balance", "use gift card", "redeem gift card"]),
        ("contact_support",      &["contact support", "talk to agent", "speak with someone"]),
        ("store_hours",          &["store hours", "when do you open", "are you open today"]),
        ("find_store",           &["find a store", "nearest location", "store near me"]),
        ("newsletter",           &["unsubscribe newsletter", "stop emails", "email preferences"]),
        ("privacy_data",         &["delete my data", "privacy request", "gdpr request"]),
        ("dispute_charge",       &["dispute a charge", "unauthorized charge", "fraudulent transaction"]),
        ("payment_failed",       &["payment failed", "card declined", "transaction error"]),
        ("set_reminder",         &["set a reminder", "remind me", "create alert"]),
        ("check_fees",           &["what are the fees", "any charges", "hidden fees"]),
        ("technical_issue",      &["app is broken", "not working", "technical problem"]),
        ("feedback",             &["give feedback", "leave a review", "rate my experience"]),
    ];

    for (id, phrases) in intents {
        router.add_intent(id, phrases);
    }

    let queries = &[
        "how do i update my payment method",
        "cancel my subscription",
        "where is my order",
        "i have a billing issue",
        "i want a refund please",
        "can i cancel and get a refund",
    ];

    // Warmup
    for q in queries.iter() {
        let _ = router.route_multi(q, 0.5);
    }

    // Time total route_multi
    let iters = 20_000usize;
    let mut total = Duration::ZERO;
    for i in 0..iters {
        let q = queries[i % queries.len()];
        let t = Instant::now();
        let _ = router.route_multi(q, 0.5);
        total += t.elapsed();
    }
    let avg_total = total / iters as u32;

    println!("=== Benchmark ({} iters, {} intents) ===", iters, intents.len());
    println!("  Avg route_multi: {:>6.1}µs", avg_total.as_secs_f64() * 1e6);
    println!();
    println!("Per-query breakdown:");
    for q in queries.iter() {
        let mut qt = Duration::ZERO;
        let n = 5000usize;
        for _ in 0..n {
            let t = Instant::now();
            let _ = router.route_multi(q, 0.5);
            qt += t.elapsed();
        }
        let avg = qt / n as u32;
        println!("  {:>6.1}µs  {:?}", avg.as_secs_f64() * 1e6, q);
    }
}
