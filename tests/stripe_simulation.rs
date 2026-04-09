//! Stripe API simulation — real-world seed guard stress test.
//!
//! Creates intents from Stripe's actual API operations (payments, refunds,
//! subscriptions, invoices, disputes, customers, charges). These domains
//! heavily overlap in vocabulary. Tests whether the guard helps or hurts.
//!
//! Run: cargo test --test stripe_simulation -- --nocapture

use asv_router::Router;

#[test]
fn stripe_full_lifecycle() {
    println!("\n============================================================");
    println!("  STRIPE API SIMULATION — Seed Guard Stress Test");
    println!("============================================================\n");

    let mut router = Router::new();

    // Phase 1: Create intents from Stripe's actual API domains
    // These are based on Stripe's real API operations and documentation
    println!("━━━ Phase 1: Create 20 Stripe-like intents ━━━\n");

    let intents: Vec<(&str, Vec<&str>)> = vec![
        ("create_payment", vec![
            "process a payment",
            "charge the customer",
            "create a new payment intent",
            "accept credit card payment",
            "initiate a transaction",
        ]),
        ("refund_payment", vec![
            "refund this payment",
            "reverse the charge",
            "issue a refund to customer",
            "return money to buyer",
            "process a refund",
        ]),
        ("create_subscription", vec![
            "set up a subscription",
            "start recurring billing",
            "create a monthly plan",
            "enroll customer in subscription",
            "activate subscription plan",
        ]),
        ("cancel_subscription", vec![
            "cancel the subscription",
            "stop recurring billing",
            "end the subscription plan",
            "terminate membership",
            "unsubscribe the customer",
        ]),
        ("update_subscription", vec![
            "change subscription plan",
            "upgrade the subscription",
            "modify billing frequency",
            "switch to annual plan",
            "downgrade subscription tier",
        ]),
        ("create_invoice", vec![
            "generate an invoice",
            "create billing statement",
            "send invoice to customer",
            "issue an invoice",
            "prepare invoice for payment",
        ]),
        ("pay_invoice", vec![
            "pay the outstanding invoice",
            "settle the invoice balance",
            "process invoice payment",
            "collect payment on invoice",
            "mark invoice as paid",
        ]),
        ("void_invoice", vec![
            "void this invoice",
            "cancel the unpaid invoice",
            "nullify the billing statement",
            "remove the invoice",
            "discard draft invoice",
        ]),
        ("create_customer", vec![
            "add a new customer",
            "register customer account",
            "create customer profile",
            "onboard new client",
            "set up customer record",
        ]),
        ("update_customer", vec![
            "update customer details",
            "change customer email",
            "modify customer information",
            "edit customer profile",
            "correct customer address",
        ]),
        ("delete_customer", vec![
            "delete the customer record",
            "remove customer from system",
            "permanently erase customer data",
            "deactivate customer account",
            "close customer profile",
        ]),
        ("create_dispute", vec![
            "file a dispute",
            "open a chargeback case",
            "initiate payment dispute",
            "contest a charge",
            "raise a dispute claim",
        ]),
        ("respond_dispute", vec![
            "respond to the dispute",
            "submit dispute evidence",
            "provide chargeback documentation",
            "upload proof for dispute",
            "defend against chargeback",
        ]),
        ("add_payment_method", vec![
            "add a credit card",
            "save new payment method",
            "register bank account",
            "link debit card",
            "store payment credentials",
        ]),
        ("remove_payment_method", vec![
            "remove the credit card",
            "delete payment method",
            "unlink bank account",
            "detach card from customer",
            "remove stored payment",
        ]),
        ("create_payout", vec![
            "send a payout",
            "transfer funds to bank",
            "initiate bank transfer",
            "withdraw balance to account",
            "process seller payout",
        ]),
        ("check_balance", vec![
            "check account balance",
            "view available funds",
            "show current balance",
            "how much is in the account",
            "remaining balance amount",
        ]),
        ("list_transactions", vec![
            "show recent transactions",
            "list all charges",
            "view transaction history",
            "display payment log",
            "pull up charge records",
        ]),
        ("apply_coupon", vec![
            "apply a discount code",
            "use promotional coupon",
            "redeem voucher",
            "activate discount",
            "enter promo code",
        ]),
        ("create_price", vec![
            "set up pricing",
            "create a price tier",
            "define product price",
            "configure billing amount",
            "establish rate plan",
        ]),
    ];

    let mut creation_conflicts: Vec<(String, String, String)> = Vec::new(); // (intent, term, conflicting_intent)

    for (id, seeds) in &intents {
        let results = router.add_intent(id, &seeds.iter().map(|s| *s).collect::<Vec<&str>>());
        for result in results.iter() {
            for conflict in &result.conflicts {
                creation_conflicts.push((
                    id.to_string(),
                    conflict.term.clone(),
                    conflict.competing_intent.clone(),
                ));
            }
        }
    }

    println!("  Created {} intents", intents.len());
    println!("  Collisions detected during creation: {}", creation_conflicts.len());
    for (intent, term, other) in &creation_conflicts {
        println!("    {} ← '{}' conflicts with {}", intent, term, other);
    }

    // Phase 2: Baseline routing accuracy with realistic queries
    println!("\n━━━ Phase 2: Route realistic queries ━━━\n");

    let test_queries: Vec<(&str, &str)> = vec![
        // Clear matches
        ("charge the customer's card", "create_payment"),
        ("I need to refund this transaction", "refund_payment"),
        ("set up monthly billing for this user", "create_subscription"),
        ("the customer wants to cancel their plan", "cancel_subscription"),
        ("generate an invoice for last month", "create_invoice"),
        ("add a new credit card to the account", "add_payment_method"),
        ("show me the transaction history", "list_transactions"),
        ("apply the 20% off coupon", "apply_coupon"),

        // Ambiguous / cross-domain
        ("process the payment on this invoice", "pay_invoice"),  // payment + invoice
        ("refund the subscription charge", "refund_payment"),    // refund + subscription
        ("cancel and refund the customer", "cancel_subscription"), // cancel + refund + customer
        ("update the card on file", "add_payment_method"),       // update + card
        ("dispute this refund", "create_dispute"),               // dispute + refund
        ("delete the customer and cancel everything", "delete_customer"), // delete + cancel
        ("send money to the seller's bank", "create_payout"),    // money + bank
        ("check if the invoice was paid", "pay_invoice"),        // check + invoice + paid
        ("change the subscription payment method", "update_subscription"), // change + subscription + payment
        ("remove the expired card", "remove_payment_method"),    // remove + card
        ("how much does the customer owe", "check_balance"),     // customer + owe
        ("void the invoice and refund", "void_invoice"),         // void + invoice + refund
    ];

    let mut correct = 0;
    let mut wrong = 0;
    let mut failures: Vec<(String, String, String)> = Vec::new(); // (query, expected, got)

    for (query, expected) in &test_queries {
        let results = router.route(query);
        let got = results.first().map(|r| r.id.as_str()).unwrap_or("(none)");
        if got == *expected {
            correct += 1;
        } else {
            wrong += 1;
            failures.push((query.to_string(), expected.to_string(), got.to_string()));
        }
    }

    println!("  Queries: {} | Correct: {} | Wrong: {} | Accuracy: {:.0}%",
        test_queries.len(), correct, wrong,
        correct as f64 / test_queries.len() as f64 * 100.0);

    if !failures.is_empty() {
        println!("\n  Failures:");
        for (q, expected, got) in &failures {
            println!("    \"{}\"", q);
            println!("      expected={}, got={}", expected, got);
        }
    }

    // Phase 3: Try adding seeds through the guard
    println!("\n━━━ Phase 3: Add seeds through guard ━━━\n");

    let fix_seeds: Vec<(&str, &str)> = vec![
        // Good seeds — should pass
        ("refund_payment", "reimburse the transaction"),
        ("cancel_subscription", "stop the recurring charges"),
        ("create_invoice", "bill the client"),
        ("check_balance", "what funds are available"),
        ("create_payout", "wire money to seller"),

        // Cross-domain seeds — may collide
        ("refund_payment", "cancel the charge"),          // "cancel" → cancel_subscription?
        ("pay_invoice", "charge the invoice amount"),     // "charge" → create_payment?
        ("void_invoice", "refund the invoice"),           // "refund" → refund_payment?
        ("update_subscription", "change payment plan"),   // "payment" → create_payment?
        ("create_dispute", "dispute the refund decision"),// "refund" → refund_payment?
        ("delete_customer", "remove customer payment data"), // "payment" → ?
        ("add_payment_method", "update credit card"),     // "update" → update_customer?
        ("cancel_subscription", "refund remaining balance"), // "refund" + "balance" → ?
        ("create_payment", "bill customer card"),         // "bill" → create_invoice?
        ("respond_dispute", "provide refund evidence"),   // "refund" → refund_payment?
    ];

    let mut added = 0;
    let mut blocked = 0;
    let mut blocked_details: Vec<(String, String, String)> = Vec::new();

    for (intent, seed) in &fix_seeds {
        let result = router.add_seed_checked(intent, seed, "en");
        if result.added {
            added += 1;
            let new = if result.new_terms.is_empty() { String::new() }
                else { format!(" (new: {})", result.new_terms.join(", ")) };
            println!("  ADDED: \"{}\" → {}{}", seed, intent, new);
        } else {
            blocked += 1;
            let reason = if result.redundant {
                "redundant".to_string()
            } else if !result.conflicts.is_empty() {
                result.conflicts.iter()
                    .map(|c| format!("'{}' conflicts with {}", c.term, c.competing_intent))
                    .collect::<Vec<_>>().join("; ")
            } else {
                result.warning.unwrap_or("unknown".to_string())
            };
            blocked_details.push((seed.to_string(), intent.to_string(), reason.clone()));
            println!("  BLOCKED: \"{}\" → {} — {}", seed, intent, reason);
        }
    }

    println!("\n  Added: {} | Blocked: {}", added, blocked);

    // Phase 4: Accuracy after fixes
    println!("\n━━━ Phase 4: Accuracy after fixes ━━━\n");

    let mut correct_after = 0;
    let mut failures_after: Vec<(String, String, String)> = Vec::new();

    for (query, expected) in &test_queries {
        let results = router.route(query);
        let got = results.first().map(|r| r.id.as_str()).unwrap_or("(none)");
        if got == *expected {
            correct_after += 1;
        } else {
            failures_after.push((query.to_string(), expected.to_string(), got.to_string()));
        }
    }

    println!("  Before: {}/{} ({:.0}%)", correct, test_queries.len(),
        correct as f64 / test_queries.len() as f64 * 100.0);
    println!("  After:  {}/{} ({:.0}%)", correct_after, test_queries.len(),
        correct_after as f64 / test_queries.len() as f64 * 100.0);

    let delta = correct_after as i32 - correct as i32;
    if delta > 0 {
        println!("  Improved by {} queries", delta);
    } else if delta < 0 {
        println!("  DEGRADED by {} queries!", -delta);
    } else {
        println!("  No change");
    }

    if !failures_after.is_empty() {
        println!("\n  Remaining failures:");
        for (q, expected, got) in &failures_after {
            println!("    \"{}\" expected={}, got={}", q, expected, got);
        }
    }

    // Phase 5: Verify no intent lost its primary routing
    println!("\n━━━ Phase 5: Primary intent routing preserved? ━━━\n");

    let primary_queries: Vec<(&str, &str)> = vec![
        ("process a payment", "create_payment"),
        ("refund this payment", "refund_payment"),
        ("create a subscription", "create_subscription"),
        ("cancel the subscription", "cancel_subscription"),
        ("update subscription plan", "update_subscription"),
        ("create an invoice", "create_invoice"),
        ("pay the invoice", "pay_invoice"),
        ("void the invoice", "void_invoice"),
        ("add a new customer", "create_customer"),
        ("update customer info", "update_customer"),
        ("delete the customer", "delete_customer"),
        ("file a dispute", "create_dispute"),
        ("respond to dispute", "respond_dispute"),
        ("add payment method", "add_payment_method"),
        ("remove payment method", "remove_payment_method"),
        ("send a payout", "create_payout"),
        ("check balance", "check_balance"),
        ("list transactions", "list_transactions"),
        ("apply coupon", "apply_coupon"),
        ("set up pricing", "create_price"),
    ];

    let mut primary_correct = 0;
    let mut primary_broken = Vec::new();

    for (query, expected) in &primary_queries {
        let results = router.route(query);
        let got = results.first().map(|r| r.id.clone()).unwrap_or_else(|| "(none)".to_string());
        if got == *expected {
            primary_correct += 1;
        } else {
            primary_broken.push((query.to_string(), expected.to_string(), got));
        }
    }

    println!("  Primary routing: {}/{} correct", primary_correct, primary_queries.len());
    if !primary_broken.is_empty() {
        println!("  BROKEN:");
        for (q, expected, got) in &primary_broken {
            println!("    \"{}\" expected={}, got={}", q, expected, got);
        }
    }

    // Assertions
    assert!(primary_correct >= 18, "Primary routing should be >= 90% ({}/{})", primary_correct, primary_queries.len());
    assert!(correct_after >= correct, "Accuracy should not degrade after guarded seed addition");
}
