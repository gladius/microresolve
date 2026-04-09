//! CJK seed guard test — verify collision detection works with
//! Chinese, Japanese, and Korean tokenization.
//!
//! Run: cargo test --test cjk_guard_test -- --nocapture

use asv_router::Router;

#[test]
fn cjk_chinese_seed_guard() {
    println!("\n=== Chinese Seed Guard Test ===\n");

    let mut router = Router::new();
    router.add_intent("cancel_order", &[
        "取消订单",        // cancel order
        "我要取消",        // I want to cancel
        "不要了",          // don't want it
    ]);
    router.add_intent("track_order", &[
        "查询订单",        // check order
        "我的包裹在哪",    // where is my package
        "物流状态",        // shipping status
    ]);
    router.add_intent("refund", &[
        "申请退款",        // apply for refund
        "退钱给我",        // return money to me
        "我要退款",        // I want a refund
    ]);

    // "订单" (order) should be shared across cancel_order and track_order
    // Adding it to refund should NOT be blocked (shared across 2+ intents)
    let result = router.check_seed("refund", "退款订单问题");  // refund order problem
    println!("  '退款订单问题' → refund:");
    println!("    conflicts: {:?}", result.conflicts.iter().map(|c| format!("{} in {}", c.term, c.competing_intent)).collect::<Vec<_>>());
    println!("    new_terms: {:?}", result.new_terms);
    println!("    redundant: {}", result.redundant);

    // "退款" (refund) is exclusive to refund intent
    // Adding it to cancel_order SHOULD be flagged
    let result = router.check_seed("cancel_order", "取消并退款");  // cancel and refund
    println!("\n  '取消并退款' → cancel_order:");
    println!("    conflicts: {:?}", result.conflicts.iter().map(|c| format!("{} in {}", c.term, c.competing_intent)).collect::<Vec<_>>());
    let has_refund_conflict = result.conflicts.iter().any(|c| c.competing_intent == "refund");
    println!("    refund collision detected: {}", has_refund_conflict);

    // Test add_seed_checked blocks the collision
    let result = router.add_seed_checked("cancel_order", "取消并退款", "zh");
    println!("\n  add_seed_checked '取消并退款' → cancel_order:");
    println!("    added: {}", result.added);
    println!("    warning: {:?}", result.warning);

    // Clean seed should pass
    let result = router.add_seed_checked("refund", "钱还没退回来", "zh");  // money hasn't been returned
    println!("\n  add_seed_checked '钱还没退回来' → refund:");
    println!("    added: {}", result.added);
    println!("    new_terms: {:?}", result.new_terms);

    // Verify routing still works
    let results = router.route("取消订单");
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "cancel_order");
    println!("\n  Routing '取消订单' → {} ✓", results[0].id);

    let results = router.route("我要退款");
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "refund");
    println!("  Routing '我要退款' → {} ✓", results[0].id);
}

#[test]
fn cjk_japanese_seed_guard() {
    println!("\n=== Japanese Seed Guard Test ===\n");

    let mut router = Router::new();
    router.add_intent("cancel_order", &[
        "注文をキャンセル",    // cancel the order
        "キャンセルしたい",    // I want to cancel
    ]);
    router.add_intent("track_order", &[
        "注文の追跡",          // track the order
        "荷物はどこですか",    // where is my package
    ]);
    router.add_intent("refund", &[
        "返金してください",    // please refund
        "お金を返して",        // return the money
    ]);

    // "返金" (refund) exclusive to refund — should flag when adding to cancel
    let result = router.check_seed("cancel_order", "キャンセルして返金");  // cancel and refund
    println!("  'キャンセルして返金' → cancel_order:");
    println!("    conflicts: {:?}", result.conflicts.iter().map(|c| format!("{} in {}", c.term, c.competing_intent)).collect::<Vec<_>>());

    // Clean addition should work
    let result = router.add_seed_checked("refund", "返品して返金", "ja");  // return and refund
    println!("\n  add_seed_checked '返品して返金' → refund:");
    println!("    added: {}", result.added);
    println!("    new_terms: {:?}", result.new_terms);
}

#[test]
fn cjk_korean_seed_guard() {
    println!("\n=== Korean Seed Guard Test ===\n");

    let mut router = Router::new();
    router.add_intent("cancel_order", &[
        "주문 취소",          // cancel order
        "취소하고 싶어요",    // I want to cancel
    ]);
    router.add_intent("refund", &[
        "환불 요청",          // refund request
        "돈 돌려주세요",      // return my money
    ]);

    // "환불" (refund) exclusive to refund
    let result = router.check_seed("cancel_order", "취소하고 환불");  // cancel and refund
    println!("  '취소하고 환불' → cancel_order:");
    println!("    conflicts: {:?}", result.conflicts.iter().map(|c| format!("{} in {}", c.term, c.competing_intent)).collect::<Vec<_>>());

    // Routing works
    let results = router.route("주문 취소");
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "cancel_order");
    println!("\n  Routing '주문 취소' → {} ✓", results[0].id);

    let results = router.route("환불 요청");
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "refund");
    println!("  Routing '환불 요청' → {} ✓", results[0].id);
}
