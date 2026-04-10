//! Router unit tests.

use asv_router::*;
use std::collections::HashMap;


#[test]
fn basic_routing() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &[
        "cancel my order",
        "I want to cancel",
        "stop my order",
    ]);
    router.add_intent("track_order", &[
        "where is my package",
        "track my order",
        "shipping status",
    ]);

    let result = router.route("I need to cancel something");
    assert!(!result.is_empty());
    assert_eq!(result[0].id, "cancel_order");

    let result = router.route("where is my package");
    assert_eq!(result[0].id, "track_order");
}

#[test]
fn learning_improves_routing() {
    let mut router = Router::new();
    router.add_intent("cancel_sub", &["cancel subscription"]);

    // Before learning: "stop charging me" has no term overlap
    let before = router.route("stop charging me");
    let cancel_before = before.iter().find(|r| r.id == "cancel_sub");

    // Learn the mapping
    router.learn("stop charging me", "cancel_sub");

    // After learning: should route correctly
    let after = router.route("stop charging me");
    assert!(!after.is_empty());
    assert_eq!(after[0].id, "cancel_sub");

    if let Some(cb) = cancel_before {
        assert!(after[0].score > cb.score);
    }
}

#[test]
fn correction_moves_signal() {
    let mut router = Router::new();
    router.add_intent("cancel", &["cancel order"]);
    router.add_intent("refund", &["get refund"]);

    router.learn("I want my money back", "cancel");
    router.correct("I want my money back", "cancel", "refund");

    let result = router.route("I want my money back");
    assert_eq!(result[0].id, "refund");
}

#[test]
fn route_best_with_threshold() {
    let mut router = Router::new();
    router.add_intent("greet", &["hello", "hi there"]);

    assert!(router.route_best("hello", 0.1).is_some());
    assert!(router.route_best("quantum physics", 0.1).is_none());
}

#[test]
fn remove_intent() {
    let mut router = Router::new();
    router.add_intent("a", &["cancel order"]);
    router.add_intent("b", &["track order"]);

    router.remove_intent("a");
    assert_eq!(router.intent_count(), 1);

    let result = router.route("cancel");
    assert!(result.is_empty() || result[0].id != "a");
}

#[test]
fn export_import_roundtrip() {
    let mut router = Router::new();
    router.add_intent("cancel", &["cancel my order", "stop order"]);
    router.learn("drop my order", "cancel");

    let json = router.export_json();
    let restored = Router::import_json(&json).unwrap();

    let result = restored.route("cancel my order");
    assert_eq!(result[0].id, "cancel");

    let result = restored.route("drop my order");
    assert!(!result.is_empty());
    assert_eq!(result[0].id, "cancel");
}

#[test]
fn empty_router_returns_empty() {
    let router = Router::new();
    assert!(router.route("anything").is_empty());
}

#[test]
fn all_stop_words_returns_empty() {
    let mut router = Router::new();
    router.add_intent("a", &["cancel"]);
    assert!(router.route("the a an in on at to").is_empty());
}

#[test]
fn learn_creates_new_intent() {
    let mut router = Router::new();
    router.learn("reset password", "password_reset");
    assert_eq!(router.intent_count(), 1);

    let result = router.route("reset password");
    assert_eq!(result[0].id, "password_reset");
}

// --- CJK routing tests ---

#[test]
fn cjk_chinese_basic_routing() {
    let mut router = Router::new();
    // Space-separated seeds (as LLM would provide)
    router.add_intent("cancel_order", &[
        "取消 订单",
        "我 要 取消",
        "退订",
    ]);
    router.add_intent("track_order", &[
        "查看 订单",
        "物流 状态",
        "快递 到 哪里",
    ]);

    // Query: "我想取消我的订单" (I want to cancel my order)
    let result = router.route("我想取消我的订单");
    assert!(!result.is_empty(), "should match CJK query");
    assert_eq!(result[0].id, "cancel_order");
}

#[test]
fn cjk_japanese_basic_routing() {
    let mut router = Router::new();
    router.add_intent("cancel", &[
        "キャンセル",
        "取り消し",
    ]);
    router.add_intent("track", &[
        "追跡",
        "配送 状況",
    ]);

    let result = router.route("キャンセルしたい");
    assert!(!result.is_empty(), "should match Japanese query");
    assert_eq!(result[0].id, "cancel");
}

#[test]
fn cjk_four_char_idiom() {
    let mut router = Router::new();
    // Test 4-character compound term (automaton handles any length)
    router.add_intent("complaint", &[
        "莫名其妙",
        "投诉",
    ]);
    router.add_intent("praise", &[
        "非常满意",
        "好评",
    ]);

    let result = router.route("这个服务莫名其妙");
    assert!(!result.is_empty());
    assert_eq!(result[0].id, "complaint");
}

#[test]
fn cjk_mixed_language_query() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &[
        "cancel my order",
        "取消 订单",
    ]);

    // Mixed: "I want to 取消订单"
    let result = router.route("I want to 取消订单");
    assert!(!result.is_empty());
    assert_eq!(result[0].id, "cancel_order");
}

#[test]
fn cjk_learning() {
    let mut router = Router::new();
    router.add_intent("refund", &[
        "退款",
        "退钱",
    ]);

    // Before learning: "要回我的钱" has no seed match
    let before = router.route("要回我的钱");
    let _had_refund = before.iter().any(|r| r.id == "refund");

    // Learn the phrase
    router.learn("要回我的钱", "refund");

    // After learning: should route to refund
    let after = router.route("要回我的钱");
    assert!(!after.is_empty());
    assert_eq!(after[0].id, "refund");
}

#[test]
fn cjk_negation_routing() {
    let mut router = Router::new();
    router.add_intent("cancel", &["取消", "退订"]);
    router.add_intent("track", &["查看", "追踪"]);

    // "不取消" — negation should suppress 取消
    let result = router.route("不取消");
    // 取消 is negated, so cancel intent should not be top
    let cancel_score = result.iter().find(|r| r.id == "cancel").map(|r| r.score).unwrap_or(0.0);
    // Without the negated term, cancel shouldn't score
    assert_eq!(cancel_score, 0.0, "negated term should not score");
}

#[test]
fn cjk_multi_intent() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &["取消 订单", "退订"]);
    router.add_intent("check_balance", &["查看 余额", "账户 余额"]);

    let result = router.route_multi("取消订单然后查看余额", 0.3);
    assert!(result.intents.len() >= 2, "should detect 2 intents, got {}", result.intents.len());

    let ids: Vec<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
    assert!(ids.contains(&"cancel_order"), "missing cancel_order in {:?}", ids);
    assert!(ids.contains(&"check_balance"), "missing check_balance in {:?}", ids);

    // Should detect sequential relation from 然后
    if !result.relations.is_empty() {
        assert!(
            matches!(result.relations[0], IntentRelation::Sequential { .. }),
            "expected Sequential from 然后, got {:?}", result.relations[0]
        );
    }
}

#[test]
fn cjk_unsegmented_seeds() {
    // LLM might generate seeds without spaces — tokenizer must still produce
    // character bigrams so the automaton can find substrings in queries
    let mut router = Router::new();
    router.add_intent("save_recipe", &[
        "保存食谱",         // unsegmented: "save recipe"
        "保存我的食谱",     // unsegmented: "save my recipe"
    ]);

    // Query with those characters embedded in longer text
    let result = router.route("你能帮我保存一下食谱吗");
    assert!(!result.is_empty(), "should match unsegmented CJK seeds");
    assert_eq!(result[0].id, "save_recipe");
}

#[test]
fn cjk_export_import_roundtrip() {
    let mut router = Router::new();
    router.add_intent("cancel", &["取消 订单"]);
    router.learn("退订服务", "cancel");

    let json = router.export_json();
    let restored = Router::import_json(&json).unwrap();

    let result = restored.route("取消订单");
    assert!(!result.is_empty());
    assert_eq!(result[0].id, "cancel");
}

#[test]
fn cjk_multi_intent_chinese_long() {
    // Chinese customer rant with multiple intents buried in complaint
    let mut router = Router::new();
    router.add_intent("cancel_order", &["取消 订单", "退订", "取消 购买"]);
    router.add_intent("refund", &["退款", "退钱", "把 钱 退 给 我"]);
    router.add_intent("track_order", &["查 订单", "物流 查询", "包裹 在 哪"]);
    router.add_intent("complaint", &["投诉", "不满意", "差评"]);
    router.add_intent("contact_human", &["转 人工", "找 客服", "人工 服务"]);
    router.add_intent("check_balance", &["查看 余额", "账户 余额"]);

    // Short: 2 intents
    let r = router.route_multi("取消订单然后退款", 0.3);
    let ids: Vec<&str> = r.intents.iter().map(|i| i.id.as_str()).collect();
    assert!(ids.contains(&"cancel_order"), "short: missing cancel_order, got {:?}", ids);
    assert!(ids.contains(&"refund"), "short: missing refund, got {:?}", ids);

    // Medium: 3 intents
    let r = router.route_multi("我要取消订单并且退款还要投诉你们的服务", 0.3);
    let ids: Vec<&str> = r.intents.iter().map(|i| i.id.as_str()).collect();
    assert!(ids.contains(&"cancel_order"), "medium: missing cancel_order, got {:?}", ids);
    assert!(ids.contains(&"refund"), "medium: missing refund, got {:?}", ids);
    assert!(ids.contains(&"complaint"), "medium: missing complaint, got {:?}", ids);

    // Long rant: should not exceed max_intents (5)
    let r = router.route_multi(
        "你们这个服务太差了我等了一个星期包裹还没到现在我要退款而且我要取消所有的订单以后再也不买了我要投诉你们还要找你们的客服经理来处理这个问题查看一下我的账户余额",
        0.3
    );
    assert!(r.intents.len() <= 5, "CJK long rant: {} intents exceeds cap of 5", r.intents.len());
}

#[test]
fn cjk_multi_intent_japanese() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &["注文 キャンセル", "注文 取り消し"]);
    router.add_intent("refund", &["返金", "払い戻し"]);
    router.add_intent("track_order", &["配送 状況", "荷物 追跡"]);
    router.add_intent("complaint", &["苦情", "クレーム"]);
    router.add_intent("contact_human", &["オペレーター", "担当者"]);

    // Short: 2 intents
    let r = router.route_multi("注文キャンセルして返金してください", 0.3);
    let ids: Vec<&str> = r.intents.iter().map(|i| i.id.as_str()).collect();
    assert!(ids.contains(&"cancel_order"), "JP short: missing cancel_order, got {:?}", ids);
    assert!(ids.contains(&"refund"), "JP short: missing refund, got {:?}", ids);

    // Long: should not exceed cap
    let r = router.route_multi(
        "もう本当にひどいです一週間も待っているのに荷物がまだ届きません返金してください注文もキャンセルしたいですそれからクレームを入れたいのでオペレーターに繋いでください",
        0.3
    );
    assert!(r.intents.len() <= 5, "JP long: {} intents exceeds cap of 5", r.intents.len());
}

#[test]
fn cjk_multi_intent_korean() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &["주문 취소", "취소 하다"]);
    router.add_intent("refund", &["환불", "돈 돌려주다"]);
    router.add_intent("track_order", &["배송 조회", "택배 추적"]);
    router.add_intent("complaint", &["불만", "항의"]);

    // Short: 2 intents
    let r = router.route_multi("주문취소하고 환불해주세요", 0.3);
    let ids: Vec<&str> = r.intents.iter().map(|i| i.id.as_str()).collect();
    assert!(ids.contains(&"cancel_order"), "KR short: missing cancel_order, got {:?}", ids);
    assert!(ids.contains(&"refund"), "KR short: missing refund, got {:?}", ids);

    // Long: cap applies
    let r = router.route_multi(
        "정말 화가 납니다 일주일이나 기다렸는데 배송조회도 안되고 환불도 안해주고 주문취소도 안되고 불만이 너무 많습니다",
        0.3
    );
    assert!(r.intents.len() <= 5, "KR long: {} intents exceeds cap of 5", r.intents.len());
}

#[test]
fn cjk_mixed_multi_intent() {
    // Mixed CJK + Latin in same query
    let mut router = Router::new();
    router.add_intent("cancel_order", &["取消 订单", "cancel order"]);
    router.add_intent("refund", &["退款", "refund"]);
    router.add_intent("track_order", &["查 物流", "track package"]);

    let r = router.route_multi("我要cancel我的订单还要退款", 0.3);
    let ids: Vec<&str> = r.intents.iter().map(|i| i.id.as_str()).collect();
    assert!(ids.contains(&"cancel_order") || ids.contains(&"refund"),
        "mixed: should detect at least one intent, got {:?}", ids);
}

#[test]
fn max_intents_configurable() {
    let mut router = Router::new();
    router.add_intent("a", &["alpha bravo"]);
    router.add_intent("b", &["charlie delta"]);
    router.add_intent("c", &["echo foxtrot"]);
    router.add_intent("d", &["golf hotel"]);

    // Default cap is 5, all 4 should be detected
    let r = router.route_multi("alpha bravo charlie delta echo foxtrot golf hotel", 0.1);
    assert_eq!(r.intents.len(), 4);

    // Set cap to 2
    router.set_max_intents(2);
    let r = router.route_multi("alpha bravo charlie delta echo foxtrot golf hotel", 0.1);
    assert_eq!(r.intents.len(), 2, "cap at 2 should limit to 2 intents, got {}", r.intents.len());

    // Cap persists through export/import
    let json = router.export_json();
    let restored = Router::import_json(&json).unwrap();
    assert_eq!(restored.max_intents(), 2);
}

#[test]
fn many_intents_still_fast() {
    let mut router = Router::new();
    for i in 0..500 {
        router.add_intent(
            &format!("intent_{}", i),
            &[&format!("action_{} thing_{}", i, i)],
        );
    }

    let result = router.route("action_42 thing_42");
    assert!(!result.is_empty());
    assert_eq!(result[0].id, "intent_42");
}

// --- Prerequisite tests ---

#[test]
fn intent_type_default_is_action() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &["cancel my order"]);
    assert_eq!(router.get_intent_type("cancel_order"), IntentType::Action);
}

#[test]
fn intent_type_set_and_get() {
    let mut router = Router::new();
    router.add_intent("check_balance", &["check my balance"]);
    router.set_intent_type("check_balance", IntentType::Context);
    assert_eq!(router.get_intent_type("check_balance"), IntentType::Context);
}

#[test]
fn intent_type_in_route_multi_output() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &["cancel my order", "I want to cancel"]);
    router.add_intent("check_balance", &["check my balance", "account balance"]);
    router.set_intent_type("check_balance", IntentType::Context);

    let result = router.route_multi("cancel my order and check my balance", 0.3);
    assert!(result.intents.len() >= 2);
    let cancel = result.intents.iter().find(|i| i.id == "cancel_order").unwrap();
    let balance = result.intents.iter().find(|i| i.id == "check_balance").unwrap();
    assert_eq!(cancel.intent_type, IntentType::Action);
    assert_eq!(balance.intent_type, IntentType::Context);
}

#[test]
fn metadata_set_and_get() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &["cancel my order"]);
    router.set_metadata("cancel_order", "context_intents", vec!["check_balance".into(), "track_order".into()]);
    router.set_metadata("cancel_order", "action_intents", vec!["refund".into()]);

    let meta = router.get_metadata("cancel_order").unwrap();
    assert_eq!(meta.get("context_intents").unwrap(), &vec!["check_balance".to_string(), "track_order".to_string()]);
    assert_eq!(meta.get("action_intents").unwrap(), &vec!["refund".to_string()]);
}

#[test]
fn metadata_key_lookup() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &["cancel my order"]);
    router.set_metadata("cancel_order", "context_intents", vec!["check_balance".into()]);

    assert_eq!(router.get_metadata_key("cancel_order", "context_intents").unwrap(), &vec!["check_balance".to_string()]);
    assert!(router.get_metadata_key("cancel_order", "nonexistent").is_none());
    assert!(router.get_metadata_key("nonexistent", "context_intents").is_none());
}

#[test]
fn metadata_in_route_multi_output() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &["cancel my order", "I want to cancel"]);
    router.add_intent("track_order", &["where is my package", "track my order"]);
    router.set_metadata("cancel_order", "context_intents", vec!["check_balance".into()]);
    router.set_metadata("track_order", "context_intents", vec!["get_shipping_info".into()]);

    let result = router.route_multi("cancel my order and track my package", 0.3);
    assert!(result.intents.len() >= 2);
    let cancel_meta = result.metadata.get("cancel_order").unwrap();
    assert_eq!(cancel_meta.get("context_intents").unwrap(), &vec!["check_balance".to_string()]);
}

#[test]
fn intent_type_and_metadata_persist_through_export_import() {
    let mut router = Router::new();
    router.add_intent("refund", &["refund my order"]);
    router.set_intent_type("refund", IntentType::Context);
    router.set_metadata("refund", "context_intents", vec!["check_balance".into()]);
    router.set_metadata("refund", "team", vec!["billing".into()]);

    let json = router.export_json();
    let restored = Router::import_json(&json).unwrap();
    assert_eq!(restored.get_intent_type("refund"), IntentType::Context);
    assert_eq!(restored.get_metadata_key("refund", "context_intents").unwrap(), &vec!["check_balance".to_string()]);
    assert_eq!(restored.get_metadata_key("refund", "team").unwrap(), &vec!["billing".to_string()]);
}

#[test]
fn remove_intent_cleans_type_and_metadata() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &["cancel my order"]);
    router.set_intent_type("cancel_order", IntentType::Context);
    router.set_metadata("cancel_order", "team", vec!["ops".into()]);

    router.remove_intent("cancel_order");
    assert_eq!(router.get_intent_type("cancel_order"), IntentType::Action); // default
    assert!(router.get_metadata("cancel_order").is_none());
}

#[test]
fn co_occurrence_tracking() {
    let mut router = Router::new();
    router.record_co_occurrence(&["cancel_order", "refund"]);
    router.record_co_occurrence(&["cancel_order", "refund"]);
    router.record_co_occurrence(&["cancel_order", "track_order"]);

    let pairs = router.get_co_occurrence();
    assert_eq!(pairs[0], ("cancel_order", "refund", 2));
    assert_eq!(pairs[1], ("cancel_order", "track_order", 1));
}

#[test]
fn suggest_intents_from_co_occurrence() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &["cancel my order"]);
    router.add_intent("refund", &["get a refund", "money back"]);
    router.add_intent("track_order", &["track my package", "where is my order"]);
    router.add_intent("complaint", &["file a complaint"]);

    // Simulate traffic: cancel+refund appear together 10 times,
    // cancel+track 3 times, cancel+complaint 1 time
    for _ in 0..10 {
        router.record_co_occurrence(&["cancel_order", "refund"]);
    }
    for _ in 0..3 {
        router.record_co_occurrence(&["cancel_order", "track_order"]);
    }
    router.record_co_occurrence(&["cancel_order", "complaint"]);

    // When cancel_order is detected, suggest refund (high prob) and track_order (moderate)
    let suggestions = router.suggest_intents(&["cancel_order"], 3, 0.2);
    assert!(!suggestions.is_empty(), "should have suggestions");

    // refund should be top suggestion (10/14 = 0.71)
    assert_eq!(suggestions[0].id, "refund");
    assert!(suggestions[0].probability > 0.6, "refund probability should be >0.6, got {}", suggestions[0].probability);
    assert_eq!(suggestions[0].observations, 10);
    assert_eq!(suggestions[0].because_of, "cancel_order");

    // track_order should be second (3/14 = 0.21)
    assert_eq!(suggestions[1].id, "track_order");
    assert!(suggestions[1].probability > 0.15);

    // complaint should NOT appear (only 1 observation, below min_observations=3)
    assert!(suggestions.iter().all(|s| s.id != "complaint"),
        "complaint should not be suggested (only 1 observation)");

    // Already-detected intents should not be suggested
    let suggestions = router.suggest_intents(&["cancel_order", "refund"], 3, 0.2);
    assert!(suggestions.iter().all(|s| s.id != "refund"),
        "refund should not be suggested when already detected");
}

#[test]
fn suggestions_in_route_multi() {
    let mut router = Router::new();
    router.add_intent("cancel_order", &["cancel my order", "stop my order"]);
    router.add_intent("refund", &["get a refund", "money back", "refund my purchase"]);
    router.add_intent("track_order", &["track my package", "where is my order"]);

    // Build co-occurrence: cancel_order + refund always together
    for _ in 0..20 {
        router.record_co_occurrence(&["cancel_order", "refund"]);
    }

    // Route a query that only triggers cancel_order
    let result = router.route_multi("cancel my order please", 0.3);
    let detected_ids: Vec<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
    assert!(detected_ids.contains(&"cancel_order"), "should detect cancel_order");

    // refund should appear as a suggestion since it wasn't detected but co-occurs
    if !detected_ids.contains(&"refund") {
        assert!(!result.suggestions.is_empty(), "should suggest refund when cancel_order is detected alone");
        assert_eq!(result.suggestions[0].id, "refund");
    }
}

#[test]
fn version_increments_on_mutation() {
    let mut router = Router::new();
    assert_eq!(router.version(), 0);

    router.add_intent("cancel", &["cancel my order"]);
    assert_eq!(router.version(), 1);

    router.learn("stop that", "cancel");
    assert_eq!(router.version(), 2);

    router.add_intent("track", &["track my order"]);
    assert_eq!(router.version(), 3);

    router.remove_intent("track");
    assert_eq!(router.version(), 4);
}

#[test]
fn merge_learned_combines_weights() {
    let mut router_a = Router::new();
    router_a.add_intent("cancel", &["cancel my order"]);
    router_a.add_intent("track", &["track my order"]);
    router_a.learn("stop that purchase", "cancel");

    let mut router_b = Router::new();
    router_b.add_intent("cancel", &["cancel my order"]);
    router_b.add_intent("track", &["track my order"]);
    router_b.learn("where is my stuff", "track");

    // Before merge: router_a doesn't know "stuff", router_b doesn't know "stop"
    let before_a = router_a.route("where is my stuff");
    let before_track_score = before_a.iter().find(|r| r.id == "track").map(|r| r.score).unwrap_or(0.0);

    router_a.merge_learned(&router_b);

    // After merge: router_a should know "stuff" from router_b's learning
    let after_a = router_a.route("where is my stuff");
    let after_track_score = after_a.iter().find(|r| r.id == "track").map(|r| r.score).unwrap_or(0.0);
    assert!(after_track_score > before_track_score,
        "merge should improve track score for 'stuff': before={}, after={}", before_track_score, after_track_score);
}

#[test]
fn merge_is_idempotent() {
    let mut router_a = Router::new();
    router_a.add_intent("cancel", &["cancel my order"]);
    router_a.learn("stop it", "cancel");

    let mut router_b = Router::new();
    router_b.add_intent("cancel", &["cancel my order"]);
    router_b.learn("halt order", "cancel");

    router_a.merge_learned(&router_b);
    let score_after_first = router_a.route("halt order")[0].score;

    router_a.merge_learned(&router_b); // same merge again
    let score_after_second = router_a.route("halt order")[0].score;

    assert!((score_after_first - score_after_second).abs() < 0.001,
        "merge should be idempotent: first={}, second={}", score_after_first, score_after_second);
}

#[test]
fn export_learned_only_is_lightweight() {
    let mut router = Router::new();
    router.add_intent("cancel", &["cancel my order"]);
    router.add_intent("track", &["track my order"]);
    router.learn("stop it", "cancel");

    let learned_json = router.export_learned_only();
    let parsed: HashMap<String, HashMap<String, f32>> =
        serde_json::from_str(&learned_json).unwrap();

    // Only "cancel" should appear (it has learned terms), not "track"
    assert!(parsed.contains_key("cancel"));
    assert!(!parsed.contains_key("track"));
}

#[test]
fn import_learned_merge_roundtrip() {
    let mut router_a = Router::new();
    router_a.add_intent("cancel", &["cancel my order"]);
    router_a.learn("stop it", "cancel");

    let learned_json = router_a.export_learned_only();

    let mut router_b = Router::new();
    router_b.add_intent("cancel", &["cancel my order"]);

    // Before import: router_b doesn't know "stop"
    let before = router_b.route("stop it");
    let before_score = before.iter().find(|r| r.id == "cancel").map(|r| r.score).unwrap_or(0.0);

    router_b.import_learned_merge(&learned_json).unwrap();

    // After import: router_b should know "stop" from router_a
    let after = router_b.route("stop it");
    let after_score = after.iter().find(|r| r.id == "cancel").map(|r| r.score).unwrap_or(0.0);
    assert!(after_score > before_score,
        "import_learned_merge should improve score: before={}, after={}", before_score, after_score);
}

#[test]
fn co_occurrence_survives_export_import() {
    let mut router = Router::new();
    router.record_co_occurrence(&["cancel_order", "refund"]);
    router.record_co_occurrence(&["cancel_order", "refund"]);

    let json = router.export_json();
    let restored = Router::import_json(&json).unwrap();

    let pairs = restored.get_co_occurrence();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0], ("cancel_order", "refund", 2));
}

#[test]
fn temporal_ordering_tracks_direction() {
    let mut router = Router::new();
    // "cancel" appears before "refund" 3 times
    router.record_intent_sequence(&["cancel_order", "refund"]);
    router.record_intent_sequence(&["cancel_order", "refund"]);
    router.record_intent_sequence(&["cancel_order", "refund"]);
    // "refund" appears before "cancel" 1 time
    router.record_intent_sequence(&["refund", "cancel_order"]);

    let order = router.get_temporal_order();
    // cancel_order → refund should dominate (3 vs 1)
    let cancel_first = order.iter().find(|(a, b, _, _)| *a == "cancel_order" && *b == "refund");
    assert!(cancel_first.is_some(), "should find cancel_order → refund ordering");
    let (_, _, prob, count) = cancel_first.unwrap();
    assert_eq!(*count, 3);
    // Probability = 3/4 co-occurrences (lexicographic key stores total=4)
    assert!(*prob > 0.5, "cancel_order should appear before refund with high probability: {}", prob);
}

#[test]
fn temporal_order_survives_export_import() {
    let mut router = Router::new();
    router.record_intent_sequence(&["cancel_order", "refund", "contact_human"]);
    router.record_intent_sequence(&["cancel_order", "refund"]);

    let json = router.export_json();
    let restored = Router::import_json(&json).unwrap();

    let order = restored.get_temporal_order();
    assert!(!order.is_empty(), "temporal ordering should survive export/import");
    // cancel_order → refund should exist
    let cancel_refund = order.iter().find(|(a, b, _, _)| *a == "cancel_order" && *b == "refund");
    assert!(cancel_refund.is_some());
}

#[test]
fn discover_workflows_finds_clusters() {
    let mut router = Router::new();
    // Cluster 1: cancel + refund + complaint (frequent)
    for _ in 0..5 {
        router.record_intent_sequence(&["cancel_order", "refund"]);
        router.record_intent_sequence(&["refund", "complaint"]);
        router.record_intent_sequence(&["cancel_order", "complaint"]);
    }
    // Cluster 2: track + shipping_status (separate)
    for _ in 0..5 {
        router.record_intent_sequence(&["track_order", "shipping_status"]);
    }

    let workflows = router.discover_workflows(3);
    assert!(workflows.len() >= 2, "should find at least 2 clusters, got {}", workflows.len());

    // Largest cluster should have 3 intents (cancel, refund, complaint)
    let largest = &workflows[0];
    assert_eq!(largest.len(), 3, "largest cluster should have 3 intents: {:?}",
        largest.iter().map(|w| &w.id).collect::<Vec<_>>());

    // Each intent in largest cluster should have neighbors
    for wi in largest {
        assert!(!wi.neighbors.is_empty(), "{} should have neighbors", wi.id);
    }
}

#[test]
fn detect_escalation_patterns_finds_sequences() {
    let mut router = Router::new();
    // Recurring escalation: track → complaint → contact_human
    for _ in 0..5 {
        router.record_intent_sequence(&["track_order", "complaint", "contact_human"]);
    }
    // Another pattern: cancel → refund
    for _ in 0..3 {
        router.record_intent_sequence(&["cancel_order", "refund"]);
    }
    // Noise
    router.record_intent_sequence(&["check_balance"]);

    let patterns = router.detect_escalation_patterns(3);
    assert!(!patterns.is_empty(), "should find escalation patterns");

    // The track → complaint → contact_human triple should appear
    let escalation = patterns.iter().find(|p|
        p.sequence == vec!["track_order", "complaint", "contact_human"]
    );
    assert!(escalation.is_some(), "should find track→complaint→contact_human pattern");
    assert_eq!(escalation.unwrap().occurrences, 5);

    // cancel → refund pair should appear
    let cancel_refund = patterns.iter().find(|p|
        p.sequence == vec!["cancel_order", "refund"]
    );
    assert!(cancel_refund.is_some(), "should find cancel→refund pattern");
    assert_eq!(cancel_refund.unwrap().occurrences, 3);
}



#[test]
fn router_config_defaults() {
    let config = RouterConfig::default();
    assert_eq!(config.top_k, 10);
    assert_eq!(config.max_intents, 5);
    assert!(config.server.is_none());
    assert_eq!(config.app_id, "default");
    assert!(config.data_path.is_none());
    assert_eq!(config.sync_interval_secs, 30);
}

#[test]
fn router_with_config() {
    let r = Router::with_config(RouterConfig {
        top_k: 3,
        max_intents: 8,
        ..Default::default()
    });
    // top_k is private, tested through routing behavior
    assert_eq!(r.max_intents(), 8);
    assert!(!r.is_connected());
}

#[test]
fn connected_mode_blocks_writes() {
    let r = Router::with_config(RouterConfig {
        server: Some("http://localhost:3001".to_string()),
        ..Default::default()
    });
    assert!(r.is_connected());
}

#[test]
#[should_panic(expected = "connected mode")]
fn connected_mode_panics_on_add_intent() {
    let mut r = Router::with_config(RouterConfig {
        server: Some("http://localhost:3001".to_string()),
        ..Default::default()
    });
    r.add_intent("test", &["test phrase"]);
}

#[test]
#[should_panic(expected = "connected mode")]
fn connected_mode_panics_on_learn() {
    let mut r = Router::with_config(RouterConfig {
        server: Some("http://localhost:3001".to_string()),
        ..Default::default()
    });
    r.learn("test query", "test_intent");
}


#[test]
fn save_and_load_file() {
    let mut r = Router::new();
    r.add_intent("test", &["test phrase"]);

    let path = "/tmp/asv_test_save.json";
    r.save(path).unwrap();

    let r2 = Router::load(path).unwrap();
    let results = r2.route("test phrase");
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "test");

    std::fs::remove_file(path).ok();
}


#[test]
fn remove_phrase() {
    let mut r = Router::new();
    r.add_intent("cancel", &["cancel my order", "stop my order", "I want to cancel"]);

    // Remove one phrase
    assert!(r.remove_phrase("cancel", "stop my order"));

    // Verify it's gone from training
    let training = r.get_training("cancel").unwrap();
    assert_eq!(training.len(), 2);
    assert!(!training.contains(&"stop my order".to_string()));

    // Routing still works with remaining phrases
    let result = r.route("cancel my order");
    assert!(!result.is_empty());

    // Remove nonexistent phrase returns false
    assert!(!r.remove_phrase("cancel", "nonexistent phrase"));

    // Remove from nonexistent intent returns false
    assert!(!r.remove_phrase("nonexistent", "cancel my order"));
}

#[test]
fn remove_last_seed_removes_intent() {
    let mut r = Router::new();
    r.add_intent("test", &["only seed"]);

    assert!(r.remove_phrase("test", "only seed"));
    assert_eq!(r.intent_count(), 0);
}

// --- Phrase Guard Tests ---

#[test]
fn check_phrase_new_terms_safe() {
    let mut r = Router::new();
    r.add_intent("refund", &["I want a refund", "money back"]);
    r.add_intent("track_order", &["where is my package", "track my order"]);

    // "reimburse me" has terms not in any intent — should be safe
    let result = r.check_phrase("refund", "reimburse me please");
    assert!(!result.redundant);
    assert!(result.new_terms.contains(&"reimburse".to_string()));
    assert!(result.conflicts.is_empty());
}

#[test]
fn check_phrase_redundant() {
    let mut r = Router::new();
    r.add_intent("refund", &["I want a refund", "money back"]);

    // "refund money" — both terms already in refund intent
    let result = r.check_phrase("refund", "refund money");
    assert!(result.redundant);
    assert!(result.warning.is_some());
}

#[test]
fn check_phrase_detects_collision() {
    let mut r = Router::new();
    r.add_intent("refund", &["I want a refund", "money back"]);
    r.add_intent("payment_method", &[
        "update my visa card",
        "change payment to visa",
        "visa card on file",
    ]);

    // "refund to visa" — "visa" is primarily in payment_method
    let result = r.check_phrase("refund", "refund to visa");
    assert!(!result.conflicts.is_empty());

    let visa_conflict = result.conflicts.iter().find(|c| c.term == "visa");
    assert!(visa_conflict.is_some(), "should detect visa collision");
    let vc = visa_conflict.unwrap();
    assert_eq!(vc.competing_intent, "payment_method");
    assert!(vc.severity > 0.5, "visa should be primarily in payment_method");
}

#[test]
fn check_phrase_no_collision_for_shared_low_weight_terms() {
    let mut r = Router::new();
    r.add_intent("cancel_order", &["cancel my order", "stop my order"]);
    r.add_intent("track_order", &["track my order", "where is my order"]);
    r.add_intent("change_order", &["change my order", "modify my order"]);

    // "order" is spread across 3 intents — low discrimination, no collision warning
    let result = r.check_phrase("cancel_order", "order status please");
    let order_conflicts: Vec<_> = result.conflicts.iter()
        .filter(|c| c.term == "order")
        .collect();
    // "order" is distributed, not concentrated — should not flag as high severity
    for c in &order_conflicts {
        assert!(c.severity <= 0.5,
            "shared term 'order' should not have high severity: {} in {}",
            c.severity, c.competing_intent);
    }
}

#[test]
fn check_phrase_empty_after_stop_words() {
    let mut r = Router::new();
    r.add_intent("refund", &["I want a refund"]);

    // "I want to" — all stop words
    let result = r.check_phrase("refund", "I want to");
    assert!(result.warning.is_some());
    assert!(result.new_terms.is_empty());
}

#[test]
fn add_phrase_checked_skips_redundant() {
    let mut r = Router::new();
    r.add_intent("refund", &["I want a refund", "money back"]);

    let result = r.add_phrase_checked("refund", "refund money", "en");
    assert!(!result.added);
    assert!(result.redundant);
}

#[test]
fn add_phrase_checked_blocks_collision() {
    let mut r = Router::new();
    r.add_intent("refund", &["I want a refund", "money back"]);
    r.add_intent("payment_method", &[
        "update my visa card",
        "change payment to visa",
        "visa card on file",
    ]);

    let result = r.add_phrase_checked("refund", "refund to visa", "en");
    // Should block — "visa" conflicts with payment_method
    assert!(!result.added);
    assert!(!result.conflicts.is_empty());
    assert!(result.warning.is_some());
}

#[test]
fn add_phrase_checked_clean_addition() {
    let mut r = Router::new();
    r.add_intent("refund", &["I want a refund", "money back"]);
    r.add_intent("track_order", &["where is my package"]);

    let result = r.add_phrase_checked("refund", "reimburse my purchase", "en");
    assert!(result.added);
    assert!(result.conflicts.is_empty());
    assert!(!result.redundant);
    assert!(result.warning.is_none());
    assert!(!result.new_terms.is_empty());
}

#[test]
fn seed_guard_does_not_block_learn() {
    // learn() should still work regardless of collisions
    // because user corrections are ground truth
    let mut r = Router::new();
    r.add_intent("refund", &["I want a refund"]);
    r.add_intent("payment_method", &["update my visa card"]);

    // Even though "visa" is in payment_method, learning from a real
    // user query should work
    r.learn("refund back to my visa", "refund");
    let results = r.route("refund back to my visa");
    assert_eq!(results[0].id, "refund");
}

#[test]
fn check_phrase_with_realistic_ecommerce_intents() {
    let mut r = Router::new();
    r.add_intent("cancel_order", &[
        "cancel my order",
        "I changed my mind",
        "stop my purchase",
    ]);
    r.add_intent("refund", &[
        "I want a refund",
        "get my money back",
        "reimburse me",
    ]);
    r.add_intent("track_order", &[
        "where is my package",
        "when will it arrive",
        "delivery status",
    ]);
    r.add_intent("billing_issue", &[
        "charged twice",
        "wrong charge on my card",
        "billing error",
    ]);
    r.add_intent("return_item", &[
        "return this product",
        "send it back",
        "wrong item received",
    ]);

    // Test 1: "cancel and refund" to cancel_order — "refund" should collide
    let result = r.check_phrase("cancel_order", "cancel and get refund");
    let refund_collision = result.conflicts.iter()
        .any(|c| c.term == "refund" && c.competing_intent == "refund");
    assert!(refund_collision, "should detect 'refund' collision with refund intent");

    // Test 2: "delivery is late" to track_order — "delivery" already in track_order
    let result = r.check_phrase("track_order", "delivery is late");
    // "delivery" is from "delivery status" phrase — already in track_order
    // "late" is new — this should not be redundant
    assert!(!result.redundant);

    // Test 3: "wrong charge" to refund — "wrong" and "charge" are in billing_issue
    let result = r.check_phrase("refund", "wrong charge refund");
    let billing_conflict = result.conflicts.iter()
        .any(|c| c.competing_intent == "billing_issue");
    assert!(billing_conflict, "should detect collision with billing_issue");

    // Test 4: completely novel phrase — no collisions
    let result = r.check_phrase("refund", "compensate me for the inconvenience");
    assert!(result.conflicts.is_empty());
    assert!(!result.redundant);
}

#[test]
fn seed_guard_preserves_routing_accuracy() {
    // After adding phrases through add_phrase_checked, routing should still work correctly
    let mut r = Router::new();
    r.add_intent("cancel_order", &["cancel my order", "stop my purchase", "cancel it now"]);
    r.add_intent("track_order", &["where is my package", "track my order", "shipping status"]);
    r.add_intent("refund", &["get my money back", "full refund", "reimburse me"]);

    // Add clean phrases
    r.add_phrase_checked("cancel_order", "I changed my mind about buying this", "en");
    r.add_phrase_checked("refund", "return the payment to my account", "en");

    // Routing should still work for clear queries
    let cancel_results = r.route("cancel my order please");
    assert_eq!(cancel_results[0].id, "cancel_order");

    let refund_results = r.route("give me a full refund");
    assert_eq!(refund_results[0].id, "refund");

    let track_results = r.route("where is my package");
    assert_eq!(track_results[0].id, "track_order");
}

#[test]
fn seed_guard_similar_intents_shared_terms() {
    // cancel_order and cancel_subscription both legitimately use "cancel"
    let mut r = Router::new();
    r.add_intent("cancel_order", &["cancel my order", "stop my purchase"]);
    r.add_intent("cancel_subscription", &["cancel my subscription", "end my membership"]);

    // Adding "cancel my service" to cancel_subscription should NOT be blocked.
    // "cancel" is already shared across 2 intents — it's a known shared term.
    let result = r.check_phrase("cancel_subscription", "cancel my service");
    let cancel_conflict = result.conflicts.iter().any(|c| c.term == "cancel");
    assert!(!cancel_conflict,
        "shared term 'cancel' across similar intents should not flag as collision");
}

#[test]
fn seed_guard_exclusive_term_still_blocked() {
    // "visa" is exclusive to payment_method — should still be blocked
    let mut r = Router::new();
    r.add_intent("refund", &["I want a refund", "money back"]);
    r.add_intent("payment_method", &["update my visa card", "change payment to visa"]);

    let result = r.check_phrase("refund", "refund to visa");
    let visa_conflict = result.conflicts.iter().any(|c| c.term == "visa");
    assert!(visa_conflict, "exclusive term 'visa' should still be blocked");
}

#[test]
fn seed_guard_three_way_shared_term_not_blocked() {
    // "order" in 3 intents — definitely shared, should not block
    let mut r = Router::new();
    r.add_intent("cancel_order", &["cancel my order"]);
    r.add_intent("track_order", &["track my order"]);
    r.add_intent("change_order", &["change my order"]);

    // Adding "order update" to a new intent should not flag "order"
    r.add_intent("order_status", &["order status"]);
    let result = r.check_phrase("order_status", "check order progress");
    let order_conflict = result.conflicts.iter().any(|c| c.term == "order");
    assert!(!order_conflict,
        "term 'order' shared across 3+ intents should not flag");
}
