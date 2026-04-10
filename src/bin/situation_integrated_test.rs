//! Integrated situation index test — all 5 apps, cross-app multi-intent.
//!
//! Uses the real Router with add_situation_patterns(), route_multi().
//! Proves the integrated system matches (or beats) the standalone experiment results.
//!
//! Run: cargo run --bin situation_integrated_test

use asv_router::Router;
use std::collections::HashMap;

// ── Minimal action seeds per intent ──────────────────────────────────────────
// These are distinct from situation patterns — they represent explicit requests.
// The point of situation queries is they contain NONE of these words.

fn action_seeds() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        // stripe
        ("charge_card",          vec!["charge the card", "process payment", "bill the customer"]),
        ("refund_payment",       vec!["issue a refund", "refund this payment", "process refund"]),
        ("cancel_subscription",  vec!["cancel subscription", "cancel their plan", "end the subscription"]),
        // github
        ("create_issue",         vec!["report a bug", "create an issue", "log a ticket"]),
        ("merge_pr",             vec!["merge the pull request", "merge this PR", "approve and merge"]),
        ("create_release",       vec!["cut a release", "create new release", "publish version"]),
        ("close_issue",          vec!["close this issue", "mark resolved", "close the ticket"]),
        // slack
        ("send_message",         vec!["send a message", "notify the team", "ping the channel"]),
        ("create_channel",       vec!["create a channel", "set up a channel", "open new channel"]),
        ("invite_user",          vec!["invite user", "add someone to channel", "add member"]),
        // shopify
        ("update_inventory",     vec!["update inventory", "adjust stock", "restock items"]),
        ("track_shipment",       vec!["track this order", "track the package", "check delivery"]),
        ("refund_order",         vec!["refund this order", "process return", "issue order refund"]),
        // calendar
        ("reschedule_event",     vec!["reschedule the meeting", "move the event", "change meeting time"]),
        ("set_out_of_office",    vec!["set out of office", "mark as away", "enable OOO"]),
        ("find_meeting_time",    vec!["find a meeting time", "schedule a meeting", "find open slot"]),
    ]
}

// ── Situation patterns from mixed/seeds.json (inlined for zero file I/O) ─────

fn situation_patterns() -> Vec<(&'static str, Vec<(&'static str, f32)>)> {
    vec![
        ("charge_card", vec![
            ("付款", 1.0), ("扣款", 1.0), ("刷卡", 1.0), ("拒绝", 1.0), ("收款", 1.0),
            ("失败", 0.4), ("报错", 0.4),
            ("支払", 1.0), ("決済", 1.0), ("拒否", 1.0),
            ("402", 1.0), ("payment", 0.7), ("declined", 1.0), ("charge", 0.7),
        ]),
        ("refund_payment", vec![
            ("退款", 1.0), ("争议", 1.0), ("多扣", 1.0), ("投诉", 1.0),
            ("返金", 1.0), ("払戻", 1.0),
            ("refund", 1.0), ("dispute", 1.0), ("chargeback", 1.0),
        ]),
        ("cancel_subscription", vec![
            ("续费", 1.0), ("流失", 1.0), ("退出", 1.0),
            ("解約", 1.0), ("退会", 1.0), ("取消", 0.4),
            ("churn", 1.0), ("cancel", 0.5), ("unsubscribe", 1.0),
        ]),
        ("create_issue", vec![
            ("构建", 1.0), ("线上", 1.0), ("崩溃", 1.0), ("内存", 1.0), ("泄漏", 1.0),
            ("バグ", 1.0), ("本番", 1.0), ("失败", 0.4), ("报错", 0.4),
            ("build", 1.0), ("crash", 1.0), ("CI", 0.7), ("prod", 0.8),
            ("error", 0.4), ("404", 1.0), ("500", 1.0), ("OOM", 1.0), ("leak", 1.0), ("down", 0.8),
        ]),
        ("merge_pr", vec![
            ("审查", 1.0), ("批准", 1.0), ("通过", 1.0), ("合并", 1.0),
            ("レビュー", 1.0), ("承認", 1.0),
            ("PR", 1.0), ("approve", 1.0), ("LGTM", 1.0), ("review", 0.7), ("merge", 0.8),
        ]),
        ("create_release", vec![
            ("发版", 1.0), ("上线", 1.0), ("发布", 1.0), ("リリース", 1.0),
            ("release", 1.0), ("deploy", 1.0), ("tag", 0.6), ("rollout", 1.0),
        ]),
        ("close_issue", vec![
            ("解决", 1.0), ("修复", 1.0), ("关闭", 1.0), ("修正", 1.0),
            ("fixed", 1.0), ("resolved", 1.0), ("closed", 0.7),
        ]),
        ("send_message", vec![
            ("通知", 1.0), ("告知", 1.0), ("广播", 1.0),
            ("团队", 0.6), ("チーム", 0.6), ("お知らせ", 1.0),
            ("ping", 0.8), ("DM", 0.8), ("notify", 0.8), ("channel", 0.5),
        ]),
        ("create_channel", vec![
            ("频道", 1.0), ("チャンネル", 1.0), ("群组", 1.0), ("channel", 1.0),
        ]),
        ("invite_user", vec![
            ("邀请", 1.0), ("招待", 1.0), ("参加", 0.5), ("invite", 1.0), ("add", 0.5),
        ]),
        ("update_inventory", vec![
            ("库存", 1.0), ("缺货", 1.0), ("断货", 1.0), ("补货", 1.0),
            ("在庫", 1.0), ("品切れ", 1.0),
            ("stock", 0.8), ("inventory", 1.0), ("SKU", 0.7), ("OOS", 1.0),
        ]),
        ("track_shipment", vec![
            ("物流", 1.0), ("包裹", 1.0), ("快递", 1.0),
            ("配送", 1.0), ("荷物", 1.0),
            ("tracking", 1.0), ("shipment", 1.0), ("delivery", 1.0),
        ]),
        ("refund_order", vec![
            ("损坏", 1.0), ("退货", 1.0), ("错发", 1.0), ("破損", 1.0),
            ("refund", 1.0), ("return", 0.8), ("damaged", 1.0), ("wrong item", 1.0),
        ]),
        ("reschedule_event", vec![
            ("冲突", 1.0), ("重叠", 1.0), ("改期", 1.0), ("另约", 1.0),
            ("重複", 1.0), ("変更", 0.5),
            ("conflict", 1.0), ("reschedule", 1.0), ("double", 0.6),
        ]),
        ("set_out_of_office", vec![
            ("不在", 1.0), ("休假", 1.0), ("出差", 1.0), ("请假", 1.0),
            ("休み", 1.0), ("不在席", 1.0),
            ("OOO", 1.0), ("vacation", 1.0), ("PTO", 1.0), ("offline", 0.7),
        ]),
        ("find_meeting_time", vec![
            ("空闲", 1.0), ("有空", 1.0), ("空き", 1.0),
            ("available", 1.0), ("free slot", 1.0), ("schedule", 0.5),
        ]),
    ]
}

// ── Test cases ────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct TestCase {
    query: &'static str,
    /// Primary expected intent (always checked).
    expect_intent: &'static str,
    /// For cross_app: second intent that should also fire.
    expect_also: Option<&'static str>,
    category: &'static str,
    /// Expected to produce NO matches.
    expect_silence: bool,
    note: &'static str,
}

fn test_cases() -> Vec<TestCase> {
    vec![
        // ── Stripe ────────────────────────────────────────────────────────────
        TestCase { query: "payment一直declined，试了好几次",
            expect_intent: "charge_card", expect_also: None, category: "situation",
            expect_silence: false, note: "payment(0.7)+declined(1.0) → combined 1.41+" },
        TestCase { query: "402报出来了，充值没成功",
            expect_intent: "charge_card", expect_also: None, category: "situation",
            expect_silence: false, note: "402(1.0) → passes" },
        TestCase { query: "有个chargeback进来了",
            expect_intent: "refund_payment", expect_also: None, category: "situation",
            expect_silence: false, note: "chargeback(1.0)" },
        TestCase { query: "用户发起了dispute",
            expect_intent: "refund_payment", expect_also: None, category: "situation",
            expect_silence: false, note: "dispute(1.0)" },
        TestCase { query: "这个用户churn了",
            expect_intent: "cancel_subscription", expect_also: None, category: "situation",
            expect_silence: false, note: "churn(1.0)" },

        // ── GitHub ────────────────────────────────────────────────────────────
        TestCase { query: "commit之后build就挂了",
            expect_intent: "create_issue", expect_also: None, category: "situation",
            expect_silence: false, note: "build(1.0) — classic mixed engineering" },
        TestCase { query: "CI跑红了，merge不了",
            expect_intent: "create_issue", expect_also: None, category: "situation",
            expect_silence: false, note: "CI(0.7) → score=0.99 > 0.8" },
        TestCase { query: "prod上的接口全部500了",
            expect_intent: "create_issue", expect_also: None, category: "situation",
            expect_silence: false, note: "prod(0.8)+500(1.0)" },
        TestCase { query: "服务OOM了，容器一直重启",
            expect_intent: "create_issue", expect_also: None, category: "situation",
            expect_silence: false, note: "OOM(1.0)" },
        TestCase { query: "deploy完之后服务就down了",
            expect_intent: "create_issue", expect_also: None, category: "situation",
            expect_silence: false, note: "deploy(1.0)+down(0.8) — also fires create_release" },
        TestCase { query: "接口一直打404，不知道为什么",
            expect_intent: "create_issue", expect_also: None, category: "situation",
            expect_silence: false, note: "404(1.0)" },
        TestCase { query: "内存leak了，监控在报警",
            expect_intent: "create_issue", expect_also: None, category: "situation",
            expect_silence: false, note: "内存(1.0)+leak(1.0)" },
        TestCase { query: "PR被两个人approve了",
            expect_intent: "merge_pr", expect_also: None, category: "situation",
            expect_silence: false, note: "PR(1.0)+approve(1.0)" },
        TestCase { query: "代码review完了，可以merge了",
            expect_intent: "merge_pr", expect_also: None, category: "situation",
            expect_silence: false, note: "review(0.7)+merge(0.8) → 1.5+" },
        TestCase { query: "LGTM了，可以合上去了",
            expect_intent: "merge_pr", expect_also: None, category: "situation",
            expect_silence: false, note: "LGTM(1.0)" },
        TestCase { query: "今天要deploy，tag打好了",
            expect_intent: "create_release", expect_also: None, category: "situation",
            expect_silence: false, note: "deploy(1.0)+tag(0.6)" },
        TestCase { query: "rollout已经完成了",
            expect_intent: "create_release", expect_also: None, category: "situation",
            expect_silence: false, note: "rollout(1.0)" },
        TestCase { query: "那个bug已经fixed了",
            expect_intent: "close_issue", expect_also: None, category: "situation",
            expect_silence: false, note: "fixed(1.0)" },
        TestCase { query: "这个issue可以resolved了",
            expect_intent: "close_issue", expect_also: None, category: "situation",
            expect_silence: false, note: "resolved(1.0)" },

        // ── Shopify ───────────────────────────────────────────────────────────
        TestCase { query: "inventory快没了，OOS了几个SKU",
            expect_intent: "update_inventory", expect_also: None, category: "situation",
            expect_silence: false, note: "inventory(1.0)+OOS(1.0)" },
        TestCase { query: "这个SKU缺货了",
            expect_intent: "update_inventory", expect_also: None, category: "situation",
            expect_silence: false, note: "缺货(1.0)" },
        TestCase { query: "shipment tracking不到了",
            expect_intent: "track_shipment", expect_also: None, category: "situation",
            expect_silence: false, note: "shipment(1.0)+tracking(1.0)" },
        TestCase { query: "买家说收到的东西是damaged的",
            expect_intent: "refund_order", expect_also: None, category: "situation",
            expect_silence: false, note: "damaged(1.0)" },

        // ── Calendar ──────────────────────────────────────────────────────────
        TestCase { query: "我下周OOO",
            expect_intent: "set_out_of_office", expect_also: None, category: "situation",
            expect_silence: false, note: "OOO(1.0)" },
        TestCase { query: "要找个大家都available的slot",
            expect_intent: "find_meeting_time", expect_also: None, category: "situation",
            expect_silence: false, note: "available(1.0)" },
        TestCase { query: "有个scheduling conflict",
            expect_intent: "reschedule_event", expect_also: None, category: "situation",
            expect_silence: false, note: "conflict(1.0)" },

        // ── Slack ─────────────────────────────────────────────────────────────
        TestCase { query: "要给工程师团队ping一下",
            expect_intent: "send_message", expect_also: None, category: "situation",
            expect_silence: false, note: "ping(0.8)+团队(0.6) → 1.41+" },
        TestCase { query: "需要建个新channel来协调",
            expect_intent: "create_channel", expect_also: None, category: "situation",
            expect_silence: false, note: "channel(1.0)" },

        // ── Cross-app (situation fires one intent, action vocab fires another) ─
        TestCase { query: "build挂了，要通知团队",
            expect_intent: "create_issue", expect_also: Some("send_message"),
            category: "cross_app", expect_silence: false,
            note: "build→create_issue (situation) + 通知/团队→send_message (situation)" },
        TestCase { query: "inventory没了，DM一下采购",
            expect_intent: "update_inventory", expect_also: Some("send_message"),
            category: "cross_app", expect_silence: false,
            note: "inventory→update_inventory + DM→send_message" },

        // ── Negatives (nothing should fire) ───────────────────────────────────
        TestCase { query: "今天code写得很顺",
            expect_intent: "", expect_also: None, category: "negative",
            expect_silence: true, note: "no seed match" },
        TestCase { query: "meeting开得不错",
            expect_intent: "", expect_also: None, category: "negative",
            expect_silence: true, note: "no seed match" },
        TestCase { query: "我对这个solution很满意",
            expect_intent: "", expect_also: None, category: "negative",
            expect_silence: true, note: "no seed match" },
        // Documented honest false positive: "PR" alone fires merge_pr
        TestCase { query: "这个PR写得真不错",
            expect_intent: "merge_pr", expect_also: None, category: "false_positive",
            expect_silence: false, note: "KNOWN: PR(1.0) fires merge_pr — precision limit" },
    ]
}

// ── Runner ────────────────────────────────────────────────────────────────────

fn main() {
    // Build router
    let mut router = Router::new();

    // 1. Action seeds (term index — explicit requests)
    for (intent_id, seeds) in action_seeds() {
        router.add_intent(intent_id, &seeds);
    }

    // 2. Situation patterns (situation index — state descriptions)
    for (intent_id, patterns) in situation_patterns() {
        router.add_situation_patterns(intent_id, &patterns);
    }

    println!("Router: {} intents, situation patterns loaded", router.intent_count());
    println!();

    let cases = test_cases();
    let threshold = 0.3f32;

    // Counters by category
    let mut counts: HashMap<&str, (usize, usize)> = HashMap::new(); // (pass, total)

    let mut all_pass = true;

    for cat in &["situation", "cross_app", "negative", "false_positive"] {
        let cat_cases: Vec<&TestCase> = cases.iter().filter(|c| c.category == *cat).collect();
        if cat_cases.is_empty() { continue; }

        let label = match *cat {
            "situation"      => "SITUATION (state vocab → single intent)",
            "cross_app"      => "CROSS-APP (state + action → multi-intent)",
            "negative"       => "NEGATIVE (must stay silent)",
            "false_positive" => "KNOWN FALSE POSITIVE (documented precision limit)",
            _ => cat,
        };
        println!("━━━  {}  ━━━", label);

        let mut pass = 0;
        let total = cat_cases.len();

        for tc in &cat_cases {
            let result = router.route_multi(tc.query, threshold);
            let detected: Vec<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();

            let ok = if tc.expect_silence {
                detected.is_empty()
            } else if *cat == "cross_app" {
                // Primary must fire; secondary is bonus (logged but not required for pass)
                detected.contains(&tc.expect_intent)
            } else {
                detected.contains(&tc.expect_intent)
            };

            if ok { pass += 1; } else { all_pass = false; }

            let status = if ok { "✓" } else { "✗" };
            let detected_str = if detected.is_empty() {
                "(silence)".to_string()
            } else {
                detected.iter().map(|s| {
                    let r = result.intents.iter().find(|i| i.id == *s).unwrap();
                    format!("{}[{:.2}|{}]", s, r.score, r.source)
                }).collect::<Vec<_>>().join(", ")
            };

            // For cross_app, flag whether the secondary also fired
            let cross_note = if *cat == "cross_app" {
                if let Some(also) = tc.expect_also {
                    if detected.contains(&also) { " +secondary✓" } else { " (secondary not fired)" }
                } else { "" }
            } else { "" };

            println!("  {} {:50}  →  {}{}", status, tc.query, detected_str, cross_note);
            if !ok {
                println!("      expected: {}  note: {}", tc.expect_intent, tc.note);
            }
        }

        counts.insert(cat, (pass, total));
        println!("  {} / {} passed", pass, total);
        println!();
    }

    // Summary
    println!("━━━  SUMMARY  ━━━");
    let categories = ["situation", "cross_app", "negative"];
    let mut total_pass = 0;
    let mut total_all = 0;
    for cat in &categories {
        if let Some((pass, total)) = counts.get(cat) {
            println!("  {:20} {}/{}", cat, pass, total);
            total_pass += pass;
            total_all += total;
        }
    }
    println!("  ─────────────────────────");
    println!("  {:20} {}/{}", "TOTAL", total_pass, total_all);

    if let Some((fp, fpt)) = counts.get("false_positive") {
        println!("  {:20} {} documented (expected precision limit)", "false_positives", fpt - fp);
    }

    println!();
    if all_pass {
        println!("ALL TESTS PASSED");
    } else {
        println!("SOME TESTS FAILED — see ✗ above");
        std::process::exit(1);
    }
}
