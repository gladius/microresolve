//! Situation→Action Inference Experiment
//!
//! Tests whether a phrase-level situation index can:
//!   1. Detect the correct APP from a situation description (no action verb)
//!   2. Detect the correct INTENT within that app
//!   3. Improve with one learning pass from corrected failures
//!
//! Run: cargo run --bin situation_exp --features server
//! (needs --features server only for the tokenizer path; no server actually starts)

use std::collections::{HashMap, HashSet};
use asv_router::tokenizer::tokenize;

// ─── Core types ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct SituationPhrase {
    phrase: String,
    tokens: Vec<String>,   // pre-tokenized, stop words already removed by tokenize()
    weight: f32,            // 1.0 seeded, 0.4 learned
}

/// Sliding-window token overlap between a phrase and a query.
///
/// Window size = phrase_len + 2 (allows 2 extra "gap" words).
/// Takes the best overlap found across all windows.
///
/// Why: "payment bounced" should match "payment totally bounced" (gap=1) ✓
///      but NOT "payment error logged, idea bounced around" (gap=5) ✗
///
/// Order insensitive within the window: "bounced payment" still matches "payment bounced".
fn windowed_overlap(phrase_tokens: &[String], query_vec: &[String]) -> f32 {
    if phrase_tokens.is_empty() || query_vec.is_empty() { return 0.0; }

    let phrase_set: HashSet<&str> = phrase_tokens.iter().map(|s| s.as_str()).collect();
    let window_size = phrase_tokens.len() + 2;
    let query_len = query_vec.len();

    // Number of windows to slide. If query fits inside one window, just use the whole query.
    let num_windows = if query_len <= window_size { 1 } else { query_len - window_size + 1 };

    let mut best_overlap = 0.0f32;
    for start in 0..num_windows {
        let end = (start + window_size).min(query_len);
        let window_set: HashSet<&str> = query_vec[start..end].iter().map(|s| s.as_str()).collect();
        let matches = phrase_set.iter().filter(|t| window_set.contains(*t)).count();
        let overlap = matches as f32 / phrase_tokens.len() as f32;
        if overlap > best_overlap { best_overlap = overlap; }
    }

    best_overlap
}

/// Key = (app_id, intent_id)
struct SituationStore {
    phrases: HashMap<(String, String), Vec<SituationPhrase>>,
}

#[derive(Debug, Clone)]
struct ScoredIntent {
    app_id: String,
    intent_id: String,
    score: f32,
    best_match: String,   // which phrase matched best
}

impl SituationStore {
    fn new() -> Self {
        Self { phrases: HashMap::new() }
    }

    fn add(&mut self, app_id: &str, intent_id: &str, phrase: &str, weight: f32) {
        let tokens = tokenize(phrase);
        if tokens.is_empty() { return; }
        let entry = self.phrases
            .entry((app_id.to_string(), intent_id.to_string()))
            .or_default();
        // Avoid duplicates
        if !entry.iter().any(|p| p.phrase == phrase) {
            entry.push(SituationPhrase {
                phrase: phrase.to_string(),
                tokens,
                weight,
            });
        }
    }

    /// Score a query against all (app, intent) pairs.
    /// Returns list sorted by score descending.
    fn score_query(&self, query: &str) -> Vec<ScoredIntent> {
        let query_vec: Vec<String> = tokenize(query);
        let mut results: Vec<ScoredIntent> = Vec::new();

        for ((app_id, intent_id), phrases) in &self.phrases {
            let mut intent_score = 0.0f32;
            let mut best_match = String::new();
            let mut best_phrase_score = 0.0f32;

            for phrase in phrases {
                if phrase.tokens.is_empty() { continue; }

                // Sliding window overlap: phrase tokens must appear within a window of
                // phrase_len+2 consecutive query positions. This handles:
                //   "bounced payment" vs "payment bounced" → same window, full match ✓
                //   "payment got bounced" → window spans both tokens ✓
                //   "payment ... [10 words] ... bounced" → no window covers both ✗
                let overlap = windowed_overlap(&phrase.tokens, &query_vec);

                // Require at least 60% overlap
                if overlap < 0.6 { continue; }

                // Length bonus: sqrt rewards longer, more specific phrases
                let length_bonus = (phrase.tokens.len() as f32).sqrt();
                let phrase_score = overlap * phrase.weight * length_bonus;

                intent_score += phrase_score;
                if phrase_score > best_phrase_score {
                    best_phrase_score = phrase_score;
                    best_match = phrase.phrase.clone();
                }
            }

            if intent_score > 0.0 {
                results.push(ScoredIntent {
                    app_id: app_id.clone(),
                    intent_id: intent_id.clone(),
                    score: intent_score,
                    best_match,
                });
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    /// Top app detection: aggregate intent scores per app.
    fn detect_app(&self, query: &str) -> Vec<(String, f32)> {
        let scored = self.score_query(query);
        let mut app_scores: HashMap<String, f32> = HashMap::new();
        for s in &scored {
            *app_scores.entry(s.app_id.clone()).or_insert(0.0) += s.score;
        }
        let mut apps: Vec<(String, f32)> = app_scores.into_iter().collect();
        apps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        apps
    }

    /// Top (app, intent) globally.
    fn detect_intent(&self, query: &str) -> Option<ScoredIntent> {
        self.score_query(query).into_iter().next()
    }

    /// Top intent within a specific app.
    fn detect_intent_in_app(&self, query: &str, app_id: &str) -> Option<ScoredIntent> {
        self.score_query(query).into_iter()
            .find(|s| s.app_id == app_id)
    }

    /// Learn from a correction: extract bigrams and trigrams, add as situation phrases.
    fn learn(&mut self, query: &str, app_id: &str, intent_id: &str) {
        let tokens = tokenize(query);
        if tokens.is_empty() { return; }

        // Extract all bigrams and trigrams
        let mut ngrams: Vec<String> = Vec::new();
        for n in 2..=3usize {
            if tokens.len() >= n {
                for i in 0..=(tokens.len() - n) {
                    let ngram = tokens[i..i+n].join(" ");
                    ngrams.push(ngram);
                }
            }
        }

        // Add up to 4 n-grams (prefer longer), at learned weight 0.4
        let mut added = 0;
        // Sort: trigrams first (more specific), then bigrams
        ngrams.sort_by(|a, b| b.split_whitespace().count().cmp(&a.split_whitespace().count()));
        for ngram in ngrams {
            if added >= 4 { break; }
            self.add(app_id, intent_id, &ngram, 0.4);
            added += 1;
        }
    }

    fn phrase_count(&self) -> usize {
        self.phrases.values().map(|v| v.len()).sum()
    }
}

// ─── Seed data ────────────────────────────────────────────────────────────────
// Hand-written situation phrases per (app, intent).
// These are NOT paraphrases of the action — they describe situations that imply the action.
// Deliberately minimal: what a developer would write in 30 minutes.

fn seed_store() -> SituationStore {
    let mut s = SituationStore::new();

    // ── STRIPE ──
    s.add("stripe", "charge_card",         "payment bounced",                1.0);
    s.add("stripe", "charge_card",         "card declined",                  1.0);
    s.add("stripe", "charge_card",         "payment failed",                  1.0);
    s.add("stripe", "charge_card",         "charge rejected",                 1.0);
    s.add("stripe", "charge_card",         "transaction failed",              1.0);
    s.add("stripe", "charge_card",         "payment not going through",       1.0);
    s.add("stripe", "charge_card",         "getting 402 errors",              1.0);

    s.add("stripe", "refund_payment",      "customer wants money back",       1.0);
    s.add("stripe", "refund_payment",      "chargeback filed",                1.0);
    s.add("stripe", "refund_payment",      "dispute opened",                  1.0);
    s.add("stripe", "refund_payment",      "incorrect amount charged",        1.0);
    s.add("stripe", "refund_payment",      "wrong charge on account",         1.0);

    s.add("stripe", "cancel_subscription", "customer churned",                1.0);
    s.add("stripe", "cancel_subscription", "they are leaving",                1.0);
    s.add("stripe", "cancel_subscription", "not renewing",                    1.0);
    s.add("stripe", "cancel_subscription", "cancellation requested",          1.0);
    s.add("stripe", "cancel_subscription", "wants out of the plan",           1.0);
    s.add("stripe", "cancel_subscription", "ending subscription",             1.0);

    s.add("stripe", "create_invoice",      "work completed",                  1.0);
    s.add("stripe", "create_invoice",      "project delivered",               1.0);
    s.add("stripe", "create_invoice",      "billing cycle ended",             1.0);
    s.add("stripe", "create_invoice",      "invoice overdue",                 1.0);
    s.add("stripe", "create_invoice",      "end of month billing",            1.0);

    s.add("stripe", "retrieve_balance",    "need to check funds",             1.0);
    s.add("stripe", "retrieve_balance",    "balance inquiry",                 1.0);
    s.add("stripe", "retrieve_balance",    "how much in account",             1.0);

    // ── GITHUB ──
    s.add("github", "create_issue",        "something broken",                1.0);
    s.add("github", "create_issue",        "bug in production",               1.0);
    s.add("github", "create_issue",        "regression found",                1.0);
    s.add("github", "create_issue",        "build is failing",                1.0);
    s.add("github", "create_issue",        "tests are failing",               1.0);
    s.add("github", "create_issue",        "getting errors in prod",          1.0);
    s.add("github", "create_issue",        "memory leak detected",            1.0);
    s.add("github", "create_issue",        "prod is down",                    1.0);

    s.add("github", "merge_pr",            "reviewers approved",              1.0);
    s.add("github", "merge_pr",            "all thumbs up",                   1.0);
    s.add("github", "merge_pr",            "review passed",                   1.0);
    s.add("github", "merge_pr",            "green lights on PR",              1.0);
    s.add("github", "merge_pr",            "approved and ready",              1.0);
    s.add("github", "merge_pr",            "PR is approved",                  1.0);

    s.add("github", "create_release",      "going live today",                1.0);
    s.add("github", "create_release",      "shipping the release",            1.0);
    s.add("github", "create_release",      "launch day",                      1.0);
    s.add("github", "create_release",      "ready to go live",                1.0);
    s.add("github", "create_release",      "cutting the release",             1.0);
    s.add("github", "create_release",      "deployment day",                  1.0);

    s.add("github", "close_issue",         "bug is fixed",                    1.0);
    s.add("github", "close_issue",         "problem resolved",                1.0);
    s.add("github", "close_issue",         "no longer reproducing",           1.0);
    s.add("github", "close_issue",         "issue is resolved",               1.0);
    s.add("github", "close_issue",         "working now",                     1.0);

    s.add("github", "create_pr",           "changes ready for review",        1.0);
    s.add("github", "create_pr",           "feature complete",                1.0);
    s.add("github", "create_pr",           "diff is ready",                   1.0);
    s.add("github", "create_pr",           "ready for eyes",                  1.0);

    // ── SLACK ──
    s.add("slack", "send_message",         "team needs to know",              1.0);
    s.add("slack", "send_message",         "important update for team",       1.0);
    s.add("slack", "send_message",         "should let people know",          1.0);
    s.add("slack", "send_message",         "team announcement",               1.0);

    s.add("slack", "create_channel",       "need a dedicated space",          1.0);
    s.add("slack", "create_channel",       "nowhere to coordinate",           1.0);
    s.add("slack", "create_channel",       "need a room for this",            1.0);
    s.add("slack", "create_channel",       "no channel for this topic",       1.0);

    s.add("slack", "invite_user",          "someone needs to be in the loop", 1.0);
    s.add("slack", "invite_user",          "should be included",              1.0);
    s.add("slack", "invite_user",          "needs access to channel",         1.0);
    s.add("slack", "invite_user",          "should be part of this",          1.0);

    s.add("slack", "set_reminder",         "need to follow up",               1.0);
    s.add("slack", "set_reminder",         "do not want to forget",           1.0);
    s.add("slack", "set_reminder",         "need to remember",                1.0);

    s.add("slack", "create_poll",          "team vote needed",                1.0);
    s.add("slack", "create_poll",          "need to decide as a group",       1.0);
    s.add("slack", "create_poll",          "democratic decision",             1.0);
    s.add("slack", "create_poll",          "need consensus from team",        1.0);

    // ── SHOPIFY ──
    s.add("shopify", "ship_order",         "order ready to go out",           1.0);
    s.add("shopify", "ship_order",         "package ready to send",           1.0);
    s.add("shopify", "ship_order",         "ready to fulfill",                1.0);
    s.add("shopify", "ship_order",         "time to send it out",             1.0);

    s.add("shopify", "track_shipment",     "customer asking where package is",1.0);
    s.add("shopify", "track_shipment",     "delivery not arrived",            1.0);
    s.add("shopify", "track_shipment",     "shipment status unknown",         1.0);
    s.add("shopify", "track_shipment",     "customer worried about delivery", 1.0);

    s.add("shopify", "update_inventory",   "running low on stock",            1.0);
    s.add("shopify", "update_inventory",   "inventory depleted",              1.0);
    s.add("shopify", "update_inventory",   "out of stock",                    1.0);
    s.add("shopify", "update_inventory",   "stock count wrong",               1.0);
    s.add("shopify", "update_inventory",   "nearly sold out",                 1.0);

    s.add("shopify", "refund_order",       "item arrived damaged",            1.0);
    s.add("shopify", "refund_order",       "wrong product sent",              1.0);
    s.add("shopify", "refund_order",       "customer unhappy with order",     1.0);
    s.add("shopify", "refund_order",       "product defective",               1.0);

    s.add("shopify", "cancel_order",       "customer changed their mind",     1.0);
    s.add("shopify", "cancel_order",       "order was a mistake",             1.0);
    s.add("shopify", "cancel_order",       "before it shipped",               1.0);

    s.add("shopify", "process_return",     "customer returning item",         1.0);
    s.add("shopify", "process_return",     "return request received",         1.0);
    s.add("shopify", "process_return",     "item being sent back",            1.0);

    s.add("shopify", "generate_report",    "need sales numbers",              1.0);
    s.add("shopify", "generate_report",    "end of quarter review",           1.0);
    s.add("shopify", "generate_report",    "revenue inquiry",                 1.0);
    s.add("shopify", "generate_report",    "how did we do this month",        1.0);

    // ── CALENDAR ──
    s.add("calendar", "create_event",      "need to meet",                    1.0);
    s.add("calendar", "create_event",      "need a time slot",                1.0);
    s.add("calendar", "create_event",      "planning a call",                 1.0);
    s.add("calendar", "create_event",      "setting up a meeting",            1.0);

    s.add("calendar", "cancel_event",      "meeting not happening",           1.0);
    s.add("calendar", "cancel_event",      "call got cancelled",              1.0);
    s.add("calendar", "cancel_event",      "event is off",                    1.0);

    s.add("calendar", "reschedule_event",  "time conflict",                   1.0);
    s.add("calendar", "reschedule_event",  "double booked",                   1.0);
    s.add("calendar", "reschedule_event",  "cannot make it then",             1.0);
    s.add("calendar", "reschedule_event",  "need to move the meeting",        1.0);

    s.add("calendar", "check_availability","wondering if free",               1.0);
    s.add("calendar", "check_availability","checking the schedule",           1.0);
    s.add("calendar", "check_availability","seeing if there is time",         1.0);

    s.add("calendar", "set_out_of_office", "going on vacation",               1.0);
    s.add("calendar", "set_out_of_office", "away next week",                  1.0);
    s.add("calendar", "set_out_of_office", "unavailable those days",          1.0);
    s.add("calendar", "set_out_of_office", "out of the office",               1.0);

    s.add("calendar", "find_meeting_time", "need a slot everyone is free",    1.0);
    s.add("calendar", "find_meeting_time", "figuring out when to meet",       1.0);
    s.add("calendar", "find_meeting_time", "coordinating schedules",          1.0);

    s.add("calendar", "invite_attendee",   "someone should be on the call",   1.0);
    s.add("calendar", "invite_attendee",   "needs to join the meeting",       1.0);
    s.add("calendar", "invite_attendee",   "should be part of the call",      1.0);

    s
}

// ─── CJK seed phrases ─────────────────────────────────────────────────────────
// CJK bigram tokenizer creates "bridge" bigrams that break when characters are
// inserted between concepts: "付款失败" → ["付款失败","付款","款失","失败"] but
// "付款一直失败" → ["付款","款一","一直","直失","失败"] — "款失" is absent → 2/4 = 0.5 overlap.
//
// Fix: write seeds as SPACE-SEPARATED key 2-char compound pairs.
// "付款 失败" → ["付款","失败"] — 2 tokens. Any query with both → 2/2 = 1.0. ✓
// Cross-domain guard: "构建失败" has "失败" but NOT "付款" → 1/2 = 0.5 < 0.6. ✗

fn seed_cjk(s: &mut SituationStore) {
    // ── STRIPE (Chinese) — space-separated compound pairs ──
    s.add("stripe", "charge_card",         "付款 失败",     1.0); // payment + failed
    s.add("stripe", "charge_card",         "卡 拒绝",       1.0); // card + rejected
    s.add("stripe", "charge_card",         "收款 失败",     1.0); // collect payment + failed
    s.add("stripe", "charge_card",         "交易 拒",       1.0); // transaction + rejected
    s.add("stripe", "refund_payment",      "客户 退款",     1.0); // customer + refund
    s.add("stripe", "refund_payment",      "扣费 错误",     1.0); // charge + error
    s.add("stripe", "refund_payment",      "发起 争议",     1.0); // raised + dispute
    s.add("stripe", "cancel_subscription", "用户 流失",     1.0); // user + churned
    s.add("stripe", "cancel_subscription", "不 续费",       1.0); // not + renewing
    s.add("stripe", "cancel_subscription", "取消 订阅",     1.0); // cancel + subscription
    s.add("stripe", "create_invoice",      "项目 交付",     1.0); // project + delivered
    s.add("stripe", "create_invoice",      "月底 账期",     1.0); // month-end + billing
    s.add("stripe", "retrieve_balance",    "查 余额",       1.0); // check + balance

    // ── GITHUB (Chinese) ──
    s.add("github", "create_issue",        "生产 挂了",     1.0); // prod + down
    s.add("github", "create_issue",        "构建 失败",     1.0); // build + failed
    s.add("github", "create_issue",        "线上 问题",     1.0); // prod + issue
    s.add("github", "create_issue",        "内存 泄漏",     1.0); // memory + leak
    s.add("github", "create_issue",        "报错 线上",     1.0); // errors + prod
    s.add("github", "merge_pr",            "审查 通过",     1.0); // review + passed
    s.add("github", "merge_pr",            "PR 批准",       1.0); // PR + approved
    s.add("github", "create_release",      "发版 今天",     1.0); // release + today
    s.add("github", "create_release",      "上线 准备",     1.0); // go-live + ready
    s.add("github", "close_issue",         "问题 解决",     1.0); // issue + resolved
    s.add("github", "close_issue",         "bug 修好",      1.0); // bug + fixed

    // ── SLACK (Chinese) ──
    s.add("slack", "send_message",         "团队 知道",     1.0); // team + know
    s.add("slack", "send_message",         "通知 大家",     1.0); // notify + everyone
    s.add("slack", "create_channel",       "专属 频道",     1.0); // dedicated + channel
    s.add("slack", "create_channel",       "没有 地方 协调", 1.0); // nowhere + coordinate
    s.add("slack", "invite_user",          "需要 加入 频道", 1.0); // need + join + channel
    s.add("slack", "set_reminder",         "不想 忘记",     1.0); // don't want + forget
    s.add("slack", "create_poll",          "团队 投票",     1.0); // team + vote

    // ── SHOPIFY (Chinese) ──
    s.add("shopify", "update_inventory",   "库存 没了",     1.0); // inventory + gone
    s.add("shopify", "update_inventory",   "缺货",          1.0); // out of stock (single compound)
    s.add("shopify", "update_inventory",   "库存 归零",     1.0); // inventory + zero
    s.add("shopify", "track_shipment",     "客户 包裹",     1.0); // customer + package
    s.add("shopify", "track_shipment",     "物流 动静",     1.0); // shipment + movement
    s.add("shopify", "refund_order",       "收到 损坏",     1.0); // received + damaged
    s.add("shopify", "refund_order",       "发错 货",       1.0); // sent + wrong item
    s.add("shopify", "cancel_order",       "客户 取消",     1.0); // customer + cancel
    s.add("shopify", "generate_report",    "季末 数据",     1.0); // quarter-end + data
    s.add("shopify", "generate_report",    "销售 怎么样",   1.0); // sales + how were

    // ── CALENDAR (Chinese) ──
    s.add("calendar", "create_event",      "安排 会议",     1.0); // arrange + meeting
    s.add("calendar", "cancel_event",      "会议 取消",     1.0); // meeting + cancelled
    s.add("calendar", "reschedule_event",  "时间 冲突",     1.0); // time + conflict
    s.add("calendar", "reschedule_event",  "日程 重叠",     1.0); // schedule + overlap
    s.add("calendar", "reschedule_event",  "改 时间",       1.0); // change + time
    s.add("calendar", "set_out_of_office", "下周 不在",     1.0); // next week + away
    s.add("calendar", "set_out_of_office", "出去 玩",       1.0); // go + trip/vacation
    s.add("calendar", "find_meeting_time", "大家 有空",     1.0); // everyone + free
    s.add("calendar", "find_meeting_time", "找 时间 开会",  1.0); // find + time + meet
    s.add("calendar", "invite_attendee",   "需要 加入 会议", 1.0); // needs + join + meeting

    // ── STRIPE (Japanese) — same pattern: space-separated key compounds ──
    s.add("stripe", "charge_card",         "支払 失敗",     1.0); // payment + failed
    s.add("stripe", "charge_card",         "カード 拒否",   1.0); // card + declined
    s.add("stripe", "refund_payment",      "返金 お願い",   1.0); // refund + request
    s.add("stripe", "cancel_subscription", "解約 したい",   1.0); // cancel + want to
    s.add("stripe", "cancel_subscription", "更新 しない",   1.0); // renew + not

    // ── GITHUB (Japanese) ──
    s.add("github", "create_issue",        "本番 落ちて",   1.0); // prod + down
    s.add("github", "create_issue",        "バグ 見つかり", 1.0); // bug + found
    s.add("github", "merge_pr",            "レビュー 承認", 1.0); // review + approved
    s.add("github", "create_release",      "リリース 日",   1.0); // release + day

    // ── SLACK (Japanese) ──
    s.add("slack", "send_message",         "チーム 知らせ", 1.0); // team + inform
    s.add("slack", "create_channel",       "専用 チャンネル", 1.0); // dedicated + channel

    // ── SHOPIFY (Japanese) ──
    s.add("shopify", "update_inventory",   "在庫 切れ",     1.0); // inventory + out
    s.add("shopify", "track_shipment",     "荷物 どこ",     1.0); // package + where
    s.add("shopify", "refund_order",       "商品 壊れ",     1.0); // item + broken

    // ── CALENDAR (Japanese) ──
    s.add("calendar", "reschedule_event",  "ダブル ブッキング", 1.0); // double + booking
    s.add("calendar", "set_out_of_office", "来週 休み",     1.0); // next week + off
    s.add("calendar", "find_meeting_time", "空き時間 探し", 1.0); // free time + find
}

fn cjk_test_cases() -> Vec<TestCase> {
    vec![
        // Chinese — pure situation queries
        tc("付款一直失败，搞不定了",              "stripe",   "charge_card",         "situation"), // payment keeps failing
        tc("客户的卡被拒了三次",                  "stripe",   "charge_card",         "situation"), // card declined 3 times
        tc("这个客户不续费了，要走了",            "stripe",   "cancel_subscription", "situation"), // customer not renewing, leaving
        tc("客户说金额扣错了要投诉",              "stripe",   "refund_payment",      "situation"), // customer says wrong amount, wants to complain
        tc("线上出大问题了，好多报错",            "github",   "create_issue",        "situation"), // big prod issue, lots of errors
        tc("构建挂了，没法发版",                  "github",   "create_issue",        "situation"), // build failed, can't release
        tc("PR已经有两个approval了",              "github",   "merge_pr",            "situation"), // PR has two approvals
        tc("今天是发版日，准备好了",              "github",   "create_release",      "situation"), // today is release day, ready
        tc("仓库快空了，热门款只剩三件",          "shopify",  "update_inventory",    "situation"), // warehouse almost empty, popular item has 3 left
        tc("客户说收到的东西坏掉了",              "shopify",  "refund_order",        "situation"), // customer says received broken item
        tc("这个季度结束了，看看销量如何",        "shopify",  "generate_report",     "situation"), // quarter ended, check sales
        tc("周二下午我有两个会重叠了",            "calendar", "reschedule_event",    "situation"), // tuesday afternoon two meetings overlap
        tc("下个月我要出去玩一周",                "calendar", "set_out_of_office",   "situation"), // going on trip next month for a week

        // Japanese — pure situation queries
        tc("支払いが何度も失敗しています",        "stripe",   "charge_card",         "situation"), // payment failing repeatedly
        tc("お客様がカードを拒否されたと言っています", "stripe", "charge_card",       "situation"), // customer says card was declined
        tc("解約したいというリクエストが来ました", "stripe",  "cancel_subscription", "situation"), // got a cancellation request
        tc("本番環境が落ちているとアラートが来た", "github",  "create_issue",        "situation"), // alert came that prod is down
        tc("在庫が全部なくなってしまいました",    "shopify",  "update_inventory",    "situation"), // all inventory is gone
        tc("来週の月曜日から木曜日まで休みます",  "calendar", "set_out_of_office",   "situation"), // off Monday to Thursday next week

        // Mixed CJK (situation + action verb)
        tc("付款失败了，重新收一下",              "stripe",   "charge_card",         "mixed"), // payment failed, charge again
        tc("线上出问题了，提个issue",             "github",   "create_issue",        "mixed"), // prod issue, file an issue
        tc("缺货了，更新一下库存",                "shopify",  "update_inventory",    "mixed"), // out of stock, update inventory

        // Negative CJK
        tc("今天天气不错",                        "none",     "none",                "negative"), // weather is nice today
        tc("我不确定这个方案怎么样",              "none",     "none",                "negative"), // not sure about this plan
    ]
}

// ─── Test cases ───────────────────────────────────────────────────────────────

#[derive(Clone)]
struct TestCase {
    query: String,
    expected_app: String,
    expected_intent: String,
    category: &'static str,   // "situation", "cross_app", "mixed", "negative"
}

fn tc(query: &str, app: &str, intent: &str, cat: &'static str) -> TestCase {
    TestCase {
        query: query.to_string(),
        expected_app: app.to_string(),
        expected_intent: intent.to_string(),
        category: cat,
    }
}

fn test_cases() -> Vec<TestCase> {
    vec![
        // ── CATEGORY A: Pure situation queries (no action verb) ──
        tc("the payment bounced on the enterprise account",          "stripe",   "charge_card",         "situation"),
        tc("card got declined three times in a row",                 "stripe",   "charge_card",         "situation"),
        tc("customer churned last night, three months in",           "stripe",   "cancel_subscription", "situation"),
        tc("they are not renewing when the plan expires",            "stripe",   "cancel_subscription", "situation"),
        tc("dispute came in for the enterprise payment",             "stripe",   "refund_payment",      "situation"),
        tc("wrong amount on the client statement this month",        "stripe",   "refund_payment",      "situation"),
        tc("project is done and delivered to the client",            "stripe",   "create_invoice",      "situation"),
        tc("there is a memory leak in the authentication service",   "github",   "create_issue",        "situation"),
        tc("prod is down for the second time today",                 "github",   "create_issue",        "situation"),
        tc("build is red on main branch",                            "github",   "create_issue",        "situation"),
        tc("all reviewers gave thumbs up on the diff",               "github",   "merge_pr",            "situation"),
        tc("PR has been sitting with approvals for two days",        "github",   "merge_pr",            "situation"),
        tc("we are going live with v3 of the API this afternoon",    "github",   "create_release",      "situation"),
        tc("launch day for the new payments feature",                "github",   "create_release",      "situation"),
        tc("the auth bug is no longer reproducing in staging",       "github",   "close_issue",         "situation"),
        tc("engineering team has nowhere to coordinate the incident","slack",    "create_channel",      "situation"),
        tc("sarah needs to be in the loop on this decision",         "slack",    "invite_user",         "situation"),
        tc("team vote needed on the new logo direction",             "slack",    "create_poll",         "situation"),
        tc("blue hoodie SKU is down to zero units",                  "shopify",  "update_inventory",    "situation"),
        tc("customer asking where their package has been",           "shopify",  "track_shipment",      "situation"),
        tc("item came back damaged from shipping",                   "shopify",  "refund_order",        "situation"),
        tc("order was a mistake, nothing has shipped yet",           "shopify",  "cancel_order",        "situation"),
        tc("end of quarter, need to see how store performed",        "shopify",  "generate_report",     "situation"),
        tc("tuesday afternoon is double booked",                     "calendar", "reschedule_event",    "situation"),
        tc("need a slot when the whole product team is free",        "calendar", "find_meeting_time",   "situation"),
        tc("away the week of the fifteenth",                         "calendar", "set_out_of_office",   "situation"),

        // ── CATEGORY B: Cross-app ambiguous (app detection test) ──
        tc("something is broken and the team needs to know",         "github",   "create_issue",        "cross_app"),
        tc("the build went red right before the demo",               "github",   "create_issue",        "cross_app"),
        tc("customer is not happy and wants their money back",        "shopify",  "refund_order",        "cross_app"),
        tc("end of sprint, time to look at the numbers",             "shopify",  "generate_report",     "cross_app"),
        tc("three people booked the same room on thursday",          "calendar", "reschedule_event",    "cross_app"),

        // ── CATEGORY C: Mixed queries (situation + action verb) ──
        tc("payment failed, process it again",                       "stripe",   "charge_card",         "mixed"),
        tc("the bug is fixed, we can close it",                      "github",   "close_issue",         "mixed"),
        tc("inventory hit zero, restock it",                         "shopify",  "update_inventory",    "mixed"),
        tc("double booked tuesday, move the meeting",                "calendar", "reschedule_event",    "mixed"),
        tc("prod is down, file a ticket immediately",                "github",   "create_issue",        "mixed"),

        // ── CATEGORY D: Negative (nothing should fire confidently) ──
        tc("the weather looks good for the launch",                  "none",     "none",                "negative"),
        tc("sounds good to me",                                      "none",     "none",                "negative"),
        tc("I am not sure about this approach",                      "none",     "none",                "negative"),
        tc("ok yes",                                                 "none",     "none",                "negative"),
        tc("banana pancakes",                                        "none",     "none",                "negative"),
    ]
}

// ─── Evaluation ───────────────────────────────────────────────────────────────

const SCORE_THRESHOLD: f32 = 0.3;

struct EvalResult {
    app_correct: bool,
    intent_correct: bool,
    top_app: String,
    top_intent: String,
    top_score: f32,
    matched_phrase: String,
    correctly_silent: bool,  // negative case, nothing fired
}

fn evaluate_one(store: &SituationStore, tc: &TestCase) -> EvalResult {
    let scored = store.score_query(&tc.query);
    let top = scored.first();

    if tc.category == "negative" {
        let silent = top.map(|s| s.score < SCORE_THRESHOLD).unwrap_or(true);
        let top_app = top.map(|s| s.app_id.clone()).unwrap_or_default();
        let top_intent = top.map(|s| s.intent_id.clone()).unwrap_or_default();
        let top_score = top.map(|s| s.score).unwrap_or(0.0);
        return EvalResult {
            app_correct: silent,
            intent_correct: silent,
            top_app,
            top_intent,
            top_score,
            matched_phrase: top.map(|s| s.best_match.clone()).unwrap_or_default(),
            correctly_silent: silent,
        };
    }

    let apps = store.detect_app(&tc.query);
    let top_app_name = apps.first().map(|(a, _)| a.clone()).unwrap_or_default();
    let top_app_score = apps.first().map(|(_, s)| *s).unwrap_or(0.0);

    let top_intent_result = store.detect_intent(&tc.query);
    let top_intent_name = top_intent_result.as_ref().map(|s| s.intent_id.clone()).unwrap_or_default();
    let top_intent_score = top_intent_result.as_ref().map(|s| s.score).unwrap_or(0.0);
    let matched_phrase = top_intent_result.as_ref().map(|s| s.best_match.clone()).unwrap_or_default();

    let app_correct = top_app_score >= SCORE_THRESHOLD && top_app_name == tc.expected_app;
    let intent_correct = top_intent_score >= SCORE_THRESHOLD && top_intent_name == tc.expected_intent;

    EvalResult {
        app_correct,
        intent_correct,
        top_app: top_app_name,
        top_intent: top_intent_name,
        top_score: top_intent_score,
        matched_phrase,
        correctly_silent: false,
    }
}

fn run_eval(store: &SituationStore, cases: &[TestCase], label: &str) -> Vec<EvalResult> {
    println!("\n{}", "=".repeat(70));
    println!("  {}", label);
    println!("{}", "=".repeat(70));

    let mut results = Vec::new();
    let mut cat_stats: HashMap<&str, (usize, usize, usize)> = HashMap::new(); // (total, app_ok, intent_ok)

    for tc in cases {
        let r = evaluate_one(store, tc);
        let (total, app_ok, intent_ok) = cat_stats.entry(tc.category).or_insert((0, 0, 0));
        *total += 1;
        if r.app_correct    { *app_ok += 1; }
        if r.intent_correct { *intent_ok += 1; }

        let app_marker    = if r.app_correct    { "✓" } else { "✗" };
        let intent_marker = if r.intent_correct { "✓" } else { "✗" };

        if tc.category == "negative" {
            let status = if r.correctly_silent { "✓ SILENT" } else { "✗ FALSE+" };
            println!("  {} [{}] {:.42}", status, tc.category, tc.query);
            if !r.correctly_silent {
                println!("       fired: {}.{} score={:.2} via '{}'",
                    r.top_app, r.top_intent, r.top_score, r.matched_phrase);
            }
        } else {
            let ok = if r.app_correct && r.intent_correct { "PASS" }
                     else if r.app_correct || r.intent_correct { "PART" }
                     else { "FAIL" };
            println!("  [{}] {} app{} intent{}", ok, &tc.query[..tc.query.len().min(55)],
                app_marker, intent_marker);
            if !r.app_correct || !r.intent_correct {
                println!("       expected: {}.{}",
                    tc.expected_app, tc.expected_intent);
                println!("       got:      {}.{} score={:.2} via '{}'",
                    r.top_app, r.top_intent, r.top_score, r.matched_phrase);
            }
        }

        results.push(r);
    }

    // Summary by category
    println!("\n  --- {} ---", label);
    let categories = ["situation", "cross_app", "mixed", "negative"];
    let mut total_app = 0usize;
    let mut total_intent = 0usize;
    let mut total_all = 0usize;

    for cat in &categories {
        if let Some((n, app_ok, intent_ok)) = cat_stats.get(cat) {
            if *n == 0 { continue; }
            total_all    += n;
            total_app    += app_ok;
            total_intent += intent_ok;
            println!("  {:10} {:2} cases | app {:2}/{:2} ({:3}%) | intent {:2}/{:2} ({:3}%)",
                cat, n,
                app_ok, n, 100 * app_ok / n,
                intent_ok, n, 100 * intent_ok / n);
        }
    }
    println!("  {:10} {:2} cases | app {:2}/{:2} ({:3}%) | intent {:2}/{:2} ({:3}%)",
        "TOTAL", total_all,
        total_app, total_all, if total_all > 0 { 100 * total_app / total_all } else { 0 },
        total_intent, total_all, if total_all > 0 { 100 * total_intent / total_all } else { 0 });

    results
}

// ─── Learning pass ────────────────────────────────────────────────────────────

fn learn_from_failures(store: &mut SituationStore, cases: &[TestCase], results: &[EvalResult]) {
    println!("\n{}", "=".repeat(70));
    println!("  LEARNING — adding n-grams from failed cases");
    println!("{}", "=".repeat(70));

    let mut learned = 0;
    for (tc, r) in cases.iter().zip(results.iter()) {
        if tc.category == "negative" { continue; }
        if !r.intent_correct {
            let before = store.phrases
                .get(&(tc.expected_app.clone(), tc.expected_intent.clone()))
                .map(|v| v.len()).unwrap_or(0);
            store.learn(&tc.query, &tc.expected_app, &tc.expected_intent);
            let after = store.phrases
                .get(&(tc.expected_app.clone(), tc.expected_intent.clone()))
                .map(|v| v.len()).unwrap_or(0);
            let added = after - before;
            if added > 0 {
                println!("  + {}.{}: +{} phrases from '{:.50}'",
                    tc.expected_app, tc.expected_intent, added, tc.query);
                learned += added;
            }
        }
    }
    println!("  Total new phrases: {}  |  Store size: {}", learned, store.phrase_count());
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("ASV Situation→Action Inference Experiment");
    println!("Testing: app detection, intent detection, learning loop");
    println!("Threshold: {:.2}  |  Phrase overlap: ≥60%", SCORE_THRESHOLD);

    let mut store = seed_store();
    println!("\nSeed phrases loaded: {}", store.phrase_count());

    let cases = test_cases();
    println!("Test cases: {} total ({} situation, {} cross_app, {} mixed, {} negative)",
        cases.len(),
        cases.iter().filter(|t| t.category == "situation").count(),
        cases.iter().filter(|t| t.category == "cross_app").count(),
        cases.iter().filter(|t| t.category == "mixed").count(),
        cases.iter().filter(|t| t.category == "negative").count(),
    );

    // Round 1 — seeds only
    let r1 = run_eval(&store, &cases, "ROUND 1 — Seed phrases only (baseline)");

    // Learn from failures
    learn_from_failures(&mut store, &cases, &r1);

    // Round 2 — after learning
    let r2 = run_eval(&store, &cases, "ROUND 2 — After learning pass");

    // Diff
    println!("\n{}", "=".repeat(70));
    println!("  IMPROVEMENT");
    println!("{}", "=".repeat(70));

    let app_r1    = r1.iter().filter(|r| r.app_correct).count();
    let intent_r1 = r1.iter().filter(|r| r.intent_correct).count();
    let app_r2    = r2.iter().filter(|r| r.app_correct).count();
    let intent_r2 = r2.iter().filter(|r| r.intent_correct).count();
    let n = cases.len();

    println!("  App detection:    {}/{} → {}/{} (+{})",
        app_r1, n, app_r2, n, app_r2 as i32 - app_r1 as i32);
    println!("  Intent detection: {}/{} → {}/{} (+{})",
        intent_r1, n, intent_r2, n, intent_r2 as i32 - intent_r1 as i32);

    // Specific improvements
    let mut improved = Vec::new();
    let mut still_failing = Vec::new();
    for ((tc, a), b) in cases.iter().zip(r1.iter()).zip(r2.iter()) {
        if tc.category == "negative" { continue; }
        if !a.intent_correct && b.intent_correct {
            improved.push(format!("{}.{}", tc.expected_app, tc.expected_intent));
        }
        if !b.intent_correct {
            still_failing.push(format!("{}.{} (got: {}.{} score={:.2})",
                tc.expected_app, tc.expected_intent, b.top_app, b.top_intent, b.top_score));
        }
    }

    if !improved.is_empty() {
        println!("\n  Newly passing ({}):", improved.len());
        for s in &improved { println!("    + {}", s); }
    }
    if !still_failing.is_empty() {
        println!("\n  Still failing ({}):", still_failing.len());
        for s in &still_failing { println!("    - {}", s); }
    }

    println!("\n  See SITUATION_EXPERIMENT.md to record these results.");

    // ── CJK EXPERIMENT ────────────────────────────────────────────────────────
    println!("\n\n{}", "#".repeat(70));
    println!("  CJK EXPERIMENT — Chinese + Japanese situation phrases");
    println!("{}", "#".repeat(70));

    let mut cjk_store = seed_store();
    seed_cjk(&mut cjk_store);
    println!("\nSeed phrases loaded (EN+CJK): {}", cjk_store.phrase_count());

    let cjk_cases = cjk_test_cases();
    println!("CJK test cases: {} total ({} situation, {} mixed, {} negative)",
        cjk_cases.len(),
        cjk_cases.iter().filter(|t| t.category == "situation").count(),
        cjk_cases.iter().filter(|t| t.category == "mixed").count(),
        cjk_cases.iter().filter(|t| t.category == "negative").count(),
    );

    let cjk_r1 = run_eval(&cjk_store, &cjk_cases, "CJK ROUND 1 — Seed phrases only");

    learn_from_failures(&mut cjk_store, &cjk_cases, &cjk_r1);

    run_eval(&cjk_store, &cjk_cases, "CJK ROUND 2 — After learning pass");
}
