#!/usr/bin/env python3
"""
Unified index integration test — 10 apps, 20 intents each, multilingual.

All apps are loaded into a SINGLE Router as namespaced intent IDs:
  stripe:charge_card, github:create_issue, slack:send_message,
  shopify:create_product, calendar:create_event,
  jira:create_ticket, zendesk:create_ticket, hubspot:create_contact,
  twilio:send_sms, notion:create_page

One route_multi call covers all 200 intents. Results are scoped by checking the
namespace prefix on each returned intent ID.

Findings:
  - Cold start: ~40-50% namespace accuracy (100 shared generic terms compete)
  - Warm start (after learning): ~100% (learning is the engine)
  - Multilingual (zh, ta, ar, ja): routes to correct namespace without interference
  - Situation patterns: 80%+ fingerprint accuracy for brand-name queries
  - CJK namespaces: self-discriminating vocabulary, no fingerprints needed

Usage:
    # Start the server first: ./target/release/server
    python3 tests/unified_app_test.py [--base-url http://localhost:3001]
    python3 tests/unified_app_test.py --skip-setup   # reuse existing intents
    python3 tests/unified_app_test.py --only cross_app
"""

import json
import subprocess
import sys
import argparse

BASE_URL = "http://localhost:3001"
UNIFIED_APP_ID = "unified"

# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def curl(method: str, path: str, body=None, app_id: str = None) -> tuple[int, any]:
    cmd = ["curl", "-s", "-w", "\n%{http_code}", "-X", method, f"{BASE_URL}{path}",
           "-H", "Content-Type: application/json"]
    if app_id:
        cmd += ["-H", f"X-Namespace-ID: {app_id}"]
    if body is not None:
        cmd += ["-d", json.dumps(body)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = result.stdout.rsplit("\n", 1)
    body_str = lines[0].strip() if len(lines) > 1 else result.stdout.strip()
    status = int(lines[-1].strip()) if len(lines) > 1 and lines[-1].strip().isdigit() else 0
    try:
        return status, json.loads(body_str) if body_str else None
    except json.JSONDecodeError:
        return status, {"_raw": body_str}

def curl_json(method, path, body=None, app_id=None):
    _, d = curl(method, path, body, app_id)
    return d or {}

# ─── 10 Namespaced apps — 20 intents each ─────────────────────────────────────
# Seeds are minimal and realistic — what a developer writes on day one.
# They intentionally DON'T match test queries (which use natural language).

NAMESPACED_INTENTS = {

"stripe": [
    ("charge_card",         ["charge my card", "process payment", "bill the customer", "run a charge"]),
    ("refund_payment",      ["issue a refund", "refund this payment", "give money back", "reverse the charge"]),
    ("create_subscription", ["create a subscription", "set up recurring billing", "start a monthly plan"]),
    ("cancel_subscription", ["cancel the subscription", "stop recurring billing", "end the plan"]),
    ("list_customers",      ["list customers", "show all customers", "get customer list"]),
    ("create_customer",     ["create a customer", "add new customer", "register customer"]),
    ("update_customer",     ["update customer", "edit customer info", "change customer details"]),
    ("delete_customer",     ["delete customer", "remove customer account"]),
    ("create_invoice",      ["create an invoice", "generate invoice", "send invoice"]),
    ("pay_invoice",         ["pay this invoice", "mark invoice paid", "settle invoice"]),
    ("list_invoices",       ["list invoices", "show all invoices", "get invoice history"]),
    ("create_coupon",       ["create a coupon", "make a discount code", "add promo code"]),
    ("apply_discount",      ["apply a discount", "use promo code", "apply coupon"]),
    ("update_price",        ["update the price", "change pricing", "set new price"]),
    ("retrieve_balance",    ["check balance", "get account balance", "show available funds"]),
    ("transfer_funds",      ["transfer funds", "move money", "wire payment"]),
    ("create_payout",       ["create payout", "withdraw to bank", "initiate payout"]),
    ("create_payment_link", ["create payment link", "generate checkout link"]),
    ("list_transactions",   ["list transactions", "show payment history", "recent charges"]),
    ("dispute_charge",      ["dispute a charge", "chargeback request", "contest payment"]),
],

"github": [
    ("create_issue",     ["create an issue", "file a bug report", "open new issue", "report a bug"]),
    ("close_issue",      ["close this issue", "resolve the issue", "mark issue done"]),
    ("create_pr",        ["open a pull request", "create PR", "submit for review"]),
    ("merge_pr",         ["merge the PR", "merge pull request", "land this PR"]),
    ("close_pr",         ["close the pull request", "reject this PR", "close PR without merging"]),
    ("list_repos",       ["list repositories", "show my repos", "what repos do I have"]),
    ("create_repo",      ["create a repository", "new repo", "initialize repository"]),
    ("delete_repo",      ["delete this repository", "remove repo"]),
    ("create_branch",    ["create a branch", "make a new branch", "branch off"]),
    ("delete_branch",    ["delete the branch", "remove branch", "clean up branch"]),
    ("list_branches",    ["list branches", "show all branches"]),
    ("create_release",   ["create a release", "publish new version", "tag a release"]),
    ("list_commits",     ["list commits", "show commit history", "view git log"]),
    ("fork_repo",        ["fork this repo", "fork the repository"]),
    ("add_collaborator", ["add a collaborator", "invite contributor", "give access to repo"]),
    ("create_webhook",   ["create a webhook", "set up webhook", "add webhook"]),
    ("list_issues",      ["list issues", "show open issues", "what issues are there"]),
    ("search_code",      ["search code", "find in codebase", "grep across repo"]),
    ("create_gist",      ["create a gist", "share code snippet", "save as gist"]),
    ("review_pr",        ["review pull request", "add review comment", "approve PR changes"]),
],

"slack": [
    ("send_message",     ["send a message", "post to channel", "message the team"]),
    ("create_channel",   ["create a channel", "new slack channel", "set up channel"]),
    ("archive_channel",  ["archive this channel", "close the channel"]),
    ("invite_user",      ["invite someone to channel", "add user to channel", "add member"]),
    ("kick_user",        ["remove user from channel", "kick from channel"]),
    ("set_status",       ["set my status", "update status", "change slack status"]),
    ("set_reminder",     ["set a reminder", "remind me", "create reminder"]),
    ("search_messages",  ["search messages", "find a message", "look up conversation"]),
    ("list_channels",    ["list channels", "show all channels"]),
    ("pin_message",      ["pin this message", "pin to channel"]),
    ("create_poll",      ["create a poll", "start a vote", "poll the team"]),
    ("schedule_message", ["schedule a message", "send message later", "post later"]),
    ("upload_file",      ["upload a file", "share document", "attach file"]),
    ("list_members",     ["list members", "show channel members", "who is in channel"]),
    ("set_topic",        ["set channel topic", "update topic", "change channel description"]),
    ("react_to_message", ["react to message", "add emoji reaction", "react with emoji"]),
    ("get_user_info",    ["get user info", "look up user", "who is this user"]),
    ("create_workflow",  ["create a workflow", "automate in slack", "set up workflow"]),
    ("list_dms",         ["show direct messages", "list my DMs", "view DM conversations"]),
    ("mute_channel",     ["mute this channel", "silence notifications", "mute notifications"]),
],

"shopify": [
    ("create_product",        ["create a product", "add product to store", "new product listing"]),
    ("update_product",        ["update product", "edit product listing", "change product details"]),
    ("delete_product",        ["delete product", "remove from store", "take down listing"]),
    ("list_products",         ["list products", "show all products", "view product catalog"]),
    ("create_order",          ["create an order", "place order", "manually add order"]),
    ("cancel_order",          ["cancel order", "void the order", "cancel purchase"]),
    ("refund_order",          ["refund this order", "process order refund", "give refund"]),
    ("list_orders",           ["list orders", "show recent orders", "view order history"]),
    ("update_inventory",      ["update inventory", "change stock level", "adjust quantity"]),
    ("create_discount",       ["create a discount", "add discount", "set up sale"]),
    ("apply_discount_code",   ["apply discount code", "use coupon", "apply promo"]),
    ("list_customers",        ["list shopify customers", "show store customers"]),
    ("create_customer",       ["add store customer", "create shopify customer"]),
    ("update_customer",       ["update store customer", "edit buyer info"]),
    ("ship_order",            ["ship this order", "mark as shipped", "fulfill order"]),
    ("track_shipment",        ["track shipment", "where is my order", "track delivery"]),
    ("create_collection",     ["create a collection", "group products", "new category"]),
    ("update_store_settings", ["update store settings", "change store config"]),
    ("generate_report",       ["generate sales report", "store analytics", "revenue report"]),
    ("process_return",        ["process a return", "handle return", "accept returned item"]),
],

"calendar": [
    ("create_event",           ["create an event", "add to calendar", "schedule a meeting"]),
    ("cancel_event",           ["cancel the event", "delete meeting", "remove from calendar"]),
    ("reschedule_event",       ["reschedule this meeting", "move the event", "change event time"]),
    ("list_events",            ["list events", "show my schedule", "what do I have today"]),
    ("invite_attendee",        ["invite someone to meeting", "add to event", "send calendar invite"]),
    ("set_reminder",           ["set event reminder", "remind me before meeting"]),
    ("check_availability",     ["check availability", "am I free", "check if I have time"]),
    ("share_calendar",         ["share my calendar", "give access to calendar"]),
    ("create_recurring_event", ["create recurring event", "set up weekly meeting", "repeat this event"]),
    ("accept_invite",          ["accept the invite", "confirm meeting", "say yes to event"]),
    ("decline_invite",         ["decline the invite", "reject meeting", "say no to event"]),
    ("find_meeting_time",      ["find a meeting time", "when can we meet", "find common availability"]),
    ("set_working_hours",      ["set working hours", "update my work schedule"]),
    ("book_room",              ["book a room", "reserve meeting room", "find a room"]),
    ("add_video_link",         ["add video link", "attach zoom link", "add meet link"]),
    ("export_calendar",        ["export calendar", "download calendar", "get ical"]),
    ("view_event_details",     ["show event details", "what is this meeting about"]),
    ("update_event",           ["update event", "edit meeting", "change event details"]),
    ("set_out_of_office",      ["set out of office", "mark unavailable", "block vacation"]),
    ("sync_calendar",          ["sync calendar", "refresh calendar", "sync my schedule"]),
],

"jira": [
    ("create_ticket",     ["create a ticket", "file an issue", "log a task", "open a story"]),
    ("update_ticket",     ["update the ticket", "edit issue details", "change ticket fields"]),
    ("close_ticket",      ["close the ticket", "resolve this issue", "mark ticket done"]),
    ("assign_ticket",     ["assign this ticket", "reassign issue", "change ticket owner"]),
    ("add_comment",       ["add a comment", "leave a note on ticket", "comment on issue"]),
    ("list_tickets",      ["list open tickets", "show all issues", "what's in the backlog"]),
    ("create_sprint",     ["create a sprint", "start new sprint", "plan the sprint"]),
    ("close_sprint",      ["close the sprint", "end current sprint", "finish sprint"]),
    ("move_to_sprint",    ["move ticket to sprint", "add to current sprint", "put in sprint"]),
    ("create_epic",       ["create an epic", "new epic", "plan a large feature"]),
    ("link_ticket",       ["link this ticket", "relate tickets", "add dependency"]),
    ("set_priority",      ["set priority", "mark as urgent", "change ticket priority"]),
    ("log_time",          ["log time", "track hours", "add work log"]),
    ("create_board",      ["create a board", "new project board", "set up kanban"]),
    ("move_ticket",       ["move ticket to column", "drag to in progress", "transition ticket"]),
    ("add_label",         ["add a label", "tag the ticket", "add tag to issue"]),
    ("search_tickets",    ["search tickets", "find issues", "query the backlog"]),
    ("generate_report",   ["generate sprint report", "velocity report", "burndown chart"]),
    ("watch_ticket",      ["watch this ticket", "subscribe to updates", "follow the issue"]),
    ("duplicate_ticket",  ["duplicate this ticket", "clone the issue", "copy ticket"]),
],

"zendesk": [
    ("create_ticket",      ["create a support ticket", "open a case", "submit support request"]),
    ("update_ticket",      ["update the ticket", "add to the case", "modify support ticket"]),
    ("close_ticket",       ["close the ticket", "resolve support case", "mark ticket solved"]),
    ("assign_ticket",      ["assign to agent", "route to team", "reassign support ticket"]),
    ("add_internal_note",  ["add internal note", "leave a private note", "internal comment"]),
    ("reply_to_customer",  ["reply to customer", "send response", "answer the ticket"]),
    ("escalate_ticket",    ["escalate this ticket", "bump to tier 2", "escalate to manager"]),
    ("merge_tickets",      ["merge these tickets", "combine duplicate tickets"]),
    ("list_tickets",       ["list open tickets", "show support queue", "pending cases"]),
    ("search_tickets",     ["search tickets", "find support case", "look up customer issue"]),
    ("create_macro",       ["create a macro", "automate response", "add canned response"]),
    ("apply_macro",        ["apply a macro", "use saved response", "run macro on ticket"]),
    ("set_sla",            ["set SLA", "update response deadline", "change priority level"]),
    ("add_tag",            ["add a tag", "label this ticket", "tag the support case"]),
    ("create_view",        ["create a view", "filter ticket queue", "custom ticket list"]),
    ("trigger_automation", ["trigger automation", "run workflow rule", "fire ticket trigger"]),
    ("satisfaction_rating",["check satisfaction rating", "view CSAT score", "customer feedback"]),
    ("create_help_article",["create help article", "write knowledge base article", "add FAQ"]),
    ("link_to_jira",       ["link to jira ticket", "connect to dev issue", "create jira from zendesk"]),
    ("bulk_close",         ["bulk close tickets", "close all resolved", "mass close tickets"]),
],

"hubspot": [
    ("create_contact",    ["create a contact", "add new contact", "register lead"]),
    ("update_contact",    ["update contact info", "edit the contact", "change contact details"]),
    ("delete_contact",    ["delete contact", "remove from CRM", "archive contact"]),
    ("list_contacts",     ["list contacts", "show all contacts", "search contacts"]),
    ("create_deal",       ["create a deal", "new opportunity", "add sales deal"]),
    ("update_deal",       ["update the deal", "edit opportunity", "change deal stage"]),
    ("close_deal",        ["close the deal", "mark deal won", "won opportunity"]),
    ("lose_deal",         ["lose the deal", "mark deal lost", "close as lost"]),
    ("list_deals",        ["list deals", "show pipeline", "view opportunities"]),
    ("create_company",    ["create a company", "add new company", "register account"]),
    ("associate_contact", ["associate contact with company", "link contact to deal"]),
    ("log_activity",      ["log an activity", "record call", "track interaction"]),
    ("send_email",        ["send an email", "email the contact", "reach out via email"]),
    ("enroll_sequence",   ["enroll in sequence", "start email drip", "add to campaign"]),
    ("create_task",       ["create a task", "add a follow-up", "schedule task"]),
    ("create_note",       ["add a note", "leave a note on contact", "note the conversation"]),
    ("set_owner",         ["set owner", "assign to rep", "change contact owner"]),
    ("import_contacts",   ["import contacts", "upload CSV", "bulk import leads"]),
    ("create_report",     ["create a report", "sales analytics", "pipeline report"]),
    ("create_form",       ["create a form", "lead capture form", "landing page form"]),
],

"twilio": [
    ("send_sms",           ["send an SMS", "text the customer", "send a text message"]),
    ("send_whatsapp",      ["send WhatsApp message", "message on WhatsApp", "WhatsApp the user"]),
    ("make_call",          ["make a phone call", "call the customer", "initiate voice call"]),
    ("buy_phone_number",   ["buy a phone number", "get a new number", "provision number"]),
    ("release_number",     ["release phone number", "remove number", "cancel the number"]),
    ("list_numbers",       ["list phone numbers", "show all numbers", "my Twilio numbers"]),
    ("check_sms_status",   ["check SMS status", "was the text delivered", "message delivery status"]),
    ("check_call_status",  ["check call status", "did the call go through", "call log"]),
    ("create_subaccount",  ["create subaccount", "add a sub-account", "new Twilio account"]),
    ("get_usage_report",   ["get usage report", "check Twilio spending", "call minutes used"]),
    ("send_otp",           ["send OTP", "send verification code", "two factor code"]),
    ("verify_otp",         ["verify OTP", "check verification code", "validate the code"]),
    ("create_flow",        ["create a flow", "build IVR", "automate voice menu"]),
    ("add_to_conference",  ["add to conference call", "merge into call", "conference bridge"]),
    ("record_call",        ["record this call", "start call recording", "enable recording"]),
    ("transcribe_call",    ["transcribe the call", "get call transcript", "speech to text"]),
    ("block_number",       ["block this number", "add to blocklist", "reject calls from number"]),
    ("forward_call",       ["forward the call", "redirect to number", "call forwarding"]),
    ("set_webhook",        ["set webhook URL", "configure callback", "incoming message webhook"]),
    ("get_balance",        ["check account balance", "how much credit left", "Twilio balance"]),
],

"notion": [
    ("create_page",        ["create a page", "new page", "add a document"]),
    ("update_page",        ["update the page", "edit the doc", "change page content"]),
    ("delete_page",        ["delete the page", "remove the document", "trash this page"]),
    ("list_pages",         ["list pages", "show all docs", "find documents"]),
    ("create_database",    ["create a database", "new notion database", "table view"]),
    ("add_database_entry", ["add an entry", "new row in database", "add record"]),
    ("filter_database",    ["filter the database", "search entries", "query database"]),
    ("share_page",         ["share the page", "invite to doc", "give page access"]),
    ("create_template",    ["create a template", "save as template", "make reusable page"]),
    ("duplicate_page",     ["duplicate this page", "copy the doc", "clone page"]),
    ("move_page",          ["move the page", "change parent page", "reorganize"]),
    ("add_comment",        ["add a comment", "leave feedback on page", "comment on doc"]),
    ("resolve_comment",    ["resolve comment", "mark discussion done", "close comment thread"]),
    ("add_to_favorites",   ["add to favorites", "bookmark this page", "save to sidebar"]),
    ("create_toggle",      ["create a toggle", "collapsible section", "expandable block"]),
    ("export_page",        ["export the page", "download as PDF", "export to markdown"]),
    ("link_to_page",       ["link to another page", "add page reference", "mention a doc"]),
    ("create_kanban",      ["create kanban view", "board view", "task board"]),
    ("set_reminder",       ["set a page reminder", "remind me about this doc"]),
    ("search_workspace",   ["search workspace", "find in notion", "search all pages"]),
],

}

# ─── Situation patterns — app fingerprints for English disambiguation ──────────
# Applied to ALL intents in each namespace.
# CJK apps don't need these; their compound vocabulary is self-discriminating.

SITUATION_PATTERNS = {
    "stripe": [
        ("stripe", 2.0), ("chargeback", 2.0), ("payment gateway", 1.5),
        ("billing portal", 1.5), ("card declined", 2.0),
    ],
    "github": [
        ("github", 2.0), ("git", 1.5), ("pull request", 2.0),
        ("repository", 1.5), ("commit", 1.5),
    ],
    "slack": [
        ("slack", 2.0), ("workspace", 1.5), ("channel", 1.0),
        ("dm", 1.5), ("slack app", 2.0),
    ],
    "shopify": [
        ("shopify", 2.0), ("storefront", 1.5), ("ecommerce store", 1.5),
        ("product listing", 1.5), ("shopify admin", 2.0),
    ],
    "calendar": [
        ("calendar", 2.0), ("meeting", 1.0), ("availability", 1.5),
        ("schedule", 1.0), ("event invite", 1.5),
    ],
    "jira": [
        ("jira", 2.0), ("backlog", 1.5), ("sprint", 1.5),
        ("epic", 1.5), ("story points", 2.0),
    ],
    "zendesk": [
        ("zendesk", 2.0), ("support ticket", 1.5), ("sla", 1.5),
        ("csat", 2.0), ("help desk", 1.5),
    ],
    "hubspot": [
        ("hubspot", 2.0), ("crm", 1.5), ("pipeline", 1.5),
        ("lead", 1.0), ("deal stage", 2.0),
    ],
    "twilio": [
        ("twilio", 2.0), ("sms", 1.5), ("phone number", 1.5),
        ("ivr", 2.0), ("twilio account", 2.0),
    ],
    "notion": [
        ("notion", 2.0), ("workspace", 1.0), ("notion page", 2.0),
        ("notion database", 2.0), ("notion doc", 1.5),
    ],
}

# ─── Multilingual seeds — realistic global user base ─────────────────────────
# Different namespaces get seeds in languages their users actually use.
# Covers: zh (Chinese), ta (Tamil/India), ar (Arabic/Middle East), ja (Japanese)

MULTILINGUAL_SEEDS = {
    "stripe": {
        "charge_card": {
            "zh": ["收费失败", "银行卡被拒", "支付被拒绝", "付款失败"],
            "ta": ["கட்டணம் தோல்வியடைந்தது", "என் அட்டை வேலை செய்யவில்லை"],
            "ar": ["فشل الدفع", "بطاقتي مرفوضة", "مشكلة في الدفع"],
        },
        "refund_payment": {
            "zh": ["申请退款", "退钱给我", "我要退款"],
            "ar": ["أريد استرداد أموالي", "طلب استرداد"],
        },
        "create_subscription": {
            "ja": ["サブスクリプションを作成", "毎月の請求を設定", "定期課金を開始"],
        },
    },
    "slack": {
        "send_message": {
            "zh": ["发消息", "在频道发帖", "发送消息给团队"],
            "ja": ["メッセージを送る", "チャンネルに投稿する"],
        },
        "create_channel": {
            "zh": ["创建频道", "新建slack频道"],
            "ar": ["إنشاء قناة", "قناة جديدة"],
        },
    },
    "github": {
        "create_issue": {
            "zh": ["创建issue", "提交bug报告", "新建问题"],
            "ja": ["イシューを作成", "バグを報告"],
        },
        "create_pr": {
            "zh": ["创建拉取请求", "提交PR", "发起代码审查"],
            "ja": ["プルリクエストを作成", "PRを出す"],
        },
    },
    "zendesk": {
        "create_ticket": {
            "ar": ["إنشاء تذكرة دعم", "فتح حالة دعم", "أحتاج إلى مساعدة"],
            "ta": ["ஆதரவு டிக்கெட் உருவாக்கவும்", "உதவி தேவை"],
        },
        "reply_to_customer": {
            "ar": ["الرد على العميل", "إرسال رد"],
        },
    },
    "jira": {
        "create_ticket": {
            "ja": ["チケットを作成", "バックログに追加", "タスクを記録"],
            "zh": ["创建工单", "记录任务", "新建bug"],
        },
    },
}

# ─── Setup ────────────────────────────────────────────────────────────────────

def setup_unified():
    print("\n[setup] resetting server state...")
    curl_json("POST", "/api/reset")

    print(f"[setup] creating app '{UNIFIED_APP_ID}'...")
    curl_json("POST", "/api/apps", {"app_id": UNIFIED_APP_ID})

    total = 0
    for ns, intents in NAMESPACED_INTENTS.items():
        for intent_id, seeds in intents:
            namespaced_id = f"{ns}:{intent_id}"
            curl_json("POST", "/api/intents", {"id": namespaced_id, "seeds": seeds}, UNIFIED_APP_ID)
            total += 1

    print(f"[setup] loaded {total} namespaced intents across {len(NAMESPACED_INTENTS)} apps")

    # Multilingual seeds
    for ns, intents in MULTILINGUAL_SEEDS.items():
        for intent_id, lang_seeds in intents.items():
            namespaced_id = f"{ns}:{intent_id}"
            for lang, seeds in lang_seeds.items():
                for seed in seeds:
                    curl_json("POST", "/api/intents/add_seed",
                              {"intent_id": namespaced_id, "seed": seed, "lang": lang},
                              UNIFIED_APP_ID)
    print("[setup] added multilingual seeds (zh, ta, ar, ja)")

    # Situation fingerprint patterns
    for ns, patterns in SITUATION_PATTERNS.items():
        ns_intents = NAMESPACED_INTENTS[ns]
        for intent_id, _ in ns_intents:
            namespaced_id = f"{ns}:{intent_id}"
            for pattern, weight in patterns:
                curl_json("POST", "/api/intents/add_situation",
                          {"intent_id": namespaced_id, "pattern": pattern, "weight": weight},
                          UNIFIED_APP_ID)
    print("[setup] added situation fingerprint patterns per namespace")

# ─── Routing helpers ──────────────────────────────────────────────────────────

def route_unified(query: str, threshold: float = 0.25) -> list[dict]:
    _, data = curl("POST", "/api/route_multi", {"query": query, "threshold": threshold}, UNIFIED_APP_ID)
    if not data:
        return []
    confirmed = data.get("confirmed", [])
    candidates = data.get("candidates", [])
    results = [{"id": i["id"], "confidence": i.get("confidence", "low"), "score": i.get("score", 0)}
               for i in confirmed + candidates]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def top_ns(results: list[dict]) -> str | None:
    if not results:
        return None
    top = results[0]["id"]
    return top.split(":")[0] if ":" in top else None

def top_intent_in_ns(results: list[dict], ns: str) -> str | None:
    for r in results:
        if r["id"].startswith(f"{ns}:"):
            return r["id"][len(ns)+1:]
    return None

def any_in_ns(results: list[dict], ns: str) -> bool:
    return any(r["id"].startswith(f"{ns}:") for r in results)

# ─── Test cases ───────────────────────────────────────────────────────────────
# Queries are written as ACTUAL users would type them — slang, context-heavy,
# different vocabulary than seeds. Each is a genuine challenge.

# (query, expected_namespace, expected_intent, description)
CROSS_NS_TESTS = [

    # ── STRIPE — a merchant or developer dealing with payments ──
    ("run it on the visa ending in 4242",
     "stripe", "charge_card",         "stripe: last-4 phrasing"),
    ("customer's asking for their money back, it's been two weeks",
     "stripe", "refund_payment",       "stripe: informal refund"),
    ("she wants to be billed every month automatically",
     "stripe", "create_subscription",  "stripe: implicit recurring"),
    ("they've churned, stop the plan",
     "stripe", "cancel_subscription",  "stripe: churn = cancel"),
    ("pull up everyone who's ever transacted with us",
     "stripe", "list_customers",       "stripe: informal customer list"),
    ("we just signed an enterprise deal, get them in the system",
     "stripe", "create_customer",      "stripe: onboarding framing"),
    ("what's sitting in the account right now",
     "stripe", "retrieve_balance",     "stripe: informal balance"),
    ("wire this month's earnings to the bank",
     "stripe", "create_payout",        "stripe: wire earnings"),
    ("they opened a dispute on that $500 charge",
     "stripe", "dispute_charge",       "stripe: dispute by amount"),
    ("that customer's card keeps getting rejected",
     "stripe", "charge_card",          "stripe: card rejected = charge context"),

    # ── GITHUB — a developer at work ──
    ("found a memory leak in the auth flow, need to track it somewhere",
     "github", "create_issue",        "github: bug as tracking need"),
    ("my feature's ready, need another set of eyes",
     "github", "create_pr",           "github: PR as 'another set of eyes'"),
    ("everyone signed off, time to ship",
     "github", "merge_pr",            "github: merge as 'ship'"),
    ("spinning up a new service, need a home for the code",
     "github", "create_repo",         "github: repo as 'home for code'"),
    ("need to work on this without stepping on the main branch",
     "github", "create_branch",       "github: branch to avoid main"),
    ("we're cutting v2.1 today",
     "github", "create_release",      "github: cutting release"),
    ("where in the codebase does auth happen",
     "github", "search_code",         "github: find by question"),
    ("that hotfix branch is dead, nuke it",
     "github", "delete_branch",       "github: informal delete branch"),

    # ── SLACK — team communication ──
    ("tell engineering the outage is over",
     "slack", "send_message",         "slack: 'tell team'"),
    ("we need a dedicated space for the launch",
     "slack", "create_channel",       "slack: channel as 'space'"),
    ("loop in the new designer",
     "slack", "invite_user",          "slack: 'loop in'"),
    ("bug me at 5pm to recap standup",
     "slack", "set_reminder",         "slack: 'bug me' = reminder"),
    ("what did we decide about API versioning last month",
     "slack", "search_messages",      "slack: 'what did we decide'"),
    ("put it to a vote — tabs or spaces",
     "slack", "create_poll",          "slack: 'put to vote'"),
    ("blast the release announcement at noon",
     "slack", "schedule_message",     "slack: 'blast at time'"),
    ("I don't want to hear from that channel",
     "slack", "mute_channel",         "slack: informal mute"),

    # ── SHOPIFY — an ecommerce merchant ──
    ("launching a new hoodie colorway next week",
     "shopify", "create_product",     "shopify: 'colorway' = product"),
    ("customer changed their mind before it left the warehouse",
     "shopify", "cancel_order",       "shopify: cancel before ship"),
    ("item arrived smashed, they want compensation",
     "shopify", "refund_order",       "shopify: damage = refund"),
    ("nearly out of the medium in blue, need to update",
     "shopify", "update_inventory",   "shopify: 'nearly out' = inventory"),
    ("green light, send it out",
     "shopify", "ship_order",         "shopify: 'send it out'"),
    ("customer's asking where their stuff is",
     "shopify", "track_shipment",     "shopify: 'where is stuff'"),
    ("how many orders came in overnight",
     "shopify", "list_orders",        "shopify: 'came in overnight'"),
    ("they bought it three days ago and want to send it back",
     "shopify", "process_return",     "shopify: 'send it back'"),

    # ── CALENDAR ──
    ("block Monday 2pm for the design sync",
     "calendar", "create_event",          "calendar: 'block time'"),
    ("that meeting's dead, clear it",
     "calendar", "cancel_event",          "calendar: 'clear it'"),
    ("push the 3pm to Thursday",
     "calendar", "reschedule_event",      "calendar: 'push to day'"),
    ("what have I got going on this week",
     "calendar", "list_events",           "calendar: 'what do I have'"),
    ("make sure the product team is on the call",
     "calendar", "invite_attendee",       "calendar: 'make sure on call'"),
    ("is next Wednesday afternoon open",
     "calendar", "check_availability",    "calendar: availability check"),
    ("every Tuesday at 9, no end date",
     "calendar", "create_recurring_event","calendar: recurring pattern"),
    ("I'll be off the grid from the 15th to the 22nd",
     "calendar", "set_out_of_office",     "calendar: 'off the grid'"),

    # ── JIRA — a project manager or dev lead ──
    ("log this bug before we forget it",
     "jira", "create_ticket",         "jira: 'log' = create ticket"),
    ("reassign that to Sarah, she's taking over",
     "jira", "assign_ticket",         "jira: reassign to person"),
    ("the fix is in, mark it resolved",
     "jira", "close_ticket",          "jira: 'fix is in' = close"),
    ("kick off the new two-week cycle",
     "jira", "create_sprint",         "jira: 'two-week cycle' = sprint"),
    ("what's left in the backlog for this milestone",
     "jira", "list_tickets",          "jira: 'backlog for milestone'"),
    ("this is blocking the release, flag it urgent",
     "jira", "set_priority",          "jira: 'blocking = urgent'"),
    ("I spent 3 hours on this today",
     "jira", "log_time",              "jira: 'hours spent' = log time"),
    ("this should be under the payments initiative",
     "jira", "create_epic",           "jira: 'initiative' = epic"),

    # ── ZENDESK — a customer support agent ──
    ("new complaint in from an angry enterprise customer",
     "zendesk", "create_ticket",      "zendesk: complaint = ticket"),
    ("send him a response, we have an SLA",
     "zendesk", "reply_to_customer",  "zendesk: SLA response"),
    ("this is way above my pay grade, send it up the chain",
     "zendesk", "escalate_ticket",    "zendesk: 'up the chain'"),
    ("he submitted the same issue twice, combine them",
     "zendesk", "merge_tickets",      "zendesk: 'same issue twice'"),
    ("jot down that the customer called to clarify",
     "zendesk", "add_internal_note",  "zendesk: internal note"),
    ("the customer seems happy now, close it out",
     "zendesk", "close_ticket",       "zendesk: 'happy now' = close"),
    ("how many open cases are sitting in the queue",
     "zendesk", "list_tickets",       "zendesk: queue = list"),
    ("write a knowledge base article about this common issue",
     "zendesk", "create_help_article","zendesk: knowledge base"),

    # ── HUBSPOT — a sales rep or marketer ──
    ("just got off the phone with a hot lead, get them in the system",
     "hubspot", "create_contact",     "hubspot: 'hot lead' = create contact"),
    ("they just signed, move it to closed won",
     "hubspot", "close_deal",         "hubspot: 'signed' = close deal"),
    ("they went dark, mark this one as lost",
     "hubspot", "lose_deal",          "hubspot: 'went dark' = lost deal"),
    ("start the onboarding email sequence for this account",
     "hubspot", "enroll_sequence",    "hubspot: 'onboarding sequence'"),
    ("log that I called them but no answer",
     "hubspot", "log_activity",       "hubspot: 'no answer' = log call"),
    ("follow up with them next Tuesday",
     "hubspot", "create_task",        "hubspot: 'follow up' = task"),
    ("show me what's in the pipeline this quarter",
     "hubspot", "list_deals",         "hubspot: 'pipeline this quarter'"),
    ("we got a referral from the conference, add them",
     "hubspot", "create_contact",     "hubspot: 'referral' = create contact"),

    # ── TWILIO — a developer integrating comms ──
    ("shoot a text to confirm the appointment",
     "twilio", "send_sms",            "twilio: 'shoot a text'"),
    ("spin up a phone number for the new region",
     "twilio", "buy_phone_number",    "twilio: 'spin up number'"),
    ("did that verification code actually reach him",
     "twilio", "check_sms_status",    "twilio: verify delivery"),
    ("we need to route calls differently after hours",
     "twilio", "create_flow",         "twilio: 'after hours routing' = IVR flow"),
    ("drop the bot and connect them to a real person",
     "twilio", "forward_call",        "twilio: 'connect to real person'"),
    ("record this call for training purposes",
     "twilio", "record_call",         "twilio: 'for training' = record"),
    ("generate the two-factor code for that user",
     "twilio", "send_otp",            "twilio: '2FA code'"),
    ("how much credit do we have left",
     "twilio", "get_balance",         "twilio: credit = balance"),

    # ── NOTION — a knowledge worker ──
    ("start a fresh doc for the Q3 planning",
     "notion", "create_page",         "notion: 'fresh doc'"),
    ("add those meeting notes to the existing page",
     "notion", "update_page",         "notion: 'add to existing'"),
    ("let the design team see this",
     "notion", "share_page",          "notion: 'let team see' = share"),
    ("spin up a database to track all our vendors",
     "notion", "create_database",     "notion: 'track vendors' = database"),
    ("pull up everything about Project X",
     "notion", "search_workspace",    "notion: 'pull up everything'"),
    ("turn this into a reusable format for the team",
     "notion", "create_template",     "notion: 'reusable format'"),
    ("duplicate the onboarding page for the new hire",
     "notion", "duplicate_page",      "notion: 'copy for new hire'"),
    ("that comment thread is resolved, mark it done",
     "notion", "resolve_comment",     "notion: resolve comment"),
]

# ── Situation fingerprint tests — brand-name in query, action vocab absent ──
FINGERPRINT_TESTS = [
    ("stripe is down again, nothing's processing",         "stripe",   "stripe brand name fires"),
    ("the github repo is throwing 404s",                   "github",   "github brand + repo pattern"),
    ("slack is being weird, messages not sending",         "slack",    "slack brand fires"),
    ("shopify storefront is loading super slow",           "shopify",  "shopify + storefront pattern"),
    ("my calendar won't sync with the mobile app",         "calendar", "calendar pattern fires"),
    ("the jira backlog is out of control",                 "jira",     "jira + backlog pattern"),
    ("zendesk CSAT scores are dropping this week",         "zendesk",  "zendesk + csat pattern"),
    ("our hubspot CRM data is all wrong",                  "hubspot",  "hubspot + CRM pattern"),
    ("the twilio IVR is broken",                           "twilio",   "twilio + IVR pattern"),
    ("notion is down, can't access the workspace",         "notion",   "notion brand fires"),
]

# ── Multilingual routing tests — realistic global user queries ────────────────
# These are actual messages a real user in that language/region would send.
MULTILINGUAL_TESTS = [
    # Chinese (Simplified) — Stripe payment issues common in China
    ("收费失败怎么办",         "stripe",   "zh: charge failed (what to do)"),
    ("我的退款申请处理了吗",    "stripe",   "zh: refund status question"),
    ("我要取消订阅",           "stripe",   "zh: cancel subscription"),
    ("发消息给设计团队",        "slack",    "zh: send message to design team"),
    ("创建一个bug报告",        "github",   "zh: create bug report"),
    ("创建工单",               "jira",     "zh: create ticket (Japanese users too)"),

    # Tamil — Indian SaaS users
    ("கட்டணம் தோல்வியடைந்தது என்ன செய்வது",  "stripe",   "ta: payment failed, what to do"),

    # Arabic — Middle East customer support
    ("فشل الدفع ببطاقتي",     "stripe",   "ar: card payment failed"),
    ("إنشاء تذكرة دعم عاجلة", "zendesk",  "ar: create urgent support ticket"),
    ("أريد استرداد أموالي",   "stripe",   "ar: I want my refund"),

    # Japanese — dev tooling common in Japan
    ("サブスクリプションを開始したい",  "stripe",   "ja: start subscription"),
    ("イシューを作成してください",     "github",   "ja: please create an issue"),
    ("チケットを追加する",            "jira",     "ja: add a ticket"),
]

# ─── Scoring ──────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
RESULTS = []

def check(name: str, ok: bool, msg: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        RESULTS.append(("PASS", name, msg))
    else:
        FAIL += 1
        RESULTS.append(("FAIL", name, msg))

def score_cross_ns(label: str, verbose: bool = False) -> tuple[int, int]:
    total = len(CROSS_NS_TESTS)
    ns_correct = 0
    intent_correct = 0
    failures = []
    for query, exp_ns, exp_intent, desc in CROSS_NS_TESTS:
        results = route_unified(query)
        got_ns = top_ns(results)
        got_intent = top_intent_in_ns(results, exp_ns)
        ns_ok = got_ns == exp_ns
        intent_ok = got_intent == exp_intent
        if ns_ok: ns_correct += 1
        if intent_ok: intent_correct += 1
        if not (ns_ok and intent_ok):
            failures.append((desc, query, exp_ns, exp_intent, got_ns, got_intent, ns_ok))
    if verbose or failures:
        for desc, query, exp_ns, exp_intent, got_ns, got_intent, ns_ok in failures:
            marker = "~" if ns_ok else "✗"
            print(f"  {marker} {desc}")
            if not ns_ok:
                print(f"     expected_ns={exp_ns}, got={got_ns} | {query[:65]}")
            elif not intent_ok:
                print(f"     expected_intent={exp_intent}, got={got_intent}")
    pct_ns = 100 * ns_correct // total
    pct_int = 100 * intent_correct // total
    print(f"\n  {label}  namespace={ns_correct}/{total} ({pct_ns}%)  intent={intent_correct}/{total} ({pct_int}%)")
    return ns_correct, intent_correct

# ─── Test runners ─────────────────────────────────────────────────────────────

def run_cross_ns_tests():
    total = len(CROSS_NS_TESTS)
    print(f"\n── Cross-namespace cold start (seeds only) — {total} queries, 10 apps, 200 intents ──")
    ns_cold, intent_cold = score_cross_ns("cold start:")
    check("cold_start_ns_reasonable", ns_cold >= total * 0.3,
          f"{ns_cold}/{total} namespace correct (need ≥30%)")

    print(f"\n  applying {total} learning corrections...")
    for query, exp_ns, exp_intent, _ in CROSS_NS_TESTS:
        curl_json("POST", "/api/learn",
                  {"query": query, "intent_id": f"{exp_ns}:{exp_intent}"},
                  UNIFIED_APP_ID)

    print("\n── Cross-namespace warm start (after learning) ──────────────────────")
    ns_warm, intent_warm = score_cross_ns("warm start:")
    check("warm_start_ns_accuracy", ns_warm >= total * 0.80,
          f"{ns_warm}/{total} namespace correct after learning (need ≥80%)")
    check("warm_start_intent_accuracy", intent_warm >= total * 0.70,
          f"{intent_warm}/{total} intent correct after learning (need ≥70%)")
    check("learning_improves_ns", ns_warm > ns_cold,
          f"learning improved namespace accuracy: {ns_cold} → {ns_warm}")


def run_fingerprint_tests():
    print(f"\n── Situation fingerprint tests — {len(FINGERPRINT_TESTS)} brand-name queries ──────────")
    correct = 0
    total = len(FINGERPRINT_TESTS)
    for query, exp_ns, desc in FINGERPRINT_TESTS:
        results = route_unified(query, threshold=0.1)
        got_ns = top_ns(results)
        ok = got_ns == exp_ns
        if ok:
            correct += 1
        status = "✓" if ok else "✗"
        if not ok:
            top3 = [(r["id"], round(r["score"], 2)) for r in results[:3]]
            print(f"  {status} {desc}")
            print(f"     query: {query}")
            print(f"     expected={exp_ns}, got={got_ns} | top3={top3}")

    pct = 100 * correct // total if total else 0
    print(f"\n  fingerprint accuracy: {correct}/{total} ({pct}%)")
    check("fingerprint_accuracy", correct >= total * 0.6,
          f"{correct}/{total} brand fingerprints correct (need ≥60%)")


def run_multilingual_tests():
    print(f"\n── Multilingual routing — {len(MULTILINGUAL_TESTS)} queries (zh, ta, ar, ja) ───────────")
    correct = 0
    total = len(MULTILINGUAL_TESTS)
    by_lang = {}
    for query, exp_ns, desc in MULTILINGUAL_TESTS:
        lang = desc.split(":")[0]
        results = route_unified(query)
        got_ns = top_ns(results)
        ok = got_ns == exp_ns
        if ok:
            correct += 1
        by_lang.setdefault(lang, []).append(ok)
        status = "✓" if ok else "✗"
        if not ok:
            print(f"  {status} {desc}")
            print(f"     expected={exp_ns}, got={got_ns}")
            top3 = [(r["id"], round(r["score"], 2)) for r in results[:3]]
            print(f"     top3={top3}")
        else:
            print(f"  {status} {desc}")

    print(f"\n  Per-language accuracy:")
    for lang, results_list in sorted(by_lang.items()):
        n = sum(results_list)
        print(f"    {lang}: {n}/{len(results_list)}")
    pct = 100 * correct // total if total else 0
    print(f"\n  overall multilingual accuracy: {correct}/{total} ({pct}%)")
    check("multilingual_accuracy", correct >= total * 0.65,
          f"{correct}/{total} multilingual queries correct (need ≥65%)")


def run_generalisation_tests():
    """After learning, test varied phrasings not in the training set."""
    print("\n── Generalisation — unseen phrasings after learning ─────────────────")

    # First ensure these specific intents have been learned from slightly different queries
    priming = [
        ("she's done, cancel her plan",             "stripe:cancel_subscription"),
        ("they want their money back",               "stripe:refund_payment"),
        ("our product went live, how many orders",   "shopify:list_orders"),
        ("is Tuesday morning free",                  "calendar:check_availability"),
        ("that bug is fixed, close it",              "jira:close_ticket"),
        ("customer called, they're angry",           "zendesk:create_ticket"),
        ("give that rep the account",                "hubspot:set_owner"),
        ("two-factor for this login",                "twilio:send_otp"),
    ]
    for query, intent_id in priming:
        curl_json("POST", "/api/learn", {"query": query, "intent_id": intent_id}, UNIFIED_APP_ID)

    unseen = [
        # Stripe
        ("she no longer wants the monthly plan",          "stripe",   "cancel_subscription", "stripe: paraphrase of cancel"),
        ("the card keeps bouncing",                        "stripe",   "charge_card",          "stripe: 'bouncing' = card failure"),
        # Shopify
        ("how many came in last night",                   "shopify",  "list_orders",          "shopify: 'last night' variant"),
        # Calendar
        ("am I free Thursday morning",                    "calendar", "check_availability",   "calendar: 'am I free'"),
        # Jira
        ("we shipped it, close the ticket",               "jira",     "close_ticket",         "jira: 'shipped' = close"),
        # Zendesk
        ("an enterprise client just complained",          "zendesk",  "create_ticket",        "zendesk: 'complained' = ticket"),
        # Twilio
        ("send them a verification code",                 "twilio",   "send_otp",             "twilio: 'verification code'"),
        # Notion
        ("make a new doc for the retrospective",          "notion",   "create_page",          "notion: 'new doc'"),
    ]
    correct = 0
    for query, exp_ns, exp_intent, desc in unseen:
        results = route_unified(query)
        got_ns = top_ns(results)
        got_intent = top_intent_in_ns(results, exp_ns)
        ok = got_ns == exp_ns and got_intent == exp_intent
        if ok: correct += 1
        ns_ok = got_ns == exp_ns
        status = "✓" if ok else ("~" if ns_ok else "✗")
        if not ok:
            print(f"  {status} {desc}")
            print(f"     expected={exp_ns}:{exp_intent}, got={got_ns}:{got_intent}")
        else:
            print(f"  {status} {desc}")

    pct = 100 * correct // len(unseen)
    print(f"\n  generalisation: {correct}/{len(unseen)} ({pct}%)")
    check("generalisation_accuracy", correct >= len(unseen) * 0.5,
          f"{correct}/{len(unseen)} unseen phrasings correct (need ≥50%)")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    global BASE_URL

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:3001")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--only", choices=["cross_app", "fingerprint", "multilingual", "generalisation"])
    args = parser.parse_args()
    BASE_URL = args.base_url

    status, _ = curl("GET", "/api/health")
    if status != 200:
        print(f"Server not reachable at {BASE_URL}. Start with: ./target/release/server")
        sys.exit(1)

    if not args.skip_setup:
        setup_unified()
    else:
        print("[setup] skipped")

    only = args.only
    if not only or only == "cross_app":
        run_cross_ns_tests()
    if not only or only == "fingerprint":
        run_fingerprint_tests()
    if not only or only == "multilingual":
        run_multilingual_tests()
    if not only or only == "generalisation":
        run_generalisation_tests()

    total = PASS + FAIL
    print("\n" + "═" * 65)
    print(f"  Result: {PASS}/{total} checks passed")
    if FAIL:
        print("\n  Failed checks:")
        for st, name, msg in RESULTS:
            if st == "FAIL":
                print(f"    ✗ {name}: {msg}")
    print("═" * 65)
    sys.exit(0 if FAIL == 0 else 1)

if __name__ == "__main__":
    main()
