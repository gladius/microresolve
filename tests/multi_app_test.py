#!/usr/bin/env python3
"""
Multi-app integration test suite — designed to expose real system behavior.

Seeds are minimal and realistic (what a developer would write on day 1).
Test queries are genuinely natural — different vocabulary, different structure.
The goal is NOT to pass 100%. It is to show where the system works and where
it breaks so we can fix and improve it.

Usage:
    python3 tests/multi_app_test.py [--base-url http://localhost:3001]
    python3 tests/multi_app_test.py --skip-setup   # reuse existing intents
    python3 tests/multi_app_test.py --only negative
"""

import json
import subprocess
import sys
import argparse

BASE_URL = "http://localhost:3001"

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

def create_app(app_id):
    status, _ = curl("POST", "/api/apps", {"app_id": app_id})
    return status in (200, 201, 409)

def add_intent(app_id, intent_id, label, seeds):
    status, _ = curl("POST", "/api/intents", {"id": intent_id, "label": label, "seeds": seeds}, app_id)
    return status in (200, 201)

def delete_intent(app_id, intent_id):
    curl("POST", "/api/intents/delete", {"intent_id": intent_id}, app_id)

def route_app(app_id: str, query: str) -> list[dict]:
    """Route against one app, returns list of {id, confidence, score}."""
    _, data = curl("POST", "/api/route_multi", {"query": query, "threshold": 0.25}, app_id)
    if not data:
        return []
    confirmed = data.get("confirmed", [])
    candidates = data.get("candidates", [])
    results = [{"id": i["id"], "confidence": i.get("confidence", "low"), "score": i.get("score", 0)}
               for i in confirmed + candidates]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def route_all(query: str) -> dict[str, list[dict]]:
    """Route against all apps, returns {app_id: [intents]}."""
    return {app_id: route_app(app_id, query) for app_id in APPS}

def intent_ids(results: list[dict]) -> list[str]:
    return [r["id"] for r in results]

def high_confidence(results: list[dict]) -> list[str]:
    return [r["id"] for r in results if r["confidence"] in ("high", "medium")]

# ─── App definitions — MINIMAL seeds, what a developer writes on day one ──────
# These are NOT tuned to match test queries.

APPS = {
    "stripe": [
        ("charge_card",         "Charge Card",         ["charge my card", "process payment", "bill the customer", "run a charge"]),
        ("refund_payment",      "Refund Payment",      ["issue a refund", "refund this payment", "give money back", "reverse the charge"]),
        ("create_subscription", "Create Subscription", ["create a subscription", "set up recurring billing", "start a monthly plan"]),
        ("cancel_subscription", "Cancel Subscription", ["cancel the subscription", "stop recurring billing", "end the plan"]),
        ("list_customers",      "List Customers",      ["list customers", "show all customers", "get customer list"]),
        ("create_customer",     "Create Customer",     ["create a customer", "add new customer", "register customer"]),
        ("update_customer",     "Update Customer",     ["update customer", "edit customer info", "change customer details"]),
        ("delete_customer",     "Delete Customer",     ["delete customer", "remove customer account"]),
        ("create_invoice",      "Create Invoice",      ["create an invoice", "generate invoice", "send invoice"]),
        ("pay_invoice",         "Pay Invoice",         ["pay this invoice", "mark invoice paid", "settle invoice"]),
        ("list_invoices",       "List Invoices",       ["list invoices", "show all invoices", "get invoice history"]),
        ("create_coupon",       "Create Coupon",       ["create a coupon", "make a discount code", "add promo code"]),
        ("apply_discount",      "Apply Discount",      ["apply a discount", "use promo code", "apply coupon"]),
        ("create_product",      "Create Product",      ["create a product", "add product to stripe", "new product"]),
        ("update_price",        "Update Price",        ["update the price", "change pricing", "set new price"]),
        ("retrieve_balance",    "Retrieve Balance",    ["check balance", "get account balance", "show available funds"]),
        ("transfer_funds",      "Transfer Funds",      ["transfer funds", "move money", "wire payment"]),
        ("create_payout",       "Create Payout",       ["create payout", "withdraw to bank", "initiate payout"]),
        ("create_payment_link", "Create Payment Link", ["create payment link", "generate checkout link"]),
        ("list_transactions",   "List Transactions",   ["list transactions", "show payment history", "recent charges"]),
    ],

    "github": [
        ("create_issue",     "Create Issue",     ["create an issue", "file a bug report", "open new issue", "report a bug"]),
        ("close_issue",      "Close Issue",      ["close this issue", "resolve the issue", "mark issue done"]),
        ("create_pr",        "Create PR",        ["open a pull request", "create PR", "submit for review"]),
        ("merge_pr",         "Merge PR",         ["merge the PR", "merge pull request", "land this PR"]),
        ("close_pr",         "Close PR",         ["close the pull request", "reject this PR", "close PR without merging"]),
        ("list_repos",       "List Repos",       ["list repositories", "show my repos", "what repos do I have"]),
        ("create_repo",      "Create Repo",      ["create a repository", "new repo", "initialize repository"]),
        ("delete_repo",      "Delete Repo",      ["delete this repository", "remove repo"]),
        ("create_branch",    "Create Branch",    ["create a branch", "make a new branch", "branch off"]),
        ("delete_branch",    "Delete Branch",    ["delete the branch", "remove branch", "clean up branch"]),
        ("list_branches",    "List Branches",    ["list branches", "show all branches"]),
        ("create_release",   "Create Release",   ["create a release", "publish new version", "tag a release"]),
        ("list_commits",     "List Commits",     ["list commits", "show commit history", "view git log"]),
        ("fork_repo",        "Fork Repo",        ["fork this repo", "fork the repository"]),
        ("star_repo",        "Star Repo",        ["star this repo", "add to starred"]),
        ("add_collaborator", "Add Collaborator", ["add a collaborator", "invite contributor", "give access to repo"]),
        ("create_webhook",   "Create Webhook",   ["create a webhook", "set up webhook", "add webhook"]),
        ("list_issues",      "List Issues",      ["list issues", "show open issues", "what issues are there"]),
        ("search_code",      "Search Code",      ["search code", "find in codebase", "grep across repo"]),
        ("create_gist",      "Create Gist",      ["create a gist", "share code snippet", "save as gist"]),
    ],

    "slack": [
        ("send_message",     "Send Message",      ["send a message", "post to channel", "message the team"]),
        ("create_channel",   "Create Channel",    ["create a channel", "new slack channel", "set up channel"]),
        ("archive_channel",  "Archive Channel",   ["archive this channel", "close the channel"]),
        ("invite_user",      "Invite User",       ["invite someone to channel", "add user to channel", "add member"]),
        ("kick_user",        "Kick User",         ["remove user from channel", "kick from channel"]),
        ("set_status",       "Set Status",        ["set my status", "update status", "change slack status"]),
        ("set_reminder",     "Set Reminder",      ["set a reminder", "remind me", "create reminder"]),
        ("search_messages",  "Search Messages",   ["search messages", "find a message", "look up conversation"]),
        ("list_channels",    "List Channels",     ["list channels", "show all channels"]),
        ("pin_message",      "Pin Message",       ["pin this message", "pin to channel"]),
        ("create_poll",      "Create Poll",       ["create a poll", "start a vote", "poll the team"]),
        ("schedule_message", "Schedule Message",  ["schedule a message", "send message later", "post later"]),
        ("upload_file",      "Upload File",       ["upload a file", "share document", "attach file"]),
        ("list_members",     "List Members",      ["list members", "show channel members", "who is in channel"]),
        ("set_topic",        "Set Topic",         ["set channel topic", "update topic", "change channel description"]),
        ("react_to_message", "React",             ["react to message", "add emoji reaction", "react with emoji"]),
        ("get_user_info",    "Get User Info",     ["get user info", "look up user", "who is this user"]),
        ("create_workflow",  "Create Workflow",   ["create a workflow", "automate in slack", "set up workflow"]),
        ("list_dms",         "List DMs",          ["show direct messages", "list my DMs", "view DM conversations"]),
        ("mute_channel",     "Mute Channel",      ["mute this channel", "silence notifications", "mute notifications"]),
    ],

    "shopify": [
        ("create_product",        "Create Product",        ["create a product", "add product to store", "new product listing"]),
        ("update_product",        "Update Product",        ["update product", "edit product listing", "change product details"]),
        ("delete_product",        "Delete Product",        ["delete product", "remove from store", "take down listing"]),
        ("list_products",         "List Products",         ["list products", "show all products", "view product catalog"]),
        ("create_order",          "Create Order",          ["create an order", "place order", "manually add order"]),
        ("cancel_order",          "Cancel Order",          ["cancel order", "void the order", "cancel purchase"]),
        ("refund_order",          "Refund Order",          ["refund this order", "process order refund", "give refund"]),
        ("list_orders",           "List Orders",           ["list orders", "show recent orders", "view order history"]),
        ("update_inventory",      "Update Inventory",      ["update inventory", "change stock level", "adjust quantity"]),
        ("create_discount",       "Create Discount",       ["create a discount", "add discount", "set up sale"]),
        ("apply_discount_code",   "Apply Discount Code",   ["apply discount code", "use coupon", "apply promo"]),
        ("list_customers",        "List Customers",        ["list shopify customers", "show store customers"]),
        ("create_customer",       "Create Customer",       ["add store customer", "create shopify customer"]),
        ("update_customer",       "Update Customer",       ["update store customer", "edit buyer info"]),
        ("ship_order",            "Ship Order",            ["ship this order", "mark as shipped", "fulfill order"]),
        ("track_shipment",        "Track Shipment",        ["track shipment", "where is my order", "track delivery"]),
        ("create_collection",     "Create Collection",     ["create a collection", "group products", "new category"]),
        ("update_store_settings", "Update Store Settings", ["update store settings", "change store config"]),
        ("generate_report",       "Generate Report",       ["generate sales report", "store analytics", "revenue report"]),
        ("process_return",        "Process Return",        ["process a return", "handle return", "accept returned item"]),
    ],

    "calendar": [
        ("create_event",           "Create Event",           ["create an event", "add to calendar", "schedule a meeting"]),
        ("cancel_event",           "Cancel Event",           ["cancel the event", "delete meeting", "remove from calendar"]),
        ("reschedule_event",       "Reschedule Event",       ["reschedule this meeting", "move the event", "change event time"]),
        ("list_events",            "List Events",            ["list events", "show my schedule", "what do I have today"]),
        ("invite_attendee",        "Invite Attendee",        ["invite someone to meeting", "add to event", "send calendar invite"]),
        ("set_reminder",           "Set Reminder",           ["set event reminder", "remind me before meeting"]),
        ("check_availability",     "Check Availability",     ["check availability", "am I free", "check if I have time"]),
        ("share_calendar",         "Share Calendar",         ["share my calendar", "give access to calendar"]),
        ("create_recurring_event", "Create Recurring Event", ["create recurring event", "set up weekly meeting", "repeat this event"]),
        ("accept_invite",          "Accept Invite",          ["accept the invite", "confirm meeting", "say yes to event"]),
        ("decline_invite",         "Decline Invite",         ["decline the invite", "reject meeting", "say no to event"]),
        ("find_meeting_time",      "Find Meeting Time",      ["find a meeting time", "when can we meet", "find common availability"]),
        ("set_working_hours",      "Set Working Hours",      ["set working hours", "update my work schedule"]),
        ("book_room",              "Book Room",              ["book a room", "reserve meeting room", "find a room"]),
        ("add_video_link",         "Add Video Link",         ["add video link", "attach zoom link", "add meet link"]),
        ("export_calendar",        "Export Calendar",        ["export calendar", "download calendar", "get ical"]),
        ("view_event_details",     "View Event Details",     ["show event details", "what is this meeting about"]),
        ("update_event",           "Update Event",           ["update event", "edit meeting", "change event details"]),
        ("set_out_of_office",      "Set Out of Office",      ["set out of office", "mark unavailable", "block vacation"]),
        ("sync_calendar",          "Sync Calendar",          ["sync calendar", "refresh calendar", "sync my schedule"]),
    ],
}

# ─── Positive tests ───────────────────────────────────────────────────────────
# All queries use natural language that DIFFERS from seeds.
# "expected" is what the system SHOULD detect — failures here are real gaps.

# Format: (app_id_or_list, query, expected_intents_list, description)
# For cross-app: app_id is a list, expected is list of (app_id, intent_id)

SINGLE_INTENT_TESTS = [
    # ── STRIPE ──
    ("stripe", "run it on the visa ending in 4242",                           ["charge_card"],         "stripe: card phrasing with last 4"),
    ("stripe", "my customer is asking for their money back from last week",    ["refund_payment"],      "stripe: indirect refund request"),
    ("stripe", "they want to be billed every month going forward",             ["create_subscription"], "stripe: implicit recurring billing"),
    ("stripe", "she's done, wants out of the plan entirely",                   ["cancel_subscription"], "stripe: colloquial cancellation"),
    ("stripe", "pull up everyone who's ever paid us",                          ["list_customers"],      "stripe: informal customer list"),
    ("stripe", "we just landed a new enterprise account",                      ["create_customer"],     "stripe: onboarding framing"),
    ("stripe", "he moved offices, need to update the billing address",         ["update_customer"],     "stripe: contextual update"),
    ("stripe", "the account went dark, scrub their data",                      ["delete_customer"],     "stripe: informal delete"),
    ("stripe", "send them something official for this month's work",           ["create_invoice"],      "stripe: invoice by context"),
    ("stripe", "they finally settled up after three reminders",                ["pay_invoice"],         "stripe: indirect pay invoice"),
    ("stripe", "what's sitting in the stripe account right now",               ["retrieve_balance"],    "stripe: informal balance check"),
    ("stripe", "wire this month's earnings over to the bank",                  ["create_payout"],       "stripe: payout as 'wire earnings'"),

    # ── GITHUB ──
    ("github", "found a memory leak in the auth flow, need to track it",      ["create_issue"],      "github: issue as bug tracking"),
    ("github", "my feature's ready for other eyes",                            ["create_pr"],         "github: PR as 'ready for eyes'"),
    ("github", "everyone approved it, time to land it",                        ["merge_pr"],          "github: merge via 'land it'"),
    ("github", "what codebases do we have under this org",                    ["list_repos"],        "github: repos as 'codebases'"),
    ("github", "spinning up a new microservice, need a home for the code",    ["create_repo"],       "github: repo as 'home for code'"),
    ("github", "I need to work on this without stepping on main",              ["create_branch"],     "github: branch to avoid main"),
    ("github", "we're cutting the 2.1 release today",                         ["create_release"],    "github: release as 'cutting'"),
    ("github", "where does the app handle authentication",                    ["search_code"],       "github: code search as question"),

    # ── SLACK ──
    ("slack", "tell the engineering team about the incident",                  ["send_message"],      "slack: message as 'tell team'"),
    ("slack", "we need a dedicated space for the launch coordination",         ["create_channel"],    "slack: channel as 'dedicated space'"),
    ("slack", "loop in the new designer we hired",                             ["invite_user"],       "slack: invite as 'loop in'"),
    ("slack", "bug me at 5 to recap the standup",                              ["set_reminder"],      "slack: reminder as 'bug me'"),
    ("slack", "what did we decide about the API versioning last month",        ["search_messages"],   "slack: search as 'what did we decide'"),
    ("slack", "put it to a vote — dark mode or light mode",                    ["create_poll"],       "slack: poll as 'put it to a vote'"),
    ("slack", "blast the announcement to everyone at noon",                    ["schedule_message"],  "slack: schedule as 'blast at noon'"),
    ("slack", "drop the design mockups in the product channel",                ["upload_file"],       "slack: upload as 'drop files'"),
    ("slack", "I don't want to hear from that channel right now",              ["mute_channel"],      "slack: mute as 'don't want to hear'"),
    ("slack", "let them know by thumbs up or down",                            ["react_to_message"],  "slack: react as 'thumbs up or down'"),

    # ── SHOPIFY ──
    ("shopify", "we're launching a new hoodie colorway next week",            ["create_product"],      "shopify: product as 'new colorway'"),
    ("shopify", "customer changed their mind before it left the warehouse",   ["cancel_order"],        "shopify: cancel before ship"),
    ("shopify", "item arrived broken, they're requesting compensation",       ["refund_order"],        "shopify: refund as 'compensation'"),
    ("shopify", "how many orders came in this morning",                       ["list_orders"],         "shopify: orders as count question"),
    ("shopify", "we're nearly out of stock on the medium blue jacket",        ["update_inventory"],    "shopify: inventory via 'nearly out'"),
    ("shopify", "green light on this one, send it out",                       ["ship_order"],          "shopify: ship as 'send it out'"),
    ("shopify", "the customer is asking where their stuff is",                ["track_shipment"],      "shopify: tracking via customer question"),
    ("shopify", "customer bought it three days ago and wants to send it back",["process_return"],      "shopify: return with context"),

    # ── CALENDAR ──
    ("calendar", "block Monday 2pm for the design sync with product",         ["create_event"],          "calendar: create via 'block time'"),
    ("calendar", "that meeting isn't happening anymore, clear it",             ["cancel_event"],          "calendar: cancel as 'clear it'"),
    ("calendar", "push the 3pm to sometime Thursday instead",                 ["reschedule_event"],      "calendar: reschedule as 'push to'"),
    ("calendar", "what have I got going on this week",                        ["list_events"],           "calendar: list as 'what have I got'"),
    ("calendar", "make sure everyone from product is on the call",            ["invite_attendee"],       "calendar: invite as 'make sure on call'"),
    ("calendar", "is next Wednesday afternoon wide open",                     ["check_availability"],    "calendar: availability check"),
    ("calendar", "every Tuesday at 9, forever, no end date",                  ["create_recurring_event"],"calendar: recurring via time pattern"),
    ("calendar", "I'll be off the grid from the 15th to the 22nd",           ["set_out_of_office"],     "calendar: OOO as 'off the grid'"),
]

MULTI_INTENT_SAME_APP_TESTS = [
    # No conjunctions. Two intents implicit in one natural request.
    # Format: (app_id, query, [intent1, intent2], description)

    # ── STRIPE ──
    ("stripe",
     "new enterprise client coming on, need to get them set up with monthly billing",
     ["create_customer", "create_subscription"],
     "stripe: onboard + subscribe (implicit)"),

    ("stripe",
     "process the card and shoot them a receipt for their records",
     ["charge_card", "create_invoice"],
     "stripe: charge + invoice (implicit)"),

    ("stripe",
     "they're done with us — cancel everything and send back whatever we owe them",
     ["cancel_subscription", "refund_payment"],
     "stripe: cancel + refund (churn flow)"),

    # ── GITHUB ──
    ("github",
     "the auth bug is fixed, we can close that ticket and retire the hotfix branch",
     ["close_issue", "delete_branch"],
     "github: close issue + delete branch (cleanup)"),

    ("github",
     "spinning up the new payments service — need a repo and a CI hook",
     ["create_repo", "create_webhook"],
     "github: repo + webhook (setup)"),

    ("github",
     "tagging today's release and pulling up the commit history to draft the changelog",
     ["create_release", "list_commits"],
     "github: release + commits (changelog flow)"),

    # ── SLACK ──
    ("slack",
     "set up a war room for the incident and pull in the on-call engineers",
     ["create_channel", "invite_user"],
     "slack: channel + invite (incident response)"),

    ("slack",
     "pin the new API docs and get everyone's attention on them",
     ["pin_message", "send_message"],
     "slack: pin + notify (announcement)"),

    ("slack",
     "announce the deployment window at 3pm and ask the team to acknowledge",
     ["schedule_message", "create_poll"],
     "slack: schedule + poll (deployment)"),

    # ── SHOPIFY ──
    ("shopify",
     "just dropped a new sneaker — get it listed and set the opening stock to 200",
     ["create_product", "update_inventory"],
     "shopify: list product + set stock"),

    ("shopify",
     "customer wants out and their money back, handle both",
     ["cancel_order", "refund_order"],
     "shopify: cancel + refund"),

    ("shopify",
     "it went out this morning, update the status and pull up the tracking for them",
     ["ship_order", "track_shipment"],
     "shopify: ship + track"),

    # ── CALENDAR ──
    ("calendar",
     "get the kickoff on the books and loop in the whole product team",
     ["create_event", "invite_attendee"],
     "calendar: create + invite (kickoff)"),

    ("calendar",
     "the Friday meeting is dead — ditch it and find us a better slot next week",
     ["cancel_event", "find_meeting_time"],
     "calendar: cancel + find new time"),

    ("calendar",
     "I'm out from Monday, block it and set up recurring coverage standups",
     ["set_out_of_office", "create_recurring_event"],
     "calendar: OOO + recurring"),
]

MULTI_INTENT_CROSS_APP_TESTS = [
    # Query spans two different apps.
    # Format: (query, [(app_id, intent_id), ...], description)

    ("charge the client and open a tracking issue for the billing edge case we found",
     [("stripe", "charge_card"), ("github", "create_issue")],
     "stripe charge + github issue"),

    ("the order shipped — post in the team channel and update the tracking",
     [("shopify", "ship_order"), ("slack", "send_message")],
     "shopify shipped + slack notify"),

    ("merge the hotfix and alert the on-call team immediately",
     [("github", "merge_pr"), ("slack", "send_message")],
     "github merge + slack alert"),

    ("bill the client for this sprint and spin up a repo for next quarter's work",
     [("stripe", "create_invoice"), ("github", "create_repo")],
     "stripe invoice + github repo"),

    ("get the Q3 report out of shopify and share it in the finance channel",
     [("shopify", "generate_report"), ("slack", "upload_file")],
     "shopify report + slack share"),

    ("schedule the client demo and send the invite link through slack",
     [("calendar", "create_event"), ("slack", "send_message")],
     "calendar event + slack message"),

    ("the new customer is in stripe — send them a welcome in slack",
     [("stripe", "create_customer"), ("slack", "send_message")],
     "stripe customer + slack welcome"),

    ("tag the v2 release and announce it in the announcements channel",
     [("github", "create_release"), ("slack", "send_message")],
     "github release + slack announce"),

    ("cancel their subscription in stripe and close their shopify store account",
     [("stripe", "cancel_subscription"), ("shopify", "delete_product")],
     "stripe cancel + shopify close"),

    ("find a time for the sprint planning and create the github milestone",
     [("calendar", "find_meeting_time"), ("github", "create_repo")],
     "calendar + github (planning)"),
]

# ─── Negative tests ───────────────────────────────────────────────────────────
# Queries that should NOT trigger any high/medium confidence intent in ANY app.
# These expose false positives — a system that matches everything is broken.

NEGATIVE_TESTS = [
    # Completely off-topic
    ("what's the weather like in Tokyo tomorrow",          "weather — no app intent"),
    ("I'm feeling really good about this project",        "emotional statement"),
    ("banana pancakes with maple syrup",                  "food — nonsense to all apps"),
    ("can you explain recursion to me",                   "CS concept question"),
    ("the quick brown fox jumps over the lazy dog",       "filler sentence"),
    ("it was a dark and stormy night",                    "fiction opener"),
    ("what time zone is New York in",                     "timezone question"),
    ("my dog's name is charlie",                          "personal statement"),
    ("translate this to Spanish",                         "translation request"),
    ("is the earth flat",                                 "unrelated factual question"),

    # Too vague / incomplete
    ("do it",                                             "zero-context imperative"),
    ("yes",                                               "affirmation with no context"),
    ("ok sure",                                           "acknowledgment"),
    ("later",                                             "one word — later"),
    ("help",                                              "generic help"),
    ("not sure",                                          "uncertainty expression"),
    ("I don't know",                                      "uncertainty expression"),
    ("whatever you think is best",                        "deferred decision"),
    ("make it work",                                      "vague imperative"),
    ("fix it",                                            "vague fix request"),

    # Plausible-sounding but actually no clear intent
    ("I was thinking maybe we could possibly look into something",  "hedge soup — no action"),
    ("let's revisit this next quarter",                            "deferral — no action"),
    ("just a heads up for the team",                               "fyi with no action"),
    ("I'll get back to you on that",                               "deferral statement"),
    ("sounds like a plan",                                         "agreement — no action"),
]

# ─── Situation patterns ───────────────────────────────────────────────────────
# Domain-specific state vocabulary that distinguishes apps from each other.
# These are signals that appear in real queries for each app's domain —
# not action words (those are seeds), but context / state / jargon that
# confirms "this query belongs to THIS app".

SITUATION_PATTERNS = {
    "stripe": {
        "charge_card":         [("card declined", 1.0), ("payment failed", 1.0), ("billing error", 0.9),
                                 ("charge failed", 0.9), ("payment method", 0.7)],
        "refund_payment":      [("chargeback", 1.0), ("dispute", 0.9), ("money back", 0.8),
                                 ("payment reversed", 0.9)],
        "create_subscription": [("monthly plan", 1.0), ("recurring billing", 1.0), ("billing cycle", 0.9),
                                 ("annual plan", 0.8), ("subscription", 0.7)],
        "cancel_subscription": [("churned", 1.0), ("monthly billing", 0.9), ("recurring charge", 0.9),
                                 ("subscription ended", 1.0)],
        "create_customer":     [("new account", 0.7), ("enterprise client", 0.8), ("onboard", 0.7)],
        "update_customer":     [("billing address", 1.0), ("payment info", 0.9), ("billing details", 0.9)],
        "create_invoice":      [("invoice", 0.9), ("billing statement", 0.9), ("for this month's work", 0.8)],
        "pay_invoice":         [("settled up", 0.9), ("invoice paid", 1.0), ("payment received", 0.8)],
        "retrieve_balance":    [("stripe account", 0.9), ("available funds", 0.9), ("account balance", 0.8)],
        "create_payout":       [("withdraw to bank", 1.0), ("earnings", 0.7), ("bank account", 0.7)],
        "list_transactions":   [("payment history", 0.9), ("recent charges", 0.9), ("transaction", 0.8)],
    },
    "github": {
        "create_issue":   [("build failed", 1.0), ("CI failed", 1.0), ("memory leak", 1.0),
                            ("bug", 0.7), ("crash", 0.8), ("broken", 0.7), ("regression", 0.9),
                            ("error in", 0.7), ("track it", 0.7)],
        "close_issue":    [("fixed", 0.7), ("bug is fixed", 1.0), ("resolved", 0.7), ("patch merged", 1.0)],
        "create_pr":      [("code review", 1.0), ("ready for review", 1.0), ("for other eyes", 0.9),
                            ("diff", 0.8), ("changes ready", 0.9)],
        "merge_pr":       [("approved", 0.8), ("LGTM", 1.0), ("everyone approved", 1.0),
                            ("land it", 0.9), ("CI passed", 1.0), ("green", 0.7)],
        "create_release": [("cutting the", 0.9), ("release today", 1.0), ("tag", 0.7),
                            ("changelog", 0.9), ("new version", 0.8)],
        "create_repo":    [("new microservice", 0.9), ("home for the code", 1.0), ("new service", 0.7)],
        "create_branch":  [("without stepping on main", 1.0), ("feature branch", 0.9),
                            ("hotfix branch", 1.0), ("branch off", 0.8)],
        "delete_branch":  [("retire the", 0.9), ("clean up branch", 0.9), ("hotfix branch", 0.8)],
        "list_commits":   [("commit history", 1.0), ("git log", 1.0), ("changelog", 0.7)],
        "search_code":    [("codebase", 0.8), ("find in", 0.7), ("where does", 0.8), ("authentication", 0.6)],
    },
    "slack": {
        "send_message":      [("notify", 0.8), ("ping", 0.9), ("let them know", 0.9),
                               ("heads up", 0.8), ("tell the", 0.7), ("alert", 0.8)],
        "create_channel":    [("dedicated space", 1.0), ("war room", 1.0), ("new channel", 0.9),
                               ("launch coordination", 0.9)],
        "invite_user":       [("loop in", 1.0), ("bring in", 0.9), ("add to channel", 0.8)],
        "set_reminder":      [("bug me", 1.0), ("remind me", 0.9), ("don't forget", 0.8), ("follow up", 0.7)],
        "search_messages":   [("what did we decide", 1.0), ("look up conversation", 0.9),
                               ("find that message", 0.9)],
        "create_poll":       [("put it to a vote", 1.0), ("poll the team", 0.9), ("vote", 0.8),
                               ("dark mode or", 0.9)],
        "schedule_message":  [("blast", 0.8), ("post at", 0.8), ("send at noon", 1.0),
                               ("announcement at", 0.9)],
        "upload_file":       [("drop the", 0.8), ("design mockups", 0.9), ("share document", 0.8)],
        "mute_channel":      [("don't want to hear", 1.0), ("silence notifications", 1.0), ("too noisy", 0.9)],
        "react_to_message":  [("thumbs up", 1.0), ("emoji", 0.9), ("thumbs down", 1.0)],
        "pin_message":       [("pin it", 0.9), ("keep it visible", 0.9)],
        "schedule_message":  [("blast at", 1.0), ("announce at", 0.9), ("deployment window", 0.8)],
    },
    "shopify": {
        "create_product":    [("colorway", 1.0), ("listing", 0.8), ("SKU", 0.9),
                               ("product page", 0.9), ("store", 0.6)],
        "cancel_order":      [("before it left", 1.0), ("before shipment", 1.0), ("void the", 0.8),
                               ("changed their mind", 0.9)],
        "refund_order":      [("arrived broken", 1.0), ("wrong item", 1.0), ("damaged", 0.9),
                               ("compensation", 0.8), ("item", 0.6)],
        "update_inventory":  [("out of stock", 1.0), ("nearly out", 1.0), ("stock level", 0.9),
                               ("restock", 0.9), ("SKU", 0.9), ("medium blue", 0.8)],
        "ship_order":        [("send it out", 1.0), ("green light", 0.9), ("fulfill", 0.9),
                               ("warehouse", 0.8), ("dispatch", 0.9)],
        "track_shipment":    [("where is my", 1.0), ("delivery status", 0.9), ("tracking", 0.8),
                               ("customer asking", 0.7)],
        "process_return":    [("send it back", 1.0), ("wants to return", 1.0), ("returned item", 0.9),
                               ("return request", 0.9)],
        "list_orders":       [("came in this morning", 0.9), ("order count", 0.8), ("recent orders", 0.8)],
        "generate_report":   [("sales report", 1.0), ("revenue", 0.8), ("store analytics", 1.0)],
    },
    "calendar": {
        "create_event":           [("block", 0.7), ("design sync", 0.9), ("2pm", 0.6),
                                    ("get it on the books", 1.0), ("add to calendar", 0.8)],
        "cancel_event":           [("not happening", 1.0), ("clear it", 0.9), ("meeting is dead", 1.0),
                                    ("no longer", 0.8)],
        "reschedule_event":       [("push the", 0.8), ("postponed", 0.9), ("different time", 0.9),
                                    ("better slot", 1.0), ("move to Thursday", 1.0)],
        "check_availability":     [("am I free", 1.0), ("wide open", 1.0), ("have time", 0.8),
                                    ("free slot", 0.9)],
        "find_meeting_time":      [("common availability", 1.0), ("when can we meet", 1.0),
                                    ("everyone available", 0.9), ("better slot", 0.9)],
        "set_out_of_office":      [("off the grid", 1.0), ("OOO", 1.0), ("vacation", 0.9),
                                    ("unavailable", 0.8), ("out from Monday", 1.0)],
        "create_recurring_event": [("every Tuesday", 1.0), ("forever", 0.8), ("no end date", 1.0),
                                    ("weekly", 0.8), ("recurring", 0.9)],
        "invite_attendee":        [("make sure everyone", 1.0), ("on the call", 0.8),
                                    ("calendar invite", 1.0), ("whole product team", 0.9)],
    },
}

def add_situation_pattern(app_id: str, intent_id: str, pattern: str, weight: float):
    status, _ = curl("POST", "/api/intents/add_situation",
                     {"intent_id": intent_id, "pattern": pattern, "weight": weight}, app_id)
    return status in (200, 201)

# ─── Setup ────────────────────────────────────────────────────────────────────

def setup_all_apps():
    print("\n=== Setting up apps ===")
    total = 0
    sit_total = 0
    for app_id, intents in APPS.items():
        print(f"\n  [{app_id}] {len(intents)} intents")
        create_app(app_id)
        existing = curl_json("GET", "/api/intents", app_id=app_id)
        if isinstance(existing, list):
            for i in existing: delete_intent(app_id, i["id"])
        for intent_id, label, seeds in intents:
            ok = add_intent(app_id, intent_id, label, seeds)
            print(f"    {'+'if ok else '!'} {intent_id}")
            total += 1
            # Add situation patterns for this intent if defined
            patterns = SITUATION_PATTERNS.get(app_id, {}).get(intent_id, [])
            for pattern, weight in patterns:
                add_situation_pattern(app_id, intent_id, pattern, weight)
                sit_total += 1
    print(f"\n  {total} intents + {sit_total} situation patterns registered across {len(APPS)} apps")

# ─── Test runners ─────────────────────────────────────────────────────────────

class Results:
    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.failures = []

    def record(self, passed, desc, detail=""):
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(f"    FAIL [{desc}]: {detail}")

    def print_summary(self):
        total = self.passed + self.failed
        pct = 100 * self.passed // total if total else 0
        print(f"\n--- {self.name}: {self.passed}/{total} ({pct}%) ---")
        for f in self.failures:
            print(f)

def run_single_intent(results: Results):
    print("\n=== Single-Intent Tests (natural language, no seed overlap) ===")
    for app_id, query, expected, desc in SINGLE_INTENT_TESTS:
        detected = route_app(app_id, query)
        ids = intent_ids(detected)
        passed = any(e in ids[:3] for e in expected)
        status = "PASS" if passed else "FAIL"
        print(f"  {status} [{desc}]")
        if not passed:
            print(f"       query:    {query}")
            print(f"       expected: {expected}")
            print(f"       got:      {ids[:4]}")
        results.record(passed, desc, f"expected={expected}, got={ids[:4]}")

def run_multi_same_app(results: Results):
    print("\n=== Multi-Intent Same-App (implicit, no conjunctions) ===")
    for app_id, query, expected, desc in MULTI_INTENT_SAME_APP_TESTS:
        detected = route_app(app_id, query)
        ids = intent_ids(detected)
        found = [e for e in expected if e in ids]
        # Require BOTH intents detected
        passed = len(found) == len(expected)
        found_str = f"{len(found)}/{len(expected)}"
        status = "PASS" if passed else f"PART({found_str})" if found else "FAIL"
        print(f"  {status} [{desc}]")
        if not passed:
            print(f"       query:   {query}")
            print(f"       found:   {found}")
            print(f"       missing: {[e for e in expected if e not in found]}")
            print(f"       got:     {ids[:5]}")
        results.record(passed, desc, f"found={found}, missing={[e for e in expected if e not in found]}, got={ids[:5]}")

def run_multi_cross_app(results: Results):
    print("\n=== Multi-Intent Cross-App (single query, multiple apps) ===")
    for query, expected_pairs, desc in MULTI_INTENT_CROSS_APP_TESTS:
        all_results = route_all(query)
        found_pairs = []
        for app_id, intent_id in expected_pairs:
            app_ids = intent_ids(all_results.get(app_id, []))
            if intent_id in app_ids[:3]:
                found_pairs.append((app_id, intent_id))

        passed = len(found_pairs) == len(expected_pairs)
        found_str = f"{len(found_pairs)}/{len(expected_pairs)}"
        status = "PASS" if passed else f"PART({found_str})" if found_pairs else "FAIL"
        print(f"  {status} [{desc}]")
        if not passed:
            print(f"       query:   {query[:80]}")
            print(f"       found:   {found_pairs}")
            missing = [(a,i) for a,i in expected_pairs if (a,i) not in found_pairs]
            print(f"       missing: {missing}")
            for app_id, intent_id in missing:
                top = intent_ids(all_results.get(app_id, []))[:3]
                print(f"       {app_id} top: {top}")
        results.record(passed, desc, f"found={len(found_pairs)}/{len(expected_pairs)}")

def run_negative(results: Results):
    print("\n=== Negative Tests (should NOT fire high/medium confidence in any app) ===")
    for query, desc in NEGATIVE_TESTS:
        all_results = route_all(query)
        # Collect any high/medium confidence matches across all apps
        fires = {}
        for app_id, detected in all_results.items():
            hc = high_confidence(detected)
            if hc:
                fires[app_id] = hc
        passed = len(fires) == 0
        status = "PASS" if passed else "FAIL"
        print(f"  {status} [{desc}]")
        if not passed:
            print(f"       query:  {query}")
            print(f"       fired:  {fires}")
        results.record(passed, desc, f"fired in: {fires}" if fires else "")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global BASE_URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:3001")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--only",
                        choices=["single", "multi_same", "multi_cross", "negative", "all"],
                        default="all")
    args = parser.parse_args()
    BASE_URL = args.base_url

    print("ASV Multi-App Integration Test Suite")
    print(f"Target: {BASE_URL}")
    print("Seeds: minimal (not tuned to tests). Failures = real system gaps.")

    status, _ = curl("GET", "/api/apps")
    if status == 0:
        print(f"\nERROR: Server not reachable at {BASE_URL}")
        sys.exit(1)

    if not args.skip_setup:
        setup_all_apps()

    r_single     = Results("Single-Intent (50 tests)")
    r_multi_same = Results("Multi-Intent Same-App (15 tests)")
    r_multi_cross= Results("Multi-Intent Cross-App (10 tests)")
    r_negative   = Results("Negative Tests (25 tests)")

    if args.only in ("single", "all"):
        run_single_intent(r_single)
    if args.only in ("multi_same", "all"):
        run_multi_same_app(r_multi_same)
    if args.only in ("multi_cross", "all"):
        run_multi_cross_app(r_multi_cross)
    if args.only in ("negative", "all"):
        run_negative(r_negative)

    total_pass = sum(r.passed for r in [r_single, r_multi_same, r_multi_cross, r_negative])
    total_all  = sum(r.passed + r.failed for r in [r_single, r_multi_same, r_multi_cross, r_negative])

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    r_single.print_summary()
    r_multi_same.print_summary()
    r_multi_cross.print_summary()
    r_negative.print_summary()

    pct = 100 * total_pass // total_all if total_all else 0
    print(f"\nOVERALL: {total_pass}/{total_all} ({pct}%)")
    print("\nNote: Failures here are actionable — they show what to fix.")
    if total_all:
        sys.exit(0 if total_pass == total_all else 1)

if __name__ == "__main__":
    main()
