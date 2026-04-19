# UI Review — ASV Router

> Reviewed: 2026-04-19. Page-by-page UX audit covering usability, terminology, nav, and flow.

---

## Global Issues (fix everywhere)

### Navigation
- **Sidebar collapse loses context** — collapsed mode shows icon only, no tooltip on hover in some states. Nav group labels ("USE", "TRAIN", "CONFIG") disappear — user loses orientation.
- **"Use" group is wrong** — Route, Intents, Import are not all "use" actions. Route = Use. Intents = Configure. Import = Setup. Grouping is misleading.
- **Import belongs near Intents, not in its own group** — users think of import as how they create intents, not a separate workflow. Suggest moving Import under Intents or merging the Use group into: **Build** (Intents + Import) / **Train** / **Config**.
- **Fix Collisions in Train group is odd** — it's a one-time setup step after import, not ongoing training. Belongs in Build alongside Import.
- **Terminology inconsistency across pages** — "Tools" (MCP), "Functions" (OpenAI), "Operations" (OpenAPI). Pick one word for the concept: **tools** everywhere. Users don't care about the source format name.

### Feedback & State
- **Destructive actions are inconsistent** — Dismiss (review), Delete namespace, Delete domain: some have confirmation modals, some don't, one requires typing "delete all". Standardise: anything irreversible needs one confirmation step.
- **Saving is invisible** — changes persist silently. User never knows if something saved. Add a subtle "Saved" flash or persistent dirty indicator where needed.
- **Error messages are raw** — HTTP error strings surfaced directly to the user (e.g. "HTTP 500: Turn 2 failed: LLM API 429"). Wrap these into plain English.
- **Loading states missing** on several pages — intents list, review queue, layers stats all fetch on mount but show nothing while loading.

### Terminology
- **"Namespace"** — fine for developers, confusing for non-technical users. Consider aliasing as "Workspace" in UI copy while keeping namespace as the API term.
- **"L0/L1/L2"** — internal layer names exposed in multiple places. Move to Layers page only.
- **"Auto-learn"** — good term, keep it. But "auto-learn toggle" in Review header is too small for how important it is.

---

## Page-by-Page

### Route (/)
**What's good:** Real-time feedback, layer trace, confidence badges, relation detection.

**Problems:**
- `/help`, `/learn`, `/correct`, `/reset` commands are hidden — only discoverable via placeholder text. Add a visible "Commands" hint or a `?` button.
- `/reset` has no confirmation. One accidental enter wipes learned state.
- "LLM Review" card appears with no explanation of when/why. Label it "AI correction" or add a one-line explanation: "LLM checked this routing — here's what it found."
- Span ranges `[12, 24]` mean nothing to a user. Show the actual text substring highlighted instead.
- Trace toggle is subtle. When trace is on, make the button visibly active (filled, not just border change).

---

### Intents (/intents)
**What's good:** Domain grouping, collapsible sections, phrase management, AI generation.

**Problems:**
- Search is a toggle (click to show input). Just show the search input always — it doesn't take up space that matters.
- "Fix Collisions" link in header is the right call but looks like a secondary link. Should match the button style of "+ New".
- AI phrase generation shows a spinner but no count until complete. Show running count: "Generated 3 phrases so far…"
- Stats tab is redundant — phrase count already shown in the list. Replace Stats with a **Health** tab: phrase count, collision score, last updated.
- Saving is invisible — no indication when a description, instruction, or guardrail was saved.
- Bulk phrase add: the textarea placeholder has sample phrases — good. But the button changes label to "Add 5 phrases" only after parsing. Show count as user types.

---

### Import Landing (/import)
**What's good:** Clean card grid, fast to scan.

**Problems:**
- No indication which namespace/domain will receive the import until after clicking through. Show "Importing into: **default**" at the top.
- Search filter with no match count. When filtered, show "2 of 4 sources".
- All 4 cards look equally available. If some formats are more mature (MCP is the flagship), show that. A subtle "recommended" badge on MCP would guide users.

---

### MCP Import (/import/mcp)
**What's good:** Live Smithery search, tool list with parameter detail, domain assignment.

**Problems:**
- Three input paths (search, paste, upload) are present simultaneously and create visual noise. Lead with search. Put paste/upload behind "Can't find it? →" link.
- Tool list: unselected tools at 40% opacity are hard to read. Use a checkbox + full opacity instead.
- After searching and selecting a server, the server name disappears from the header. Keep it: "Importing from: **stripe**".
- "Fix Collisions →" button in ImportReport is good placement. But it's amber — looks like a warning. Make it violet to match the action tone.

---

### OpenAPI Import (/import/openapi)
**Problems:**
- Three input methods (URL, paste, file) shown together with unclear priority. Lead with URL input, secondary paste, tertiary upload.
- HTTP method colour scheme (GET=green, POST=blue, PUT=orange, DELETE=red, PATCH=yellow) may not be accessible. Add method text label next to colour badge.
- Collision guard notice is a tiny info box. It's one of the most important features — give it a section header.
- "Paste spec" is hidden in divider text. Make it a proper button.

---

### OpenAI Functions + LangChain Import
**Problems:**
- These two pages are nearly identical. Consider merging into a single "Function Definition" import page with a format toggle (OpenAI / LangChain).
- "Load example" button gives no feedback. Flash the textarea or show a toast.
- Terminology: "Functions" vs "Tools" — use Tools throughout.

---

### Simulate (/simulate)
**What's good:** Phase-based flow, before/after accuracy, per-query breakdown.

**Problems:**
- Phase names are bare words ("generating", "routing", "learning"). Show full sentence: "Generating test queries via LLM…"
- Before/after metrics only appear after the full run. Show baseline accuracy after the routing phase so the user sees the gap before learning starts — creates anticipation.
- Result icons: ✓ (pass), ✗ (fail), ! (degraded), ✓ (fixed) — two different meanings for ✓. Use ✦ for fixed, ✗ for fail, △ for degraded.
- "Run again (keep improving)" button appears too late. Show it as soon as the run completes.
- Missed intents only visible in detailed breakdown. Add to summary: "3 intents never detected — consider adding more seed phrases."

---

### Fix Collisions (/collisions)
**What's good:** Auto-preview on load, domain filter, colour-coded overlap scores.

**Problems:**
- "Overlap threshold — 15%" needs a one-line explanation: "Pairs sharing more than 15% of their phrase words are flagged."
- No message while auto-previewing on load. Show: "Scanning for collisions…" immediately.
- After applying fixes, show which phrases were added per intent (expandable). Currently just shows "+5 for notion-fetch" with no way to inspect.
- Phrases/pair input is technical — label it "New examples to generate per pair" and default to 5 (already is).
- No empty state illustration — when 0 collisions found, just shows bare text. Add: "✓ Your intents are well separated" with a subtle green check.

---

### Layers (/layers)
**What's good:** Probe tool is excellent for debugging. Edge management is clean.

**Problems:**
- "Distill" button is completely unexplained. Users have no idea what this does. Rename to "Generate synonyms via AI" and add a one-line description.
- L1 edge kinds (Synonym, Morphological, Abbreviation) with weights (0.88, 0.98, 0.99) — defaults shown in form but never explained. Add tooltip: "Weight 1.0 = identical. Lower = looser match."
- Probe shows L0/L1/L2 transformations — great for developers. But "no change" in gray is easy to miss. Use a subtle strikethrough or "→ unchanged" label.
- This page is intentionally expert-only. Add a header note: "Advanced — for inspecting and tuning the routing layers directly."

---

### Review (/review)
**What's good:** SSE live updates, manual + auto modes, detailed analysis panel.

**Problems:**
- Auto-learn toggle is a small switch in the top-right. This is the most important setting on this page — the entire operating mode. Make it more prominent: a clearly labelled toggle with "Auto" / "Manual" text next to it.
- Queue list item shows detected intents as raw IDs in small text. Hard to scan. Bold the intent name, small the domain prefix.
- "Analyze with AI" button — what does this do that the worker didn't already do? Explain: "Re-run LLM analysis on this query."
- AI Analysis strikethrough on wrong detections looks like deleted text. Use a red ✗ badge instead.
- Language dropdown in phrase blocks is tiny and easy to miss. Errors here mean phrases go into the wrong language. Make language selector more visible.
- Dismiss has no confirmation. One click removes from queue permanently.
- Empty queue state: "Queue is empty. Routed queries appear here for LLM review." — good. But add next step: "Enable Auto-learn to process queries automatically."

---

### Namespaces (/namespaces)
**Problems:**
- Delete button visible for all namespaces including "default". Disable for default with tooltip: "Default namespace cannot be deleted."
- "Active" badge shown but clicking it does nothing. Make it a "Switch to" button.
- Inline edit form breaks the list layout. Use a slide-out panel or modal.
- Auto-learn label in the list is a badge ("auto") — hard to distinguish from the "active" badge at a glance. Use an icon instead.

---

### Domains (/namespaces/:nsId)
**Problems:**
- Inline description edit loses focus easily — no explicit Save button. Add "Save" / "Cancel" on focus.
- Delete domain: tooltip says "intents unaffected" but users will assume deleting a domain deletes its intents. Clarify: "Domain label removed — intents keep their prefix and remain active."
- No way to rename a domain (only edit description). Add rename.
- Intent count badge is not clickable. Should link to /intents filtered to that domain.

---

### Models (/models)
**Problems:**
- Empty state says "Add one below" but the add form is below the fold on smaller screens. Anchor the empty state to the form.
- No validation of model_id format — user can enter gibberish.
- Label vs model_id — explain which appears where: "Label shows in the routing dropdown. Model ID is sent to the LLM API."
- Remove (×) has no confirmation.

---

### Languages (/languages)
**What's good:** Per-namespace config, stop words section with status, inline generate prompt.

**Problems:**
- Language picker is a dropdown that opens below the chips — on smaller viewports it may go offscreen. Consider an inline search that filters the chip grid directly.
- "Stop words are global — shared across all namespaces" label is tiny. This is a meaningful distinction — surface it more clearly.
- "Missing" stop words in amber with a "Generate" button — users won't know what stop words are. Add a one-liner: "Words like 'is', 'the', 'を' that are ignored when matching phrases."
- The generate prompt (inline after adding a language) can be skipped and forgotten. Show missing stop words as amber in the status section so users remember.
- "Regenerate" for already-generated languages — no explanation of when to use this.

---

### Settings (/settings)
**Problems:**
- LLM configuration requires editing `.env` on the server — the UI just shows the instructions. This is fine for the server tool, but should say explicitly: "Configuration requires server restart after `.env` changes."
- Confidence threshold slider: "0% = always judge, 100% = never judge" — the 100% end should be labelled "trust routing" not "never judge" which sounds dangerous.
- Warning when threshold > 0 uses "Turn 1" jargon. Replace: "Queries routed with high confidence will skip LLM review."
- Import State file picker gives no hint about expected file format. Say: "Select a `.json` export file."
- "Reset to Defaults" and "Clear All Data" are adjacent but very different in severity. Separate them visually — Clear All Data should be in a red-bordered danger zone.

---

## Priority Summary

### Fix before launch
1. `/reset` command confirmation on Route page
2. Delete confirmation standardisation (Namespace delete, Dismiss in Review)
3. Auto-learn toggle prominence on Review page
4. Import — clarify which namespace receives the import (show it at top)
5. Error messages — wrap raw HTTP errors in plain English
6. "Distill" rename → "Generate synonyms via AI"

### Fix soon after launch
7. Nav restructure — Use → Build, move Fix Collisions to Build
8. Terminology pass — Tools everywhere, Workspace alias for Namespace
9. Inline save feedback
10. Simulate — show baseline before learning starts
11. Span ranges → highlighted text substrings on Route page

### Low priority / polish
12. Stats tab → Health tab on Intents
13. Merge OpenAI Functions + LangChain import pages
14. Domain delete copy — clarify intents are unaffected
15. Models page — label vs model_id explanation
