//! Knowledge Mesh Experiment: Five-index architecture with online learning.
//!
//! Tests three phases:
//! - Phase 1: Baseline routing index only, with learning
//! - Phase 2: Routing + Paraphrase index, with learning
//! - Phase 3: Full mesh (all 5 indexes), with learning
//!
//! Each phase runs:
//! - First pass: 30 scenarios with sequential learning
//! - Second pass: same 30 scenarios (no new learning) = generalization
//! - New scenarios: 10 unseen scenarios = true generalization
//!
//! Run with: cargo run --bin mesh_experiment --release

use asv_router::{IntentType, Router, MultiRouteOutput, MultiRouteResult};
use std::collections::{HashMap, HashSet};

// ============= Data Structures =============

#[derive(serde::Deserialize, Clone)]
struct Scenario {
    id: String,
    category: String,
    #[allow(dead_code)]
    persona: serde_json::Value,
    turns: Vec<Turn>,
}

#[derive(serde::Deserialize, Clone)]
struct Turn {
    message: String,
    ground_truth: Vec<String>,
}

#[derive(serde::Deserialize)]
struct ModifierEntry {
    pattern: String,
    effect: String,
    #[serde(default = "default_scope")]
    scope: usize,
}

fn default_scope() -> usize { 2 }

#[derive(Debug, Clone, serde::Serialize)]
struct TurnRecord {
    phase: String,
    pass_type: String,
    scenario: String,
    turn: usize,
    message: String,
    word_count: usize,
    ground_truth: Vec<String>,
    detected_top5: Vec<DetectedIntent>,
    exact_match: bool,
    top5_recall: bool,
    false_positives: usize,
    missed: Vec<String>,
    corrections_applied_so_far: usize,
    learning_curve_point: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
struct DetectedIntent {
    id: String,
    score: f32,
    source: String,
}

// ============= Paraphrase Index =============

struct ParaphraseIndex {
    /// phrase -> (intent_id, confidence_weight)
    phrases: HashMap<String, (String, f32)>,
    /// Built from phrases keys for fast scanning
    automaton: Option<aho_corasick::AhoCorasick>,
    patterns: Vec<String>,
}

impl ParaphraseIndex {
    fn new() -> Self {
        Self {
            phrases: HashMap::new(),
            automaton: None,
            patterns: Vec::new(),
        }
    }

    fn load_from_json(data: &HashMap<String, Vec<String>>) -> Self {
        let mut idx = Self::new();
        for (intent_id, phrases) in data {
            for phrase in phrases {
                let lower = phrase.to_lowercase();
                idx.phrases.insert(lower, (intent_id.clone(), 0.8));
            }
        }
        idx.rebuild_automaton();
        idx
    }

    fn rebuild_automaton(&mut self) {
        self.patterns = self.phrases.keys().cloned().collect();
        if self.patterns.is_empty() {
            self.automaton = None;
            return;
        }
        // Sort by length descending so longer phrases match first
        self.patterns.sort_by(|a, b| b.len().cmp(&a.len()));
        self.automaton = aho_corasick::AhoCorasick::builder()
            .match_kind(aho_corasick::MatchKind::LeftmostLongest)
            .build(&self.patterns)
            .ok();
    }

    fn scan(&self, message: &str) -> Vec<(String, f32, usize)> {
        let lower = message.to_lowercase();
        let mut results: Vec<(String, f32, usize)> = Vec::new();
        let mut seen_intents: HashSet<String> = HashSet::new();

        if let Some(ref ac) = self.automaton {
            for mat in ac.find_iter(&lower) {
                let pattern = &self.patterns[mat.pattern().as_usize()];
                if let Some((intent_id, weight)) = self.phrases.get(pattern) {
                    if seen_intents.insert(intent_id.clone()) {
                        results.push((intent_id.clone(), *weight, mat.start()));
                    }
                }
            }
        }
        results
    }

    fn add_phrase(&mut self, phrase: &str, intent_id: &str, weight: f32) {
        let lower = phrase.to_lowercase();
        if lower.split_whitespace().count() >= 2 && lower.len() >= 5 {
            self.phrases.insert(lower, (intent_id.to_string(), weight));
        }
    }

    fn learn_from_message(&mut self, message: &str, intent_id: &str) {
        let lower = message.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        // Extract windows of 3-5 consecutive words as new phrases
        for window_size in 3..=5 {
            if words.len() >= window_size {
                for start in 0..=(words.len() - window_size) {
                    let phrase: String = words[start..start + window_size].join(" ");
                    if phrase.len() >= 8 {
                        self.add_phrase(&phrase, intent_id, 0.5);
                    }
                }
            }
        }
        // Also add the full message if it's not too long
        if words.len() >= 3 && words.len() <= 12 {
            self.add_phrase(&lower, intent_id, 0.6);
        }
    }
}

// ============= Correction Index =============

struct CorrectionIndex {
    mappings: HashMap<String, String>,
}

impl CorrectionIndex {
    fn load_from_json(data: &HashMap<String, String>) -> Self {
        Self { mappings: data.clone() }
    }

    fn apply(&self, message: &str) -> String {
        let words: Vec<&str> = message.split_whitespace().collect();
        let mut result: Vec<String> = Vec::new();
        for word in &words {
            let lower = word.to_lowercase();
            // Strip trailing punctuation for lookup
            let clean: String = lower.chars()
                .take_while(|c| c.is_alphanumeric() || *c == '\'')
                .collect();
            let trailing: String = lower.chars()
                .skip_while(|c| c.is_alphanumeric() || *c == '\'')
                .collect();
            if let Some(replacement) = self.mappings.get(&clean) {
                result.push(format!("{}{}", replacement, trailing));
            } else {
                result.push(word.to_string());
            }
        }
        result.join(" ")
    }
}

// ============= Modifier Index =============

#[derive(Clone)]
struct ModifierPattern {
    words: Vec<String>,
    effect: ModifierEffect,
    scope: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum ModifierEffect {
    Suppress,
    Reduce,
    Boost,
    Flag,
}

struct ModifierIndex {
    patterns: Vec<ModifierPattern>,
}

impl ModifierIndex {
    fn load_from_json(entries: &[ModifierEntry]) -> Self {
        let patterns = entries.iter().map(|e| {
            let effect = match e.effect.as_str() {
                "suppress" => ModifierEffect::Suppress,
                "reduce" => ModifierEffect::Reduce,
                "boost" => ModifierEffect::Boost,
                _ => ModifierEffect::Flag,
            };
            ModifierPattern {
                words: e.pattern.to_lowercase().split_whitespace().map(String::from).collect(),
                effect,
                scope: e.scope,
            }
        }).collect();
        Self { patterns }
    }

    /// Returns a map of word_index -> effect for words in the message
    fn analyze(&self, message: &str) -> HashMap<usize, ModifierEffect> {
        let words: Vec<String> = message.to_lowercase()
            .split_whitespace()
            .map(|w| w.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect())
            .collect();

        let mut effects: HashMap<usize, ModifierEffect> = HashMap::new();

        for pattern in &self.patterns {
            let pat_len = pattern.words.len();
            if words.len() < pat_len { continue; }

            for i in 0..=(words.len() - pat_len) {
                let matches = pattern.words.iter()
                    .zip(&words[i..i + pat_len])
                    .all(|(p, w)| p == w);

                if matches {
                    // Apply effect to the next `scope` words after the pattern
                    let start = i + pat_len;
                    for j in start..words.len().min(start + pattern.scope) {
                        effects.insert(j, pattern.effect);
                    }
                }
            }
        }
        effects
    }

    /// Apply modifiers to route_multi output: suppress negated, boost urgent
    fn apply_to_results(
        &self,
        message: &str,
        output: &MultiRouteOutput,
        soft: bool,
    ) -> Vec<MultiRouteResult> {
        let effects = self.analyze(message);
        if effects.is_empty() {
            return output.intents.clone();
        }

        let words: Vec<&str> = message.to_lowercase().leak().split_whitespace().collect();
        let has_suppress = effects.values().any(|e| *e == ModifierEffect::Suppress);
        let has_boost = effects.values().any(|e| *e == ModifierEffect::Boost);

        let mut results = output.intents.clone();

        if has_boost {
            for r in &mut results {
                r.score *= 1.2;
            }
        }

        if has_suppress {
            let suppressed_words: HashSet<String> = effects.iter()
                .filter(|(_, e)| **e == ModifierEffect::Suppress)
                .filter_map(|(idx, _)| words.get(*idx).map(|w| w.to_string()))
                .collect();

            if soft {
                // Soft mode: only DEMOTE (halve score), don't remove entirely
                // And only if the intent has a low score relative to top
                let top_score = results.iter().map(|r| r.score).fold(0.0f32, f32::max);
                for r in &mut results {
                    let id_parts: Vec<&str> = r.id.split('_').collect();
                    if id_parts.iter().any(|p| suppressed_words.contains(*p)) {
                        r.score *= 0.3; // heavy discount but don't remove
                    }
                }
                // Only remove if score drops below 10% of top
                results.retain(|r| r.score >= top_score * 0.1);
            } else {
                results.retain(|r| {
                    let id_parts: Vec<&str> = r.id.split('_').collect();
                    !id_parts.iter().any(|p| suppressed_words.contains(*p))
                });
            }
        }

        results
    }
}

// ============= Exclusion Index =============

struct ExclusionIndex {
    /// Static exclusions (from config)
    static_exclusions: HashMap<String, HashSet<String>>,
    /// Learned exclusions: (intent_a, intent_b) -> false_pair_count
    learned: HashMap<(String, String), u32>,
    /// Minimum count before learned exclusion activates
    min_count: u32,
}

impl ExclusionIndex {
    fn load_from_json(data: &HashMap<String, Vec<String>>) -> Self {
        let mut static_exclusions = HashMap::new();
        for (intent, excluded) in data {
            let set: HashSet<String> = excluded.iter().cloned().collect();
            static_exclusions.insert(intent.clone(), set);
        }
        Self {
            static_exclusions,
            learned: HashMap::new(),
            min_count: 3,
        }
    }

    fn should_exclude(&self, intent_a: &str, intent_b: &str, soft: bool) -> bool {
        if !soft {
            // Hard mode: check static exclusions
            if let Some(excluded) = self.static_exclusions.get(intent_a) {
                if excluded.contains(intent_b) { return true; }
            }
            if let Some(excluded) = self.static_exclusions.get(intent_b) {
                if excluded.contains(intent_a) { return true; }
            }
        }
        // Soft mode: only use learned exclusions (no static)

        // Check learned (both modes)
        let key = if intent_a < intent_b {
            (intent_a.to_string(), intent_b.to_string())
        } else {
            (intent_b.to_string(), intent_a.to_string())
        };
        let threshold = if soft { 5 } else { self.min_count };
        if let Some(&count) = self.learned.get(&key) {
            return count >= threshold;
        }

        false
    }

    fn apply_to_results(&self, results: &[MultiRouteResult], soft: bool) -> Vec<MultiRouteResult> {
        if results.is_empty() { return results.to_vec(); }

        let mut kept: Vec<MultiRouteResult> = Vec::new();
        let mut excluded_ids: HashSet<String> = HashSet::new();

        // Process in score order (highest first)
        let mut sorted = results.to_vec();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        for intent in &sorted {
            if excluded_ids.contains(&intent.id) { continue; }

            // Check if this intent is excluded by any already-kept intent
            let excluded_by_kept = kept.iter().any(|k| self.should_exclude(&k.id, &intent.id, soft));
            if excluded_by_kept {
                excluded_ids.insert(intent.id.clone());
                continue;
            }

            kept.push(intent.clone());
        }

        // Restore position ordering
        kept.sort_by_key(|i| i.position);
        kept
    }

    fn learn_false_pair(&mut self, intent_a: &str, intent_b: &str) {
        let key = if intent_a < intent_b {
            (intent_a.to_string(), intent_b.to_string())
        } else {
            (intent_b.to_string(), intent_a.to_string())
        };
        *self.learned.entry(key).or_insert(0) += 1;
    }
}

// ============= Mesh Router =============

struct MeshRouter {
    router: Router,
    paraphrase_index: Option<ParaphraseIndex>,
    correction_index: Option<CorrectionIndex>,
    modifier_index: Option<ModifierIndex>,
    exclusion_index: Option<ExclusionIndex>,
    corrections_applied: usize,
    /// When true, modifier/exclusion use softer thresholds
    soft_mode: bool,
}

impl MeshRouter {
    fn new(router: Router) -> Self {
        Self {
            router,
            paraphrase_index: None,
            correction_index: None,
            modifier_index: None,
            exclusion_index: None,
            corrections_applied: 0,
            soft_mode: false,
        }
    }

    fn with_paraphrase(mut self, idx: ParaphraseIndex) -> Self {
        self.paraphrase_index = Some(idx);
        self
    }

    fn with_correction(mut self, idx: CorrectionIndex) -> Self {
        self.correction_index = Some(idx);
        self
    }

    fn with_modifier(mut self, idx: ModifierIndex) -> Self {
        self.modifier_index = Some(idx);
        self
    }

    fn with_exclusion(mut self, idx: ExclusionIndex) -> Self {
        self.exclusion_index = Some(idx);
        self
    }

    fn with_soft_mode(mut self) -> Self {
        self.soft_mode = true;
        self
    }

    /// Route a message through all active indexes
    fn route(&self, message: &str, threshold: f32) -> Vec<DetectedIntent> {
        // Step 1: Apply correction index (normalize text)
        let normalized = if let Some(ref ci) = self.correction_index {
            ci.apply(message)
        } else {
            message.to_string()
        };

        // Step 2: Route through the main routing index
        let routing_output = self.router.route_multi(&normalized, threshold);

        // Step 3: Scan paraphrase index (on original message, not normalized)
        let mut paraphrase_hits: Vec<(String, f32)> = Vec::new();
        if let Some(ref pi) = self.paraphrase_index {
            for (intent_id, weight, _pos) in pi.scan(message) {
                paraphrase_hits.push((intent_id, weight));
            }
        }

        // Step 4: Merge routing + paraphrase results
        let mut merged: HashMap<String, (f32, String)> = HashMap::new();

        for intent in &routing_output.intents {
            merged.insert(intent.id.clone(), (intent.score, "routing".to_string()));
        }

        for (intent_id, weight) in &paraphrase_hits {
            let entry = merged.entry(intent_id.clone())
                .or_insert((0.0, "paraphrase".to_string()));
            // Combine: if paraphrase matched, boost score
            entry.0 += weight * 3.0; // paraphrase matches are high-confidence
            if entry.1 == "routing" {
                entry.1 = "both".to_string();
            }
        }

        // Build result list
        let mut results: Vec<MultiRouteResult> = merged.iter().map(|(id, (score, _))| {
            MultiRouteResult {
                id: id.clone(),
                score: *score,
                position: 0,
                span: (0, 0),
                intent_type: IntentType::Action,
                confidence: "low".to_string(),
                source: "routing".to_string(),
                negated: false,
            }
        }).collect();

        // Step 5: Apply modifier index
        if let Some(ref mi) = self.modifier_index {
            results = mi.apply_to_results(message, &MultiRouteOutput {
                intents: results,
                relations: vec![],
                metadata: HashMap::new(),
            }, self.soft_mode);
        }

        // Step 6: Apply exclusion index
        if let Some(ref ei) = self.exclusion_index {
            results = ei.apply_to_results(&results, self.soft_mode);
        }

        // Sort by score descending, take top 5
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(5);

        // Convert to DetectedIntent with source info
        results.iter().map(|r| {
            let source = merged.get(&r.id)
                .map(|(_, s)| s.clone())
                .unwrap_or_else(|| "routing".to_string());
            DetectedIntent {
                id: r.id.clone(),
                score: (r.score * 100.0).round() / 100.0,
                source,
            }
        }).collect()
    }

    /// Learn from ground truth: update all active indexes
    fn learn(&mut self, message: &str, ground_truth: &[String], detected: &[DetectedIntent]) {
        let gt_set: HashSet<&str> = ground_truth.iter().map(|s| s.as_str()).collect();
        let det_set: HashSet<&str> = detected.iter().map(|d| d.id.as_str()).collect();

        // Learn missed intents
        for gt_intent in &gt_set {
            if !det_set.contains(gt_intent) {
                // This intent was missed — learn the message for it
                self.router.learn(message, gt_intent);
                if let Some(ref mut pi) = self.paraphrase_index {
                    pi.learn_from_message(message, gt_intent);
                }
                self.corrections_applied += 1;
            }
        }

        // Correct false positives
        for det_intent in &det_set {
            if !gt_set.contains(det_intent) {
                // This was a false positive
                // If there's a ground truth intent that was detected, correct toward it
                if let Some(correct) = gt_set.iter().next() {
                    self.router.correct(message, det_intent, correct);
                    self.corrections_applied += 1;
                }

                // Learn exclusion pairs (false positive + correct intent)
                if let Some(ref mut ei) = self.exclusion_index {
                    for gt in &gt_set {
                        ei.learn_false_pair(det_intent, gt);
                    }
                }
            }
        }

        // For intents that were correctly detected, reinforce
        for gt_intent in gt_set.intersection(&det_set) {
            self.router.learn(message, gt_intent);
            if let Some(ref mut pi) = self.paraphrase_index {
                pi.learn_from_message(message, gt_intent);
            }
        }
    }

    fn rebuild_paraphrase_automaton(&mut self) {
        if let Some(ref mut pi) = self.paraphrase_index {
            pi.rebuild_automaton();
        }
    }
}

// ============= Evaluation =============

fn evaluate_turn(
    ground_truth: &[String],
    detected: &[DetectedIntent],
) -> (bool, bool, usize, Vec<String>) {
    let gt_set: HashSet<&str> = ground_truth.iter().map(|s| s.as_str()).collect();
    let det_ids: Vec<&str> = detected.iter().map(|d| d.id.as_str()).collect();
    let det_set: HashSet<&str> = det_ids.iter().copied().collect();

    // Exact match: detected == ground_truth exactly
    let exact = gt_set == det_set;

    // Top-5 recall: all GT intents appear in detected (regardless of extras)
    let top5_recall = gt_set.iter().all(|gt| det_set.contains(gt));

    // False positives: detected but not in GT
    let fp_count = det_set.difference(&gt_set).count();

    // Missed: in GT but not detected
    let missed: Vec<String> = gt_set.difference(&det_set).map(|s| s.to_string()).collect();

    (exact, top5_recall, fp_count, missed)
}

// ============= Stats =============

struct PhaseStats {
    total: usize,
    exact_pass: usize,
    top5_recall: usize,
    total_fp: usize,
    total_missed: usize,
    by_category: HashMap<String, (usize, usize, usize)>, // (total, exact, recall)
    by_wordcount: HashMap<String, (usize, usize, usize)>,
    learning_curve: Vec<(usize, f32, f32)>, // (turn_number, exact_rate, recall_rate)
}

fn compute_phase_stats(records: &[TurnRecord]) -> PhaseStats {
    let total = records.len();
    let exact_pass = records.iter().filter(|r| r.exact_match).count();
    let top5_recall = records.iter().filter(|r| r.top5_recall).count();
    let total_fp: usize = records.iter().map(|r| r.false_positives).sum();
    let total_missed: usize = records.iter().map(|r| r.missed.len()).sum();

    let mut by_category: HashMap<String, (usize, usize, usize)> = HashMap::new();
    let mut by_wc: HashMap<String, (usize, usize, usize)> = HashMap::new();

    for r in records {
        // Extract category from scenario ID
        let cat = r.scenario.split('_').next().unwrap_or("unknown").to_string();
        // Actually let's use the full pass_type grouping
        let bucket = if r.word_count <= 5 { "1-5" }
            else if r.word_count <= 10 { "6-10" }
            else if r.word_count <= 20 { "11-20" }
            else if r.word_count <= 40 { "21-40" }
            else { "41+" };

        let e = by_wc.entry(bucket.to_string()).or_insert((0, 0, 0));
        e.0 += 1;
        if r.exact_match { e.1 += 1; }
        if r.top5_recall { e.2 += 1; }
    }

    // Learning curve: compute running accuracy at granular milestones
    let milestones: Vec<usize> = (1..=records.len()).step_by(10)
        .chain([1, 5, 25, 50, 75, 100, 138].iter().copied())
        .filter(|&m| m <= records.len())
        .collect::<std::collections::BTreeSet<usize>>()
        .into_iter().collect();
    let mut learning_curve = Vec::new();
    for &m in &milestones {
        let slice = &records[..m];
        let exact_rate = slice.iter().filter(|r| r.exact_match).count() as f32 / m as f32;
        let recall_rate = slice.iter().filter(|r| r.top5_recall).count() as f32 / m as f32;
        learning_curve.push((m, exact_rate * 100.0, recall_rate * 100.0));
    }

    PhaseStats {
        total, exact_pass, top5_recall, total_fp, total_missed,
        by_category, by_wordcount: by_wc, learning_curve,
    }
}

fn write_phase_report(out: &mut String, name: &str, stats: &PhaseStats) {
    let exact_pct = (stats.exact_pass as f64 / stats.total.max(1) as f64) * 100.0;
    let recall_pct = (stats.top5_recall as f64 / stats.total.max(1) as f64) * 100.0;
    let fp_avg = stats.total_fp as f64 / stats.total.max(1) as f64;

    out.push_str(&format!("\n### {}\n\n", name));
    out.push_str("| Metric | Value |\n|---|---|\n");
    out.push_str(&format!("| Total turns | {} |\n", stats.total));
    out.push_str(&format!("| Exact pass | {} ({:.1}%) |\n", stats.exact_pass, exact_pct));
    out.push_str(&format!("| Top-5 recall | {} ({:.1}%) |\n", stats.top5_recall, recall_pct));
    out.push_str(&format!("| Avg FP/turn | {:.2} |\n", fp_avg));
    out.push_str(&format!("| Total missed | {} |\n", stats.total_missed));

    if !stats.learning_curve.is_empty() {
        out.push_str("\n**Learning curve:**\n\n");
        out.push_str("| Turn | Exact% | Recall% |\n|---|---|---|\n");
        for (turn, exact, recall) in &stats.learning_curve {
            out.push_str(&format!("| {} | {:.1}% | {:.1}% |\n", turn, exact, recall));
        }
    }

    if !stats.by_wordcount.is_empty() {
        out.push_str("\n**By word count:**\n\n");
        out.push_str("| Words | Total | Exact | Recall | Exact% | Recall% |\n|---|---|---|---|---|---|\n");
        for bucket in &["1-5", "6-10", "11-20", "21-40", "41+"] {
            if let Some((t, e, r)) = stats.by_wordcount.get(*bucket) {
                let ep = if *t > 0 { (*e as f64 / *t as f64) * 100.0 } else { 0.0 };
                let rp = if *t > 0 { (*r as f64 / *t as f64) * 100.0 } else { 0.0 };
                out.push_str(&format!("| {} | {} | {} | {} | {:.1}% | {:.1}% |\n",
                    bucket, t, e, r, ep, rp));
            }
        }
    }
}

// ============= Intent Setup =============

fn setup_router() -> Router {
    let mut router = Router::new();
    let actions: &[(&str, &[&str])] = &[
        ("cancel_order", &["cancel my order","I need to cancel an order I just placed","please stop my order from shipping","I changed my mind and want to cancel the purchase","how do I cancel something I ordered yesterday","cancel order number","I accidentally ordered the wrong thing, cancel it","withdraw my order before it ships"]),
        ("refund", &["I want a refund","get my money back","I received a damaged item and need a refund","the product was nothing like the description, refund please","how long does it take to process a return","I returned it two weeks ago and still no refund","I want to return this for a full refund","money back"]),
        ("contact_human", &["talk to a human","I need to speak with a real person not a bot","connect me to customer service please","this bot is useless, get me an agent","transfer me to a representative","I want to talk to someone who can actually help","live agent please","escalate this to a manager"]),
        ("reset_password", &["reset my password","I forgot my password and can't log in","my account is locked out","how do I change my password","the password reset email never arrived","I keep getting invalid password error","locked out of my account need help getting back in","send me a password reset link"]),
        ("update_address", &["change my address","I moved and need to update my shipping address","update my delivery address before it ships","my address is wrong on the order","change the shipping destination","I need to correct my mailing address","ship it to a different address instead","new address for future orders"]),
        ("billing_issue", &["wrong charge on my account","I was charged twice for the same order","there's a billing error on my statement","I see an unauthorized charge","you overcharged me by twenty dollars","my credit card was charged the wrong amount","dispute a charge","the amount on my bill doesn't match what I ordered"]),
        ("change_plan", &["upgrade my plan","I want to switch to the premium subscription","downgrade my account to the basic tier","change my subscription plan","what plans are available for upgrade","I want a cheaper plan","switch me to the annual billing"]),
        ("close_account", &["delete my account","I want to close my account permanently","how do I deactivate my profile","remove all my data and close the account","I no longer want to use this service","cancel my membership entirely","please terminate my account"]),
        ("report_fraud", &["someone used my card without permission","I think my account was hacked","there are transactions I did not make","report unauthorized access to my account","fraudulent activity on my card","someone stole my identity and made purchases","I need to report suspicious charges"]),
        ("apply_coupon", &["I have a discount code","apply my coupon to the order","where do I enter a promo code","this coupon isn't working","I forgot to apply my discount before checkout","can I use two coupons on one order","my promotional code was rejected"]),
        ("schedule_callback", &["can someone call me back","I'd like to schedule a phone call","have an agent call me at this number","request a callback for tomorrow morning","I prefer a phone call over chat","when can I expect a call back","set up a time for support to call me"]),
        ("file_complaint", &["I want to file a formal complaint","this is unacceptable, I'm filing a complaint","how do I report poor service","I want to submit a grievance","your service has been terrible and I want it documented","escalate my complaint to upper management","I need to make an official complaint"]),
        ("request_invoice", &["send me an invoice for my purchase","I need a receipt for tax purposes","can I get a PDF of my invoice","email me the billing statement","I need documentation of this transaction","where can I download my invoice","generate an invoice for order number"]),
        ("pause_subscription", &["pause my subscription for a month","I want to temporarily stop my membership","can I freeze my account without canceling","put my plan on hold","suspend my subscription until next quarter","I'm traveling and want to pause billing","temporarily deactivate my subscription"]),
        ("transfer_funds", &["transfer money to another account","send funds to my savings account","move money between my accounts","I want to wire money to someone","initiate a bank transfer","how do I send money to another person","transfer fifty dollars to my checking"]),
        ("add_payment_method", &["add a new credit card to my account","I want to register a different payment method","update my card information","save a new debit card for payments","link my bank account for direct payment","replace my expired card on file","add PayPal as a payment option"]),
        ("remove_item", &["remove an item from my order","take this product out of my cart","I don't want one of the items in my order anymore","delete the second item from my purchase","can I remove something before it ships","take off the extra item I added by mistake","drop one item from my order"]),
        ("reorder", &["reorder my last purchase","I want to buy the same thing again","repeat my previous order","order the same items as last time","can I quickly reorder what I got before","place the same order again","buy this product again"]),
        ("upgrade_shipping", &["upgrade to express shipping","I need this delivered faster","can I switch to overnight delivery","expedite my shipment","change my shipping to two-day delivery","I'll pay extra for faster shipping","rush delivery please"]),
        ("gift_card_redeem", &["redeem my gift card","I have a gift card code to apply","how do I use a gift certificate","enter my gift card balance","apply a gift card to my purchase","my gift card isn't being accepted","check the balance on my gift card"]),
    ];
    let contexts: &[(&str, &[&str])] = &[
        ("track_order", &["where is my package","track my order","my order still hasn't arrived and it's been a week","I need a shipping update on my recent purchase","when will my delivery arrive","package tracking number","it says delivered but I never got it","how long until my order gets here"]),
        ("check_balance", &["check my balance","how much money is in my account","what's my current account balance","show me my available funds","I need to know how much I have left","account summary","remaining balance on my card","what do I owe right now"]),
        ("account_status", &["is my account in good standing","check my account status","am I verified","what is the state of my account","is my account active or suspended","show me my account details","my account status page"]),
        ("order_history", &["show me my past orders","what did I order last month","view my order history","list all my previous purchases","I need to see what I bought before","pull up my recent orders","my purchase history"]),
        ("payment_history", &["show me my payment history","list all charges to my account","what payments have I made","view my transaction log","when was my last payment","how much have I spent this month","pull up my billing history"]),
        ("shipping_options", &["what shipping methods are available","how much does express shipping cost","what are my delivery options","do you offer free shipping","compare shipping speeds and prices","international shipping rates","same day delivery available"]),
        ("return_policy", &["what is your return policy","how many days do I have to return something","can I return a used product","do you accept returns without receipt","what items are not returnable","is there a restocking fee for returns","return and exchange policy"]),
        ("product_availability", &["is this item in stock","when will this product be available again","check if you have this in my size","is this item available for delivery","out of stock notification","do you carry this brand","product availability in my area"]),
        ("warranty_info", &["what does the warranty cover","how long is the warranty period","is my product still under warranty","warranty claim process","does this come with a manufacturer warranty","extended warranty options","what voids the warranty"]),
        ("loyalty_points", &["how many reward points do I have","check my loyalty balance","when do my points expire","how can I redeem my reward points","how many points do I earn per dollar","my rewards program status","transfer loyalty points"]),
        ("subscription_status", &["what plan am I on","when does my subscription renew","show me my current plan details","how much am I paying monthly","when is my next billing date","what features are included in my plan","subscription renewal date"]),
        ("delivery_estimate", &["when will my order arrive","estimated delivery date","how long does shipping take","expected arrival for my package","delivery timeframe for my area","how many business days until delivery","will it arrive before the weekend"]),
        ("price_check", &["how much does this cost","what is the price of this item","is this on sale right now","price match guarantee","compare prices for this product","total cost including shipping","any discounts on this item"]),
        ("account_limits", &["what is my spending limit","daily transfer limit on my account","maximum withdrawal amount","how much can I send per transaction","increase my account limits","what are the restrictions on my account","transaction limits for my plan"]),
        ("transaction_details", &["show me details of my last transaction","what was that charge for","transaction reference number lookup","I need details about a specific payment","when exactly was this charge made","who was the merchant for this transaction","breakdown of charges on my statement"]),
        ("eligibility_check", &["am I eligible for an upgrade","do I qualify for a discount","can I apply for this program","check my eligibility for the promotion","what are the requirements to qualify","am I eligible for a credit increase","do I meet the criteria for this offer"]),
    ];
    for (id, seeds) in actions { router.add_intent(id, seeds); router.set_intent_type(id, IntentType::Action); }
    for (id, seeds) in contexts { router.add_intent(id, seeds); router.set_intent_type(id, IntentType::Context); }
    router
}

// ============= Phase Runner =============

fn run_phase(
    mesh: &mut MeshRouter,
    scenarios: &[Scenario],
    phase_name: &str,
    pass_type: &str,
    do_learn: bool,
    threshold: f32,
) -> Vec<TurnRecord> {
    run_phase_with_rate(mesh, scenarios, phase_name, pass_type, if do_learn { 1.0 } else { 0.0 }, threshold)
}

fn run_phase_with_rate(
    mesh: &mut MeshRouter,
    scenarios: &[Scenario],
    phase_name: &str,
    pass_type: &str,
    learn_rate: f64,
    threshold: f32,
) -> Vec<TurnRecord> {
    let mut records = Vec::new();
    // Granular milestones: every 10 turns + key points
    let learning_milestones: HashSet<usize> = (1..=200).step_by(10)
        .chain([1, 5, 25, 50, 75, 100, 138].iter().copied())
        .collect();
    let mut global_turn = 0;
    let mut rng_state: u64 = 42; // deterministic pseudo-random for partial learning

    for scenario in scenarios {
        for (turn_idx, turn) in scenario.turns.iter().enumerate() {
            global_turn += 1;

            // Route BEFORE learning
            let detected = mesh.route(&turn.message, threshold);

            let (exact, recall, fp, missed) = evaluate_turn(&turn.ground_truth, &detected);

            records.push(TurnRecord {
                phase: phase_name.to_string(),
                pass_type: pass_type.to_string(),
                scenario: scenario.id.clone(),
                turn: turn_idx + 1,
                message: turn.message.clone(),
                word_count: turn.message.split_whitespace().count(),
                ground_truth: turn.ground_truth.clone(),
                detected_top5: detected.clone(),
                exact_match: exact,
                top5_recall: recall,
                false_positives: fp,
                missed,
                corrections_applied_so_far: mesh.corrections_applied,
                learning_curve_point: learning_milestones.contains(&global_turn),
            });

            // Learn AFTER evaluation (with probability = learn_rate)
            if learn_rate > 0.0 {
                let should_learn = if learn_rate >= 1.0 {
                    true
                } else {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let r = (rng_state >> 33) as f64 / (u32::MAX as f64);
                    r < learn_rate
                };
                if should_learn {
                    mesh.learn(&turn.message, &turn.ground_truth, &detected);
                }
            }
        }
    }

    // Rebuild paraphrase automaton after all learning in this phase
    if learn_rate > 0.0 {
        mesh.rebuild_paraphrase_automaton();
    }

    records
}

// ============= Main =============

fn main() {
    eprintln!("Loading data files...");

    // Load scenarios
    let scenario_data = std::fs::read_to_string("tests/scenarios/scenarios.json")
        .expect("Failed to read scenarios.json");
    let scenarios: Vec<Scenario> = serde_json::from_str(&scenario_data)
        .expect("Failed to parse scenarios.json");

    let new_scenario_data = std::fs::read_to_string("tests/scenarios/new_scenarios.json")
        .expect("Failed to read new_scenarios.json");
    let new_scenarios: Vec<Scenario> = serde_json::from_str(&new_scenario_data)
        .expect("Failed to parse new_scenarios.json");

    // Load index data
    let paraphrase_data: HashMap<String, Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string("tests/data/paraphrases.json")
            .expect("Failed to read paraphrases.json"))
        .expect("Failed to parse paraphrases.json");

    let correction_data: HashMap<String, String> =
        serde_json::from_str(&std::fs::read_to_string("tests/data/corrections.json")
            .expect("Failed to read corrections.json"))
        .expect("Failed to parse corrections.json");

    let modifier_entries: Vec<ModifierEntry> =
        serde_json::from_str(&std::fs::read_to_string("tests/data/modifiers.json")
            .expect("Failed to read modifiers.json"))
        .expect("Failed to parse modifiers.json");

    let exclusion_data: HashMap<String, Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string("tests/data/exclusions.json")
            .expect("Failed to read exclusions.json"))
        .expect("Failed to parse exclusions.json");

    let total_turns: usize = scenarios.iter().map(|s| s.turns.len()).sum();
    let new_turns: usize = new_scenarios.iter().map(|s| s.turns.len()).sum();

    eprintln!("Data loaded: {} scenarios ({} turns), {} new scenarios ({} turns)",
        scenarios.len(), total_turns, new_scenarios.len(), new_turns);
    eprintln!("Paraphrases: {} intents, {} total phrases",
        paraphrase_data.len(),
        paraphrase_data.values().map(|v| v.len()).sum::<usize>());
    eprintln!("Corrections: {} mappings", correction_data.len());
    eprintln!("Modifiers: {} patterns", modifier_entries.len());
    eprintln!("Exclusions: {} intents covered", exclusion_data.len());

    let threshold = 0.3;
    let mut report = String::new();
    let mut all_records: Vec<TurnRecord> = Vec::new();

    report.push_str("# ASV Knowledge Mesh — Experiment Results\n\n");
    report.push_str(&format!("Scenarios: {} ({} turns) + {} new ({} turns)\n\n",
        scenarios.len(), total_turns, new_scenarios.len(), new_turns));

    // ============================
    // PHASE 1: Baseline + Learning (Routing Index Only)
    // ============================
    eprintln!("\n=== PHASE 1: Routing Index Only ===");
    report.push_str("---\n\n## PHASE 1: Routing Index Only (with learning)\n");

    // First pass with learning
    eprintln!("  First pass (with learning)...");
    let router1 = setup_router();
    let mut mesh1 = MeshRouter::new(router1);
    let p1_first = run_phase(&mut mesh1, &scenarios, "1", "first_pass", true, threshold);
    let p1_first_stats = compute_phase_stats(&p1_first);
    write_phase_report(&mut report, "First Pass (with learning)", &p1_first_stats);
    all_records.extend(p1_first);

    // Second pass (no new learning)
    eprintln!("  Second pass (generalization)...");
    let p1_second = run_phase(&mut mesh1, &scenarios, "1", "second_pass", false, threshold);
    let p1_second_stats = compute_phase_stats(&p1_second);
    write_phase_report(&mut report, "Second Pass (generalization)", &p1_second_stats);
    all_records.extend(p1_second);

    // New scenarios
    eprintln!("  New scenarios (true generalization)...");
    let p1_new = run_phase(&mut mesh1, &new_scenarios, "1", "new_scenario", false, threshold);
    let p1_new_stats = compute_phase_stats(&p1_new);
    write_phase_report(&mut report, "New Scenarios (unseen)", &p1_new_stats);
    all_records.extend(p1_new);

    eprintln!("  Phase 1 corrections applied: {}", mesh1.corrections_applied);
    report.push_str(&format!("\n**Total corrections applied:** {}\n", mesh1.corrections_applied));

    // ============================
    // PHASE 2: Routing + Paraphrase Index
    // ============================
    eprintln!("\n=== PHASE 2: Routing + Paraphrase Index ===");
    report.push_str("\n---\n\n## PHASE 2: Routing + Paraphrase Index (with learning)\n");

    let router2 = setup_router();
    let pi = ParaphraseIndex::load_from_json(&paraphrase_data);
    let mut mesh2 = MeshRouter::new(router2).with_paraphrase(pi);

    // First pass with learning
    eprintln!("  First pass (with learning)...");
    let p2_first = run_phase(&mut mesh2, &scenarios, "2", "first_pass", true, threshold);
    let p2_first_stats = compute_phase_stats(&p2_first);
    write_phase_report(&mut report, "First Pass (with learning)", &p2_first_stats);
    all_records.extend(p2_first);

    // Second pass
    eprintln!("  Second pass (generalization)...");
    let p2_second = run_phase(&mut mesh2, &scenarios, "2", "second_pass", false, threshold);
    let p2_second_stats = compute_phase_stats(&p2_second);
    write_phase_report(&mut report, "Second Pass (generalization)", &p2_second_stats);
    all_records.extend(p2_second);

    // New scenarios
    eprintln!("  New scenarios (true generalization)...");
    let p2_new = run_phase(&mut mesh2, &new_scenarios, "2", "new_scenario", false, threshold);
    let p2_new_stats = compute_phase_stats(&p2_new);
    write_phase_report(&mut report, "New Scenarios (unseen)", &p2_new_stats);
    all_records.extend(p2_new);

    eprintln!("  Phase 2 corrections applied: {}", mesh2.corrections_applied);
    report.push_str(&format!("\n**Total corrections applied:** {}\n", mesh2.corrections_applied));

    // ============================
    // PHASE 3: Full Knowledge Mesh
    // ============================
    eprintln!("\n=== PHASE 3: Full Knowledge Mesh ===");
    report.push_str("\n---\n\n## PHASE 3: Full Knowledge Mesh (with learning)\n");
    report.push_str("\n**Pipeline:** Correction → Modifier → Paraphrase → Routing → Exclusion\n");

    let router3 = setup_router();
    let pi3 = ParaphraseIndex::load_from_json(&paraphrase_data);
    let ci3 = CorrectionIndex::load_from_json(&correction_data);
    let mi3 = ModifierIndex::load_from_json(&modifier_entries);
    let ei3 = ExclusionIndex::load_from_json(&exclusion_data);
    let mut mesh3 = MeshRouter::new(router3)
        .with_paraphrase(pi3)
        .with_correction(ci3)
        .with_modifier(mi3)
        .with_exclusion(ei3);

    // First pass with learning
    eprintln!("  First pass (with learning)...");
    let p3_first = run_phase(&mut mesh3, &scenarios, "3", "first_pass", true, threshold);
    let p3_first_stats = compute_phase_stats(&p3_first);
    write_phase_report(&mut report, "First Pass (with learning)", &p3_first_stats);
    all_records.extend(p3_first);

    // Second pass
    eprintln!("  Second pass (generalization)...");
    let p3_second = run_phase(&mut mesh3, &scenarios, "3", "second_pass", false, threshold);
    let p3_second_stats = compute_phase_stats(&p3_second);
    write_phase_report(&mut report, "Second Pass (generalization)", &p3_second_stats);
    all_records.extend(p3_second);

    // New scenarios
    eprintln!("  New scenarios (true generalization)...");
    let p3_new = run_phase(&mut mesh3, &new_scenarios, "3", "new_scenario", false, threshold);
    let p3_new_stats = compute_phase_stats(&p3_new);
    write_phase_report(&mut report, "New Scenarios (unseen)", &p3_new_stats);
    all_records.extend(p3_new);

    eprintln!("  Phase 3 corrections applied: {}", mesh3.corrections_applied);
    report.push_str(&format!("\n**Total corrections applied:** {}\n", mesh3.corrections_applied));

    // ============================
    // PHASE 3b: Softened Knowledge Mesh
    // ============================
    eprintln!("\n=== PHASE 3b: Softened Knowledge Mesh ===");
    report.push_str("\n---\n\n## PHASE 3b: Softened Mesh (modifier demotes, exclusion learned-only min_count=5)\n");

    let router3b = setup_router();
    let pi3b = ParaphraseIndex::load_from_json(&paraphrase_data);
    let ci3b = CorrectionIndex::load_from_json(&correction_data);
    let mi3b = ModifierIndex::load_from_json(&modifier_entries);
    let ei3b = ExclusionIndex::load_from_json(&exclusion_data);
    let mut mesh3b = MeshRouter::new(router3b)
        .with_paraphrase(pi3b)
        .with_correction(ci3b)
        .with_modifier(mi3b)
        .with_exclusion(ei3b)
        .with_soft_mode();

    eprintln!("  First pass (with learning)...");
    let p3b_first = run_phase(&mut mesh3b, &scenarios, "3b", "first_pass", true, threshold);
    let p3b_first_stats = compute_phase_stats(&p3b_first);
    write_phase_report(&mut report, "First Pass (with learning)", &p3b_first_stats);
    all_records.extend(p3b_first);

    eprintln!("  Second pass (generalization)...");
    let p3b_second = run_phase(&mut mesh3b, &scenarios, "3b", "second_pass", false, threshold);
    let p3b_second_stats = compute_phase_stats(&p3b_second);
    write_phase_report(&mut report, "Second Pass (generalization)", &p3b_second_stats);
    all_records.extend(p3b_second);

    eprintln!("  New scenarios...");
    let p3b_new = run_phase(&mut mesh3b, &new_scenarios, "3b", "new_scenario", false, threshold);
    let p3b_new_stats = compute_phase_stats(&p3b_new);
    write_phase_report(&mut report, "New Scenarios (unseen)", &p3b_new_stats);
    all_records.extend(p3b_new);

    eprintln!("  Phase 3b corrections applied: {}", mesh3b.corrections_applied);
    report.push_str(&format!("\n**Total corrections applied:** {}\n", mesh3b.corrections_applied));

    // ============================
    // PHASE 4: Partial Learning (30% correction rate) — Phase 2 architecture
    // ============================
    eprintln!("\n=== PHASE 4: Partial Learning (30% correction rate) ===");
    report.push_str("\n---\n\n## PHASE 4: Partial Learning — 30% correction rate (Routing + Paraphrase)\n");
    report.push_str("\n**Simulates production: only 1 in 3 turns gets human review/correction.**\n");

    let router4 = setup_router();
    let pi4 = ParaphraseIndex::load_from_json(&paraphrase_data);
    let mut mesh4 = MeshRouter::new(router4).with_paraphrase(pi4);

    eprintln!("  First pass (30% learning)...");
    let p4_first = run_phase_with_rate(&mut mesh4, &scenarios, "4", "first_pass", 0.3, threshold);
    let p4_first_stats = compute_phase_stats(&p4_first);
    write_phase_report(&mut report, "First Pass (30% learning rate)", &p4_first_stats);
    all_records.extend(p4_first);

    eprintln!("  Second pass (generalization)...");
    let p4_second = run_phase(&mut mesh4, &scenarios, "4", "second_pass", false, threshold);
    let p4_second_stats = compute_phase_stats(&p4_second);
    write_phase_report(&mut report, "Second Pass (generalization)", &p4_second_stats);
    all_records.extend(p4_second);

    eprintln!("  New scenarios...");
    let p4_new = run_phase(&mut mesh4, &new_scenarios, "4", "new_scenario", false, threshold);
    let p4_new_stats = compute_phase_stats(&p4_new);
    write_phase_report(&mut report, "New Scenarios (unseen)", &p4_new_stats);
    all_records.extend(p4_new);

    eprintln!("  Phase 4 corrections applied: {}", mesh4.corrections_applied);
    report.push_str(&format!("\n**Total corrections applied:** {}\n", mesh4.corrections_applied));

    // ============================
    // PHASE 5: Partial Learning (10% correction rate) — Phase 2 architecture
    // ============================
    eprintln!("\n=== PHASE 5: Minimal Learning (10% correction rate) ===");
    report.push_str("\n---\n\n## PHASE 5: Minimal Learning — 10% correction rate (Routing + Paraphrase)\n");
    report.push_str("\n**Simulates low-supervision: only 1 in 10 turns gets corrected.**\n");

    let router5 = setup_router();
    let pi5 = ParaphraseIndex::load_from_json(&paraphrase_data);
    let mut mesh5 = MeshRouter::new(router5).with_paraphrase(pi5);

    eprintln!("  First pass (10% learning)...");
    let p5_first = run_phase_with_rate(&mut mesh5, &scenarios, "5", "first_pass", 0.1, threshold);
    let p5_first_stats = compute_phase_stats(&p5_first);
    write_phase_report(&mut report, "First Pass (10% learning rate)", &p5_first_stats);
    all_records.extend(p5_first);

    eprintln!("  Second pass (generalization)...");
    let p5_second = run_phase(&mut mesh5, &scenarios, "5", "second_pass", false, threshold);
    let p5_second_stats = compute_phase_stats(&p5_second);
    write_phase_report(&mut report, "Second Pass (generalization)", &p5_second_stats);
    all_records.extend(p5_second);

    eprintln!("  Phase 5 corrections applied: {}", mesh5.corrections_applied);
    report.push_str(&format!("\n**Total corrections applied:** {}\n", mesh5.corrections_applied));

    // ============================
    // COMPARISON SUMMARY
    // ============================
    report.push_str("\n---\n\n## COMPARISON SUMMARY\n\n");
    report.push_str("### First Pass (30 scenarios, with learning)\n\n");
    report.push_str("| Phase | Exact% | Top-5 Recall% | Avg FP/turn | Corrections |\n");
    report.push_str("|---|---|---|---|---|\n");

    for (name, stats, corrections) in &[
        ("1: Routing Only", &p1_first_stats, mesh1.corrections_applied),
        ("2: +Paraphrase", &p2_first_stats, mesh2.corrections_applied),
        ("3: Full Mesh", &p3_first_stats, mesh3.corrections_applied),
        ("3b: Soft Mesh", &p3b_first_stats, mesh3b.corrections_applied),
        ("4: 30% Learning", &p4_first_stats, mesh4.corrections_applied),
        ("5: 10% Learning", &p5_first_stats, mesh5.corrections_applied),
    ] {
        let exact = (stats.exact_pass as f64 / stats.total.max(1) as f64) * 100.0;
        let recall = (stats.top5_recall as f64 / stats.total.max(1) as f64) * 100.0;
        let fp = stats.total_fp as f64 / stats.total.max(1) as f64;
        report.push_str(&format!("| {} | {:.1}% | {:.1}% | {:.2} | {} |\n",
            name, exact, recall, fp, corrections));
    }

    report.push_str("\n### Second Pass (generalization on same 30 scenarios)\n\n");
    report.push_str("| Phase | Exact% | Top-5 Recall% | Avg FP/turn |\n");
    report.push_str("|---|---|---|---|\n");

    for (name, stats) in &[
        ("1: Routing Only", &p1_second_stats),
        ("2: +Paraphrase", &p2_second_stats),
        ("3: Full Mesh", &p3_second_stats),
        ("3b: Soft Mesh", &p3b_second_stats),
        ("4: 30% Learning", &p4_second_stats),
        ("5: 10% Learning", &p5_second_stats),
    ] {
        let exact = (stats.exact_pass as f64 / stats.total.max(1) as f64) * 100.0;
        let recall = (stats.top5_recall as f64 / stats.total.max(1) as f64) * 100.0;
        let fp = stats.total_fp as f64 / stats.total.max(1) as f64;
        report.push_str(&format!("| {} | {:.1}% | {:.1}% | {:.2} |\n",
            name, exact, recall, fp));
    }

    report.push_str("\n### New Scenarios (10 unseen, true generalization)\n\n");
    report.push_str("| Phase | Exact% | Top-5 Recall% | Avg FP/turn |\n");
    report.push_str("|---|---|---|---|\n");

    for (name, stats) in &[
        ("1: Routing Only", &p1_new_stats),
        ("2: +Paraphrase", &p2_new_stats),
        ("3: Full Mesh", &p3_new_stats),
        ("3b: Soft Mesh", &p3b_new_stats),
        ("4: 30% Learning", &p4_new_stats),
    ] {
        let exact = (stats.exact_pass as f64 / stats.total.max(1) as f64) * 100.0;
        let recall = (stats.top5_recall as f64 / stats.total.max(1) as f64) * 100.0;
        let fp = stats.total_fp as f64 / stats.total.max(1) as f64;
        report.push_str(&format!("| {} | {:.1}% | {:.1}% | {:.2} |\n",
            name, exact, recall, fp));
    }

    // Learning curves comparison — granular
    report.push_str("\n### Learning Curves — Second Pass (Generalization)\n\n");
    report.push_str("| Turn | P1 Exact% | P2 Exact% | P3b Exact% | P4(30%) Exact% | P5(10%) Exact% |\n");
    report.push_str("|---|---|---|---|---|---|\n");
    let curve_milestones = [1, 5, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 138];
    for &m in &curve_milestones {
        let get_exact = |stats: &PhaseStats| -> Option<f32> {
            stats.learning_curve.iter().find(|(t, _, _)| *t == m).map(|(_, e, _)| *e)
        };
        if let (Some(p1e), Some(p2e), Some(p3be), Some(p4e), Some(p5e)) = (
            get_exact(&p1_second_stats),
            get_exact(&p2_second_stats),
            get_exact(&p3b_second_stats),
            get_exact(&p4_second_stats),
            get_exact(&p5_second_stats),
        ) {
            report.push_str(&format!("| {} | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% |\n",
                m, p1e, p2e, p3be, p4e, p5e));
        }
    }

    // Minimum supervision analysis
    report.push_str("\n### Minimum Supervision Required\n\n");
    report.push_str("What accuracy do you get with less human review?\n\n");
    report.push_str("| Supervision Rate | Corrections | 2nd Pass Exact% | 2nd Pass Recall% |\n");
    report.push_str("|---|---|---|---|\n");
    for (rate, stats, corr) in &[
        ("100%", &p2_second_stats, mesh2.corrections_applied),
        ("30%", &p4_second_stats, mesh4.corrections_applied),
        ("10%", &p5_second_stats, mesh5.corrections_applied),
    ] {
        let exact = (stats.exact_pass as f64 / stats.total.max(1) as f64) * 100.0;
        let recall = (stats.top5_recall as f64 / stats.total.max(1) as f64) * 100.0;
        report.push_str(&format!("| {} | {} | {:.1}% | {:.1}% |\n", rate, corr, exact, recall));
    }

    // Break-even analysis (second pass)
    report.push_str("\n### Break-Even Analysis (Second Pass)\n\n");
    report.push_str("At what turn in the second pass does cumulative accuracy cross thresholds?\n\n");
    report.push_str("| Target | P1 | P2 | P3b | P4(30%) | P5(10%) |\n");
    report.push_str("|---|---|---|---|---|---|\n");

    let find_breakeven = |records: &[TurnRecord], target: f64| -> String {
        let mut cumulative_exact = 0;
        for (i, r) in records.iter().enumerate() {
            if r.exact_match { cumulative_exact += 1; }
            let rate = (cumulative_exact as f64 / (i + 1) as f64) * 100.0;
            if rate >= target {
                return format!("Turn {}", i + 1);
            }
        }
        "Never".to_string()
    };

    for target in [30.0, 50.0, 60.0, 70.0] {
        let collect_second = |phase: &str| -> Vec<TurnRecord> {
            all_records.iter()
                .filter(|r| r.phase == phase && r.pass_type == "second_pass")
                .cloned().collect()
        };

        let p1r = collect_second("1");
        let p2r = collect_second("2");
        let p3br = collect_second("3b");
        let p4r = collect_second("4");
        let p5r = collect_second("5");

        report.push_str(&format!("| {:.0}% | {} | {} | {} | {} | {} |\n",
            target,
            find_breakeven(&p1r, target),
            find_breakeven(&p2r, target),
            find_breakeven(&p3br, target),
            find_breakeven(&p4r, target),
            find_breakeven(&p5r, target)));
    }

    // Index contribution analysis
    report.push_str("\n### Index Contribution Analysis\n\n");

    for (label, phase_id) in &[("Phase 3 (hard)", "3"), ("Phase 3b (soft)", "3b")] {
        let phase_records: Vec<&TurnRecord> = all_records.iter()
            .filter(|r| r.phase == *phase_id && r.pass_type == "first_pass")
            .collect();

        let mut paraphrase_only = 0;
        let mut routing_only = 0;
        let mut both_sources = 0;
        for r in &phase_records {
            for d in &r.detected_top5 {
                if r.ground_truth.contains(&d.id) {
                    match d.source.as_str() {
                        "paraphrase" => paraphrase_only += 1,
                        "routing" => routing_only += 1,
                        "both" => both_sources += 1,
                        _ => {}
                    }
                }
            }
        }
        report.push_str(&format!("\n**{}** correct detections by source:\n", label));
        report.push_str(&format!("- Routing only: {}\n", routing_only));
        report.push_str(&format!("- Paraphrase only: {}\n", paraphrase_only));
        report.push_str(&format!("- Both indexes: {}\n", both_sources));
    }

    // Save report
    std::fs::write("mesh_experiment_results.md", &report)
        .expect("Failed to write results");
    eprintln!("\nResults saved to mesh_experiment_results.md");

    // Save detailed turn data
    let json_output = serde_json::to_string_pretty(&all_records)
        .expect("Failed to serialize turn records");
    std::fs::write("mesh_experiment_turns.json", &json_output)
        .expect("Failed to write turn data");
    eprintln!("Detailed turn data saved to mesh_experiment_turns.json");

    // Print summary
    eprintln!("\n=== SUMMARY ===");
    eprintln!("Phase 1 (Routing Only):  Exact={:.1}% → {:.1}% (gen) | Recall={:.1}% → {:.1}%",
        (p1_first_stats.exact_pass as f64 / p1_first_stats.total.max(1) as f64) * 100.0,
        (p1_second_stats.exact_pass as f64 / p1_second_stats.total.max(1) as f64) * 100.0,
        (p1_first_stats.top5_recall as f64 / p1_first_stats.total.max(1) as f64) * 100.0,
        (p1_second_stats.top5_recall as f64 / p1_second_stats.total.max(1) as f64) * 100.0);
    eprintln!("Phase 2 (+Paraphrase):   Exact={:.1}% → {:.1}% (gen) | Recall={:.1}% → {:.1}%",
        (p2_first_stats.exact_pass as f64 / p2_first_stats.total.max(1) as f64) * 100.0,
        (p2_second_stats.exact_pass as f64 / p2_second_stats.total.max(1) as f64) * 100.0,
        (p2_first_stats.top5_recall as f64 / p2_first_stats.total.max(1) as f64) * 100.0,
        (p2_second_stats.top5_recall as f64 / p2_second_stats.total.max(1) as f64) * 100.0);
    eprintln!("Phase 3 (Full Mesh):     Exact={:.1}% → {:.1}% (gen) | Recall={:.1}% → {:.1}%",
        (p3_first_stats.exact_pass as f64 / p3_first_stats.total.max(1) as f64) * 100.0,
        (p3_second_stats.exact_pass as f64 / p3_second_stats.total.max(1) as f64) * 100.0,
        (p3_first_stats.top5_recall as f64 / p3_first_stats.total.max(1) as f64) * 100.0,
        (p3_second_stats.top5_recall as f64 / p3_second_stats.total.max(1) as f64) * 100.0);
}
