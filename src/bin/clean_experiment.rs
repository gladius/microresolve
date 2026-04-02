//! Clean Pipeline Experiments: Phase 2 cleanup + Boundary Pattern Index.
//!
//! Implements the cleaned-up ASV routing pipeline:
//! - Modifier Index: metadata-only (flags negation, doesn't affect scores)
//! - Exclusion Index: data-collection-only (records false pairs, doesn't filter)
//! - Boundary Pattern Index: Aho-Corasick segmentation for verbose queries
//!
//! Experiments:
//! A: Phase 2 + Correction Index (isolated test)
//! B: Phase 2 + Boundary Segmentation
//! C: Phase 2 + Correction + Boundary (full clean pipeline)
//! D: Dual-source confidence signal (analysis of existing data)
//! E: Cumulative learning (3 passes)
//! F: Score ratio analysis (analysis of existing data)
//!
//! Run with: cargo run --bin clean_experiment --release

use asv_router::{IntentType, Router, MultiRouteOutput};
use std::collections::{HashMap, HashSet};

// ============= Data Structures =============

#[derive(serde::Deserialize, Clone)]
struct Scenario {
    id: String,
    #[allow(dead_code)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    #[serde(default)]
    segments_used: usize,
    #[serde(default)]
    negation_flags: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DetectedIntent {
    id: String,
    score: f32,
    source: String,
}

// ============= Paraphrase Index =============

struct ParaphraseIndex {
    phrases: HashMap<String, (String, f32)>,
    automaton: Option<aho_corasick::AhoCorasick>,
    patterns: Vec<String>,
}

impl ParaphraseIndex {
    fn load_from_json(data: &HashMap<String, Vec<String>>) -> Self {
        let mut idx = Self {
            phrases: HashMap::new(),
            automaton: None,
            patterns: Vec::new(),
        };
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

// ============= Modifier Index (METADATA ONLY) =============
// Detects negation/urgency patterns and outputs flags.
// Does NOT affect routing scores.

struct ModifierIndex {
    patterns: Vec<ModifierPattern>,
}

#[derive(Clone)]
struct ModifierPattern {
    words: Vec<String>,
    effect: String,
    scope: usize,
}

#[derive(serde::Deserialize)]
struct ModifierEntry {
    pattern: String,
    effect: String,
    #[serde(default = "default_scope")]
    scope: usize,
}

fn default_scope() -> usize { 2 }

impl ModifierIndex {
    fn load_from_json(entries: &[ModifierEntry]) -> Self {
        let patterns = entries.iter().map(|e| {
            ModifierPattern {
                words: e.pattern.to_lowercase().split_whitespace().map(String::from).collect(),
                effect: e.effect.clone(),
                scope: e.scope,
            }
        }).collect();
        Self { patterns }
    }

    /// Detect modifier patterns and return metadata flags.
    /// Returns: Vec of (effect, nearby_word) pairs as metadata.
    fn detect_flags(&self, message: &str) -> Vec<String> {
        let words: Vec<String> = message.to_lowercase()
            .split_whitespace()
            .map(|w| w.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect())
            .collect();

        let mut flags: Vec<String> = Vec::new();

        for pattern in &self.patterns {
            let pat_len = pattern.words.len();
            if words.len() < pat_len { continue; }

            for i in 0..=(words.len() - pat_len) {
                let matches = pattern.words.iter()
                    .zip(&words[i..i + pat_len])
                    .all(|(p, w)| p == w);

                if matches {
                    let start = i + pat_len;
                    let nearby: Vec<&str> = words[start..words.len().min(start + pattern.scope)]
                        .iter().map(|s| s.as_str()).collect();
                    flags.push(format!("{} detected near: {}",
                        pattern.effect, nearby.join(" ")));
                }
            }
        }
        flags
    }
}

// ============= Exclusion Index (DATA COLLECTION ONLY) =============
// Records which intent pairs are corrected as false co-occurrences.
// Does NOT filter or affect results.

struct ExclusionCollector {
    /// (intent_a, intent_b) -> false_pair_count (sorted key order)
    learned: HashMap<(String, String), u32>,
}

impl ExclusionCollector {
    fn new() -> Self {
        Self { learned: HashMap::new() }
    }

    fn record_false_pair(&mut self, intent_a: &str, intent_b: &str) {
        let key = if intent_a < intent_b {
            (intent_a.to_string(), intent_b.to_string())
        } else {
            (intent_b.to_string(), intent_a.to_string())
        };
        *self.learned.entry(key).or_insert(0) += 1;
    }

    #[allow(dead_code)]
    fn get_count(&self, intent_a: &str, intent_b: &str) -> u32 {
        let key = if intent_a < intent_b {
            (intent_a.to_string(), intent_b.to_string())
        } else {
            (intent_b.to_string(), intent_a.to_string())
        };
        self.learned.get(&key).copied().unwrap_or(0)
    }
}

// ============= Boundary Pattern Index =============

struct BoundaryPatternIndex {
    automaton: Option<aho_corasick::AhoCorasick>,
    patterns: Vec<String>,
    min_prefix_chars: usize,
}

impl BoundaryPatternIndex {
    fn load_from_json(data: &HashMap<String, Vec<String>>) -> Self {
        let mut all_patterns: Vec<String> = Vec::new();
        for (_category, patterns) in data {
            for p in patterns {
                all_patterns.push(p.to_lowercase());
            }
        }
        // Deduplicate
        all_patterns.sort();
        all_patterns.dedup();

        let mut idx = Self {
            automaton: None,
            patterns: all_patterns,
            min_prefix_chars: 15,
        };
        idx.rebuild_automaton();
        idx
    }

    fn rebuild_automaton(&mut self) {
        if self.patterns.is_empty() {
            self.automaton = None;
            return;
        }
        // Sort by length descending for leftmost-longest matching
        self.patterns.sort_by(|a, b| b.len().cmp(&a.len()));
        self.automaton = aho_corasick::AhoCorasick::builder()
            .match_kind(aho_corasick::MatchKind::LeftmostLongest)
            .build(&self.patterns)
            .ok();
    }

    /// Segment a message into clauses based on boundary patterns.
    /// Returns segments. If no boundaries found, returns the whole message as one segment.
    fn segment(&self, message: &str) -> Vec<String> {
        let lower = message.to_lowercase();

        // Find all boundary positions
        let mut boundary_positions: Vec<usize> = Vec::new();

        if let Some(ref ac) = self.automaton {
            for mat in ac.find_iter(&lower) {
                let pos = mat.start();
                // Only insert boundary if enough text precedes it
                if pos >= self.min_prefix_chars {
                    boundary_positions.push(pos);
                }
            }
        }

        if boundary_positions.is_empty() {
            return vec![message.to_string()];
        }

        // Deduplicate and sort positions
        boundary_positions.sort();
        boundary_positions.dedup();

        // Split at boundary positions (using original case message)
        let mut segments: Vec<String> = Vec::new();
        let mut last_pos = 0;

        for &pos in &boundary_positions {
            if pos > last_pos {
                let seg = message[last_pos..pos].trim().to_string();
                if !seg.is_empty() {
                    segments.push(seg);
                }
            }
            last_pos = pos;
        }
        // Final segment
        if last_pos < message.len() {
            let seg = message[last_pos..].trim().to_string();
            if !seg.is_empty() {
                segments.push(seg);
            }
        }

        // Merge segments that are too short (< 3 words) with adjacent
        self.merge_short_segments(segments)
    }

    fn merge_short_segments(&self, segments: Vec<String>) -> Vec<String> {
        if segments.len() <= 1 {
            return segments;
        }

        let mut result: Vec<String> = Vec::new();
        let mut carry: Option<String> = None;

        for seg in segments {
            if let Some(prev) = carry.take() {
                // Merge carried short segment with current
                result.push(format!("{} {}", prev, seg));
            } else if seg.split_whitespace().count() < 3 {
                // Too short — carry forward to merge with next
                carry = Some(seg);
            } else {
                result.push(seg);
            }
        }

        // If last segment was carried, merge with previous
        if let Some(leftover) = carry {
            if let Some(last) = result.last_mut() {
                *last = format!("{} {}", last, leftover);
            } else {
                result.push(leftover);
            }
        }

        if result.is_empty() {
            // Shouldn't happen, but safety fallback
            result.push(String::new());
        }

        result
    }

    fn add_pattern(&mut self, pattern: &str) {
        let lower = pattern.to_lowercase();
        if lower.split_whitespace().count() >= 2 && !self.patterns.contains(&lower) {
            self.patterns.push(lower);
        }
    }

    /// Learn boundary patterns from a multi-intent correction.
    /// When GT has 2+ intents and the system missed at least one in an unsegmented message,
    /// try to identify boundary words from route_multi spans.
    fn learn_from_correction(
        &mut self,
        message: &str,
        routing_output: &MultiRouteOutput,
        _ground_truth: &[String],
    ) {
        // Use route_multi spans to find gap regions between detected intents
        if routing_output.intents.len() >= 2 {
            let mut sorted_intents = routing_output.intents.clone();
            sorted_intents.sort_by_key(|i| i.position);

            for window in sorted_intents.windows(2) {
                let gap_start = window[0].span.1;
                let gap_end = window[1].span.0;

                if gap_end > gap_start {
                    let gap_text = &message[gap_start.min(message.len())..gap_end.min(message.len())];
                    let gap_words: Vec<&str> = gap_text.split_whitespace().collect();

                    // Add 2-word windows from the gap as potential boundary patterns
                    if gap_words.len() >= 2 {
                        for window in gap_words.windows(2) {
                            let pattern = window.join(" ");
                            if pattern.len() >= 3 {
                                self.add_pattern(&pattern);
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============= Clean Router =============

struct CleanRouter {
    router: Router,
    paraphrase_index: ParaphraseIndex,
    correction_index: Option<CorrectionIndex>,
    boundary_index: Option<BoundaryPatternIndex>,
    modifier_index: Option<ModifierIndex>,
    exclusion_collector: ExclusionCollector,
    corrections_applied: usize,
}

impl CleanRouter {
    fn new(router: Router, paraphrase: ParaphraseIndex) -> Self {
        Self {
            router,
            paraphrase_index: paraphrase,
            correction_index: None,
            boundary_index: None,
            modifier_index: None,
            exclusion_collector: ExclusionCollector::new(),
            corrections_applied: 0,
        }
    }

    fn with_correction(mut self, ci: CorrectionIndex) -> Self {
        self.correction_index = Some(ci);
        self
    }

    fn with_boundary(mut self, bi: BoundaryPatternIndex) -> Self {
        self.boundary_index = Some(bi);
        self
    }

    fn with_modifier(mut self, mi: ModifierIndex) -> Self {
        self.modifier_index = Some(mi);
        self
    }

    /// Route a message through the clean pipeline.
    /// Returns (detected_intents, segments_used, negation_flags).
    fn route(&self, message: &str, threshold: f32) -> (Vec<DetectedIntent>, usize, Vec<String>) {
        // Step 1: Correction normalization (optional)
        let normalized = if let Some(ref ci) = self.correction_index {
            ci.apply(message)
        } else {
            message.to_string()
        };

        // Step 2: Modifier metadata (optional, does NOT affect scores)
        let negation_flags = if let Some(ref mi) = self.modifier_index {
            mi.detect_flags(&normalized)
        } else {
            vec![]
        };

        // Step 3: Segment or route whole
        let (detected, segments_used) = if let Some(ref bi) = self.boundary_index {
            let segments = bi.segment(&normalized);
            let seg_count = segments.len();

            if seg_count <= 1 {
                // No segmentation needed — route normally
                (self.route_single(&normalized, message, threshold), 1)
            } else {
                // Route each segment, merge results
                (self.route_segmented(&segments, message, threshold), seg_count)
            }
        } else {
            (self.route_single(&normalized, message, threshold), 1)
        };

        (detected, segments_used, negation_flags)
    }

    /// Route a single message (no segmentation).
    fn route_single(&self, normalized: &str, original: &str, threshold: f32) -> Vec<DetectedIntent> {
        let routing_output = self.router.route_multi(normalized, threshold);

        // Scan paraphrase on original message
        let paraphrase_hits = self.paraphrase_index.scan(original);

        // Merge routing + paraphrase
        self.merge_results(&routing_output, &paraphrase_hits)
    }

    /// Route segmented message: each segment through paraphrase + routing, then merge.
    fn route_segmented(&self, segments: &[String], original: &str, threshold: f32) -> Vec<DetectedIntent> {
        let mut merged: HashMap<String, (f32, String)> = HashMap::new();

        for segment in segments {
            // Route segment through routing index
            let routing_output = self.router.route_multi(segment, threshold);

            for intent in &routing_output.intents {
                let entry = merged.entry(intent.id.clone())
                    .or_insert((0.0, "routing".to_string()));
                if intent.score > entry.0 {
                    entry.0 = intent.score;
                }
            }

            // Scan segment through paraphrase
            for (intent_id, weight, _pos) in self.paraphrase_index.scan(segment) {
                let boosted = weight * 3.0;
                let entry = merged.entry(intent_id)
                    .or_insert((0.0, "paraphrase".to_string()));
                if entry.1 == "routing" && boosted > 0.0 {
                    entry.1 = "both".to_string();
                }
                entry.0 += boosted;
            }
        }

        // Also scan original message for cross-segment paraphrase matches
        for (intent_id, weight, _pos) in self.paraphrase_index.scan(original) {
            let boosted = weight * 3.0;
            let entry = merged.entry(intent_id)
                .or_insert((0.0, "paraphrase".to_string()));
            if entry.1 == "routing" {
                entry.1 = "both".to_string();
            }
            // Only add if not already higher
            if boosted > entry.0 {
                entry.0 = boosted;
            }
        }

        // Sort by score, take top 5
        let mut results: Vec<DetectedIntent> = merged.into_iter().map(|(id, (score, source))| {
            DetectedIntent {
                id,
                score: (score * 100.0).round() / 100.0,
                source,
            }
        }).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(5);
        results
    }

    /// Merge routing output with paraphrase hits into DetectedIntent list.
    fn merge_results(
        &self,
        routing_output: &MultiRouteOutput,
        paraphrase_hits: &[(String, f32, usize)],
    ) -> Vec<DetectedIntent> {
        let mut merged: HashMap<String, (f32, String)> = HashMap::new();

        for intent in &routing_output.intents {
            merged.insert(intent.id.clone(), (intent.score, "routing".to_string()));
        }

        for (intent_id, weight, _pos) in paraphrase_hits {
            let entry = merged.entry(intent_id.clone())
                .or_insert((0.0, "paraphrase".to_string()));
            entry.0 += weight * 3.0;
            if entry.1 == "routing" {
                entry.1 = "both".to_string();
            }
        }

        let mut results: Vec<DetectedIntent> = merged.into_iter().map(|(id, (score, source))| {
            DetectedIntent {
                id,
                score: (score * 100.0).round() / 100.0,
                source,
            }
        }).collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(5);
        results
    }

    /// Learn from ground truth.
    fn learn(&mut self, message: &str, ground_truth: &[String], detected: &[DetectedIntent]) {
        let gt_set: HashSet<&str> = ground_truth.iter().map(|s| s.as_str()).collect();
        let det_set: HashSet<&str> = detected.iter().map(|d| d.id.as_str()).collect();

        // Learn missed intents
        for gt_intent in &gt_set {
            if !det_set.contains(gt_intent) {
                self.router.learn(message, gt_intent);
                self.paraphrase_index.learn_from_message(message, gt_intent);
                self.corrections_applied += 1;
            }
        }

        // Correct false positives
        for det_intent in &det_set {
            if !gt_set.contains(det_intent) {
                if let Some(correct) = gt_set.iter().next() {
                    self.router.correct(message, det_intent, correct);
                    self.corrections_applied += 1;
                }
                // Record exclusion data (collection only, no filtering)
                for gt in &gt_set {
                    self.exclusion_collector.record_false_pair(det_intent, gt);
                }
            }
        }

        // Reinforce correct detections
        for gt_intent in gt_set.intersection(&det_set) {
            self.router.learn(message, gt_intent);
            self.paraphrase_index.learn_from_message(message, gt_intent);
        }

        // Boundary learning: if GT has 2+ intents and we missed at least one
        if ground_truth.len() >= 2 && !gt_set.iter().all(|g| det_set.contains(g)) {
            if let Some(ref mut bi) = self.boundary_index {
                let routing_output = self.router.route_multi(message, 0.3);
                bi.learn_from_correction(message, &routing_output, ground_truth);
            }
        }
    }

    fn rebuild_indexes(&mut self) {
        self.paraphrase_index.rebuild_automaton();
        if let Some(ref mut bi) = self.boundary_index {
            bi.rebuild_automaton();
        }
    }
}

// ============= Evaluation =============

fn evaluate_turn(
    ground_truth: &[String],
    detected: &[DetectedIntent],
) -> (bool, bool, usize, Vec<String>) {
    let gt_set: HashSet<&str> = ground_truth.iter().map(|s| s.as_str()).collect();
    let det_set: HashSet<&str> = detected.iter().map(|d| d.id.as_str()).collect();

    let exact = gt_set == det_set;
    let top5_recall = gt_set.iter().all(|gt| det_set.contains(gt));
    let fp_count = det_set.difference(&gt_set).count();
    let missed: Vec<String> = gt_set.difference(&det_set).map(|s| s.to_string()).collect();

    (exact, top5_recall, fp_count, missed)
}

// ============= Stats =============

struct PhaseStats {
    total: usize,
    exact_pass: usize,
    top5_recall: usize,
    total_fp: usize,
    by_wordcount: HashMap<String, (usize, usize, usize)>,
}

fn compute_stats(records: &[TurnRecord]) -> PhaseStats {
    let total = records.len();
    let exact_pass = records.iter().filter(|r| r.exact_match).count();
    let top5_recall = records.iter().filter(|r| r.top5_recall).count();
    let total_fp: usize = records.iter().map(|r| r.false_positives).sum();

    let mut by_wc: HashMap<String, (usize, usize, usize)> = HashMap::new();
    for r in records {
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

    PhaseStats { total, exact_pass, top5_recall, total_fp, by_wordcount: by_wc }
}

fn write_experiment_report(out: &mut String, name: &str, pipeline: &str, stats: &PhaseStats,
    new_stats: Option<&PhaseStats>) {
    out.push_str(&format!("\n## {}\n", name));
    out.push_str(&format!("**Pipeline:** {}\n\n", pipeline));

    let exact_pct = (stats.exact_pass as f64 / stats.total.max(1) as f64) * 100.0;
    let recall_pct = (stats.top5_recall as f64 / stats.total.max(1) as f64) * 100.0;
    let fp_avg = stats.total_fp as f64 / stats.total.max(1) as f64;

    out.push_str("**Second Pass Results:**\n\n");
    out.push_str(&format!("- Exact match: {:.1}%\n", exact_pct));
    out.push_str(&format!("- Top-5 recall: {:.1}%\n", recall_pct));
    out.push_str(&format!("- Avg FP/turn: {:.2}\n\n", fp_avg));

    out.push_str("**By word count:**\n\n");
    out.push_str("| Words | Total | Exact% | Recall% |\n|---|---|---|---|\n");
    for bucket in &["1-5", "6-10", "11-20", "21-40", "41+"] {
        if let Some((t, e, r)) = stats.by_wordcount.get(*bucket) {
            let ep = if *t > 0 { (*e as f64 / *t as f64) * 100.0 } else { 0.0 };
            let rp = if *t > 0 { (*r as f64 / *t as f64) * 100.0 } else { 0.0 };
            out.push_str(&format!("| {} | {} | {:.1}% | {:.1}% |\n", bucket, t, ep, rp));
        }
    }

    if let Some(ns) = new_stats {
        let ne = (ns.exact_pass as f64 / ns.total.max(1) as f64) * 100.0;
        let nr = (ns.top5_recall as f64 / ns.total.max(1) as f64) * 100.0;
        out.push_str(&format!("\n**New scenarios:** {:.1}% exact, {:.1}% recall\n", ne, nr));
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

fn run_experiment_phase(
    mesh: &mut CleanRouter,
    scenarios: &[Scenario],
    experiment: &str,
    pass_type: &str,
    do_learn: bool,
    threshold: f32,
) -> Vec<TurnRecord> {
    let mut records = Vec::new();

    for scenario in scenarios {
        for (turn_idx, turn) in scenario.turns.iter().enumerate() {
            let (detected, segments_used, negation_flags) = mesh.route(&turn.message, threshold);
            let (exact, recall, fp, missed) = evaluate_turn(&turn.ground_truth, &detected);

            records.push(TurnRecord {
                phase: experiment.to_string(),
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
                segments_used,
                negation_flags,
            });

            if do_learn {
                mesh.learn(&turn.message, &turn.ground_truth, &detected);
            }
        }
    }

    if do_learn {
        mesh.rebuild_indexes();
    }

    records
}

// ============= Analysis Experiments (D and F) =============

fn run_experiment_d(report: &mut String) {
    eprintln!("\n=== Experiment D: Dual-Source Confidence Signal ===");
    report.push_str("\n---\n\n## Experiment D: Dual-Source Confidence Signal\n");
    report.push_str("**Analysis of existing Phase 2 data from mesh_experiment_turns.json**\n\n");

    let data = match std::fs::read_to_string("mesh_experiment_turns.json") {
        Ok(d) => d,
        Err(e) => {
            report.push_str(&format!("ERROR: Could not load mesh_experiment_turns.json: {}\n", e));
            report.push_str("Run mesh_experiment first to generate the data.\n");
            eprintln!("  ERROR: {}", e);
            return;
        }
    };

    let records: Vec<TurnRecord> = match serde_json::from_str(&data) {
        Ok(r) => r,
        Err(e) => {
            report.push_str(&format!("ERROR: Could not parse data: {}\n", e));
            eprintln!("  ERROR: {}", e);
            return;
        }
    };

    // Filter Phase 2 second_pass records
    let phase2_records: Vec<&TurnRecord> = records.iter()
        .filter(|r| r.phase == "2" && r.pass_type == "second_pass")
        .collect();

    if phase2_records.is_empty() {
        report.push_str("No Phase 2 second_pass records found.\n");
        return;
    }

    let mut dual_correct = 0u32;
    let mut dual_total = 0u32;
    let mut routing_only_correct = 0u32;
    let mut routing_only_total = 0u32;
    let mut paraphrase_only_correct = 0u32;
    let mut paraphrase_only_total = 0u32;

    for record in &phase2_records {
        let gt_set: HashSet<&str> = record.ground_truth.iter().map(|s| s.as_str()).collect();

        for det in &record.detected_top5 {
            let is_correct = gt_set.contains(det.id.as_str());

            match det.source.as_str() {
                "both" => {
                    dual_total += 1;
                    if is_correct { dual_correct += 1; }
                },
                "routing" => {
                    routing_only_total += 1;
                    if is_correct { routing_only_correct += 1; }
                },
                "paraphrase" => {
                    paraphrase_only_total += 1;
                    if is_correct { paraphrase_only_correct += 1; }
                },
                _ => {}
            }
        }
    }

    let dual_tpr = if dual_total > 0 { dual_correct as f64 / dual_total as f64 * 100.0 } else { 0.0 };
    let routing_tpr = if routing_only_total > 0 { routing_only_correct as f64 / routing_only_total as f64 * 100.0 } else { 0.0 };
    let paraphrase_tpr = if paraphrase_only_total > 0 { paraphrase_only_correct as f64 / paraphrase_only_total as f64 * 100.0 } else { 0.0 };

    report.push_str("**Detection source analysis (Phase 2, second pass):**\n\n");
    report.push_str("| Source | Total Detections | Correct | TPR |\n|---|---|---|---|\n");
    report.push_str(&format!("| Both (dual-source) | {} | {} | {:.1}% |\n", dual_total, dual_correct, dual_tpr));
    report.push_str(&format!("| Routing only | {} | {} | {:.1}% |\n", routing_only_total, routing_only_correct, routing_tpr));
    report.push_str(&format!("| Paraphrase only | {} | {} | {:.1}% |\n", paraphrase_only_total, paraphrase_only_correct, paraphrase_tpr));

    report.push_str(&format!("\n**Dual-source confidence lift:** {:.1}x over routing-only",
        if routing_tpr > 0.0 { dual_tpr / routing_tpr } else { 0.0 }));
    if dual_tpr >= 95.0 {
        report.push_str("\n\n**FINDING:** Dual-source detections are 95%+ accurate. ");
        report.push_str("Can be used as high-confidence auto-route signal. ");
        report.push_str("Single-source detections should be escalated for human review.\n");
    } else if dual_tpr >= 85.0 {
        report.push_str("\n\n**FINDING:** Dual-source detections are 85%+ accurate. ");
        report.push_str("Moderately reliable as confidence signal but not sufficient for full auto-routing.\n");
    } else {
        report.push_str(&format!("\n\n**FINDING:** Dual-source TPR at {:.1}% is not high enough for auto-routing.\n", dual_tpr));
    }

    eprintln!("  Dual-source TPR: {:.1}%, Routing-only TPR: {:.1}%, Paraphrase-only TPR: {:.1}%",
        dual_tpr, routing_tpr, paraphrase_tpr);
}

fn run_experiment_f(report: &mut String) {
    eprintln!("\n=== Experiment F: Score Ratio Analysis ===");
    report.push_str("\n---\n\n## Experiment F: Score Ratio Analysis\n");
    report.push_str("**Analysis of score ratios from existing Phase 2 data**\n\n");

    let data = match std::fs::read_to_string("mesh_experiment_turns.json") {
        Ok(d) => d,
        Err(e) => {
            report.push_str(&format!("ERROR: Could not load mesh_experiment_turns.json: {}\n", e));
            eprintln!("  ERROR: {}", e);
            return;
        }
    };

    let records: Vec<TurnRecord> = match serde_json::from_str(&data) {
        Ok(r) => r,
        Err(e) => {
            report.push_str(&format!("ERROR: Could not parse data: {}\n", e));
            eprintln!("  ERROR: {}", e);
            return;
        }
    };

    // Use both baseline (Phase 1) and Phase 2 second_pass
    for phase_label in &["1", "2"] {
        let phase_records: Vec<&TurnRecord> = records.iter()
            .filter(|r| r.phase == *phase_label && r.pass_type == "second_pass")
            .collect();

        if phase_records.is_empty() { continue; }

        report.push_str(&format!("\n### Phase {} Score Ratios\n\n", phase_label));

        // For each turn, compute ratio: highest_correct_score / highest_fp_score
        let mut clean_ratios: Vec<f32> = Vec::new(); // turns with ONLY correct detections
        let mut mixed_ratios: Vec<f32> = Vec::new(); // turns with some FP
        let mut fp_only_scores: Vec<f32> = Vec::new(); // FP scores for threshold analysis

        for record in &phase_records {
            let gt_set: HashSet<&str> = record.ground_truth.iter().map(|s| s.as_str()).collect();

            let mut best_correct: f32 = 0.0;
            let mut best_fp: f32 = 0.0;
            let mut has_correct = false;
            let mut has_fp = false;

            for det in &record.detected_top5 {
                if gt_set.contains(det.id.as_str()) {
                    if det.score > best_correct { best_correct = det.score; }
                    has_correct = true;
                } else {
                    if det.score > best_fp { best_fp = det.score; }
                    has_fp = true;
                    fp_only_scores.push(det.score);
                }
            }

            if has_correct && !has_fp {
                clean_ratios.push(best_correct);
            } else if has_correct && has_fp && best_fp > 0.0 {
                mixed_ratios.push(best_correct / best_fp);
            }
        }

        // Distribution of ratios
        let ratio_buckets = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
        report.push_str("**Score ratio distribution (correct/FP) for mixed turns:**\n\n");
        report.push_str("| Ratio Range | Count | Cumulative% |\n|---|---|---|\n");

        let total_mixed = mixed_ratios.len();
        if total_mixed > 0 {
            let mut cumulative = 0;
            let mut prev_bucket = 0.0f32;
            for &bucket in &ratio_buckets {
                let count = mixed_ratios.iter().filter(|&&r| r >= prev_bucket && r < bucket).count();
                cumulative += count;
                let cum_pct = cumulative as f64 / total_mixed as f64 * 100.0;
                report.push_str(&format!("| {:.1}-{:.1}x | {} | {:.1}% |\n",
                    prev_bucket, bucket, count, cum_pct));
                prev_bucket = bucket;
            }
            let remaining = mixed_ratios.iter().filter(|&&r| r >= *ratio_buckets.last().unwrap()).count();
            cumulative += remaining;
            let cum_pct = cumulative as f64 / total_mixed as f64 * 100.0;
            report.push_str(&format!("| {:.1}x+ | {} | {:.1}% |\n",
                ratio_buckets.last().unwrap(), remaining, cum_pct));
        } else {
            report.push_str("| (no mixed turns) | 0 | - |\n");
        }

        // Summary stats
        report.push_str(&format!("\n**Clean turns (correct only):** {}\n", clean_ratios.len()));
        report.push_str(&format!("**Mixed turns (correct + FP):** {}\n", mixed_ratios.len()));

        if !mixed_ratios.is_empty() {
            let mut sorted = mixed_ratios.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sorted[sorted.len() / 2];
            let mean: f32 = sorted.iter().sum::<f32>() / sorted.len() as f32;
            let p25 = sorted[sorted.len() / 4];
            let p75 = sorted[3 * sorted.len() / 4];

            report.push_str(&format!("**Ratio stats:** median={:.2}x, mean={:.2}x, P25={:.2}x, P75={:.2}x\n",
                median, mean, p25, p75));

            // Threshold recommendation
            let above_2x = mixed_ratios.iter().filter(|&&r| r >= 2.0).count();
            let above_2x_pct = above_2x as f64 / total_mixed as f64 * 100.0;
            report.push_str(&format!("\n**Threshold finding:** {:.1}% of mixed turns have ratio >= 2.0x\n", above_2x_pct));
            if median >= 2.0 {
                report.push_str("A 2.0x ratio threshold could separate clean from noisy results.\n");
            } else {
                report.push_str("Correct and FP scores are too close — ratio alone is not a reliable separator.\n");
            }
        }

        // FP score distribution
        if !fp_only_scores.is_empty() {
            let mut sorted_fp = fp_only_scores.clone();
            sorted_fp.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let fp_median = sorted_fp[sorted_fp.len() / 2];
            let fp_p90 = sorted_fp[(sorted_fp.len() as f64 * 0.9) as usize];

            report.push_str(&format!("\n**FP score distribution:** median={:.2}, P90={:.2} (N={})\n",
                fp_median, fp_p90, fp_only_scores.len()));
        }
    }
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

    let boundary_data: HashMap<String, Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string("tests/data/boundary_patterns.json")
            .expect("Failed to read boundary_patterns.json"))
        .expect("Failed to parse boundary_patterns.json");

    let modifier_entries: Vec<ModifierEntry> =
        serde_json::from_str(&std::fs::read_to_string("tests/data/modifiers.json")
            .expect("Failed to read modifiers.json"))
        .expect("Failed to parse modifiers.json");

    let total_turns: usize = scenarios.iter().map(|s| s.turns.len()).sum();
    let new_turns: usize = new_scenarios.iter().map(|s| s.turns.len()).sum();
    let total_boundary_patterns: usize = boundary_data.values().map(|v| v.len()).sum();

    eprintln!("Data loaded: {} scenarios ({} turns), {} new scenarios ({} turns)",
        scenarios.len(), total_turns, new_scenarios.len(), new_turns);
    eprintln!("Paraphrases: {} intents, {} total phrases",
        paraphrase_data.len(),
        paraphrase_data.values().map(|v| v.len()).sum::<usize>());
    eprintln!("Corrections: {} mappings", correction_data.len());
    eprintln!("Boundary patterns: {} total", total_boundary_patterns);
    eprintln!("Modifiers: {} patterns (metadata-only)", modifier_entries.len());

    let threshold = 0.3;
    let mut report = String::new();
    let mut all_records: Vec<TurnRecord> = Vec::new();

    report.push_str("# ASV Clean Pipeline — Experiment Results\n\n");
    report.push_str(&format!("Scenarios: {} ({} turns) + {} new ({} turns)\n",
        scenarios.len(), total_turns, new_scenarios.len(), new_turns));
    report.push_str(&format!("Boundary patterns: {}\n\n", total_boundary_patterns));

    // ============================
    // BASELINE: Phase 2 (Paraphrase + Routing + Learning)
    // ============================
    eprintln!("\n=== BASELINE: Phase 2 (Paraphrase + Routing) ===");
    report.push_str("---\n\n## BASELINE: Phase 2 (Paraphrase + Routing + Learning)\n\n");

    let router_b = setup_router();
    let pi_b = ParaphraseIndex::load_from_json(&paraphrase_data);
    let mi_b = ModifierIndex::load_from_json(&modifier_entries);
    let mut baseline = CleanRouter::new(router_b, pi_b).with_modifier(mi_b);

    eprintln!("  First pass (with learning)...");
    let baseline_first = run_experiment_phase(&mut baseline, &scenarios, "baseline", "first_pass", true, threshold);
    all_records.extend(baseline_first);

    eprintln!("  Second pass (generalization)...");
    let baseline_second = run_experiment_phase(&mut baseline, &scenarios, "baseline", "second_pass", false, threshold);
    let baseline_stats = compute_stats(&baseline_second);
    all_records.extend(baseline_second);

    eprintln!("  New scenarios...");
    let baseline_new = run_experiment_phase(&mut baseline, &new_scenarios, "baseline", "new_scenario", false, threshold);
    let baseline_new_stats = compute_stats(&baseline_new);
    all_records.extend(baseline_new);

    write_experiment_report(&mut report, "Baseline (Phase 2)", "Paraphrase + Routing + Learning",
        &baseline_stats, Some(&baseline_new_stats));
    report.push_str(&format!("\n**Corrections applied:** {}\n", baseline.corrections_applied));

    // ============================
    // EXPERIMENT A: Phase 2 + Correction Index
    // ============================
    eprintln!("\n=== Experiment A: Phase 2 + Correction Index ===");

    let router_a = setup_router();
    let pi_a = ParaphraseIndex::load_from_json(&paraphrase_data);
    let ci_a = CorrectionIndex::load_from_json(&correction_data);
    let mi_a = ModifierIndex::load_from_json(&modifier_entries);
    let mut mesh_a = CleanRouter::new(router_a, pi_a).with_correction(ci_a).with_modifier(mi_a);

    eprintln!("  First pass (with learning)...");
    let a_first = run_experiment_phase(&mut mesh_a, &scenarios, "A", "first_pass", true, threshold);
    all_records.extend(a_first);

    eprintln!("  Second pass (generalization)...");
    let a_second = run_experiment_phase(&mut mesh_a, &scenarios, "A", "second_pass", false, threshold);
    let a_stats = compute_stats(&a_second);
    all_records.extend(a_second);

    eprintln!("  New scenarios...");
    let a_new = run_experiment_phase(&mut mesh_a, &new_scenarios, "A", "new_scenario", false, threshold);
    let a_new_stats = compute_stats(&a_new);
    all_records.extend(a_new);

    write_experiment_report(&mut report, "Experiment A: Phase 2 + Correction",
        "Correction -> Paraphrase + Routing + Learning",
        &a_stats, Some(&a_new_stats));
    report.push_str(&format!("\n**Corrections applied:** {}\n", mesh_a.corrections_applied));

    // Delta vs baseline
    let baseline_exact = baseline_stats.exact_pass as f64 / baseline_stats.total.max(1) as f64 * 100.0;
    let a_exact = a_stats.exact_pass as f64 / a_stats.total.max(1) as f64 * 100.0;
    report.push_str(&format!("\n**Delta vs baseline:** {:.1}% ({})\n",
        a_exact - baseline_exact,
        if a_exact > baseline_exact { "improvement" } else if a_exact < baseline_exact { "regression" } else { "neutral" }));

    // ============================
    // EXPERIMENT B: Phase 2 + Boundary Segmentation
    // ============================
    eprintln!("\n=== Experiment B: Phase 2 + Boundary Segmentation ===");

    let router_b2 = setup_router();
    let pi_b2 = ParaphraseIndex::load_from_json(&paraphrase_data);
    let bi_b = BoundaryPatternIndex::load_from_json(&boundary_data);
    let mi_b2 = ModifierIndex::load_from_json(&modifier_entries);
    let mut mesh_b = CleanRouter::new(router_b2, pi_b2).with_boundary(bi_b).with_modifier(mi_b2);

    eprintln!("  First pass (with learning)...");
    let b_first = run_experiment_phase(&mut mesh_b, &scenarios, "B", "first_pass", true, threshold);
    all_records.extend(b_first);

    eprintln!("  Second pass (generalization)...");
    let b_second = run_experiment_phase(&mut mesh_b, &scenarios, "B", "second_pass", false, threshold);
    let b_stats = compute_stats(&b_second);
    all_records.extend(b_second);

    eprintln!("  New scenarios...");
    let b_new = run_experiment_phase(&mut mesh_b, &new_scenarios, "B", "new_scenario", false, threshold);
    let b_new_stats = compute_stats(&b_new);
    all_records.extend(b_new);

    write_experiment_report(&mut report, "Experiment B: Phase 2 + Boundary Segmentation",
        "Boundary Split -> [per segment: Paraphrase + Routing] -> Merge + Learning",
        &b_stats, Some(&b_new_stats));
    report.push_str(&format!("\n**Corrections applied:** {}\n", mesh_b.corrections_applied));

    // Delta vs baseline with word count focus
    let b_exact = b_stats.exact_pass as f64 / b_stats.total.max(1) as f64 * 100.0;
    report.push_str(&format!("\n**Delta vs baseline:** {:.1}%\n", b_exact - baseline_exact));

    // Segmentation impact by word count
    report.push_str("\n**Segmentation impact by word count (Experiment B vs Baseline):**\n\n");
    report.push_str("| Words | Baseline Exact% | Exp B Exact% | Delta |\n|---|---|---|---|\n");
    for bucket in &["1-5", "6-10", "11-20", "21-40", "41+"] {
        let base_e = baseline_stats.by_wordcount.get(*bucket);
        let b_e = b_stats.by_wordcount.get(*bucket);
        if let (Some((bt, be, _)), Some((et, ee, _))) = (base_e, b_e) {
            let bpct = if *bt > 0 { *be as f64 / *bt as f64 * 100.0 } else { 0.0 };
            let epct = if *et > 0 { *ee as f64 / *et as f64 * 100.0 } else { 0.0 };
            let marker = if *bucket == "21-40" || *bucket == "41+" { " **" } else { "" };
            report.push_str(&format!("| {}{} | {:.1}% | {:.1}% | {:+.1}% |\n",
                bucket, marker, bpct, epct, epct - bpct));
        }
    }

    // Segmentation statistics
    let b_second_records: Vec<&TurnRecord> = all_records.iter()
        .filter(|r| r.phase == "B" && r.pass_type == "second_pass")
        .collect();
    let total_segments: usize = b_second_records.iter().map(|r| r.segments_used).sum();
    let multi_seg = b_second_records.iter().filter(|r| r.segments_used > 1).count();
    report.push_str(&format!("\n**Segmentation stats:** {}/{} turns segmented, avg {:.1} segments/turn\n",
        multi_seg, b_second_records.len(),
        total_segments as f64 / b_second_records.len().max(1) as f64));

    // ============================
    // EXPERIMENT C: Phase 2 + Correction + Boundary
    // ============================
    eprintln!("\n=== Experiment C: Full Clean Pipeline ===");

    let router_c = setup_router();
    let pi_c = ParaphraseIndex::load_from_json(&paraphrase_data);
    let ci_c = CorrectionIndex::load_from_json(&correction_data);
    let bi_c = BoundaryPatternIndex::load_from_json(&boundary_data);
    let mi_c = ModifierIndex::load_from_json(&modifier_entries);
    let mut mesh_c = CleanRouter::new(router_c, pi_c)
        .with_correction(ci_c)
        .with_boundary(bi_c)
        .with_modifier(mi_c);

    eprintln!("  First pass (with learning)...");
    let c_first = run_experiment_phase(&mut mesh_c, &scenarios, "C", "first_pass", true, threshold);
    all_records.extend(c_first);

    eprintln!("  Second pass (generalization)...");
    let c_second = run_experiment_phase(&mut mesh_c, &scenarios, "C", "second_pass", false, threshold);
    let c_stats = compute_stats(&c_second);
    all_records.extend(c_second);

    eprintln!("  New scenarios...");
    let c_new = run_experiment_phase(&mut mesh_c, &new_scenarios, "C", "new_scenario", false, threshold);
    let c_new_stats = compute_stats(&c_new);
    all_records.extend(c_new);

    write_experiment_report(&mut report, "Experiment C: Full Clean Pipeline",
        "Correction -> Boundary Split -> [per segment: Paraphrase + Routing] -> Merge + Learning",
        &c_stats, Some(&c_new_stats));
    report.push_str(&format!("\n**Corrections applied:** {}\n", mesh_c.corrections_applied));

    let c_exact = c_stats.exact_pass as f64 / c_stats.total.max(1) as f64 * 100.0;
    report.push_str(&format!("\n**Delta vs baseline:** {:.1}%\n", c_exact - baseline_exact));
    report.push_str(&format!("**Delta vs Exp A (correction only):** {:.1}%\n", c_exact - a_exact));
    report.push_str(&format!("**Delta vs Exp B (boundary only):** {:.1}%\n", c_exact - b_exact));

    // ============================
    // EXPERIMENT D: Dual-source confidence signal (analysis)
    // ============================
    run_experiment_d(&mut report);

    // ============================
    // EXPERIMENT E: Cumulative learning (3 passes)
    // ============================
    eprintln!("\n=== Experiment E: Cumulative Learning (3 passes) ===");
    report.push_str("\n---\n\n## Experiment E: Cumulative Learning (3 passes)\n");
    report.push_str("**Phase 2 architecture, 30 scenarios run 3 times with continuous learning**\n\n");

    let router_e = setup_router();
    let pi_e = ParaphraseIndex::load_from_json(&paraphrase_data);
    let mi_e = ModifierIndex::load_from_json(&modifier_entries);
    let mut mesh_e = CleanRouter::new(router_e, pi_e).with_modifier(mi_e);

    report.push_str("| Pass | Exact% | Top-5 Recall% | Avg FP/turn | Corrections (cumulative) |\n");
    report.push_str("|---|---|---|---|---|\n");

    for pass_num in 1..=3 {
        eprintln!("  Pass {} (with learning)...", pass_num);
        let pass_name = format!("E_pass{}", pass_num);
        let pass_records = run_experiment_phase(
            &mut mesh_e, &scenarios, &pass_name, "learning_pass", true, threshold);

        let stats = compute_stats(&pass_records);
        let exact_pct = stats.exact_pass as f64 / stats.total.max(1) as f64 * 100.0;
        let recall_pct = stats.top5_recall as f64 / stats.total.max(1) as f64 * 100.0;
        let fp_avg = stats.total_fp as f64 / stats.total.max(1) as f64;

        report.push_str(&format!("| Pass {} | {:.1}% | {:.1}% | {:.2} | {} |\n",
            pass_num, exact_pct, recall_pct, fp_avg, mesh_e.corrections_applied));

        eprintln!("  Pass {}: Exact={:.1}%, Recall={:.1}%, Corrections={}",
            pass_num, exact_pct, recall_pct, mesh_e.corrections_applied);

        all_records.extend(pass_records);
    }

    // Generalization test after 3 passes
    eprintln!("  Generalization pass...");
    let e_gen = run_experiment_phase(&mut mesh_e, &scenarios, "E_gen", "generalization", false, threshold);
    let e_gen_stats = compute_stats(&e_gen);
    let e_gen_exact = e_gen_stats.exact_pass as f64 / e_gen_stats.total.max(1) as f64 * 100.0;
    let e_gen_recall = e_gen_stats.top5_recall as f64 / e_gen_stats.total.max(1) as f64 * 100.0;
    let e_gen_fp = e_gen_stats.total_fp as f64 / e_gen_stats.total.max(1) as f64;

    report.push_str(&format!("| Generalization | {:.1}% | {:.1}% | {:.2} | {} |\n",
        e_gen_exact, e_gen_recall, e_gen_fp, mesh_e.corrections_applied));

    all_records.extend(e_gen);

    // New scenarios after 3 passes
    eprintln!("  New scenarios...");
    let e_new = run_experiment_phase(&mut mesh_e, &new_scenarios, "E_new", "new_scenario", false, threshold);
    let e_new_stats = compute_stats(&e_new);
    let e_new_exact = e_new_stats.exact_pass as f64 / e_new_stats.total.max(1) as f64 * 100.0;
    let e_new_recall = e_new_stats.top5_recall as f64 / e_new_stats.total.max(1) as f64 * 100.0;

    report.push_str(&format!("\n**After 3 passes new scenarios:** {:.1}% exact, {:.1}% recall\n", e_new_exact, e_new_recall));
    report.push_str(&format!("**Total corrections applied:** {}\n", mesh_e.corrections_applied));

    // Plateau/degrade analysis
    report.push_str("\n**Learning trajectory:** ");
    report.push_str("Does the system keep improving or plateau/degrade across passes?\n");

    all_records.extend(e_new);

    // ============================
    // EXPERIMENT F: Score ratio analysis
    // ============================
    run_experiment_f(&mut report);

    // ============================
    // COMPARISON SUMMARY
    // ============================
    report.push_str("\n---\n\n## COMPARISON SUMMARY\n\n");
    report.push_str("### Second Pass Results (30 scenarios, generalization)\n\n");
    report.push_str("| Experiment | Exact% | Top-5 Recall% | Avg FP/turn | Delta vs Baseline |\n");
    report.push_str("|---|---|---|---|---|\n");

    for (name, stats) in &[
        ("Baseline (Phase 2)", &baseline_stats),
        ("A: +Correction", &a_stats),
        ("B: +Boundary", &b_stats),
        ("C: +Correction+Boundary", &c_stats),
    ] {
        let exact = (stats.exact_pass as f64 / stats.total.max(1) as f64) * 100.0;
        let recall = (stats.top5_recall as f64 / stats.total.max(1) as f64) * 100.0;
        let fp = stats.total_fp as f64 / stats.total.max(1) as f64;
        let delta = exact - baseline_exact;
        report.push_str(&format!("| {} | {:.1}% | {:.1}% | {:.2} | {:+.1}% |\n",
            name, exact, recall, fp, delta));
    }

    // Word count comparison for segmentation experiments
    report.push_str("\n### By Word Count — Segmentation Focus\n\n");
    report.push_str("| Words | Baseline | Exp B (Boundary) | Exp C (Full) | B Delta | C Delta |\n");
    report.push_str("|---|---|---|---|---|---|\n");
    for bucket in &["1-5", "6-10", "11-20", "21-40", "41+"] {
        let get_pct = |stats: &PhaseStats| -> f64 {
            stats.by_wordcount.get(*bucket)
                .map(|(t, e, _)| if *t > 0 { *e as f64 / *t as f64 * 100.0 } else { 0.0 })
                .unwrap_or(0.0)
        };
        let base_pct = get_pct(&baseline_stats);
        let b_pct = get_pct(&b_stats);
        let c_pct = get_pct(&c_stats);
        report.push_str(&format!("| {} | {:.1}% | {:.1}% | {:.1}% | {:+.1}% | {:+.1}% |\n",
            bucket, base_pct, b_pct, c_pct, b_pct - base_pct, c_pct - base_pct));
    }

    // New scenario comparison
    report.push_str("\n### New Scenarios (10 unseen)\n\n");
    report.push_str("| Experiment | Exact% | Top-5 Recall% |\n|---|---|---|\n");
    for (name, stats) in &[
        ("Baseline", &baseline_new_stats),
        ("A: +Correction", &a_new_stats),
        ("B: +Boundary", &b_new_stats),
        ("C: Full Clean", &c_new_stats),
    ] {
        let exact = (stats.exact_pass as f64 / stats.total.max(1) as f64) * 100.0;
        let recall = (stats.top5_recall as f64 / stats.total.max(1) as f64) * 100.0;
        report.push_str(&format!("| {} | {:.1}% | {:.1}% |\n", name, exact, recall));
    }

    // Save report
    std::fs::write("clean_experiment_results.md", &report)
        .expect("Failed to write results");
    eprintln!("\nResults saved to clean_experiment_results.md");

    // Save detailed turn data
    let json_output = serde_json::to_string_pretty(&all_records)
        .expect("Failed to serialize turn records");
    std::fs::write("clean_experiment_turns.json", &json_output)
        .expect("Failed to write turn data");
    eprintln!("Detailed turn data saved to clean_experiment_turns.json");

    // Print summary
    eprintln!("\n=== SUMMARY ===");
    eprintln!("Baseline (Phase 2): Exact={:.1}%", baseline_exact);
    eprintln!("Exp A (+Correction): Exact={:.1}% (delta: {:+.1}%)", a_exact, a_exact - baseline_exact);
    eprintln!("Exp B (+Boundary):   Exact={:.1}% (delta: {:+.1}%)", b_exact, b_exact - baseline_exact);
    eprintln!("Exp C (Full Clean):  Exact={:.1}% (delta: {:+.1}%)", c_exact, c_exact - baseline_exact);
    eprintln!("Exp E (3-pass gen):  Exact={:.1}%", e_gen_exact);
}
