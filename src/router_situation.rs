//! Router: situation index (state-description → action inference).
//!
//! Situation patterns are matched by direct substring search against raw query text.
//! Works for Latin, CJK, and mixed-script queries without any script detection.
//!
//! Score per match: `weight × sqrt(char_count)` — longer patterns are more specific.
//! Threshold: 0.8 — requires at least one strong domain-specific match.
//!
//! Sits alongside the term index as an additive signal. Pure action queries
//! ("charge the card") score ~0 on the situation index; situation queries
//! ("payment bounced three times") score 0 on the term index.

use crate::*;
use crate::tokenizer::universal_stop_set;
use std::collections::HashMap;

/// Blending factor: situation score added to term score.
/// 0.4 keeps situation inference subordinate to direct vocabulary matching.
pub(crate) const SITUATION_ALPHA: f32 = 0.4;

/// Minimum combined situation score to surface a detection.
pub(crate) const SITUATION_THRESHOLD: f32 = 0.8;

impl Router {
    /// Add situation patterns for an intent.
    ///
    /// Situation patterns describe states that imply an action — not the action itself.
    /// They are matched by direct substring search (case-insensitive) against the raw query.
    ///
    /// Weight guide:
    /// - 1.0: strongly domain-specific (e.g., "OOM", "付款", "LGTM", "402")
    /// - 0.7-0.8: moderately specific (e.g., "prod", "CI", "payment")
    /// - 0.4-0.6: generic signals that need a partner to exceed threshold
    ///
    /// ```
    /// use asv_router::Router;
    ///
    /// let mut router = Router::new();
    /// router.add_intent("create_issue", &["report a bug", "log an issue"]);
    /// router.add_situation_patterns("create_issue", &[
    ///     ("build failed", 1.0),
    ///     ("OOM", 1.0),
    ///     ("500", 1.0),
    ///     ("prod", 0.8),
    /// ]);
    ///
    /// // Now "the build failed on prod" routes to create_issue
    /// // even though it contains no action vocabulary.
    /// let result = router.route("the build failed on prod");
    /// assert_eq!(result[0].id, "create_issue");
    /// ```
    pub fn add_situation_patterns(&mut self, intent_id: &str, patterns: &[(&str, f32)]) {
        self.require_local();
        let entry = self.situation_patterns.entry(intent_id.to_string()).or_default();
        for (pattern, weight) in patterns {
            entry.push((pattern.to_string(), *weight));
        }
        self.version += 1;
    }

    /// Add situation patterns from a nested map (for bulk loading from JSON).
    /// Format: `{ intent_id: [(pattern, weight), ...] }`
    pub fn add_situation_patterns_bulk(&mut self, data: &HashMap<String, Vec<(String, f32)>>) {
        self.require_local();
        for (intent_id, patterns) in data {
            let entry = self.situation_patterns.entry(intent_id.clone()).or_default();
            for (pattern, weight) in patterns {
                entry.push((pattern.clone(), *weight));
            }
        }
        self.version += 1;
    }

    /// Get all situation patterns for an intent.
    pub fn get_situation_patterns(&self, intent_id: &str) -> Option<&Vec<(String, f32)>> {
        self.situation_patterns.get(intent_id)
    }

    /// Remove a single situation pattern from an intent by exact pattern string.
    /// Returns true if the pattern was found and removed.
    pub fn remove_situation_pattern(&mut self, intent_id: &str, pattern: &str) -> bool {
        if let Some(patterns) = self.situation_patterns.get_mut(intent_id) {
            let before = patterns.len();
            patterns.retain(|(p, _)| p != pattern);
            let removed = patterns.len() < before;
            if removed {
                self.version += 1;
            }
            return removed;
        }
        false
    }

    /// Scan a query against all situation patterns.
    ///
    /// Returns `(intent_id, score)` pairs sorted by score descending.
    /// Only includes intents whose score meets `SITUATION_THRESHOLD`.
    pub(crate) fn situation_scan(&self, query: &str) -> Vec<(String, f32)> {
        if self.situation_patterns.is_empty() {
            return vec![];
        }

        let query_lower = query.to_lowercase();
        let mut results: Vec<(String, f32)> = Vec::new();

        for (intent_id, patterns) in &self.situation_patterns {
            let mut score = 0.0f32;
            for (pattern, weight) in patterns {
                let matched = query.contains(pattern.as_str())
                    || query_lower.contains(&pattern.to_lowercase());
                if matched {
                    let char_len = pattern.chars().count() as f32;
                    score += weight * char_len.sqrt();
                }
            }
            if score >= SITUATION_THRESHOLD {
                results.push((intent_id.clone(), score));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Like `situation_scan` but also returns which patterns matched per intent.
    /// Returns (intent_id, total_score, matched_patterns: Vec<(pattern, contribution)>)
    pub(crate) fn situation_scan_detailed(&self, query: &str) -> Vec<(String, f32, Vec<(String, f32)>)> {
        if self.situation_patterns.is_empty() {
            return vec![];
        }
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        for (intent_id, patterns) in &self.situation_patterns {
            let mut score = 0.0f32;
            let mut matched: Vec<(String, f32)> = Vec::new();
            for (pattern, weight) in patterns {
                let matched_flag = query.contains(pattern.as_str())
                    || query_lower.contains(&pattern.to_lowercase());
                if matched_flag {
                    let char_len = pattern.chars().count() as f32;
                    let contribution = weight * char_len.sqrt();
                    score += contribution;
                    matched.push((pattern.clone(), (contribution * 100.0).round() / 100.0));
                }
            }
            if score >= SITUATION_THRESHOLD {
                results.push((intent_id.clone(), score, matched));
            }
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Check a situation pattern before adding it. Pure read — no mutation.
    ///
    /// Returns a `SituationGuardResult` describing any conflicts, duplicates, or
    /// genericness. The caller decides whether to proceed.
    ///
    /// Conflict: exact pattern already in another intent → both would fire equally → noise.
    /// Duplicate: pattern already in THIS intent → skip.
    /// Too generic: pattern is too short or all-stopword → weak signal at best, noise at worst.
    pub fn check_situation_pattern(&self, intent_id: &str, pattern: &str) -> SituationGuardResult {
        use crate::tokenizer::universal_stop_set;

        // Duplicate check
        if let Some(patterns) = self.situation_patterns.get(intent_id) {
            let pat_lower = pattern.to_lowercase();
            if patterns.iter().any(|(p, _)| p.to_lowercase() == pat_lower) {
                return SituationGuardResult {
                    added: false,
                    conflicts: vec![],
                    duplicate: true,
                    too_generic: false,
                    warning: Some("Pattern already exists in this intent".to_string()),
                };
            }
        }

        // Too-generic check
        // Latin patterns: all words are stop words, or pattern is < 3 chars and ASCII-only
        // CJK patterns: fewer than 2 CJK chars (single char is too ambiguous)
        let cjk_char_count = pattern.chars().filter(|c| is_cjk_char(*c)).count();
        let is_cjk_pattern = cjk_char_count > 0;
        let too_generic = if is_cjk_pattern {
            cjk_char_count < 2
        } else {
            let stops = universal_stop_set();
            let words: Vec<&str> = pattern.split_whitespace().collect();
            let all_stop = !words.is_empty() && words.iter().all(|w| stops.contains(*w));
            let too_short = pattern.chars().count() < 3;
            all_stop || too_short
        };

        if too_generic {
            return SituationGuardResult {
                added: false,
                conflicts: vec![],
                duplicate: false,
                too_generic: true,
                warning: Some(format!("Pattern '{}' is too generic to be a useful signal", pattern)),
            };
        }

        // Conflict check: same exact pattern in another intent
        let pat_lower = pattern.to_lowercase();
        let mut conflicts = vec![];
        for (other_id, patterns) in &self.situation_patterns {
            if other_id == intent_id { continue; }
            if let Some((_, w)) = patterns.iter().find(|(p, _)| p.to_lowercase() == pat_lower) {
                conflicts.push(SituationConflict {
                    competing_intent: other_id.clone(),
                    competing_weight: *w,
                });
            }
        }

        let warning = if !conflicts.is_empty() {
            let names: Vec<String> = conflicts.iter().map(|c| c.competing_intent.clone()).collect();
            Some(format!("Pattern '{}' already in: {} — both intents will fire", pattern, names.join(", ")))
        } else {
            None
        };

        SituationGuardResult { added: false, conflicts, duplicate: false, too_generic: false, warning }
    }

    /// Add a situation pattern with guard checking.
    ///
    /// Blocks if: duplicate, too generic, or conflicts with another intent.
    /// Returns the guard result — `result.added` tells you if it landed.
    pub fn add_situation_pattern_checked(&mut self, intent_id: &str, pattern: &str, weight: f32) -> SituationGuardResult {
        let mut result = self.check_situation_pattern(intent_id, pattern);
        if result.duplicate || result.too_generic || !result.conflicts.is_empty() {
            return result;
        }
        let entry = self.situation_patterns.entry(intent_id.to_string()).or_default();
        entry.push((pattern.to_string(), weight));
        self.version += 1;
        result.added = true;
        result
    }

    /// Learn situation n-grams from a query for an intent (online learning).
    ///
    /// Extracts char 2-grams and 3-grams from meaningful characters
    /// (CJK ideographs, kana, hangul, ASCII alphanumeric) and adds them
    /// at weight 0.4 — weak signals that need a partner to fire.
    ///
    /// Guard is applied: patterns that conflict with other intents or are too
    /// generic are silently skipped. This keeps auto-learned patterns clean.
    pub fn learn_situation(&mut self, query: &str, intent_id: &str) {
        self.require_local();
        let chars: Vec<char> = query.chars()
            .filter(|c| is_situation_meaningful(*c))
            .collect();

        let mut candidates: Vec<String> = Vec::new();

        for i in 0..chars.len().saturating_sub(1) {
            candidates.push(chars[i..i + 2].iter().collect());
        }
        if chars.len() >= 3 {
            for i in 0..chars.len().saturating_sub(2) {
                candidates.push(chars[i..i + 3].iter().collect());
            }
        }

        for pattern in candidates {
            let result = self.check_situation_pattern(intent_id, &pattern);
            if result.added || (!result.duplicate && !result.too_generic && result.conflicts.is_empty()) {
                let entry = self.situation_patterns.entry(intent_id.to_string()).or_default();
                if !entry.iter().any(|(p, _)| p == &pattern) {
                    entry.push((pattern, 0.4));
                }
            }
        }

        self.version += 1;
    }
}

/// Returns true for CJK ideograph characters (used for genericness check).
fn is_cjk_char(c: char) -> bool {
    let cp = c as u32;
    (cp >= 0x4E00 && cp <= 0x9FFF)
        || (cp >= 0x3400 && cp <= 0x4DBF)
        || (cp >= 0x3040 && cp <= 0x309F)
        || (cp >= 0x30A0 && cp <= 0x30FF)
        || (cp >= 0xAC00 && cp <= 0xD7AF)
}

/// Returns true for characters worth including in situation n-gram extraction.
fn is_situation_meaningful(c: char) -> bool {
    let cp = c as u32;
    matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9')
        || (cp >= 0x4E00 && cp <= 0x9FFF)   // CJK Unified Ideographs
        || (cp >= 0x3400 && cp <= 0x4DBF)   // CJK Extension A
        || (cp >= 0x3040 && cp <= 0x309F)   // Hiragana
        || (cp >= 0x30A0 && cp <= 0x30FF)   // Katakana
        || (cp >= 0xAC00 && cp <= 0xD7AF)   // Hangul Syllables
}
