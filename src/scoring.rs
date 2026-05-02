//! # L2 — IDF-weighted intent index with multi-round token consumption
//!
//! Word → intent posting list with cached IDF and incrementally maintained
//! reverse-index for confidence computation. Multi-intent decomposition runs
//! up to three rounds, consuming tokens claimed by confirmed intents
//! between rounds. Cross-provider tiebreak picks the best provider when
//! the same action appears under multiple namespaces.
//!
//! L0 (typo correction) and L1 (lexical-graph query rewriting) were
//! removed in v0.2.0 — see launch notes.

use crate::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A conjunction rule fires when ALL listed words appear in the normalized query.
/// Adds a bonus activation to the target intent on top of individual word weights.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConjunctionRule {
    pub words: Vec<String>,
    pub intent: String,
    pub bonus: f32,
}

/// Full routing result with disposition and ranked candidates.
#[derive(Debug, Clone)]
pub struct RouteResult {
    pub confirmed: Vec<(String, f32)>,
    pub ranked: Vec<(String, f32)>,
    pub disposition: String,
    pub has_negation: bool,
}

/// One round of multi-intent resolution captured for debug/inspection.
#[derive(Serialize, Clone, Debug)]
pub struct RoundTrace {
    pub tokens_in: Vec<String>,
    pub scored: Vec<(String, f32)>,
    pub confirmed: Vec<String>,
    pub consumed: Vec<String>,
}

/// Full multi-intent resolution trace returned by `score_multi_traced`.
#[derive(Serialize, Clone, Debug)]
pub struct MultiIntentTrace {
    pub rounds: Vec<RoundTrace>,
    pub stop_reason: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct IntentIndex {
    /// word → [(intent_id, weight 0.0–1.0)]
    #[serde(default)]
    pub word_intent: HashMap<String, Vec<(String, f32)>>,

    /// Conjunction bonuses — word pairs that together strongly indicate an intent.
    #[serde(default)]
    pub conjunctions: Vec<ConjunctionRule>,

    /// Char-ngram tiebreaker index: intent_id → set of char 4-grams from seed phrases.
    #[serde(default)]
    pub char_ngrams: HashMap<String, std::collections::HashSet<String>>,

    /// Cached count of distinct intents in the index.
    #[serde(default)]
    pub intent_count: usize,

    /// Per-word IDF cache — rebuilt on load via `rebuild_caches`.
    #[serde(skip)]
    idf_cache: FxHashMap<String, f32>,

    #[serde(skip)]
    known_intents: FxHashSet<String>,

    #[serde(skip)]
    known_words: FxHashSet<String>,

    /// Reverse index: `intent_id → set of tokens that fire on this intent`.
    #[serde(skip)]
    intent_to_tokens: FxHashMap<String, FxHashSet<String>>,
}

impl IntentIndex {
    pub fn new() -> Self {
        Self::default()
    }

    const PHRASE_RATE: f32 = 0.4;
    const LEARN_RATE: f32 = 0.3;

    /// Recompute intent_count and idf_cache from word_intent in one pass.
    pub fn rebuild_caches(&mut self) {
        self.known_intents.clear();
        self.known_words.clear();
        self.intent_to_tokens.clear();
        for (word, entries) in &self.word_intent {
            self.known_words.insert(word.clone());
            for (id, _) in entries {
                self.known_intents.insert(id.clone());
                self.intent_to_tokens
                    .entry(id.clone())
                    .or_default()
                    .insert(word.clone());
            }
        }
        let n = self.known_intents.len();
        self.intent_count = n;
        let n_f = n.max(1) as f32;
        self.idf_cache.clear();
        for (word, entries) in &self.word_intent {
            let idf = (n_f / entries.len() as f32).ln().max(0.0);
            self.idf_cache.insert(word.clone(), idf);
        }
    }

    pub fn known_words(&self) -> &FxHashSet<String> {
        &self.known_words
    }

    fn refresh_idf_for(&mut self, word: &str) {
        if let Some(entries) = self.word_intent.get(word) {
            let n_f = self.intent_count.max(1) as f32;
            let idf = (n_f / entries.len() as f32).ln().max(0.0);
            self.idf_cache.insert(word.to_string(), idf);
        } else {
            self.idf_cache.remove(word);
        }
    }

    #[inline]
    fn idf(&self, word: &str) -> f32 {
        self.idf_cache.get(word).copied().unwrap_or_else(|| {
            self.word_intent
                .get(word)
                .map(|e| {
                    (self.intent_count.max(1) as f32 / e.len() as f32)
                        .ln()
                        .max(0.0)
                })
                .unwrap_or(0.0)
        })
    }

    pub fn intent_max_score(&self, tokens: &[String], intent: &str) -> f32 {
        let Some(intent_vocab) = self.intent_to_tokens.get(intent) else {
            return 0.0;
        };
        tokens
            .iter()
            .map(|t| t.strip_prefix("not_").unwrap_or(t.as_str()))
            .filter(|base| intent_vocab.contains(*base))
            .map(|base| self.idf(base))
            .sum()
    }

    pub fn confidence_for(&self, raw_score: f32, tokens: &[String], intent: &str) -> f32 {
        let max = self.intent_max_score(tokens, intent);
        if max < 1e-6 {
            return 0.0;
        }
        (raw_score / max).clamp(0.0, 1.0)
    }

    pub fn learn_word(&mut self, word: &str, intent: &str, rate: f32) {
        if word.is_empty() {
            return;
        }
        let entries = self.word_intent.entry(word.to_string()).or_default();
        if let Some(e) = entries.iter_mut().find(|(id, _)| id == intent) {
            e.1 = (e.1 + rate * (1.0 - e.1)).min(1.0);
        } else {
            let new_intent = self.known_intents.insert(intent.to_string());
            if new_intent {
                self.intent_count += 1;
            }
            entries.push((intent.to_string(), rate));
            self.known_words.insert(word.to_string());
            self.intent_to_tokens
                .entry(intent.to_string())
                .or_default()
                .insert(word.to_string());
            if new_intent {
                self.rebuild_caches();
            } else {
                self.refresh_idf_for(word);
            }
        }
    }

    pub fn learn_phrase(&mut self, words: &[&str], intent: &str) {
        for word in words {
            self.learn_word(word, intent, Self::PHRASE_RATE);
        }
    }

    pub fn index_char_ngrams(&mut self, phrase: &str, intent: &str) {
        let normalized: String = phrase.to_lowercase();
        let s: String = format!(
            "  {}  ",
            normalized.split_whitespace().collect::<Vec<_>>().join(" ")
        );
        if s.chars().count() < 4 {
            return;
        }
        let chars: Vec<char> = s.chars().collect();
        let set = self.char_ngrams.entry(intent.to_string()).or_default();
        for window in chars.windows(4) {
            let ngram: String = window.iter().collect();
            set.insert(ngram);
        }
    }

    pub fn apply_char_ngram_tiebreaker(
        &self,
        query: &str,
        ranked: Vec<(String, f32)>,
        ratio_threshold: f32,
        alpha: f32,
    ) -> Vec<(String, f32)> {
        if ranked.len() < 2 {
            return ranked;
        }
        let top1 = ranked[0].1;
        let top2 = ranked[1].1;
        if top1 + top2 <= 0.0 {
            return ranked;
        }
        let ratio = top1 / (top1 + top2);
        if ratio >= ratio_threshold {
            return ranked;
        }
        let normalized: String = query.to_lowercase();
        let s: String = format!(
            "  {}  ",
            normalized.split_whitespace().collect::<Vec<_>>().join(" ")
        );
        if s.chars().count() < 4 {
            return ranked;
        }
        let chars: Vec<char> = s.chars().collect();
        let mut q_ngrams: FxHashSet<String> = FxHashSet::default();
        for window in chars.windows(4) {
            let ngram: String = window.iter().collect();
            q_ngrams.insert(ngram);
        }
        if q_ngrams.is_empty() {
            return ranked;
        }
        let k = ranked.len().min(5);
        let (head, tail) = ranked.split_at(k);
        let mut rescored: Vec<(String, f32)> = head
            .iter()
            .map(|(id, score)| {
                let intent_set = self.char_ngrams.get(id);
                let jaccard = match intent_set {
                    Some(iset) if !iset.is_empty() => {
                        let inter = q_ngrams.iter().filter(|n| iset.contains(*n)).count();
                        let uni = q_ngrams.len() + iset.len() - inter;
                        if uni == 0 {
                            0.0
                        } else {
                            inter as f32 / uni as f32
                        }
                    }
                    _ => 0.0,
                };
                (id.clone(), score + alpha * jaccard)
            })
            .collect();
        rescored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rescored.extend_from_slice(tail);
        rescored
    }

    pub fn reinforce_tokens(&mut self, words: &[&str], intent: &str) {
        for word in words {
            self.learn_word(word, intent, Self::LEARN_RATE);
        }
    }

    pub fn default_threshold(&self) -> f32 {
        0.3
    }

    pub fn default_gap(&self) -> f32 {
        1.5
    }

    pub fn reinforce(&mut self, words: &[&str], intent: &str, delta: f32) {
        for word in words {
            let entries = self.word_intent.entry(word.to_string()).or_default();
            if let Some(e) = entries.iter_mut().find(|(id, _)| id == intent) {
                if delta >= 0.0 {
                    e.1 = (e.1 + delta * (1.0 - e.1)).min(1.0);
                } else {
                    e.1 = (e.1 * (1.0 + delta)).max(0.0);
                }
            } else if delta > 0.0 {
                entries.push((intent.to_string(), delta.min(1.0)));
                self.refresh_idf_for(word);
            }
        }
    }

    pub fn fired_conjunction_indices(&self, words: &[&str]) -> Vec<usize> {
        let word_set: FxHashSet<&str> = words.iter().copied().collect();
        self.conjunctions
            .iter()
            .enumerate()
            .filter(|(_, rule)| rule.words.iter().all(|w| word_set.contains(w.as_str())))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn reinforce_conjunction(&mut self, idx: usize, delta: f32) {
        if let Some(rule) = self.conjunctions.get_mut(idx) {
            if delta >= 0.0 {
                rule.bonus = (rule.bonus + delta * (1.0 - rule.bonus)).min(1.0);
            } else {
                rule.bonus = (rule.bonus * (1.0 + delta)).max(0.0);
            }
        }
    }

    /// IDF-weighted 1-gram scoring.
    pub fn score(&self, normalized: &str) -> (Vec<(String, f32)>, bool) {
        const CJK_NEG: &[char] = &['不', '没', '别', '未'];
        let cjk_negated = normalized.chars().any(|c| CJK_NEG.contains(&c));
        let query_for_tokenize: std::borrow::Cow<str> = if cjk_negated {
            std::borrow::Cow::Owned(
                normalized
                    .chars()
                    .map(|c| if CJK_NEG.contains(&c) { ' ' } else { c })
                    .collect(),
            )
        } else {
            std::borrow::Cow::Borrowed(normalized)
        };

        let tokens = crate::tokenizer::tokenize(&query_for_tokenize);
        let mut scores: FxHashMap<String, f32> = FxHashMap::default();
        let mut has_negation = cjk_negated;

        let all_bases: FxHashSet<&str> = tokens
            .iter()
            .map(|t| t.strip_prefix("not_").unwrap_or(t.as_str()))
            .collect();

        for token in &tokens {
            let is_negated = token.starts_with("not_");
            let base = if is_negated {
                &token["not_".len()..]
            } else {
                token.as_str()
            };
            if is_negated {
                has_negation = true;
            }
            if let Some(activations) = self.word_intent.get(base) {
                let idf = self.idf(base);
                for (intent, weight) in activations {
                    let delta = weight * idf;
                    *scores.entry(intent.clone()).or_insert(0.0) +=
                        if is_negated { -delta } else { delta };
                }
            }
        }

        for rule in &self.conjunctions {
            if rule.words.iter().all(|w| all_bases.contains(w.as_str())) {
                *scores.entry(rule.intent.clone()).or_insert(0.0) += rule.bonus;
            }
        }

        let mut result: Vec<(String, f32)> = scores.into_iter().filter(|(_, s)| *s > 0.0).collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        (result, has_negation)
    }

    pub fn score_multi(
        &self,
        normalized: &str,
        threshold: f32,
        gap: f32,
    ) -> (Vec<(String, f32)>, bool) {
        let (results, neg, _trace) = self.score_multi_traced(normalized, threshold, gap, false);
        (results, neg)
    }

    pub fn score_multi_traced(
        &self,
        normalized: &str,
        threshold: f32,
        _gap: f32,
        with_trace: bool,
    ) -> (Vec<(String, f32)>, bool, Option<MultiIntentTrace>) {
        const GATE_RATIO: f32 = 0.55;
        const MAX_ROUNDS: usize = 3;

        const CJK_NEG: &[char] = &['不', '没', '别', '未'];
        let cjk_negated = normalized.chars().any(|c| CJK_NEG.contains(&c));
        let query_for_tokenize: std::borrow::Cow<str> = if cjk_negated {
            std::borrow::Cow::Owned(
                normalized
                    .chars()
                    .map(|c| if CJK_NEG.contains(&c) { ' ' } else { c })
                    .collect(),
            )
        } else {
            std::borrow::Cow::Borrowed(normalized)
        };

        let all_tokens: Vec<String> = crate::tokenizer::tokenize(&query_for_tokenize);
        let has_negation = cjk_negated || all_tokens.iter().any(|t| t.starts_with("not_"));

        let mut remaining: Vec<String> = all_tokens;
        let mut confirmed: Vec<(String, f32)> = Vec::new();
        let mut confirmed_ids: FxHashSet<String> = FxHashSet::default();
        let mut original_top: f32 = 0.0;
        let mut trace_rounds: Vec<RoundTrace> = Vec::new();
        let mut stop_reason: Option<String> = None;

        for round in 0..MAX_ROUNDS {
            let scored = self.score_tokens(&remaining, &confirmed_ids);
            if scored.is_empty() {
                if with_trace {
                    stop_reason = Some("no scores".into());
                }
                break;
            }

            let round_top = scored[0].1;
            if round == 0 {
                original_top = round_top;
            }
            if round_top < threshold {
                if with_trace {
                    stop_reason =
                        Some(format!("top {:.2} < threshold {:.2}", round_top, threshold));
                    trace_rounds.push(RoundTrace {
                        tokens_in: remaining.clone(),
                        scored: scored.iter().take(5).cloned().collect(),
                        confirmed: vec![],
                        consumed: vec![],
                    });
                }
                break;
            }
            if round > 0 && round_top < original_top * GATE_RATIO {
                if with_trace {
                    stop_reason = Some(format!(
                        "top {:.2} < gate {:.2}",
                        round_top,
                        original_top * GATE_RATIO
                    ));
                    trace_rounds.push(RoundTrace {
                        tokens_in: remaining.clone(),
                        scored: scored.iter().take(5).cloned().collect(),
                        confirmed: vec![],
                        consumed: vec![],
                    });
                }
                break;
            }

            let mut round_confirmed: Vec<(String, f32)> = Vec::new();
            for (intent, score) in &scored {
                if *score >= round_top * 0.90 && *score >= threshold {
                    round_confirmed.push((intent.clone(), *score));
                    confirmed_ids.insert(intent.clone());
                }
            }

            if round_confirmed.is_empty() {
                if with_trace {
                    stop_reason = Some("no confirmed".into());
                }
                break;
            }
            confirmed.extend(round_confirmed.iter().cloned());

            let tokens_before: Vec<String> = remaining.clone();
            remaining.retain(|token| {
                let base = token.strip_prefix("not_").unwrap_or(token.as_str());
                if let Some(activations) = self.word_intent.get(base) {
                    let best_intent = activations
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    match best_intent {
                        Some((id, _)) => !confirmed_ids.contains(id.as_str()),
                        None => true,
                    }
                } else {
                    true
                }
            });

            if with_trace {
                let remaining_set: FxHashSet<&String> = remaining.iter().collect();
                let consumed: Vec<String> = tokens_before
                    .iter()
                    .filter(|t| !remaining_set.contains(t))
                    .cloned()
                    .collect();
                trace_rounds.push(RoundTrace {
                    tokens_in: tokens_before,
                    scored: scored.iter().take(5).cloned().collect(),
                    confirmed: round_confirmed.iter().map(|(id, _)| id.clone()).collect(),
                    consumed,
                });
            }

            if remaining.is_empty() {
                if with_trace {
                    stop_reason = Some("all tokens consumed".into());
                }
                break;
            }
        }

        let trace = if with_trace {
            Some(MultiIntentTrace {
                rounds: trace_rounds,
                stop_reason: stop_reason.unwrap_or_else(|| "max rounds reached".into()),
            })
        } else {
            None
        };

        (confirmed, has_negation, trace)
    }

    fn score_tokens(
        &self,
        tokens: &[String],
        exclude_intents: &FxHashSet<String>,
    ) -> Vec<(String, f32)> {
        let mut scores: FxHashMap<String, f32> = FxHashMap::default();

        for token in tokens {
            let is_negated = token.starts_with("not_");
            let base = if is_negated {
                &token["not_".len()..]
            } else {
                token.as_str()
            };
            if let Some(activations) = self.word_intent.get(base) {
                let idf = self.idf(base);
                for (intent, weight) in activations {
                    if exclude_intents.contains(intent) {
                        continue;
                    }
                    let delta = weight * idf;
                    *scores.entry(intent.clone()).or_insert(0.0) +=
                        if is_negated { -delta } else { delta };
                }
            }
        }

        let all_bases: FxHashSet<&str> = tokens
            .iter()
            .map(|t| t.strip_prefix("not_").unwrap_or(t.as_str()))
            .collect();
        for rule in &self.conjunctions {
            if !exclude_intents.contains(&rule.intent)
                && rule.words.iter().all(|w| all_bases.contains(w.as_str()))
            {
                *scores.entry(rule.intent.clone()).or_insert(0.0) += rule.bonus;
            }
        }

        let mut result: Vec<(String, f32)> = scores.into_iter().filter(|(_, s)| *s > 0.0).collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(std::io::Error::other)
    }

    /// Resolve a query end-to-end: IDF scoring → token consumption →
    /// cross-provider disambiguation → disposition.
    pub fn resolve(&self, query: &str, threshold: f32, top_n: usize) -> RouteResult {
        let (raw, has_negation) = self.score(query);
        let ranked: Vec<(String, f32)> = raw.into_iter().take(top_n).collect();

        let (mut confirmed, _) = self.score_multi(query, threshold, 0.0);

        if confirmed.is_empty() {
            return RouteResult {
                confirmed: vec![],
                ranked,
                disposition: "no_match".to_string(),
                has_negation,
            };
        }

        if confirmed.len() > 1 {
            self.disambiguate_providers(&mut confirmed, query);
        }

        let top = confirmed[0].1;
        let disposition = if confirmed.len() >= 3 && confirmed[2].1 / top >= 0.75 {
            "escalate"
        } else if top < threshold * 2.0 {
            "low_confidence"
        } else {
            "confident"
        };

        RouteResult {
            confirmed,
            ranked,
            disposition: disposition.to_string(),
            has_negation,
        }
    }

    fn disambiguate_providers(&self, confirmed: &mut Vec<(String, f32)>, query: &str) {
        if confirmed.len() < 2 {
            return;
        }

        let mut action_groups: FxHashMap<String, Vec<usize>> = FxHashMap::default();
        for (i, (id, _)) in confirmed.iter().enumerate() {
            let action = id.split(':').nth(1).unwrap_or(id.as_str());
            action_groups.entry(action.to_string()).or_default().push(i);
        }

        let duplicate_groups: Vec<Vec<usize>> = action_groups
            .values()
            .filter(|g| g.len() > 1)
            .cloned()
            .collect();

        if duplicate_groups.is_empty() {
            return;
        }

        let tokens = crate::tokenizer::tokenize(query);
        let confirmed_ids: FxHashSet<&str> = confirmed.iter().map(|(id, _)| id.as_str()).collect();

        let mut unique_count: FxHashMap<&str, usize> = FxHashMap::default();
        for token in &tokens {
            let base = token.strip_prefix("not_").unwrap_or(token.as_str());
            if let Some(activations) = self.word_intent.get(base) {
                let matching: Vec<&str> = activations
                    .iter()
                    .filter(|(id, _)| confirmed_ids.contains(id.as_str()))
                    .map(|(id, _)| id.as_str())
                    .collect();
                if matching.len() == 1 {
                    *unique_count.entry(matching[0]).or_insert(0) += 1;
                }
            }
        }

        let mut to_remove: FxHashSet<usize> = FxHashSet::default();
        for group in &duplicate_groups {
            let best = group.iter().max_by_key(|&&i| {
                unique_count
                    .get(confirmed[i].0.as_str())
                    .copied()
                    .unwrap_or(0)
            });
            if let Some(&best_idx) = best {
                if unique_count
                    .get(confirmed[best_idx].0.as_str())
                    .copied()
                    .unwrap_or(0)
                    > 0
                {
                    for &i in group {
                        if i != best_idx {
                            to_remove.insert(i);
                        }
                    }
                }
            }
        }

        if !to_remove.is_empty() {
            let mut i = 0;
            confirmed.retain(|_| {
                let keep = !to_remove.contains(&i);
                i += 1;
                keep
            });
        }
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        let activation_edges: usize = self.word_intent.values().map(|v| v.len()).sum();
        (
            self.word_intent.len(),
            activation_edges,
            self.conjunctions.len(),
        )
    }
}
