//! Multi-intent routing with positional decomposition and relation detection.
//!
//! Decomposes a query containing multiple intents via greedy term consumption,
//! tracks character positions to recover the user's original ordering, and detects
//! relationships (sequential, conditional, negation) from gap text between spans.

use crate::index::{InvertedIndex, ScoredIntent};
use crate::tokenizer::PositionedTerm;
use crate::vector::LearnedVector;
use std::collections::{HashMap, HashSet};

/// A single detected intent from multi-intent routing.
#[derive(Debug, Clone)]
pub struct MultiRouteResult {
    /// The intent identifier.
    pub id: String,
    /// Match score from term accumulation.
    pub score: f32,
    /// First character position in original query (determines ordering).
    pub position: usize,
    /// Character position span `(start, end)` consumed by this intent.
    pub span: (usize, usize),
    /// Whether this is an action or context intent.
    pub intent_type: crate::IntentType,
    /// Detection confidence tier: "high" (dual-source), "medium" (paraphrase-only), "low" (routing-only).
    pub confidence: String,
    /// Detection source: "dual" (both indexes), "paraphrase" (phrase match only), "routing" (term match only).
    pub source: String,
    /// Whether this intent was negated in the query (e.g. "don't cancel" → cancel_order with negated=true).
    pub negated: bool,
}

/// Relationship between two consecutive detected intents.
#[derive(Debug, Clone, PartialEq)]
pub enum IntentRelation {
    /// "and", "also" — independent, any execution order.
    Parallel,
    /// "and then", "after that" — must execute in order.
    Sequential { first: usize, then: usize },
    /// "or", "otherwise" — second is fallback if first fails.
    Conditional { primary: usize, fallback: usize },
    /// "but first", "before that" — reverse the stated order.
    Reverse { stated_first: usize, execute_first: usize },
    /// "but don't", "except", "without" — exclude second intent.
    Negation { do_this: usize, not_this: usize },
}

/// Complete multi-intent routing output.
#[derive(Debug, Clone)]
pub struct MultiRouteOutput {
    /// Detected intents in positional order (left to right in user's message).
    pub intents: Vec<MultiRouteResult>,
    /// Relations between consecutive intent pairs (`intents[i]` → `intents[i+1]`).
    pub relations: Vec<IntentRelation>,
    /// Opaque metadata per detected intent, keyed by intent id.
    /// Populated by `Router::route_multi()` from configured metadata.
    pub metadata: HashMap<String, HashMap<String, Vec<String>>>,
    /// Suggested intents based on co-occurrence patterns — intents that frequently
    /// co-occur with detected intents but were NOT found in this query.
    /// Populated by `Router::route_multi()` from accumulated co-occurrence data.
    pub suggestions: Vec<crate::IntentSuggestion>,
}

impl MultiRouteOutput {
    /// Return only "strong" intents — those scoring >= `ratio` of the top intent.
    ///
    /// Useful for filtering noise from co-occurrence recording and display.
    /// Intents below the relative threshold are likely false positives from
    /// incidental term overlap (e.g., "order" matching apply_coupon).
    pub fn strong_intents(&self, ratio: f32) -> Vec<&MultiRouteResult> {
        let top = self.intents.iter().map(|i| i.score).fold(0.0f32, f32::max);
        if top <= 0.0 {
            return vec![];
        }
        let floor = top * ratio;
        self.intents.iter().filter(|i| i.score >= floor).collect()
    }
}

/// Run multi-intent greedy decomposition.
///
/// Algorithm:
/// 1. Receive positioned terms (from dual-path extraction)
/// 2. Score all intents against remaining terms
/// 3. Take highest-scoring intent, consume its matching terms
/// 4. Repeat until no intent scores above threshold
/// 5. Re-sort detected intents by position (user's original order)
/// 6. Detect relations from gap text between spans
pub(crate) fn route_multi(
    index: &InvertedIndex,
    vectors: &HashMap<String, LearnedVector>,
    positioned: Vec<PositionedTerm>,
    query_chars: Vec<char>,
    threshold: f32,
    max_intents: usize,
) -> MultiRouteOutput {
    if positioned.is_empty() {
        return MultiRouteOutput {
            intents: vec![],
            relations: vec![],
            metadata: HashMap::new(),
            suggestions: vec![],
        };
    }

    let mut remaining = positioned;
    let mut detected: Vec<MultiRouteResult> = Vec::new();
    let mut seen_intents: HashSet<String> = HashSet::new();

    // Term discrimination: a term is "defining" if it appears in a small
    // fraction of intents. Generic vocabulary (shared across many intents)
    // should not be enough to detect an intent in multi-intent decomposition.
    let n = index.intent_count();
    let disc_max_df = (n / 15).max(3);

    // Greedy consumption: find best intent, consume its terms, repeat
    loop {
        if remaining.is_empty() || detected.len() >= max_intents {
            break;
        }

        // Build search terms from remaining.
        // Strip not_ prefix for index search so negated terms still match intents.
        // The prefix is preserved in the positioned terms for negation tracking.
        let search_terms = build_search_terms(&remaining);
        let search_terms_stripped: Vec<String> = search_terms.iter()
            .map(|t| strip_negation_prefix(t))
            .collect();

        // Score all intents using stripped terms
        let mut results = index.search(&search_terms_stripped, 10);

        // Adjacency coherence: boost intents whose matching terms cluster
        // together in the query. Scattered matches get no boost.
        // Transparent for short queries (all terms adjacent → uniform boost).
        apply_adjacency_boost(&mut results, &remaining, vectors);

        // Find best intent we haven't already detected
        let best = results
            .iter()
            .find(|r| r.score >= threshold && !seen_intents.contains(&r.id));

        let best = match best {
            Some(b) => b,
            None => break,
        };

        // Get effective terms for the winning intent
        let effective = match vectors.get(&best.id) {
            Some(v) => v.effective_terms(),
            None => break,
        };

        // Identify which positioned terms would be consumed.
        // Match against effective terms using the stripped (un-negated) form.
        let consumed_indices: Vec<usize> = remaining.iter().enumerate()
            .filter(|(_, pt)| {
                let bare = strip_negation_prefix(&pt.term);
                effective.contains_key(&bare)
            })
            .map(|(i, _)| i)
            .collect();

        if consumed_indices.is_empty() {
            break;
        }

        // Discrimination check for secondary intents: at least one consumed
        // term must be "defining" — appearing in fewer than disc_max_df intents.
        // This prevents noise from generic vocabulary (e.g. "order", "have")
        // creating false detections after the strongest intent consumes its terms.
        if !detected.is_empty() {
            let has_defining_term = consumed_indices.iter().any(|&i| {
                let bare = strip_negation_prefix(&remaining[i].term);
                index.df(&bare) < disc_max_df
            });
            if !has_defining_term {
                // All consumed terms are generic — skip this intent
                seen_intents.insert(best.id.clone());
                continue;
            }
        }

        // Check if this intent is negated: majority of consumed content terms have not_ prefix
        let negated_count = consumed_indices.iter()
            .filter(|&&i| remaining[i].term.starts_with("not_"))
            .count();
        let is_negated = negated_count > 0 && negated_count >= consumed_indices.len() / 2;

        // Commit: consume matched terms, track positions
        let consumed_set: HashSet<usize> = consumed_indices.into_iter().collect();
        let mut consumed_offsets: Vec<(usize, usize)> = Vec::new();
        let mut new_remaining: Vec<PositionedTerm> = Vec::new();

        for (i, pt) in remaining.into_iter().enumerate() {
            if consumed_set.contains(&i) {
                consumed_offsets.push((pt.offset, pt.end_offset));
            } else {
                new_remaining.push(pt);
            }
        }

        let min_offset = consumed_offsets.iter().map(|(s, _)| *s).min().unwrap();
        let max_end = consumed_offsets.iter().map(|(_, e)| *e).max().unwrap();

        seen_intents.insert(best.id.clone());
        detected.push(MultiRouteResult {
            id: best.id.clone(),
            score: best.score,
            position: min_offset,
            span: (min_offset, max_end),
            intent_type: crate::IntentType::Action, // Default; overridden by Router::route_multi()
            confidence: "low".to_string(),
            source: "routing".to_string(),
            negated: is_negated,
        });

        remaining = new_remaining;
    }

    // Re-sort by position (user's original ordering)
    detected.sort_by_key(|d| d.position);

    // Detect relations from gap text between consecutive spans
    let relations = detect_relations(&detected, &query_chars);

    MultiRouteOutput {
        intents: detected,
        relations,
        metadata: HashMap::new(), // Populated by Router::route_multi()
        suggestions: vec![],      // Populated by Router::route_multi()
    }
}

/// Adjacency coherence boost: intents whose matching terms are positionally
/// clustered in the query get a score bonus. This helps long multi-intent
/// queries where scattered term overlap would otherwise create false signals.
///
/// For short queries, all competing intents see similar adjacency patterns,
/// so relative rankings are unchanged.
fn apply_adjacency_boost(
    results: &mut Vec<ScoredIntent>,
    remaining: &[PositionedTerm],
    vectors: &HashMap<String, LearnedVector>,
) {
    if remaining.len() < 3 {
        return;
    }

    // Sort remaining by position for adjacency checks
    let mut by_pos: Vec<&PositionedTerm> = remaining.iter().collect();
    by_pos.sort_by_key(|pt| pt.offset);

    for r in results.iter_mut() {
        let effective = match vectors.get(&r.id) {
            Some(v) => v.effective_terms(),
            None => continue,
        };

        // Find which positional slots match this intent
        let matching_seq: Vec<usize> = by_pos
            .iter()
            .enumerate()
            .filter(|(_, pt)| {
                let bare = strip_negation_prefix(&pt.term);
                effective.contains_key(&bare)
            })
            .map(|(i, _)| i)
            .collect();

        if matching_seq.len() < 2 {
            continue;
        }

        // Count adjacent pairs among matching positions
        let adjacent = matching_seq
            .windows(2)
            .filter(|w| w[1] - w[0] == 1)
            .count();

        let max_pairs = matching_seq.len() - 1;
        let ratio = adjacent as f32 / max_pairs as f32;

        // Up to 30% boost for fully clustered matches
        r.score *= 1.0 + 0.3 * ratio;
    }

    // Re-sort by adjusted scores
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Strip the `not_` prefix from a negated term, returning the bare form.
/// Non-negated terms are returned as-is.
fn strip_negation_prefix(term: &str) -> String {
    term.strip_prefix("not_").unwrap_or(term).to_string()
}

/// Check if a term carries a negation prefix.
pub fn is_negated_term(term: &str) -> bool {
    term.starts_with("not_")
}

/// Build search terms (unigrams + adjacency bigrams) from remaining positioned terms.
fn build_search_terms(remaining: &[PositionedTerm]) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // Unigrams
    for pt in remaining {
        if seen.insert(pt.term.clone()) {
            terms.push(pt.term.clone());
        }
    }

    // Bigrams from position-adjacent Latin terms only (CJK terms are already complete units)
    let mut sorted: Vec<&PositionedTerm> = remaining.iter().collect();
    sorted.sort_by_key(|pt| pt.offset);

    for window in sorted.windows(2) {
        // Only create bigrams between Latin (non-CJK) terms within 25 chars
        if !window[0].is_cjk && !window[1].is_cjk
            && window[1].offset.saturating_sub(window[0].end_offset) <= 25
        {
            let bigram = format!("{} {}", window[0].term, window[1].term);
            if seen.insert(bigram.clone()) {
                terms.push(bigram);
            }
        }
    }

    terms
}

/// Public entry point for relation detection (used by segmented route_multi).
pub fn detect_relations_public(
    intents: &[MultiRouteResult],
    query_chars: &[char],
) -> Vec<IntentRelation> {
    detect_relations(intents, query_chars)
}

/// Detect relations between consecutive intents by examining gap text.
fn detect_relations(
    intents: &[MultiRouteResult],
    query_chars: &[char],
) -> Vec<IntentRelation> {
    let mut relations = Vec::new();

    for i in 0..intents.len().saturating_sub(1) {
        let gap_start = intents[i].span.1;
        let gap_end = intents[i + 1].span.0;

        if gap_start >= gap_end || gap_end > query_chars.len() {
            relations.push(IntentRelation::Parallel);
            continue;
        }

        let gap_text: String = query_chars[gap_start..gap_end].iter().collect();
        relations.push(classify_relation(&gap_text, i));
    }

    relations
}

/// Classify the relationship between two intents from their gap text.
fn classify_relation(gap: &str, index: usize) -> IntentRelation {
    let g = gap.to_lowercase();

    // Sequential: "then", "after", "next", "followed by", "once done"
    // CJK: 然后 (then), 之后 (after)
    if has_word(&g, "then") || has_word(&g, "after") || has_word(&g, "next")
        || g.contains("followed by") || g.contains("once done")
        || g.contains("然后") || g.contains("之后")
    {
        return IntentRelation::Sequential {
            first: index,
            then: index + 1,
        };
    }

    // Conditional: "or", "otherwise", "if not", "failing that"
    // CJK: 或者 (or/otherwise)
    if has_word(&g, "otherwise") || g.contains("if not") || g.contains("failing that")
        || has_word(&g, "or")
        || g.contains("或者")
    {
        return IntentRelation::Conditional {
            primary: index,
            fallback: index + 1,
        };
    }

    // Reverse: "but first", "before that"
    if g.contains("but first") || g.contains("before that") || g.contains("before this") {
        return IntentRelation::Reverse {
            stated_first: index,
            execute_first: index + 1,
        };
    }

    // Negation: "not" (from expanded "don't"), "except", "without", "never"
    // CJK: 但不 (but not)
    if has_word(&g, "not") || has_word(&g, "except") || has_word(&g, "without")
        || has_word(&g, "never")
        || g.contains("但不")
    {
        return IntentRelation::Negation {
            do_this: index,
            not_this: index + 1,
        };
    }

    // Default: independent/parallel
    IntentRelation::Parallel
}

/// Check if a word appears as a whole word in text.
fn has_word(text: &str, word: &str) -> bool {
    text.split_whitespace().any(|w| w == word)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Router;

    fn setup_router() -> Router {
        let mut router = Router::new();
        router.add_intent("cancel_order", &[
            "cancel my order", "cancel order", "stop my order",
        ]);
        router.add_intent("track_order", &[
            "track my order", "where is my package", "shipping status",
        ]);
        router.add_intent("check_credit", &[
            "check store credit", "store credit balance", "how much credit",
        ]);
        router.add_intent("refund", &[
            "get a refund", "money back", "refund my purchase",
        ]);
        router.add_intent("delete_profile", &[
            "delete my profile", "remove my account", "erase profile",
        ]);
        router.add_intent("close_account", &[
            "close my account", "shut down account", "deactivate account",
        ]);
        router.add_intent("transfer_balance", &[
            "transfer my balance", "move my balance", "send balance",
        ]);
        router
    }

    #[test]
    fn single_intent_still_works() {
        let router = setup_router();
        let result = router.route_multi("cancel my order", 0.5);
        assert_eq!(result.intents.len(), 1);
        assert_eq!(result.intents[0].id, "cancel_order");
        assert!(result.relations.is_empty());
    }

    #[test]
    fn two_intents_detected() {
        let router = setup_router();
        let result = router.route_multi(
            "cancel my order and also track my package",
            0.3,
        );
        assert!(result.intents.len() >= 2);
        let ids: Vec<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        assert!(ids.contains(&"cancel_order"), "missing cancel_order in {:?}", ids);
        assert!(ids.contains(&"track_order"), "missing track_order in {:?}", ids);
    }

    #[test]
    fn positional_ordering() {
        let router = setup_router();
        let result = router.route_multi(
            "cancel my order and also check store credit",
            0.3,
        );
        let cancel_idx = result.intents.iter().position(|i| i.id == "cancel_order");
        let credit_idx = result.intents.iter().position(|i| i.id == "check_credit");
        if let (Some(ci), Some(cri)) = (cancel_idx, credit_idx) {
            assert!(ci < cri, "cancel_order (idx {}) should appear before check_credit (idx {})", ci, cri);
        }
    }

    #[test]
    fn three_intents_decomposition() {
        let router = setup_router();
        let result = router.route_multi(
            "cancel my order and track the package and check store credit",
            0.3,
        );
        let ids: Vec<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        assert!(ids.contains(&"cancel_order"), "missing cancel_order in {:?}", ids);
        assert!(ids.contains(&"track_order"), "missing track_order in {:?}", ids);
        assert!(ids.contains(&"check_credit"), "missing check_credit in {:?}", ids);

        // Verify positional ordering
        for i in 0..result.intents.len() - 1 {
            assert!(
                result.intents[i].position <= result.intents[i + 1].position,
                "intents not in positional order"
            );
        }
    }

    #[test]
    fn sequential_relation() {
        let router = setup_router();
        let result = router.route_multi(
            "transfer my balance then close my account",
            0.3,
        );
        if result.intents.len() >= 2 && !result.relations.is_empty() {
            assert!(
                matches!(result.relations[0], IntentRelation::Sequential { .. }),
                "expected Sequential, got {:?}", result.relations[0]
            );
        }
    }

    #[test]
    fn negation_relation() {
        let router = setup_router();
        // Use cancel_order + delete_profile to avoid shared "account" term
        let result = router.route_multi(
            "cancel my order but don't delete my profile",
            0.3,
        );
        assert!(
            result.intents.len() >= 2,
            "expected 2+ intents, got {}: {:?}",
            result.intents.len(),
            result.intents.iter().map(|i| &i.id).collect::<Vec<_>>()
        );
        assert!(
            !result.relations.is_empty(),
            "expected relations, got none"
        );
        assert!(
            matches!(result.relations[0], IntentRelation::Negation { .. }),
            "expected Negation, got {:?}", result.relations[0]
        );
    }

    #[test]
    fn conditional_relation() {
        let router = setup_router();
        let result = router.route_multi(
            "get a refund or otherwise check store credit",
            0.3,
        );
        if result.intents.len() >= 2 && !result.relations.is_empty() {
            assert!(
                matches!(result.relations[0], IntentRelation::Conditional { .. }),
                "expected Conditional, got {:?}", result.relations[0]
            );
        }
    }

    #[test]
    fn parallel_relation_default() {
        let router = setup_router();
        let result = router.route_multi(
            "cancel my order and also check store credit",
            0.3,
        );
        if result.intents.len() >= 2 && !result.relations.is_empty() {
            assert!(
                matches!(result.relations[0], IntentRelation::Parallel),
                "expected Parallel, got {:?}", result.relations[0]
            );
        }
    }

    #[test]
    fn below_threshold_stops() {
        let router = setup_router();
        let result = router.route_multi("cancel my order", 100.0);
        assert!(result.intents.is_empty());
    }

    #[test]
    fn empty_query_returns_empty() {
        let router = setup_router();
        let result = router.route_multi("", 0.5);
        assert!(result.intents.is_empty());
    }

    #[test]
    fn all_stop_words_returns_empty() {
        let router = setup_router();
        let result = router.route_multi("the a an in on at to of", 0.5);
        assert!(result.intents.is_empty());
    }

    #[test]
    fn spans_are_valid() {
        let router = setup_router();
        let result = router.route_multi(
            "cancel my order and check store credit",
            0.3,
        );
        for intent in &result.intents {
            assert!(intent.span.0 <= intent.span.1);
            assert_eq!(intent.position, intent.span.0);
        }
    }

    #[test]
    fn no_duplicate_intents() {
        let router = setup_router();
        let result = router.route_multi(
            "cancel my order and cancel my order",
            0.3,
        );
        let ids: Vec<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();
        let unique: HashSet<&&str> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len(), "duplicate intents detected: {:?}", ids);
    }

    // --- relation detection unit tests ---

    #[test]
    fn classify_sequential_patterns() {
        assert!(matches!(
            classify_relation("and then", 0),
            IntentRelation::Sequential { first: 0, then: 1 }
        ));
        assert!(matches!(
            classify_relation("after that", 0),
            IntentRelation::Sequential { .. }
        ));
        assert!(matches!(
            classify_relation("next", 0),
            IntentRelation::Sequential { .. }
        ));
    }

    #[test]
    fn classify_conditional_patterns() {
        assert!(matches!(
            classify_relation("or", 0),
            IntentRelation::Conditional { primary: 0, fallback: 1 }
        ));
        assert!(matches!(
            classify_relation("otherwise", 0),
            IntentRelation::Conditional { .. }
        ));
        assert!(matches!(
            classify_relation("if not", 0),
            IntentRelation::Conditional { .. }
        ));
    }

    #[test]
    fn classify_negation_patterns() {
        assert!(matches!(
            classify_relation("but do not", 0),
            IntentRelation::Negation { do_this: 0, not_this: 1 }
        ));
        assert!(matches!(
            classify_relation("except", 0),
            IntentRelation::Negation { .. }
        ));
        assert!(matches!(
            classify_relation("without", 0),
            IntentRelation::Negation { .. }
        ));
    }

    #[test]
    fn classify_reverse_patterns() {
        assert!(matches!(
            classify_relation("but first", 0),
            IntentRelation::Reverse { stated_first: 0, execute_first: 1 }
        ));
        assert!(matches!(
            classify_relation("before that", 0),
            IntentRelation::Reverse { .. }
        ));
    }

    #[test]
    fn classify_parallel_default() {
        assert!(matches!(
            classify_relation("and also", 0),
            IntentRelation::Parallel
        ));
        assert!(matches!(
            classify_relation("and", 0),
            IntentRelation::Parallel
        ));
    }

    #[test]
    fn classify_cjk_sequential() {
        assert!(matches!(
            classify_relation("然后", 0),
            IntentRelation::Sequential { first: 0, then: 1 }
        ));
    }

    #[test]
    fn classify_cjk_conditional() {
        assert!(matches!(
            classify_relation("或者", 0),
            IntentRelation::Conditional { primary: 0, fallback: 1 }
        ));
    }

    // --- adjacency boost tests ---

    #[test]
    fn adjacency_boost_long_confused_query() {
        // Scenario: user mentions "order food delivery" early in the message
        // but actually wants to cancel an order and track a delivery.
        // Without adjacency boost, "order_food" could outscore "cancel_order"
        // because it matches scattered terms across the whole query.
        let mut router = Router::new();
        router.add_intent("order_food", &[
            "order food", "order meal", "food delivery", "order pizza",
        ]);
        router.add_intent("cancel_order", &[
            "cancel order", "cancel my purchase", "stop my order",
        ]);
        router.add_intent("track_delivery", &[
            "track delivery", "delivery status", "where is delivery",
        ]);

        let query = "I ordered some food yesterday and the delivery has not \
                      arrived so I want to cancel my order and track the delivery";
        let result = router.route_multi(query, 0.3);
        let ids: Vec<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();

        // cancel_order and track_delivery should be detected (user's actual intent)
        assert!(ids.contains(&"cancel_order"),
            "should detect cancel_order in {:?}", ids);
        assert!(ids.contains(&"track_delivery"),
            "should detect track_delivery in {:?}", ids);
    }

    #[test]
    fn adjacency_boost_scattered_vs_clustered() {
        // "account" and "order" appear in multiple intents.
        // The user wants to cancel their order and close their account.
        // Terms for each intent cluster together in the query.
        let mut router = Router::new();
        router.add_intent("cancel_order", &[
            "cancel order", "cancel my order", "stop order",
        ]);
        router.add_intent("close_account", &[
            "close account", "close my account", "delete account",
        ]);
        router.add_intent("check_order", &[
            "check order", "order status", "order details",
        ]);

        let query = "I am so frustrated right now I need to cancel my order \
                      immediately and then I also want to close my account \
                      because your service is terrible";
        let result = router.route_multi(query, 0.3);
        let ids: Vec<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();

        assert!(ids.contains(&"cancel_order"),
            "should detect cancel_order in {:?}", ids);
        assert!(ids.contains(&"close_account"),
            "should detect close_account in {:?}", ids);
        // check_order should NOT appear — "order" is near "cancel", not "check"
        assert!(!ids.contains(&"check_order"),
            "should NOT detect check_order in {:?}", ids);
    }

    #[test]
    fn adjacency_boost_four_intents_long_ramble() {
        let mut router = Router::new();
        router.add_intent("cancel_order", &[
            "cancel my order", "cancel order", "stop my order",
        ]);
        router.add_intent("track_order", &[
            "track my order", "where is my package", "shipping status",
        ]);
        router.add_intent("refund", &[
            "get a refund", "money back", "refund my purchase",
        ]);
        router.add_intent("contact_service", &[
            "talk to someone", "speak to agent", "customer service",
            "talk to a human",
        ]);

        let query = "this is absolutely ridiculous I have been waiting for \
                      two weeks and I need to cancel my order because it is \
                      taking forever and I also want to track my other package \
                      that should have arrived and I want a refund for the \
                      broken item and honestly I just want to talk to someone \
                      who can actually help me with this terrible service";
        let result = router.route_multi(query, 0.3);
        let ids: Vec<&str> = result.intents.iter().map(|i| i.id.as_str()).collect();

        assert!(ids.contains(&"cancel_order"),
            "should detect cancel_order in {:?}", ids);
        assert!(ids.contains(&"track_order"),
            "should detect track_order in {:?}", ids);
        assert!(ids.contains(&"refund"),
            "should detect refund in {:?}", ids);
        assert!(ids.contains(&"contact_service"),
            "should detect contact_service in {:?}", ids);
    }

    #[test]
    fn adjacency_boost_short_query_unchanged() {
        // Short queries should work exactly as before
        let router = setup_router();
        let result = router.route_multi("cancel my order", 0.5);
        assert_eq!(result.intents.len(), 1);
        assert_eq!(result.intents[0].id, "cancel_order");
    }

    #[test]
    fn adjacency_boost_comparison() {
        // Side-by-side: index.search (no boost) vs route_multi (with boost)
        // Shows how adjacency changes scoring for long queries
        let mut router = Router::new();
        router.add_intent("order_food", &[
            "order food", "order meal", "food delivery", "order pizza",
        ]);
        router.add_intent("cancel_order", &[
            "cancel order", "cancel my purchase", "stop my order",
        ]);
        router.add_intent("track_delivery", &[
            "track delivery", "delivery status", "where is delivery",
        ]);

        let query = "I ordered some food yesterday and the delivery has not \
                      arrived so I want to cancel my order and track the delivery";

        // Raw index scores (no adjacency boost)
        let terms = router.extract_terms(query);
        let raw = router.route(query);
        let raw_ids: Vec<(&str, f32)> = raw.iter()
            .map(|r| (r.id.as_str(), r.score)).collect();

        // Full route_multi (with adjacency boost)
        let multi = router.route_multi(query, 0.1);
        let multi_ids: Vec<(&str, f32)> = multi.intents.iter()
            .map(|i| (i.id.as_str(), i.score)).collect();

        println!("\n  Query: {}", query);
        println!("  Terms: {:?}", terms);
        println!("\n  RAW index scores (no adjacency):");
        for (id, score) in &raw_ids {
            println!("    {:<20} {:.3}", id, score);
        }
        println!("\n  MULTI route (with adjacency boost):");
        for (id, score) in &multi_ids {
            println!("    {:<20} {:.3}", id, score);
        }

        // cancel_order and track_delivery should be detected, NOT order_food
        assert!(multi_ids.iter().any(|(id, _)| *id == "cancel_order"),
            "cancel_order should be in multi results");
        assert!(multi_ids.iter().any(|(id, _)| *id == "track_delivery"),
            "track_delivery should be in multi results");
    }

    #[test]
    fn adjacency_boost_edge_cases() {
        let mut router = Router::new();
        router.add_intent("cancel_order", &[
            "cancel my order", "cancel order", "stop my order",
        ]);
        router.add_intent("track_order", &[
            "track my order", "where is my package", "shipping status",
        ]);
        router.add_intent("refund", &[
            "get a refund", "money back", "refund my purchase",
        ]);
        router.add_intent("close_account", &[
            "close my account", "delete account", "deactivate account",
        ]);
        router.add_intent("change_address", &[
            "change my address", "update shipping address", "new address",
        ]);
        router.add_intent("billing", &[
            "billing issue", "charged twice", "wrong charge", "payment problem",
        ]);

        let cases: Vec<(&str, Vec<&str>, Vec<&str>)> = vec![
            // (query, must_contain, must_not_contain)

            // Edge 1: negation + adjacency — "don't cancel" should still detect cancel as negated
            ("I want to track my order but don't cancel it",
             vec!["track_order", "cancel_order"], vec![]),

            // Edge 2: single word query — should not crash or behave weird
            ("refund", vec!["refund"], vec![]),

            // Edge 3: 5 intents in one ramble (max_intents default = 5)
            ("I need to cancel my order and track my other package and \
              get a refund and close my account and change my address",
             vec!["cancel_order", "track_order", "refund", "close_account", "change_address"], vec![]),

            // Edge 4: repeated word "order" in different contexts
            ("cancel my first order and track my second order",
             vec!["cancel_order", "track_order"], vec![]),

            // Edge 5: filler-heavy — lots of noise words between intent terms
            ("so basically what happened is that I really really need to \
              like cancel the order I placed and also maybe get a refund \
              if that is even possible at this point",
             vec!["cancel_order", "refund"], vec![]),

            // Edge 6: adjacent terms for WRONG intent — "close order" not a real intent
            ("I want to close my account",
             vec!["close_account"], vec!["cancel_order"]),
        ];

        println!("\n  ═══ ADJACENCY BOOST EDGE CASE PROBES ═══\n");
        let mut pass = 0;
        let mut fail = 0;

        for (i, (query, must, must_not)) in cases.iter().enumerate() {
            let result = router.route_multi(query, 0.3);
            let ids: Vec<&str> = result.intents.iter().map(|r| r.id.as_str()).collect();
            let scores: Vec<String> = result.intents.iter()
                .map(|r| format!("{}({:.2})", r.id, r.score)).collect();

            let mut ok = true;
            for &m in must {
                if !ids.contains(&m) {
                    println!("  FAIL #{}: missing '{}' in {:?}", i+1, m, scores);
                    ok = false;
                }
            }
            for &m in must_not {
                if ids.contains(&m) {
                    println!("  FAIL #{}: unwanted '{}' in {:?}", i+1, m, scores);
                    ok = false;
                }
            }
            if ok {
                println!("  OK   #{}: {} → {:?}", i+1, &query[..query.len().min(60)], scores);
                pass += 1;
            } else {
                fail += 1;
            }
        }

        println!("\n  Results: {}/{} passed\n", pass, pass + fail);
        assert_eq!(fail, 0, "{} edge cases failed", fail);
    }
}
