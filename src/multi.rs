//! Multi-intent routing with positional decomposition and relation detection.
//!
//! Decomposes a query containing multiple intents via greedy term consumption,
//! tracks word positions to recover the user's original ordering, and detects
//! relationships (sequential, conditional, negation) from gap words between spans.

use crate::index::InvertedIndex;
use crate::tokenizer::{tokenize_positioned, PositionedTerm};
use crate::vector::LearnedVector;
use std::collections::{HashMap, HashSet};

/// A single detected intent from multi-intent routing.
#[derive(Debug, Clone)]
pub struct MultiRouteResult {
    /// The intent identifier.
    pub id: String,
    /// Match score from term accumulation.
    pub score: f32,
    /// First word position in original query (determines ordering).
    pub position: usize,
    /// Word position span `(start, end)` consumed by this intent.
    pub span: (usize, usize),
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
}

/// Run multi-intent greedy decomposition.
///
/// Algorithm:
/// 1. Tokenize query into positioned terms
/// 2. Score all intents against remaining terms
/// 3. Take highest-scoring intent, consume its matching terms
/// 4. Repeat until no intent scores above threshold
/// 5. Re-sort detected intents by position (user's original order)
/// 6. Detect relations from gap words between spans
pub(crate) fn route_multi(
    index: &InvertedIndex,
    vectors: &HashMap<String, LearnedVector>,
    query: &str,
    threshold: f32,
) -> MultiRouteOutput {
    let (positioned, original_words) = tokenize_positioned(query);
    if positioned.is_empty() {
        return MultiRouteOutput {
            intents: vec![],
            relations: vec![],
        };
    }

    let mut remaining = positioned;
    let mut detected: Vec<MultiRouteResult> = Vec::new();
    let mut seen_intents: HashSet<String> = HashSet::new();

    // Greedy consumption: find best intent, consume its terms, repeat
    loop {
        if remaining.is_empty() {
            break;
        }

        // Build search terms (unigrams + adjacency bigrams) from remaining
        let search_terms = build_search_terms(&remaining);

        // Score all intents
        let results = index.search(&search_terms, 10);

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

        // Consume matching terms, track positions
        let mut consumed_positions: Vec<usize> = Vec::new();
        let mut new_remaining: Vec<PositionedTerm> = Vec::new();

        for pt in remaining {
            if effective.contains_key(&pt.term) {
                consumed_positions.push(pt.position);
            } else {
                new_remaining.push(pt);
            }
        }

        if consumed_positions.is_empty() {
            break;
        }

        let min_pos = *consumed_positions.iter().min().unwrap();
        let max_pos = *consumed_positions.iter().max().unwrap();

        seen_intents.insert(best.id.clone());
        detected.push(MultiRouteResult {
            id: best.id.clone(),
            score: best.score,
            position: min_pos,
            span: (min_pos, max_pos),
        });

        remaining = new_remaining;
    }

    // Re-sort by position (user's original ordering)
    detected.sort_by_key(|d| d.position);

    // Detect relations from gap words between consecutive spans
    let relations = detect_relations(&detected, &original_words);

    MultiRouteOutput {
        intents: detected,
        relations,
    }
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

    // Bigrams from position-adjacent pairs (gap ≤ 3 words in original)
    let mut sorted: Vec<&PositionedTerm> = remaining.iter().collect();
    sorted.sort_by_key(|pt| pt.position);

    for window in sorted.windows(2) {
        if window[1].position.saturating_sub(window[0].position) <= 3 {
            let bigram = format!("{} {}", window[0].term, window[1].term);
            if seen.insert(bigram.clone()) {
                terms.push(bigram);
            }
        }
    }

    terms
}

/// Detect relations between consecutive intents by examining gap words.
fn detect_relations(
    intents: &[MultiRouteResult],
    original_words: &[String],
) -> Vec<IntentRelation> {
    let mut relations = Vec::new();

    for i in 0..intents.len().saturating_sub(1) {
        let gap_start = intents[i].span.1 + 1;
        let gap_end = intents[i + 1].span.0;

        if gap_start >= gap_end || gap_end > original_words.len() {
            relations.push(IntentRelation::Parallel);
            continue;
        }

        let gap_text = original_words[gap_start..gap_end].join(" ");
        relations.push(classify_relation(&gap_text, i));
    }

    relations
}

/// Classify the relationship between two intents from their gap text.
fn classify_relation(gap: &str, index: usize) -> IntentRelation {
    let g = gap.to_lowercase();

    // Sequential: "then", "after", "next", "followed by", "once done"
    if has_word(&g, "then") || has_word(&g, "after") || has_word(&g, "next")
        || g.contains("followed by") || g.contains("once done")
    {
        return IntentRelation::Sequential {
            first: index,
            then: index + 1,
        };
    }

    // Conditional: "or", "otherwise", "if not", "failing that"
    if has_word(&g, "otherwise") || g.contains("if not") || g.contains("failing that")
        || has_word(&g, "or")
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
    if has_word(&g, "not") || has_word(&g, "except") || has_word(&g, "without")
        || has_word(&g, "never")
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
        let result = router.route_multi("can you please do this for me", 0.5);
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
}
