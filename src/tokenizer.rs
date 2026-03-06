//! Tokenizer — query and training phrase processing.
//!
//! Produces unigrams + bigrams from natural language, filtering stop words.
//! Also converts training phrases into term-weight maps for seeding intents.

use std::collections::{HashMap, HashSet};

/// Expand common English contractions to prevent garbage tokens from apostrophe splitting.
/// "don't" → "do not", "can't" → "can not", "what's" → "what", etc.
fn expand_contractions(text: &str) -> String {
    // Normalize unicode right single quotation mark to ASCII apostrophe
    let text = text.replace('\u{2019}', "'");

    // Irregular contractions (must come before general n't rule)
    let text = text.replace("won't", "will not");
    let text = text.replace("can't", "can not");
    let text = text.replace("shan't", "shall not");

    // Regular n't → " not"
    let text = text.replace("n't", " not");

    // Other contractions → space (expansions are all stop words anyway)
    let text = text.replace("'re", " ");
    let text = text.replace("'ve", " ");
    let text = text.replace("'ll", " ");
    let text = text.replace("'d", " ");
    let text = text.replace("'m", " ");
    let text = text.replace("'s", " ");

    text
}

/// Detect word positions that are negated (should be suppressed from scoring).
///
/// Conservative: only negates after "do not" (from expanded "don't") and
/// explicit negation words ("never", "without", "except"). Does NOT negate
/// after "can not" (can't = inability) or "is not" (isn't = state).
fn find_negated_positions(words: &[&str], stop_set: &HashSet<&str>) -> HashSet<usize> {
    let mut negated: HashSet<usize> = HashSet::new();
    let mut negate_next = 0u8; // counter: negate next N content words

    for (i, &word) in words.iter().enumerate() {
        // "do not" → true intent negation (from expanded "don't")
        if word == "not" && i > 0 && words[i - 1] == "do" {
            negate_next = 1;
            continue;
        }

        // Standalone negation words
        if word == "never" || word == "without" || word == "except" {
            negate_next = 1;
            continue;
        }

        // Clause boundaries reset negation
        if word == "and" || word == "but" || word == "or" || word == "then" {
            negate_next = 0;
            continue;
        }

        // Negation scope counts all words (stop words absorb it too).
        // Only content words actually get marked as negated.
        if negate_next > 0 {
            if !stop_set.contains(word) {
                negated.insert(i);
            }
            negate_next -= 1;
        }
    }

    negated
}

/// Common English words that don't carry intent signal.
const STOP_WORDS: &[&str] = &[
    // Articles and pronouns
    "a", "an", "the", "i", "my", "me", "we", "our", "you", "your",
    "it", "its", "he", "she", "they", "them",
    // Prepositions
    "in", "on", "at", "to", "for", "of", "with", "from", "by", "as",
    // Conjunctions
    "and", "or", "but", "if", "then", "so",
    // Conversational filler
    "please", "can", "could", "would", "want", "need", "like",
    "just", "also", "about", "how", "what", "which", "that", "this",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "will", "shall", "should", "may", "might", "must",
    "not", "no", "yes", "all", "some", "any", "each", "every",
];

/// Tokenize a query into searchable terms (unigrams + bigrams).
///
/// ```
/// use asv_router::tokenizer::tokenize;
///
/// let terms = tokenize("charge my credit card");
/// assert!(terms.contains(&"charge".to_string()));
/// assert!(terms.contains(&"credit".to_string()));
/// assert!(terms.contains(&"card".to_string()));
/// assert!(terms.contains(&"credit card".to_string()));
/// assert!(!terms.contains(&"my".to_string())); // stop word
/// ```
pub fn tokenize(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    let expanded = expand_contractions(&lower);

    let words: Vec<&str> = expanded
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| !w.is_empty())
        .collect();

    let stop_set: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    let negated = find_negated_positions(&words, &stop_set);

    let mut terms: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // Unigrams (excluding stop words and negated terms)
    for (i, word) in words.iter().enumerate() {
        if !stop_set.contains(word) && !negated.contains(&i) {
            let w = word.to_string();
            if seen.insert(w.clone()) {
                terms.push(w);
            }
        }
    }

    // Bigrams (consecutive non-stop, non-negated word pairs)
    let non_stop: Vec<&str> = words.iter()
        .enumerate()
        .filter(|(i, w)| !stop_set.contains(**w) && !negated.contains(i))
        .map(|(_, w)| *w)
        .collect();

    for window in non_stop.windows(2) {
        let bigram = format!("{} {}", window[0], window[1]);
        if seen.insert(bigram.clone()) {
            terms.push(bigram);
        }
    }

    terms
}

/// Convert training phrases into term weights.
///
/// Weight formula: `0.3 + 0.65 * (term_frequency / max_frequency)`, capped at 0.95.
/// Terms appearing in more training phrases get higher weights.
///
/// ```
/// use asv_router::tokenizer::training_to_terms;
///
/// let terms = training_to_terms(&[
///     "cancel my order".to_string(),
///     "I want to cancel".to_string(),
///     "stop my order".to_string(),
/// ]);
/// // "cancel" appears in 2/3, "order" in 2/3, "stop" in 1/3
/// assert!(terms["cancel"] > terms["stop"]);
/// ```
pub fn training_to_terms(queries: &[String]) -> HashMap<String, f32> {
    if queries.is_empty() {
        return HashMap::new();
    }

    let mut term_counts: HashMap<String, u32> = HashMap::new();

    for query in queries {
        let tokens = tokenize(query);
        for token in tokens {
            *term_counts.entry(token).or_insert(0) += 1;
        }
    }

    if term_counts.is_empty() {
        return HashMap::new();
    }

    let max_count = *term_counts.values().max().unwrap_or(&1);

    term_counts
        .into_iter()
        .map(|(term, count)| {
            let weight = (0.3 + 0.65 * (count as f32 / max_count as f32)).min(0.95);
            (term, (weight * 100.0).round() / 100.0)
        })
        .collect()
}

/// A term with its position in the original query.
#[derive(Debug, Clone)]
pub struct PositionedTerm {
    /// The term (lowercase, after stop word removal).
    pub term: String,
    /// Word index in the original query's word array.
    pub position: usize,
}

/// Tokenize with position tracking for multi-intent decomposition.
///
/// Unlike `tokenize()`, this preserves duplicate terms at different positions
/// so each occurrence can be consumed independently by different intents.
///
/// Returns `(positioned_terms, all_words)` where `all_words` includes stop words
/// (used for gap analysis in relation detection).
pub fn tokenize_positioned(query: &str) -> (Vec<PositionedTerm>, Vec<String>) {
    let lower = query.to_lowercase();
    let expanded = expand_contractions(&lower);

    let words: Vec<&str> = expanded
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| !w.is_empty())
        .collect();

    let stop_set: HashSet<&str> = STOP_WORDS.iter().copied().collect();
    let original_words: Vec<String> = words.iter().map(|w| w.to_string()).collect();

    let positioned: Vec<PositionedTerm> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| !stop_set.contains(**w))
        .map(|(pos, w)| PositionedTerm {
            term: w.to_string(),
            position: pos,
        })
        .collect();

    (positioned, original_words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_simple() {
        let terms = tokenize("list my repos");
        assert!(terms.contains(&"list".to_string()));
        assert!(terms.contains(&"repos".to_string()));
        assert!(!terms.contains(&"my".to_string()));
    }

    #[test]
    fn tokenize_bigrams() {
        let terms = tokenize("charge credit card");
        assert!(terms.contains(&"charge".to_string()));
        assert!(terms.contains(&"credit".to_string()));
        assert!(terms.contains(&"card".to_string()));
        assert!(terms.contains(&"charge credit".to_string()));
        assert!(terms.contains(&"credit card".to_string()));
    }

    #[test]
    fn tokenize_strips_punctuation() {
        let terms = tokenize("what's my repo?");
        assert!(terms.contains(&"repo".to_string()));
    }

    #[test]
    fn tokenize_empty() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn tokenize_all_stop_words() {
        assert!(tokenize("can you please do this for me").is_empty());
    }

    #[test]
    fn tokenize_deduplication() {
        let terms = tokenize("charge charge charge");
        assert_eq!(terms.iter().filter(|t| *t == "charge").count(), 1);
    }

    #[test]
    fn tokenize_contractions() {
        // "don't" expands cleanly — no garbage "don"/"t" tokens
        // "cancel" survives: "want" (stop word) absorbs the negation scope
        let terms = tokenize("I don't want to cancel");
        assert!(!terms.contains(&"don".to_string()));
        assert!(!terms.contains(&"t".to_string()));
        assert!(terms.contains(&"cancel".to_string())); // survives — scope absorbed by "want"

        // "can't" expands cleanly — NOT negation (inability)
        let terms = tokenize("I can't log in");
        assert!(!terms.contains(&"t".to_string()));
        assert!(terms.contains(&"log".to_string())); // preserved — "can't" is inability

        // "what's" should not produce "s"
        let terms = tokenize("what's happening");
        assert!(!terms.contains(&"s".to_string()));
        assert!(terms.contains(&"happening".to_string()));

        // "won't" irregular contraction — NOT negation (refusal)
        let terms = tokenize("it won't work");
        assert!(!terms.contains(&"won".to_string()));
        assert!(terms.contains(&"work".to_string())); // preserved
    }

    #[test]
    fn tokenize_negation_suppression() {
        // "don't cancel" → "do not cancel" → "cancel" is immediate next → negated
        let terms = tokenize("don't cancel my order");
        assert!(!terms.contains(&"cancel".to_string()));
        assert!(terms.contains(&"order".to_string()));

        // "don't want to cancel" → "want" absorbs scope → "cancel" survives
        let terms = tokenize("don't want to cancel my order");
        assert!(terms.contains(&"cancel".to_string()));
        assert!(terms.contains(&"order".to_string()));

        // "don't have my card" → "have" (stop word) absorbs scope → "card" survives
        let terms = tokenize("I don't have my card");
        assert!(terms.contains(&"card".to_string()));

        // "can't log in" → NOT negated (inability, not intent negation)
        let terms = tokenize("I can't log in");
        assert!(terms.contains(&"log".to_string()));

        // "never received" → "received" is immediate next → negated
        let terms = tokenize("I never received my card");
        assert!(!terms.contains(&"received".to_string()));
        assert!(terms.contains(&"card".to_string()));

        // "without cancelling" → "cancelling" is immediate next → negated
        let terms = tokenize("track my order without cancelling");
        assert!(terms.contains(&"track".to_string()));
        assert!(terms.contains(&"order".to_string()));
        assert!(!terms.contains(&"cancelling".to_string()));

        // No negation → normal behavior
        let terms = tokenize("cancel my order");
        assert!(terms.contains(&"cancel".to_string()));
        assert!(terms.contains(&"order".to_string()));
    }

    #[test]
    fn training_basic() {
        let terms = training_to_terms(&[
            "pause the music".to_string(),
            "stop playing".to_string(),
            "stop the music".to_string(),
        ]);
        assert!(terms.contains_key("music"));
        assert!(terms.contains_key("stop"));
        assert!(terms.contains_key("pause"));
        assert!(terms["music"] > terms["pause"]);
        assert!((terms["music"] - terms["stop"]).abs() < 0.01);
    }

    #[test]
    fn training_includes_bigrams() {
        let terms = training_to_terms(&[
            "stop the music".to_string(),
            "stop playing now".to_string(),
        ]);
        assert!(terms.contains_key("stop music"));
        assert!(terms.contains_key("stop playing"));
    }

    #[test]
    fn training_empty() {
        assert!(training_to_terms(&[]).is_empty());
        assert!(training_to_terms(&["the a an".to_string()]).is_empty());
    }

    #[test]
    fn training_weight_range() {
        let queries: Vec<String> = (0..15).map(|i| {
            if i < 10 { "music".to_string() } else { "song".to_string() }
        }).collect();
        let terms = training_to_terms(&queries);
        for (_, weight) in &terms {
            assert!(*weight >= 0.3);
            assert!(*weight <= 0.95);
        }
    }
}
