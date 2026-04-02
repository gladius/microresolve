//! Tokenizer — query and training phrase processing.
//!
//! Produces unigrams + bigrams from natural language, filtering stop words.
//! Also converts training phrases into term-weight maps for seeding intents.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

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

/// Public wrapper for expand_contractions (used by Router for dual-path tokenization).
pub fn expand_contractions_public(text: &str) -> String {
    expand_contractions(text)
}

/// Detect word positions that are negated (should be suppressed from scoring).
///
/// Conservative: only negates after "do not" (from expanded "don't") and
/// explicit negation words ("never", "without", "except"). Does NOT negate
/// after "can not" (can't = inability) or "is not" (isn't = state).
fn find_negated_positions(words: &[&str], stop_set: &HashSet<String>) -> HashSet<usize> {
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

/// Stop word data loaded from languages/stopwords.json.
///
/// Only two categories:
/// - `universal`: minimal Latin function words safe across all languages
/// - `cjk_chars`: script-specific CJK particles (zh/ja/ko in separate Unicode ranges)
///
/// Do NOT add per-language Latin stop lists — cross-language collisions
/// make them unsafe (e.g. German "die" = "the" but also English "die").
struct StopWordData {
    universal: HashSet<String>,
    cjk_chars: HashSet<char>,
}

/// Raw JSON structure matching stopwords.json.
#[derive(serde::Deserialize)]
struct StopWordsJson {
    universal: Vec<String>,
    unsegmented: HashMap<String, Vec<String>>,
}

fn stop_data() -> &'static StopWordData {
    static DATA: OnceLock<StopWordData> = OnceLock::new();
    DATA.get_or_init(|| {
        let json_str = include_str!("../languages/stopwords.json");
        let raw: StopWordsJson = serde_json::from_str(json_str).expect("invalid stopwords.json");

        let universal: HashSet<String> = raw.universal.into_iter().collect();

        let mut cjk_chars = HashSet::new();
        for words in raw.unsegmented.values() {
            for w in words {
                for c in w.chars() {
                    cjk_chars.insert(c);
                }
            }
        }

        StopWordData { universal, cjk_chars }
    })
}

/// Get the universal stop word set (minimal, cross-language safe).
fn universal_stop_set() -> &'static HashSet<String> {
    &stop_data().universal
}

/// Get the combined CJK stop character set (script-specific, no collision risk).
pub fn cjk_stop_char_set() -> &'static HashSet<char> {
    &stop_data().cjk_chars
}

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

    // Split into words, but also break CJK runs from Latin text
    let raw_words: Vec<&str> = expanded
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| !w.is_empty())
        .collect();

    // Expand CJK tokens into character bigrams (and individual chars as unigrams)
    let stop_set = universal_stop_set();
    let cjk_stop_set = cjk_stop_char_set();

    let mut words: Vec<String> = Vec::new();
    let mut is_word_cjk: Vec<bool> = Vec::new();

    for word in &raw_words {
        let has_cjk = word.chars().any(is_cjk);
        if !has_cjk {
            words.push(word.to_string());
            is_word_cjk.push(false);
        } else {
            // Split CJK token into sub-runs (CJK chars vs non-CJK chars)
            let chars: Vec<char> = word.chars().collect();
            let mut cjk_run = String::new();

            for &c in &chars {
                if is_cjk(c) {
                    cjk_run.push(c);
                } else {
                    if !cjk_run.is_empty() {
                        // Generate individual CJK words (non-stop chars) and bigrams
                        expand_cjk_run(&cjk_run, &cjk_stop_set, &mut words, &mut is_word_cjk);
                        cjk_run.clear();
                    }
                    // Non-CJK char in a mixed token — add as Latin word
                    let s = c.to_string();
                    if !s.is_empty() && s.chars().any(|c| c.is_alphanumeric()) {
                        words.push(s);
                        is_word_cjk.push(false);
                    }
                }
            }
            if !cjk_run.is_empty() {
                expand_cjk_run(&cjk_run, &cjk_stop_set, &mut words, &mut is_word_cjk);
            }
        }
    }

    let word_refs: Vec<&str> = words.iter().map(|w| w.as_str()).collect();
    let negated = find_negated_positions(&word_refs, &stop_set);

    let mut terms: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // Unigrams (excluding stop words, CJK stop chars, and negated terms)
    for (i, word) in words.iter().enumerate() {
        if !stop_set.contains(word.as_str()) && !negated.contains(&i) {
            if seen.insert(word.clone()) {
                terms.push(word.clone());
            }
        }
    }

    // Bigrams (consecutive non-stop, non-negated, non-CJK word pairs)
    // CJK bigrams are already generated by expand_cjk_run, so skip CJK words here
    let non_stop: Vec<&str> = words.iter()
        .enumerate()
        .filter(|(i, w)| !stop_set.contains(w.as_str()) && !negated.contains(i) && !is_word_cjk[*i])
        .map(|(_, w)| w.as_str())
        .collect();

    for window in non_stop.windows(2) {
        let bigram = format!("{} {}", window[0], window[1]);
        if seen.insert(bigram.clone()) {
            terms.push(bigram);
        }
    }

    terms
}

/// Expand a CJK character run into indexable terms.
///
/// Generates character bigrams from contiguous CJK text, filtering stop characters.
/// For space-separated seeds like "保存 食谱", each word arrives as a separate run,
/// so "保存" goes in as-is. For unsegmented text like "保存食谱", this produces
/// bigrams "保存", "存食", "食谱".
fn expand_cjk_run(
    run: &str,
    stop_set: &HashSet<char>,
    words: &mut Vec<String>,
    is_cjk: &mut Vec<bool>,
) {
    let chars: Vec<char> = run.chars()
        .filter(|c| !stop_set.contains(c))
        .collect();

    if chars.is_empty() {
        return;
    }

    // Single char: add as unigram
    if chars.len() == 1 {
        words.push(chars[0].to_string());
        is_cjk.push(true);
        return;
    }

    // For 2+ chars: add the full cleaned run as a term (exact match),
    // plus character bigrams for substring matching
    let full: String = chars.iter().collect();
    words.push(full);
    is_cjk.push(true);

    for window in chars.windows(2) {
        let bigram: String = window.iter().collect();
        words.push(bigram);
        is_cjk.push(true);
    }
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

/// A term with its position in the processed query.
#[derive(Debug, Clone)]
pub struct PositionedTerm {
    /// The term (lowercase, after stop word removal).
    pub term: String,
    /// Character offset in the processed query.
    pub offset: usize,
    /// Exclusive end character offset.
    pub end_offset: usize,
    /// Whether this term was extracted from CJK text (skip bigram generation).
    pub is_cjk: bool,
}

/// Tokenize with position tracking for multi-intent decomposition.
///
/// Unlike `tokenize()`, this preserves duplicate terms at different positions
/// so each occurrence can be consumed independently by different intents.
///
/// Returns `(positioned_terms, query_chars)` where `query_chars` is the
/// processed query as a char array (used for gap analysis in relation detection).
pub fn tokenize_positioned(query: &str) -> (Vec<PositionedTerm>, Vec<char>) {
    let lower = query.to_lowercase();
    let expanded = expand_contractions(&lower);
    let chars: Vec<char> = expanded.chars().collect();

    // Find words with their char positions
    let mut words_positions: Vec<(String, usize, usize)> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_alphanumeric() || chars[i] == '-' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '-') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            words_positions.push((word, start, i));
        } else {
            i += 1;
        }
    }

    let stop_set = universal_stop_set();
    let word_strs: Vec<&str> = words_positions.iter().map(|(w, _, _)| w.as_str()).collect();
    let negated = find_negated_positions(&word_strs, stop_set);

    let positioned: Vec<PositionedTerm> = words_positions
        .iter()
        .enumerate()
        .filter(|(idx, (w, _, _))| !stop_set.contains(w.as_str()) && !negated.contains(idx))
        .map(|(_, (w, start, end))| PositionedTerm {
            term: w.clone(),
            offset: *start,
            end_offset: *end,
            is_cjk: false,
        })
        .collect();

    (positioned, chars)
}

// --- Unsegmented Script Support ---

/// Check if a character belongs to an unsegmented script (no spaces between words).
///
/// These scripts require automaton-based tokenization rather than whitespace splitting.
/// Covers CJK (Chinese, Japanese, Korean), Thai, Myanmar (Burmese), Khmer, and Lao.
pub fn is_cjk(c: char) -> bool {
    matches!(c,
        // CJK
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Korean Hangul Syllables
        | '\u{3400}'..='\u{4DBF}' // CJK Unified Ideographs Extension A
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{1100}'..='\u{11FF}' // Hangul Jamo
        | '\u{3130}'..='\u{318F}' // Hangul Compatibility Jamo
        // Southeast Asian unsegmented scripts
        | '\u{0E00}'..='\u{0E7F}' // Thai
        | '\u{0E80}'..='\u{0EFF}' // Lao
        | '\u{1000}'..='\u{109F}' // Myanmar (Burmese)
        | '\u{1780}'..='\u{17FF}' // Khmer
    )
}

/// CJK negation markers.
pub const CJK_NEGATION_MARKERS: &[&str] = &["不", "没", "别", "未"];

/// Japanese multi-character negation suffixes.
pub const JA_NEGATION_SUFFIXES: &[&str] = &["ない", "しない", "できない"];

/// CJK clause boundary characters.
pub const CJK_CLAUSE_BOUNDARIES: &[char] = &['，', '、', '。', '；'];

/// CJK conjunction words (multi-character).
pub const CJK_CONJUNCTIONS: &[&str] = &["但", "然后", "而且", "或者"];

/// Script type for a text run.
#[derive(Debug, Clone, PartialEq)]
pub enum ScriptType {
    Latin,
    Cjk,
}

/// A contiguous run of same-script text.
#[derive(Debug, Clone)]
pub struct ScriptRun {
    pub script: ScriptType,
    pub text: String,
    /// Character offset of this run in the full query.
    pub char_offset: usize,
}

/// Split text into runs of Latin vs CJK script.
pub fn split_script_runs(text: &str) -> Vec<ScriptRun> {
    let mut runs = Vec::new();
    let mut current_text = String::new();
    let mut current_is_cjk: Option<bool> = None;
    let mut run_start = 0;
    let mut char_idx = 0;

    for c in text.chars() {
        if is_cjk(c) {
            if current_is_cjk == Some(false) {
                // Flush Latin run
                if !current_text.is_empty() {
                    runs.push(ScriptRun {
                        script: ScriptType::Latin,
                        text: std::mem::take(&mut current_text),
                        char_offset: run_start,
                    });
                }
                run_start = char_idx;
            }
            if current_is_cjk.is_none() {
                run_start = char_idx;
            }
            current_is_cjk = Some(true);
            current_text.push(c);
        } else if c.is_alphanumeric() {
            if current_is_cjk == Some(true) {
                // Flush CJK run
                if !current_text.is_empty() {
                    runs.push(ScriptRun {
                        script: ScriptType::Cjk,
                        text: std::mem::take(&mut current_text),
                        char_offset: run_start,
                    });
                }
                run_start = char_idx;
            }
            if current_is_cjk.is_none() {
                run_start = char_idx;
            }
            current_is_cjk = Some(false);
            current_text.push(c);
        } else {
            // Non-alphanumeric (spaces, punctuation): attach to current run
            current_text.push(c);
        }
        char_idx += 1;
    }

    if !current_text.is_empty() {
        let script = match current_is_cjk {
            Some(true) => ScriptType::Cjk,
            _ => ScriptType::Latin,
        };
        runs.push(ScriptRun {
            script,
            text: current_text,
            char_offset: run_start,
        });
    }

    runs
}

/// Find CJK negated character regions in a CJK text run.
///
/// Returns Vec of (start_char, end_char) ranges where terms should be suppressed.
/// Negation starts at a marker and extends to the next clause boundary or end.
pub fn find_cjk_negated_regions(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let text_len = chars.len();
    let mut regions = Vec::new();
    let stop_set: HashSet<char> = CJK_CLAUSE_BOUNDARIES.iter().copied().collect();

    // Check single-char Chinese negation markers
    for (i, &c) in chars.iter().enumerate() {
        let s: String = c.to_string();
        if CJK_NEGATION_MARKERS.contains(&s.as_str()) {
            // Find end: next clause boundary, conjunction, or end of text
            let neg_start = i + 1; // suppress after the marker
            let mut neg_end = text_len;
            for j in neg_start..text_len {
                if stop_set.contains(&chars[j]) {
                    neg_end = j;
                    break;
                }
                // Check multi-char conjunctions
                let remaining: String = chars[j..].iter().collect();
                if CJK_CONJUNCTIONS.iter().any(|conj| remaining.starts_with(conj)) {
                    neg_end = j;
                    break;
                }
            }
            if neg_start < neg_end {
                regions.push((neg_start, neg_end));
            }
        }
    }

    // Check Japanese multi-char negation suffixes
    let text_str: String = chars.iter().collect();
    for suffix in JA_NEGATION_SUFFIXES {
        let suffix_chars: Vec<char> = suffix.chars().collect();
        let suffix_len = suffix_chars.len();
        if text_len >= suffix_len {
            for i in 0..=(text_len - suffix_len) {
                if chars[i..i + suffix_len] == suffix_chars[..] {
                    // Negate terms after this suffix until clause boundary
                    let neg_start = i + suffix_len;
                    let mut neg_end = text_len;
                    for j in neg_start..text_len {
                        if stop_set.contains(&chars[j]) {
                            neg_end = j;
                            break;
                        }
                    }
                    if neg_start < neg_end {
                        regions.push((neg_start, neg_end));
                    }
                }
            }
        }
    }
    let _ = text_str; // suppress unused warning

    regions
}

/// Generate bigrams from unmatched CJK residual text.
///
/// Strips stop characters (but keeps negation markers), then generates character bigrams.
pub fn generate_cjk_residual_bigrams(text: &str) -> Vec<String> {
    let stop_set = cjk_stop_char_set();

    // Remove stop chars (keep negation markers — they're not in stop set)
    let cleaned: Vec<char> = text.chars()
        .filter(|c| is_cjk(*c) && !stop_set.contains(c))
        .collect();

    let mut bigrams = Vec::new();
    for window in cleaned.windows(2) {
        let bigram: String = window.iter().collect();
        bigrams.push(bigram);
    }
    bigrams
}

/// Segment a query into clause boundaries for independent scoring.
///
/// Splits at sentence-ending punctuation (. ? !), semicolons,
/// comma+conjunction patterns, and bare conjunctions between clauses.
/// Returns character positions where breaks occur.
///
/// Each segment is scored independently in route_multi to prevent
/// noise term accumulation across clause boundaries.
pub fn segment_breaks(query: &str) -> Vec<usize> {
    let chars: Vec<char> = query.chars().collect();
    let len = chars.len();
    let mut breaks: Vec<usize> = Vec::new();

    let lower: String = query.to_lowercase();
    let lower_chars: Vec<char> = lower.chars().collect();

    // Phase 1: Sentence boundaries and semicolons
    for (i, &c) in chars.iter().enumerate() {
        match c {
            '.' | '?' | '!' => breaks.push(i + 1),
            ';' => breaks.push(i),
            _ => {}
        }
    }

    // Phase 2: Comma + conjunction
    for (i, &c) in chars.iter().enumerate() {
        if c == ',' {
            let remaining_start = i + 1;
            if remaining_start < len {
                let end = len.min(remaining_start + 15);
                let rest: String = lower_chars[remaining_start..end].iter().collect();
                let trimmed = rest.trim_start();
                if trimmed.starts_with("and ")
                    || trimmed.starts_with("but ")
                    || trimmed.starts_with("or ")
                    || trimmed.starts_with("so ")
                    || trimmed.starts_with("because ")
                    || trimmed.starts_with("also ")
                    || trimmed.starts_with("then ")
                    || trimmed.starts_with("however ")
                {
                    breaks.push(i);
                }
            }
        }
    }

    // Phase 3: Bare conjunctions between clauses.
    // Split on "and", "but", "or", "because", "however", "also" when
    // they appear as standalone words with sufficient context on both sides.
    // This catches the common case: "I was charged twice and I still haven't..."
    let words: Vec<&str> = query.split_whitespace().collect();
    if words.len() >= 5 {
        // Build word-to-char-offset map
        let mut word_offsets: Vec<usize> = Vec::new();
        let mut pos = 0;
        for &word in &words {
            // Find this word's position in the original string
            if let Some(idx) = query[pos..].find(word) {
                word_offsets.push(pos + idx);
                pos = pos + idx + word.len();
            }
        }

        let conjunctions = ["and", "but", "or", "because", "however", "also"];
        for (wi, &word) in words.iter().enumerate() {
            let lower_word = word.to_lowercase();
            // Strip trailing punctuation for matching
            let clean = lower_word.trim_end_matches(|c: char| !c.is_alphabetic());
            if conjunctions.contains(&clean) {
                // Require at least 3 words before and 2 words after
                if wi >= 3 && wi + 2 < words.len() {
                    if let Some(&offset) = word_offsets.get(wi) {
                        breaks.push(offset);
                    }
                }
            }
        }
    }

    breaks.sort();
    breaks.dedup();

    // Phase 4: Merge tiny segments. If a segment has < 3 characters of content,
    // remove the break that created it.
    if !breaks.is_empty() {
        let mut filtered = Vec::new();
        let mut prev = 0usize;
        for &brk in &breaks {
            let seg = &query[byte_offset_from_char(&chars, prev)..byte_offset_from_char(&chars, brk)];
            let content_chars = seg.chars().filter(|c| c.is_alphanumeric()).count();
            if content_chars >= 6 {
                filtered.push(brk);
                prev = brk;
            }
            // else: skip this break, merge segment with next
        }
        // Check last segment
        if let Some(&last_brk) = filtered.last() {
            let seg = &query[byte_offset_from_char(&chars, last_brk)..];
            let content_chars = seg.chars().filter(|c| c.is_alphanumeric()).count();
            if content_chars < 6 {
                filtered.pop(); // merge last tiny segment back
            }
        }
        breaks = filtered;
    }

    breaks
}

/// Convert character offset to byte offset in a string.
fn byte_offset_from_char(chars: &[char], char_pos: usize) -> usize {
    chars[..char_pos.min(chars.len())]
        .iter()
        .map(|c| c.len_utf8())
        .sum()
}

/// Check if a CJK residual bigram is clean enough to learn.
///
/// Returns false for bigrams that are entirely stop characters or negation markers.
pub fn is_learnable_cjk_bigram(bigram: &str) -> bool {
    let stop_set = cjk_stop_char_set();
    let neg_chars: HashSet<char> = ['不', '没', '别', '未'].iter().copied().collect();
    let chars: Vec<char> = bigram.chars().collect();
    if chars.len() < 2 {
        return false;
    }
    // At least one char must be neither a stop char nor a negation marker
    chars.iter().any(|c| !stop_set.contains(c) && !neg_chars.contains(c))
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
        // Only universal stops: "a", "the", "is", "in", "to", etc.
        assert!(tokenize("the a an in on at to of for by").is_empty());
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

    // --- CJK tests ---

    #[test]
    fn is_cjk_detection() {
        assert!(is_cjk('取'));  // Chinese
        assert!(is_cjk('の'));  // Hiragana
        assert!(is_cjk('カ'));  // Katakana
        assert!(is_cjk('한'));  // Korean
        assert!(!is_cjk('a'));
        assert!(!is_cjk('1'));
        assert!(!is_cjk(' '));
    }

    #[test]
    fn split_script_runs_latin_only() {
        let runs = split_script_runs("cancel my order");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].script, ScriptType::Latin);
    }

    #[test]
    fn split_script_runs_cjk_only() {
        let runs = split_script_runs("取消订单");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].script, ScriptType::Cjk);
        assert_eq!(runs[0].char_offset, 0);
    }

    #[test]
    fn split_script_runs_mixed() {
        let runs = split_script_runs("cancel 取消订单 order");
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].script, ScriptType::Latin);
        assert_eq!(runs[1].script, ScriptType::Cjk);
        assert_eq!(runs[2].script, ScriptType::Latin);
    }

    #[test]
    fn cjk_residual_bigrams() {
        // "取消订单" → bigrams: "取消", "消订", "订单"
        let bigrams = generate_cjk_residual_bigrams("取消订单");
        assert_eq!(bigrams.len(), 3);
        assert!(bigrams.contains(&"取消".to_string()));
        assert!(bigrams.contains(&"订单".to_string()));
    }

    #[test]
    fn cjk_residual_bigrams_filters_stop_chars() {
        // "我的订单" → 的 is a stop char (particle), 我 is NOT (appears in compounds)
        // After removing 的: "我", "订", "单" → bigrams: "我订", "订单"
        let bigrams = generate_cjk_residual_bigrams("我的订单");
        assert!(bigrams.contains(&"订单".to_string()));
        assert!(!bigrams.iter().any(|b| b.contains('的'))); // 的 filtered out
    }

    #[test]
    fn cjk_negation_regions() {
        // "不取消" → negation at pos 0, suppresses from pos 1 to end
        let regions = find_cjk_negated_regions("不取消");
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0], (1, 3));
    }

    #[test]
    fn cjk_negation_stops_at_clause_boundary() {
        // "不取消，查看订单" → negation suppresses "取消" but not "查看订单"
        let regions = find_cjk_negated_regions("不取消，查看订单");
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].0, 1);
        assert_eq!(regions[0].1, 3); // stops at ，
    }

    #[test]
    fn cjk_negation_stops_at_conjunction() {
        // "不取消然后查看" → negation suppresses "取消" but not after "然后"
        let regions = find_cjk_negated_regions("不取消然后查看");
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].0, 1);
        assert_eq!(regions[0].1, 3); // stops at 然后
    }

    #[test]
    fn learnable_cjk_bigram_checks() {
        assert!(is_learnable_cjk_bigram("取消"));
        assert!(is_learnable_cjk_bigram("订单"));
        assert!(is_learnable_cjk_bigram("我的"));  // 我 is not a stop char, 的 is — one real char is enough
        assert!(!is_learnable_cjk_bigram("的了"));  // both are stop chars (pure particles)
        assert!(!is_learnable_cjk_bigram("a"));     // too short
    }

    #[test]
    fn positioned_terms_have_char_offsets() {
        let (terms, _chars) = tokenize_positioned("cancel my order");
        // "cancel" starts at char 0, "order" starts at char 10
        let cancel = terms.iter().find(|t| t.term == "cancel").unwrap();
        assert_eq!(cancel.offset, 0);
        assert_eq!(cancel.end_offset, 6);
        let order = terms.iter().find(|t| t.term == "order").unwrap();
        assert_eq!(order.offset, 10);
        assert_eq!(order.end_offset, 15);
    }
}
