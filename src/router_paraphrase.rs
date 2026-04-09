//! Router: paraphrase index (Aho-Corasick phrase matching).

use crate::*;
use crate::tokenizer::*;
use aho_corasick::AhoCorasick;
use std::collections::HashMap;

impl Router {
    pub fn add_paraphrases(&mut self, intent_id: &str, phrases: &[&str]) {
        self.require_local();
        for phrase in phrases {
            let lower = phrase.to_lowercase();
            if lower.split_whitespace().count() >= 2 && lower.len() >= 5 {
                self.paraphrase_phrases.insert(lower, (intent_id.to_string(), 0.8));
            }
        }
        self.rebuild_paraphrase_automaton();
    }

    /// Add paraphrases from a map of intent_id -> phrases (for bulk loading).
    pub fn add_paraphrases_bulk(&mut self, data: &HashMap<String, Vec<String>>) {
        self.require_local();
        for (intent_id, phrases) in data {
            for phrase in phrases {
                let lower = phrase.to_lowercase();
                if lower.split_whitespace().count() >= 2 && lower.len() >= 5 {
                    self.paraphrase_phrases.insert(lower, (intent_id.clone(), 0.8));
                }
            }
        }
        self.rebuild_paraphrase_automaton();
    }

    /// Scan a message against the paraphrase automaton.
    /// Returns: Vec of (intent_id, weight, match_start_position).
    pub(crate) fn paraphrase_scan(&self, message: &str) -> Vec<(String, f32, usize)> {
        let lower = message.to_lowercase();
        let mut results: Vec<(String, f32, usize)> = Vec::new();
        let mut seen_intents: HashSet<String> = HashSet::new();

        if let Some(ref ac) = self.paraphrase_automaton {
            for mat in ac.find_iter(&lower) {
                let pattern = &self.paraphrase_patterns[mat.pattern().as_usize()];
                if let Some((intent_id, weight)) = self.paraphrase_phrases.get(pattern) {
                    if seen_intents.insert(intent_id.clone()) {
                        results.push((intent_id.clone(), *weight, mat.start()));
                    }
                }
            }
        }
        results
    }

    /// Learn paraphrase n-grams from a message for an intent.
    /// Extracts all overlapping 3-5 word windows as paraphrase phrases.
    /// Multi-word phrases are inherently discriminative so no filtering is needed.
    /// Matches clean experiment's extraction: min 2 words + 5 chars, overwrites allowed.
    pub(crate) fn learn_paraphrases(&mut self, message: &str, intent_id: &str) {
        let lower = message.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        for window_size in 3..=5 {
            if words.len() >= window_size {
                for start in 0..=(words.len() - window_size) {
                    let phrase: String = words[start..start + window_size].join(" ");
                    if phrase.split_whitespace().count() >= 2 && phrase.len() >= 5 {
                        self.paraphrase_phrases.insert(phrase, (intent_id.to_string(), 0.5));
                    }
                }
            }
        }

        // Also add the full message if it's short enough
        if words.len() >= 3 && words.len() <= 12 {
            self.paraphrase_phrases.insert(lower, (intent_id.to_string(), 0.6));
        }

        self.rebuild_paraphrase_automaton();
    }

    /// Request a paraphrase automaton rebuild. Deferred if in batch mode.
    pub(crate) fn rebuild_paraphrase_automaton(&mut self) {
        if self.batch_mode {
            self.paraphrase_dirty = true;
        } else {
            self.rebuild_paraphrase_automaton_now();
        }
    }

    /// Unconditionally rebuild the paraphrase Aho-Corasick automaton.
    pub(crate) fn rebuild_paraphrase_automaton_now(&mut self) {
        self.paraphrase_patterns = self.paraphrase_phrases.keys().cloned().collect();
        if self.paraphrase_patterns.is_empty() {
            self.paraphrase_automaton = None;
            return;
        }
        // Sort by length descending for leftmost-longest matching
        self.paraphrase_patterns.sort_by(|a, b| b.len().cmp(&a.len()));
        self.paraphrase_automaton = AhoCorasick::builder()
            .match_kind(aho_corasick::MatchKind::LeftmostLongest)
            .build(&self.paraphrase_patterns)
            .ok();
    }

    /// Get paraphrase count (for diagnostics).
    pub fn paraphrase_count(&self) -> usize {
        self.paraphrase_phrases.len()
    }

    /// Extract terms from a query using dual-path (Latin tokenizer + CJK automaton).
    pub(crate) fn extract_terms(&self, query: &str) -> Vec<String> {
        if !query.chars().any(is_cjk) {
            return tokenize(query);
        }

        let lower = query.to_lowercase();
        let runs = split_script_runs(&lower);
        let mut all_terms = Vec::new();
        let mut seen = HashSet::new();

        for run in &runs {
            match run.script {
                ScriptType::Latin => {
                    for term in tokenize(&run.text) {
                        if seen.insert(term.clone()) {
                            all_terms.push(term);
                        }
                    }
                }
                ScriptType::Cjk => {
                    let terms = self.extract_cjk_run_terms(&run.text, false);
                    for term in terms {
                        if seen.insert(term.clone()) {
                            all_terms.push(term);
                        }
                    }
                }
            }
        }

        all_terms
    }

    /// Extract terms for learning — more selective for CJK to prevent noise pollution.
    pub(crate) fn extract_terms_for_learning(&self, query: &str) -> Vec<String> {
        if !query.chars().any(is_cjk) {
            return tokenize(query);
        }

        let lower = query.to_lowercase();
        let runs = split_script_runs(&lower);
        let mut all_terms = Vec::new();
        let mut seen = HashSet::new();

        for run in &runs {
            match run.script {
                ScriptType::Latin => {
                    for term in tokenize(&run.text) {
                        if seen.insert(term.clone()) {
                            all_terms.push(term);
                        }
                    }
                }
                ScriptType::Cjk => {
                    let terms = self.extract_cjk_run_terms(&run.text, true);
                    for term in terms {
                        if seen.insert(term.clone()) {
                            all_terms.push(term);
                        }
                    }
                }
            }
        }

        all_terms
    }

    /// Extract terms from a CJK text run.
    ///
    /// 1. Detect negation marker positions
    /// 2. Scan automaton on original text (overlapping matches)
    /// 3. Find unmatched residual regions
    /// 4. Generate bigrams from cleaned residuals
    ///
    /// If `for_learning` is true, only include residual bigrams that pass the noise filter.
    pub(crate) fn extract_cjk_run_terms(&self, cjk_text: &str, for_learning: bool) -> Vec<String> {
        let negated_regions = find_cjk_negated_regions(cjk_text);
        let chars: Vec<char> = cjk_text.chars().collect();
        let mut matched_terms = Vec::new();
        let mut covered: HashSet<usize> = HashSet::new();

        // Step 1: Automaton scan (if available)
        if let Some(ref automaton) = self.cjk_automaton {
            for mat in automaton.find_overlapping_iter(cjk_text) {
                let pattern_idx = mat.pattern().as_usize();
                let term = &self.cjk_patterns[pattern_idx];

                // Convert byte offset to char offset
                let start_char = cjk_text[..mat.start()].chars().count();
                let end_char = cjk_text[..mat.end()].chars().count();

                // Check if this match falls in a negated region — prefix instead of skip
                let is_neg = negated_regions.iter().any(|(ns, ne)| start_char >= *ns && start_char < *ne);
                if is_neg {
                    matched_terms.push(format!("not_{}", term));
                } else {
                    matched_terms.push(term.clone());
                }
                for i in start_char..end_char {
                    covered.insert(i);
                }
            }
        }

        // Step 2: Find unmatched residual regions
        let mut residual_runs: Vec<String> = Vec::new();
        let mut current_run = String::new();

        for (i, &c) in chars.iter().enumerate() {
            if !covered.contains(&i) && is_cjk(c) {
                current_run.push(c);
            } else if !current_run.is_empty() {
                residual_runs.push(std::mem::take(&mut current_run));
            }
        }
        if !current_run.is_empty() {
            residual_runs.push(current_run);
        }

        // Step 3: Generate bigrams from residuals (with stop char filtering)
        for residual in &residual_runs {
            let bigrams = generate_cjk_residual_bigrams(residual);
            for bg in bigrams {
                // For learning, apply stricter filter
                if for_learning && !is_learnable_cjk_bigram(&bg) {
                    continue;
                }

                // Check negation for residual bigrams — prefix instead of skip
                let is_neg = if let Some(pos) = cjk_text.find(&bg) {
                    let char_pos = cjk_text[..pos].chars().count();
                    negated_regions.iter().any(|(ns, ne)| char_pos >= *ns && char_pos < *ne)
                } else {
                    false
                };

                if is_neg {
                    matched_terms.push(format!("not_{}", bg));
                } else {
                    matched_terms.push(bg);
                }
            }
        }

        matched_terms
    }

    /// Extract positioned terms for multi-intent decomposition.
    ///
    /// Returns positioned terms with character offsets and the processed query as chars.
    pub(crate) fn extract_terms_positioned(&self, query: &str) -> (Vec<PositionedTerm>, Vec<char>) {
        let lower = query.to_lowercase();

        if !lower.chars().any(is_cjk) {
            // Fast path: Latin only
            return tokenizer::tokenize_positioned(&lower);
        }

        // Dual path: expand contractions, split into script runs
        let expanded = tokenizer::expand_contractions_public(&lower);
        let full_chars: Vec<char> = expanded.chars().collect();
        let runs = split_script_runs(&expanded);

        let mut all_positioned = Vec::new();

        for run in &runs {
            match run.script {
                ScriptType::Latin => {
                    // Tokenize the Latin run and adjust offsets
                    let (terms, _) = tokenizer::tokenize_positioned(&run.text);
                    for mut pt in terms {
                        pt.offset += run.char_offset;
                        pt.end_offset += run.char_offset;
                        all_positioned.push(pt);
                    }
                }
                ScriptType::Cjk => {
                    let cjk_terms = self.extract_cjk_run_positioned(&run.text, run.char_offset);
                    all_positioned.extend(cjk_terms);
                }
            }
        }

        (all_positioned, full_chars)
    }

    /// Extract positioned CJK terms from a CJK text run using the automaton.
    pub(crate) fn extract_cjk_run_positioned(&self, cjk_text: &str, base_offset: usize) -> Vec<PositionedTerm> {
        let negated_regions = find_cjk_negated_regions(cjk_text);
        let chars: Vec<char> = cjk_text.chars().collect();

        let mut positioned = Vec::new();
        let mut covered: HashSet<usize> = HashSet::new();

        // Automaton scan
        if let Some(ref automaton) = self.cjk_automaton {
            for mat in automaton.find_overlapping_iter(cjk_text) {
                let pattern_idx = mat.pattern().as_usize();
                let term = &self.cjk_patterns[pattern_idx];

                let start_char = cjk_text[..mat.start()].chars().count();
                let end_char = cjk_text[..mat.end()].chars().count();

                let is_neg = negated_regions.iter().any(|(ns, ne)| start_char >= *ns && start_char < *ne);
                let final_term = if is_neg {
                    format!("not_{}", term)
                } else {
                    term.clone()
                };

                positioned.push(PositionedTerm {
                    term: final_term,
                    offset: base_offset + start_char,
                    end_offset: base_offset + end_char,
                    is_cjk: true,
                });

                for i in start_char..end_char {
                    covered.insert(i);
                }
            }
        }

        // Residual bigrams
        let mut current_run_start = None;
        let mut current_run = String::new();

        for (i, &c) in chars.iter().enumerate() {
            if !covered.contains(&i) && is_cjk(c) {
                if current_run_start.is_none() {
                    current_run_start = Some(i);
                }
                current_run.push(c);
            } else if !current_run.is_empty() {
                let run_start = current_run_start.take().unwrap();
                let bigrams = generate_cjk_residual_bigrams(&current_run);
                let mut bi = 0;
                for bg in bigrams {
                    positioned.push(PositionedTerm {
                        term: bg,
                        offset: base_offset + run_start + bi,
                        end_offset: base_offset + run_start + bi + 2,
                        is_cjk: true,
                    });
                    bi += 1;
                }
                current_run.clear();
            }
        }
        if !current_run.is_empty() {
            let run_start = current_run_start.unwrap();
            let bigrams = generate_cjk_residual_bigrams(&current_run);
            let mut bi = 0;
            for bg in bigrams {
                positioned.push(PositionedTerm {
                    term: bg,
                    offset: base_offset + run_start + bi,
                    end_offset: base_offset + run_start + bi + 2,
                    is_cjk: true,
                });
                bi += 1;
            }
        }

        positioned
    }

    // ===== Experimental methods for scenario testing =====

    /// Document frequency of a term across all intents.
    pub fn term_df(&self, term: &str) -> usize {
        self.index.df(term)
    }

    /// Analyze query terms: returns (term, idf, df) for each content term.
    pub fn analyze_query_terms(&self, query: &str) -> Vec<(String, f32, usize)> {
        let terms = tokenize(&query.to_lowercase());
        let n = self.index.intent_count().max(1) as f32;
        terms.into_iter().map(|t| {
            let df = self.index.df(&t);
            let idf = if df > 0 { 1.0 + 0.5 * (n / df as f32).ln() } else { 0.0 };
            (t, idf, df)
        }).collect()
    }

    /// Route multi-intent with noise gate: exclude terms appearing in > max_df intents.
    pub fn route_multi_noise_gated(&self, query: &str, threshold: f32, max_df: usize) -> MultiRouteOutput {
        let (positioned, query_chars) = self.extract_terms_positioned(query);
        let filtered: Vec<PositionedTerm> = positioned.into_iter()
            .filter(|pt| self.index.df(&pt.term) <= max_df)
            .collect();

        let mut output = multi::route_multi(&self.index, &self.vectors, filtered, query_chars, threshold, self.max_intents);
        for intent in &mut output.intents {
            intent.intent_type = self.get_intent_type(&intent.id);
        }
        output
    }

    /// Route multi-intent with anchor-based scoring.
    /// Only detects intents that have an anchor term (high discrimination) in the query.
    /// Scores using a local window of terms around each anchor.
    pub fn route_multi_anchored(&self, query: &str, threshold: f32, window: usize) -> MultiRouteOutput {
        let (positioned, query_chars) = self.extract_terms_positioned(query);
        if positioned.is_empty() {
            return MultiRouteOutput { intents: vec![], relations: vec![], metadata: HashMap::new(), suggestions: vec![] };
        }

        let n = self.index.intent_count();
        let disc_max_df = (n / 15).max(3);

        // Build reverse map: term -> intents it can anchor
        let mut term_to_intents: HashMap<&str, Vec<&str>> = HashMap::new();
        for (intent_id, vector) in &self.vectors {
            for (term, weight) in vector.effective_terms() {
                if self.index.df(&term) <= disc_max_df && weight >= 0.5 {
                    term_to_intents.entry(
                        // Leak string to get &str with right lifetime — only for experiments
                        // In production this would use a proper data structure
                        Box::leak(term.into_boxed_str()) as &str
                    ).or_default().push(
                        Box::leak(intent_id.clone().into_boxed_str()) as &str
                    );
                }
            }
        }

        // Find anchor matches in query positioned terms
        let mut anchored: HashMap<String, Vec<usize>> = HashMap::new(); // intent -> [term indices]
        for (idx, pt) in positioned.iter().enumerate() {
            if let Some(intents) = term_to_intents.get(pt.term.as_str()) {
                for &intent in intents {
                    anchored.entry(intent.to_string()).or_default().push(idx);
                }
            }
        }

        // For each anchored intent, score in local window around anchor
        let mut results: Vec<MultiRouteResult> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for (intent_id, anchor_positions) in &anchored {
            if seen.contains(intent_id) { continue; }

            let mut best_score = 0.0f32;
            let mut best_anchor_idx = 0usize;

            for &anchor_idx in anchor_positions {
                let start = anchor_idx.saturating_sub(window);
                let end = (anchor_idx + window + 1).min(positioned.len());
                let window_terms: Vec<String> = positioned[start..end].iter()
                    .map(|pt| pt.term.clone())
                    .collect();

                let search_results = self.index.search(&window_terms, 10);
                if let Some(sr) = search_results.iter().find(|r| r.id == *intent_id) {
                    if sr.score > best_score {
                        best_score = sr.score;
                        best_anchor_idx = anchor_idx;
                    }
                }
            }

            if best_score >= threshold {
                seen.insert(intent_id.clone());
                let start = best_anchor_idx.saturating_sub(window);
                let end = (best_anchor_idx + window + 1).min(positioned.len());
                let min_off = positioned[start..end].iter().map(|p| p.offset).min().unwrap_or(0);
                let max_off = positioned[start..end].iter().map(|p| p.end_offset).max().unwrap_or(0);

                results.push(MultiRouteResult {
                    id: intent_id.clone(),
                    score: best_score,
                    position: positioned[best_anchor_idx].offset,
                    span: (min_off, max_off),
                    intent_type: self.get_intent_type(intent_id),
                    confidence: "low".to_string(),
                    source: "routing".to_string(),
                    negated: false,
                });
            }
        }

        results.sort_by_key(|r| r.position);
        let relations = multi::detect_relations_public(&results, &query_chars);

        MultiRouteOutput { intents: results, relations, metadata: HashMap::new(), suggestions: vec![] }
    }

    /// Query coverage: (known_terms, total_terms) — fraction of terms in the index.
    pub fn query_coverage(&self, query: &str) -> (usize, usize) {
        let terms = tokenize(&query.to_lowercase());
        let total = terms.len();
        let known = terms.iter().filter(|t| self.index.df(t) > 0).count();
        (known, total)
    }

    /// Search the index directly (for experimental scoring).
    pub fn search_terms(&self, terms: &[String], top_k: usize) -> Vec<RouteResult> {
        self.index.search(terms, top_k).iter().map(|si| RouteResult {
            id: si.id.clone(),
            score: si.score,
        }).collect()
    }
}

