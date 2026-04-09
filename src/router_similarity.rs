//! Router: distributional similarity and merge operations.

use crate::*;
use crate::tokenizer::*;
use crate::vector::LearnedVector;
use crate::index::InvertedIndex;
use std::collections::{HashMap, HashSet};

impl Router {
    pub fn build_similarity(&mut self, texts: &[String]) {
        const DIM: usize = 64;
        const TOP_K: usize = 10;
        const WINDOW: usize = 3;

        if texts.is_empty() { return; }

        // Collect vocabulary
        let mut word_to_id: HashMap<String, usize> = HashMap::new();
        let mut id_to_word: Vec<String> = Vec::new();

        let tokenized: Vec<Vec<usize>> = texts.iter().map(|text| {
            let terms = tokenize(text);
            terms.iter().map(|t| {
                let len = word_to_id.len();
                *word_to_id.entry(t.clone()).or_insert_with(|| {
                    id_to_word.push(t.clone());
                    len
                })
            }).collect()
        }).collect();

        let vs = id_to_word.len();
        if vs < 10 { return; } // too few words

        // Random vectors (deterministic from word ID)
        let rand_vecs: Vec<Vec<f32>> = (0..vs).map(|id| {
            let mut v = vec![0.0f32; DIM];
            for i in 0..8 {
                let h = (id as u32).wrapping_mul(2654435761).wrapping_add(i as u32 * 1013904223);
                let pos = (h as usize) % DIM;
                v[pos] += if (h >> 16) & 1 == 0 { 1.0 } else { -1.0 };
            }
            v
        }).collect();

        // Accumulate context vectors
        let mut ctx: Vec<Vec<f32>> = vec![vec![0.0f32; DIM]; vs];
        let mut ctx_count: Vec<u32> = vec![0; vs];

        for tokens in &tokenized {
            for (i, &a) in tokens.iter().enumerate() {
                for k in 1..=WINDOW {
                    if i + k < tokens.len() {
                        let b = tokens[i + k];
                        for d in 0..DIM {
                            ctx[a][d] += rand_vecs[b][d];
                            ctx[b][d] += rand_vecs[a][d];
                        }
                        ctx_count[a] += 1;
                        ctx_count[b] += 1;
                    }
                }
            }
        }

        // Normalize vectors
        for i in 0..vs {
            let norm: f32 = ctx[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for d in 0..DIM { ctx[i][d] /= norm; }
            }
        }

        // Compute top-K similar words for each single-word term.
        // Skip multi-word bigrams (contain spaces) — they produce noisy expansions.
        let mut similarity: HashMap<String, Vec<(String, f32)>> = HashMap::new();

        for i in 0..vs {
            if ctx_count[i] < 5 { continue; } // need enough observations
            if id_to_word[i].contains(' ') { continue; } // skip bigrams as source

            let mut sims: Vec<(usize, f32)> = (0..vs)
                .filter(|&j| j != i && ctx_count[j] >= 5 && !id_to_word[j].contains(' '))
                .map(|j| {
                    let dot: f32 = (0..DIM).map(|d| ctx[i][d] * ctx[j][d]).sum();
                    (j, dot)
                })
                .filter(|(_, s)| *s > 0.3) // similarity threshold
                .collect();

            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sims.truncate(TOP_K);

            if !sims.is_empty() {
                similarity.insert(
                    id_to_word[i].clone(),
                    sims.iter().map(|(j, s)| (id_to_word[*j].clone(), *s)).collect(),
                );
            }
        }

        self.similarity = similarity;
    }

    /// Get similar terms for a word (from the similarity index).
    pub fn similar_terms(&self, term: &str) -> &[(String, f32)] {
        self.similarity.get(term).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Check if the similarity index has been built.
    pub fn has_similarity(&self) -> bool {
        !self.similarity.is_empty()
    }

    /// Set the expansion discount factor for similarity-based term expansion.
    /// Default is 0.3. Lower = more conservative, higher = more aggressive.
    pub fn set_expansion_discount(&mut self, discount: f32) {
        self.expansion_discount = discount;
    }

    /// Merge learned weights from another router into this one.
    ///
    /// Uses max() per term per intent — this is a CRDT merge:
    /// commutative, associative, idempotent, conflict-free.
    ///
    /// - Seed weights are never modified (immutable layer preserved)
    /// - Only learned weights are combined
    /// - New intents in `other` that don't exist here are ignored
    ///   (they have no seed layer to anchor them)
    /// - Co-occurrence counts are summed
    /// - Paraphrase phrases are merged (other's phrases added if not present)
    ///
    /// After merge, the inverted index is rebuilt to reflect new weights.
    pub fn merge_learned(&mut self, other: &Router) {
        let mut changed = false;

        // Merge learned weights per intent
        for (intent_id, other_vector) in &other.vectors {
            if let Some(self_vector) = self.vectors.get_mut(intent_id) {
                if other_vector.has_learned() {
                    self_vector.merge_learned(other_vector);
                    changed = true;
                }
            }
            // Intents only in `other` are skipped — no seed layer here to anchor them
        }

        // Merge co-occurrence (additive)
        for ((a, b), &count) in &other.co_occurrence {
            *self.co_occurrence.entry((a.clone(), b.clone())).or_insert(0) += count;
        }

        // Merge temporal ordering (additive)
        for ((a, b), &count) in &other.temporal_order {
            *self.temporal_order.entry((a.clone(), b.clone())).or_insert(0) += count;
        }

        // Merge intent sequences (append, cap at 1000)
        for seq in &other.intent_sequences {
            self.intent_sequences.push(seq.clone());
        }
        if self.intent_sequences.len() > 1000 {
            let excess = self.intent_sequences.len() - 1000;
            self.intent_sequences.drain(0..excess);
        }

        // Merge paraphrase phrases (keep existing if conflict, add new)
        for (phrase, (intent_id, weight)) in &other.paraphrase_phrases {
            self.paraphrase_phrases
                .entry(phrase.clone())
                .or_insert_with(|| (intent_id.clone(), *weight));
        }

        // Merge training phrases (union)
        for (intent_id, other_lang_map) in &other.training {
            let self_lang_map = self.training.entry(intent_id.clone()).or_default();
            for (lang, other_phrases) in other_lang_map {
                let self_phrases = self_lang_map.entry(lang.clone()).or_default();
                let existing: HashSet<String> = self_phrases.iter().cloned().collect();
                for phrase in other_phrases {
                    if !existing.contains(phrase) {
                        self_phrases.push(phrase.clone());
                    }
                }
            }
        }

        if changed {
            self.rebuild_index();
            self.rebuild_paraphrase_automaton_now();
            self.version += 1;
        }
    }

    /// Export only the learned layer weights for lightweight sync.
    ///
    /// Returns a JSON object: { intent_id: { term: weight, ... }, ... }
    /// Only includes intents that have learned weights.
    /// Much smaller than full export — just the delta from seed state.
    pub fn export_learned_only(&self) -> String {
        let learned: HashMap<&str, &HashMap<String, f32>> = self.vectors.iter()
            .filter(|(_, v)| v.has_learned())
            .map(|(id, v)| (id.as_str(), v.learned_terms()))
            .collect();
        serde_json::to_string(&learned).unwrap_or_default()
    }

    /// Import and merge learned weights from a lightweight sync payload.
    ///
    /// Input format: { intent_id: { term: weight, ... }, ... }
    /// Uses max() merge — safe to call multiple times with same data (idempotent).
    pub fn import_learned_merge(&mut self, json: &str) -> Result<(), String> {
        let learned: HashMap<String, HashMap<String, f32>> =
            serde_json::from_str(json).map_err(|e| format!("invalid JSON: {}", e))?;

        let mut changed = false;
        for (intent_id, other_terms) in &learned {
            if let Some(vector) = self.vectors.get_mut(intent_id) {
                let other_vec = LearnedVector::from_parts(HashMap::new(), other_terms.clone());
                vector.merge_learned(&other_vec);
                changed = true;
            }
        }

        if changed {
            self.rebuild_index();
            self.version += 1;
        }
        Ok(())
    }

    pub(crate) fn rebuild_index(&mut self) {
        self.index = InvertedIndex::build(&self.vectors);
        // Full index rebuild always rebuilds automaton immediately (not deferred)
        self.rebuild_cjk_automaton_now();
    }

    /// Request a CJK automaton rebuild. Deferred if in batch mode.
    pub(crate) fn rebuild_cjk_automaton(&mut self) {
        if self.batch_mode {
            self.cjk_dirty = true;
        } else {
            self.rebuild_cjk_automaton_now();
        }
    }

    /// Unconditionally rebuild the Aho-Corasick automaton from CJK terms in the index.
    pub(crate) fn rebuild_cjk_automaton_now(&mut self) {
        let cjk_terms: Vec<String> = self.index.terms()
            .filter(|t| t.chars().any(is_cjk))
            .cloned()
            .collect();

        if cjk_terms.is_empty() {
            self.cjk_automaton = None;
            self.cjk_patterns = Vec::new();
            return;
        }

        self.cjk_automaton = Some(
            AhoCorasick::builder()
                .match_kind(aho_corasick::MatchKind::Standard)
                .build(&cjk_terms)
                .expect("failed to build CJK automaton")
        );
        self.cjk_patterns = cjk_terms;
    }

    // ===== Paraphrase Index =====

    // Add paraphrase phrases for an intent.
    //
    // Paraphrases are multi-word expressions scanned via Aho-Corasick automaton.
    // When both the routing index and paraphrase index detect the same intent,
    // the detection is tagged as "dual-source" with "high" confidence.
    //
    // ```
    // use asv_router::Router;
    //
    // let mut router = Router::new();
    // router.add_intent("refund", &["I want a refund", "money back"]);
    // router.add_paraphrases("refund", &[
    //     "get my money back",
    //     "return for a full refund",
    //     "I need a refund please",
    // ]);
    // ```

}
