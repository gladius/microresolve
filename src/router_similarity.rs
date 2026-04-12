//! Router: distributional similarity.
//!
//! Builds a word co-occurrence graph from accumulated text using random projections.
//! Used for analysis; routing is handled by the Hebbian L1+L3 system.

use crate::*;
use crate::tokenizer::tokenize;
use std::collections::HashMap;

impl Router {
    /// Build a distributional similarity index from the given texts.
    ///
    /// Uses random projections (count-sketch style) in a co-occurrence window.
    /// After this call, `similar_terms()` returns related words for any known term.
    pub fn build_similarity(&mut self, texts: &[String]) {
        const DIM: usize = 64;
        const TOP_K: usize = 10;
        const WINDOW: usize = 3;

        if texts.is_empty() { return; }

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
        if vs < 10 { return; }

        // Deterministic random vectors (hash-based)
        let rand_vecs: Vec<Vec<f32>> = (0..vs).map(|id| {
            let mut v = vec![0.0f32; DIM];
            for i in 0..8 {
                let h = (id as u32).wrapping_mul(2654435761).wrapping_add(i as u32 * 1013904223);
                let pos = (h as usize) % DIM;
                v[pos] += if (h >> 16) & 1 == 0 { 1.0 } else { -1.0 };
            }
            v
        }).collect();

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

        for i in 0..vs {
            let norm: f32 = ctx[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for d in 0..DIM { ctx[i][d] /= norm; }
            }
        }

        let mut similarity: HashMap<String, Vec<(String, f32)>> = HashMap::new();

        for i in 0..vs {
            if ctx_count[i] < 5 { continue; }
            if id_to_word[i].contains(' ') { continue; }

            let mut sims: Vec<(usize, f32)> = (0..vs)
                .filter(|&j| j != i && ctx_count[j] >= 5 && !id_to_word[j].contains(' '))
                .map(|j| {
                    let dot: f32 = (0..DIM).map(|d| ctx[i][d] * ctx[j][d]).sum();
                    (j, dot)
                })
                .filter(|(_, s)| *s > 0.3)
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

    /// Get similar terms for a word.
    pub fn similar_terms(&self, term: &str) -> &[(String, f32)] {
        self.similarity.get(term).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Check if the similarity index has been built.
    pub fn has_similarity(&self) -> bool {
        !self.similarity.is_empty()
    }
}
