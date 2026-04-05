//! Neocortex LM v2: Position-Offset PMI + Low-Rank Residual on TinyStories
//!
//! Fixes from v1:
//! - Real corpus (TinyStories, 3.7M words of simple stories)
//! - Low-rank SVD residual instead of bigram correction table
//! - Proper evaluation with perplexity comparison
//!
//! Architecture: score(next_token | context) = PMI_signal + SVD_residual
//!   PMI_signal = Σ_k weight_k · PMI_offset_k[context[t-k], next_token]
//!   SVD_residual = low-rank correction learned from prediction errors
//!
//! Zero gradients. Zero neural network. Statistics + linear algebra.

use std::collections::HashMap;
use std::time::Instant;

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

fn build_vocab(texts: &[Vec<String>], max_vocab: usize) -> (HashMap<String, u32>, Vec<String>) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for doc in texts {
        for w in doc { *counts.entry(w.clone()).or_insert(0) += 1; }
    }
    let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.truncate(max_vocab - 1);

    let mut w2i: HashMap<String, u32> = HashMap::new();
    let mut i2w: Vec<String> = vec!["<unk>".to_string()];
    w2i.insert("<unk>".to_string(), 0);
    for (i, (word, _)) in sorted.iter().enumerate() {
        w2i.insert(word.clone(), (i + 1) as u32);
        i2w.push(word.clone());
    }
    (w2i, i2w)
}

// ── Position-Offset PMI (same as v1 but optimized) ──
struct OffsetPMI {
    // For each offset k: sparse map of (prev_token, next_token) → PMI
    offsets: Vec<HashMap<u64, f32>>,  // key = prev * vocab_size + next
    vocab_size: u32,
    max_offset: usize,
    unigram_log_prob: Vec<f32>,
}

impl OffsetPMI {
    fn key(a: u32, b: u32, vocab: u32) -> u64 { a as u64 * vocab as u64 + b as u64 }

    fn build(seqs: &[Vec<u32>], vocab_size: u32, max_offset: usize) -> Self {
        let mut pair_counts: Vec<HashMap<u64, u32>> = vec![HashMap::new(); max_offset];
        let mut unigram = vec![0u32; vocab_size as usize];
        let mut total_pairs = vec![0u64; max_offset];

        for seq in seqs {
            for (t, &tok) in seq.iter().enumerate() {
                unigram[tok as usize] += 1;
                for k in 1..=max_offset {
                    if t + k < seq.len() {
                        let next = seq[t + k];
                        *pair_counts[k-1].entry(Self::key(tok, next, vocab_size)).or_insert(0) += 1;
                        total_pairs[k-1] += 1;
                    }
                }
            }
        }

        let total_tok: f64 = unigram.iter().map(|&c| c as f64).sum();
        let unigram_log_prob: Vec<f32> = unigram.iter()
            .map(|&c| ((c as f64 + 1.0) / (total_tok + vocab_size as f64)).ln() as f32)
            .collect();

        let unigram_f: Vec<f64> = unigram.iter()
            .map(|&c| (c as f64 + 1.0) / (total_tok + vocab_size as f64))
            .collect();

        let mut offsets = Vec::with_capacity(max_offset);
        for k in 0..max_offset {
            let np = total_pairs[k] as f64;
            if np == 0.0 { offsets.push(HashMap::new()); continue; }
            let mut map = HashMap::new();
            for (&key, &count) in &pair_counts[k] {
                let a = (key / vocab_size as u64) as usize;
                let b = (key % vocab_size as u64) as usize;
                let p_ab = count as f64 / np;
                let pmi = (p_ab / (unigram_f[a] * unigram_f[b])).ln();
                if pmi > 0.3 {
                    map.insert(key, pmi as f32);
                }
            }
            offsets.push(map);
        }

        OffsetPMI { offsets, vocab_size, max_offset, unigram_log_prob }
    }

    /// Score all candidate next tokens given context
    fn score_next(&self, context: &[u32], scores: &mut [f32]) {
        let ctx_len = context.len();
        for s in scores.iter_mut() { *s = 0.0; }

        for k in 1..=self.max_offset.min(ctx_len) {
            let prev = context[ctx_len - k];
            let weight = 1.0 / (k as f32).sqrt(); // sqrt decay
            let map = &self.offsets[k - 1];
            for next in 0..self.vocab_size {
                let key = Self::key(prev, next, self.vocab_size);
                if let Some(&pmi) = map.get(&key) {
                    scores[next as usize] += pmi * weight;
                }
            }
        }
    }
}

// ── Low-Rank Residual for LM ──
// Instead of per-bigram corrections, learn a low-rank matrix:
// residual_score[next] = Σ_k (U_k · context_vec) · V_k[next]
//
// context_vec = bag-of-recent-words (weighted by recency)
// U is (rank × vocab), V is (rank × vocab)
// This generalizes: similar contexts get similar corrections.
struct LowRankLMResidual {
    u: Vec<f32>,      // rank × vocab (projects context)
    v: Vec<f32>,      // rank × vocab (projects candidate)
    rank: usize,
    vocab: usize,
}

impl LowRankLMResidual {
    fn new(rank: usize, vocab: usize) -> Self {
        LowRankLMResidual {
            u: vec![0.0; rank * vocab],
            v: vec![0.0; rank * vocab],
            rank, vocab,
        }
    }

    /// Add residual scores: for each candidate, score = (U · ctx_vec)^T · V[:, candidate]
    fn apply(&self, context: &[u32], scores: &mut [f32]) {
        // Build weighted context vector
        let ctx_len = context.len();
        // Project context through U: z = U · ctx_weighted
        let mut z = vec![0.0f32; self.rank];
        for (i, &tok) in context.iter().enumerate() {
            let weight = 1.0 / ((ctx_len - i) as f32).sqrt(); // recency weight
            let tok = tok as usize;
            for r in 0..self.rank {
                z[r] += self.u[r * self.vocab + tok] * weight;
            }
        }
        // Score each candidate: score[next] = z^T · V[:, next]
        for next in 0..self.vocab {
            let mut s = 0.0f32;
            for r in 0..self.rank {
                s += z[r] * self.v[r * self.vocab + next];
            }
            scores[next] += s;
        }
    }

    /// Perceptron correction: push toward correct, away from wrong
    fn correct(&mut self, context: &[u32], wrong: u32, correct: u32, lr: f32) {
        let ctx_len = context.len();
        // z = U · ctx (already computed, recompute for simplicity)
        let mut z = vec![0.0f32; self.rank];
        for (i, &tok) in context.iter().enumerate() {
            let weight = 1.0 / ((ctx_len - i) as f32).sqrt();
            let tok = tok as usize;
            for r in 0..self.rank {
                z[r] += self.u[r * self.vocab + tok] * weight;
            }
        }

        // Update V: push V[:, correct] toward z, V[:, wrong] away from z
        let correct = correct as usize;
        let wrong = wrong as usize;
        for r in 0..self.rank {
            self.v[r * self.vocab + correct] += lr * z[r];
            self.v[r * self.vocab + wrong] -= lr * z[r];
        }

        // Update U: push U projection toward correct direction
        for (i, &tok) in context.iter().enumerate() {
            let weight = lr / ((ctx_len - i) as f32).sqrt();
            let tok = tok as usize;
            for r in 0..self.rank {
                // direction: V[:, correct] - V[:, wrong]
                let dir = self.v[r * self.vocab + correct] - self.v[r * self.vocab + wrong];
                self.u[r * self.vocab + tok] += weight * dir * 0.1; // smaller U update
            }
        }
    }
}

// ── Evaluation ──
fn evaluate(
    seqs: &[Vec<u32>],
    pmi: &OffsetPMI,
    residual: &LowRankLMResidual,
    ctx_len: usize,
) -> (f64, f64, f64) {
    let vocab = pmi.vocab_size as usize;
    let mut total_ll = 0.0f64;
    let mut n_tok = 0u64;
    let mut top1 = 0u64;
    let mut top5 = 0u64;

    let mut scores = vec![0.0f32; vocab];

    for seq in seqs {
        for t in ctx_len..seq.len() {
            let ctx = &seq[t.saturating_sub(ctx_len)..t];
            let target = seq[t] as usize;

            // PMI scores
            pmi.score_next(ctx, &mut scores);
            // Add unigram prior
            for i in 0..vocab { scores[i] += pmi.unigram_log_prob[i] * 0.5; }
            // Add residual
            residual.apply(ctx, &mut scores);

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|s| (s - max_s).exp()).sum();
            let log_prob = (scores[target] - max_s) as f64 - (sum as f64).ln();
            total_ll += log_prob;
            n_tok += 1;

            // Top-1
            let pred = scores.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == target { top1 += 1; }

            // Top-5
            let mut idx: Vec<(usize, f32)> = scores.iter().enumerate()
                .map(|(i, &s)| (i, s)).collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if idx.iter().take(5).any(|(i, _)| *i == target) { top5 += 1; }
        }
    }

    let ppl = (-total_ll / n_tok as f64).exp();
    (ppl, top1 as f64 / n_tok as f64 * 100.0, top5 as f64 / n_tok as f64 * 100.0)
}

fn generate(
    pmi: &OffsetPMI, residual: &LowRankLMResidual,
    seed: &[u32], n: usize, i2w: &[String], temp: f32,
) -> String {
    let vocab = pmi.vocab_size as usize;
    let mut seq = seed.to_vec();
    let mut rng: u64 = 98765;
    let mut scores = vec![0.0f32; vocab];

    for _ in 0..n {
        let ctx = &seq[seq.len().saturating_sub(8)..];
        pmi.score_next(ctx, &mut scores);
        for i in 0..vocab { scores[i] += pmi.unigram_log_prob[i] * 0.5; }
        residual.apply(ctx, &mut scores);

        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| ((s - max_s) / temp).exp()).collect();
        let sum: f32 = exps.iter().sum();

        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let r = (rng % 1_000_000) as f32 / 1_000_000.0;
        let mut cum = 0.0;
        let mut chosen = 0;
        for (i, &e) in exps.iter().enumerate() {
            cum += e / sum;
            if cum >= r { chosen = i; break; }
        }
        seq.push(chosen as u32);
    }

    seq.iter().map(|&id| i2w.get(id as usize).map(|s| s.as_str()).unwrap_or("?"))
        .collect::<Vec<_>>().join(" ")
}

fn main() {
    let t0 = Instant::now();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Neocortex LM v2: TinyStories + Low-Rank Residual              ║");
    println!("║  Gradient-free language model on real stories                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Load TinyStories ──
    let raw = std::fs::read_to_string(
        // Try neocortex data dir first, then relative path
        std::path::Path::new("/home/gladius/Workspace/neocortex/data/tinystories.txt")
    ).expect("TinyStories not found — download to neocortex/data/tinystories.txt");

    // Split into stories
    let stories: Vec<Vec<String>> = raw.split("<|endoftext|>")
        .map(|s| words(s.trim()))
        .filter(|w| w.len() >= 10)
        .collect();

    println!("Stories: {}", stories.len());
    let total_words: usize = stories.iter().map(|s| s.len()).sum();
    println!("Total words: {}", total_words);

    // ── Vocab ──
    let max_vocab = 4096;
    let (vocab, i2w) = build_vocab(&stories, max_vocab);
    let vs = i2w.len() as u32;
    println!("Vocabulary: {} tokens", vs);

    // ── Tokenize ──
    let sequences: Vec<Vec<u32>> = stories.iter()
        .map(|s| s.iter().map(|w| *vocab.get(w).unwrap_or(&0)).collect())
        .collect();

    // ── Split ──
    let split = sequences.len() * 90 / 100;
    let train = &sequences[..split];
    let test_seqs = &sequences[split..];
    let train_tokens: usize = train.iter().map(|s| s.len()).sum();
    let test_tokens: usize = test_seqs.iter().map(|s| s.len()).sum();
    println!("Train: {} stories ({} tokens)", train.len(), train_tokens);
    println!("Test:  {} stories ({} tokens)", test_seqs.len(), test_tokens);

    // ── Build PMI ──
    println!();
    println!("Building position-offset PMI...");
    let t1 = Instant::now();
    let max_offset = 5;
    let pmi = OffsetPMI::build(train, vs, max_offset);
    let total_entries: usize = pmi.offsets.iter().map(|m| m.len()).sum();
    println!("PMI: {} total entries across {} offsets ({:.1}s)",
        total_entries, max_offset, t1.elapsed().as_secs_f64());

    // ── Baselines ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  BASELINES");
    println!("═══════════════════════════════════════════════════════════════════");

    let ctx_len = 5;
    let no_resid = LowRankLMResidual::new(0, vs as usize);

    let (ppl_uni, _, _) = {
        let mut ll = 0.0f64;
        let mut n = 0u64;
        for seq in test_seqs {
            for &tok in seq.iter().skip(1) {
                ll += pmi.unigram_log_prob[tok as usize] as f64;
                n += 1;
            }
        }
        ((-ll / n as f64).exp(), 0.0, 0.0)
    };
    println!("  Random:  perplexity = {}", vs);
    println!("  Unigram: perplexity = {:.1}", ppl_uni);

    let (ppl_pmi, top1_pmi, top5_pmi) = evaluate(test_seqs, &pmi, &no_resid, ctx_len);
    println!("  PMI:     perplexity = {:.1}, top-1 = {:.1}%, top-5 = {:.1}%", ppl_pmi, top1_pmi, top5_pmi);

    // ── Train low-rank residual ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  TRAINING (low-rank corrections, rank=16)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let rank = 16;
    let mut residual = LowRankLMResidual::new(rank, vs as usize);
    let lr = 0.01f32;
    let n_passes = 3;
    let mut scores_buf = vec![0.0f32; vs as usize];

    println!("  {:>4} | {:>10} | {:>7} | {:>7} | {:>10}",
        "Pass", "Perplexity", "Top-1", "Top-5", "Corrections");
    println!("  ───────────────────────────────────────────────────────");

    let (p, t1a, t5a) = evaluate(test_seqs, &pmi, &residual, ctx_len);
    println!("  {:>4} | {:>10.1} | {:>6.1}% | {:>6.1}% | {:>10}", 0, p, t1a, t5a, 0);

    let t_train = Instant::now();

    for pass in 1..=n_passes {
        let mut corrections = 0usize;
        let sample_rate = if pass == 1 { 1 } else { 2 }; // subsample later passes

        for (si, seq) in train.iter().enumerate() {
            if si % sample_rate != 0 { continue; }
            for t in ctx_len..seq.len() {
                let ctx = &seq[t.saturating_sub(ctx_len)..t];
                let target = seq[t];

                pmi.score_next(ctx, &mut scores_buf);
                for i in 0..vs as usize { scores_buf[i] += pmi.unigram_log_prob[i] * 0.5; }
                residual.apply(ctx, &mut scores_buf);

                let pred = scores_buf.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;

                if pred != target {
                    residual.correct(ctx, pred, target, lr);
                    corrections += 1;
                }
            }
        }

        let (ppl, top1, top5) = evaluate(test_seqs, &pmi, &residual, ctx_len);
        println!("  {:>4} | {:>10.1} | {:>6.1}% | {:>6.1}% | {:>10}",
            pass, ppl, top1, top5, corrections);
    }

    println!();
    println!("  Training time: {:.1}s", t_train.elapsed().as_secs_f64());

    // ── Generate ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  TEXT GENERATION (temperature=0.8)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let seeds = [
        "once upon a time",
        "the little girl",
        "one day the boy",
        "she was very happy",
        "they played together",
    ];

    for seed in &seeds {
        let ids: Vec<u32> = words(seed).iter()
            .map(|w| *vocab.get(w).unwrap_or(&0)).collect();
        let text = generate(&pmi, &residual, &ids, 30, &i2w, 0.8);
        println!("  \"{}\"", text);
        println!();
    }

    // ── Summary ──
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    let (final_ppl, final_top1, final_top5) = evaluate(test_seqs, &pmi, &residual, ctx_len);
    println!("  Vocab:         {} tokens", vs);
    println!("  Random PPL:    {}", vs);
    println!("  Unigram PPL:   {:.1}", ppl_uni);
    println!("  PMI-only PPL:  {:.1}  (top-1: {:.1}%)", ppl_pmi, top1_pmi);
    println!("  Final PPL:     {:.1}  (top-1: {:.1}%, top-5: {:.1}%)", final_ppl, final_top1, final_top5);
    println!("  PPL reduction: {:.1}x over unigram, {:.1}x over random",
        ppl_uni / final_ppl, vs as f64 / final_ppl);
    println!();
    println!("  No neural network. No gradients. No loss function.");
    println!("  Statistics + linear algebra + corrections.");
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
