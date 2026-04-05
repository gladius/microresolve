//! Test 2: Given 150 candidates, can SVD+corrections pick the correct next word?
//! This is 150-way classification — we proved 97.5% on 27-way.
//! Zero neural. Zero gradients. Rust only.

use std::collections::HashMap;
use std::time::Instant;

const CUTOFF_SECS: u64 = 900; // 15 min cutoff

fn main() {
    let t0 = Instant::now();
    println!("Test 2: Discriminate correct word from 150 candidates");
    println!();

    // Load & vocab (same as Test 1)
    let raw = std::fs::read_to_string("/home/gladius/Workspace/neocortex/data/tinystories.txt")
        .expect("TinyStories not found");
    let stories: Vec<Vec<String>> = raw.split("<|endoftext|>")
        .map(|s| s.to_lowercase().split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty()).map(|w| w.to_string()).collect::<Vec<_>>())
        .filter(|w| w.len() >= 10).collect();

    let mut counts: HashMap<String, usize> = HashMap::new();
    for s in &stories { for w in s { *counts.entry(w.clone()).or_insert(0) += 1; } }
    let mut w2i: HashMap<String, u32> = HashMap::new();
    let mut i2w: Vec<String> = vec!["<unk>".to_string()];
    w2i.insert("<unk>".to_string(), 0);
    let mut sorted: Vec<(String, usize)> = counts.into_iter().filter(|(_,c)| *c >= 3).collect();
    sorted.sort_by(|a,b| b.1.cmp(&a.1));
    for (w, _) in &sorted { let id = i2w.len() as u32; w2i.insert(w.clone(), id); i2w.push(w.clone()); }
    let vs = i2w.len();

    let mut all_ids: Vec<u32> = Vec::new();
    for s in &stories { for w in s { all_ids.push(*w2i.get(w).unwrap_or(&0)); } }
    let split = all_ids.len() * 90 / 100;
    let train = &all_ids[..split];
    let test = &all_ids[split..];
    println!("Vocab: {}, Train: {}, Test: {}", vs, train.len(), test.len());

    // Build conditional probability neighbors + frequency list
    let max_offset = 5;
    let ctx_len = 5;
    let mut unigram = vec![0u32; vs];
    let mut pair_counts: Vec<HashMap<(u32,u32), u32>> = (0..max_offset).map(|_| HashMap::new()).collect();

    for t in 0..train.len() {
        unigram[train[t] as usize] += 1;
        for k in 1..=max_offset {
            if t+k < train.len() {
                *pair_counts[k-1].entry((train[t], train[t+k])).or_insert(0) += 1;
            }
        }
    }

    // Conditional probability neighbors per token
    let mut cprob: Vec<Vec<(u32, f64)>> = vec![Vec::new(); vs];
    for k in 0..max_offset {
        for (&(a, b), &count) in &pair_counts[k] {
            let p = count as f64 / unigram[a as usize].max(1) as f64 / ((k+1) as f64).sqrt();
            if let Some(entry) = cprob[a as usize].iter_mut().find(|(id,_)| *id == b) {
                entry.1 += p;
            } else {
                cprob[a as usize].push((b, p));
            }
        }
    }
    for n in &mut cprob { n.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap()); }

    // Top 100 frequent words
    let mut freq_sorted: Vec<(u32, u32)> = unigram.iter().enumerate().map(|(i,&c)| (i as u32, c)).collect();
    freq_sorted.sort_by(|a,b| b.1.cmp(&a.1));
    let top_100: Vec<u32> = freq_sorted.iter().take(100).map(|(id,_)| *id).collect();

    println!("PMI + conditional prob built: {:.1}s", t0.elapsed().as_secs_f64());

    // ── Generate candidates for each test position ──
    // Candidate set: top-50 by cond.prob from context + top-100 frequent
    // Then discriminate: which candidate is correct?

    // Discrimination method: context features → score each candidate
    // Context features: which tokens are in the context window (bag of recent words)
    // Score: for each candidate, sum of P(candidate | context_token) for context tokens

    println!("\nPhase 1: Baseline discriminator (conditional probability ranking)...");
    let t1 = Instant::now();
    let eval_size = test.len().min(30000);
    let mut base_top1 = 0u64;
    let mut base_top5 = 0u64;
    let mut in_candidates = 0u64;
    let mut total = 0u64;

    for t in ctx_len..eval_size {
        let true_next = test[t];
        if true_next == 0 { continue; }

        // Generate candidates: top-50 cond.prob + top-100 freq
        let mut cand_scores: HashMap<u32, f64> = HashMap::new();
        for k in 1..=ctx_len {
            if t < k { continue; }
            let ctx_tok = test[t-k] as usize;
            let w = 1.0 / (k as f64).sqrt();
            for &(n, p) in cprob[ctx_tok].iter().take(50) {
                *cand_scores.entry(n).or_insert(0.0) += p * w;
            }
        }
        // Add frequent words with unigram score
        let n_total = train.len() as f64;
        for &fid in &top_100 {
            cand_scores.entry(fid).or_insert(unigram[fid as usize] as f64 / n_total * 0.5);
        }

        let mut candidates: Vec<(u32, f64)> = cand_scores.into_iter().collect();
        candidates.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

        // Is true_next in candidates?
        let in_cand = candidates.iter().any(|(id,_)| *id == true_next);
        if in_cand { in_candidates += 1; }

        // Top-1 accuracy (best candidate by cond.prob score)
        if candidates.first().map(|(id,_)| *id) == Some(true_next) { base_top1 += 1; }
        if candidates.iter().take(5).any(|(id,_)| *id == true_next) { base_top5 += 1; }

        total += 1;
    }

    let cov = in_candidates as f64 / total as f64 * 100.0;
    let bt1 = base_top1 as f64 / total as f64 * 100.0;
    let bt5 = base_top5 as f64 / total as f64 * 100.0;
    println!("  Coverage: {:.1}%, Top-1: {:.1}%, Top-5: {:.1}% ({} tokens, {:.1}s)",
        cov, bt1, bt5, total, t1.elapsed().as_secs_f64());

    // ── Phase 2: Perceptron corrections on candidate ranking ──
    println!("\nPhase 2: Perceptron corrections (fix ranking mistakes)...");

    // For discrimination, we need a scoring function that takes (context, candidate) → score
    // Simple: for each (context_pattern, candidate) pair, maintain a correction weight
    // Context pattern = hash of last 3 tokens (trigram context)

    // Correction table: (context_hash, candidate_id) → correction weight
    let mut corrections: HashMap<u64, f64> = HashMap::new();
    let corr_lr = 0.5;

    let hash_context = |tokens: &[u32]| -> u64 {
        let mut h: u64 = 0;
        for &t in tokens {
            h = h.wrapping_mul(31).wrapping_add(t as u64);
        }
        h
    };

    let n_passes = 3;
    let sample = (train.len() / 200_000).max(1);

    for pass in 1..=n_passes {
        if t0.elapsed().as_secs() > CUTOFF_SECS { println!("  CUTOFF"); break; }

        let mut n_corrections = 0u64;
        for t in (ctx_len..train.len()).step_by(sample) {
            let true_next = train[t];
            if true_next == 0 { continue; }

            // Generate candidates
            let mut cand_scores: HashMap<u32, f64> = HashMap::new();
            for k in 1..=ctx_len {
                if t < k { continue; }
                let ctx_tok = train[t-k] as usize;
                let w = 1.0 / (k as f64).sqrt();
                for &(n, p) in cprob[ctx_tok].iter().take(50) {
                    *cand_scores.entry(n).or_insert(0.0) += p * w;
                }
            }
            for &fid in &top_100 {
                let n_total = train.len() as f64;
                cand_scores.entry(fid).or_insert(unigram[fid as usize] as f64 / n_total * 0.5);
            }

            // Apply existing corrections
            let ctx_start = if t >= 3 { t - 3 } else { 0 };
            let ctx_hash = hash_context(&train[ctx_start..t]);

            for (cand, score) in cand_scores.iter_mut() {
                let key = ctx_hash.wrapping_mul(65537).wrapping_add(*cand as u64);
                if let Some(&corr) = corrections.get(&key) {
                    *score += corr;
                }
            }

            let mut candidates: Vec<(u32, f64)> = cand_scores.into_iter().collect();
            candidates.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

            let pred = candidates.first().map(|(id,_)| *id).unwrap_or(0);
            if pred != true_next {
                n_corrections += 1;
                // Boost correct, suppress wrong
                let key_correct = ctx_hash.wrapping_mul(65537).wrapping_add(true_next as u64);
                let key_wrong = ctx_hash.wrapping_mul(65537).wrapping_add(pred as u64);
                *corrections.entry(key_correct).or_insert(0.0) += corr_lr;
                *corrections.entry(key_wrong).or_insert(0.0) -= corr_lr;
            }
        }

        // Evaluate
        if t0.elapsed().as_secs() > CUTOFF_SECS { println!("  CUTOFF"); break; }

        let mut ct1 = 0u64; let mut ct5 = 0u64; let mut ct = 0u64;
        for t in ctx_len..eval_size {
            let true_next = test[t];
            if true_next == 0 { continue; }

            let mut cand_scores: HashMap<u32, f64> = HashMap::new();
            for k in 1..=ctx_len {
                if t < k { continue; }
                let ctx_tok = test[t-k] as usize;
                let w = 1.0 / (k as f64).sqrt();
                for &(n, p) in cprob[ctx_tok].iter().take(50) {
                    *cand_scores.entry(n).or_insert(0.0) += p * w;
                }
            }
            let n_total = train.len() as f64;
            for &fid in &top_100 {
                cand_scores.entry(fid).or_insert(unigram[fid as usize] as f64 / n_total * 0.5);
            }

            let ctx_start = if t >= 3 { t - 3 } else { 0 };
            let ctx_hash = hash_context(&test[ctx_start..t]);
            for (cand, score) in cand_scores.iter_mut() {
                let key = ctx_hash.wrapping_mul(65537).wrapping_add(*cand as u64);
                if let Some(&corr) = corrections.get(&key) { *score += corr; }
            }

            let mut candidates: Vec<(u32, f64)> = cand_scores.into_iter().collect();
            candidates.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

            if candidates.first().map(|(id,_)| *id) == Some(true_next) { ct1 += 1; }
            if candidates.iter().take(5).any(|(id,_)| *id == true_next) { ct5 += 1; }
            ct += 1;
        }

        let t1p = ct1 as f64 / ct as f64 * 100.0;
        let t5p = ct5 as f64 / ct as f64 * 100.0;
        println!("  Pass {}: top-1={:.1}%, top-5={:.1}%, corrections={}, table={} [{:.0}s]",
            pass, t1p, t5p, n_corrections, corrections.len(), t0.elapsed().as_secs_f64());
    }

    // ── Text Generation ──
    println!("\n═══════════════════════════════════════════════════════");
    println!("  TEXT GENERATION");
    println!("═══════════════════════════════════════════════════════\n");

    let seeds = ["once upon a time", "the little girl was", "she was happy because",
                 "he wanted to play", "one day the boy"];
    let mut rng: u64 = 42;

    for seed in &seeds {
        let mut seq: Vec<u32> = seed.split_whitespace()
            .map(|w| *w2i.get(&w.to_lowercase()).unwrap_or(&0)).collect();

        for _ in 0..30 {
            let t = seq.len();
            let mut cand_scores: HashMap<u32, f64> = HashMap::new();
            for k in 1..=ctx_len.min(t) {
                let ctx_tok = seq[t-k] as usize;
                let w = 1.0 / (k as f64).sqrt();
                for &(n, p) in cprob[ctx_tok].iter().take(50) {
                    *cand_scores.entry(n).or_insert(0.0) += p * w;
                }
            }
            let n_total = train.len() as f64;
            for &fid in &top_100 {
                cand_scores.entry(fid).or_insert(unigram[fid as usize] as f64 / n_total * 0.5);
            }

            // Apply corrections
            let ctx_start = if t >= 3 { t - 3 } else { 0 };
            let ctx_hash = hash_context(&seq[ctx_start..t]);
            for (cand, score) in cand_scores.iter_mut() {
                let key = ctx_hash.wrapping_mul(65537).wrapping_add(*cand as u64);
                if let Some(&corr) = corrections.get(&key) { *score += corr; }
            }

            // Temperature sampling from candidates
            let mut candidates: Vec<(u32, f64)> = cand_scores.into_iter().collect();
            candidates.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
            candidates.truncate(30); // sample from top 30

            let temp = 0.7;
            let max_s = candidates.iter().map(|(_,s)| *s).fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = candidates.iter().map(|(_,s)| ((s - max_s) / temp).exp()).collect();
            let sum: f64 = exps.iter().sum();

            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let r = (rng % 1_000_000) as f64 / 1_000_000.0;
            let mut cum = 0.0;
            let mut chosen = candidates[0].0;
            for (i, &e) in exps.iter().enumerate() {
                cum += e / sum;
                if cum >= r { chosen = candidates[i].0; break; }
            }
            seq.push(chosen);
        }

        let text: String = seq.iter()
            .map(|&id| i2w.get(id as usize).map(|s| s.as_str()).unwrap_or("?"))
            .collect::<Vec<_>>().join(" ");
        println!("  \"{}\"", text);
        println!();
    }

    println!("═══════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("═══════════════════════════════════════════════════════");
    println!("  Baseline (cond.prob ranking): top-1={:.1}%, top-5={:.1}%", bt1, bt5);
    println!("  Candidate coverage: {:.1}%", cov);
    println!("  Correction table: {} entries", corrections.len());
    println!("  Zero neural. Zero gradients. Zero PyTorch.");
    println!("  Time: {:.0}s", t0.elapsed().as_secs_f64());
}
