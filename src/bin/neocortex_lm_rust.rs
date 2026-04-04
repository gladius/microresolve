//! Neocortex LM: Non-Neural Next-Token Prediction
//!
//! Pure Rust. No PyTorch. No gradients. No transformer.
//! Pipeline: PMI (counting) → SVD (factorization) → Corrections (fix mistakes)
//!
//! Same pipeline that got 97.5% on classification. Now applied to next-token prediction.

use std::collections::HashMap;
use std::time::Instant;

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

// ── Dense matrix (reused from svd_attention.rs) ──
struct Mat { rows: usize, cols: usize, data: Vec<f64> }
impl Mat {
    fn zeros(r: usize, c: usize) -> Self { Mat { rows: r, cols: c, data: vec![0.0; r*c] } }
    fn get(&self, i: usize, j: usize) -> f64 { self.data[i * self.cols + j] }
    fn set(&mut self, i: usize, j: usize, v: f64) { self.data[i * self.cols + j] = v; }
    fn add(&mut self, i: usize, j: usize, v: f64) { self.data[i * self.cols + j] += v; }
    fn frobenius(&self) -> f64 { self.data.iter().map(|v| v*v).sum::<f64>().sqrt() }
}

// ── Power iteration SVD (top-k singular vectors) ──
fn svd_top_k(m: &Mat, k: usize, max_iter: usize) -> (Mat, Vec<f64>, Mat) {
    let mut u_mat = Mat::zeros(m.rows, k);
    let mut v_mat = Mat::zeros(m.cols, k);
    let mut sigmas = vec![0.0; k];
    let mut residual = Mat::zeros(m.rows, m.cols);
    residual.data.copy_from_slice(&m.data);

    for s in 0..k {
        let mut v = vec![0.0; residual.cols];
        for (i, x) in v.iter_mut().enumerate() { *x = ((i*7+s*13+1) % 100) as f64 - 50.0; }
        let norm: f64 = v.iter().map(|x| x*x).sum::<f64>().sqrt();
        for x in v.iter_mut() { *x /= norm; }

        let mut sigma = 0.0;
        let mut u = vec![0.0; residual.rows];

        for _ in 0..max_iter {
            u = vec![0.0; residual.rows];
            for i in 0..residual.rows {
                for j in 0..residual.cols { u[i] += residual.get(i,j) * v[j]; }
            }
            sigma = u.iter().map(|x| x*x).sum::<f64>().sqrt();
            if sigma < 1e-10 { break; }
            for x in u.iter_mut() { *x /= sigma; }

            v = vec![0.0; residual.cols];
            for j in 0..residual.cols {
                for i in 0..residual.rows { v[j] += residual.get(i,j) * u[i]; }
            }
            let vn: f64 = v.iter().map(|x| x*x).sum::<f64>().sqrt();
            if vn < 1e-10 { break; }
            for x in v.iter_mut() { *x /= vn; }
        }

        sigmas[s] = sigma;
        for i in 0..m.rows { u_mat.set(i, s, u[i]); }
        for j in 0..m.cols { v_mat.set(j, s, v[j]); }
        for i in 0..residual.rows {
            for j in 0..residual.cols {
                residual.add(i, j, -sigma * u[i] * v[j]);
            }
        }
    }
    (u_mat, sigmas, v_mat)
}

fn main() {
    let t0 = Instant::now();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Neocortex LM: Non-Neural Next-Token Prediction             ║");
    println!("║  Pure Rust. Zero gradients. Zero PyTorch. Zero transformer.  ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Load TinyStories ──
    let raw = std::fs::read_to_string("/home/gladius/Workspace/neocortex/data/tinystories.txt")
        .expect("TinyStories not found");
    let stories: Vec<Vec<String>> = raw.split("<|endoftext|>")
        .map(|s| words(s.trim()))
        .filter(|w| w.len() >= 10)
        .collect();
    println!("Stories: {}", stories.len());

    // ── Build vocab (all words appearing 3+ times) ──
    let mut counts: HashMap<String, usize> = HashMap::new();
    for s in &stories { for w in s { *counts.entry(w.clone()).or_insert(0) += 1; } }
    let mut w2i: HashMap<String, u32> = HashMap::new();
    let mut i2w: Vec<String> = vec!["<unk>".to_string()];
    w2i.insert("<unk>".to_string(), 0);
    let mut sorted: Vec<(String, usize)> = counts.into_iter().filter(|(_,c)| *c >= 3).collect();
    sorted.sort_by(|a,b| b.1.cmp(&a.1));
    for (w, _) in &sorted {
        let id = i2w.len() as u32;
        w2i.insert(w.clone(), id);
        i2w.push(w.clone());
    }
    let vs = i2w.len();
    println!("Vocab: {} (words appearing 3+ times)", vs);

    // ── Tokenize ──
    let mut all_ids: Vec<u32> = Vec::new();
    for s in &stories {
        for w in s { all_ids.push(*w2i.get(w).unwrap_or(&0)); }
    }
    let unk_count = all_ids.iter().filter(|&&id| id == 0).count();
    println!("Tokens: {}, unk: {:.1}%", all_ids.len(), unk_count as f64 / all_ids.len() as f64 * 100.0);

    // ── Split train/test ──
    let split = all_ids.len() * 90 / 100;
    let train = &all_ids[..split];
    let test = &all_ids[split..];

    // ── Step 1: PMI (counting) ──
    println!("\nStep 1: Computing PMI...");
    let t1 = Instant::now();
    let max_offset = 5;
    let mut unigram = vec![0u32; vs];
    let mut pair_counts: Vec<HashMap<(u32,u32), u32>> = (0..max_offset).map(|_| HashMap::new()).collect();
    let mut total_pairs = vec![0u64; max_offset];

    for t in 0..train.len() {
        let tok = train[t];
        unigram[tok as usize] += 1;
        for k in 1..=max_offset {
            if t+k < train.len() {
                let next = train[t+k];
                *pair_counts[k-1].entry((tok, next)).or_insert(0) += 1;
                total_pairs[k-1] += 1;
            }
        }
    }

    // Compute PMI scores for next-token prediction
    // For each (context_token, next_token) pair at each offset: PMI score
    let n_total = train.len() as f64;
    let unigram_p: Vec<f64> = unigram.iter().map(|&c| (c as f64 + 1.0) / (n_total + vs as f64)).collect();

    // Build score matrix: for each context window, score each candidate next token
    // Simple approach: weighted sum of PMI at each offset
    // score(next | context) = Σ_k weight_k × PMI(context[t-k], next) for k=1..5
    // Build indexed PMI lookup: token → Vec<(neighbor, pmi_score)>
    println!("  Building indexed PMI lookup...");
    let mut pmi_lookup: Vec<Vec<Vec<(u32, f64)>>> = vec![vec![Vec::new(); vs]; max_offset];
    for k in 0..max_offset {
        let np = total_pairs[k] as f64;
        if np == 0.0 { continue; }
        for (&(a, b), &count) in &pair_counts[k] {
            let p_ab = count as f64 / np;
            let pmi = (p_ab / (unigram_p[a as usize] * unigram_p[b as usize])).ln();
            if pmi > 0.0 {
                pmi_lookup[k][a as usize].push((b, pmi));
            }
        }
    }
    println!("  PMI computed + indexed in {:.1}s", t1.elapsed().as_secs_f64());

    let ctx_len = 5;

    // Helper: score candidates from context using indexed PMI
    let score_from_context = |context: &[u32], scores: &mut Vec<f64>| {
        for s in scores.iter_mut() { *s = 0.0; }
        let cl = context.len();
        for k in 1..=ctx_len.min(max_offset) {
            if cl < k { continue; }
            let ctx_tok = context[cl - k] as usize;
            let weight = 1.0 / (k as f64).sqrt();
            for &(b, pmi) in &pmi_lookup[k-1][ctx_tok] {
                scores[b as usize] += pmi * weight;
            }
        }
        // Unigram prior
        for i in 0..scores.len() { scores[i] += unigram_p[i].ln() * 0.3; }
    };

    // ── Step 2: PMI-only baseline ──
    println!("\nStep 2: PMI-only baseline...");
    let t2 = Instant::now();
    let mut correct_top1 = 0u64;
    let mut correct_top5 = 0u64;
    let mut total_eval = 0u64;
    let mut scores = vec![0.0f64; vs];

    for t in ctx_len..test.len().min(100_000) {
        let true_next = test[t] as usize;
        let ctx_start = if t >= ctx_len { t - ctx_len } else { 0 };
        score_from_context(&test[ctx_start..t], &mut scores);

        let pred = scores.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        if pred == true_next { correct_top1 += 1; }

        let mut indexed: Vec<(usize, f64)> = scores.iter().enumerate().map(|(i,&s)| (i,s)).collect();
        indexed.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        if indexed.iter().take(5).any(|(i,_)| *i == true_next) { correct_top5 += 1; }

        total_eval += 1;
    }

    let pmi_top1 = correct_top1 as f64 / total_eval as f64 * 100.0;
    let pmi_top5 = correct_top5 as f64 / total_eval as f64 * 100.0;
    println!("  PMI-only: top-1 = {:.1}%, top-5 = {:.1}% ({} tokens, {:.1}s)",
        pmi_top1, pmi_top5, total_eval, t2.elapsed().as_secs_f64());

    // ── Step 3: Build context→next_token matrix for SVD ──
    println!("\nStep 3: Computing target matrix for SVD...");
    let t3 = Instant::now();

    // Context representation: for each position, bag-of-words of previous ctx_len tokens
    // Target: the next token (one-hot)
    // We want A such that A × context_repr ≈ next_token_repr
    //
    // Use a simpler approach: bigram correction matrix
    // For each (prev_token, next_token) pair: how much does PMI under/over predict?
    // correction[prev][next] = (actual frequency) - (PMI prediction)

    // Collect training predictions and errors
    let dim = vs.min(500); // limit for SVD tractability
    let mut qtq = Mat::zeros(dim, dim); // Q^T Q
    let mut ctq = Mat::zeros(vs, dim);  // C^T Q (target × context)

    let sample_every = (train.len() / 500_000).max(1); // sample for speed

    for t in (ctx_len..train.len()).step_by(sample_every) {
        let true_next = train[t] as usize;

        // Context vector: sparse, only ctx_len entries
        // Map context tokens to dim-space using modular hashing
        let mut ctx_vec = vec![0.0f64; dim];
        for k in 1..=ctx_len {
            if t < k { continue; }
            let tok = train[t-k] as usize;
            let idx = tok % dim;
            ctx_vec[idx] += 1.0 / (k as f64).sqrt();
        }

        // Accumulate Q^T Q
        for i in 0..dim {
            if ctx_vec[i] == 0.0 { continue; }
            for j in 0..dim {
                qtq.add(i, j, ctx_vec[i] * ctx_vec[j]);
            }
        }

        // Accumulate C^T Q (target row = true_next)
        for j in 0..dim {
            if ctx_vec[j] == 0.0 { continue; }
            ctq.add(true_next, j, ctx_vec[j]);
        }
    }

    // Regularize Q^T Q
    for i in 0..dim { qtq.add(i, i, 0.01); }

    // Invert Q^T Q via Gauss-Jordan
    println!("  Inverting {}×{} matrix...", dim, dim);
    let mut work = Mat::zeros(dim, 2*dim);
    for i in 0..dim {
        for j in 0..dim { work.set(i, j, qtq.get(i, j)); }
        work.set(i, dim+i, 1.0);
    }
    for col in 0..dim {
        let mut max_val = work.get(col, col).abs();
        let mut max_row = col;
        for row in (col+1)..dim {
            let v = work.get(row, col).abs();
            if v > max_val { max_val = v; max_row = row; }
        }
        if max_val < 1e-12 { continue; }
        if max_row != col {
            for j in 0..2*dim {
                let a = work.get(col, j); let b = work.get(max_row, j);
                work.set(col, j, b); work.set(max_row, j, a);
            }
        }
        let pivot = work.get(col, col);
        for j in 0..2*dim { let v = work.get(col, j); work.set(col, j, v/pivot); }
        for row in 0..dim {
            if row == col { continue; }
            let f = work.get(row, col);
            if f.abs() < 1e-15 { continue; }
            for j in 0..2*dim { let v = work.get(row, j) - f * work.get(col, j); work.set(row, j, v); }
        }
    }
    let mut inv = Mat::zeros(dim, dim);
    for i in 0..dim { for j in 0..dim { inv.set(i, j, work.get(i, dim+j)); } }

    // A* = C^T Q × (Q^T Q)^{-1} → gives (vs × dim) target matrix
    let mut a_star = Mat::zeros(vs, dim);
    for i in 0..vs {
        for j in 0..dim {
            let mut v = 0.0;
            for k in 0..dim { v += ctq.get(i, k) * inv.get(k, j); }
            a_star.set(i, j, v);
        }
    }

    println!("  Target matrix computed in {:.1}s", t3.elapsed().as_secs_f64());

    // ── Step 4: Evaluate A* (direct, no SVD compression yet) ──
    println!("\nStep 4: Evaluating target matrix...");
    let t4 = Instant::now();
    let mut svd_top1 = 0u64;
    let mut svd_top5 = 0u64;
    let mut svd_total = 0u64;

    for t in (ctx_len..test.len().min(100_000)).step_by(1) {
        let true_next = test[t] as usize;

        // Context vector
        let mut ctx_vec = vec![0.0f64; dim];
        for k in 1..=ctx_len {
            if t < k { continue; }
            let tok = test[t-k] as usize;
            ctx_vec[tok % dim] += 1.0 / (k as f64).sqrt();
        }

        // Score = A* × context
        let mut scores = vec![0.0f64; vs];
        for i in 0..vs {
            for j in 0..dim {
                if ctx_vec[j] == 0.0 { continue; }
                scores[i] += a_star.get(i, j) * ctx_vec[j];
            }
        }

        let pred = scores.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        if pred == true_next { svd_top1 += 1; }

        let mut indexed: Vec<(usize, f64)> = scores.iter().enumerate().map(|(i,&s)| (i,s)).collect();
        indexed.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        if indexed.iter().take(5).any(|(i,_)| *i == true_next as usize) { svd_top5 += 1; }

        svd_total += 1;
    }

    let ls_top1 = svd_top1 as f64 / svd_total as f64 * 100.0;
    let ls_top5 = svd_top5 as f64 / svd_total as f64 * 100.0;
    println!("  Least-squares: top-1 = {:.1}%, top-5 = {:.1}% ({:.1}s)",
        ls_top1, ls_top5, t4.elapsed().as_secs_f64());

    // ── Step 5: Perceptron corrections ──
    println!("\nStep 5: Perceptron corrections...");
    let t5 = Instant::now();

    // For each wrong prediction, adjust A*
    let n_passes = 5;
    let corr_lr = 0.01;

    for pass in 1..=n_passes {
        let mut corrections = 0u64;
        let sample = (train.len() / 200_000).max(1);

        for t in (ctx_len..train.len()).step_by(sample) {
            let true_next = train[t] as usize;

            let mut ctx_vec = vec![0.0f64; dim];
            for k in 1..=ctx_len {
                if t < k { continue; }
                ctx_vec[train[t-k] as usize % dim] += 1.0 / (k as f64).sqrt();
            }

            let mut sc = vec![0.0f64; vs];
            for i in 0..vs {
                for j in 0..dim {
                    if ctx_vec[j] == 0.0 { continue; }
                    sc[i] += a_star.get(i, j) * ctx_vec[j];
                }
            }
            let pred = sc.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

            if pred != true_next {
                corrections += 1;
                for j in 0..dim {
                    if ctx_vec[j] == 0.0 { continue; }
                    a_star.add(true_next, j, corr_lr * ctx_vec[j]);
                    a_star.add(pred, j, -corr_lr * ctx_vec[j]);
                }
            }
        }

        // Evaluate
        let mut c1 = 0u64; let mut c5 = 0u64; let mut ct = 0u64;
        for t in (ctx_len..test.len().min(50_000)).step_by(1) {
            let true_next = test[t] as usize;
            let mut ctx_vec = vec![0.0f64; dim];
            for k in 1..=ctx_len {
                if t < k { continue; }
                ctx_vec[test[t-k] as usize % dim] += 1.0 / (k as f64).sqrt();
            }
            let mut sc = vec![0.0f64; vs];
            for i in 0..vs {
                for j in 0..dim {
                    if ctx_vec[j] == 0.0 { continue; }
                    sc[i] += a_star.get(i, j) * ctx_vec[j];
                }
            }
            let pred = sc.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred == true_next { c1 += 1; }
            let mut idx: Vec<(usize,f64)> = sc.iter().enumerate().map(|(i,&s)|(i,s)).collect();
            idx.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
            if idx.iter().take(5).any(|(i,_)| *i == true_next) { c5 += 1; }
            ct += 1;
        }
        let t1_pct = c1 as f64 / ct as f64 * 100.0;
        let t5_pct = c5 as f64 / ct as f64 * 100.0;
        println!("  Pass {}: top-1 = {:.1}%, top-5 = {:.1}%, corrections = {} [{:.0}s]",
            pass, t1_pct, t5_pct, corrections, t5.elapsed().as_secs_f64());
    }

    // ── Step 6: Generate text ──
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  TEXT GENERATION (no neural network, no gradients)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let seeds = vec![
        "once upon a time",
        "the little girl",
        "one day the boy",
        "she was happy because",
        "he wanted to play but",
    ];

    let mut rng: u64 = 42;
    for seed in &seeds {
        let seed_words: Vec<u32> = seed.split_whitespace()
            .map(|w| *w2i.get(&w.to_lowercase()).unwrap_or(&0))
            .collect();
        let mut seq = seed_words.clone();

        for _ in 0..30 {
            let t = seq.len();
            let mut ctx_vec = vec![0.0f64; dim];
            for k in 1..=ctx_len.min(t) {
                ctx_vec[seq[t-k] as usize % dim] += 1.0 / (k as f64).sqrt();
            }
            let mut scores = vec![0.0f64; vs];
            for i in 0..vs {
                for j in 0..dim {
                    if ctx_vec[j] == 0.0 { continue; }
                    scores[i] += a_star.get(i, j) * ctx_vec[j];
                }
            }
            // Add unigram prior
            for i in 0..vs { scores[i] += unigram_p[i].ln() * 0.2; }

            // Temperature sampling
            let temp = 0.8;
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| ((s - max_s) / temp).exp()).collect();
            let sum: f64 = exps.iter().sum();

            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let r = (rng % 1_000_000) as f64 / 1_000_000.0;
            let mut cum = 0.0;
            let mut chosen = 0u32;
            for (i, &e) in exps.iter().enumerate() {
                cum += e / sum;
                if cum >= r { chosen = i as u32; break; }
            }
            seq.push(chosen);
        }

        let text: String = seq.iter()
            .map(|&id| i2w.get(id as usize).map(|s| s.as_str()).unwrap_or("<unk>"))
            .collect::<Vec<_>>().join(" ");
        println!("  \"{}\"", text);
        println!();
    }

    // ── Summary ──
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Vocab:        {} (zero neural, all from PMI)", vs);
    println!("  PMI-only:     top-1 = {:.1}%, top-5 = {:.1}%", pmi_top1, pmi_top5);
    println!("  Least-squares: top-1 = {:.1}%, top-5 = {:.1}%", ls_top1, ls_top5);
    println!("  Random:       top-1 = {:.2}%", 100.0 / vs as f64);
    println!();
    println!("  ZERO gradients. ZERO PyTorch. ZERO transformer.");
    println!("  Pure counting + matrix factorization + corrections.");
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
