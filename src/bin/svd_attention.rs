//! Experiment: SVD vs SGD for PMI-Residual Attention
//!
//! Can we compute the optimal low-rank residual in ONE SHOT via SVD,
//! matching or beating 60K steps of gradient descent?
//!
//! If yes: this architecture needs NO gradients, NO training loop.
//! Just counting (PMI) + matrix factorization (SVD).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(serde::Deserialize)]
struct Example { text: String, intents: Vec<String> }

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .filter(|w| !matches!(*w, "is" | "a" | "an" | "the" | "and" | "or" | "in" | "of"
            | "to" | "for" | "with" | "can" | "are" | "it" | "be" | "not" | "that"
            | "has" | "but" | "from" | "by" | "on" | "at" | "as" | "its" | "do"
            | "does" | "was" | "were" | "been" | "being" | "have" | "had" | "my"
            | "me" | "if" | "so" | "up" | "no" | "get" | "got" | "just" | "how"
            | "what" | "when" | "where" | "why" | "would" | "could" | "should"
            | "need" | "want" | "like" | "about" | "some" | "this" | "there"
            | "than" | "will" | "also" | "know"))
        .map(|w| w.to_string())
        .collect()
}

type SparseVec = Vec<(usize, f64)>;

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

// ── Dense matrix (row-major) ──
struct Mat { rows: usize, cols: usize, data: Vec<f64> }
impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }
    fn get(&self, i: usize, j: usize) -> f64 { self.data[i * self.cols + j] }
    fn set(&mut self, i: usize, j: usize, v: f64) { self.data[i * self.cols + j] = v; }
    fn add(&mut self, i: usize, j: usize, v: f64) { self.data[i * self.cols + j] += v; }

    /// Multiply: C = A × B
    fn mul(&self, b: &Mat) -> Mat {
        assert_eq!(self.cols, b.rows);
        let mut c = Mat::zeros(self.rows, b.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_ik = self.get(i, k);
                if a_ik == 0.0 { continue; }
                for j in 0..b.cols {
                    c.add(i, j, a_ik * b.get(k, j));
                }
            }
        }
        c
    }

    /// Transpose
    fn t(&self) -> Mat {
        let mut m = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                m.set(j, i, self.get(i, j));
            }
        }
        m
    }

    /// Column vector from dense slice
    fn from_col(v: &[f64]) -> Mat {
        let mut m = Mat::zeros(v.len(), 1);
        for (i, &x) in v.iter().enumerate() { m.set(i, 0, x); }
        m
    }

    fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }

    fn frobenius(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

// ── Power iteration SVD (simple, sufficient for our dimensions) ──
// Finds top-k singular vectors of M via repeated power iteration.
fn svd_top_k(m: &Mat, k: usize, max_iter: usize) -> (Mat, Vec<f64>, Mat) {
    // Returns: U (rows × k), sigma (k), V (cols × k)
    let mut u_mat = Mat::zeros(m.rows, k);
    let mut v_mat = Mat::zeros(m.cols, k);
    let mut sigmas = vec![0.0; k];

    // Work on a deflated copy
    let mut residual = Mat::zeros(m.rows, m.cols);
    residual.data.copy_from_slice(&m.data);

    for s in 0..k {
        // Random starting vector
        let mut v = vec![0.0; residual.cols];
        for (i, x) in v.iter_mut().enumerate() { *x = ((i * 7 + s * 13 + 1) % 100) as f64 - 50.0; }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in v.iter_mut() { *x /= norm; }

        let mut sigma = 0.0;
        let mut u = vec![0.0; residual.rows];

        for _iter in 0..max_iter {
            // u = M · v
            u = vec![0.0; residual.rows];
            for i in 0..residual.rows {
                for j in 0..residual.cols {
                    u[i] += residual.get(i, j) * v[j];
                }
            }
            sigma = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            if sigma < 1e-10 { break; }
            for x in u.iter_mut() { *x /= sigma; }

            // v = M^T · u
            v = vec![0.0; residual.cols];
            for j in 0..residual.cols {
                for i in 0..residual.rows {
                    v[j] += residual.get(i, j) * u[i];
                }
            }
            let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if v_norm < 1e-10 { break; }
            for x in v.iter_mut() { *x /= v_norm; }
        }

        sigmas[s] = sigma;
        for i in 0..m.rows { u_mat.set(i, s, u[i]); }
        for j in 0..m.cols { v_mat.set(j, s, v[j]); }

        // Deflate: residual -= sigma * u * v^T
        for i in 0..residual.rows {
            for j in 0..residual.cols {
                residual.add(i, j, -sigma * u[i] * v[j]);
            }
        }
    }

    (u_mat, sigmas, v_mat)
}

// ── Classification using A = PMI + U·Σ·V^T ──
fn classify_svd(
    query: &SparseVec,
    pmi: &Mat,
    u: &Mat, sigmas: &[f64], v: &Mat,
    centroids: &[Vec<f64>],
) -> usize {
    // project = (PMI + UΣV^T) · q
    let dim = pmi.rows;
    let rank = sigmas.len();
    let mut proj = vec![0.0; dim];

    // PMI part
    for &(j, qj) in query {
        for i in 0..dim { proj[i] += pmi.get(i, j) * qj; }
    }

    // SVD residual part: U · diag(Σ) · V^T · q
    // First: z = V^T · q (rank-dim)
    let mut z = vec![0.0; rank];
    for &(j, qj) in query {
        for r in 0..rank { z[r] += v.get(j, r) * qj * sigmas[r]; }
    }
    // Then: proj += U · z
    for i in 0..dim {
        for r in 0..rank { proj[i] += u.get(i, r) * z[r]; }
    }

    // Find best intent
    centroids.iter().enumerate()
        .map(|(k, c)| (k, dot(c, &proj)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap().0
}

fn evaluate_svd(
    data: &[(SparseVec, usize)],
    pmi: &Mat, u: &Mat, sigmas: &[f64], v: &Mat,
    centroids: &[Vec<f64>],
) -> f64 {
    let c = data.iter()
        .filter(|(q, t)| classify_svd(q, pmi, u, sigmas, v, centroids) == *t)
        .count();
    c as f64 / data.len() as f64 * 100.0
}

fn evaluate_pmi_only(
    data: &[(SparseVec, usize)],
    pmi: &Mat,
    centroids: &[Vec<f64>],
) -> f64 {
    let dim = pmi.rows;
    let c = data.iter().filter(|(q, t)| {
        let mut proj = vec![0.0; dim];
        for &(j, qj) in q.iter() {
            for i in 0..dim { proj[i] += pmi.get(i, j) * qj; }
        }
        let pred = centroids.iter().enumerate()
            .map(|(k, c)| (k, dot(c, &proj)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap().0;
        pred == *t
    }).count();
    c as f64 / data.len() as f64 * 100.0
}

fn full_eval(data: &[(SparseVec, usize)], mat: &Mat, centroids: &[Vec<f64>]) -> f64 {
    let dim = mat.rows;
    let c = data.iter().filter(|(q, t)| {
        let mut proj = vec![0.0; dim];
        for &(j, qj) in q.iter() {
            for i in 0..dim { proj[i] += mat.get(i, j) * qj; }
        }
        let pred = centroids.iter().enumerate()
            .map(|(k, c)| (k, dot(c, &proj)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap().0;
        pred == *t
    }).count();
    c as f64 / data.len() as f64 * 100.0
}

fn main() {
    let t0 = Instant::now();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  SVD vs SGD: Gradient-Free PMI-Residual Attention               ║");
    println!("║  Can we match 60K steps of training with ONE matrix operation?   ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Load data (same as before) ──
    let examples: Vec<Example> = serde_json::from_str(
        &std::fs::read_to_string("tests/data/benchmarks/bitext_all.json").unwrap()
    ).unwrap();

    let docs: Vec<Vec<String>> = examples.iter().map(|e| tokenize(&e.text)).collect();
    let mut df: HashMap<String, usize> = HashMap::new();
    for doc in &docs {
        let unique: HashSet<&String> = doc.iter().collect();
        for t in unique { *df.entry(t.clone()).or_insert(0) += 1; }
    }
    let mut by_df: Vec<(String, usize)> = df.into_iter()
        .filter(|(_, c)| *c >= 5 && *c < docs.len() / 2).collect();
    by_df.sort_by(|a, b| b.1.cmp(&a.1));
    by_df.truncate(300);
    let dim = by_df.len();
    let term_to_idx: HashMap<String, usize> = by_df.iter().enumerate()
        .map(|(i, (t, _))| (t.clone(), i)).collect();

    let mut intent_map: HashMap<String, usize> = HashMap::new();
    for ex in &examples { let n = intent_map.len(); intent_map.entry(ex.intents[0].clone()).or_insert(n); }
    let n_intents = intent_map.len();

    let all_data: Vec<(SparseVec, usize)> = examples.iter().filter_map(|ex| {
        let sparse: SparseVec = tokenize(&ex.text).iter()
            .filter_map(|t| term_to_idx.get(t).map(|&i| (i, 1.0)))
            .collect::<HashMap<usize, f64>>().into_iter().collect();
        if sparse.is_empty() { None } else { Some((sparse, intent_map[&ex.intents[0]])) }
    }).collect();

    // Same split as before (seed=42)
    let mut rng_state: u64 = 42;
    let mut indices: Vec<usize> = (0..all_data.len()).collect();
    for i in (1..indices.len()).rev() {
        rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
        let j = rng_state as usize % (i + 1);
        indices.swap(i, j);
    }
    let split = all_data.len() * 80 / 100;
    let train: Vec<(SparseVec, usize)> = indices[..split].iter().map(|&i| all_data[i].clone()).collect();
    let test: Vec<(SparseVec, usize)> = indices[split..].iter().map(|&i| all_data[i].clone()).collect();

    println!("Data: {} train, {} test, {} intents, dim={}", train.len(), test.len(), n_intents, dim);

    // ── PMI matrix ──
    let n = docs.len() as f64;
    let mut term_df: HashMap<usize, f64> = HashMap::new();
    let mut cooccur: HashMap<(usize, usize), f64> = HashMap::new();
    for doc in &docs {
        let unique: Vec<usize> = doc.iter()
            .filter_map(|t| term_to_idx.get(t).copied())
            .collect::<HashSet<_>>().into_iter().collect();
        for &idx in &unique { *term_df.entry(idx).or_insert(0.0) += 1.0; }
        for i in 0..unique.len() {
            for j in (i+1)..unique.len() {
                let key = if unique[i] < unique[j] { (unique[i], unique[j]) } else { (unique[j], unique[i]) };
                *cooccur.entry(key).or_insert(0.0) += 1.0;
            }
        }
    }
    let mut pmi = Mat::zeros(dim, dim);
    for (&(a, b), &count) in &cooccur {
        let p_ab = count / n;
        let p_a = term_df.get(&a).copied().unwrap_or(1.0) / n;
        let p_b = term_df.get(&b).copied().unwrap_or(1.0) / n;
        let v = (p_ab / (p_a * p_b)).ln();
        if v > 0.0 {
            let ppmi = v.min(5.0);
            pmi.set(a, b, ppmi);
            pmi.set(b, a, ppmi);
        }
    }

    // Normalize PMI (same as previous experiments)
    let pmi_norm = pmi.frobenius();
    let ref_norm = (dim as f64).sqrt(); // ~17.3
    let scale = ref_norm / pmi_norm;
    for v in pmi.data.iter_mut() { *v *= scale; }

    // ── Centroids ──
    let mut centroids = vec![vec![0.0f64; dim]; n_intents];
    let mut counts = vec![0usize; n_intents];
    for (q, intent) in &train {
        for &(i, v) in q { centroids[*intent][i] += v; }
        counts[*intent] += 1;
    }
    for k in 0..n_intents {
        if counts[k] > 0 { let c = counts[k] as f64; for v in centroids[k].iter_mut() { *v /= c; } }
    }

    // ── PMI-only baseline ──
    let pmi_acc = evaluate_pmi_only(&test, &pmi, &centroids);
    println!("PMI static: {:.1}%", pmi_acc);
    println!();

    // ══════════════════════════════════════════════════════════════
    // THE KEY COMPUTATION: Target matrix and SVD
    // ══════════════════════════════════════════════════════════════

    println!("Computing target attention matrix (what attention SHOULD be)...");
    let t1 = Instant::now();

    // Target: for each training query, the ideal output is the correct intent's centroid.
    // We want A such that A · q ≈ centroid[intent(q)] for all training queries.
    //
    // In matrix form: A · Q ≈ C  where Q is (dim × n_train) and C is (dim × n_train)
    // Least squares: A = C · Q^T · (Q · Q^T)^{-1}
    //
    // But Q·Q^T is dim×dim and may be singular. Use regularized pseudoinverse.

    // Build Q^T · Q (dim × dim) and C · Q^T (dim × dim)
    // Q is sparse, so we accumulate directly.
    let mut qtq = Mat::zeros(dim, dim);  // Q · Q^T (dim × dim)
    let mut cqt = Mat::zeros(dim, dim);  // C · Q^T (dim × dim) where C_col = centroid[intent]

    for (q, intent) in &train {
        let c = &centroids[*intent];
        for &(j1, qj1) in q {
            for &(j2, qj2) in q {
                qtq.add(j1, j2, qj1 * qj2);
            }
            for i in 0..dim {
                cqt.add(i, j1, c[i] * qj1);
            }
        }
    }

    // Regularize Q·Q^T for invertibility
    for i in 0..dim { qtq.add(i, i, 0.01); }

    // Invert Q·Q^T via Cholesky-like approach (it's positive definite after regularization)
    // For dim=300, direct Gauss-Jordan is fine.
    println!("Inverting {}×{} matrix...", dim, dim);

    let mut inv = Mat::zeros(dim, dim);
    let mut work = Mat::zeros(dim, 2 * dim);
    for i in 0..dim {
        for j in 0..dim { work.set(i, j, qtq.get(i, j)); }
        work.set(i, dim + i, 1.0);
    }
    for col in 0..dim {
        // Find pivot
        let mut max_val = work.get(col, col).abs();
        let mut max_row = col;
        for row in (col+1)..dim {
            let v = work.get(row, col).abs();
            if v > max_val { max_val = v; max_row = row; }
        }
        if max_val < 1e-12 { continue; }
        // Swap rows
        if max_row != col {
            for j in 0..2*dim {
                let a = work.get(col, j);
                let b = work.get(max_row, j);
                work.set(col, j, b);
                work.set(max_row, j, a);
            }
        }
        // Scale pivot row
        let pivot = work.get(col, col);
        for j in 0..2*dim { let v = work.get(col, j); work.set(col, j, v / pivot); }
        // Eliminate
        for row in 0..dim {
            if row == col { continue; }
            let factor = work.get(row, col);
            if factor.abs() < 1e-15 { continue; }
            for j in 0..2*dim {
                let v = work.get(row, j) - factor * work.get(col, j);
                work.set(row, j, v);
            }
        }
    }
    for i in 0..dim {
        for j in 0..dim { inv.set(i, j, work.get(i, dim + j)); }
    }

    // A* = C·Q^T · (Q·Q^T)^{-1}
    let a_star = cqt.mul(&inv);
    println!("Target matrix computed in {:.2}s", t1.elapsed().as_secs_f64());

    // ── Residual ──
    let mut residual = Mat::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            residual.set(i, j, a_star.get(i, j) - pmi.get(i, j));
        }
    }
    let res_norm = residual.frobenius();
    let pmi_f = pmi.frobenius();
    println!("‖PMI‖ = {:.2}, ‖Residual‖ = {:.2}, ratio = {:.1}%",
        pmi_f, res_norm, res_norm / pmi_f * 100.0);

    // ── SVD of residual ──
    println!();
    println!("Computing SVD of residual matrix...");
    let t2 = Instant::now();

    let ranks_to_test = [1, 2, 4, 8, 16, 32, 64];
    let max_rank = *ranks_to_test.last().unwrap();
    let (u_full, sigmas, v_full) = svd_top_k(&residual, max_rank, 200);

    println!("SVD computed in {:.2}s", t2.elapsed().as_secs_f64());
    println!();

    // Show singular values
    println!("Top singular values:");
    for (i, s) in sigmas.iter().enumerate().take(16) {
        let bar_len = (s / sigmas[0] * 40.0) as usize;
        println!("  σ_{:<2} = {:>8.2}  {}", i+1, s, "█".repeat(bar_len));
    }

    // Show energy captured
    let total_energy: f64 = sigmas.iter().map(|s| s * s).sum();
    println!();
    println!("Energy captured by rank:");
    let mut cumulative = 0.0;
    for &r in &ranks_to_test {
        cumulative = sigmas.iter().take(r).map(|s| s * s).sum();
        println!("  rank {:>2}: {:.1}% of residual energy", r, cumulative / total_energy * 100.0);
    }

    // ── Evaluate each rank ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RESULTS: SVD (one-shot) vs SGD (60K steps)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  {:>6} | {:>8} | {:>10} | {:>15} | {:>8}",
        "Rank", "Params", "SVD Acc", "SGD Acc (prev)", "Winner");
    println!("  ────────────────────────────────────────────────────────────────");

    // SGD results from pmi_residual_sim.rs for comparison
    let sgd_results: Vec<(usize, f64)> = vec![
        (0, 71.5),    // PMI only
        (8, 98.2),    // rank 8 SGD
        (32, 98.2),   // rank 32 SGD
    ];

    let pmi_only_acc = evaluate_pmi_only(&test, &pmi, &centroids);
    println!("  {:>6} | {:>8} | {:>9.1}% | {:>13.1}% | {}",
        0, 0, pmi_only_acc, 71.5, "—");

    for &r in &ranks_to_test {
        let acc = evaluate_svd(&test, &pmi, &u_full, &sigmas[..r], &v_full, &centroids);
        let params = 2 * dim * r;

        let sgd_acc = sgd_results.iter()
            .find(|(rank, _)| *rank == r)
            .map(|(_, a)| format!("{:.1}%", a))
            .unwrap_or("—".to_string());

        let winner = if sgd_results.iter().any(|(rank, a)| *rank == r && acc > *a) {
            "SVD ✓"
        } else if sgd_results.iter().any(|(rank, a)| *rank == r && acc < *a - 0.5) {
            "SGD"
        } else if sgd_results.iter().any(|(rank, _)| *rank == r) {
            "≈ tie"
        } else {
            "—"
        };

        println!("  {:>6} | {:>8} | {:>9.1}% | {:>14} | {}",
            r, params, acc, sgd_acc, winner);
    }

    // ── Timing comparison ──
    let total_svd_time = t0.elapsed().as_secs_f64();
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  TIMING");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  SVD (this run):     {:.2}s  (PMI + target + invert + SVD + eval)", total_svd_time);
    println!("  SGD (previous run): 33.6s  (60K gradient descent steps)");
    println!("  Speedup: {:.0}x", 33.6 / total_svd_time);

    // ── Verdict ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  VERDICT");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let svd_r8 = evaluate_svd(&test, &pmi, &u_full, &sigmas[..8], &v_full, &centroids);
    if svd_r8 >= 97.5 {
        println!("  SVD rank-8 achieves {:.1}% — matches or beats SGD's 98.2%.", svd_r8);
        println!("  ZERO gradients needed. ZERO training iterations.");
        println!("  The architecture learns from STATISTICS ALONE:");
        println!("    1. Count co-occurrences → PMI (seconds)");
        println!("    2. Factorize the residual → SVD (seconds)");
        println!("    3. Done. No loss function. No optimizer. No epochs.");
    } else {
        println!("  SVD rank-8: {:.1}% vs SGD rank-8: 98.2%", svd_r8);
        if svd_r8 > 90.0 {
            println!("  SVD is competitive but SGD still wins. The iterative");
            println!("  refinement of SGD captures something SVD misses.");
        } else {
            println!("  SVD underperforms. The closed-form solution doesn't");
            println!("  capture the nonlinear optimization landscape that SGD finds.");
        }
    }

    // ══════════════════════════════════════════════════════════════
    // EXPERIMENT 2: SVD + Online Corrections (Perceptron-style)
    //
    // Start from SVD solution. For each mistake, apply a rank-1
    // correction: push output toward correct intent, away from wrong.
    // No loss function. No gradient. Just "fix this mistake."
    // Periodically re-project to low rank via SVD.
    // ══════════════════════════════════════════════════════════════

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 2: SVD + Online Corrections (no gradients)           ║");
    println!("║  Perceptron-style: fix each mistake with a rank-1 update         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Build the full residual matrix from SVD at rank=32 (starting point)
    // A = PMI + U · diag(Σ) · V^T
    let start_rank = 32;
    let mut attn = Mat::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let mut residual_val = 0.0;
            for r in 0..start_rank {
                residual_val += u_full.get(i, r) * sigmas[r] * v_full.get(j, r);
            }
            attn.set(i, j, pmi.get(i, j) + residual_val);
        }
    }

    let svd_start_acc = full_eval(&test, &attn, &centroids);
    println!("  Starting from SVD rank-{}: {:.1}%", start_rank, svd_start_acc);
    println!("  Applying perceptron corrections (no gradients)...");
    println!();

    let correction_lr = 0.01;
    let n_passes = 5; // passes through training data
    let eval_every_pass = 1;

    println!("  {:>6} | {:>10} | {:>8} | {:>10}",
        "Pass", "Mistakes", "Accuracy", "Corrections");
    println!("  ─────────────────────────────────────────────────");

    let t_corr = Instant::now();
    let mut total_corrections = 0usize;

    for pass in 0..n_passes {
        let mut mistakes = 0;

        // Shuffle order each pass
        let mut order: Vec<usize> = (0..train.len()).collect();
        for i in (1..order.len()).rev() {
            rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
            let j = rng_state as usize % (i + 1);
            order.swap(i, j);
        }

        for &idx in &order {
            let (ref q, true_intent) = train[idx];

            // Classify with current attention matrix
            let mut proj = vec![0.0; dim];
            for &(j, qj) in q {
                for i in 0..dim { proj[i] += attn.get(i, j) * qj; }
            }
            let scores: Vec<f64> = centroids.iter().map(|c| dot(c, &proj)).collect();
            let pred = scores.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap().0;

            if pred != true_intent {
                mistakes += 1;
                total_corrections += 1;

                // Perceptron correction:
                // Push output toward correct centroid, away from wrong centroid
                // correction_direction = centroid[correct] - centroid[wrong]
                // A += lr * correction_direction ⊗ query
                let c_correct = &centroids[true_intent];
                let c_wrong = &centroids[pred];

                for &(j, qj) in q {
                    let lrq = correction_lr * qj;
                    for i in 0..dim {
                        let correction = lrq * (c_correct[i] - c_wrong[i]);
                        attn.add(i, j, correction);
                    }
                }
            }
        }

        let acc = full_eval(&test, &attn, &centroids);
        println!("  {:>6} | {:>10} | {:>7.1}% | {:>10}",
            pass + 1, mistakes, acc, total_corrections);
    }

    let corr_time = t_corr.elapsed().as_secs_f64();
    let final_corr_acc = full_eval(&test, &attn, &centroids);

    println!();
    println!("  Correction time: {:.2}s ({} total corrections)", corr_time, total_corrections);

    // ── Now project corrected matrix back to low rank ──
    println!();
    println!("  Projecting corrected matrix back to low rank via SVD...");

    // Extract the learned residual
    let mut learned_residual = Mat::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            learned_residual.set(i, j, attn.get(i, j) - pmi.get(i, j));
        }
    }

    let (u_corr, sig_corr, v_corr) = svd_top_k(&learned_residual, 32, 200);

    println!();
    println!("  Post-correction accuracy at each rank:");
    println!("  {:>6} | {:>13} | {:>13} | {:>10}", "Rank", "SVD-only", "SVD+Correct", "Δ");
    println!("  ──────────────────────────────────────────────────────");

    for &r in &[1, 2, 4, 8, 16, 32] {
        let svd_only = evaluate_svd(&test, &pmi, &u_full, &sigmas[..r], &v_full, &centroids);
        let svd_corr = evaluate_svd(&test, &pmi, &u_corr, &sig_corr[..r], &v_corr, &centroids);
        let delta = svd_corr - svd_only;
        println!("  {:>6} | {:>12.1}% | {:>12.1}% | {:>+9.1}%", r, svd_only, svd_corr, delta);
    }

    // ── Final summary ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  FINAL COMPARISON");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  Method                              | Accuracy | Gradients?");
    println!("  ──────────────────────────────────────────────────────────────");
    println!("  PMI only (counting)                 | {:>7.1}% | No", pmi_only_acc);
    println!("  SVD rank-8 (factorization)          | {:>7.1}% | No", evaluate_svd(&test, &pmi, &u_full, &sigmas[..8], &v_full, &centroids));
    println!("  SVD rank-32 (factorization)         | {:>7.1}% | No", evaluate_svd(&test, &pmi, &u_full, &sigmas[..32], &v_full, &centroids));
    println!("  SVD + corrections (full matrix)     | {:>7.1}% | No", final_corr_acc);

    let corr_r8 = evaluate_svd(&test, &pmi, &u_corr, &sig_corr[..8], &v_corr, &centroids);
    let corr_r32 = evaluate_svd(&test, &pmi, &u_corr, &sig_corr[..32], &v_corr, &centroids);
    println!("  SVD + corrections → rank-8          | {:>7.1}% | No", corr_r8);
    println!("  SVD + corrections → rank-32         | {:>7.1}% | No", corr_r32);
    println!("  ──────────────────────────────────────────────────────────────");
    println!("  SGD rank-8 (gradient descent)       | {:>7.1}% | YES", 98.2);
    println!("  SGD full matrix (gradient descent)  | {:>7.1}% | YES", 93.8);
    println!();

    let total_time = t0.elapsed().as_secs_f64();

    if corr_r8 > 95.0 {
        println!("  SVD + corrections at rank-8: {:.1}% — CLOSES THE GAP.", corr_r8);
        println!("  No gradients. No loss function. No backpropagation.");
        println!("  Just: counting → factorizing → correcting mistakes.");
    } else if final_corr_acc > 95.0 {
        println!("  Full-matrix corrections reach {:.1}% — gap closed!", final_corr_acc);
        println!("  But low-rank projection loses accuracy ({:.1}% at rank-8).", corr_r8);
        println!("  The discriminative corrections don't compress as well as SGD's.");
    } else {
        println!("  Best gradient-free: {:.1}%. SGD still leads at 98.2%.", final_corr_acc.max(corr_r8));
        println!("  The discriminative refinement helps but doesn't fully close the gap.");
    }

    // ══════════════════════════════════════════════════════════════
    // EXPERIMENT 3: Rank-Constrained Corrections
    // Correct U and V directly instead of the full matrix.
    // ══════════════════════════════════════════════════════════════

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 3: Rank-Constrained Corrections                     ║");
    println!("║  Update U/V directly — keep everything in rank-K subspace       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    for &test_rank in &[8, 16, 32] {
        // Start from SVD at this rank
        let mut u_r = Mat::zeros(dim, test_rank);
        let mut v_r = Mat::zeros(dim, test_rank);
        for i in 0..dim {
            for r in 0..test_rank {
                u_r.set(i, r, u_full.get(i, r) * sigmas[r].sqrt());
                v_r.set(i, r, v_full.get(i, r) * sigmas[r].sqrt());
            }
        }
        // A = PMI + U_r · V_r^T

        let eval_rank = |u: &Mat, v: &Mat| -> f64 {
            let c = test.iter().filter(|(q, t)| {
                let mut proj = vec![0.0; dim];
                // PMI part
                for &(j, qj) in q.iter() {
                    for i in 0..dim { proj[i] += pmi.get(i, j) * qj; }
                }
                // Low-rank part: U · V^T · q
                let mut z = vec![0.0; test_rank];
                for &(j, qj) in q.iter() {
                    for r in 0..test_rank { z[r] += v.get(j, r) * qj; }
                }
                for i in 0..dim {
                    for r in 0..test_rank { proj[i] += u.get(i, r) * z[r]; }
                }
                let pred = centroids.iter().enumerate()
                    .map(|(k, c)| (k, dot(c, &proj)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap().0;
                pred == *t
            }).count();
            c as f64 / test.len() as f64 * 100.0
        };

        let start_acc = eval_rank(&u_r, &v_r);
        let corr_lr = 0.01;
        let mut total_corr = 0usize;

        for pass in 0..5 {
            let mut order: Vec<usize> = (0..train.len()).collect();
            for i in (1..order.len()).rev() {
                rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
                let j = rng_state as usize % (i + 1);
                order.swap(i, j);
            }

            for &idx in &order {
                let (ref q, true_intent) = train[idx];

                // Classify: proj = (PMI + U·V^T) · q
                let mut proj = vec![0.0; dim];
                for &(j, qj) in q {
                    for i in 0..dim { proj[i] += pmi.get(i, j) * qj; }
                }
                let mut z = vec![0.0; test_rank];
                for &(j, qj) in q {
                    for r in 0..test_rank { z[r] += v_r.get(j, r) * qj; }
                }
                for i in 0..dim {
                    for r in 0..test_rank { proj[i] += u_r.get(i, r) * z[r]; }
                }

                let scores: Vec<f64> = centroids.iter().map(|c| dot(c, &proj)).collect();
                let pred = scores.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap().0;

                if pred != true_intent {
                    total_corr += 1;
                    let c_correct = &centroids[true_intent];
                    let c_wrong = &centroids[pred];

                    // Correction direction: d = c_correct - c_wrong
                    // We want: (U + ΔU)·V^T·q to move toward d
                    // Project d onto U-space: d_proj = d (update U rows)
                    // Project q onto V-space: q is already projected as z = V^T·q
                    //
                    // ΔU: for each rank dimension r, update U[:,r] += lr * d[i] * z[r]
                    // ΔV: for each rank dimension r, update V[:,r] += lr * q[j] * (U^T·d)[r]

                    // U^T · d
                    let mut utd = vec![0.0; test_rank];
                    for i in 0..dim {
                        let di = c_correct[i] - c_wrong[i];
                        for r in 0..test_rank { utd[r] += u_r.get(i, r) * di; }
                    }

                    // Update U: U += lr * d · z^T
                    for i in 0..dim {
                        let di = corr_lr * (c_correct[i] - c_wrong[i]);
                        for r in 0..test_rank {
                            u_r.add(i, r, di * z[r]);
                        }
                    }

                    // Update V: V += lr * q · (U^T·d)^T
                    for &(j, qj) in q {
                        let lrq = corr_lr * qj;
                        for r in 0..test_rank {
                            v_r.add(j, r, lrq * utd[r]);
                        }
                    }
                }
            }
        }

        let final_acc = eval_rank(&u_r, &v_r);
        println!("  Rank {:>2}: {:.1}% → {:.1}%  ({} corrections, {} params)",
            test_rank, start_acc, final_acc, total_corr, 2 * dim * test_rank);
    }

    println!();
    println!("  Reference: SGD rank-8 = 98.2%, SGD full = 93.8%");
    println!();
    println!("Total time: {:.2}s", t0.elapsed().as_secs_f64());
}
