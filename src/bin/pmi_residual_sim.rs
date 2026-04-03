//! PMI Residual Attention Simulation
//!
//! Tests three levels of PMI integration:
//!   A: Full matrix, random init (baseline) — 90K params
//!   B: Residual: PMI_fixed + Delta — 90K learnable, PMI never destroyed
//!   C: Low-rank residual rank=8: PMI + U·W^T — 4,800 params (5.3%)
//!   D: Low-rank residual rank=32: PMI + U·W^T — 19,200 params (21.3%)
//!   E: Router 70%: top 70% tokens frozen to PMI, learn rest — ~27K params (30%)
//!
//! The key question: can PMI stay active DURING learning, not just at init?

use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ── PRNG ──
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.max(1)) }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f64(&mut self) -> f64 { (self.next_u64() % 10_000_000) as f64 / 10_000_000.0 }
    fn normal(&mut self, std: f64) -> f64 {
        let u1 = self.f64().max(1e-10);
        let u2 = self.f64();
        std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn shuffle_indices(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        for i in (1..v.len()).rev() {
            let j = self.next_u64() as usize % (i + 1);
            v.swap(i, j);
        }
        v
    }
}

// ── Data ──
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

// ── Dense Matrix ──
struct Matrix { dim: usize, data: Vec<f64> }
impl Matrix {
    fn zeros(dim: usize) -> Self { Matrix { dim, data: vec![0.0; dim * dim] } }
    fn random(dim: usize, rng: &mut Rng) -> Self {
        let std = 1.0 / (dim as f64).sqrt();
        let mut m = Self::zeros(dim);
        for v in m.data.iter_mut() { *v = rng.normal(std); }
        m
    }
    fn get(&self, i: usize, j: usize) -> f64 { self.data[i * self.dim + j] }
    fn set(&mut self, i: usize, j: usize, v: f64) { self.data[i * self.dim + j] = v; }
    fn add(&mut self, i: usize, j: usize, v: f64) { self.data[i * self.dim + j] += v; }
    fn mul_sparse(&self, x: &[(usize, f64)]) -> Vec<f64> {
        let mut result = vec![0.0; self.dim];
        for &(j, xj) in x {
            for i in 0..self.dim { result[i] += self.data[i * self.dim + j] * xj; }
        }
        result
    }
    fn frobenius_norm(&self) -> f64 { self.data.iter().map(|v| v * v).sum::<f64>().sqrt() }
    fn clone_matrix(&self) -> Self { Matrix { dim: self.dim, data: self.data.clone() } }
    fn normalize_to(&mut self, target: f64) {
        let n = self.frobenius_norm();
        if n > 0.0 { let s = target / n; for v in self.data.iter_mut() { *v *= s; } }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

// ── Track A: Full matrix (random or PMI init) ──
fn full_classify(q: &SparseVec, mat: &Matrix, centroids: &[Vec<f64>]) -> (usize, Vec<f64>) {
    let proj = mat.mul_sparse(q);
    let scores: Vec<f64> = centroids.iter().map(|c| dot(c, &proj)).collect();
    let probs = softmax(&scores);
    let pred = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    (pred, probs)
}

fn full_sgd(mat: &mut Matrix, q: &SparseVec, target: usize, centroids: &[Vec<f64>], lr: f64) {
    let (_, probs) = full_classify(q, mat, centroids);
    for (k, c) in centroids.iter().enumerate() {
        let err = probs[k] - if k == target { 1.0 } else { 0.0 };
        if err.abs() < 1e-8 { continue; }
        let f = -lr * err;
        for &(j, qj) in q {
            let fq = f * qj;
            for i in 0..mat.dim { mat.add(i, j, fq * c[i]); }
        }
    }
}

fn full_eval(data: &[(SparseVec, usize)], mat: &Matrix, centroids: &[Vec<f64>]) -> f64 {
    let c = data.iter().filter(|(q, t)| full_classify(q, mat, centroids).0 == *t).count();
    c as f64 / data.len() as f64 * 100.0
}

// ── Track B: Residual (PMI_fixed + Delta) ──
// Score = c^T · (PMI + Delta) · q = pmi_score + c^T · Delta · q
// SGD only on Delta. PMI is never modified.

fn residual_scores(q: &SparseVec, delta: &Matrix, pmi_scores: &[f64], centroids: &[Vec<f64>]) -> Vec<f64> {
    let delta_proj = delta.mul_sparse(q);
    centroids.iter().enumerate()
        .map(|(k, c)| pmi_scores[k] + dot(c, &delta_proj))
        .collect()
}

fn residual_classify(q: &SparseVec, delta: &Matrix, pmi_scores: &[f64], centroids: &[Vec<f64>]) -> (usize, Vec<f64>) {
    let scores = residual_scores(q, delta, pmi_scores, centroids);
    let probs = softmax(&scores);
    let pred = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    (pred, probs)
}

fn residual_sgd(delta: &mut Matrix, q: &SparseVec, target: usize, pmi_scores: &[f64], centroids: &[Vec<f64>], lr: f64) {
    let (_, probs) = residual_classify(q, delta, pmi_scores, centroids);
    for (k, c) in centroids.iter().enumerate() {
        let err = probs[k] - if k == target { 1.0 } else { 0.0 };
        if err.abs() < 1e-8 { continue; }
        let f = -lr * err;
        for &(j, qj) in q {
            let fq = f * qj;
            for i in 0..delta.dim { delta.add(i, j, fq * c[i]); }
        }
    }
}

fn residual_eval(data: &[(SparseVec, usize)], delta: &Matrix, pmi_scores_all: &[Vec<f64>], centroids: &[Vec<f64>]) -> f64 {
    let c = data.iter().enumerate()
        .filter(|(i, (q, t))| residual_classify(q, delta, &pmi_scores_all[*i], centroids).0 == *t)
        .count();
    c as f64 / data.len() as f64 * 100.0
}

// ── Track C/D: Low-rank residual (PMI + U·W^T) ──
// Score = pmi_score + (U^T · c)^T · (W^T · q)
// Parameters: U (dim×rank) + W (dim×rank) = 2·dim·rank

struct LowRank {
    u: Vec<f64>,    // dim × rank, row-major
    w: Vec<f64>,    // dim × rank, row-major
    rank: usize,
    dim: usize,
}

impl LowRank {
    fn new(dim: usize, rank: usize, rng: &mut Rng) -> Self {
        let std = 1.0 / (rank as f64).sqrt();
        let n = dim * rank;
        let mut u = vec![0.0; n];
        let mut w = vec![0.0; n];
        for v in u.iter_mut() { *v = rng.normal(std) * 0.1; }
        for v in w.iter_mut() { *v = rng.normal(std) * 0.1; }
        LowRank { u, w, rank, dim }
    }

    fn params(&self) -> usize { 2 * self.dim * self.rank }

    // W^T · q → rank-dim vector (sparse q)
    fn project_q(&self, q: &SparseVec) -> Vec<f64> {
        let mut p = vec![0.0; self.rank];
        for &(j, qj) in q {
            for l in 0..self.rank {
                p[l] += self.w[j * self.rank + l] * qj;
            }
        }
        p
    }

    // U^T · c → rank-dim vector
    fn project_centroid(&self, c: &[f64]) -> Vec<f64> {
        let mut r = vec![0.0; self.rank];
        for i in 0..self.dim {
            if c[i] == 0.0 { continue; }
            for l in 0..self.rank {
                r[l] += self.u[i * self.rank + l] * c[i];
            }
        }
        r
    }

    fn classify(&self, q: &SparseVec, pmi_scores: &[f64], centroids: &[Vec<f64>]) -> (usize, Vec<f64>) {
        let proj_q = self.project_q(q);
        let scores: Vec<f64> = centroids.iter().enumerate().map(|(k, c)| {
            let proj_c = self.project_centroid(c);
            pmi_scores[k] + dot(&proj_c, &proj_q)
        }).collect();
        let probs = softmax(&scores);
        let pred = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        (pred, probs)
    }

    fn sgd(&mut self, q: &SparseVec, target: usize, pmi_scores: &[f64], centroids: &[Vec<f64>], lr: f64) {
        let proj_q = self.project_q(q);
        let scores: Vec<f64> = centroids.iter().enumerate().map(|(k, c)| {
            let proj_c = self.project_centroid(c);
            pmi_scores[k] + dot(&proj_c, &proj_q)
        }).collect();
        let probs = softmax(&scores);

        // grad_signal = sum_k e_k · c_k
        let mut grad_signal = vec![0.0; self.dim];
        for (k, c) in centroids.iter().enumerate() {
            let err = probs[k] - if k == target { 1.0 } else { 0.0 };
            if err.abs() < 1e-8 { continue; }
            for i in 0..self.dim { grad_signal[i] += err * c[i]; }
        }

        // h = U^T · grad_signal
        let mut h = vec![0.0; self.rank];
        for i in 0..self.dim {
            if grad_signal[i] == 0.0 { continue; }
            for l in 0..self.rank {
                h[l] += self.u[i * self.rank + l] * grad_signal[i];
            }
        }

        // dL/dU = grad_signal ⊗ proj_q (outer product)
        for i in 0..self.dim {
            if grad_signal[i] == 0.0 { continue; }
            for l in 0..self.rank {
                self.u[i * self.rank + l] -= lr * grad_signal[i] * proj_q[l];
            }
        }

        // dL/dW: only for non-zero q entries
        for &(j, qj) in q {
            for l in 0..self.rank {
                self.w[j * self.rank + l] -= lr * qj * h[l];
            }
        }
    }

    fn eval(&self, data: &[(SparseVec, usize)], pmi_scores_all: &[Vec<f64>], centroids: &[Vec<f64>]) -> f64 {
        let c = data.iter().enumerate()
            .filter(|(i, (q, t))| self.classify(q, &pmi_scores_all[*i], centroids).0 == *t)
            .count();
        c as f64 / data.len() as f64 * 100.0
    }
}

// ── Track E: Router (top-K tokens frozen to PMI, rest learned) ──
struct Router {
    matrix: Matrix,
    frozen: Vec<bool>,
    n_frozen: usize,
}

impl Router {
    fn new(pmi: &Matrix, skip_rate: f64, rng: &mut Rng) -> Self {
        let dim = pmi.dim;
        // PMI confidence = sum of positive PMI edges for each token
        let mut conf: Vec<(usize, f64)> = (0..dim).map(|i| {
            let s: f64 = (0..dim).map(|j| pmi.get(i, j).max(0.0)).sum();
            (i, s)
        }).collect();
        conf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let n_frozen = (dim as f64 * skip_rate) as usize;
        let mut frozen = vec![false; dim];
        for &(idx, _) in conf.iter().take(n_frozen) {
            frozen[idx] = true;
        }

        // Init: frozen rows = PMI, learned rows = small random
        let mut matrix = pmi.clone_matrix();
        let std = 0.01;
        for i in 0..dim {
            if !frozen[i] {
                for j in 0..dim {
                    matrix.set(i, j, pmi.get(i, j) + rng.normal(std));
                }
            }
        }

        Router { matrix, frozen, n_frozen }
    }

    fn effective_params(&self) -> usize {
        (self.matrix.dim - self.n_frozen) * self.matrix.dim
    }

    fn sgd(&mut self, q: &SparseVec, target: usize, centroids: &[Vec<f64>], lr: f64) {
        let (_, probs) = full_classify(q, &self.matrix, centroids);
        for (k, c) in centroids.iter().enumerate() {
            let err = probs[k] - if k == target { 1.0 } else { 0.0 };
            if err.abs() < 1e-8 { continue; }
            let f = -lr * err;
            for &(j, qj) in q {
                let fq = f * qj;
                for i in 0..self.matrix.dim {
                    if !self.frozen[i] {
                        self.matrix.add(i, j, fq * c[i]);
                    }
                }
            }
        }
    }
}

// ── Main ──
fn main() {
    let t0 = Instant::now();
    let mut rng = Rng::new(42);

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  PMI Residual Attention: Can PMI Stay Active During Learning?    ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Load & prep (same as sim1) ──
    let path = "tests/data/benchmarks/bitext_all.json";
    let examples: Vec<Example> = serde_json::from_str(
        &std::fs::read_to_string(path).expect("bitext_all.json not found")
    ).expect("parse failed");

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

    let indices = rng.shuffle_indices(all_data.len());
    let split = all_data.len() * 80 / 100;
    let train: Vec<(SparseVec, usize)> = indices[..split].iter().map(|&i| all_data[i].clone()).collect();
    let test: Vec<(SparseVec, usize)> = indices[split..].iter().map(|&i| all_data[i].clone()).collect();

    // PMI matrix
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
    let mut pmi_matrix = Matrix::zeros(dim);
    for (&(a, b), &count) in &cooccur {
        let p_ab = count / n;
        let p_a = term_df.get(&a).copied().unwrap_or(1.0) / n;
        let p_b = term_df.get(&b).copied().unwrap_or(1.0) / n;
        let pmi = (p_ab / (p_a * p_b)).ln();
        if pmi > 0.0 {
            let v = pmi.min(5.0);
            pmi_matrix.set(a, b, v);
            pmi_matrix.set(b, a, v);
        }
    }

    // Centroids
    let mut centroids = vec![vec![0.0f64; dim]; n_intents];
    let mut counts = vec![0usize; n_intents];
    for (q, intent) in &train {
        for &(i, v) in q { centroids[*intent][i] += v; }
        counts[*intent] += 1;
    }
    for k in 0..n_intents {
        if counts[k] > 0 { let c = counts[k] as f64; for v in centroids[k].iter_mut() { *v /= c; } }
    }

    // Normalize PMI
    let ref_norm = Matrix::random(dim, &mut rng).frobenius_norm();
    let mut pmi_scaled = pmi_matrix.clone_matrix();
    pmi_scaled.normalize_to(ref_norm);

    // ── Precompute PMI scores ──
    println!("Data: {} train, {} test, {} intents, dim={}", train.len(), test.len(), n_intents, dim);
    println!("Precomputing PMI projections...");

    let pmi_scores_train: Vec<Vec<f64>> = train.iter().map(|(q, _)| {
        let proj = pmi_scaled.mul_sparse(q);
        centroids.iter().map(|c| dot(c, &proj)).collect()
    }).collect();

    let pmi_scores_test: Vec<Vec<f64>> = test.iter().map(|(q, _)| {
        let proj = pmi_scaled.mul_sparse(q);
        centroids.iter().map(|c| dot(c, &proj)).collect()
    }).collect();

    // ── Initialize all tracks ──
    // A: Full random
    let mut mat_random = Matrix::random(dim, &mut rng);
    mat_random.normalize_to(ref_norm * 0.1);

    // B: Residual (PMI + Delta)
    let mut delta = Matrix::zeros(dim);

    // C: Low-rank rank=8
    let mut lr8 = LowRank::new(dim, 8, &mut rng);

    // D: Low-rank rank=32
    let mut lr32 = LowRank::new(dim, 32, &mut rng);

    // E: Router 70%
    let mut router70 = Router::new(&pmi_scaled, 0.70, &mut rng);

    println!();
    println!("  Track                          | Learnable Params | % of Full");
    println!("  ─────────────────────────────────────────────────────────────");
    println!("  A: Full matrix (random init)   | {:>16} | {:>8}", dim*dim, "100%");
    println!("  B: Residual (PMI + Delta)      | {:>16} | {:>8}", dim*dim, "100%");
    println!("  C: Low-rank rank=8 (PMI + UW^T)| {:>16} | {:>7.1}%", lr8.params(), lr8.params() as f64 / (dim*dim) as f64 * 100.0);
    println!("  D: Low-rank rank=32(PMI + UW^T)| {:>16} | {:>7.1}%", lr32.params(), lr32.params() as f64 / (dim*dim) as f64 * 100.0);
    println!("  E: Router 70% skip (PMI+learn) | {:>16} | {:>7.1}%", router70.effective_params(), router70.effective_params() as f64 / (dim*dim) as f64 * 100.0);

    // ── Static baselines ──
    println!();
    let pmi_static = full_eval(&test, &pmi_scaled, &centroids);
    println!("  PMI static (no learning): {:.1}%", pmi_static);

    // ── Training ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  LEARNING CURVES");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let lr_full = 0.005;
    let lr_lowrank = 0.02;
    let lr_router = 0.005;
    let n_epochs = 3;
    let eval_every = 3000;

    struct Curve { points: Vec<(usize, f64)> }
    impl Curve {
        fn new() -> Self { Curve { points: Vec::new() } }
        fn add(&mut self, step: usize, acc: f64) { self.points.push((step, acc)); }
        fn steps_to(&self, thresh: f64) -> Option<usize> {
            self.points.iter().find(|(_, a)| *a >= thresh).map(|(s, _)| *s)
        }
        fn final_acc(&self) -> f64 { self.points.last().map(|(_, a)| *a).unwrap_or(0.0) }
    }

    let mut curve_a = Curve::new();
    let mut curve_b = Curve::new();
    let mut curve_c = Curve::new();
    let mut curve_d = Curve::new();
    let mut curve_e = Curve::new();

    // Initial eval
    let ea = full_eval(&test, &mat_random, &centroids);
    let eb = residual_eval(&test, &delta, &pmi_scores_test, &centroids);
    let ec = lr8.eval(&test, &pmi_scores_test, &centroids);
    let ed = lr32.eval(&test, &pmi_scores_test, &centroids);
    let ee = full_eval(&test, &router70.matrix, &centroids);

    curve_a.add(0, ea); curve_b.add(0, eb); curve_c.add(0, ec); curve_d.add(0, ed); curve_e.add(0, ee);

    println!("  {:>6} | {:>8} | {:>8} | {:>10} | {:>10} | {:>10}",
        "Step", "A:Random", "B:Resid", "C:LR-8", "D:LR-32", "E:Route70");
    println!("  {:>6} | {:>8} | {:>8} | {:>10} | {:>10} | {:>10}",
        "", "(90Kp)", "(90Kp)", format!("({}p)", lr8.params()), format!("({}p)", lr32.params()), format!("({}p)", router70.effective_params()));
    println!("  ────────────────────────────────────────────────────────────────");
    println!("  {:>6} | {:>7.1}% | {:>7.1}% | {:>9.1}% | {:>9.1}% | {:>9.1}%",
        0, ea, eb, ec, ed, ee);

    let t_learn = Instant::now();
    let mut global_step = 0;

    for _epoch in 0..n_epochs {
        let order = rng.shuffle_indices(train.len());
        for &idx in &order {
            let (ref q, target) = train[idx];
            let pmi_s = &pmi_scores_train[idx];

            // Update all tracks
            full_sgd(&mut mat_random, q, target, &centroids, lr_full);
            residual_sgd(&mut delta, q, target, pmi_s, &centroids, lr_full);
            lr8.sgd(q, target, pmi_s, &centroids, lr_lowrank);
            lr32.sgd(q, target, pmi_s, &centroids, lr_lowrank);
            router70.sgd(q, target, &centroids, lr_router);

            global_step += 1;

            if global_step % eval_every == 0 {
                let ea = full_eval(&test, &mat_random, &centroids);
                let eb = residual_eval(&test, &delta, &pmi_scores_test, &centroids);
                let ec = lr8.eval(&test, &pmi_scores_test, &centroids);
                let ed = lr32.eval(&test, &pmi_scores_test, &centroids);
                let ee = full_eval(&test, &router70.matrix, &centroids);

                curve_a.add(global_step, ea);
                curve_b.add(global_step, eb);
                curve_c.add(global_step, ec);
                curve_d.add(global_step, ed);
                curve_e.add(global_step, ee);

                println!("  {:>6} | {:>7.1}% | {:>7.1}% | {:>9.1}% | {:>9.1}% | {:>9.1}%",
                    global_step, ea, eb, ec, ed, ee);
            }
        }
    }

    println!();
    println!("  Time: {:.1}s ({} steps, {:.0} steps/sec)",
        t_learn.elapsed().as_secs_f64(), global_step,
        global_step as f64 / t_learn.elapsed().as_secs_f64());

    // ── Analysis ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  CONVERGENCE SPEED");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let thresholds = [60.0, 70.0, 75.0, 80.0, 85.0, 90.0];
    println!("  Steps to reach threshold (and speedup vs Random):");
    println!("  {:>6} | {:>8} | {:>13} | {:>13} | {:>13} | {:>13}",
        "Thresh", "A:Rand", "B:Residual", "C:LR-8", "D:LR-32", "E:Route70");
    println!("  ────────────────────────────────────────────────────────────────────");

    for &th in &thresholds {
        let sa = curve_a.steps_to(th);
        let sb = curve_b.steps_to(th);
        let sc = curve_c.steps_to(th);
        let sd = curve_d.steps_to(th);
        let se = curve_e.steps_to(th);

        let fmt = |s: Option<usize>, base: Option<usize>| -> String {
            match (s, base) {
                (Some(v), Some(b)) if b > 0 => format!("{:>6} ({:.1}x)", v, b as f64 / v as f64),
                (Some(v), _) => format!("{:>6}      ", v),
                (None, _) => "  never      ".to_string(),
            }
        };

        println!("  {:>5.0}% | {:>6} | {} | {} | {} | {}",
            th,
            sa.map(|v| format!("{}", v)).unwrap_or("never".to_string()),
            fmt(sb, sa), fmt(sc, sa), fmt(sd, sa), fmt(se, sa));
    }

    // ── Parameter efficiency ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  PARAMETER EFFICIENCY");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let fa = curve_a.final_acc();
    println!("  Track                    | Params   | Final Acc | % of Full Acc");
    println!("  ─────────────────────────────────────────────────────────────────");
    println!("  A: Full random           | {:>6}   | {:>7.1}%  | {:>8}", dim*dim, fa, "baseline");
    println!("  B: Residual (PMI+Delta)  | {:>6}   | {:>7.1}%  | {:>7.1}%",
        dim*dim, curve_b.final_acc(), curve_b.final_acc() / fa * 100.0);
    println!("  C: Low-rank r=8 (PMI+UW) | {:>6}   | {:>7.1}%  | {:>7.1}%",
        lr8.params(), curve_c.final_acc(), curve_c.final_acc() / fa * 100.0);
    println!("  D: Low-rank r=32(PMI+UW) | {:>6}   | {:>7.1}%  | {:>7.1}%",
        lr32.params(), curve_d.final_acc(), curve_d.final_acc() / fa * 100.0);
    println!("  E: Router 70% skip       | {:>6}   | {:>7.1}%  | {:>7.1}%",
        router70.effective_params(), curve_e.final_acc(), curve_e.final_acc() / fa * 100.0);
    println!("  (PMI static, no learning)| {:>6}   | {:>7.1}%  | {:>7.1}%",
        0, pmi_static, pmi_static / fa * 100.0);

    // ── Residual magnitude ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  RESIDUAL ANALYSIS (Track B)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    let pmi_norm = pmi_scaled.frobenius_norm();
    let delta_norm = delta.frobenius_norm();
    println!("  ‖PMI‖   = {:.2}", pmi_norm);
    println!("  ‖Delta‖ = {:.2}", delta_norm);
    println!("  Ratio ‖Delta‖/‖PMI‖ = {:.1}%", delta_norm / pmi_norm * 100.0);
    println!();
    if delta_norm < pmi_norm {
        println!("  Delta is SMALLER than PMI → the learned correction is a refinement,");
        println!("  not a replacement. PMI carries the majority of the signal.");
    } else {
        println!("  Delta grew larger than PMI → learning overrode the PMI base.");
    }

    // ── Verdict ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  VERDICT");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let fb = curve_b.final_acc();
    if fb > fa + 0.5 {
        println!("  RESIDUAL PMI BEATS RANDOM INIT: {:.1}% vs {:.1}%", fb, fa);
        println!("  → Keeping PMI as permanent base IS better than letting SGD overwrite it.");
    }

    // Check if low-rank is viable
    let fd = curve_d.final_acc();
    if fd > fa * 0.95 {
        println!("  LOW-RANK r=32 ({} params) reaches {:.1}% of full accuracy.",
            lr32.params(), fd / fa * 100.0);
        println!("  → {:.0}% parameter reduction with <{:.1}% accuracy loss!",
            (1.0 - lr32.params() as f64 / (dim*dim) as f64) * 100.0,
            (fa - fd).max(0.0));
    }

    // Check router
    let fe = curve_e.final_acc();
    if fe > fa * 0.95 {
        println!("  ROUTER 70% ({} params) reaches {:.1}% of full accuracy.",
            router70.effective_params(), fe / fa * 100.0);
        println!("  → 70% of tokens need NO learned attention at all.");
    }

    // Overall thesis
    let any_beat_random = fb > fa || fd > fa * 0.98;
    if any_beat_random {
        println!();
        println!("  ┌─────────────────────────────────────────────────────────┐");
        println!("  │  PMI is NOT just a kickstarter. It's a permanent       │");
        println!("  │  structural foundation that reduces what attention      │");
        println!("  │  needs to learn, both in params and in training time.   │");
        println!("  └─────────────────────────────────────────────────────────┘");
    }

    println!();
    println!("Total: {:.1}s", t0.elapsed().as_secs_f64());
}
