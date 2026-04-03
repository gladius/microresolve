//! Mathematical Simulation: Can PMI Serve as Attention Initialization?
//!
//! Model: Single-layer "attention" for intent classification
//!   score_k = centroid_k^T * A * query    (for each intent k)
//!   prediction = argmax_k score_k
//!   Training: SGD on A via cross-entropy loss
//!
//! Tracks compared:
//!   1. A = I           (identity, no mixing — baseline)
//!   2. A = PMI         (static co-occurrence, zero learning)
//!   3. A = random → SGD     (simulates transformer training from scratch)
//!   4. A = PMI → SGD        (simulates PMI-initialized transformer)
//!   5. A = PMI, learn gate  (simulates gated PMI with tiny parameter count)
//!
//! If Track 4 converges faster than Track 3, PMI initialization works.
//! If Track 5 matches Track 3 with fewer params, gated PMI works.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ============================================================================
// Simple PRNG (xorshift64)
// ============================================================================
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.max(1)) }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f64(&mut self) -> f64 {
        (self.next_u64() % 10_000_000) as f64 / 10_000_000.0
    }
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

// ============================================================================
// Data structures
// ============================================================================

#[derive(serde::Deserialize)]
struct Example {
    text: String,
    intents: Vec<String>,
}

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

// ============================================================================
// Matrix operations (dense, small)
// ============================================================================

/// V×V matrix stored flat
struct Matrix {
    dim: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn zeros(dim: usize) -> Self {
        Matrix { dim, data: vec![0.0; dim * dim] }
    }

    fn identity(dim: usize) -> Self {
        let mut m = Self::zeros(dim);
        for i in 0..dim { m.data[i * dim + i] = 1.0; }
        m
    }

    fn random(dim: usize, rng: &mut Rng) -> Self {
        let std = 1.0 / (dim as f64).sqrt();
        let mut m = Self::zeros(dim);
        for v in m.data.iter_mut() { *v = rng.normal(std); }
        m
    }

    fn get(&self, i: usize, j: usize) -> f64 { self.data[i * self.dim + j] }
    fn set(&mut self, i: usize, j: usize, v: f64) { self.data[i * self.dim + j] = v; }
    fn add(&mut self, i: usize, j: usize, v: f64) { self.data[i * self.dim + j] += v; }

    /// Multiply: result = A * x (sparse x)
    fn mul_sparse(&self, x: &[(usize, f64)]) -> Vec<f64> {
        let mut result = vec![0.0; self.dim];
        for &(j, xj) in x {
            if xj == 0.0 { continue; }
            for i in 0..self.dim {
                result[i] += self.data[i * self.dim + j] * xj;
            }
        }
        result
    }

    fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    fn clone_matrix(&self) -> Self {
        Matrix { dim: self.dim, data: self.data.clone() }
    }

    /// Scale to match target Frobenius norm
    fn normalize_to(&mut self, target_norm: f64) {
        let norm = self.frobenius_norm();
        if norm > 0.0 {
            let scale = target_norm / norm;
            for v in self.data.iter_mut() { *v *= scale; }
        }
    }
}

// ============================================================================
// Attention classifier
// ============================================================================

/// Sparse vector: (index, value) pairs
type SparseVec = Vec<(usize, f64)>;

fn dot_dense(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn dot_sparse_dense(sparse: &SparseVec, dense: &[f64]) -> f64 {
    sparse.iter().map(|(i, v)| v * dense[*i]).sum()
}

/// Stable softmax
fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Classify: score_k = centroid_k · (A * query), return (prediction, probabilities)
fn classify(
    query: &SparseVec,
    matrix: &Matrix,
    centroids: &[Vec<f64>],
) -> (usize, Vec<f64>) {
    let projected = matrix.mul_sparse(query);
    let scores: Vec<f64> = centroids.iter().map(|c| dot_dense(c, &projected)).collect();
    let probs = softmax(&scores);
    let pred = probs.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;
    (pred, probs)
}

/// SGD update on full matrix A
/// gradient: dL/dA = sum_k (prob_k - target_k) * centroid_k ⊗ query
fn sgd_step(
    matrix: &mut Matrix,
    query: &SparseVec,
    true_intent: usize,
    centroids: &[Vec<f64>],
    lr: f64,
) {
    let (_, probs) = classify(query, matrix, centroids);
    for (k, centroid) in centroids.iter().enumerate() {
        let target = if k == true_intent { 1.0 } else { 0.0 };
        let error = probs[k] - target;
        if error.abs() < 1e-8 { continue; }
        let factor = -lr * error;
        for &(j, qj) in query {
            if qj == 0.0 { continue; }
            let fq = factor * qj;
            for i in 0..matrix.dim {
                matrix.add(i, j, fq * centroid[i]);
            }
        }
    }
}

// ============================================================================
// Gated PMI: A = PMI ⊙ (g * g^T), learn g ∈ R^V
// ============================================================================

struct GatedPMI {
    pmi: Matrix,       // fixed PMI matrix
    gate: Vec<f64>,    // learnable per-token gate, V parameters
    dim: usize,
}

impl GatedPMI {
    fn new(pmi: Matrix) -> Self {
        let dim = pmi.dim;
        GatedPMI {
            pmi,
            gate: vec![1.0; dim],
            dim,
        }
    }

    /// Effective matrix: A[i,j] = sigmoid(gate[i]) * PMI[i,j] * sigmoid(gate[j])
    fn effective_value(&self, i: usize, j: usize) -> f64 {
        let gi = sigmoid(self.gate[i]);
        let gj = sigmoid(self.gate[j]);
        gi * self.pmi.get(i, j) * gj
    }

    fn mul_sparse(&self, x: &SparseVec) -> Vec<f64> {
        let mut result = vec![0.0; self.dim];
        for &(j, xj) in x {
            if xj == 0.0 { continue; }
            let gj = sigmoid(self.gate[j]);
            let xjg = xj * gj;
            for i in 0..self.dim {
                let gi = sigmoid(self.gate[i]);
                result[i] += gi * self.pmi.get(i, j) * xjg;
            }
        }
        result
    }

    fn classify(&self, query: &SparseVec, centroids: &[Vec<f64>]) -> (usize, Vec<f64>) {
        let projected = self.mul_sparse(query);
        let scores: Vec<f64> = centroids.iter().map(|c| dot_dense(c, &projected)).collect();
        let probs = softmax(&scores);
        let pred = probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;
        (pred, probs)
    }

    fn sgd_step(&mut self, query: &SparseVec, true_intent: usize, centroids: &[Vec<f64>], lr: f64) {
        let (_, probs) = self.classify(query, centroids);

        // dL/dgate[i] via chain rule through sigmoid
        let mut grad = vec![0.0; self.dim];

        for (k, centroid) in centroids.iter().enumerate() {
            let target = if k == true_intent { 1.0 } else { 0.0 };
            let error = probs[k] - target;
            if error.abs() < 1e-8 { continue; }

            // For each gate[i], collect gradient from row i and column i
            for i in 0..self.dim {
                let gi = sigmoid(self.gate[i]);
                let dsig_i = gi * (1.0 - gi); // sigmoid derivative

                // Row contribution: sum_j centroid[i] * PMI[i,j] * sig(gate[j]) * query[j]
                let mut row_sum = 0.0;
                for &(j, qj) in query {
                    if qj == 0.0 { continue; }
                    row_sum += self.pmi.get(i, j) * sigmoid(self.gate[j]) * qj;
                }
                grad[i] += error * centroid[i] * dsig_i * row_sum;

                // Column contribution (if i is a query term)
                // sum_row centroid[row] * sig(gate[row]) * PMI[row, i] * query[i]
                if let Some(&(_, qi)) = query.iter().find(|(idx, _)| *idx == i) {
                    if qi != 0.0 {
                        let mut col_sum = 0.0;
                        for row in 0..self.dim {
                            col_sum += centroid[row] * sigmoid(self.gate[row]) * self.pmi.get(row, i);
                        }
                        grad[i] += error * qi * dsig_i * col_sum;
                    }
                }
            }
        }

        // Apply gradient
        for i in 0..self.dim {
            self.gate[i] -= lr * grad[i];
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================================
// Evaluation
// ============================================================================

fn evaluate(
    data: &[(SparseVec, usize)],
    matrix: &Matrix,
    centroids: &[Vec<f64>],
) -> f64 {
    let correct = data.iter()
        .filter(|(q, intent)| classify(q, matrix, centroids).0 == *intent)
        .count();
    correct as f64 / data.len() as f64 * 100.0
}

fn evaluate_gated(
    data: &[(SparseVec, usize)],
    gated: &GatedPMI,
    centroids: &[Vec<f64>],
) -> f64 {
    let correct = data.iter()
        .filter(|(q, intent)| gated.classify(q, centroids).0 == *intent)
        .count();
    correct as f64 / data.len() as f64 * 100.0
}

// ============================================================================
// Main simulation
// ============================================================================

fn main() {
    let t0 = Instant::now();
    let mut rng = Rng::new(42);

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  PMI vs Attention: Mathematical Simulation                      ║");
    println!("║  Can PMI initialization speed up attention learning?             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Load data ──
    let path = "tests/data/benchmarks/bitext_all.json";
    let data = std::fs::read_to_string(path).expect("bitext_all.json not found");
    let examples: Vec<Example> = serde_json::from_str(&data).expect("parse failed");
    println!("Loaded {} examples", examples.len());

    // ── Build vocab (top discriminating terms) ──
    let docs: Vec<Vec<String>> = examples.iter().map(|e| tokenize(&e.text)).collect();
    let mut df: HashMap<String, usize> = HashMap::new();
    for doc in &docs {
        let unique: HashSet<&String> = doc.iter().collect();
        for t in unique { *df.entry(t.clone()).or_insert(0) += 1; }
    }
    let mut by_df: Vec<(String, usize)> = df.into_iter()
        .filter(|(_, c)| *c >= 5 && *c < docs.len() / 2)
        .collect();
    by_df.sort_by(|a, b| b.1.cmp(&a.1));
    by_df.truncate(300);
    let dim = by_df.len();

    let term_to_idx: HashMap<String, usize> = by_df.iter().enumerate()
        .map(|(i, (t, _))| (t.clone(), i))
        .collect();

    println!("Vocabulary: {} terms (dim={})", dim, dim);

    // ── Build intent mapping ──
    let mut intent_map: HashMap<String, usize> = HashMap::new();
    for ex in &examples {
        let len = intent_map.len();
        intent_map.entry(ex.intents[0].clone()).or_insert(len);
    }
    let n_intents = intent_map.len();
    println!("Intents: {}", n_intents);

    // ── Convert to sparse vectors ──
    let mut all_data: Vec<(SparseVec, usize)> = Vec::new();
    for ex in &examples {
        let terms = tokenize(&ex.text);
        let sparse: SparseVec = terms.iter()
            .filter_map(|t| term_to_idx.get(t).map(|&i| (i, 1.0)))
            .collect::<HashMap<usize, f64>>()
            .into_iter()
            .collect();
        if sparse.is_empty() { continue; }
        let intent = intent_map[&ex.intents[0]];
        all_data.push((sparse, intent));
    }

    // ── Train/test split (80/20, shuffled) ──
    let indices = rng.shuffle_indices(all_data.len());
    let split = all_data.len() * 80 / 100;
    let train: Vec<(SparseVec, usize)> = indices[..split].iter()
        .map(|&i| all_data[i].clone())
        .collect();
    let test: Vec<(SparseVec, usize)> = indices[split..].iter()
        .map(|&i| all_data[i].clone())
        .collect();
    println!("Train: {}, Test: {}", train.len(), test.len());

    // ── Compute PMI matrix ──
    println!("Computing PMI matrix...");
    let n = docs.len() as f64;
    let mut term_df: HashMap<usize, f64> = HashMap::new();
    let mut cooccur: HashMap<(usize, usize), f64> = HashMap::new();

    for doc in &docs {
        let unique: Vec<usize> = doc.iter()
            .filter_map(|t| term_to_idx.get(t).copied())
            .collect::<HashSet<_>>().into_iter().collect();
        for &idx in &unique {
            *term_df.entry(idx).or_insert(0.0) += 1.0;
        }
        for i in 0..unique.len() {
            for j in (i+1)..unique.len() {
                let key = if unique[i] < unique[j] { (unique[i], unique[j]) } else { (unique[j], unique[i]) };
                *cooccur.entry(key).or_insert(0.0) += 1.0;
            }
        }
    }

    let mut pmi_matrix = Matrix::zeros(dim);
    let mut pmi_nonzero = 0;
    for (&(a, b), &count) in &cooccur {
        let p_ab = count / n;
        let p_a = term_df.get(&a).copied().unwrap_or(1.0) / n;
        let p_b = term_df.get(&b).copied().unwrap_or(1.0) / n;
        let pmi = (p_ab / (p_a * p_b)).ln();
        if pmi > 0.0 {
            let ppmi = pmi.min(5.0);  // cap PPMI
            pmi_matrix.set(a, b, ppmi);
            pmi_matrix.set(b, a, ppmi);
            pmi_nonzero += 2;
        }
    }
    let pmi_density = pmi_nonzero as f64 / (dim * dim) as f64 * 100.0;
    println!("PMI matrix: {}×{}, {:.1}% non-zero ({} entries)", dim, dim, pmi_density, pmi_nonzero);

    // ── Build intent centroids from training data ──
    let mut centroids = vec![vec![0.0f64; dim]; n_intents];
    let mut counts = vec![0usize; n_intents];
    for (q, intent) in &train {
        for &(i, v) in q {
            centroids[*intent][i] += v;
        }
        counts[*intent] += 1;
    }
    for k in 0..n_intents {
        if counts[k] > 0 {
            let c = counts[k] as f64;
            for v in centroids[k].iter_mut() { *v /= c; }
        }
    }

    // ── Normalize PMI to similar scale as random init ──
    let random_ref = Matrix::random(dim, &mut rng);
    let ref_norm = random_ref.frobenius_norm();
    let mut pmi_scaled = pmi_matrix.clone_matrix();
    pmi_scaled.normalize_to(ref_norm);

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  PHASE 1: Static Evaluation (no learning)");
    println!("═══════════════════════════════════════════════════════════════════");

    let identity = Matrix::identity(dim);
    let acc_identity = evaluate(&test, &identity, &centroids);
    let acc_pmi = evaluate(&test, &pmi_scaled, &centroids);
    let acc_random = evaluate(&test, &random_ref, &centroids);

    println!("  Identity (no mixing):    {:.1}%", acc_identity);
    println!("  Random matrix:           {:.1}%", acc_random);
    println!("  PMI matrix (static):     {:.1}%", acc_pmi);
    println!();
    println!("  PMI already captures {:.1} percentage points above random.",
        acc_pmi - acc_random);

    // ── PHASE 2: Learning curves ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  PHASE 2: Learning Curves (SGD on attention matrix)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  Simulating transformer training: learn the full V×V matrix");
    println!("  Track A: Random init → SGD (simulates normal training)");
    println!("  Track B: PMI init → SGD (simulates PMI-initialized training)");
    println!("  Parameters: {} (full matrix)", dim * dim);
    println!();

    let lr = 0.005;
    let n_epochs = 5;
    let eval_every = 500;

    // Track A: Random init
    let mut mat_random = Matrix::random(dim, &mut rng);
    mat_random.normalize_to(ref_norm * 0.1); // small init

    // Track B: PMI init
    let mut mat_pmi = pmi_scaled.clone_matrix();

    // Track C: Gated PMI (tiny: V parameters)
    let pmi_for_gate = pmi_scaled.clone_matrix();
    let mut gated = GatedPMI::new(pmi_for_gate);

    let mut curve_random: Vec<(usize, f64)> = Vec::new();
    let mut curve_pmi: Vec<(usize, f64)> = Vec::new();
    let mut curve_gated: Vec<(usize, f64)> = Vec::new();

    // Initial evaluation
    curve_random.push((0, evaluate(&test, &mat_random, &centroids)));
    curve_pmi.push((0, evaluate(&test, &mat_pmi, &centroids)));
    curve_gated.push((0, evaluate_gated(&test, &gated, &centroids)));

    println!("  {:>6} | {:>12} | {:>12} | {:>16}", "Step", "Random Init", "PMI Init", "PMI+Gate({}p)");
    println!("  {:>6} | {:>12} | {:>12} | {:>16}", "", format!("({}p)", dim*dim), format!("({}p)", dim*dim), format!("({}p)", dim));
    println!("  ─────────────────────────────────────────────────────────");
    println!("  {:>6} | {:>11.1}% | {:>11.1}% | {:>15.1}%",
        0, curve_random[0].1, curve_pmi[0].1, curve_gated[0].1);

    let mut global_step = 0;
    let t_learn = Instant::now();

    for epoch in 0..n_epochs {
        let order = rng.shuffle_indices(train.len());
        for (batch_idx, &idx) in order.iter().enumerate() {
            let (ref q, intent) = train[idx];

            // Update all three tracks
            sgd_step(&mut mat_random, q, intent, &centroids, lr);
            sgd_step(&mut mat_pmi, q, intent, &centroids, lr);
            gated.sgd_step(q, intent, &centroids, lr * 2.0); // higher LR for gate (fewer params)

            global_step += 1;

            if global_step % eval_every == 0 {
                let ar = evaluate(&test, &mat_random, &centroids);
                let ap = evaluate(&test, &mat_pmi, &centroids);
                let ag = evaluate_gated(&test, &gated, &centroids);

                curve_random.push((global_step, ar));
                curve_pmi.push((global_step, ap));
                curve_gated.push((global_step, ag));

                println!("  {:>6} | {:>11.1}% | {:>11.1}% | {:>15.1}%",
                    global_step, ar, ap, ag);
            }
        }
    }

    let learn_time = t_learn.elapsed().as_secs_f64();
    println!();
    println!("  Learning time: {:.1}s ({} total steps, {:.0} steps/sec)",
        learn_time, global_step, global_step as f64 / learn_time);

    // ── PHASE 3: Analysis ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  PHASE 3: Convergence Analysis");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let thresholds = [50.0, 60.0, 70.0, 75.0, 80.0];

    println!("  Steps to reach accuracy threshold:");
    println!("  {:>10} | {:>12} | {:>12} | {:>12} | {:>6}",
        "Threshold", "Random Init", "PMI Init", "PMI+Gate", "Speedup");
    println!("  ──────────────────────────────────────────────────────────────");

    for &thresh in &thresholds {
        let steps_random = curve_random.iter()
            .find(|(_, acc)| *acc >= thresh)
            .map(|(s, _)| *s);
        let steps_pmi = curve_pmi.iter()
            .find(|(_, acc)| *acc >= thresh)
            .map(|(s, _)| *s);
        let steps_gated = curve_gated.iter()
            .find(|(_, acc)| *acc >= thresh)
            .map(|(s, _)| *s);

        let sr = steps_random.map(|s| format!("{}", s)).unwrap_or("never".to_string());
        let sp = steps_pmi.map(|s| format!("{}", s)).unwrap_or("never".to_string());
        let sg = steps_gated.map(|s| format!("{}", s)).unwrap_or("never".to_string());

        let speedup = match (steps_random, steps_pmi) {
            (Some(r), Some(p)) if p > 0 => format!("{:.1}x", r as f64 / p as f64),
            (None, Some(_)) => "∞".to_string(),
            _ => "n/a".to_string(),
        };

        println!("  {:>9.0}% | {:>12} | {:>12} | {:>12} | {:>6}",
            thresh, sr, sp, sg, speedup);
    }

    // Final accuracies
    let final_random = curve_random.last().unwrap().1;
    let final_pmi = curve_pmi.last().unwrap().1;
    let final_gated = curve_gated.last().unwrap().1;

    println!();
    println!("  Final accuracies (after {} steps):", global_step);
    println!("    Random init (full matrix, {} params): {:.1}%", dim * dim, final_random);
    println!("    PMI init (full matrix, {} params):    {:.1}%", dim * dim, final_pmi);
    println!("    PMI + gate ({} params):               {:.1}%", dim, final_gated);
    println!();

    let param_ratio = dim as f64 / (dim * dim) as f64 * 100.0;
    println!("  Parameter efficiency:");
    println!("    Gate uses {:.1}% of full matrix parameters ({} vs {})",
        param_ratio, dim, dim * dim);
    if final_gated > 0.0 && final_pmi > 0.0 {
        println!("    Gate achieves {:.1}% of full-matrix accuracy ({:.1}% vs {:.1}%)",
            final_gated / final_pmi * 100.0, final_gated, final_pmi);
    }

    // ── PHASE 4: What does the gate learn? ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  PHASE 4: What the Gate Learned");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let idx_to_term: Vec<&str> = {
        let mut v = vec![""; dim];
        for (t, &i) in &term_to_idx { v[i] = t.as_str(); }
        v
    };

    let mut gate_values: Vec<(f64, &str)> = gated.gate.iter()
        .enumerate()
        .map(|(i, &g)| (sigmoid(g), idx_to_term[i]))
        .collect();
    gate_values.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("  Most AMPLIFIED tokens (gate → 1.0 means keep PMI connection):");
    for (g, term) in gate_values.iter().take(15) {
        let bar_len = (g * 30.0) as usize;
        println!("    {:<20} {:.3}  {}", term, g, "█".repeat(bar_len));
    }

    println!();
    println!("  Most SUPPRESSED tokens (gate → 0.0 means ignore PMI):");
    for (g, term) in gate_values.iter().rev().take(15) {
        let bar_len = (g * 30.0) as usize;
        println!("    {:<20} {:.3}  {}", term, g, "█".repeat(bar_len));
    }

    // ── Summary ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  VERDICT");
    println!("═══════════════════════════════════════════════════════════════════");

    if final_pmi > final_random + 1.0 {
        println!("  PMI init BEATS random init by {:.1} percentage points.", final_pmi - final_random);
    } else if final_random > final_pmi + 1.0 {
        println!("  Random init caught up and beats PMI init by {:.1} pp.", final_random - final_pmi);
    } else {
        println!("  PMI and random init converge to similar accuracy.");
    }

    // Check if PMI init was faster at any threshold
    let any_speedup = thresholds.iter().any(|&thresh| {
        let sr = curve_random.iter().find(|(_, a)| *a >= thresh).map(|(s, _)| *s);
        let sp = curve_pmi.iter().find(|(_, a)| *a >= thresh).map(|(s, _)| *s);
        matches!((sr, sp), (Some(r), Some(p)) if p < r)
    });

    if any_speedup {
        println!("  PMI initialization DOES speed up convergence.");
        println!("  → The thesis holds: precomputed PMI = free head start for attention.");
    } else {
        println!("  PMI initialization did NOT help convergence.");
        println!("  → The thesis needs revision.");
    }

    if final_gated > final_random * 0.9 {
        println!("  Gated PMI ({} params) reaches {:.0}% of full-matrix accuracy.",
            dim, final_gated / final_random * 100.0);
        println!("  → Tiny gate on PMI is a viable cheap attention substitute.");
    }

    println!();
    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
