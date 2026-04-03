//! Neocortex Language Model: Gradient-Free Next-Token Prediction
//!
//! Can PMI + corrections predict the next token in a sequence?
//! No neural network. No gradients. Just statistics + corrections.
//!
//! Architecture:
//!   - Position-offset PMI: PMI_k[a,b] = co-occurrence of token a with token b at distance k
//!   - Combined attention: weighted sum of PMI at different offsets
//!   - Corrections: perceptron-style, fix wrong predictions
//!   - Single layer first, then stack if it works
//!
//! Run: cargo run --release --bin neocortex_lm

use std::collections::HashMap;
use std::time::Instant;

// ── Simple tokenizer (word-level, lowercase) ──
fn build_vocab(texts: &[Vec<String>], max_vocab: usize) -> (HashMap<String, u32>, Vec<String>) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for doc in texts {
        for w in doc { *counts.entry(w.clone()).or_insert(0) += 1; }
    }
    let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.truncate(max_vocab - 1); // reserve 0 for <unk>

    let mut word_to_id: HashMap<String, u32> = HashMap::new();
    let mut id_to_word: Vec<String> = vec!["<unk>".to_string()];
    word_to_id.insert("<unk>".to_string(), 0);

    for (i, (word, _)) in sorted.iter().enumerate() {
        let id = (i + 1) as u32;
        word_to_id.insert(word.clone(), id);
        id_to_word.push(word.clone());
    }
    (word_to_id, id_to_word)
}

fn tokenize_to_ids(text: &[String], vocab: &HashMap<String, u32>) -> Vec<u32> {
    text.iter().map(|w| *vocab.get(w).unwrap_or(&0)).collect()
}

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

// ── Position-Offset PMI ──
// PMI_k[a][b] = log(P(a at pos t, b at pos t+k) / P(a)·P(b))
// Stored sparse: for each offset k, a HashMap of (token_a, token_b) → PMI value
struct OffsetPMI {
    offsets: Vec<HashMap<(u32, u32), f32>>,  // one map per offset distance
    max_offset: usize,
    unigram_freq: Vec<f32>,  // P(token)
    vocab_size: usize,
}

impl OffsetPMI {
    fn build(sequences: &[Vec<u32>], vocab_size: usize, max_offset: usize) -> Self {
        let mut pair_counts: Vec<HashMap<(u32, u32), u32>> = vec![HashMap::new(); max_offset];
        let mut unigram_counts = vec![0u32; vocab_size];
        let mut total_tokens = 0u64;
        let mut total_pairs = vec![0u64; max_offset];

        for seq in sequences {
            for (t, &token) in seq.iter().enumerate() {
                unigram_counts[token as usize] += 1;
                total_tokens += 1;
                for k in 1..=max_offset {
                    if t + k < seq.len() {
                        let next = seq[t + k];
                        *pair_counts[k - 1].entry((token, next)).or_insert(0) += 1;
                        total_pairs[k - 1] += 1;
                    }
                }
            }
        }

        let n = total_tokens as f32;
        let unigram_freq: Vec<f32> = unigram_counts.iter()
            .map(|&c| (c as f32 + 1.0) / (n + vocab_size as f32))
            .collect();

        let mut offsets = Vec::with_capacity(max_offset);
        for k in 0..max_offset {
            let n_pairs = total_pairs[k] as f32;
            if n_pairs == 0.0 {
                offsets.push(HashMap::new());
                continue;
            }
            let mut pmi_map = HashMap::new();
            for (&(a, b), &count) in &pair_counts[k] {
                let p_ab = count as f32 / n_pairs;
                let p_a = unigram_freq[a as usize];
                let p_b = unigram_freq[b as usize];
                let pmi = (p_ab / (p_a * p_b)).ln();
                if pmi > 0.5 { // only keep significant associations
                    pmi_map.insert((a, b), pmi);
                }
            }
            offsets.push(pmi_map);
        }

        OffsetPMI { offsets, max_offset, unigram_freq, vocab_size }
    }

    /// For a context [..., t-3, t-2, t-1], score each candidate next token.
    /// Returns scores for all vocab tokens.
    fn score_next(&self, context: &[u32]) -> Vec<f32> {
        let mut scores = vec![0.0f32; self.vocab_size];
        let ctx_len = context.len();

        for k in 1..=self.max_offset.min(ctx_len) {
            let prev_token = context[ctx_len - k];
            let offset_map = &self.offsets[k - 1];
            // Weight closer offsets more (1/k decay)
            let weight = 1.0 / k as f32;
            for candidate in 0..self.vocab_size as u32 {
                if let Some(&pmi) = offset_map.get(&(prev_token, candidate)) {
                    scores[candidate as usize] += pmi * weight;
                }
            }
        }
        scores
    }
}

// ── Correction Table ──
// For each (context_pattern, predicted_wrong, correct_token), store a correction.
// Context pattern = last N tokens as a tuple.
struct CorrectionTable {
    // (context_token, wrong_token) → corrections: Vec<(correct_token, strength)>
    bigram_corrections: HashMap<(u32, u32), Vec<(u32, f32)>>,
    // (ctx_2, ctx_1, wrong) → corrections
    trigram_corrections: HashMap<(u32, u32, u32), Vec<(u32, f32)>>,
    correction_weight: f32,
}

impl CorrectionTable {
    fn new(weight: f32) -> Self {
        CorrectionTable {
            bigram_corrections: HashMap::new(),
            trigram_corrections: HashMap::new(),
            correction_weight: weight,
        }
    }

    fn add_correction(&mut self, context: &[u32], wrong: u32, correct: u32) {
        let ctx_len = context.len();

        // Bigram correction: last token → wrong should have been correct
        if ctx_len >= 1 {
            let prev = context[ctx_len - 1];
            self.bigram_corrections.entry((prev, wrong))
                .or_default()
                .push((correct, 1.0));
        }

        // Trigram correction
        if ctx_len >= 2 {
            let prev2 = context[ctx_len - 2];
            let prev1 = context[ctx_len - 1];
            self.trigram_corrections.entry((prev2, prev1, wrong))
                .or_default()
                .push((correct, 2.0)); // trigram gets higher weight
        }
    }

    fn apply(&self, context: &[u32], scores: &mut Vec<f32>) {
        let ctx_len = context.len();

        // Apply bigram corrections
        if ctx_len >= 1 {
            let prev = context[ctx_len - 1];
            for candidate in 0..scores.len() as u32 {
                if let Some(corrs) = self.bigram_corrections.get(&(prev, candidate)) {
                    // This (prev, candidate) pair was wrong before — penalize
                    let penalty: f32 = corrs.iter().map(|(_, s)| s).sum::<f32>().min(5.0);
                    scores[candidate as usize] -= self.correction_weight * penalty;
                    // Boost the correct alternatives
                    for &(correct, strength) in corrs {
                        scores[correct as usize] += self.correction_weight * strength;
                    }
                }
            }
        }

        // Apply trigram corrections
        if ctx_len >= 2 {
            let prev2 = context[ctx_len - 2];
            let prev1 = context[ctx_len - 1];
            for candidate in 0..scores.len() as u32 {
                if let Some(corrs) = self.trigram_corrections.get(&(prev2, prev1, candidate)) {
                    let penalty: f32 = corrs.iter().map(|(_, s)| s).sum::<f32>().min(10.0);
                    scores[candidate as usize] -= self.correction_weight * penalty;
                    for &(correct, strength) in corrs {
                        scores[correct as usize] += self.correction_weight * strength;
                    }
                }
            }
        }
    }
}

// ── Evaluation ──
fn perplexity(
    sequences: &[Vec<u32>],
    pmi: &OffsetPMI,
    corrections: &CorrectionTable,
    context_len: usize,
) -> (f64, f64, f64) {
    let mut total_log_prob = 0.0f64;
    let mut total_tokens = 0u64;
    let mut correct_top1 = 0u64;
    let mut correct_top5 = 0u64;

    for seq in sequences {
        for t in context_len..seq.len() {
            let context = &seq[t.saturating_sub(context_len)..t];
            let true_token = seq[t];

            let mut scores = pmi.score_next(context);
            corrections.apply(context, &mut scores);

            // Softmax for perplexity
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let prob = exps[true_token as usize] / sum;
            total_log_prob += (prob as f64).max(1e-20).ln();
            total_tokens += 1;

            // Top-1 accuracy
            let pred = scores.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap().0 as u32;
            if pred == true_token { correct_top1 += 1; }

            // Top-5 accuracy
            let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate()
                .map(|(i, &s)| (i, s)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if indexed.iter().take(5).any(|(i, _)| *i == true_token as usize) {
                correct_top5 += 1;
            }
        }
    }

    let ppl = (-total_log_prob / total_tokens as f64).exp();
    let top1 = correct_top1 as f64 / total_tokens as f64 * 100.0;
    let top5 = correct_top5 as f64 / total_tokens as f64 * 100.0;
    (ppl, top1, top5)
}

// ── Text Generation ──
fn generate(
    pmi: &OffsetPMI,
    corrections: &CorrectionTable,
    seed: &[u32],
    n_tokens: usize,
    id_to_word: &[String],
    temperature: f32,
) -> String {
    let mut sequence = seed.to_vec();
    let mut rng_state: u64 = 12345;

    for _ in 0..n_tokens {
        let ctx_start = sequence.len().saturating_sub(8);
        let context = &sequence[ctx_start..];

        let mut scores = pmi.score_next(context);
        corrections.apply(context, &mut scores);

        // Temperature-scaled sampling
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| ((s - max_s) / temperature).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        // Sample from distribution
        rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
        let r = (rng_state % 1_000_000) as f32 / 1_000_000.0;
        let mut cumsum = 0.0;
        let mut chosen = 0u32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r { chosen = i as u32; break; }
        }
        sequence.push(chosen);
    }

    sequence.iter()
        .map(|&id| id_to_word.get(id as usize).map(|s| s.as_str()).unwrap_or("<unk>"))
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Main ──
fn main() {
    let t0 = Instant::now();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Neocortex Language Model: Gradient-Free Next-Token Prediction   ║");
    println!("║  Can PMI + corrections learn language? No neural network.        ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── Load text data ──
    // Use a built-in small corpus: fairy tales / simple stories
    // We'll construct one inline for now, then scale to files
    let corpus = include_str!("../../tests/data/benchmarks/bitext_all.json");
    let examples: Vec<serde_json::Value> = serde_json::from_str(corpus).unwrap();

    // Extract all text as a simple corpus
    let mut all_texts: Vec<Vec<String>> = Vec::new();
    for ex in &examples {
        let text = ex["text"].as_str().unwrap_or("");
        let w = words(text);
        if w.len() >= 3 { all_texts.push(w); }
    }

    // Also load any text files we can find for richer data
    let extra_text = r#"
once upon a time there was a little girl who lived in a small village near the forest
she loved to play in the garden and pick flowers for her mother
one day she found a small bird with a broken wing in the garden
she carefully picked up the bird and brought it home
her mother helped her make a small nest for the bird
every day she fed the bird and gave it water
slowly the bird got better and stronger
one morning the bird spread its wings and flew away
the girl was sad but happy that the bird was free
she knew the bird would remember her kindness
the next spring the bird came back with its family
they built a nest in the tree near her window
every morning they sang beautiful songs for her
the girl smiled and knew she had made a friend for life
the old man lived alone in a house on the hill
he spent his days reading books and tending his garden
the children in the village were afraid of him
they thought he was strange because he never spoke to anyone
one brave boy decided to visit the old man
he knocked on the door and waited nervously
the old man opened the door and smiled warmly
come in he said i have been waiting for a visitor
the boy was surprised by the kindness in his voice
they sat together and the old man told stories of his travels
he had been a sailor and had visited many countries
the boy listened with wide eyes to tales of distant lands
from that day on the boy visited the old man every week
soon the other children came too curious about the stories
the old man was no longer alone and the children had a wonderful friend
the cat sat on the mat and watched the birds outside
the dog chased the ball across the green field
the farmer planted seeds in the brown earth
rain fell softly on the leaves of the tall trees
the sun rose over the mountains and lit the valley below
fish swam in the clear blue river
children laughed and played in the warm summer afternoon
the baker made fresh bread every morning before dawn
the smell of bread filled the street and woke the neighbors
everyone agreed it was the best bread in the whole town
"#;

    for line in extra_text.lines() {
        let w = words(line);
        if w.len() >= 3 { all_texts.push(w); }
    }

    println!("Corpus: {} sentences", all_texts.len());

    // ── Build vocabulary ──
    let max_vocab = 5000;
    let (vocab, id_to_word) = build_vocab(&all_texts, max_vocab);
    let actual_vocab = id_to_word.len();
    println!("Vocabulary: {} tokens", actual_vocab);

    // ── Tokenize ──
    let sequences: Vec<Vec<u32>> = all_texts.iter()
        .map(|t| tokenize_to_ids(t, &vocab))
        .filter(|s| s.len() >= 4)
        .collect();

    let total_tokens: usize = sequences.iter().map(|s| s.len()).sum();
    println!("Sequences: {}, total tokens: {}", sequences.len(), total_tokens);

    // ── Split train/test ──
    let split = sequences.len() * 85 / 100;
    let train = &sequences[..split];
    let test = &sequences[split..];
    println!("Train: {} sequences, Test: {} sequences", train.len(), test.len());

    // ── Build Position-Offset PMI ──
    println!();
    println!("Building position-offset PMI (offsets 1-8)...");
    let t1 = Instant::now();
    let max_offset = 8;
    let pmi = OffsetPMI::build(train, actual_vocab, max_offset);

    let mut total_entries = 0;
    for (k, map) in pmi.offsets.iter().enumerate() {
        println!("  Offset {}: {} PMI entries", k + 1, map.len());
        total_entries += map.len();
    }
    println!("  Total: {} PMI entries ({:.1}s)", total_entries, t1.elapsed().as_secs_f64());

    // ── Baseline: unigram (predict most common token) ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  BASELINES");
    println!("═══════════════════════════════════════════════════════════════════");

    let no_corrections = CorrectionTable::new(0.0);

    // Random baseline
    let random_ppl = (actual_vocab as f64).exp(); // truly random
    println!("  Random:          perplexity = {:.0} (vocab size)", random_ppl);

    // Unigram baseline
    {
        let mut total_log_prob = 0.0f64;
        let mut total_tok = 0u64;
        for seq in test {
            for &token in seq.iter().skip(1) {
                let prob = pmi.unigram_freq[token as usize] as f64;
                total_log_prob += prob.max(1e-20).ln();
                total_tok += 1;
            }
        }
        let unigram_ppl = (-total_log_prob / total_tok as f64).exp();
        println!("  Unigram:         perplexity = {:.1}", unigram_ppl);
    }

    // PMI only (no corrections)
    let context_len = 8;
    let (pmi_ppl, pmi_top1, pmi_top5) = perplexity(test, &pmi, &no_corrections, context_len);
    println!("  PMI (offset 1-8): perplexity = {:.1}, top-1 = {:.1}%, top-5 = {:.1}%",
        pmi_ppl, pmi_top1, pmi_top5);

    // ── Apply corrections ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  CORRECTION PASSES");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let mut corrections = CorrectionTable::new(0.3);
    let n_passes = 5;

    println!("  {:>4} | {:>10} | {:>10} | {:>7} | {:>7}",
        "Pass", "Perplexity", "Corrections", "Top-1", "Top-5");
    println!("  ───────────────────────────────────────────────────────");

    let (p, t1a, t5a) = perplexity(test, &pmi, &corrections, context_len);
    println!("  {:>4} | {:>10.1} | {:>10} | {:>6.1}% | {:>6.1}%", 0, p, 0, t1a, t5a);

    let mut total_corr = 0usize;
    for pass in 1..=n_passes {
        // Go through training data, correct mistakes
        for seq in train {
            for t in context_len..seq.len() {
                let context = &seq[t.saturating_sub(context_len)..t];
                let true_token = seq[t];

                let mut scores = pmi.score_next(context);
                corrections.apply(context, &mut scores);

                let pred = scores.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap().0 as u32;

                if pred != true_token {
                    corrections.add_correction(context, pred, true_token);
                    total_corr += 1;
                }
            }
        }

        let (ppl, top1, top5) = perplexity(test, &pmi, &corrections, context_len);
        println!("  {:>4} | {:>10.1} | {:>10} | {:>6.1}% | {:>6.1}%",
            pass, ppl, total_corr, top1, top5);
    }

    // ── Generate text ──
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  TEXT GENERATION");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let seeds = vec![
        vec!["i", "want", "to"],
        vec!["the", "old", "man"],
        vec!["once", "upon", "a"],
        vec!["can", "you", "help"],
        vec!["please", "cancel", "my"],
    ];

    for seed_words in &seeds {
        let seed_ids: Vec<u32> = seed_words.iter()
            .map(|w| *vocab.get(&w.to_string()).unwrap_or(&0))
            .collect();

        let text = generate(&pmi, &corrections, &seed_ids, 20, &id_to_word, 0.8);
        println!("  \"{}\"", text);
        println!();
    }

    // ── Summary ──
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let (final_ppl, final_top1, final_top5) = perplexity(test, &pmi, &corrections, context_len);
    println!("  Final perplexity:  {:.1} (lower = better)", final_ppl);
    println!("  Final top-1:       {:.1}%", final_top1);
    println!("  Final top-5:       {:.1}%", final_top5);
    println!("  Random baseline:   {:.0}", random_ppl);
    println!("  PMI improvement:   {:.1}x over random", random_ppl / final_ppl);
    println!("  Total corrections: {}", total_corr);
    println!("  Total time:        {:.1}s", t0.elapsed().as_secs_f64());
    println!();

    if final_ppl < pmi_ppl * 0.8 {
        println!("  Corrections REDUCED perplexity by {:.0}%.",
            (1.0 - final_ppl / pmi_ppl) * 100.0);
        println!("  The gradient-free system learns language patterns.");
    }
    if final_top1 > 20.0 {
        println!("  Top-1 accuracy > 20% — the model predicts the right next word");
        println!("  more than 1 in 5 times. On a {}-token vocabulary, random is {:.1}%.",
            actual_vocab, 100.0 / actual_vocab as f64);
    }
}
