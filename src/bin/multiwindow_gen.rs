//! Multi-window text generation: 3 levels of context, all from counting.
//! Window 1: offset 1-5 (grammar)
//! Window 2: sentence co-occurrence (sentence coherence)
//! Window 3: all generated so far (topic consistency)
//! Zero neural. Zero gradients.

use std::collections::HashMap;
use std::time::Instant;
use std::io::BufRead;

fn main() {
    let t0 = Instant::now();
    println!("Multi-Window Generation: 3 levels of statistical context");
    println!();

    // Load TinyStories
    let raw = std::fs::read_to_string("/home/gladius/Workspace/neocortex/data/tinystories.txt")
        .expect("TinyStories not found");
    let stories: Vec<Vec<String>> = raw.split("<|endoftext|>")
        .map(|s| {
            s.to_lowercase()
                .split(|c: char| !c.is_alphanumeric() && c != '\'')
                .filter(|w| !w.is_empty())
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
        })
        .filter(|w| w.len() >= 10)
        .collect();

    // Vocab
    let mut counts: HashMap<String, usize> = HashMap::new();
    for s in &stories { for w in s { *counts.entry(w.clone()).or_insert(0) += 1; } }
    let mut w2i: HashMap<String, u32> = HashMap::new();
    let mut i2w: Vec<String> = vec!["<unk>".to_string()];
    w2i.insert("<unk>".to_string(), 0);
    let mut sorted: Vec<(String, usize)> = counts.into_iter().filter(|(_,c)| *c >= 3).collect();
    sorted.sort_by(|a,b| b.1.cmp(&a.1));
    for (w, _) in &sorted { let id = i2w.len() as u32; w2i.insert(w.clone(), id); i2w.push(w.clone()); }
    let vs = i2w.len();

    let all_ids: Vec<Vec<u32>> = stories.iter()
        .map(|s| s.iter().map(|w| *w2i.get(w).unwrap_or(&0)).collect())
        .collect();

    println!("Stories: {}, Vocab: {}", all_ids.len(), vs);

    // ── Window 1: Offset conditional probability (grammar) ──
    println!("Computing Window 1 (offset 1-5, grammar)...");
    let max_offset = 5;
    let mut unigram = vec![0u32; vs];
    let mut pair_counts: Vec<HashMap<(u32,u32), u32>> = (0..max_offset).map(|_| HashMap::new()).collect();

    for story in &all_ids {
        for t in 0..story.len() {
            unigram[story[t] as usize] += 1;
            for k in 1..=max_offset {
                if t+k < story.len() {
                    *pair_counts[k-1].entry((story[t], story[t+k])).or_insert(0) += 1;
                }
            }
        }
    }

    let mut w1_cprob: Vec<Vec<(u32, f64)>> = vec![Vec::new(); vs];
    for k in 0..max_offset {
        for (&(a,b), &count) in &pair_counts[k] {
            let p = count as f64 / unigram[a as usize].max(1) as f64 / ((k+1) as f64).sqrt();
            if let Some(e) = w1_cprob[a as usize].iter_mut().find(|(id,_)| *id == b) { e.1 += p; }
            else { w1_cprob[a as usize].push((b, p)); }
        }
    }
    for n in &mut w1_cprob { n.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap()); n.truncate(100); }

    // ── Window 2: Sentence co-occurrence (sentence coherence) ──
    // Two words in the same sentence → they co-occur
    println!("Computing Window 2 (sentence co-occurrence)...");
    let mut sent_cooccur: HashMap<(u32, u32), u32> = HashMap::new();
    let mut sent_count = vec![0u32; vs]; // how many sentences each word appears in

    for story in &all_ids {
        // Split story into sentences (roughly by common sentence-ending patterns)
        // Simple: every 15-20 tokens is roughly a sentence in TinyStories
        for chunk in story.chunks(15) {
            let unique: Vec<u32> = {
                let mut s: Vec<u32> = chunk.to_vec();
                s.sort(); s.dedup(); s
            };
            for &w in &unique { sent_count[w as usize] += 1; }
            for i in 0..unique.len() {
                for j in (i+1)..unique.len() {
                    let a = unique[i].min(unique[j]);
                    let b = unique[i].max(unique[j]);
                    *sent_cooccur.entry((a, b)).or_insert(0) += 1;
                }
            }
        }
    }

    // Build sentence-level cond.prob: given word A in sentence, P(word B also in sentence)
    let mut w2_assoc: Vec<Vec<(u32, f64)>> = vec![Vec::new(); vs];
    let total_sents = all_ids.iter().map(|s| s.len() / 15 + 1).sum::<usize>() as f64;
    for (&(a, b), &count) in &sent_cooccur {
        if count >= 3 {
            let p_ab = count as f64 / total_sents;
            let p_a = sent_count[a as usize] as f64 / total_sents;
            let p_b = sent_count[b as usize] as f64 / total_sents;
            let score = p_ab / (p_a * p_b + 1e-10);
            if score > 1.5 {
                w2_assoc[a as usize].push((b, score));
                w2_assoc[b as usize].push((a, score));
            }
        }
    }
    for n in &mut w2_assoc { n.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap()); n.truncate(100); }

    // Top 100 frequent
    let mut freq: Vec<(u32,u32)> = unigram.iter().enumerate().map(|(i,&c)|(i as u32,c)).collect();
    freq.sort_by(|a,b| b.1.cmp(&a.1));
    let top100: Vec<u32> = freq.iter().take(100).map(|(id,_)| *id).collect();
    let total_tok = unigram.iter().map(|&c| c as f64).sum::<f64>();

    println!("All windows computed: {:.1}s\n", t0.elapsed().as_secs_f64());

    // ── Generation ──
    println!("═══════════════════════════════════════════════════════");
    println!("  MULTI-WINDOW TEXT GENERATION");
    println!("  W1: offset cond.prob (grammar)");
    println!("  W2: sentence co-occurrence (coherence)");
    println!("  W3: bag of all generated words (topic)");
    println!("  Zero neural. Zero gradients. Pure counting.");
    println!("═══════════════════════════════════════════════════════\n");

    let seeds = [
        "once upon a time there",
        "the little girl was very",
        "she was happy because her",
        "he wanted to play with his",
        "one day the boy found a big",
        "the dog ran fast and",
        "mommy said you need to be",
        "they went to the park and played",
        "it was a sunny day and",
        "the cat was sitting on the",
        "but then something bad happened",
        "she loved to eat ice cream",
        "the bird was singing in the",
        "he was scared because the",
        "after dinner they went outside to",
    ];

    let ctx_len = 5;
    let alpha = 3.0;   // W1 weight (grammar) — DOMINANT
    let beta = 0.15;   // W2 weight (sentence coherence) — light touch
    let gamma = 0.05;  // W3 weight (topic) — very light

    let mut rng: u64 = 54321;

    for seed in &seeds {
        let mut seq: Vec<u32> = seed.split_whitespace()
            .map(|w| *w2i.get(&w.to_lowercase()).unwrap_or(&0)).collect();

        // Track all generated words for Window 3 (topic)
        let mut generated_bag: HashMap<u32, f64> = HashMap::new();
        for &id in &seq { *generated_bag.entry(id).or_insert(0.0) += 1.0; }

        for step in 0..50 {
            let t = seq.len();
            let mut cand_scores: HashMap<u32, f64> = HashMap::new();

            // Window 1: offset conditional probability (grammar)
            for k in 1..=ctx_len.min(t) {
                let ctx_tok = seq[t-k] as usize;
                let w = 1.0 / (k as f64).sqrt();
                for &(n, p) in w1_cprob[ctx_tok].iter().take(50) {
                    *cand_scores.entry(n).or_insert(0.0) += alpha * p * w;
                }
            }

            // Window 2: sentence co-occurrence (coherence with recent ~15 words)
            let sent_start = if t > 15 { t - 15 } else { 0 };
            for pos in sent_start..t {
                let tok = seq[pos] as usize;
                for &(n, score) in w2_assoc[tok].iter().take(30) {
                    *cand_scores.entry(n).or_insert(0.0) += beta * score * 0.1;
                }
            }

            // Window 3: topic (all generated words so far)
            // Boost candidates that co-occur with the overall topic
            for (&gen_tok, &gen_weight) in &generated_bag {
                let gen_tok = gen_tok as usize;
                let topic_w = gen_weight.sqrt() * gamma / (t as f64).sqrt();
                for &(n, score) in w2_assoc[gen_tok].iter().take(20) {
                    *cand_scores.entry(n).or_insert(0.0) += topic_w * score * 0.05;
                }
            }

            // Always include top frequent words
            for &fid in &top100 {
                cand_scores.entry(fid).or_insert(unigram[fid as usize] as f64 / total_tok * 0.3);
            }

            // Strong repetition penalty
            let recent_window = 20.min(t);
            for i in 0..recent_window {
                let recent = seq[t - 1 - i];
                let decay = 0.05 + 0.05 * i as f64; // very harsh for recent, softer for older
                if let Some(s) = cand_scores.get_mut(&recent) {
                    *s *= decay;
                }
            }

            // N-gram blocking: never repeat same 3-gram
            if t >= 2 {
                let prev2 = seq[t-2];
                let prev1 = seq[t-1];
                // Check all previous 3-grams
                for i in 2..t {
                    if seq[i-2] == prev2 && seq[i-1] == prev1 {
                        // This 3-gram already exists — block the word that followed
                        if i < seq.len() {
                            if let Some(s) = cand_scores.get_mut(&seq[i]) {
                                *s = -100.0; // block completely
                            }
                        }
                    }
                }
            }

            // Sort and select
            let mut candidates: Vec<(u32, f64)> = cand_scores.into_iter().collect();
            candidates.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

            // Temperature sampling from top 15
            let top_n = 15;
            let temp = 0.5;
            let cands: Vec<(u32, f64)> = candidates.iter().take(top_n).cloned().collect();
            if cands.is_empty() { break; }

            let max_s = cands.iter().map(|(_,s)| *s).fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = cands.iter().map(|(_,s)| ((s - max_s) / temp).exp()).collect();
            let sum: f64 = exps.iter().sum();

            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let r = (rng % 1_000_000) as f64 / 1_000_000.0;
            let mut cum = 0.0;
            let mut chosen = cands[0].0;
            for (i, &e) in exps.iter().enumerate() {
                cum += e / sum;
                if cum >= r { chosen = cands[i].0; break; }
            }

            seq.push(chosen);
            *generated_bag.entry(chosen).or_insert(0.0) += 1.0;
        }

        let text: String = seq.iter()
            .map(|&id| i2w.get(id as usize).map(|s| s.as_str()).unwrap_or("?"))
            .collect::<Vec<_>>().join(" ");
        println!("  \"{}\"\n", text);
    }

    println!("Total: {:.1}s", t0.elapsed().as_secs_f64());
}
