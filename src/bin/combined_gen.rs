//! Combined Architecture: N-gram + PMI + Multi-window + Candidate Selection
//! Everything we proved works, in one system.
//! Trained on Wikipedia. Zero neural. Zero gradients.

use std::collections::HashMap;
use std::time::Instant;
use std::io::BufRead;

const MAX_TOKENS: usize = 10_000_000;

fn main() {
    let t0 = Instant::now();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Combined Architecture: N-gram + PMI + Multi-window      ║");
    println!("║  Wikipedia-trained. Zero neural. Zero gradients.         ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // ── Load Wikipedia (10M tokens, memory-safe) ──
    println!("Loading Wikipedia...");
    let mut words: Vec<String> = Vec::new();
    let file = std::io::BufReader::new(
        std::fs::File::open("/tmp/wiki_full.txt").expect("wiki not found")
    );
    for line in file.lines() {
        if let Ok(l) = line {
            for w in l.to_lowercase().split(|c: char| !c.is_alphanumeric() && c != '\'') {
                if !w.is_empty() {
                    words.push(w.to_string());
                    if words.len() >= MAX_TOKENS { break; }
                }
            }
            if words.len() >= MAX_TOKENS { break; }
        }
    }
    println!("  {} tokens loaded", words.len());

    // ── Vocab (min_count=5) ──
    let mut counts: HashMap<String, usize> = HashMap::new();
    for w in &words { *counts.entry(w.clone()).or_insert(0) += 1; }
    let mut w2i: HashMap<String, u32> = HashMap::new();
    let mut i2w: Vec<String> = vec!["<unk>".to_string()];
    w2i.insert("<unk>".to_string(), 0);
    let mut sorted: Vec<(String, usize)> = counts.into_iter().filter(|(_,c)| *c >= 5).collect();
    sorted.sort_by(|a,b| b.1.cmp(&a.1));
    for (w, _) in &sorted { w2i.insert(w.clone(), i2w.len() as u32); i2w.push(w.clone()); }
    let vs = i2w.len();

    let ids: Vec<u32> = words.iter().map(|w| *w2i.get(w).unwrap_or(&0)).collect();
    drop(words); // free memory
    let unk_pct = ids.iter().filter(|&&id| id == 0).count() as f64 / ids.len() as f64 * 100.0;
    println!("  Vocab: {}, unk: {:.1}%", vs, unk_pct);

    let split = ids.len() * 95 / 100;
    let train = &ids[..split];
    println!("  Train: {} tokens", train.len());

    // ── Component 1: 3-gram + 2-gram + unigram ──
    println!("\nBuilding n-grams...");
    let t1 = Instant::now();
    let mut unigram = vec![0u32; vs];
    let mut bigram: HashMap<(u32, u32), HashMap<u32, u32>> = HashMap::new();
    let mut trigram: HashMap<(u32, u32, u32), HashMap<u32, u32>> = HashMap::new();

    for t in 0..train.len() {
        let w = train[t];
        unigram[w as usize] += 1;

        if t >= 1 {
            let prev1 = train[t-1];
            bigram.entry((prev1, w)).or_default();
            if t + 1 < train.len() {
                let next = train[t+1];
                bigram.entry((prev1, w)).or_default()
                    .entry(next).and_modify(|c| *c += 1).or_insert(1);
            }
        }

        if t >= 2 {
            let prev2 = train[t-2];
            let prev1 = train[t-1];
            if t + 1 < train.len() {
                let next = train[t+1];
                // Store as: given (prev2, prev1, current), what comes next?
                // Actually: given (prev2, prev1), what's the next word?
                // Wait, let me restructure: trigram[(w_{t-2}, w_{t-1})] → {w_t: count}
            }
        }
    }

    // Rebuild properly: trigram[prev2, prev1] → {next: count}
    let mut trigram: HashMap<u64, HashMap<u32, u32>> = HashMap::new();
    let mut bigram_next: HashMap<u32, HashMap<u32, u32>> = HashMap::new();

    for t in 0..train.len() {
        unigram[train[t] as usize] = unigram[train[t] as usize]; // already counted

        if t >= 1 && t < train.len() {
            let prev = train[t-1];
            let cur = train[t];
            bigram_next.entry(prev).or_default()
                .entry(cur).and_modify(|c| *c += 1).or_insert(1);
        }

        if t >= 2 && t < train.len() {
            let key = (train[t-2] as u64) * (vs as u64) + train[t-1] as u64;
            let cur = train[t];
            trigram.entry(key).or_default()
                .entry(cur).and_modify(|c| *c += 1).or_insert(1);
        }
    }

    let bigram_entries: usize = bigram_next.values().map(|m| m.len()).sum();
    let trigram_entries: usize = trigram.values().map(|m| m.len()).sum();
    println!("  Bigram entries: {}, Trigram entries: {}", bigram_entries, trigram_entries);
    println!("  N-grams built: {:.1}s", t1.elapsed().as_secs_f64());

    // ── Component 2: PMI sparse neighbors ──
    println!("Building PMI neighbors...");
    let t2 = Instant::now();
    let max_offset = 3;
    let mut pair_counts: Vec<HashMap<(u32,u32), u32>> = (0..max_offset).map(|_| HashMap::new()).collect();
    for t in 0..train.len() {
        for k in 1..=max_offset {
            if t+k < train.len() {
                *pair_counts[k-1].entry((train[t], train[t+k])).or_insert(0) += 1;
            }
        }
    }
    let total_tok = train.len() as f64;
    let up: Vec<f64> = unigram.iter().map(|&c| (c as f64 + 1.0) / (total_tok + vs as f64)).collect();

    // Build sparse PMI: top 50 neighbors per token
    let mut pmi_neighbors: Vec<Vec<(u32, f32)>> = vec![Vec::new(); vs];
    for k in 0..max_offset {
        let np: f64 = pair_counts[k].values().map(|&c| c as f64).sum();
        if np == 0.0 { continue; }
        for (&(a, b), &count) in &pair_counts[k] {
            let pmi = ((count as f64 / np) / (up[a as usize] * up[b as usize])).ln();
            if pmi > 0.5 {
                pmi_neighbors[a as usize].push((b, pmi as f32));
            }
        }
    }
    for n in &mut pmi_neighbors { n.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap()); n.truncate(50); }
    drop(pair_counts); // free memory
    println!("  PMI built: {:.1}s", t2.elapsed().as_secs_f64());

    // ── Component 3: Sentence co-occurrence (topic) ──
    println!("Building sentence co-occurrence...");
    let t3 = Instant::now();
    let mut sent_cooccur: HashMap<u64, u32> = HashMap::new();
    for chunk in train.chunks(20) { // ~20 tokens per sentence
        let mut unique: Vec<u32> = chunk.to_vec();
        unique.sort(); unique.dedup();
        for i in 0..unique.len().min(30) {
            for j in (i+1)..unique.len().min(30) {
                let key = (unique[i].min(unique[j]) as u64) * (vs as u64) + unique[i].max(unique[j]) as u64;
                *sent_cooccur.entry(key).or_insert(0) += 1;
            }
        }
    }
    println!("  Sentence co-occur: {} entries, {:.1}s", sent_cooccur.len(), t3.elapsed().as_secs_f64());

    // Top 100 frequent words
    let mut freq: Vec<(u32, u32)> = unigram.iter().enumerate().map(|(i,&c)| (i as u32, c)).collect();
    freq.sort_by(|a,b| b.1.cmp(&a.1));
    let top100: Vec<u32> = freq.iter().take(100).map(|(id,_)| *id).collect();

    println!("\nAll components ready. Total setup: {:.1}s\n", t0.elapsed().as_secs_f64());
    println!("Memory estimate: ~{}MB", (trigram_entries * 12 + bigram_entries * 12 + sent_cooccur.len() * 12 + vs * 200) / 1_000_000);

    // ═══════════════════════════════════════════════════
    // GENERATION
    // ═══════════════════════════════════════════════════

    let generate = |seed: &str, max_words: usize| -> String {
        let mut seq: Vec<u32> = seed.split_whitespace()
            .map(|w| *w2i.get(&w.to_lowercase()).unwrap_or(&0)).collect();
        let mut generated_bag: HashMap<u32, f64> = HashMap::new();
        for &id in &seq { *generated_bag.entry(id).or_insert(0.0) += 1.0; }
        let mut rng: u64 = seed.len() as u64 * 31 + 12345;

        for _ in 0..max_words {
            let t = seq.len();
            let mut scores: HashMap<u32, f64> = HashMap::new();

            // ── N-GRAM SCORES (grammar + facts) ──
            // Trigram: P(next | prev2, prev1)
            if t >= 2 {
                let key = (seq[t-2] as u64) * (vs as u64) + seq[t-1] as u64;
                if let Some(nexts) = trigram.get(&key) {
                    let total: u32 = nexts.values().sum();
                    for (&next, &count) in nexts {
                        let p = count as f64 / total as f64;
                        *scores.entry(next).or_insert(0.0) += 3.0 * p; // trigram weight 3
                    }
                }
            }

            // Bigram: P(next | prev1) — backoff
            if t >= 1 {
                if let Some(nexts) = bigram_next.get(&seq[t-1]) {
                    let total: u32 = nexts.values().sum();
                    for (&next, &count) in nexts {
                        let p = count as f64 / total as f64;
                        *scores.entry(next).or_insert(0.0) += 1.0 * p; // bigram weight 1
                    }
                }
            }

            // Unigram — final backoff
            for &fid in &top100 {
                scores.entry(fid).or_insert(unigram[fid as usize] as f64 / total_tok * 0.1);
            }

            // ── PMI ASSOCIATION (conceptual leaps) ──
            // From recent words, suggest associated concepts
            let pmi_window = 5.min(t);
            for i in 0..pmi_window {
                let tok = seq[t - 1 - i] as usize;
                let w = 0.3 / ((i + 1) as f64).sqrt();
                for &(neighbor, pmi_score) in pmi_neighbors[tok].iter().take(20) {
                    *scores.entry(neighbor).or_insert(0.0) += w * pmi_score as f64 * 0.2;
                }
            }

            // ── MULTI-WINDOW TOPIC (coherence) ──
            // Boost candidates that co-occur with words in generated text
            let topic_window = 15.min(t);
            for i in 0..topic_window {
                let tok = seq[t - 1 - i];
                for (&cand, score) in &scores.clone() {
                    let a = tok.min(cand); let b = tok.max(cand);
                    let key = (a as u64) * (vs as u64) + b as u64;
                    if let Some(&cooc) = sent_cooccur.get(&key) {
                        if cooc > 2 {
                            *scores.entry(cand).or_insert(0.0) += 0.05 * (cooc as f64).sqrt();
                        }
                    }
                }
            }

            // ── ANTI-REPETITION ──
            let recent = 15.min(t);
            for i in 0..recent {
                let recent_word = seq[t - 1 - i];
                if let Some(s) = scores.get_mut(&recent_word) {
                    *s *= 0.02 + 0.03 * i as f64; // very harsh for recent words
                }
            }
            // 3-gram blocking
            if t >= 2 {
                for i in 2..t {
                    if seq[i-2] == seq[t-2] && seq[i-1] == seq[t-1] && i < t {
                        if let Some(s) = scores.get_mut(&seq[i]) {
                            *s = -100.0;
                        }
                    }
                }
            }

            // ── SELECT ──
            let mut candidates: Vec<(u32, f64)> = scores.into_iter()
                .filter(|(_, s)| *s > 0.0)
                .collect();
            candidates.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

            if candidates.is_empty() { break; }

            // Temperature sampling from top 10
            let top_n = 10;
            let temp = 0.4;
            let cands: Vec<(u32, f64)> = candidates.iter().take(top_n).cloned().collect();
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

        seq.iter()
            .map(|&id| i2w.get(id as usize).map(|s| s.as_str()).unwrap_or("?"))
            .collect::<Vec<_>>().join(" ")
    };

    println!("═══════════════════════════════════════════════════════════");
    println!("  FACTUAL QUESTIONS");
    println!("═══════════════════════════════════════════════════════════\n");

    let factual = [
        "the capital of france",
        "the united states of america",
        "water is made of",
        "the sun is a",
        "the largest ocean in the world",
        "the first president of the united states",
        "the population of china",
        "photosynthesis is the process",
        "the earth orbits around the",
        "world war ii ended in",
    ];
    for q in &factual {
        let result = generate(q, 30);
        println!("  Q: {}", q);
        println!("  A: {}\n", result);
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  TOPIC GENERATION");
    println!("═══════════════════════════════════════════════════════════\n");

    let topics = [
        "the history of ancient rome",
        "the human body has many",
        "in the field of mathematics",
        "the climate of the earth",
        "music has been part of",
    ];
    for q in &topics {
        let result = generate(q, 50);
        println!("  \"{}\"\n", result);
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("  CAUSE-EFFECT / REASONING PROMPTS");
    println!("═══════════════════════════════════════════════════════════\n");

    let reasoning = [
        "because the temperature was very",
        "the reason for the war was",
        "if water freezes it becomes",
        "plants need sunlight to",
        "the discovery of electricity changed",
    ];
    for q in &reasoning {
        let result = generate(q, 40);
        println!("  \"{}\"\n", result);
    }

    println!("Total: {:.0}s", t0.elapsed().as_secs_f64());
}
