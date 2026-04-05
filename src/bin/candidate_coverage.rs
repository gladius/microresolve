//! Test 1: Does PMI top-K contain the correct next token?
//! Pure counting. No training. No neural. Under 2 minutes.

use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let t0 = Instant::now();
    println!("Candidate Coverage Test: is the correct next word in PMI's top-K?");
    println!();

    // Load TinyStories
    let raw = std::fs::read_to_string("/home/gladius/Workspace/neocortex/data/tinystories.txt")
        .expect("TinyStories not found");
    let stories: Vec<Vec<String>> = raw.split("<|endoftext|>")
        .map(|s| s.to_lowercase().split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty()).map(|w| w.to_string()).collect::<Vec<_>>())
        .filter(|w| w.len() >= 10).collect();

    // Vocab (words appearing 3+ times)
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

    // Build PMI: for each token, its top neighbors sorted by PMI
    let max_offset = 5;
    let mut unigram = vec![0u32; vs];
    let mut pair_counts: Vec<HashMap<(u32,u32), u32>> = (0..max_offset).map(|_| HashMap::new()).collect();
    let mut total_pairs = vec![0u64; max_offset];

    for t in 0..train.len() {
        unigram[train[t] as usize] += 1;
        for k in 1..=max_offset {
            if t+k < train.len() {
                *pair_counts[k-1].entry((train[t], train[t+k])).or_insert(0) += 1;
                total_pairs[k-1] += 1;
            }
        }
    }

    let n = train.len() as f64;
    let up: Vec<f64> = unigram.iter().map(|&c| (c as f64 + 1.0) / (n + vs as f64)).collect();

    // ── Method 1: PMI neighbors (what we tested before) ──
    println!("Building PMI neighbor lists...");
    let t1 = Instant::now();
    let mut pmi_neighbors: Vec<Vec<(u32, f64)>> = vec![Vec::new(); vs];
    let mut agg: HashMap<(u32, u32), f64> = HashMap::new();
    for k in 0..max_offset {
        let np = total_pairs[k] as f64;
        if np == 0.0 { continue; }
        for (&(a, b), &count) in &pair_counts[k] {
            let pmi = ((count as f64 / np) / (up[a as usize] * up[b as usize])).ln();
            if pmi > 0.0 {
                *agg.entry((a, b)).or_insert(0.0) += pmi / ((k + 1) as f64).sqrt();
            }
        }
    }
    for (&(a, b), &score) in &agg { pmi_neighbors[a as usize].push((b, score)); }
    for n in &mut pmi_neighbors { n.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); }

    // ── Method 2: Conditional probability P(next | prev) ──
    println!("Building conditional probability lists...");
    let mut cond_prob_neighbors: Vec<Vec<(u32, f64)>> = vec![Vec::new(); vs];
    for k in 0..max_offset {
        for (&(a, b), &count) in &pair_counts[k] {
            let p_next_given_prev = count as f64 / unigram[a as usize].max(1) as f64;
            let weighted = p_next_given_prev / ((k + 1) as f64).sqrt();
            // Aggregate into the same list
            // Find existing entry or create new
            let neighbors = &mut cond_prob_neighbors[a as usize];
            if let Some(entry) = neighbors.iter_mut().find(|(id, _)| *id == b) {
                entry.1 += weighted;
            } else {
                neighbors.push((b, weighted));
            }
        }
    }
    for n in &mut cond_prob_neighbors { n.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); }

    // ── Top frequent words (always-available candidates) ──
    let mut freq_sorted: Vec<(u32, u32)> = unigram.iter().enumerate()
        .map(|(i, &c)| (i as u32, c)).collect();
    freq_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    let top_100_freq: Vec<u32> = freq_sorted.iter().take(100).map(|(id, _)| *id).collect();

    println!("  Built in {:.1}s", t1.elapsed().as_secs_f64());

    // Show examples
    println!("\n  Sample CONDITIONAL PROB neighbors (what's LIKELY, not surprising):");
    for word in &["once", "the", "happy", "because", "on", "she"] {
        if let Some(&id) = w2i.get(&word.to_string()) {
            let top5: Vec<String> = cond_prob_neighbors[id as usize].iter().take(5)
                .map(|(id, s)| format!("{}({:.3})", i2w[*id as usize], s)).collect();
            println!("    {} → [{}]", word, top5.join(", "));
        }
    }

    // ── Test all three methods ──
    println!("\nTesting coverage on {} tokens...", test.len().min(50000));
    let ctx_len = 5;
    let k_values = [5, 10, 20, 50, 100, 200];
    let eval_size = test.len().min(50000);

    // Method A: PMI only (what failed at 27.6%)
    // Method B: Conditional probability only
    // Method C: Conditional probability + top-100 frequent words

    let mut cov_pmi = vec![0u64; k_values.len()];
    let mut cov_cprob = vec![0u64; k_values.len()];
    let mut cov_combined = vec![0u64; k_values.len()];
    let mut total = 0u64;

    for t in ctx_len..eval_size {
        let true_next = test[t];
        if true_next == 0 { continue; }

        // Gather PMI candidates
        let mut pmi_cands: HashMap<u32, f64> = HashMap::new();
        let mut cprob_cands: HashMap<u32, f64> = HashMap::new();

        for k in 1..=ctx_len {
            if t < k { continue; }
            let ctx_tok = test[t - k] as usize;
            let weight = 1.0 / (k as f64).sqrt();
            for &(n, s) in &pmi_neighbors[ctx_tok] {
                *pmi_cands.entry(n).or_insert(0.0) += s * weight;
            }
            for &(n, s) in &cond_prob_neighbors[ctx_tok] {
                *cprob_cands.entry(n).or_insert(0.0) += s * weight;
            }
        }

        let mut pmi_sorted: Vec<(u32, f64)> = pmi_cands.into_iter().collect();
        pmi_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cprob_sorted: Vec<(u32, f64)> = cprob_cands.into_iter().collect();
        cprob_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Combined: top-K from cond_prob + always include top-100 frequent
        let mut combined: Vec<u32> = cprob_sorted.iter().map(|(id, _)| *id).collect();
        for &fid in &top_100_freq {
            if !combined.contains(&fid) { combined.push(fid); }
        }

        for (ki, &k) in k_values.iter().enumerate() {
            if pmi_sorted.iter().take(k).any(|(id, _)| *id == true_next) { cov_pmi[ki] += 1; }
            if cprob_sorted.iter().take(k).any(|(id, _)| *id == true_next) { cov_cprob[ki] += 1; }
            // Combined: first K from cprob, plus always the top-100 frequent
            if combined.iter().take(k + 100).any(|id| *id == true_next) { cov_combined[ki] += 1; }
        }
        total += 1;
    }

    // Results
    println!("\n═══════════════════════════════════════════════════════");
    println!("  CANDIDATE COVERAGE RESULTS");
    println!("═══════════════════════════════════════════════════════");
    println!("  {:>6} | {:>10} | {:>10} | {:>14}", "Top-K", "PMI", "Cond.Prob", "CProb+Freq100");
    println!("  ───────────────────────────────────────────────────");
    for (ki, &k) in k_values.iter().enumerate() {
        let p = cov_pmi[ki] as f64 / total as f64 * 100.0;
        let c = cov_cprob[ki] as f64 / total as f64 * 100.0;
        let cb = cov_combined[ki] as f64 / total as f64 * 100.0;
        println!("  {:>6} | {:>9.1}% | {:>9.1}% | {:>13.1}%", k, p, c, cb);
    }

    println!("\n  Tokens: {}, Vocab: {}", total, vs);

    let best50 = cov_combined[3] as f64 / total as f64 * 100.0;
    if best50 > 50.0 {
        println!("\n  >>> COMBINED TOP-50+100freq: {:.1}% COVERAGE — VIABLE! ✓", best50);
    } else {
        println!("\n  >>> Best coverage at 50+100freq: {:.1}%", best50);
    }

    println!("\n  Time: {:.1}s", t0.elapsed().as_secs_f64());
}
