//! Deep coverage test: full dataset, scaling analysis, nomination-count discrimination.
//! One dataset (TinyStories), thorough analysis.

use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let t0 = Instant::now();
    println!("Deep Coverage Test: Full TinyStories + Nomination Discrimination");
    println!();

    // Load ALL TinyStories
    let raw = std::fs::read_to_string("/home/gladius/Workspace/neocortex/data/tinystories.txt")
        .expect("TinyStories not found");
    let all_stories: Vec<Vec<String>> = raw.split("<|endoftext|>")
        .map(|s| s.to_lowercase().split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty()).map(|w| w.to_string()).collect::<Vec<_>>())
        .filter(|w| w.len() >= 10).collect();
    println!("Total stories: {}", all_stories.len());

    // Build vocab from ALL data
    let mut counts: HashMap<String, usize> = HashMap::new();
    for s in &all_stories { for w in s { *counts.entry(w.clone()).or_insert(0) += 1; } }
    let mut w2i: HashMap<String, u32> = HashMap::new();
    let mut i2w: Vec<String> = vec!["<unk>".to_string()];
    w2i.insert("<unk>".to_string(), 0);
    let mut sorted: Vec<(String, usize)> = counts.into_iter().filter(|(_,c)| *c >= 3).collect();
    sorted.sort_by(|a,b| b.1.cmp(&a.1));
    for (w, _) in &sorted { let id = i2w.len() as u32; w2i.insert(w.clone(), id); i2w.push(w.clone()); }
    let vs = i2w.len();

    let mut all_ids: Vec<u32> = Vec::new();
    for s in &all_stories { for w in s { all_ids.push(*w2i.get(w).unwrap_or(&0)); } }
    let split = all_ids.len() * 90 / 100;
    let train = &all_ids[..split];
    let test = &all_ids[split..];
    println!("Vocab: {}, Total tokens: {}, Train: {}, Test: {}", vs, all_ids.len(), train.len(), test.len());

    // ── Test at different training sizes ──
    let data_sizes = [
        (train.len() / 4, "25%"),
        (train.len() / 2, "50%"),
        (train.len() * 3 / 4, "75%"),
        (train.len(), "100%"),
    ];

    let max_offset = 5;
    let ctx_len = 5;
    let eval_size = test.len().min(30000);

    println!("\n═══════════════════════════════════════════════════════");
    println!("  SCALING: Does more data improve coverage?");
    println!("═══════════════════════════════════════════════════════\n");

    for &(n_train, label) in &data_sizes {
        let t1 = Instant::now();
        let train_subset = &train[..n_train];

        // Count
        let mut unigram = vec![0u32; vs];
        let mut pair_counts: Vec<HashMap<(u32,u32), u32>> = (0..max_offset).map(|_| HashMap::new()).collect();
        for t in 0..train_subset.len() {
            unigram[train_subset[t] as usize] += 1;
            for k in 1..=max_offset {
                if t+k < train_subset.len() {
                    *pair_counts[k-1].entry((train_subset[t], train_subset[t+k])).or_insert(0) += 1;
                }
            }
        }

        // Cond prob
        let mut cprob: Vec<Vec<(u32, f64)>> = vec![Vec::new(); vs];
        for k in 0..max_offset {
            for (&(a,b), &count) in &pair_counts[k] {
                let p = count as f64 / unigram[a as usize].max(1) as f64 / ((k+1) as f64).sqrt();
                if let Some(e) = cprob[a as usize].iter_mut().find(|(id,_)| *id == b) { e.1 += p; }
                else { cprob[a as usize].push((b, p)); }
            }
        }
        for n in &mut cprob { n.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap()); }

        // Top 100 freq
        let mut freq: Vec<(u32,u32)> = unigram.iter().enumerate().map(|(i,&c)|(i as u32,c)).collect();
        freq.sort_by(|a,b| b.1.cmp(&a.1));
        let top100: Vec<u32> = freq.iter().take(100).map(|(id,_)| *id).collect();
        let nt = train_subset.len() as f64;

        // Coverage + nomination test
        let mut cov50 = 0u64; let mut cov100 = 0u64;
        let mut nom_top1 = 0u64; let mut nom_top5 = 0u64;
        let mut cp_top1 = 0u64; let mut cp_top5 = 0u64;
        let mut total = 0u64;

        for t in ctx_len..eval_size {
            let true_next = test[t];
            if true_next == 0 { continue; }

            // Generate candidates + track nominations
            let mut cand_scores: HashMap<u32, f64> = HashMap::new();
            let mut nominations: HashMap<u32, u32> = HashMap::new(); // how many ctx positions nominate this

            for k in 1..=ctx_len {
                if t < k { continue; }
                let ctx_tok = test[t-k] as usize;
                let w = 1.0 / (k as f64).sqrt();
                let top50_for_this_ctx: Vec<u32> = cprob[ctx_tok].iter().take(50).map(|(id,_)| *id).collect();
                for &(n, p) in cprob[ctx_tok].iter().take(50) {
                    *cand_scores.entry(n).or_insert(0.0) += p * w;
                }
                // Count nominations
                for &n in &top50_for_this_ctx {
                    *nominations.entry(n).or_insert(0) += 1;
                }
            }

            // Add frequent words
            for &fid in &top100 {
                cand_scores.entry(fid).or_insert(unigram[fid as usize] as f64 / nt * 0.5);
                // Frequent words get 0 nominations (they're always-available, not context-specific)
            }

            // Coverage
            let mut combined: Vec<(u32, f64)> = cand_scores.iter().map(|(&k,&v)|(k,v)).collect();
            combined.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

            if combined.iter().take(150).any(|(id,_)| *id == true_next) { cov50 += 1; }
            if combined.iter().take(200).any(|(id,_)| *id == true_next) { cov100 += 1; }

            // Cond.prob only discrimination
            if combined.first().map(|(id,_)| *id) == Some(true_next) { cp_top1 += 1; }
            if combined.iter().take(5).any(|(id,_)| *id == true_next) { cp_top5 += 1; }

            // Nomination-count discrimination:
            // Re-rank by: nomination_count * weight + cond_prob_score
            // Words nominated by MORE context positions rank higher
            let mut nom_ranked: Vec<(u32, f64)> = combined.iter().map(|&(id, cp_score)| {
                let nom_count = *nominations.get(&id).unwrap_or(&0) as f64;
                // Score = nominations × 10 + cond_prob (nominations dominate)
                (id, nom_count * 10.0 + cp_score)
            }).collect();
            nom_ranked.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

            if nom_ranked.first().map(|(id,_)| *id) == Some(true_next) { nom_top1 += 1; }
            if nom_ranked.iter().take(5).any(|(id,_)| *id == true_next) { nom_top5 += 1; }

            total += 1;
        }

        let c50 = cov50 as f64 / total as f64 * 100.0;
        let c100 = cov100 as f64 / total as f64 * 100.0;
        let ct1 = cp_top1 as f64 / total as f64 * 100.0;
        let ct5 = cp_top5 as f64 / total as f64 * 100.0;
        let nt1 = nom_top1 as f64 / total as f64 * 100.0;
        let nt5 = nom_top5 as f64 / total as f64 * 100.0;

        println!("  Data: {} ({} tokens)", label, n_train);
        println!("    Coverage @150: {:.1}%, @200: {:.1}%", c50, c100);
        println!("    Cond.Prob:     top-1={:.1}%, top-5={:.1}%", ct1, ct5);
        println!("    Nomination:    top-1={:.1}%, top-5={:.1}%", nt1, nt5);
        println!("    [{:.1}s]", t1.elapsed().as_secs_f64());
        println!();
    }

    // ── Detailed nomination analysis on full data ──
    println!("═══════════════════════════════════════════════════════");
    println!("  NOMINATION ANALYSIS (full data)");
    println!("═══════════════════════════════════════════════════════\n");

    // Rebuild full stats
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
    let mut cprob: Vec<Vec<(u32, f64)>> = vec![Vec::new(); vs];
    for k in 0..max_offset {
        for (&(a,b), &count) in &pair_counts[k] {
            let p = count as f64 / unigram[a as usize].max(1) as f64 / ((k+1) as f64).sqrt();
            if let Some(e) = cprob[a as usize].iter_mut().find(|(id,_)| *id == b) { e.1 += p; }
            else { cprob[a as usize].push((b, p)); }
        }
    }
    for n in &mut cprob { n.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap()); }

    // Analyze: when the correct word gets 3+ nominations, what's the accuracy?
    let mut by_nom_count: Vec<(u64, u64)> = vec![(0,0); 6]; // [0,1,2,3,4,5] nominations

    for t in ctx_len..eval_size {
        let true_next = test[t];
        if true_next == 0 { continue; }

        let mut nominations: HashMap<u32, u32> = HashMap::new();
        for k in 1..=ctx_len {
            if t < k { continue; }
            let ctx_tok = test[t-k] as usize;
            for &(n, _) in cprob[ctx_tok].iter().take(50) {
                *nominations.entry(n).or_insert(0) += 1;
            }
        }

        let true_noms = *nominations.get(&true_next).unwrap_or(&0) as usize;
        let true_noms_capped = true_noms.min(5);
        by_nom_count[true_noms_capped].1 += 1; // total with this many noms

        // Is the highest-nominated candidate the correct one?
        let max_nom = nominations.values().copied().max().unwrap_or(0);
        if *nominations.get(&true_next).unwrap_or(&0) == max_nom && max_nom > 0 {
            by_nom_count[true_noms_capped].0 += 1;
        }
    }

    println!("  When correct word has N nominations, how often is it the TOP nominee?");
    println!("  {:>10} | {:>10} | {:>10} | {:>10}", "Nominations", "Occurrences", "Is Top?", "Rate");
    println!("  ──────────────────────────────────────────────────");
    for (noms, &(correct, total)) in by_nom_count.iter().enumerate() {
        if total > 0 {
            let rate = correct as f64 / total as f64 * 100.0;
            println!("  {:>10} | {:>10} | {:>10} | {:>9.1}%", noms, total, correct, rate);
        }
    }

    // Show examples where nomination count is high
    println!("\n  Examples of high-nomination correct predictions:");
    let mut shown = 0;
    for t in ctx_len..eval_size.min(50000) {
        if shown >= 5 { break; }
        let true_next = test[t];
        if true_next == 0 { continue; }

        let mut nominations: HashMap<u32, u32> = HashMap::new();
        let mut cand_scores: HashMap<u32, f64> = HashMap::new();
        for k in 1..=ctx_len {
            if t < k { continue; }
            let ctx_tok = test[t-k] as usize;
            let w = 1.0 / (k as f64).sqrt();
            for &(n, p) in cprob[ctx_tok].iter().take(50) {
                *nominations.entry(n).or_insert(0) += 1;
                *cand_scores.entry(n).or_insert(0.0) += p * w;
            }
        }

        let true_noms = *nominations.get(&true_next).unwrap_or(&0);
        if true_noms >= 3 {
            let ctx_words: Vec<&str> = (1..=ctx_len.min(t))
                .map(|k| i2w[test[t-k] as usize].as_str()).collect();
            let top3: Vec<String> = {
                let mut sorted: Vec<(u32,u32)> = nominations.iter().map(|(&k,&v)|(k,v)).collect();
                sorted.sort_by(|a,b| b.1.cmp(&a.1));
                sorted.iter().take(3).map(|(id,n)| format!("{}({}noms)", i2w[*id as usize], n)).collect()
            };
            println!("    Context: [{}] → correct: {}({}noms) | top nominees: [{}]",
                ctx_words.join(", "), i2w[true_next as usize], true_noms, top3.join(", "));
            shown += 1;
        }
    }

    println!("\n  Total time: {:.0}s", t0.elapsed().as_secs_f64());
}
