//! Coverage + nomination test on Simple English Wikipedia
//! Same methodology as coverage_deep.rs but on Wikipedia data.

use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let t0 = Instant::now();
    println!("Coverage Test: Simple English Wikipedia (12K articles, ~10M tokens)");
    println!();

    // Read up to 50M tokens to stay within memory (~6GB max)
    let max_tokens = 50_000_000;
    println!("Loading Wikipedia (capped at {}M tokens to fit in memory)...", max_tokens / 1_000_000);
    let mut words: Vec<String> = Vec::new();
    let file = std::io::BufReader::new(std::fs::File::open("/tmp/wiki_full.txt").expect("wiki not found"));
    use std::io::BufRead;
    for line in file.lines() {
        if let Ok(l) = line {
            for w in l.to_lowercase().split(|c: char| !c.is_alphanumeric() && c != '\'') {
                if !w.is_empty() {
                    words.push(w.to_string());
                    if words.len() >= max_tokens { break; }
                }
            }
            if words.len() >= max_tokens { break; }
        }
    }
    println!("Loaded {} tokens ({:.0}MB text)", words.len(), words.len() as f64 * 5.0 / 1e6);

    // Vocab (words appearing 5+ times for diverse text)
    let mut counts: HashMap<String, usize> = HashMap::new();
    for w in &words { *counts.entry(w.clone()).or_insert(0) += 1; }
    let mut w2i: HashMap<String, u32> = HashMap::new();
    let mut i2w: Vec<String> = vec!["<unk>".to_string()];
    w2i.insert("<unk>".to_string(), 0);
    let mut sorted: Vec<(String, usize)> = counts.into_iter().filter(|(_,c)| *c >= 5).collect();
    sorted.sort_by(|a,b| b.1.cmp(&a.1));
    for (w, _) in &sorted { let id = i2w.len() as u32; w2i.insert(w.clone(), id); i2w.push(w.clone()); }
    let vs = i2w.len();

    let ids: Vec<u32> = words.iter().map(|w| *w2i.get(w).unwrap_or(&0)).collect();
    let unk = ids.iter().filter(|&&id| id == 0).count();
    println!("Vocab: {} (words appearing 5+), unk: {:.1}%", vs, unk as f64 / ids.len() as f64 * 100.0);

    let split = ids.len() * 90 / 100;
    let train = &ids[..split];
    let test = &ids[split..];
    println!("Train: {}, Test: {}", train.len(), test.len());

    // Build conditional probability + PMI
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
    let up: Vec<f64> = unigram.iter().map(|&c| (c as f64+1.0)/(train.len() as f64+vs as f64)).collect();

    let mut cprob: Vec<Vec<(u32, f64)>> = vec![Vec::new(); vs];
    for k in 0..max_offset {
        for (&(a,b), &count) in &pair_counts[k] {
            let p = count as f64 / unigram[a as usize].max(1) as f64 / ((k+1) as f64).sqrt();
            if let Some(e) = cprob[a as usize].iter_mut().find(|(id,_)| *id == b) { e.1 += p; }
            else { cprob[a as usize].push((b, p)); }
        }
    }
    for n in &mut cprob { n.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap()); }

    // Top 100 frequent
    let mut freq: Vec<(u32,u32)> = unigram.iter().enumerate().map(|(i,&c)|(i as u32,c)).collect();
    freq.sort_by(|a,b| b.1.cmp(&a.1));
    let top100: Vec<u32> = freq.iter().take(100).map(|(id,_)| *id).collect();
    let nt = train.len() as f64;

    println!("Stats computed: {:.1}s\n", t0.elapsed().as_secs_f64());

    // Coverage + Nomination analysis
    let eval_size = test.len().min(50000);
    let k_values = [10, 20, 50, 100, 200];
    let mut cov_cp = vec![0u64; k_values.len()];
    let mut cov_combined = vec![0u64; k_values.len()];

    // Nomination buckets
    let mut by_nom: Vec<(u64,u64)> = vec![(0,0); 6];
    let mut cp_top1 = 0u64; let mut cp_top5 = 0u64;
    let mut nom_top1 = 0u64; let mut nom_top5 = 0u64;
    // Product-based discrimination
    let mut prod_top1 = 0u64; let mut prod_top5 = 0u64;
    let mut total = 0u64;

    for t in ctx_len..eval_size {
        let true_next = test[t];
        if true_next == 0 { continue; }

        let mut cand_scores: HashMap<u32, f64> = HashMap::new();
        let mut nominations: HashMap<u32, u32> = HashMap::new();
        // Track per-position conditional probs for product
        let mut per_pos_probs: HashMap<u32, Vec<f64>> = HashMap::new();

        for k in 1..=ctx_len {
            if t < k { continue; }
            let ctx_tok = test[t-k] as usize;
            let w = 1.0 / (k as f64).sqrt();
            for &(n, p) in cprob[ctx_tok].iter().take(50) {
                *cand_scores.entry(n).or_insert(0.0) += p * w;
                *nominations.entry(n).or_insert(0) += 1;
                per_pos_probs.entry(n).or_insert_with(Vec::new).push(p);
            }
        }
        for &fid in &top100 {
            cand_scores.entry(fid).or_insert(unigram[fid as usize] as f64 / nt * 0.5);
        }

        let mut combined: Vec<(u32, f64)> = cand_scores.iter().map(|(&k,&v)|(k,v)).collect();
        combined.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

        // Coverage
        for (ki, &kv) in k_values.iter().enumerate() {
            if combined.iter().take(kv).any(|(id,_)| *id == true_next) { cov_cp[ki] += 1; }
            if combined.iter().take(kv + 100).any(|(id,_)| *id == true_next) { cov_combined[ki] += 1; }
        }

        // Cond.prob discrimination
        if combined.first().map(|(id,_)| *id) == Some(true_next) { cp_top1 += 1; }
        if combined.iter().take(5).any(|(id,_)| *id == true_next) { cp_top5 += 1; }

        // Nomination discrimination
        let mut nom_ranked: Vec<(u32, f64)> = combined.iter().map(|&(id, cp)| {
            let nc = *nominations.get(&id).unwrap_or(&0) as f64;
            (id, nc * 10.0 + cp)
        }).collect();
        nom_ranked.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        if nom_ranked.first().map(|(id,_)| *id) == Some(true_next) { nom_top1 += 1; }
        if nom_ranked.iter().take(5).any(|(id,_)| *id == true_next) { nom_top5 += 1; }

        // Product-based: geometric mean of per-position probabilities
        let mut prod_ranked: Vec<(u32, f64)> = combined.iter().map(|&(id, cp)| {
            let probs = per_pos_probs.get(&id);
            let prod_score = if let Some(ps) = probs {
                if ps.len() >= 2 {
                    // Geometric mean of probabilities × nomination bonus
                    let geo_mean = ps.iter().map(|p| (p + 0.001).ln()).sum::<f64>() / ps.len() as f64;
                    geo_mean.exp() * (ps.len() as f64).sqrt()
                } else {
                    cp * 0.5 // single nomination = weak
                }
            } else {
                cp * 0.1 // freq-only = weakest
            };
            (id, prod_score)
        }).collect();
        prod_ranked.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        if prod_ranked.first().map(|(id,_)| *id) == Some(true_next) { prod_top1 += 1; }
        if prod_ranked.iter().take(5).any(|(id,_)| *id == true_next) { prod_top5 += 1; }

        // Nomination bucket
        let true_noms = *nominations.get(&true_next).unwrap_or(&0) as usize;
        let capped = true_noms.min(5);
        by_nom[capped].1 += 1;
        let max_nom = nominations.values().copied().max().unwrap_or(0);
        if *nominations.get(&true_next).unwrap_or(&0) == max_nom && max_nom > 0 {
            by_nom[capped].0 += 1;
        }

        total += 1;
    }

    // Results
    println!("═══════════════════════════════════════════════════════");
    println!("  COVERAGE (Simple English Wikipedia)");
    println!("═══════════════════════════════════════════════════════");
    println!("  {:>6} | {:>10} | {:>14}", "Top-K", "Cond.Prob", "CProb+Freq100");
    println!("  ─────────────────────────────────────");
    for (ki, &kv) in k_values.iter().enumerate() {
        let c = cov_cp[ki] as f64 / total as f64 * 100.0;
        let cb = cov_combined[ki] as f64 / total as f64 * 100.0;
        println!("  {:>6} | {:>9.1}% | {:>13.1}%", kv, c, cb);
    }

    println!("\n  DISCRIMINATION:");
    println!("  {:>25} | {:>8} | {:>8}", "Method", "Top-1", "Top-5");
    println!("  ──────────────────────────────────────────");
    println!("  {:>25} | {:>7.1}% | {:>7.1}%", "Cond.Prob sum", cp_top1 as f64/total as f64*100.0, cp_top5 as f64/total as f64*100.0);
    println!("  {:>25} | {:>7.1}% | {:>7.1}%", "Nomination count", nom_top1 as f64/total as f64*100.0, nom_top5 as f64/total as f64*100.0);
    println!("  {:>25} | {:>7.1}% | {:>7.1}%", "Product (geometric mean)", prod_top1 as f64/total as f64*100.0, prod_top5 as f64/total as f64*100.0);
    println!("  {:>25} | {:>7.1}% | {:>7.1}%", "Random from candidates", 100.0/150.0, 500.0/150.0);

    println!("\n  NOMINATION BREAKDOWN:");
    println!("  {:>10} | {:>10} | {:>10} | {:>8}", "Noms", "Count", "Top?", "Rate");
    println!("  ──────────────────────────────────────────");
    for (n, &(correct, tot)) in by_nom.iter().enumerate() {
        if tot > 0 {
            println!("  {:>10} | {:>10} | {:>10} | {:>7.1}%", n, tot, correct, correct as f64/tot as f64*100.0);
        }
    }

    println!("\n  Tokens: {}, Vocab: {}, Time: {:.0}s", total, vs, t0.elapsed().as_secs_f64());
}
