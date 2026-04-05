//! Coverage test across multiple datasets.
//! Downloads 3 datasets, tests candidate coverage on each.
//! Also tests two-signal discrimination: cond.prob for candidates, PMI for re-ranking.

use std::collections::HashMap;
use std::time::Instant;

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

struct Dataset {
    name: String,
    train: Vec<u32>,
    test: Vec<u32>,
    w2i: HashMap<String, u32>,
    i2w: Vec<String>,
    vs: usize,
}

fn load_dataset(name: &str, text: &str) -> Dataset {
    let words = tokenize(text);
    let mut counts: HashMap<String, usize> = HashMap::new();
    for w in &words { *counts.entry(w.clone()).or_insert(0) += 1; }

    let mut w2i: HashMap<String, u32> = HashMap::new();
    let mut i2w: Vec<String> = vec!["<unk>".to_string()];
    w2i.insert("<unk>".to_string(), 0);
    let mut sorted: Vec<(String, usize)> = counts.into_iter().filter(|(_,c)| *c >= 3).collect();
    sorted.sort_by(|a,b| b.1.cmp(&a.1));
    for (w, _) in &sorted { let id = i2w.len() as u32; w2i.insert(w.clone(), id); i2w.push(w.clone()); }
    let vs = i2w.len();

    let ids: Vec<u32> = words.iter().map(|w| *w2i.get(w).unwrap_or(&0)).collect();
    let split = ids.len() * 90 / 100;

    Dataset {
        name: name.to_string(),
        train: ids[..split].to_vec(),
        test: ids[split..].to_vec(),
        w2i, i2w, vs
    }
}

fn test_dataset(ds: &Dataset) {
    let t0 = Instant::now();
    println!("\n============================================================");
    println!("  Dataset: {} | Vocab: {} | Train: {} | Test: {}",
        ds.name, ds.vs, ds.train.len(), ds.test.len());
    println!("============================================================");

    let max_offset = 5;
    let ctx_len = 5;

    // Count
    let mut unigram = vec![0u32; ds.vs];
    let mut pair_counts: Vec<HashMap<(u32,u32), u32>> = (0..max_offset).map(|_| HashMap::new()).collect();
    for t in 0..ds.train.len() {
        unigram[ds.train[t] as usize] += 1;
        for k in 1..=max_offset {
            if t+k < ds.train.len() {
                *pair_counts[k-1].entry((ds.train[t], ds.train[t+k])).or_insert(0) += 1;
            }
        }
    }
    let up: Vec<f64> = unigram.iter().map(|&c| (c as f64+1.0)/(ds.train.len() as f64+ds.vs as f64)).collect();

    // Conditional probability neighbors
    let mut cprob: Vec<Vec<(u32, f64)>> = vec![Vec::new(); ds.vs];
    for k in 0..max_offset {
        for (&(a,b), &count) in &pair_counts[k] {
            let p = count as f64 / unigram[a as usize].max(1) as f64 / ((k+1) as f64).sqrt();
            if let Some(e) = cprob[a as usize].iter_mut().find(|(id,_)| *id == b) { e.1 += p; }
            else { cprob[a as usize].push((b, p)); }
        }
    }
    for n in &mut cprob { n.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap()); }

    // PMI neighbors
    let mut pmi_neighbors: Vec<Vec<(u32, f64)>> = vec![Vec::new(); ds.vs];
    for k in 0..max_offset {
        let np: f64 = pair_counts[k].values().map(|&c| c as f64).sum();
        if np == 0.0 { continue; }
        for (&(a,b), &count) in &pair_counts[k] {
            let pmi = ((count as f64/np)/(up[a as usize]*up[b as usize])).ln();
            if pmi > 0.0 {
                if let Some(e) = pmi_neighbors[a as usize].iter_mut().find(|(id,_)| *id == b) {
                    e.1 += pmi / ((k+1) as f64).sqrt();
                } else {
                    pmi_neighbors[a as usize].push((b, pmi / ((k+1) as f64).sqrt()));
                }
            }
        }
    }
    for n in &mut pmi_neighbors { n.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap()); }

    // Top 100 frequent
    let mut freq: Vec<(u32,u32)> = unigram.iter().enumerate().map(|(i,&c)|(i as u32,c)).collect();
    freq.sort_by(|a,b| b.1.cmp(&a.1));
    let top100: Vec<u32> = freq.iter().take(100).map(|(id,_)| *id).collect();

    println!("  Stats computed: {:.1}s", t0.elapsed().as_secs_f64());

    // ── Coverage test ──
    let eval_size = ds.test.len().min(30000);
    let k_values = [10, 20, 50, 100, 200];

    let mut cov_cprob = vec![0u64; k_values.len()];
    let mut cov_combined = vec![0u64; k_values.len()];
    let mut total = 0u64;

    // ── Discrimination test (two-signal) ──
    let mut disc_cprob_only_top1 = 0u64;
    let mut disc_twosignal_top1 = 0u64;
    let mut disc_cprob_only_top5 = 0u64;
    let mut disc_twosignal_top5 = 0u64;

    for t in ctx_len..eval_size {
        let true_next = ds.test[t];
        if true_next == 0 { continue; }

        // Cond.prob candidates
        let mut cp_scores: HashMap<u32, f64> = HashMap::new();
        let mut pmi_scores: HashMap<u32, f64> = HashMap::new();

        for k in 1..=ctx_len {
            if t < k { continue; }
            let ctx_tok = ds.test[t-k] as usize;
            let w = 1.0 / (k as f64).sqrt();
            for &(n, p) in cprob[ctx_tok].iter().take(50) {
                *cp_scores.entry(n).or_insert(0.0) += p * w;
            }
            for &(n, p) in pmi_neighbors[ctx_tok].iter().take(50) {
                *pmi_scores.entry(n).or_insert(0.0) += p * w;
            }
        }

        // Combined candidates: cprob + top100 freq
        let mut combined: Vec<(u32, f64)> = cp_scores.iter().map(|(&k,&v)| (k,v)).collect();
        let nt = ds.train.len() as f64;
        for &fid in &top100 {
            if !cp_scores.contains_key(&fid) {
                combined.push((fid, unigram[fid as usize] as f64 / nt * 0.5));
            }
        }
        combined.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

        // Coverage
        let mut cp_sorted: Vec<(u32,f64)> = cp_scores.iter().map(|(&k,&v)|(k,v)).collect();
        cp_sorted.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

        for (ki, &kv) in k_values.iter().enumerate() {
            if cp_sorted.iter().take(kv).any(|(id,_)| *id == true_next) { cov_cprob[ki] += 1; }
            if combined.iter().take(kv + 100).any(|(id,_)| *id == true_next) { cov_combined[ki] += 1; }
        }

        // ── Discrimination: cond.prob only vs two-signal ──
        // Cond.prob only ranking (among combined candidates)
        if combined.first().map(|(id,_)| *id) == Some(true_next) { disc_cprob_only_top1 += 1; }
        if combined.iter().take(5).any(|(id,_)| *id == true_next) { disc_cprob_only_top5 += 1; }

        // Two-signal: re-rank by α×cprob + β×pmi
        let alpha = 1.0;
        let beta = 2.0; // PMI weighted more for discrimination
        let mut two_signal: Vec<(u32, f64)> = combined.iter().map(|&(id, cp)| {
            let pmi = pmi_scores.get(&id).copied().unwrap_or(0.0);
            (id, alpha * cp + beta * pmi)
        }).collect();
        two_signal.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

        if two_signal.first().map(|(id,_)| *id) == Some(true_next) { disc_twosignal_top1 += 1; }
        if two_signal.iter().take(5).any(|(id,_)| *id == true_next) { disc_twosignal_top5 += 1; }

        total += 1;
    }

    // Print results
    println!("\n  COVERAGE:");
    println!("  {:>6} | {:>10} | {:>14}", "Top-K", "Cond.Prob", "CProb+Freq100");
    println!("  ─────────────────────────────────────");
    for (ki, &kv) in k_values.iter().enumerate() {
        let c = cov_cprob[ki] as f64 / total as f64 * 100.0;
        let cb = cov_combined[ki] as f64 / total as f64 * 100.0;
        println!("  {:>6} | {:>9.1}% | {:>13.1}%", kv, c, cb);
    }

    let d1_cp = disc_cprob_only_top1 as f64 / total as f64 * 100.0;
    let d5_cp = disc_cprob_only_top5 as f64 / total as f64 * 100.0;
    let d1_ts = disc_twosignal_top1 as f64 / total as f64 * 100.0;
    let d5_ts = disc_twosignal_top5 as f64 / total as f64 * 100.0;

    println!("\n  DISCRIMINATION (from candidates):");
    println!("  {:>20} | {:>8} | {:>8}", "Method", "Top-1", "Top-5");
    println!("  ─────────────────────────────────────");
    println!("  {:>20} | {:>7.1}% | {:>7.1}%", "Cond.Prob only", d1_cp, d5_cp);
    println!("  {:>20} | {:>7.1}% | {:>7.1}%", "CProb + PMI (2-sig)", d1_ts, d5_ts);
    println!("  {:>20} | {:>7.1}% | {:>7.1}%", "Random from cands", 100.0/150.0, 500.0/150.0);

    println!("\n  Tokens: {}, Time: {:.1}s", total, t0.elapsed().as_secs_f64());
}

fn main() {
    let t0 = Instant::now();
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Multi-Dataset Coverage + Two-Signal Discrimination ║");
    println!("╚══════════════════════════════════════════════════════╝");

    // Dataset 1: TinyStories
    let ts_raw = std::fs::read_to_string("/home/gladius/Workspace/neocortex/data/tinystories.txt")
        .expect("TinyStories not found");
    let ts_text: String = ts_raw.split("<|endoftext|>").take(5000).collect::<Vec<_>>().join(" ");
    let ds1 = load_dataset("TinyStories (children's stories)", &ts_text);
    test_dataset(&ds1);

    // Dataset 2: Grimm's Fairy Tales (download)
    println!("\n  Downloading Grimm's Fairy Tales...");
    let grimm_path = "/tmp/grimm.txt";
    if !std::path::Path::new(grimm_path).exists() {
        std::process::Command::new("curl").args(&["-sL", "https://www.gutenberg.org/cache/epub/2591/pg2591.txt", "-o", grimm_path]).output().ok();
    }
    if let Ok(grimm_raw) = std::fs::read_to_string(grimm_path) {
        let ds2 = load_dataset("Grimm's Fairy Tales (classic lit)", &grimm_raw);
        test_dataset(&ds2);
    } else { println!("  Failed to load Grimm's"); }

    // Dataset 3: Simple Wikipedia (use a different text source)
    println!("\n  Downloading Tiny Shakespeare...");
    let shakes_path = "/tmp/shakespeare.txt";
    if !std::path::Path::new(shakes_path).exists() {
        std::process::Command::new("curl").args(&["-sL", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "-o", shakes_path]).output().ok();
    }
    if let Ok(shakes_raw) = std::fs::read_to_string(shakes_path) {
        let ds3 = load_dataset("Tiny Shakespeare (plays/dialogue)", &shakes_raw);
        test_dataset(&ds3);
    } else { println!("  Failed to load Shakespeare"); }

    println!("\n\nTotal time: {:.0}s", t0.elapsed().as_secs_f64());
}
