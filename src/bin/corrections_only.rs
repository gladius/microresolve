//! Corrections-only: skip PMI/SVD (we have those numbers), go straight to
//! least-squares + corrections. 16 minute hard cutoff.

use std::collections::HashMap;
use std::time::Instant;

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

struct Mat { rows: usize, cols: usize, data: Vec<f64> }
impl Mat {
    fn zeros(r: usize, c: usize) -> Self { Mat { rows: r, cols: c, data: vec![0.0; r*c] } }
    fn get(&self, i: usize, j: usize) -> f64 { self.data[i * self.cols + j] }
    fn set(&mut self, i: usize, j: usize, v: f64) { self.data[i * self.cols + j] = v; }
    fn add(&mut self, i: usize, j: usize, v: f64) { self.data[i * self.cols + j] += v; }
}

const CUTOFF_SECS: u64 = 960; // 16 minutes hard cutoff

fn main() {
    let t0 = Instant::now();
    println!("Corrections-only test. 16 min cutoff.");
    println!();

    // Load
    let raw = std::fs::read_to_string("/home/gladius/Workspace/neocortex/data/tinystories.txt")
        .expect("TinyStories not found");
    let stories: Vec<Vec<String>> = raw.split("<|endoftext|>")
        .map(|s| words(s.trim())).filter(|w| w.len() >= 10).collect();

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

    // PMI indexed lookup
    let max_offset = 5;
    let ctx_len = 5;
    let n_total = train.len() as f64;
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
    let unigram_p: Vec<f64> = unigram.iter().map(|&c| (c as f64+1.0)/(n_total+vs as f64)).collect();

    let mut pmi_lookup: Vec<Vec<Vec<(u32, f64)>>> = vec![vec![Vec::new(); vs]; max_offset];
    for k in 0..max_offset {
        let np = total_pairs[k] as f64;
        if np == 0.0 { continue; }
        for (&(a,b), &count) in &pair_counts[k] {
            let pmi = ((count as f64/np) / (unigram_p[a as usize]*unigram_p[b as usize])).ln();
            if pmi > 0.0 { pmi_lookup[k][a as usize].push((b, pmi)); }
        }
    }
    println!("PMI indexed: {:.1}s", t0.elapsed().as_secs_f64());

    // Build least-squares target matrix (same as before but faster dim)
    let dim = 300; // smaller for speed
    let sample_every = (train.len() / 300_000).max(1);
    let mut qtq = Mat::zeros(dim, dim);
    let mut ctq = Mat::zeros(vs, dim);

    for t in (ctx_len..train.len()).step_by(sample_every) {
        let true_next = train[t] as usize;
        let mut ctx_vec = vec![0.0f64; dim];
        for k in 1..=ctx_len {
            if t < k { continue; }
            ctx_vec[train[t-k] as usize % dim] += 1.0 / (k as f64).sqrt();
        }
        for i in 0..dim { if ctx_vec[i] == 0.0 { continue; }
            for j in 0..dim { qtq.add(i, j, ctx_vec[i]*ctx_vec[j]); }
        }
        for j in 0..dim { if ctx_vec[j] == 0.0 { continue; }
            ctq.add(true_next, j, ctx_vec[j]);
        }
    }
    for i in 0..dim { qtq.add(i, i, 0.01); }

    // Invert
    let mut work = Mat::zeros(dim, 2*dim);
    for i in 0..dim { for j in 0..dim { work.set(i,j,qtq.get(i,j)); } work.set(i,dim+i,1.0); }
    for col in 0..dim {
        let mut mx = work.get(col,col).abs(); let mut mr = col;
        for r in (col+1)..dim { let v = work.get(r,col).abs(); if v > mx { mx=v; mr=r; } }
        if mx < 1e-12 { continue; }
        if mr != col { for j in 0..2*dim { let a=work.get(col,j);let b=work.get(mr,j);work.set(col,j,b);work.set(mr,j,a);} }
        let p = work.get(col,col);
        for j in 0..2*dim { let v=work.get(col,j); work.set(col,j,v/p); }
        for r in 0..dim { if r==col{continue;} let f=work.get(r,col); if f.abs()<1e-15{continue;}
            for j in 0..2*dim { let v=work.get(r,j)-f*work.get(col,j); work.set(r,j,v); }
        }
    }
    let mut inv = Mat::zeros(dim,dim);
    for i in 0..dim { for j in 0..dim { inv.set(i,j,work.get(i,dim+j)); } }

    let mut a_star = Mat::zeros(vs, dim);
    for i in 0..vs { for j in 0..dim {
        let mut v = 0.0; for k in 0..dim { v += ctq.get(i,k)*inv.get(k,j); } a_star.set(i,j,v);
    }}
    println!("Least-squares: {:.1}s", t0.elapsed().as_secs_f64());

    // Quick eval before corrections
    let eval_size = 10_000;
    let eval = |a: &Mat, label: &str| -> (f64, f64) {
        let mut c1=0u64; let mut c5=0u64; let mut ct=0u64;
        for t in ctx_len..test.len().min(eval_size) {
            let tn = test[t] as usize;
            let mut cv = vec![0.0f64; dim];
            for k in 1..=ctx_len { if t<k{continue;} cv[test[t-k] as usize%dim]+=1.0/(k as f64).sqrt(); }
            let mut sc = vec![0.0f64; vs];
            for i in 0..vs { for j in 0..dim { if cv[j]==0.0{continue;} sc[i]+=a.get(i,j)*cv[j]; } }
            // Add PMI
            for k in 1..=ctx_len.min(max_offset) { if t<k{continue;}
                let ct_tok=test[t-k] as usize; let w=1.0/(k as f64).sqrt();
                for &(b,pmi) in &pmi_lookup[k-1][ct_tok] { sc[b as usize]+=pmi*w*0.5; }
            }
            for i in 0..vs { sc[i]+=unigram_p[i].ln()*0.2; }
            let pred=sc.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==tn{c1+=1;}
            let mut idx:Vec<(usize,f64)>=sc.iter().enumerate().map(|(i,&s)|(i,s)).collect();
            idx.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap());
            if idx.iter().take(5).any(|(i,_)|*i==tn){c5+=1;}
            ct+=1;
        }
        let t1=c1 as f64/ct as f64*100.0; let t5=c5 as f64/ct as f64*100.0;
        println!("  {}: top-1={:.1}%, top-5={:.1}% ({} tokens, {:.0}s)", label, t1, t5, ct, t0.elapsed().as_secs_f64());
        (t1, t5)
    };

    eval(&a_star, "Before corrections");

    // Corrections
    println!("\nCorrections (perceptron, 5 passes)...");
    let corr_lr = 0.01;
    let sample = (train.len() / 150_000).max(1);

    for pass in 1..=5 {
        if t0.elapsed().as_secs() > CUTOFF_SECS { println!("  CUTOFF reached at pass {}", pass); break; }
        let mut corrections = 0u64;
        for t in (ctx_len..train.len()).step_by(sample) {
            let tn = train[t] as usize;
            let mut cv = vec![0.0f64; dim];
            for k in 1..=ctx_len { if t<k{continue;} cv[train[t-k] as usize%dim]+=1.0/(k as f64).sqrt(); }
            let mut sc = vec![0.0f64; vs];
            for i in 0..vs { for j in 0..dim { if cv[j]==0.0{continue;} sc[i]+=a_star.get(i,j)*cv[j]; } }
            for k in 1..=ctx_len.min(max_offset) { if t<k{continue;}
                let ct_tok=train[t-k] as usize; let w=1.0/(k as f64).sqrt();
                for &(b,pmi) in &pmi_lookup[k-1][ct_tok] { sc[b as usize]+=pmi*w*0.5; }
            }
            for i in 0..vs { sc[i]+=unigram_p[i].ln()*0.2; }
            let pred=sc.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred!=tn {
                corrections+=1;
                for j in 0..dim { if cv[j]==0.0{continue;} a_star.add(tn,j,corr_lr*cv[j]); a_star.add(pred,j,-corr_lr*cv[j]); }
            }
        }
        let (t1, t5) = eval(&a_star, &format!("Pass {}", pass));
        println!("    corrections={}, elapsed={:.0}s", corrections, t0.elapsed().as_secs_f64());

        if t0.elapsed().as_secs() > CUTOFF_SECS { println!("  CUTOFF"); break; }
    }

    // Generate text
    println!("\n═══════════════════════════════════════════");
    println!("  TEXT GENERATION (zero neural, zero gradients)");
    println!("═══════════════════════════════════════════\n");

    let seeds = ["once upon a time", "the little girl", "she was happy because",
                 "he wanted to play but", "one day the boy found"];
    let mut rng: u64 = 42;
    for seed in &seeds {
        let mut seq: Vec<u32> = seed.split_whitespace().map(|w| *w2i.get(w).unwrap_or(&0)).collect();
        for _ in 0..25 {
            let t = seq.len();
            let mut cv = vec![0.0f64; dim];
            for k in 1..=ctx_len.min(t) { cv[seq[t-k] as usize%dim]+=1.0/(k as f64).sqrt(); }
            let mut sc = vec![0.0f64; vs];
            for i in 0..vs { for j in 0..dim { if cv[j]==0.0{continue;} sc[i]+=a_star.get(i,j)*cv[j]; } }
            for k in 1..=ctx_len.min(max_offset).min(t) {
                let ct=seq[t-k] as usize; let w=1.0/(k as f64).sqrt();
                for &(b,pmi) in &pmi_lookup[k-1][ct] { sc[b as usize]+=pmi*w*0.5; }
            }
            for i in 0..vs { sc[i]+=unigram_p[i].ln()*0.2; }
            let temp=0.8; let mx=sc.iter().cloned().fold(f64::NEG_INFINITY,f64::max);
            let exps:Vec<f64>=sc.iter().map(|s|((s-mx)/temp).exp()).collect();
            let sum:f64=exps.iter().sum();
            rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;
            let r=(rng%1_000_000) as f64/1_000_000.0;
            let mut cum=0.0; let mut chosen=0u32;
            for (i,&e) in exps.iter().enumerate() { cum+=e/sum; if cum>=r{chosen=i as u32;break;} }
            seq.push(chosen);
        }
        let text:String=seq.iter().map(|&id|i2w.get(id as usize).map(|s|s.as_str()).unwrap_or("?")).collect::<Vec<_>>().join(" ");
        println!("  \"{}\"", text);
        println!();
    }

    println!("Total: {:.0}s", t0.elapsed().as_secs_f64());
}
