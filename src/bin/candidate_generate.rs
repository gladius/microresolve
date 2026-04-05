//! Generate text using candidate-then-select. No neural network.
//! Candidates from conditional probability + top-100 frequent.
//! Select by: cond.prob score (best method so far).
//! READ the output. Is it coherent English?

use std::collections::HashMap;
use std::time::Instant;
use std::io::BufRead;

fn main() {
    let t0 = Instant::now();
    println!("Text Generation via Candidate-Select (zero neural)");
    println!();

    // Load TinyStories
    let raw = std::fs::read_to_string("/home/gladius/Workspace/neocortex/data/tinystories.txt")
        .expect("TinyStories not found");
    let stories: Vec<Vec<String>> = raw.split("<|endoftext|>")
        .map(|s| s.to_lowercase().split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|w| !w.is_empty()).map(|w| w.to_string()).collect::<Vec<_>>())
        .filter(|w| w.len() >= 10).collect();

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

    // Build conditional probability
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

    let mut cprob: Vec<Vec<(u32, f64)>> = vec![Vec::new(); vs];
    for k in 0..max_offset {
        for (&(a,b), &count) in &pair_counts[k] {
            let p = count as f64 / unigram[a as usize].max(1) as f64 / ((k+1) as f64).sqrt();
            if let Some(e) = cprob[a as usize].iter_mut().find(|(id,_)| *id == b) { e.1 += p; }
            else { cprob[a as usize].push((b, p)); }
        }
    }
    for n in &mut cprob { n.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap()); }

    let mut freq: Vec<(u32,u32)> = unigram.iter().enumerate().map(|(i,&c)|(i as u32,c)).collect();
    freq.sort_by(|a,b| b.1.cmp(&a.1));
    let top100: Vec<u32> = freq.iter().take(100).map(|(id,_)| *id).collect();
    let nt = train.len() as f64;

    println!("Vocab: {}, Stats computed: {:.1}s\n", vs, t0.elapsed().as_secs_f64());

    // ── Generate text ──
    let seeds = [
        "once upon a time there",
        "the little girl was very",
        "she was happy because her",
        "he wanted to play with",
        "one day the boy found a",
        "the dog ran to the",
        "they went to the park and",
        "mommy said you need to",
        "the sun was shining and the",
        "but then something happened the",
        "he was sad because he",
        "she looked at the big",
        "the bird flew over the",
        "it was a very cold",
        "after school they decided to",
    ];

    let mut rng: u64 = 12345;

    println!("═══════════════════════════════════════════════════════");
    println!("  TEXT GENERATION: Candidate-Select (zero neural)");
    println!("  Method: top-50 cond.prob + top-100 frequent");
    println!("  Selection: weighted score, temperature sampling");
    println!("═══════════════════════════════════════════════════════\n");

    for seed in &seeds {
        let mut seq: Vec<u32> = seed.split_whitespace()
            .map(|w| *w2i.get(&w.to_lowercase()).unwrap_or(&0)).collect();

        for _ in 0..40 {
            let t = seq.len();
            let mut cand_scores: HashMap<u32, f64> = HashMap::new();
            let mut nominations: HashMap<u32, u32> = HashMap::new();

            for k in 1..=ctx_len.min(t) {
                let ctx_tok = seq[t-k] as usize;
                let w = 1.0 / (k as f64).sqrt();
                for &(n, p) in cprob[ctx_tok].iter().take(50) {
                    *cand_scores.entry(n).or_insert(0.0) += p * w;
                    *nominations.entry(n).or_insert(0) += 1;
                }
            }
            for &fid in &top100 {
                cand_scores.entry(fid).or_insert(unigram[fid as usize] as f64 / nt * 0.3);
            }

            // Boost candidates with more nominations
            let mut final_scores: Vec<(u32, f64)> = cand_scores.iter().map(|(&id, &score)| {
                let nom_bonus = *nominations.get(&id).unwrap_or(&0) as f64;
                (id, score * (1.0 + nom_bonus * 0.5))
            }).collect();
            final_scores.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

            // Temperature sampling from top candidates
            let top_n = 20;
            let temp = 0.6;
            let candidates: Vec<(u32, f64)> = final_scores.iter().take(top_n).cloned().collect();

            if candidates.is_empty() { break; }

            let max_s = candidates.iter().map(|(_,s)| *s).fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = candidates.iter().map(|(_,s)| ((s - max_s) / temp).exp()).collect();
            let sum: f64 = exps.iter().sum();

            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let r = (rng % 1_000_000) as f64 / 1_000_000.0;
            let mut cum = 0.0;
            let mut chosen = candidates[0].0;
            for (i, &e) in exps.iter().enumerate() {
                cum += e / sum;
                if cum >= r { chosen = candidates[i].0; break; }
            }
            seq.push(chosen);
        }

        let text: String = seq.iter()
            .map(|&id| i2w.get(id as usize).map(|s| s.as_str()).unwrap_or("?"))
            .collect::<Vec<_>>().join(" ");
        println!("  \"{}\"\n", text);
    }

    // ── Also generate from Wikipedia data if available ──
    if std::path::Path::new("/tmp/wiki_full.txt").exists() {
        println!("\n═══════════════════════════════════════════════════════");
        println!("  TEXT GENERATION: Wikipedia-trained candidate-select");
        println!("═══════════════════════════════════════════════════════\n");

        // Load wiki (limited to 10M tokens for memory)
        let mut wiki_words: Vec<String> = Vec::new();
        let file = std::io::BufReader::new(std::fs::File::open("/tmp/wiki_full.txt").unwrap());
        for line in file.lines() {
            if let Ok(l) = line {
                for w in l.to_lowercase().split(|c: char| !c.is_alphanumeric() && c != '\'') {
                    if !w.is_empty() {
                        wiki_words.push(w.to_string());
                        if wiki_words.len() >= 10_000_000 { break; }
                    }
                }
                if wiki_words.len() >= 10_000_000 { break; }
            }
        }

        let mut wcounts: HashMap<String, usize> = HashMap::new();
        for w in &wiki_words { *wcounts.entry(w.clone()).or_insert(0) += 1; }
        let mut ww2i: HashMap<String, u32> = HashMap::new();
        let mut wi2w: Vec<String> = vec!["<unk>".to_string()];
        ww2i.insert("<unk>".to_string(), 0);
        let mut wsorted: Vec<(String,usize)> = wcounts.into_iter().filter(|(_,c)|*c>=5).collect();
        wsorted.sort_by(|a,b| b.1.cmp(&a.1));
        for (w,_) in &wsorted { let id=wi2w.len() as u32; ww2i.insert(w.clone(),id); wi2w.push(w.clone()); }
        let wvs = wi2w.len();
        let wids: Vec<u32> = wiki_words.iter().map(|w| *ww2i.get(w).unwrap_or(&0)).collect();

        let mut wunigram = vec![0u32; wvs];
        let mut wpair: Vec<HashMap<(u32,u32),u32>> = (0..max_offset).map(|_| HashMap::new()).collect();
        for t in 0..wids.len() {
            wunigram[wids[t] as usize] += 1;
            for k in 1..=max_offset {
                if t+k < wids.len() { *wpair[k-1].entry((wids[t],wids[t+k])).or_insert(0) += 1; }
            }
        }
        let mut wcprob: Vec<Vec<(u32,f64)>> = vec![Vec::new(); wvs];
        for k in 0..max_offset {
            for (&(a,b),&c) in &wpair[k] {
                let p = c as f64 / wunigram[a as usize].max(1) as f64 / ((k+1) as f64).sqrt();
                if let Some(e) = wcprob[a as usize].iter_mut().find(|(id,_)|*id==b) { e.1+=p; }
                else { wcprob[a as usize].push((b,p)); }
            }
        }
        for n in &mut wcprob { n.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap()); }
        let mut wfreq: Vec<(u32,u32)> = wunigram.iter().enumerate().map(|(i,&c)|(i as u32,c)).collect();
        wfreq.sort_by(|a,b| b.1.cmp(&a.1));
        let wtop100: Vec<u32> = wfreq.iter().take(100).map(|(id,_)|*id).collect();
        let wnt = wids.len() as f64;

        println!("  Wiki vocab: {}, tokens: {}\n", wvs, wids.len());

        let wiki_seeds = [
            "the united states of america",
            "the population of the city",
            "in the year of the",
            "water is made up of",
            "the first world war was",
            "scientists discovered that the",
            "the temperature of the earth",
            "many people believe that",
        ];

        for seed in &wiki_seeds {
            let mut seq: Vec<u32> = seed.split_whitespace()
                .map(|w| *ww2i.get(&w.to_lowercase()).unwrap_or(&0)).collect();

            for _ in 0..40 {
                let t = seq.len();
                let mut cs: HashMap<u32, f64> = HashMap::new();
                let mut noms: HashMap<u32, u32> = HashMap::new();
                for k in 1..=ctx_len.min(t) {
                    let ct = seq[t-k] as usize;
                    let w = 1.0 / (k as f64).sqrt();
                    for &(n,p) in wcprob[ct].iter().take(50) {
                        *cs.entry(n).or_insert(0.0) += p * w;
                        *noms.entry(n).or_insert(0) += 1;
                    }
                }
                for &fid in &wtop100 { cs.entry(fid).or_insert(wunigram[fid as usize] as f64/wnt*0.3); }

                let mut fs: Vec<(u32,f64)> = cs.iter().map(|(&id,&s)| {
                    let nb = *noms.get(&id).unwrap_or(&0) as f64;
                    (id, s * (1.0 + nb * 0.5))
                }).collect();
                fs.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

                let top_n = 20; let temp = 0.6;
                let cands: Vec<(u32,f64)> = fs.iter().take(top_n).cloned().collect();
                if cands.is_empty() { break; }
                let mx = cands.iter().map(|(_,s)|*s).fold(f64::NEG_INFINITY,f64::max);
                let exps: Vec<f64> = cands.iter().map(|(_,s)|((s-mx)/temp).exp()).collect();
                let sum: f64 = exps.iter().sum();
                rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;
                let r=(rng%1_000_000) as f64/1_000_000.0;
                let mut cum=0.0; let mut chosen=cands[0].0;
                for (i,&e) in exps.iter().enumerate() { cum+=e/sum; if cum>=r{chosen=cands[i].0;break;} }
                seq.push(chosen);
            }

            let text: String = seq.iter()
                .map(|&id| wi2w.get(id as usize).map(|s|s.as_str()).unwrap_or("?"))
                .collect::<Vec<_>>().join(" ");
            println!("  \"{}\"\n", text);
        }
    }

    println!("Total: {:.1}s", t0.elapsed().as_secs_f64());
}
