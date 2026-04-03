//! Experiment: Can PMI + Spreading Activation Do Multi-Hop Reasoning?
//!
//! Thesis: Spreading activation over a co-occurrence graph can answer questions
//! that require multi-hop reasoning, WITHOUT any neural network.
//!
//! Two experiments:
//! 1. Synthetic knowledge base with known multi-hop paths
//! 2. Bitext customer support data — can spreading activation improve routing
//!    when query terms don't directly match the intent?
//!
//! Run: cargo run --release --bin reasoning_experiment

use std::collections::{HashMap, HashSet, BTreeMap};
use std::time::Instant;

// ============================================================================
// PART 1: Graph + Spreading Activation Engine
// ============================================================================

/// Co-occurrence graph: term → {neighbor → PMI weight}
struct KnowledgeGraph {
    edges: HashMap<String, HashMap<String, f64>>,
    // Document frequency for IDF
    df: HashMap<String, usize>,
    n_docs: usize,
}

impl KnowledgeGraph {
    fn new() -> Self {
        Self { edges: HashMap::new(), df: HashMap::new(), n_docs: 0 }
    }

    /// Build graph from documents. Each document is a bag of terms.
    /// Terms co-occurring in the same document are connected.
    fn build_from_docs(&mut self, docs: &[Vec<String>]) {
        self.n_docs = docs.len();
        // Count co-occurrences and document frequencies
        let mut cooccur: HashMap<(String, String), usize> = HashMap::new();
        let mut term_freq: HashMap<String, usize> = HashMap::new();

        for doc in docs {
            let unique: HashSet<&String> = doc.iter().collect();
            for term in &unique {
                *self.df.entry((*term).clone()).or_insert(0) += 1;
                *term_freq.entry((*term).clone()).or_insert(0) += 1;
            }
            let terms: Vec<&String> = unique.into_iter().collect();
            for i in 0..terms.len() {
                for j in (i+1)..terms.len() {
                    let pair = if terms[i] < terms[j] {
                        (terms[i].clone(), terms[j].clone())
                    } else {
                        (terms[j].clone(), terms[i].clone())
                    };
                    *cooccur.entry(pair).or_insert(0) += 1;
                }
            }
        }

        // Compute PMI and build edges
        let n = docs.len() as f64;
        for ((a, b), count) in &cooccur {
            let p_ab = *count as f64 / n;
            let p_a = *term_freq.get(a).unwrap() as f64 / n;
            let p_b = *term_freq.get(b).unwrap() as f64 / n;
            let pmi = (p_ab / (p_a * p_b)).ln();
            if pmi > 0.0 {
                // Normalize PMI to 0-1 range (cap at 5.0)
                let weight = (pmi / 5.0).min(1.0);
                self.edges.entry(a.clone()).or_default().insert(b.clone(), weight);
                self.edges.entry(b.clone()).or_default().insert(a.clone(), weight);
            }
        }
    }

    fn num_nodes(&self) -> usize {
        self.edges.len()
    }

    fn num_edges(&self) -> usize {
        self.edges.values().map(|e| e.len()).sum::<usize>() / 2
    }
}

/// Spreading activation over the knowledge graph.
/// Returns activation levels for all reached nodes.
fn spreading_activation(
    graph: &KnowledgeGraph,
    query_terms: &[String],
    max_rounds: usize,
    decay: f64,
    top_k: usize,
) -> Vec<(String, f64)> {
    let mut activation: HashMap<String, f64> = HashMap::new();

    // Initialize: query terms get activation 1.0
    for term in query_terms {
        if graph.edges.contains_key(term) {
            *activation.entry(term.clone()).or_insert(0.0) += 1.0;
        }
    }

    let query_set: HashSet<String> = query_terms.iter().cloned().collect();

    for _round in 0..max_rounds {
        let mut new_activation: HashMap<String, f64> = HashMap::new();

        // Spread from active nodes to neighbors
        for (term, act) in &activation {
            if let Some(neighbors) = graph.edges.get(term) {
                for (neighbor, weight) in neighbors {
                    let spread = act * weight * decay;
                    if spread > 0.01 { // threshold
                        *new_activation.entry(neighbor.clone()).or_insert(0.0) += spread;
                    }
                }
            }
            // Retain own activation with decay
            *new_activation.entry(term.clone()).or_insert(0.0) += act * 0.5;
        }

        // Convergence bonus: nodes reached from multiple paths get boosted
        // (This is the key reasoning mechanism — multiple evidence lines converge)
        for (term, act) in new_activation.iter_mut() {
            // Count how many query terms have a path to this node
            let paths = query_terms.iter().filter(|q| {
                graph.edges.get(*q)
                    .map(|e| e.contains_key(term))
                    .unwrap_or(false)
            }).count();
            if paths >= 2 {
                *act *= 1.0 + 0.3 * (paths as f64 - 1.0); // 30% bonus per extra path
            }
        }

        // Lateral inhibition: keep top-K
        let mut sorted: Vec<(String, f64)> = new_activation.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.truncate(top_k);

        activation = sorted.into_iter().collect();
    }

    // Remove query terms from results (we want DISCOVERED nodes, not echoes)
    let mut results: Vec<(String, f64)> = activation.into_iter()
        .filter(|(t, _)| !query_set.contains(t))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

/// Direct term match baseline: which answer terms co-occur with query terms?
fn direct_match(
    graph: &KnowledgeGraph,
    query_terms: &[String],
) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    let query_set: HashSet<String> = query_terms.iter().cloned().collect();

    for term in query_terms {
        if let Some(neighbors) = graph.edges.get(term) {
            for (neighbor, weight) in neighbors {
                if !query_set.contains(neighbor) {
                    *scores.entry(neighbor.clone()).or_insert(0.0) += weight;
                }
            }
        }
    }

    let mut results: Vec<(String, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

// ============================================================================
// PART 2: Synthetic Knowledge Base Experiment
// ============================================================================

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .filter(|w| !matches!(*w, "is" | "a" | "an" | "the" | "and" | "or" | "in" | "of"
            | "to" | "for" | "with" | "can" | "are" | "it" | "be" | "not" | "that"
            | "has" | "but" | "from" | "by" | "on" | "at" | "as" | "its" | "do"
            | "does" | "was" | "were" | "been" | "being" | "have" | "had"))
        .map(|w| w.to_string())
        .collect()
}

fn run_synthetic_experiment() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 1: Synthetic Multi-Hop Reasoning                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Knowledge base: facts as sentences
    let facts = vec![
        "aspirin is a nonsteroidal anti-inflammatory drug called NSAID",
        "ibuprofen is a nonsteroidal anti-inflammatory drug called NSAID",
        "NSAIDs reduce inflammation and treat pain",
        "headaches cause pain in the head region",
        "migraines are severe headaches with visual aura",
        "dehydration can cause headaches and dizziness",
        "drinking water prevents dehydration",
        "stress can cause headaches and muscle tension",
        "exercise reduces stress and improves mood",
        "exercise strengthens the heart and cardiovascular system",
        "the heart pumps blood through blood vessels",
        "high blood pressure damages blood vessels over time",
        "NSAIDs can increase blood pressure as side effect",
        "paracetamol treats pain but is not an NSAID",
        "paracetamol is safe for patients with high blood pressure",
        "fever indicates infection in the body",
        "ibuprofen reduces fever effectively",
        "aspirin reduces fever and blood clotting",
        "paracetamol reduces fever safely",
        "diabetes affects blood sugar levels",
        "insulin regulates blood sugar in the body",
        "metformin treats diabetes by reducing blood sugar",
        "kidney disease can result from diabetes complications",
        "kidneys filter blood and remove waste products",
        "NSAIDs can damage kidneys with prolonged use",
    ];

    let docs: Vec<Vec<String>> = facts.iter().map(|f| tokenize(f)).collect();

    let mut graph = KnowledgeGraph::new();
    graph.build_from_docs(&docs);

    println!("Knowledge base: {} facts", facts.len());
    println!("Graph: {} nodes, {} edges", graph.num_nodes(), graph.num_edges());
    println!();

    // Reasoning questions with known answers and hop distance
    struct Question {
        query: &'static str,
        answer_terms: Vec<&'static str>, // any of these = correct
        hops: usize,
        explanation: &'static str,
    }

    let questions = vec![
        Question {
            query: "what treats pain",
            answer_terms: vec!["aspirin", "ibuprofen", "paracetamol", "nsaid", "nsaids", "nonsteroidal"],
            hops: 1,
            explanation: "Direct: pain → {aspirin, ibuprofen, paracetamol, NSAID}",
        },
        Question {
            query: "what causes headaches",
            answer_terms: vec!["dehydration", "stress"],
            hops: 1,
            explanation: "Direct: headaches → {dehydration, stress}",
        },
        Question {
            query: "what treats headaches",
            answer_terms: vec!["aspirin", "ibuprofen", "paracetamol", "nsaid", "nsaids"],
            hops: 2,
            explanation: "2-hop: headaches→pain→{NSAIDs→aspirin/ibuprofen, paracetamol}",
        },
        Question {
            query: "how to prevent headaches",
            answer_terms: vec!["water", "exercise", "drinking"],
            hops: 2,
            explanation: "2-hop: headaches←dehydration←water, headaches←stress←exercise",
        },
        Question {
            query: "what reduces fever",
            answer_terms: vec!["aspirin", "ibuprofen", "paracetamol"],
            hops: 1,
            explanation: "Direct: fever → {ibuprofen, aspirin, paracetamol}",
        },
        Question {
            query: "exercise benefits",
            answer_terms: vec!["stress", "heart", "cardiovascular", "mood"],
            hops: 1,
            explanation: "Direct: exercise → {stress reduction, heart, mood}",
        },
        Question {
            query: "danger of aspirin with blood pressure",
            answer_terms: vec!["nsaid", "nsaids", "damage", "vessels", "increase"],
            hops: 2,
            explanation: "2-hop: aspirin→NSAID→blood pressure increase→damage",
        },
        Question {
            query: "safe painkiller for blood pressure patients",
            answer_terms: vec!["paracetamol"],
            hops: 2,
            explanation: "2-hop: pain+safe+blood pressure → paracetamol (not NSAIDs)",
        },
        Question {
            query: "connection between diabetes and kidneys",
            answer_terms: vec!["kidney", "disease", "complications", "blood"],
            hops: 1,
            explanation: "Direct: diabetes→kidney disease (via complications)",
        },
        Question {
            query: "why avoid NSAIDs with kidney disease",
            answer_terms: vec!["damage", "kidneys", "prolonged"],
            hops: 1,
            explanation: "Direct: NSAIDs→kidney damage",
        },
        Question {
            query: "aspirin ibuprofen common",
            answer_terms: vec!["nsaid", "nsaids", "nonsteroidal", "pain", "fever", "anti"],
            hops: 1,
            explanation: "Convergence: aspirin→NSAID←ibuprofen (shared category)",
        },
        Question {
            query: "water headaches connection",
            answer_terms: vec!["dehydration", "prevents", "cause"],
            hops: 2,
            explanation: "2-hop: water→dehydration→headaches",
        },
        Question {
            query: "exercise heart blood",
            answer_terms: vec!["cardiovascular", "vessels", "strengthens", "pumps"],
            hops: 2,
            explanation: "2-hop: exercise→heart→blood→vessels",
        },
        Question {
            query: "insulin diabetes",
            answer_terms: vec!["sugar", "blood", "regulates", "metformin"],
            hops: 1,
            explanation: "Direct: insulin→blood sugar←diabetes←metformin",
        },
        Question {
            query: "migraine visual symptoms",
            answer_terms: vec!["aura", "headaches", "severe"],
            hops: 1,
            explanation: "Direct: migraine→aura→visual",
        },
        // Hard 3-hop questions
        Question {
            query: "how does exercise help headaches",
            answer_terms: vec!["stress", "reduces"],
            hops: 2,
            explanation: "2-hop: exercise→reduces stress→stress causes headaches",
        },
        Question {
            query: "can aspirin affect kidneys",
            answer_terms: vec!["nsaid", "nsaids", "damage", "prolonged"],
            hops: 2,
            explanation: "2-hop: aspirin→NSAID→kidney damage",
        },
        Question {
            query: "diabetes blood vessel damage",
            answer_terms: vec!["kidney", "pressure", "complications"],
            hops: 2,
            explanation: "2-hop: diabetes→complications→kidney→blood→vessels",
        },
    ];

    println!("Testing {} questions (1-hop and 2-hop reasoning):", questions.len());
    println!("─────────────────────────────────────────────────────────────────");
    println!();

    let mut baseline_correct = 0;
    let mut spreading_correct = 0;
    let mut baseline_by_hop: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut spreading_by_hop: HashMap<usize, (usize, usize)> = HashMap::new();

    for q in &questions {
        let query_terms = tokenize(q.query);

        // Baseline: direct 1-hop neighbors only
        let baseline_results = direct_match(&graph, &query_terms);
        let baseline_top10: Vec<&str> = baseline_results.iter()
            .take(10)
            .map(|(t, _)| t.as_str())
            .collect();

        let baseline_hit = q.answer_terms.iter()
            .any(|a| baseline_top10.iter().any(|b| b.contains(a) || a.contains(b)));

        // Spreading activation: multi-hop
        let spread_results = spreading_activation(&graph, &query_terms, 3, 0.5, 30);
        let spread_top10: Vec<&str> = spread_results.iter()
            .take(10)
            .map(|(t, _)| t.as_str())
            .collect();

        let spread_hit = q.answer_terms.iter()
            .any(|a| spread_top10.iter().any(|b| b.contains(a) || a.contains(b)));

        if baseline_hit { baseline_correct += 1; }
        if spread_hit { spreading_correct += 1; }

        let bh = baseline_by_hop.entry(q.hops).or_insert((0, 0));
        bh.1 += 1;
        if baseline_hit { bh.0 += 1; }

        let sh = spreading_by_hop.entry(q.hops).or_insert((0, 0));
        sh.1 += 1;
        if spread_hit { sh.0 += 1; }

        let baseline_mark = if baseline_hit { "✓" } else { "✗" };
        let spread_mark = if spread_hit { "✓" } else { "✗" };
        let improved = if spread_hit && !baseline_hit { " ← SPREADING WINS" } else { "" };

        println!("  Q: \"{}\" ({}-hop)", q.query, q.hops);
        println!("    Expected: {:?}", q.answer_terms);
        println!("    Baseline top-5:  {} {:?}", baseline_mark,
            baseline_results.iter().take(5).map(|(t, s)| format!("{}({:.2})", t, s)).collect::<Vec<_>>());
        println!("    Spreading top-5: {} {:?}", spread_mark,
            spread_results.iter().take(5).map(|(t, s)| format!("{}({:.2})", t, s)).collect::<Vec<_>>());
        println!("    Path: {}{}", q.explanation, improved);
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("RESULTS:");
    println!("  Overall:  Baseline {}/{} ({:.0}%)  vs  Spreading {}/{} ({:.0}%)",
        baseline_correct, questions.len(), baseline_correct as f64 / questions.len() as f64 * 100.0,
        spreading_correct, questions.len(), spreading_correct as f64 / questions.len() as f64 * 100.0);
    println!();

    let mut hops: Vec<usize> = baseline_by_hop.keys().cloned().collect();
    hops.sort();
    for hop in &hops {
        let (bc, bt) = baseline_by_hop.get(hop).unwrap_or(&(0, 0));
        let (sc, st) = spreading_by_hop.get(hop).unwrap_or(&(0, 0));
        println!("  {}-hop:    Baseline {}/{} ({:.0}%)  vs  Spreading {}/{} ({:.0}%)",
            hop, bc, bt, *bc as f64 / *bt as f64 * 100.0,
            sc, st, *sc as f64 / *st as f64 * 100.0);
    }

    println!();
    if spreading_correct > baseline_correct {
        println!("  SPREADING ACTIVATION FOUND {} ANSWERS THAT DIRECT MATCH MISSED.",
            spreading_correct - baseline_correct);
        println!("  This demonstrates multi-hop reasoning without any neural network.");
    }
    println!();
}

// ============================================================================
// PART 3: Bitext Real-Data Experiment
// ============================================================================

#[derive(serde::Deserialize)]
struct Example {
    text: String,
    intents: Vec<String>,
}

fn run_bitext_experiment() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 2: Bitext Intent Routing via Spreading          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load Bitext data
    let path = "tests/data/benchmarks/bitext_all.json";
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => {
            println!("  Bitext data not found at {}, skipping.", path);
            return;
        }
    };

    let examples: Vec<Example> = match serde_json::from_str(&data) {
        Ok(e) => e,
        Err(e) => {
            println!("  Failed to parse Bitext: {}", e);
            return;
        }
    };

    println!("Loaded {} examples", examples.len());

    // Build PMI graph from all queries
    let docs: Vec<Vec<String>> = examples.iter().map(|e| tokenize(&e.text)).collect();
    let mut graph = KnowledgeGraph::new();
    graph.build_from_docs(&docs);

    println!("Graph: {} nodes, {} edges", graph.num_nodes(), graph.num_edges());

    // Build intent → term associations (which terms appear in queries of each intent)
    let mut intent_terms: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for ex in &examples {
        let terms = tokenize(&ex.text);
        for intent in &ex.intents {
            let entry = intent_terms.entry(intent.clone()).or_default();
            for term in &terms {
                *entry.entry(term.clone()).or_insert(0) += 1;
            }
        }
    }

    // For each intent, find its top discriminating terms (high TF, moderate DF)
    let mut intent_signatures: HashMap<String, Vec<String>> = HashMap::new();
    for (intent, terms) in &intent_terms {
        let mut scored: Vec<(String, f64)> = terms.iter()
            .filter_map(|(term, count)| {
                let df = graph.df.get(term).copied().unwrap_or(1) as f64;
                let idf = (graph.n_docs as f64 / df).ln();
                if idf > 1.0 { // discriminating
                    Some((term.clone(), *count as f64 * idf))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        intent_signatures.insert(intent.clone(), scored.into_iter().take(10).map(|(t, _)| t).collect());
    }

    // Test: For each query, route using:
    // A) Direct term match against intent signatures
    // B) Spreading activation from query terms, then match against signatures

    let mut baseline_correct = 0;
    let mut spreading_correct = 0;
    let total = examples.len();
    let mut improvements = Vec::new();

    for ex in &examples {
        let query_terms = tokenize(&ex.text);
        let true_intent = &ex.intents[0];

        // A) Baseline: score each intent by direct term overlap with query
        let mut baseline_scores: Vec<(String, f64)> = intent_signatures.iter()
            .map(|(intent, sig_terms)| {
                let overlap: f64 = sig_terms.iter()
                    .filter(|t| query_terms.contains(t))
                    .count() as f64;
                (intent.clone(), overlap)
            })
            .collect();
        baseline_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let baseline_top1 = baseline_scores.first().map(|(i, _)| i.as_str()).unwrap_or("");

        // B) Spreading: activate query terms, spread, then match enriched activation against signatures
        let spread_results = spreading_activation(&graph, &query_terms, 2, 0.4, 40);
        let spread_terms: HashMap<String, f64> = spread_results.into_iter().collect();

        let mut spread_scores: Vec<(String, f64)> = intent_signatures.iter()
            .map(|(intent, sig_terms)| {
                let score: f64 = sig_terms.iter()
                    .filter_map(|t| spread_terms.get(t))
                    .sum();
                (intent.clone(), score)
            })
            .collect();
        spread_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let spread_top1 = spread_scores.first().map(|(i, _)| i.as_str()).unwrap_or("");

        if baseline_top1 == true_intent { baseline_correct += 1; }
        if spread_top1 == true_intent {
            spreading_correct += 1;
            if baseline_top1 != true_intent {
                improvements.push(format!(
                    "  \"{}\" → {} (baseline said: {})",
                    if ex.text.len() > 60 { &ex.text[..60] } else { &ex.text },
                    true_intent, baseline_top1
                ));
            }
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("RESULTS (intent routing on {} queries):", total);
    println!("  Baseline (direct match):     {}/{} ({:.1}%)",
        baseline_correct, total, baseline_correct as f64 / total as f64 * 100.0);
    println!("  Spreading activation:        {}/{} ({:.1}%)",
        spreading_correct, total, spreading_correct as f64 / total as f64 * 100.0);

    let diff = spreading_correct as i64 - baseline_correct as i64;
    if diff > 0 {
        println!("  Improvement: +{} queries ({:.1}%)",
            diff, diff as f64 / total as f64 * 100.0);
        println!();
        println!("  Examples where spreading found the right answer:");
        for imp in improvements.iter().take(10) {
            println!("{}", imp);
        }
    } else if diff == 0 {
        println!("  No difference — spreading didn't help for intent routing.");
    } else {
        println!("  Spreading was WORSE by {} queries.", -diff);
    }
    println!();
}

// ============================================================================
// PART 4: Circular Convolution Experiment (brain_v25 hypothesis)
// ============================================================================

fn circular_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    assert_eq!(n, b.len());
    let mut result = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            result[(i + j) % n] += a[i] * b[j];
        }
    }
    result
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

fn run_convolution_experiment() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 3: Circular Convolution of PMI Vectors          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Testing brain_v25 hypothesis: convolving PMI vectors creates");
    println!("compositional representations that distinguish intent pairs.");
    println!();

    // Load Bitext
    let path = "tests/data/benchmarks/bitext_all.json";
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => { println!("  Bitext data not found, skipping."); return; }
    };
    let examples: Vec<Example> = match serde_json::from_str(&data) {
        Ok(e) => e,
        Err(_) => { println!("  Parse failed, skipping."); return; }
    };

    // Build vocabulary and co-occurrence matrix
    let docs: Vec<Vec<String>> = examples.iter().map(|e| tokenize(&e.text)).collect();
    let mut vocab: BTreeMap<String, usize> = BTreeMap::new();
    for doc in &docs {
        for term in doc {
            let len = vocab.len();
            vocab.entry(term.clone()).or_insert(len);
        }
    }

    // Limit vocab to top-200 by DF for tractable convolution
    let mut df: HashMap<String, usize> = HashMap::new();
    for doc in &docs {
        let unique: HashSet<&String> = doc.iter().collect();
        for t in unique { *df.entry(t.clone()).or_insert(0) += 1; }
    }
    let mut by_df: Vec<(String, usize)> = df.into_iter()
        .filter(|(_, c)| *c >= 5 && *c < docs.len() / 2) // not too rare, not too common
        .collect();
    by_df.sort_by(|a, b| b.1.cmp(&a.1));
    by_df.truncate(200);

    let dim = by_df.len();
    if dim < 20 {
        println!("  Not enough terms for convolution experiment.");
        return;
    }

    let term_to_idx: HashMap<String, usize> = by_df.iter().enumerate()
        .map(|(i, (t, _))| (t.clone(), i))
        .collect();
    let idx_to_term: Vec<String> = by_df.iter().map(|(t, _)| t.clone()).collect();

    println!("Vocabulary: {} discriminating terms (dim={})", dim, dim);

    // Build PMI vectors: each term's row in the co-occurrence matrix, PMI-weighted
    let n = docs.len() as f64;
    let mut term_df: HashMap<String, f64> = HashMap::new();
    for doc in &docs {
        let unique: HashSet<&String> = doc.iter().collect();
        for t in unique {
            if term_to_idx.contains_key(t.as_str()) {
                *term_df.entry(t.clone()).or_insert(0.0) += 1.0;
            }
        }
    }

    // PMI vectors
    let mut pmi_vectors: HashMap<String, Vec<f64>> = HashMap::new();
    let mut cooccur: HashMap<(usize, usize), f64> = HashMap::new();

    for doc in &docs {
        let unique: Vec<&String> = doc.iter()
            .filter(|t| term_to_idx.contains_key(t.as_str()))
            .collect::<HashSet<_>>().into_iter().collect();
        for i in 0..unique.len() {
            for j in (i+1)..unique.len() {
                let a = term_to_idx[unique[i]];
                let b = term_to_idx[unique[j]];
                let key = if a < b { (a, b) } else { (b, a) };
                *cooccur.entry(key).or_insert(0.0) += 1.0;
            }
        }
    }

    for (term, &idx) in &term_to_idx {
        let mut vec = vec![0.0; dim];
        let p_a = term_df.get(term).copied().unwrap_or(1.0) / n;
        for (other_term, &other_idx) in &term_to_idx {
            if idx == other_idx { continue; }
            let key = if idx < other_idx { (idx, other_idx) } else { (other_idx, idx) };
            let count = cooccur.get(&key).copied().unwrap_or(0.0);
            if count > 0.0 {
                let p_b = term_df.get(other_term).copied().unwrap_or(1.0) / n;
                let p_ab = count / n;
                let pmi = (p_ab / (p_a * p_b)).ln();
                vec[other_idx] = pmi.max(0.0); // PPMI
            }
        }
        pmi_vectors.insert(term.clone(), vec);
    }

    println!("Built PMI vectors for {} terms", pmi_vectors.len());
    println!();

    // Test: Do convolved PMI vectors distinguish intent pairs?
    // "cancel" ⊛ "order" should be different from "track" ⊛ "order"
    // But "cancel" ⊛ "order" should be similar to "cancel" ⊛ "subscription"
    let test_pairs = vec![
        // (a1, b1, a2, b2, should_be_similar)
        ("cancel", "order", "track", "order", false),      // different action, same object
        ("cancel", "order", "cancel", "subscription", true), // same action, different object
        ("track", "order", "check", "order", true),         // similar action
        ("cancel", "order", "cancel", "account", true),     // same action
        ("change", "address", "change", "password", true),  // same action
        ("change", "address", "track", "order", false),     // different everything
        ("refund", "order", "cancel", "order", true),       // related actions
        ("delete", "account", "cancel", "account", true),   // synonym actions
    ];

    println!("Compositional similarity test (circular convolution):");
    println!("  If convolution captures compositionality, same-action pairs");
    println!("  should be MORE similar than different-action pairs.");
    println!();

    let mut correct_comparisons = 0;
    let mut total_comparisons = 0;

    for (a1, b1, a2, b2, should_similar) in &test_pairs {
        let va1 = match pmi_vectors.get(*a1) { Some(v) => v, None => { continue; } };
        let vb1 = match pmi_vectors.get(*b1) { Some(v) => v, None => { continue; } };
        let va2 = match pmi_vectors.get(*a2) { Some(v) => v, None => { continue; } };
        let vb2 = match pmi_vectors.get(*b2) { Some(v) => v, None => { continue; } };

        let conv1 = circular_convolve(va1, vb1);
        let conv2 = circular_convolve(va2, vb2);

        let sim = cosine_similarity(&conv1, &conv2);
        let raw_sim = cosine_similarity(va1, va2); // without convolution

        let label = if *should_similar { "SIMILAR" } else { "DIFFERENT" };
        let check = if (*should_similar && sim > 0.3) || (!should_similar && sim < 0.3) {
            correct_comparisons += 1;
            "✓"
        } else {
            "✗"
        };
        total_comparisons += 1;

        println!("  {} \"{} {}\" vs \"{} {}\" (expect {})",
            check, a1, b1, a2, b2, label);
        println!("    Convolved similarity: {:.3}  |  Raw PMI similarity: {:.3}",
            sim, raw_sim);
    }

    println!();
    println!("Convolution accuracy: {}/{} ({:.0}%)",
        correct_comparisons, total_comparisons,
        correct_comparisons as f64 / total_comparisons as f64 * 100.0);
    println!();

    // Bonus: does convolution separate intents better than raw PMI?
    // For each query, build convolved representation and find nearest intent centroid
    println!("Intent separation test:");
    println!("  Building convolved representations for each intent...");

    // Build intent centroids from convolved query representations
    let mut intent_centroids_raw: HashMap<String, Vec<f64>> = HashMap::new();
    let mut intent_centroids_conv: HashMap<String, Vec<f64>> = HashMap::new();
    let mut intent_counts: HashMap<String, usize> = HashMap::new();

    for ex in &examples {
        let terms: Vec<String> = tokenize(&ex.text).into_iter()
            .filter(|t| pmi_vectors.contains_key(t))
            .collect();
        if terms.len() < 2 { continue; }
        let intent = &ex.intents[0];

        // Raw: average PMI vectors
        let mut raw_avg = vec![0.0; dim];
        for t in &terms {
            if let Some(v) = pmi_vectors.get(t) {
                for (i, val) in v.iter().enumerate() { raw_avg[i] += val; }
            }
        }
        let n_terms = terms.len() as f64;
        for v in raw_avg.iter_mut() { *v /= n_terms; }

        // Convolved: pairwise convolution of adjacent terms, averaged
        let mut conv_avg = vec![0.0; dim];
        let mut n_pairs = 0;
        for i in 0..terms.len().saturating_sub(1) {
            if let (Some(va), Some(vb)) = (pmi_vectors.get(&terms[i]), pmi_vectors.get(&terms[i+1])) {
                let conv = circular_convolve(va, vb);
                for (j, val) in conv.iter().enumerate() { conv_avg[j] += val; }
                n_pairs += 1;
            }
        }
        if n_pairs > 0 {
            for v in conv_avg.iter_mut() { *v /= n_pairs as f64; }
        }

        // Accumulate into centroids
        let raw_cent = intent_centroids_raw.entry(intent.clone()).or_insert_with(|| vec![0.0; dim]);
        for (i, v) in raw_avg.iter().enumerate() { raw_cent[i] += v; }

        let conv_cent = intent_centroids_conv.entry(intent.clone()).or_insert_with(|| vec![0.0; dim]);
        for (i, v) in conv_avg.iter().enumerate() { conv_cent[i] += v; }

        *intent_counts.entry(intent.clone()).or_insert(0) += 1;
    }

    // Normalize centroids
    for (intent, count) in &intent_counts {
        if let Some(c) = intent_centroids_raw.get_mut(intent) {
            for v in c.iter_mut() { *v /= *count as f64; }
        }
        if let Some(c) = intent_centroids_conv.get_mut(intent) {
            for v in c.iter_mut() { *v /= *count as f64; }
        }
    }

    // Test: nearest-centroid classification
    let mut raw_correct = 0;
    let mut conv_correct = 0;
    let mut tested = 0;

    for ex in examples.iter().step_by(10) { // sample every 10th for speed
        let terms: Vec<String> = tokenize(&ex.text).into_iter()
            .filter(|t| pmi_vectors.contains_key(t))
            .collect();
        if terms.len() < 2 { continue; }
        let true_intent = &ex.intents[0];

        // Raw representation
        let mut raw_vec = vec![0.0; dim];
        for t in &terms {
            if let Some(v) = pmi_vectors.get(t) {
                for (i, val) in v.iter().enumerate() { raw_vec[i] += val; }
            }
        }

        // Convolved representation
        let mut conv_vec = vec![0.0; dim];
        let mut n_pairs = 0;
        for i in 0..terms.len().saturating_sub(1) {
            if let (Some(va), Some(vb)) = (pmi_vectors.get(&terms[i]), pmi_vectors.get(&terms[i+1])) {
                let conv = circular_convolve(va, vb);
                for (j, val) in conv.iter().enumerate() { conv_vec[j] += val; }
                n_pairs += 1;
            }
        }

        // Find nearest centroid (raw)
        let mut best_raw = ("", -1.0f64);
        for (intent, cent) in &intent_centroids_raw {
            let sim = cosine_similarity(&raw_vec, cent);
            if sim > best_raw.1 { best_raw = (intent, sim); }
        }

        // Find nearest centroid (convolved)
        let mut best_conv = ("", -1.0f64);
        for (intent, cent) in &intent_centroids_conv {
            let sim = cosine_similarity(&conv_vec, cent);
            if sim > best_conv.1 { best_conv = (intent, sim); }
        }

        if best_raw.0 == true_intent { raw_correct += 1; }
        if best_conv.0 == true_intent { conv_correct += 1; }
        tested += 1;
    }

    println!();
    println!("  Nearest-centroid classification (sampled {} queries):", tested);
    println!("    Raw PMI vectors:      {}/{} ({:.1}%)",
        raw_correct, tested, raw_correct as f64 / tested as f64 * 100.0);
    println!("    Convolved PMI vectors: {}/{} ({:.1}%)",
        conv_correct, tested, conv_correct as f64 / tested as f64 * 100.0);

    if conv_correct > raw_correct {
        println!("    CONVOLUTION IMPROVES classification by +{} ({:.1}%)",
            conv_correct - raw_correct,
            (conv_correct - raw_correct) as f64 / tested as f64 * 100.0);
    }
    println!();
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let t = Instant::now();

    run_synthetic_experiment();
    run_bitext_experiment();
    run_convolution_experiment();

    println!("Total time: {:.2}s", t.elapsed().as_secs_f64());
}
