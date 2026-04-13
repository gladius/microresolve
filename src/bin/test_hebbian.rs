//! Standalone Hebbian graph demo — run with:
//!   cargo run --bin test_hebbian

use asv_router::hebbian::saas_test_graph;

fn main() {
    let g = saas_test_graph();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          Hebbian Association Graph — Query Preprocessing     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let cases: &[(&str, &str)] = &[
        // Morphological normalization
        ("canceling my subscription",         "morph"),
        ("the order was cancelled",           "morph"),
        ("merged the pr",                     "morph+abbrev"),
        ("scheduling a msg for the chan",      "morph+abbrev"),
        // Abbreviation normalization
        ("cancel my sub",                     "abbrev"),
        ("list all repos",                    "abbrev"),
        ("close the pr in that repo",         "abbrev"),
        // Synonym expansion
        ("terminate my plan",                 "synonym"),
        ("kill the subscription",             "synonym"),
        ("ping the team",                     "synonym"),
        ("run their card",                    "synonym"),
        ("show me all invoices",              "synonym"),
        ("bill them for the extra usage",     "synonym"),
        ("reimburse the customer",            "synonym"),
        // Combined
        ("terminate my sub",                  "morph+abbrev+synonym"),
        ("canceling my sub",                  "morph+abbrev"),
        ("killing the subs",                  "morph+synonym"),
        ("merged the pr and closed the issue","morph+abbrev×2"),
        // Semantic — should NOT expand
        ("stop sending me emails",            "semantic only"),
        ("at the end of the month",           "semantic only"),
        // Clean — no change
        ("cancel my subscription",            "clean"),
        ("merge the pull request",            "clean"),
    ];

    for (query, label) in cases {
        let r = g.preprocess(query);
        println!("\n  [{label}]");
        println!("  IN:   {}", r.original);
        if r.normalized != r.original.to_lowercase() {
            println!("  NORM: {}", r.normalized);
        }
        if !r.injected.is_empty() {
            println!("  EXP:  {} [+{}]", r.expanded, r.injected.join(", "));
        }
        if !r.semantic_hits.is_empty() {
            let hits: Vec<String> = r.semantic_hits.iter()
                .map(|(s, t, w)| format!("{s}→{t}({w:.2})"))
                .collect();
            println!("  SEM:  {}", hits.join(", "));
        }
        if !r.was_modified {
            println!("  →     (no change)");
        }
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                        Graph Stats                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    use asv_router::hebbian::EdgeKind;
    let mut morph = 0usize;
    let mut abbrev = 0usize;
    let mut synonym = 0usize;
    let mut semantic = 0usize;
    for edges in g.edges.values() {
        for e in edges {
            match e.kind {
                EdgeKind::Morphological => morph += 1,
                EdgeKind::Abbreviation  => abbrev += 1,
                EdgeKind::Synonym       => synonym += 1,
                EdgeKind::Semantic      => semantic += 1,
            }
        }
    }
    println!("  Source terms:  {}", g.edges.len());
    println!("  Morphological: {morph}");
    println!("  Abbreviation:  {abbrev}");
    println!("  Synonym:       {synonym}");
    println!("  Semantic:      {semantic}");
    println!("  Total edges:   {}", morph + abbrev + synonym + semantic);

    println!("\n  Threshold: {}", g.synonym_threshold);
    println!("  (synonyms below this weight → semantic signal only, no expansion)");
}
