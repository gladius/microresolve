//! Verify the empty-subscribe = auto-subscribe-all behaviour for v0.1.5+.
//!
//! Prerequisites: a Studio running on http://localhost:3001 with several
//! namespaces created. Run:
//!   cargo run --release --example hybrid_subscribe --features connect

use microresolve::{MicroResolve, MicroResolveConfig, ServerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("─── auto-subscribe (empty list) ─────────────────────────────");
    let auto = MicroResolve::new(MicroResolveConfig {
        server: Some(ServerConfig {
            url: "http://localhost:3001".into(),
            api_key: None,
            subscribe: vec![], // empty = auto-subscribe to all
            tick_interval_secs: 30,
            log_buffer_max: 500,
        }),
        ..Default::default()
    })?;
    let mut all = auto.namespaces();
    all.sort();
    println!("  pulled namespaces: {:?}", all);

    println!("\n─── explicit subscribe (allow-list) ─────────────────────────");
    let explicit = MicroResolve::new(MicroResolveConfig {
        server: Some(ServerConfig {
            url: "http://localhost:3001".into(),
            api_key: None,
            subscribe: vec!["alpha".into()],
            tick_interval_secs: 30,
            log_buffer_max: 500,
        }),
        ..Default::default()
    })?;
    let mut explicit_ns = explicit.namespaces();
    explicit_ns.sort();
    println!("  pulled namespaces: {:?}", explicit_ns);

    Ok(())
}
