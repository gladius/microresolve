//! Tests for oplog pruning and persistence round-trip.

use microresolve::oplog::OPLOG_MAX;
use microresolve::{MicroResolve, MicroResolveConfig};

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::path::PathBuf::from(format!("/tmp/mr_oplog_test_{}_{}", tag, nanos));
    std::fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn oplog_pruned_to_max() {
    let e = MicroResolve::new(MicroResolveConfig::default()).unwrap();
    let h = e.namespace("ns");

    // Each add_intent emits at least 1 op. Add OPLOG_MAX + 10 intents.
    for i in 0..(OPLOG_MAX + 10) {
        h.add_intent(&format!("intent_{}", i), vec![format!("phrase for {}", i)])
            .unwrap();
    }

    let oplog_len = h.with_resolver(|r| r.oplog.len());
    assert!(
        oplog_len <= OPLOG_MAX,
        "oplog grew beyond OPLOG_MAX: {}",
        oplog_len
    );
}

#[test]
fn oplog_persists_and_loads() {
    let dir = tmp_dir("persist");
    let ns_dir = dir.join("testns");

    {
        let e = MicroResolve::new(MicroResolveConfig {
            data_dir: Some(dir.clone()),
            ..Default::default()
        })
        .unwrap();
        let h = e.namespace("testns");
        h.add_intent("foo", vec!["foo bar".to_string()]).unwrap();
        h.add_intent("baz", vec!["baz qux".to_string()]).unwrap();
        h.flush().unwrap();
    }

    // Oplog file should exist.
    assert!(
        ns_dir.join("_oplog.json").exists(),
        "_oplog.json must be written"
    );

    // Re-load and verify oplog is non-empty.
    let e2 = MicroResolve::new(MicroResolveConfig {
        data_dir: Some(dir.clone()),
        ..Default::default()
    })
    .unwrap();
    let h2 = e2.namespace("testns");
    let oplog_len = h2.with_resolver(|r| r.oplog.len());
    assert!(oplog_len > 0, "oplog should be non-empty after reload");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn oplog_missing_file_ok() {
    // A namespace dir with no _oplog.json should load fine with empty oplog.
    let dir = tmp_dir("missing_oplog");
    let ns_dir = dir.join("testns");
    std::fs::create_dir_all(&ns_dir).unwrap();
    // Write a minimal _ns.json so the namespace loads
    std::fs::write(
        ns_dir.join("_ns.json"),
        r#"{"name":"test","description":""}"#,
    )
    .unwrap();

    let e = MicroResolve::new(MicroResolveConfig {
        data_dir: Some(dir.clone()),
        ..Default::default()
    })
    .unwrap();
    let h = e.namespace("testns");
    let oplog_len = h.with_resolver(|r| r.oplog.len());
    assert_eq!(oplog_len, 0, "missing _oplog.json should yield empty oplog");

    std::fs::remove_dir_all(&dir).ok();
}
