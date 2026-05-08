//! Audit chain export — JSONL to stdout for handoff to SIEM / auditor.
//!
//! Walks every per-key chain in `{data_dir}/_audit/`, applies optional
//! filters (key / since / event_prefix / namespace), writes entries as
//! length-unprefixed JSON-per-line. Output is suitable for:
//!
//! ```bash
//! microresolve-studio export-log --since 30d > audit-q4.jsonl
//! microresolve-studio export-log --key prod-east-python | curl -X POST splunk.example.com/...
//! microresolve-studio export-log --event-prefix intent. > rule-changes.jsonl
//! ```
//!
//! Each line is one valid JSON object — the `AuditEntry` shape with
//! `id`, `ts_ms`, `kid`, `ns`, `event_type`, `payload`, `prev_hash`,
//! `entry_hash`. Auditors can pipe this into their tooling without
//! parsing a custom format.

use crate::audit_log::AuditEntry;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;

/// Filter options for `export_chains`. All `None` = export everything.
#[derive(Debug, Default)]
pub struct ExportFilter {
    /// If `Some(kid)`, only export that key's chain.
    pub key: Option<String>,
    /// If `Some(ms)`, only entries with `ts_ms >= since_ms`.
    pub since_ms: Option<u64>,
    /// If `Some(p)`, only entries whose `event_type` starts with `p`.
    pub event_prefix: Option<String>,
    /// If `Some(ns)`, only entries affecting that namespace.
    pub namespace: Option<String>,
}

/// Walk all per-key chains and write filtered entries as JSONL to
/// stdout. Returns `(entries_written, chains_visited)`.
pub fn export_chains(data_dir: &str, filter: &ExportFilter) -> io::Result<(u64, usize)> {
    let dir = PathBuf::from(data_dir).join("_audit");
    let mut written = 0u64;
    let mut chains = 0usize;
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Ok((0, 0)),
    };

    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "log").unwrap_or(false))
        .collect();
    paths.sort();

    for path in paths {
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };
        if let Some(ref k) = filter.key {
            if stem != k {
                continue;
            }
        }
        chains += 1;

        let mut file = fs::File::open(&path)?;
        loop {
            let mut len_buf = [0u8; 4];
            match file.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            if file.read_exact(&mut payload).is_err() {
                break;
            }
            let entry: AuditEntry = match serde_json::from_slice(&payload) {
                Ok(e) => e,
                Err(_) => continue,
            };
            // Apply filters.
            if let Some(min_ts) = filter.since_ms {
                if entry.ts_ms < min_ts {
                    continue;
                }
            }
            if let Some(ref p) = filter.event_prefix {
                if !entry.event_type.starts_with(p) {
                    continue;
                }
            }
            if let Some(ref ns) = filter.namespace {
                if &entry.ns != ns {
                    continue;
                }
            }
            // Write one JSON object per line.
            if let Ok(line) = serde_json::to_string(&entry) {
                writeln!(out, "{}", line)?;
                written += 1;
            }
        }
    }
    Ok((written, chains))
}

/// Parse a human-readable duration like "7d", "24h", "30m", "90s" into
/// milliseconds. Returns `Err` for unknown suffixes.
pub fn parse_duration_ms(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty duration".into());
    }
    let (num_str, suffix) = s.split_at(s.len() - 1);
    let multiplier = match suffix {
        "s" => 1_000u64,
        "m" => 60 * 1_000,
        "h" => 60 * 60 * 1_000,
        "d" => 24 * 60 * 60 * 1_000,
        _ => {
            return Err(format!(
                "unknown duration suffix '{}' — use s | m | h | d",
                suffix
            ))
        }
    };
    let num: u64 = num_str
        .parse()
        .map_err(|_| format!("invalid duration number '{}'", num_str))?;
    Ok(num.saturating_mul(multiplier))
}
