//! Audit chain verification.
//!
//! Walks every per-key chain in `{data_dir}/_audit/`, recomputes each
//! entry's `entry_hash`, and reports any divergence. Used by:
//! - `POST /api/audit/verify` HTTP endpoint
//! - `microresolve-studio verify-log` CLI subcommand

use crate::audit_log::{compute_entry_hash, AuditEntry};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, serde::Serialize)]
pub struct ChainReport {
    pub kid: String,
    pub entries: u64,
    pub head_hash: String,
    pub ok: bool,
    /// Description of the first break, if any.
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VerifyReport {
    pub ok: bool,
    pub total_entries: u64,
    pub chains_verified: usize,
    pub chains_with_errors: usize,
    pub chains: Vec<ChainReport>,
}

pub fn verify_all(data_dir: Option<&str>) -> VerifyReport {
    let Some(dir) = data_dir.map(|d| PathBuf::from(d).join("_audit")) else {
        return VerifyReport {
            ok: true,
            total_entries: 0,
            chains_verified: 0,
            chains_with_errors: 0,
            chains: vec![],
        };
    };
    let mut chains: Vec<ChainReport> = Vec::new();
    let mut total = 0u64;
    let mut errors = 0usize;
    let Ok(entries) = fs::read_dir(&dir) else {
        return VerifyReport {
            ok: true,
            total_entries: 0,
            chains_verified: 0,
            chains_with_errors: 0,
            chains: vec![],
        };
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map(|e| e == "log").unwrap_or(false) {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                let report = verify_chain(stem, &path);
                if !report.ok {
                    errors += 1;
                }
                total += report.entries;
                chains.push(report);
            }
        }
    }
    chains.sort_by(|a, b| a.kid.cmp(&b.kid));
    VerifyReport {
        ok: errors == 0,
        total_entries: total,
        chains_verified: chains.len(),
        chains_with_errors: errors,
        chains,
    }
}

pub fn verify_chain(kid: &str, path: &Path) -> ChainReport {
    let mut report = ChainReport {
        kid: kid.to_string(),
        entries: 0,
        head_hash: String::new(),
        ok: true,
        error: None,
    };
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            report.ok = false;
            report.error = Some(format!("open: {}", e));
            return report;
        }
    };
    let mut prev_hash = String::new();
    let mut expected_id = 0u64;
    loop {
        let mut len_buf = [0u8; 4];
        match file.read_exact(&mut len_buf) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                report.ok = false;
                report.error = Some(format!("read len at entry {}: {}", expected_id, e));
                return report;
            }
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut payload = vec![0u8; len];
        if let Err(e) = file.read_exact(&mut payload) {
            report.ok = false;
            report.error = Some(format!("truncated at entry {}: {}", expected_id, e));
            return report;
        }
        let entry: AuditEntry = match serde_json::from_slice(&payload) {
            Ok(e) => e,
            Err(e) => {
                report.ok = false;
                report.error = Some(format!("parse at entry {}: {}", expected_id, e));
                return report;
            }
        };
        if entry.id != expected_id {
            report.ok = false;
            report.error = Some(format!(
                "id mismatch: expected {}, got {}",
                expected_id, entry.id
            ));
            return report;
        }
        if entry.prev_hash != prev_hash {
            report.ok = false;
            report.error = Some(format!(
                "prev_hash mismatch at entry {}: expected {}, got {}",
                entry.id,
                short(&prev_hash),
                short(&entry.prev_hash)
            ));
            return report;
        }
        let canonical_payload = serde_json::to_vec(&entry.payload).unwrap_or_default();
        let recomputed = compute_entry_hash(
            &entry.prev_hash,
            entry.id,
            entry.ts_ms,
            &entry.kid,
            &entry.ns,
            &entry.event_type,
            &canonical_payload,
        );
        if recomputed != entry.entry_hash {
            report.ok = false;
            report.error = Some(format!(
                "hash mismatch at entry {}: stored {}, recomputed {}",
                entry.id,
                short(&entry.entry_hash),
                short(&recomputed)
            ));
            return report;
        }
        prev_hash = entry.entry_hash.clone();
        expected_id = entry.id + 1;
        report.entries += 1;
        report.head_hash = entry.entry_hash;
    }
    report
}

fn short(h: &str) -> String {
    if h.len() <= 12 {
        h.to_string()
    } else {
        format!("{}…", &h[..12])
    }
}
