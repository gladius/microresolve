//! Tamper-evident audit log — per-key hash-chained event store.
//!
//! Lineage: this module is the natural extension of the v0.2.0 compliance
//! pack work (`eu-ai-act-prohibited`, `hipaa-triage`). EU AI Act Art. 13
//! and HIPAA §164.312(b) both require tamper-evident decision logs for
//! high-risk AI systems; this module lands the infrastructure those packs
//! need to be audit-defensible end-to-end.
//!
//! # Architecture
//!
//! Per-key linear hash chains. Each API key writes its own append-only
//! file under `{data_dir}/_audit/{key_name}.log`. The key's `name` field
//! (the slug part of `mr_<name>_<hex>`) is used as the chain identifier;
//! the bearer token itself never appears in audit data.
//!
//! Why per-key, not per-namespace:
//! - A single key may write to many namespaces (one agent, multiple
//!   resolvers); attribution stays with the key
//! - Embedded library mode (v0.3+) — each library instance has its own
//!   key, naturally produces an isolated chain
//! - Multi-tenant — tenant isolation falls out for free
//! - Workflow correlation — one key's chain IS the agent's interaction
//!   history, suitable for next-action prediction (v0.3+)
//!
//! # Hash chain
//!
//! Each entry stores `prev_hash` and `entry_hash` (32 bytes each, hex).
//! `entry_hash = sha256(prev_hash_bytes || id_le || ts_le || kid || ns ||
//!                      event_type || payload_json)`.
//!
//! Verification: walk the file, recompute each hash, flag any divergence.
//! See `audit_verify.rs` for the verifier.
//!
//! # Wire format (per entry)
//! ```text
//! [u32 LE: payload_len][json bytes: payload_len]
//! ```
//! No tombstones — audit chain is strictly append-only. Tampering is
//! detected by hash mismatch, not by a delete bit.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use sha2::{Digest, Sha256};

/// Audit policy. Two states in v0.2.2: `Off` (no logging) or `Default`
/// (mutations + resolve metadata, query hashed not stored). Set once at
/// server boot via `[audit].mode` in config.toml or `MICRORESOLVE_AUDIT_MODE`
/// env var.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditMode {
    /// No audit logging.
    Off,
    /// Mutations + resolve metadata (intent ids, scores, query hash).
    /// Raw query content is NOT stored. Default for new deployments.
    #[default]
    Default,
}

impl AuditMode {
    /// Should this mode log anything (mutations + resolves)?
    pub fn enabled(&self) -> bool {
        matches!(self, Self::Default)
    }
}

/// One audit chain entry. Persisted as length-prefixed JSON.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AuditEntry {
    /// Per-chain monotonic id, starts at 0 for each key.
    pub id: u64,
    /// Wall-clock millis at write time.
    pub ts_ms: u64,
    /// API key name — the chain identifier. Slug-safe.
    pub kid: String,
    /// Namespace this event affected. Empty string for global events
    /// (e.g. `key.create`).
    pub ns: String,
    /// Event taxonomy: "resolve", "intent.add", "intent.update",
    /// "intent.delete", "namespace.create", "namespace.update",
    /// "namespace.delete", "threshold.change", "voting_tokens.change",
    /// "learn.apply", "pack.install", "key.create", "key.revoke".
    pub event_type: String,
    /// Event-specific structured data. Resolve events store
    /// `{ns, query_hash, query?, intents, threshold_applied}`.
    /// Mutation events store the relevant fields for that mutation.
    pub payload: serde_json::Value,
    /// Hex-encoded sha256 of the previous entry's `entry_hash`. Empty
    /// string for the genesis entry (id=0).
    pub prev_hash: String,
    /// Hex-encoded sha256 of (prev_hash || id_le || ts_le || kid || ns
    ///   || event_type || canonical_payload_json).
    pub entry_hash: String,
}

/// One per-key chain.
struct KeyChain {
    file: Option<File>,
    next_id: u64,
    head_hash: String,
    /// Total entries (for diagnostics / fast count without re-scan).
    count: u64,
}

impl KeyChain {
    fn in_memory() -> Self {
        Self {
            file: None,
            next_id: 0,
            head_hash: String::new(),
            count: 0,
        }
    }

    fn open(path: &Path) -> std::io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(path)?;

        // Re-open a separate read handle so we can scan from offset 0.
        let mut scan = OpenOptions::new().read(true).open(path)?;
        let mut next_id = 0u64;
        let mut head_hash = String::new();
        let mut count = 0u64;
        loop {
            let mut len_buf = [0u8; 4];
            match scan.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut payload = vec![0u8; len];
            if scan.read_exact(&mut payload).is_err() {
                break;
            }
            if let Ok(entry) = serde_json::from_slice::<AuditEntry>(&payload) {
                next_id = entry.id + 1;
                head_hash = entry.entry_hash.clone();
                count += 1;
            }
        }

        // The append handle is positioned at end-of-file because of
        // `append(true)`; nothing else needed.
        let _ = file.flush();
        Ok(Self {
            file: Some(file),
            next_id,
            head_hash,
            count,
        })
    }
}

/// Audit log store — owns one chain per API key.
///
/// # Concurrency
///
/// Internal locking — callers do NOT need an outer Mutex. Different
/// keys' chains write in parallel; only same-key writes serialize on
/// the per-chain mutex. This is the load-bearing scale property:
/// 1,000 workloads writing to 1,000 different chains see no contention.
pub struct AuditLog {
    /// Root directory `{data_dir}/_audit/`. None = in-memory mode (tests).
    dir: Option<PathBuf>,
    /// One mutex per kid. Outer RwLock guards the map (insert on first
    /// write per kid); inner Mutex guards the chain itself.
    chains: RwLock<HashMap<String, Arc<Mutex<KeyChain>>>>,
    /// Server-wide audit mode. Immutable after construction (set at
    /// boot from config). Two-state in v0.2.2: `Off` or `Default`.
    mode: AuditMode,
}

impl AuditLog {
    pub fn new(data_dir: Option<&str>, mode: AuditMode) -> Self {
        let dir = data_dir.map(|d| {
            let p = PathBuf::from(d).join("_audit");
            let _ = fs::create_dir_all(&p);
            p
        });
        let s = Self {
            dir,
            chains: RwLock::new(HashMap::new()),
            mode,
        };
        s.scan_existing();
        s
    }

    /// Server-wide audit mode.
    pub fn mode(&self) -> AuditMode {
        self.mode
    }

    fn scan_existing(&self) {
        let Some(dir) = self.dir.clone() else {
            return;
        };
        let Ok(entries) = fs::read_dir(&dir) else {
            return;
        };
        let mut map = self.chains.write().unwrap();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "log").unwrap_or(false) {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    match KeyChain::open(&path) {
                        Ok(chain) => {
                            eprintln!(
                                "[audit_log] loaded kid={} entries={} head={}",
                                stem,
                                chain.count,
                                short(&chain.head_hash)
                            );
                            map.insert(stem.to_string(), Arc::new(Mutex::new(chain)));
                        }
                        Err(e) => eprintln!("[audit_log] error loading {}: {}", stem, e),
                    }
                }
            }
        }
    }

    /// Look up the chain for `kid`, creating it if it doesn't yet exist.
    /// Returns an `Arc<Mutex<KeyChain>>` so the caller can drop the
    /// outer `RwLock` guard before locking the chain itself — this is
    /// what lets different keys' chains write in parallel.
    fn chain_for(&self, kid: &str) -> Arc<Mutex<KeyChain>> {
        // Fast path — read lock, look up.
        {
            let map = self.chains.read().unwrap();
            if let Some(c) = map.get(kid) {
                return c.clone();
            }
        }
        // Slow path — upgrade to write lock and insert (with double-check
        // in case a concurrent writer beat us here).
        let mut map = self.chains.write().unwrap();
        if let Some(c) = map.get(kid) {
            return c.clone();
        }
        let chain = match self.dir.as_ref().map(|d| d.join(format!("{}.log", kid))) {
            Some(path) => KeyChain::open(&path).unwrap_or_else(|e| {
                eprintln!("[audit_log] cannot open {}: {}", path.display(), e);
                KeyChain::in_memory()
            }),
            None => KeyChain::in_memory(),
        };
        let arc = Arc::new(Mutex::new(chain));
        map.insert(kid.to_string(), arc.clone());
        arc
    }

    /// Append an entry. Caller has already decided that this event should
    /// be logged given the resolved mode.
    ///
    /// Concurrency: takes `&self` (interior locking). Different kids
    /// proceed in parallel; same-kid writes serialize on the per-chain
    /// mutex.
    pub fn record(&self, kid: &str, ns: &str, event_type: &str, payload: serde_json::Value) -> u64 {
        let chain_arc = self.chain_for(kid);
        let mut chain = chain_arc.lock().unwrap();
        let id = chain.next_id;
        let ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let prev_hash = chain.head_hash.clone();
        let canonical_payload = serde_json::to_vec(&payload).unwrap_or_default();
        let entry_hash = compute_entry_hash(
            &prev_hash,
            id,
            ts_ms,
            kid,
            ns,
            event_type,
            &canonical_payload,
        );

        let entry = AuditEntry {
            id,
            ts_ms,
            kid: kid.to_string(),
            ns: ns.to_string(),
            event_type: event_type.to_string(),
            payload,
            prev_hash,
            entry_hash: entry_hash.clone(),
        };

        let serialized = serde_json::to_vec(&entry).unwrap_or_default();
        if let Some(ref mut file) = chain.file {
            let len = (serialized.len() as u32).to_le_bytes();
            let _ = file.write_all(&len);
            let _ = file.write_all(&serialized);
            // OS-buffered write — durability is best-effort. For
            // strict-durability deployments, periodic sync_all() lands
            // in v0.2.3 (configurable interval).
        }
        chain.next_id = id + 1;
        chain.head_hash = entry_hash;
        chain.count += 1;
        id
    }

    /// Snapshot: (kid, head_hash, count) for every known chain. Used by
    /// the `/api/audit/heads` endpoint and the verify CLI.
    pub fn heads(&self) -> Vec<ChainHead> {
        let map = self.chains.read().unwrap();
        let mut out: Vec<ChainHead> = map
            .iter()
            .map(|(kid, c)| {
                let chain = c.lock().unwrap();
                ChainHead {
                    kid: kid.clone(),
                    head_hash: chain.head_hash.clone(),
                    count: chain.count,
                }
            })
            .collect();
        out.sort_by(|a, b| a.kid.cmp(&b.kid));
        out
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ChainHead {
    pub kid: String,
    pub head_hash: String,
    pub count: u64,
}

/// Compute `entry_hash`. Stable across processes — verifier must match
/// exactly.
pub fn compute_entry_hash(
    prev_hash: &str,
    id: u64,
    ts_ms: u64,
    kid: &str,
    ns: &str,
    event_type: &str,
    canonical_payload: &[u8],
) -> String {
    let mut h = Sha256::new();
    // Decode prev_hash from hex if non-empty; for genesis we feed 32 zero
    // bytes so the first chain link has a defined preimage.
    let prev_bytes = if prev_hash.is_empty() {
        [0u8; 32].to_vec()
    } else {
        hex::decode(prev_hash).unwrap_or_else(|_| vec![0u8; 32])
    };
    h.update(&prev_bytes);
    h.update(id.to_le_bytes());
    h.update(ts_ms.to_le_bytes());
    h.update(kid.as_bytes());
    h.update([0u8]); // separator
    h.update(ns.as_bytes());
    h.update([0u8]);
    h.update(event_type.as_bytes());
    h.update([0u8]);
    h.update(canonical_payload);
    hex::encode(h.finalize())
}

/// Stable hash of a query string for `query_hash` in resolve entries.
/// Allows regulators/operators to confirm a specific query was processed
/// without storing the raw query content.
pub fn hash_query(query: &str) -> String {
    let mut h = Sha256::new();
    h.update(query.as_bytes());
    hex::encode(h.finalize())
}

fn short(h: &str) -> String {
    if h.len() <= 12 {
        h.to_string()
    } else {
        format!("{}…", &h[..12])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_records_and_links() {
        let log = AuditLog::new(None, AuditMode::Default);
        let id0 = log.record(
            "test-key",
            "ns1",
            "intent.add",
            serde_json::json!({"intent_id": "foo"}),
        );
        let id1 = log.record(
            "test-key",
            "ns1",
            "intent.add",
            serde_json::json!({"intent_id": "bar"}),
        );
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        let heads = log.heads();
        assert_eq!(heads.len(), 1);
        assert_eq!(heads[0].count, 2);
        assert!(!heads[0].head_hash.is_empty());
    }

    #[test]
    fn separate_chains_per_key() {
        let log = AuditLog::new(None, AuditMode::Default);
        log.record("key-a", "ns", "x", serde_json::json!({}));
        log.record("key-b", "ns", "x", serde_json::json!({}));
        log.record("key-a", "ns", "x", serde_json::json!({}));
        let heads = log.heads();
        assert_eq!(heads.len(), 2);
        let a = heads.iter().find(|h| h.kid == "key-a").unwrap();
        let b = heads.iter().find(|h| h.kid == "key-b").unwrap();
        assert_eq!(a.count, 2);
        assert_eq!(b.count, 1);
        assert_ne!(a.head_hash, b.head_hash);
    }

    #[test]
    fn modes_simple_two_state() {
        assert!(!AuditMode::Off.enabled());
        assert!(AuditMode::Default.enabled());
    }

    #[test]
    fn entry_hashes_change_with_payload() {
        let h1 = compute_entry_hash("", 0, 1000, "k", "n", "x", b"{\"a\":1}");
        let h2 = compute_entry_hash("", 0, 1000, "k", "n", "x", b"{\"a\":2}");
        assert_ne!(h1, h2);
    }
}
