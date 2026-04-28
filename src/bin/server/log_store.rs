//! Binary append log — single source of truth for all queries and review.
//!
//! Replaces both the JSONL query log and the in-memory review queue.
//! One file per app_id: `{data_dir}/_logs/{app_id}.bin`
//! Underscore prefix is the convention for non-namespace directories
//! (boot loader skips them when scanning for namespaces).
//!
//! ## Record format (per entry)
//! ```text
//! [u8: alive][u32 LE: payload_len][json bytes: payload_len]
//! ```
//! - `alive = 1` → live record
//! - `alive = 0` → resolved/dismissed (tombstone, written in-place by seek)
//!
//! ## Startup
//! Full file scan rebuilds an in-memory `Vec<LogMeta>` index (~50 bytes each).
//! 1M entries ≈ 50 MB in memory, ms to scan.
//!
//! ## Thread safety
//! Caller wraps in `Mutex<LogStore>`.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

// ─── Public types ────────────────────────────────────────────────────────────

/// LLM review outcome attached to a log entry (in-memory only, not persisted).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReviewStatus {
    /// "pending" | "processing" | "done" | "escalated"
    pub status: String,
    pub llm_reviewed: bool,
    pub llm_model: Option<String>,
    pub llm_result: Option<serde_json::Value>,
    pub applied: bool,
    pub phrases_added: usize,
    pub version_before: u64,
    pub version_after: Option<u64>,
    pub summary: Option<String>,
}

impl ReviewStatus {
    pub fn pending() -> Self {
        Self {
            status: "pending".to_string(),
            llm_reviewed: false,
            llm_model: None,
            llm_result: None,
            applied: false,
            phrases_added: 0,
            version_before: 0,
            version_after: None,
            summary: None,
        }
    }
}

/// A single routed query. Replaces JSONL log entry + ReviewItem.
///
/// Historical note: used to carry a `flag: Option<String>` ("miss" /
/// "low_confidence" / "false_positive"). Removed 2026-04-24 because those
/// labels were misleading — they suggested the system could self-assess
/// routing correctness, which it can't. The worker picks up every
/// unreviewed record and lets the LLM judge decide.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogRecord {
    pub id: u64,
    pub query: String,
    pub app_id: String,
    pub detected_intents: Vec<String>,
    /// "high", "medium", "low", "none"
    pub confidence: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub timestamp_ms: u64,
    pub router_version: u64,
    /// "local" or "connected"
    pub source: String,
}

pub struct LogQuery {
    pub app_id: Option<String>,
    /// None = all; Some(false) = unresolved only; Some(true) = resolved only
    pub resolved: Option<bool>,
    pub since_ms: Option<u64>,
    pub limit: usize,
    pub offset: usize,
}

impl Default for LogQuery {
    fn default() -> Self {
        Self {
            app_id: None,
            resolved: Some(false),
            since_ms: None,
            limit: 50,
            offset: 0,
        }
    }
}

pub struct LogQueryResult {
    pub total: usize,  // matching records before offset/limit
    pub records: Vec<LogRecord>,
}

// ─── Internal ────────────────────────────────────────────────────────────────

/// Compact index entry — kept in memory, avoids reading file for filtering.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LogMeta {
    offset: u64,
    payload_len: u32,
    id: u64,
    timestamp_ms: u64,
    confidence: String,
    alive: bool,
    /// Payload cached when no backing file (in-memory mode).
    cached: Option<Vec<u8>>,
}

struct AppLog {
    file: Option<File>,
    size: u64,
    index: Vec<LogMeta>,
    next_id: u64,
    /// In-memory LLM review status per record id — not persisted.
    review_status: HashMap<u64, ReviewStatus>,
}

impl AppLog {
    fn in_memory() -> Self {
        Self { file: None, size: 0, index: Vec::new(), next_id: 0, review_status: HashMap::new() }
    }

    fn open(path: &PathBuf) -> std::io::Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
        let mut index = Vec::new();
        let mut offset = 0u64;
        let mut next_id = 0u64;

        loop {
            let mut header = [0u8; 5];
            match file.read_exact(&mut header) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let alive = header[0] == 1;
            let payload_len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]);
            let mut payload = vec![0u8; payload_len as usize];
            if let Err(e) = file.read_exact(&mut payload) {
                eprintln!("[log_store] truncated record at offset {}: {}", offset, e);
                break;
            }

            if let Ok(record) = serde_json::from_slice::<LogRecord>(&payload) {
                next_id = next_id.max(record.id + 1);
                index.push(LogMeta {
                    offset,
                    payload_len,
                    id: record.id,
                    timestamp_ms: record.timestamp_ms,
                    confidence: record.confidence,
                    alive,
                    cached: None,
                });
            }
            offset += 5 + payload_len as u64;
        }

        Ok(Self { file: Some(file), size: offset, index, next_id, review_status: HashMap::new() })
    }
}

// ─── LogStore ────────────────────────────────────────────────────────────────

pub struct LogStore {
    data_dir: Option<PathBuf>,
    apps: HashMap<String, AppLog>,
}

impl LogStore {
    /// Create store. If `data_dir` is Some, persists to `{data_dir}/_logs/`.
    /// Existing log files are scanned and indexed on startup.
    pub fn new(data_dir: Option<&str>) -> Self {
        let data_dir = data_dir.map(|d| {
            let p = PathBuf::from(d).join("_logs");
            let _ = fs::create_dir_all(&p);
            p
        });
        let mut store = Self { data_dir, apps: HashMap::new() };
        store.scan_existing();
        store
    }

    fn scan_existing(&mut self) {
        let dir = match self.data_dir.clone() {
            Some(d) => d,
            None => return,
        };
        let Ok(entries) = fs::read_dir(&dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "bin").unwrap_or(false) {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    match AppLog::open(&path) {
                        Ok(app_log) => {
                            let count = app_log.index.len();
                            eprintln!("[log_store] loaded app={} records={}", stem, count);
                            self.apps.insert(stem.to_string(), app_log);
                        }
                        Err(e) => eprintln!("[log_store] error loading {}: {}", stem, e),
                    }
                }
            }
        }
    }

    fn get_or_create(&mut self, app_id: &str) -> &mut AppLog {
        if self.apps.contains_key(app_id) {
            return self.apps.get_mut(app_id).unwrap();
        }
        let app_log = match self.data_dir.as_ref().map(|d| d.join(format!("{}.bin", app_id))) {
            Some(path) => AppLog::open(&path).unwrap_or_else(|e| {
                eprintln!("[log_store] cannot open {}: {}", path.display(), e);
                AppLog::in_memory()
            }),
            None => AppLog::in_memory(),
        };
        self.apps.insert(app_id.to_string(), app_log);
        self.apps.get_mut(app_id).unwrap()
    }

    /// Append a record. Returns the assigned id.
    pub fn append(&mut self, mut record: LogRecord) -> u64 {
        let app_id = record.app_id.clone();
        let al = self.get_or_create(&app_id);

        record.id = al.next_id;
        al.next_id += 1;

        let payload = serde_json::to_vec(&record).unwrap_or_default();
        let payload_len = payload.len() as u32;
        let offset = al.size;

        if let Some(ref mut file) = al.file {
            let mut header = [1u8; 5]; // alive=1
            header[1..5].copy_from_slice(&payload_len.to_le_bytes());
            let _ = file.seek(SeekFrom::Start(offset));
            let _ = file.write_all(&header);
            let _ = file.write_all(&payload);
            let _ = file.flush();
        }

        al.size += 5 + payload_len as u64;
        al.index.push(LogMeta {
            offset,
            payload_len,
            id: record.id,
            timestamp_ms: record.timestamp_ms,
            confidence: record.confidence.clone(),
            alive: true,
            cached: if al.file.is_none() { Some(payload) } else { None },
        });

        record.id
    }

    /// Resolve (dismiss/fix) a record. Flips the alive byte in-place.
    /// Returns true if found and updated.
    pub fn resolve(&mut self, app_id: &str, id: u64) -> bool {
        let al = match self.apps.get_mut(app_id) {
            Some(a) => a,
            None => return false,
        };
        let meta = match al.index.iter_mut().find(|m| m.id == id && m.alive) {
            Some(m) => m,
            None => return false,
        };
        let offset = meta.offset;
        let size = al.size;
        if let Some(ref mut file) = al.file {
            let _ = file.seek(SeekFrom::Start(offset));
            let _ = file.write_all(&[0u8]); // tombstone
            let _ = file.seek(SeekFrom::Start(size)); // restore write position
        }
        meta.alive = false;
        true
    }

    /// Query records with filters. Returns most-recent-first.
    pub fn query(&mut self, q: &LogQuery) -> LogQueryResult {
        let app_ids: Vec<String> = match &q.app_id {
            Some(id) => vec![id.clone()],
            None => self.apps.keys().cloned().collect(),
        };

        // Collect owned copies of matching meta fields to avoid holding a borrow into self.apps
        struct Candidate { app_id: String, offset: u64, payload_len: u32, timestamp_ms: u64, cached: Option<Vec<u8>> }

        let mut candidates: Vec<Candidate> = Vec::new();
        for app_id in &app_ids {
            let Some(al) = self.apps.get(app_id) else { continue };
            for meta in &al.index {
                if !Self::matches(meta, q) { continue; }
                candidates.push(Candidate {
                    app_id: app_id.clone(),
                    offset: meta.offset,
                    payload_len: meta.payload_len,
                    timestamp_ms: meta.timestamp_ms,
                    cached: meta.cached.clone(),
                });
            }
        }

        candidates.sort_by(|a, b| b.timestamp_ms.cmp(&a.timestamp_ms));
        let total = candidates.len();

        let mut records = Vec::new();
        for c in candidates.into_iter().skip(q.offset).take(q.limit) {
            let record = if let Some(cached) = c.cached {
                serde_json::from_slice(&cached).ok()
            } else {
                self.read_at(&c.app_id, c.offset, c.payload_len)
            };
            if let Some(r) = record { records.push(r); }
        }

        LogQueryResult { total, records }
    }

    fn matches(meta: &LogMeta, q: &LogQuery) -> bool {
        if let Some(resolved) = q.resolved {
            // resolved=false → want unresolved → alive must be true
            // resolved=true  → want resolved   → alive must be false
            if meta.alive == resolved { return false; }
        }
        if let Some(since) = q.since_ms {
            if meta.timestamp_ms < since { return false; }
        }
        true
    }

    fn read_at(&mut self, app_id: &str, offset: u64, payload_len: u32) -> Option<LogRecord> {
        let al = self.apps.get_mut(app_id)?;
        let file = al.file.as_mut()?;
        file.seek(SeekFrom::Start(offset + 5)).ok()?; // skip 5-byte header
        let mut buf = vec![0u8; payload_len as usize];
        file.read_exact(&mut buf).ok()?;
        serde_json::from_slice(&buf).ok()
    }

    /// Number of alive (unresolved) records for an app.
    pub fn count_alive(&self, app_id: &str) -> usize {
        self.apps.get(app_id)
            .map(|al| al.index.iter().filter(|m| m.alive).count())
            .unwrap_or(0)
    }

    /// Total records (alive + resolved) for an app.
    pub fn count_total(&self, app_id: &str) -> usize {
        self.apps.get(app_id).map(|al| al.index.len()).unwrap_or(0)
    }

    /// Read a single record by id from any app.
    pub fn get_record(&mut self, app_id: &str, id: u64) -> Option<LogRecord> {
        let al = self.apps.get(app_id)?;
        let meta = al.index.iter().find(|m| m.id == id)?;
        if let Some(ref cached) = meta.cached {
            return serde_json::from_slice(cached).ok();
        }
        let offset = meta.offset;
        let payload_len = meta.payload_len;
        self.read_at(app_id, offset, payload_len)
    }

    /// Set the LLM review status for a record (in-memory only).
    pub fn set_review_status(&mut self, app_id: &str, id: u64, status: ReviewStatus) {
        if let Some(al) = self.apps.get_mut(app_id) {
            al.review_status.insert(id, status);
        }
    }

    /// Return (app_id, id) for all alive unreviewed records.
    /// Used by the background worker to find pending work.
    /// Every routed request is a candidate — the confidence threshold inside
    /// full_review gates whether LLM is actually called.
    pub fn pending_worker_ids(&self, app_id_filter: Option<&str>) -> Vec<(String, u64)> {
        let mut pending = Vec::new();
        for (app_id, al) in &self.apps {
            if let Some(filter) = app_id_filter {
                if app_id != filter { continue; }
            }
            for meta in &al.index {
                if !meta.alive { continue; }
                let already_done = al.review_status.get(&meta.id)
                    .map(|s| s.status == "done" || s.status == "escalated" || s.status == "processing")
                    .unwrap_or(false);
                if already_done { continue; }
                pending.push((app_id.clone(), meta.id));
            }
        }
        pending
    }

    /// Drop a single app's log: remove the on-disk `.bin` file and the
    /// in-memory index. Used when a namespace is deleted so its query
    /// history doesn't linger as a privacy leak or re-appear after restart.
    pub fn drop_app(&mut self, app_id: &str) {
        self.apps.remove(app_id);
        if let Some(ref dir) = self.data_dir {
            let _ = fs::remove_file(dir.join(format!("{}.bin", app_id)));
        }
    }

    /// Stats for all apps.
    /// Delete all log files and reset in-memory state.
    pub fn clear_all(&mut self) {
        if let Some(ref dir) = self.data_dir.clone() {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().map(|e| e == "bin").unwrap_or(false) {
                        let _ = fs::remove_file(&p);
                    }
                }
            }
        }
        self.apps.clear();
    }

    pub fn stats(&self) -> Vec<serde_json::Value> {
        self.apps.iter().map(|(app_id, al)| {
            let total = al.index.len();
            let alive = al.index.iter().filter(|m| m.alive).count();
            serde_json::json!({
                "app_id": app_id,
                "total": total,
                "unresolved": alive,
                "size_bytes": al.size,
            })
        }).collect()
    }
}
