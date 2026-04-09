//! Background sync worker: version polling and log shipping.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use crate::Router;
use super::types::{ConnectConfig, LogEntry, SyncResponse};

pub(crate) fn run_background(
    apps: Arc<RwLock<HashMap<String, Arc<Router>>>>,
    versions: Arc<RwLock<HashMap<String, u64>>>,
    config: ConnectConfig,
    log_buf: Arc<Mutex<Vec<LogEntry>>>,
) {
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => { eprintln!("[asv-connect] failed to build HTTP client: {}", e); return; }
    };

    let sync_interval = Duration::from_secs(config.sync_interval_secs);
    let flush_interval = Duration::from_secs(config.log_flush_secs);

    let mut last_sync  = Instant::now().checked_sub(sync_interval).unwrap_or_else(Instant::now);
    let mut last_flush = Instant::now().checked_sub(flush_interval).unwrap_or_else(Instant::now);

    loop {
        std::thread::sleep(Duration::from_secs(5));
        let now = Instant::now();

        if now.duration_since(last_sync) >= sync_interval {
            for app_id in &config.app_ids {
                let local_ver = versions.read().unwrap().get(app_id).copied().unwrap_or(0);
                match check_and_apply(&client, &config, app_id, local_ver, &apps, &versions) {
                    Ok(true)  => eprintln!("[asv-connect] reloaded app={}", app_id),
                    Ok(false) => {}
                    Err(e)    => eprintln!("[asv-connect] sync error app={}: {}", app_id, e),
                }
            }
            last_sync = now;
        }

        if now.duration_since(last_flush) >= flush_interval {
            flush_logs(&client, &config, &log_buf);
            last_flush = now;
        }
    }
}

fn check_and_apply(
    client: &reqwest::blocking::Client,
    config: &ConnectConfig,
    app_id: &str,
    local_version: u64,
    apps: &Arc<RwLock<HashMap<String, Arc<Router>>>>,
    versions: &Arc<RwLock<HashMap<String, u64>>>,
) -> Result<bool, String> {
    let url = format!("{}/api/sync?version={}", config.server_url, local_version);
    let mut req = client.get(&url).header("X-App-ID", app_id);
    if let Some(ref key) = config.api_key {
        req = req.header("X-Api-Key", key);
    }
    let resp = req.send().map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let sync: SyncResponse = resp.json().map_err(|e| e.to_string())?;
    if sync.up_to_date { return Ok(false); }

    let json = sync.export.ok_or_else(|| "no export in response".to_string())?;
    let new_router = Router::import_json(&json).map_err(|e| format!("import failed: {}", e))?;

    apps.write().unwrap().insert(app_id.to_string(), Arc::new(new_router));
    versions.write().unwrap().insert(app_id.to_string(), sync.version);
    Ok(true)
}

fn flush_logs(
    client: &reqwest::blocking::Client,
    config: &ConnectConfig,
    log_buf: &Arc<Mutex<Vec<LogEntry>>>,
) {
    let entries: Vec<LogEntry> = {
        let mut buf = log_buf.lock().unwrap();
        std::mem::take(&mut *buf)
    };
    if entries.is_empty() { return; }

    let mut by_app: HashMap<String, Vec<&LogEntry>> = HashMap::new();
    for e in &entries { by_app.entry(e.app_id.clone()).or_default().push(e); }

    let mut failed: Vec<LogEntry> = Vec::new();
    for (app_id, batch) in by_app {
        let url = format!("{}/api/ingest", config.server_url);
        let mut req = client.post(&url).header("X-App-ID", &app_id).json(&batch);
        if let Some(ref key) = config.api_key { req = req.header("X-Api-Key", key); }
        if let Err(e) = req.send() {
            eprintln!("[asv-connect] log flush error app={}: {}", app_id, e);
            failed.extend(batch.into_iter().cloned());
        }
    }

    if !failed.is_empty() {
        let mut buf = log_buf.lock().unwrap();
        failed.extend(buf.drain(..));
        failed.truncate(config.log_buffer_max);
        *buf = failed;
    }
}

pub(crate) fn build_client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())
}
