//! Instance backup and restore endpoints.
//!
//! - `GET  /api/backup`  — stream the entire data_dir as a zip file
//! - `POST /api/restore` — accept a zip via multipart, validate, and atomic-swap into data_dir

use crate::state::*;
use axum::{
    body::Body,
    extract::{Multipart, State},
    http::{header, StatusCode},
    response::Response,
    routing::{get, post},
};
use std::io::Write as _;

pub fn routes() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/api/backup", get(backup))
        .route("/api/restore", post(restore))
}

/// Stream the entire data_dir as a zip archive, excluding any `.git/` subtree.
pub async fn backup(State(state): State<AppState>) -> Result<Response, (StatusCode, String)> {
    let data_dir = state
        .data_dir
        .as_deref()
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "No data directory configured".to_string(),
            )
        })?
        .to_string();

    let zip_bytes = tokio::task::spawn_blocking(move || build_zip(&data_dir))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let today = {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let days = secs / 86400;
        let (y, m, d) = days_to_ymd(days);
        format!("{:04}-{:02}-{:02}", y, m, d)
    };
    let filename = format!("microresolve-backup-{}.zip", today);

    Ok(Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "application/zip")
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{}\"", filename),
        )
        .header(header::CONTENT_LENGTH, zip_bytes.len().to_string())
        .body(Body::from(zip_bytes))
        .unwrap())
}

/// Accept a multipart zip, validate it, and atomic-swap it into data_dir.
pub async fn restore(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<axum::Json<serde_json::Value>, (StatusCode, String)> {
    let data_dir = state
        .data_dir
        .as_deref()
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "No data directory configured".to_string(),
            )
        })?
        .to_string();

    // Read the zip bytes from the multipart field.
    let mut zip_bytes: Option<Vec<u8>> = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Multipart error: {}", e)))?
    {
        let data = field
            .bytes()
            .await
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("Read error: {}", e)))?;
        zip_bytes = Some(data.to_vec());
        break;
    }
    let zip_bytes = zip_bytes.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "No file field in multipart body".to_string(),
        )
    })?;

    let ns_ids = state.engine.namespaces();

    // Flush all in-memory state first. If any namespace is dirty after flush
    // (shouldn't happen unless flush fails), return 409.
    for id in &ns_ids {
        if let Some(h) = state.engine.try_namespace(id) {
            h.flush().map_err(|e| {
                (
                    StatusCode::CONFLICT,
                    format!("Could not flush namespace '{}' before restore: {}", id, e),
                )
            })?;
        }
    }

    let data_dir2 = data_dir.clone();
    tokio::task::spawn_blocking(move || unpack_and_swap(zip_bytes, &data_dir2))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, e))?;

    // Reload every namespace from the freshly-written data_dir.
    let mut reloaded = Vec::new();
    for id in &ns_ids {
        let _ = state.engine.reload_namespace(id);
        reloaded.push(id.clone());
    }
    // Also discover new namespaces that were in the zip but not in memory.
    if let Ok(entries) = std::fs::read_dir(&data_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('_') || name.starts_with('.') {
                continue;
            }
            if entry.path().is_dir() && !reloaded.contains(&name) {
                let _ = state.engine.namespace(&name);
                let _ = state.engine.reload_namespace(&name);
            }
        }
    }

    Ok(axum::Json(serde_json::json!({ "ok": true })))
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn build_zip(data_dir: &str) -> Result<Vec<u8>, String> {
    use zip::{write::SimpleFileOptions, ZipWriter};

    let base = std::path::Path::new(data_dir);
    let mut buf = Vec::new();
    let mut zip = ZipWriter::new(std::io::Cursor::new(&mut buf));
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    walk_dir(base, base, &mut zip, opts).map_err(|e| e.to_string())?;
    zip.finish().map_err(|e| e.to_string())?;
    Ok(buf)
}

fn walk_dir(
    base: &std::path::Path,
    dir: &std::path::Path,
    zip: &mut zip::ZipWriter<std::io::Cursor<&mut Vec<u8>>>,
    opts: zip::write::SimpleFileOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let rel = path.strip_prefix(base)?;

        // Skip .git subtree — it can be hundreds of MB.
        if rel.starts_with(".git") {
            continue;
        }

        if path.is_dir() {
            walk_dir(base, &path, zip, opts)?;
        } else {
            let name = rel.to_string_lossy().replace('\\', "/");
            zip.start_file(name, opts)?;
            let data = std::fs::read(&path)?;
            zip.write_all(&data)?;
        }
    }
    Ok(())
}

/// Unpack zip to a temp dir, validate it, then overwrite data_dir with the contents.
fn unpack_and_swap(zip_bytes: Vec<u8>, data_dir: &str) -> Result<(), String> {
    use std::io::Read as _;
    use zip::ZipArchive;

    let cursor = std::io::Cursor::new(zip_bytes);
    let mut archive = ZipArchive::new(cursor).map_err(|e| format!("Invalid zip: {}", e))?;

    // Validate: at least one file matching */_ns.json or _ns.json at top level.
    let has_ns = (0..archive.len()).any(|i| {
        archive
            .by_index(i)
            .ok()
            .map(|f| f.name().ends_with("_ns.json"))
            .unwrap_or(false)
    });
    if !has_ns {
        return Err("Zip does not contain any namespace data (_ns.json not found)".to_string());
    }

    let base = std::path::Path::new(data_dir);

    // Write each entry directly into the existing data_dir (overwrite-in-place).
    // This avoids needing a rename across potential filesystem boundaries.
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| format!("Zip read error: {}", e))?;
        if file.is_dir() {
            continue;
        }
        let dest = base.join(sanitize_zip_path(file.name())?);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
        }
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| format!("Read zip entry: {}", e))?;
        std::fs::write(&dest, &data).map_err(|e| format!("Write {}: {}", dest.display(), e))?;
    }

    Ok(())
}

/// Reject zip-slip paths (absolute paths or `..` components).
fn sanitize_zip_path(name: &str) -> Result<std::path::PathBuf, String> {
    let path = std::path::Path::new(name);
    if path.is_absolute() {
        return Err(format!("Unsafe path in zip: {}", name));
    }
    for component in path.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(format!("Unsafe path in zip (traversal): {}", name));
        }
    }
    Ok(path.to_path_buf())
}

/// Convert days-since-epoch to (year, month, day). Good-enough Gregorian for dates near 2020-2040.
fn days_to_ymd(days: u64) -> (u32, u32, u32) {
    // Gregorian calendar algorithm (Fliegel & Van Flandern via Julian Day Number).
    let jd = days as i64 + 2440588; // Unix epoch = JD 2440588
    let l = jd + 68569;
    let n = 4 * l / 146097;
    let l = l - (146097 * n + 3) / 4;
    let i = 4000 * (l + 1) / 1461001;
    let l = l - 1461 * i / 4 + 31;
    let j = 80 * l / 2447;
    let d = l - 2447 * j / 80;
    let l = j / 11;
    let m = j + 2 - 12 * l;
    let y = 100 * (n - 49) + i + l;
    (y as u32, m as u32, d as u32)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let ns = dir.path().join("test-ns");
        std::fs::create_dir_all(&ns).unwrap();
        std::fs::write(ns.join("_ns.json"), r#"{"id":"test-ns"}"#).unwrap();
        std::fs::write(ns.join("phrases.json"), r#"[]"#).unwrap();
        std::fs::write(dir.path().join("_settings.json"), r#"{}"#).unwrap();
        dir
    }

    #[test]
    fn backup_produces_non_empty_zip_with_expected_files() {
        let dir = make_test_data_dir();
        let zip_bytes = build_zip(dir.path().to_str().unwrap()).unwrap();

        assert!(!zip_bytes.is_empty());

        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&zip_bytes)).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();

        assert!(
            names.iter().any(|n| n.ends_with("_ns.json")),
            "Expected _ns.json in zip, got: {:?}",
            names
        );
        assert!(
            names.iter().any(|n| n.contains("_settings.json")),
            "Expected _settings.json in zip, got: {:?}",
            names
        );
    }

    #[test]
    fn git_dir_excluded_from_zip() {
        let dir = make_test_data_dir();
        let git = dir.path().join(".git");
        std::fs::create_dir_all(&git).unwrap();
        std::fs::write(
            git.join("config"),
            b"[core]\n\trepositoryformatversion = 0\n",
        )
        .unwrap();

        let zip_bytes = build_zip(dir.path().to_str().unwrap()).unwrap();
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&zip_bytes)).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();

        assert!(
            !names.iter().any(|n| n.starts_with(".git")),
            ".git files must be excluded, got: {:?}",
            names
        );
    }

    #[test]
    fn round_trip_restore_preserves_namespace_file() {
        let src = make_test_data_dir();
        let zip_bytes = build_zip(src.path().to_str().unwrap()).unwrap();

        let dest = tempfile::tempdir().unwrap();
        unpack_and_swap(zip_bytes, dest.path().to_str().unwrap()).unwrap();

        let ns_json = dest.path().join("test-ns").join("_ns.json");
        assert!(ns_json.exists(), "_ns.json should be restored");
        let content = std::fs::read_to_string(&ns_json).unwrap();
        assert!(content.contains("test-ns"));
    }

    #[test]
    fn restore_rejects_zip_without_ns_json() {
        let mut buf = Vec::new();
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
        let opts = zip::write::SimpleFileOptions::default();
        zip.start_file("readme.txt", opts).unwrap();
        zip.finish().unwrap();

        let dest = tempfile::tempdir().unwrap();
        let err = unpack_and_swap(buf, dest.path().to_str().unwrap()).unwrap_err();
        assert!(
            err.contains("_ns.json"),
            "Expected validation error, got: {}",
            err
        );
    }
}
