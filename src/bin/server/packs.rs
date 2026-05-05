//! Reference pack installation and listing.
//!
//! Packs are namespace bundles distributed as GitHub release tarballs. Each
//! tarball expands to a single top-level directory whose name matches the pack
//! (e.g. `safety-filter/`), containing `_ns.json` + per-intent JSON files.
//!
//! URL pattern:
//!   https://github.com/gladius/microresolve/releases/download/v<VERSION>/pack-<NAME>-v<VERSION>.tar.gz

use std::io;
use std::path::Path;

/// The 4 officially-distributed reference packs.
pub fn known_packs() -> &'static [&'static str] {
    &[
        "safety-filter",
        "eu-ai-act-prohibited",
        "hipaa-triage",
        "mcp-tools-generic",
    ]
}

/// Build the download URL for a pack tarball.
fn pack_url(pack: &str) -> String {
    let version = env!("CARGO_PKG_VERSION");
    format!(
        "https://github.com/gladius/microresolve/releases/download/v{version}/pack-{pack}-v{version}.tar.gz"
    )
}

/// Install a reference pack into `data_dir`.
///
/// Steps:
/// 1. Validate `pack` is in [`known_packs`].
/// 2. Fetch the tarball from the GitHub release for this binary's version.
/// 3. Extract to `data_dir/<pack>/`.
/// 4. Verify `data_dir/<pack>/_ns.json` exists after extraction.
pub fn install(pack: &str, data_dir: &Path) -> io::Result<()> {
    // 1. Validate pack name.
    if !known_packs().contains(&pack) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unknown pack '{}'. Valid pack names are: {}",
                pack,
                known_packs().join(", ")
            ),
        ));
    }

    // 2. Check target dir — bail if non-empty (--force not yet implemented).
    let target_dir = data_dir.join(pack);
    if target_dir.exists() {
        let is_empty = target_dir
            .read_dir()
            .map(|mut d| d.next().is_none())
            .unwrap_or(false);
        if !is_empty {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "pack '{}' is already installed at {}. \
                     To reinstall, remove the directory and run again. \
                     (--force will be available in v0.2.2)",
                    pack,
                    target_dir.display()
                ),
            ));
        }
    }

    // 3. Fetch tarball.
    let url = pack_url(pack);
    let version = env!("CARGO_PKG_VERSION");
    eprintln!("Downloading {}…", url);

    let response = reqwest::blocking::get(&url).map_err(|e| {
        io::Error::new(
            io::ErrorKind::ConnectionRefused,
            format!("HTTP request failed for {}: {}", url, e),
        )
    })?;

    if !response.status().is_success() {
        let status = response.status();
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "failed to fetch pack '{}' (HTTP {}). URL: {}\n\
                 This may mean the v{} release tarball hasn't been uploaded yet \
                 (common for pre-release / nightly builds). Check \
                 https://github.com/gladius/microresolve/releases/tag/v{} \
                 for available assets.",
                pack, status, url, version, version
            ),
        ));
    }

    // 4. Stream through GzDecoder → tar::Archive, extract to data_dir.
    let bytes = response.bytes().map_err(|e| {
        io::Error::new(
            io::ErrorKind::BrokenPipe,
            format!("failed to read response body: {}", e),
        )
    })?;

    // Ensure data_dir exists before extraction.
    std::fs::create_dir_all(data_dir)?;

    let cursor = std::io::Cursor::new(bytes);
    let gz = flate2::read::GzDecoder::new(cursor);
    let mut archive = tar::Archive::new(gz);
    archive.unpack(data_dir)?;

    // 5. Verify _ns.json was extracted.
    let ns_json = target_dir.join("_ns.json");
    if !ns_json.exists() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "tarball extracted but '{}' not found. \
                 The archive may be malformed or the pack name '{}' \
                 does not match the top-level directory inside the tarball.",
                ns_json.display(),
                pack
            ),
        ));
    }

    println!("Installed pack '{}' to {}", pack, target_dir.display());
    Ok(())
}

/// Print a table of all 4 reference packs and their install status against `data_dir`.
pub fn list(data_dir: &Path) {
    println!("{:<25} STATUS", "PACK");
    for &pack in known_packs() {
        let ns_json = data_dir.join(pack).join("_ns.json");
        let status = if ns_json.exists() {
            "installed"
        } else {
            "not installed"
        };
        println!("{:<25} {}", pack, status);
    }
    println!();
    println!("Data dir: {}", data_dir.display());
    println!("Install with: microresolve-studio install <pack>");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_packs_complete() {
        let packs = known_packs();
        assert!(packs.contains(&"safety-filter"), "missing safety-filter");
        assert!(
            packs.contains(&"eu-ai-act-prohibited"),
            "missing eu-ai-act-prohibited"
        );
        assert!(packs.contains(&"hipaa-triage"), "missing hipaa-triage");
        assert!(
            packs.contains(&"mcp-tools-generic"),
            "missing mcp-tools-generic"
        );
        assert_eq!(packs.len(), 4, "expected exactly 4 packs");
    }

    #[test]
    fn test_install_url_construction() {
        let version = env!("CARGO_PKG_VERSION");
        let url = pack_url("safety-filter");
        assert_eq!(
            url,
            format!(
                "https://github.com/gladius/microresolve/releases/download/v{}/pack-safety-filter-v{}.tar.gz",
                version, version
            )
        );
    }

    #[test]
    fn test_install_unknown_pack_errors() {
        let tmp = std::path::PathBuf::from("/tmp/mr-packs-test-nonexistent");
        let err = install("not-a-real-pack", &tmp).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unknown pack"),
            "expected 'unknown pack' in error: {}",
            msg
        );
        // Must list valid pack names in the error message.
        for &pack in known_packs() {
            assert!(
                msg.contains(pack),
                "expected '{}' in error message: {}",
                pack,
                msg
            );
        }
    }
}
