//! Best-effort git wrapper for the namespace data directory.
//!
//! The server treats the `--data` directory as a git repo: every namespace
//! mutation auto-commits (see `state::git_commit`), and operators can browse
//! history or roll back via the HTTP API. All operations are best-effort —
//! if git isn't installed or the dir isn't a repo, callers degrade silently
//! instead of panicking.

use std::path::Path;
use std::process::Command;

/// One entry in the data dir's commit log.
#[derive(serde::Serialize, Debug)]
pub struct Commit {
    /// Full 40-char SHA.
    pub sha: String,
    /// Subject line of the commit message.
    pub message: String,
    /// Unix seconds.
    pub ts: i64,
    /// Author name (%an from git log).
    pub author: String,
}

/// Initialize a git repo at `dir` if one doesn't already exist.
///
/// Sets a default `user.name` / `user.email` so commits don't fail when the
/// global git identity isn't configured (common on minimal Docker images).
/// Best-effort: any failure is logged + ignored. Safe to call on startup.
pub fn ensure_repo(dir: &Path) {
    if !dir.exists() {
        return;
    }
    if dir.join(".git").exists() {
        return;
    }

    let init = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(dir)
        .status();
    if init.map(|s| !s.success()).unwrap_or(true) {
        eprintln!(
            "[data_git] git init failed in {} (git not installed?)",
            dir.display()
        );
        return;
    }

    // Set local identity so the very first commit doesn't fail when global
    // user.* is unset. Operators with their own identity remain unaffected.
    let _ = Command::new("git")
        .args(["config", "user.email", "microresolve@local"])
        .current_dir(dir)
        .status();
    let _ = Command::new("git")
        .args(["config", "user.name", "microresolve"])
        .current_dir(dir)
        .status();

    // Initial commit so subsequent rollbacks always have somewhere to land.
    let _ = Command::new("git")
        .args(["add", "-A"])
        .current_dir(dir)
        .status();
    let _ = Command::new("git")
        .args(["commit", "--quiet", "--allow-empty", "-m", "init"])
        .current_dir(dir)
        .status();
}

/// Last `limit` commits (oldest-last), filtered to ones that touched
/// `ns_id/`. If `ns_id` is empty, returns engine-wide history.
pub fn log(dir: &Path, ns_id: &str, limit: usize) -> Vec<Commit> {
    if !dir.join(".git").exists() {
        return Vec::new();
    }

    let mut args: Vec<String> = vec![
        "-C".into(),
        dir.display().to_string(),
        "log".into(),
        format!("-n{}", limit),
        // %H = full sha, %ct = unix ts, %an = author name, %s = subject; \x1f = unit separator
        "--pretty=format:%H\x1f%ct\x1f%an\x1f%s".into(),
    ];
    if !ns_id.is_empty() {
        args.push("--".into());
        args.push(format!("{}/", ns_id));
    }

    let out = match Command::new("git").args(&args).output() {
        Ok(o) if o.status.success() => o.stdout,
        _ => return Vec::new(),
    };

    String::from_utf8_lossy(&out)
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(4, '\x1f').collect();
            if parts.len() != 4 {
                return None;
            }
            Some(Commit {
                sha: parts[0].to_string(),
                ts: parts[1].parse().ok()?,
                author: parts[2].to_string(),
                message: parts[3].to_string(),
            })
        })
        .collect()
}

/// Configure the `origin` remote for the data-dir repo.
///
/// Idempotent: replaces an existing `origin` if the URL differs, no-ops if
/// it already matches. No network call — the URL is just registered locally.
pub fn set_remote(dir: &Path, url: &str) -> Result<(), String> {
    if !dir.join(".git").exists() {
        return Err("data dir is not a git repo".into());
    }
    let existing = Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(dir)
        .output()
        .map_err(|e| e.to_string())?;
    let current = String::from_utf8_lossy(&existing.stdout).trim().to_string();

    if !existing.status.success() {
        // No remote yet — add it.
        let r = Command::new("git")
            .args(["remote", "add", "origin", url])
            .current_dir(dir)
            .status()
            .map_err(|e| e.to_string())?;
        if !r.success() {
            return Err("git remote add failed".into());
        }
    } else if current != url {
        // Remote exists but mismatches — update.
        let r = Command::new("git")
            .args(["remote", "set-url", "origin", url])
            .current_dir(dir)
            .status()
            .map_err(|e| e.to_string())?;
        if !r.success() {
            return Err("git remote set-url failed".into());
        }
    }
    Ok(())
}

/// Push the current branch to `origin`. Returns Ok(()) on success,
/// Err with stderr on failure. Sets upstream on first push.
pub fn push(dir: &Path) -> Result<(), String> {
    if !dir.join(".git").exists() {
        return Err("data dir is not a git repo".into());
    }
    let out = Command::new("git")
        .args(["push", "--set-upstream", "origin", "HEAD"])
        .current_dir(dir)
        .output()
        .map_err(|e| format!("git push spawn failed: {}", e))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

// ── Semantic diff ─────────────────────────────────────────────────────────────

#[derive(serde::Serialize, Debug)]
pub struct IntentWithPhrases {
    pub id: String,
    pub phrases_sample: Vec<String>,
    pub total_phrases: usize,
}

#[derive(serde::Serialize, Debug, Default)]
pub struct NamespaceDiff {
    pub from: String,
    pub to: String,
    pub intents_added: Vec<IntentWithPhrases>,
    pub intents_removed: Vec<IntentWithPhrases>,
    pub phrases_added: Vec<PhraseChange>,
    pub phrases_removed: Vec<PhraseChange>,
    pub metadata_changes: Vec<MetadataChange>,
    pub l2_edges_changed: usize,
    pub l1_edges_changed: usize,
}

#[derive(serde::Serialize, Debug)]
pub struct PhraseChange {
    pub intent_id: String,
    pub lang: String,
    pub phrase: String,
}

#[derive(serde::Serialize, Debug)]
pub struct MetadataChange {
    pub intent_id: String,
    pub field: String,
    pub from: String,
    pub to: String,
}

/// Read a file at a specific git sha. Returns None if not present at that sha.
fn git_show(dir: &Path, sha: &str, path: &str) -> Option<String> {
    let out = Command::new("git")
        .args([
            "-C",
            &dir.display().to_string(),
            "show",
            &format!("{}:{}", sha, path),
        ])
        .output()
        .ok()?;
    if out.status.success() {
        String::from_utf8(out.stdout).ok()
    } else {
        None
    }
}

/// List files under ns_id/ at a given sha.
fn ls_tree(dir: &Path, sha: &str, ns_id: &str) -> Vec<String> {
    let out = match Command::new("git")
        .args([
            "-C",
            &dir.display().to_string(),
            "ls-tree",
            "-r",
            "--name-only",
            sha,
            "--",
            &format!("{}/", ns_id),
        ])
        .output()
    {
        Ok(o) if o.status.success() => o.stdout,
        _ => return Vec::new(),
    };
    String::from_utf8_lossy(&out)
        .lines()
        .map(|s| s.to_string())
        .collect()
}

/// Verify a sha exists in the repo.
fn sha_exists(dir: &Path, sha: &str) -> bool {
    Command::new("git")
        .args([
            "-C",
            &dir.display().to_string(),
            "cat-file",
            "-e",
            &format!("{}^{{commit}}", sha),
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Parse an intent JSON blob into phrases + metadata fields we diff.
fn parse_intent(
    json: &str,
) -> (
    std::collections::HashMap<String, Vec<String>>,
    std::collections::HashMap<String, String>,
) {
    let v: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return Default::default(),
    };
    let mut phrases: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    if let Some(ph) = v.get("phrases").and_then(|p| p.as_object()) {
        for (lang, arr) in ph {
            let words: Vec<String> = arr
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|s| s.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            phrases.insert(lang.clone(), words);
        }
    }
    let mut meta: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for field in &["description", "instructions", "persona", "type"] {
        if let Some(s) = v.get(field).and_then(|f| f.as_str()) {
            meta.insert(field.to_string(), s.to_string());
        }
    }
    for field in &["guardrails"] {
        if let Some(arr) = v.get(field).and_then(|f| f.as_array()) {
            let joined: Vec<String> = arr
                .iter()
                .filter_map(|s| s.as_str().map(|s| s.to_string()))
                .collect();
            meta.insert(field.to_string(), joined.join(", "));
        }
    }
    if let Some(target) = v.get("target") {
        if let Some(url) = target.get("url").and_then(|u| u.as_str()) {
            meta.insert("target.url".to_string(), url.to_string());
        }
        if let Some(model) = target.get("model").and_then(|m| m.as_str()) {
            meta.insert("target.model".to_string(), model.to_string());
        }
    }
    (phrases, meta)
}

/// Parse _ns.json into metadata map.
fn parse_ns(json: &str) -> std::collections::HashMap<String, String> {
    let v: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return Default::default(),
    };
    let mut meta = std::collections::HashMap::new();
    for field in &["name", "description"] {
        if let Some(s) = v.get(field).and_then(|f| f.as_str()) {
            meta.insert(field.to_string(), s.to_string());
        }
    }
    if let Some(t) = v.get("default_threshold").and_then(|t| t.as_f64()) {
        meta.insert("default_threshold".to_string(), t.to_string());
    }
    meta
}

/// Count how many edges differ between two L1/L2 JSON blobs (flat object or nested).
/// We compare serialised values: any key missing from either side, or any changed value.
fn count_edge_diff(from_json: Option<&str>, to_json: Option<&str>) -> usize {
    fn flatten(
        v: &serde_json::Value,
        prefix: &str,
        out: &mut std::collections::HashMap<String, String>,
    ) {
        match v {
            serde_json::Value::Object(map) => {
                for (k, child) in map {
                    let key = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{}.{}", prefix, k)
                    };
                    flatten(child, &key, out);
                }
            }
            other => {
                out.insert(prefix.to_string(), other.to_string());
            }
        }
    }

    let parse_flat = |s: Option<&str>| -> std::collections::HashMap<String, String> {
        let v: serde_json::Value = s
            .and_then(|j| serde_json::from_str(j).ok())
            .unwrap_or(serde_json::Value::Object(Default::default()));
        let mut m = std::collections::HashMap::new();
        flatten(&v, "", &mut m);
        m
    };

    let from_flat = parse_flat(from_json);
    let to_flat = parse_flat(to_json);
    let mut diff = 0usize;
    for (k, fv) in &from_flat {
        match to_flat.get(k) {
            Some(tv) if tv == fv => {}
            _ => diff += 1,
        }
    }
    for k in to_flat.keys() {
        if !from_flat.contains_key(k) {
            diff += 1;
        }
    }
    diff
}

/// Compute a semantic diff between two commits for the given namespace.
pub fn diff(dir: &Path, ns_id: &str, from: &str, to: &str) -> Result<NamespaceDiff, String> {
    if !sha_exists(dir, from) {
        return Err(format!("sha not found: {}", from));
    }
    if !sha_exists(dir, to) {
        return Err(format!("sha not found: {}", to));
    }

    if from == to {
        return Ok(NamespaceDiff {
            from: from.into(),
            to: to.into(),
            ..Default::default()
        });
    }

    let from_files: std::collections::HashSet<String> =
        ls_tree(dir, from, ns_id).into_iter().collect();
    let to_files: std::collections::HashSet<String> = ls_tree(dir, to, ns_id).into_iter().collect();

    // Intent files: {ns_id}/{intent_id}.json or {ns_id}/{domain}/{intent_id}.json
    // Exclude _ns.json, _l1.json, _l2.json, _domain.json
    let is_intent = |path: &str| -> Option<String> {
        let filename = std::path::Path::new(path).file_name()?.to_str()?;
        if filename.starts_with('_') {
            return None;
        }
        if !filename.ends_with(".json") {
            return None;
        }
        // intent_id is the relative path under ns_id/, without extension
        let rel = path.strip_prefix(&format!("{}/", ns_id))?;
        let id = rel.strip_suffix(".json")?;
        Some(id.replace('/', ":"))
    };

    let from_intents: std::collections::HashMap<String, String> = from_files
        .iter()
        .filter_map(|p| is_intent(p).map(|id| (id, p.clone())))
        .collect();
    let to_intents: std::collections::HashMap<String, String> = to_files
        .iter()
        .filter_map(|p| is_intent(p).map(|id| (id, p.clone())))
        .collect();

    let mut result = NamespaceDiff {
        from: from.into(),
        to: to.into(),
        ..Default::default()
    };

    // Added / removed intents
    for (id, to_path) in &to_intents {
        if !from_intents.contains_key(id) {
            let json = git_show(dir, to, to_path);
            let (phrases, _) = parse_intent(json.as_deref().unwrap_or("{}"));
            let mut seen = std::collections::HashSet::new();
            let deduped: Vec<String> = phrases
                .values()
                .flat_map(|v| v.iter().cloned())
                .filter(|p| seen.insert(p.clone()))
                .collect();
            let total = deduped.len();
            let sample = deduped.into_iter().take(3).collect();
            result.intents_added.push(IntentWithPhrases {
                id: id.clone(),
                phrases_sample: sample,
                total_phrases: total,
            });
        }
    }
    for (id, from_path) in &from_intents {
        if !to_intents.contains_key(id) {
            let json = git_show(dir, from, from_path);
            let (phrases, _) = parse_intent(json.as_deref().unwrap_or("{}"));
            let mut seen = std::collections::HashSet::new();
            let deduped: Vec<String> = phrases
                .values()
                .flat_map(|v| v.iter().cloned())
                .filter(|p| seen.insert(p.clone()))
                .collect();
            let total = deduped.len();
            let sample = deduped.into_iter().take(3).collect();
            result.intents_removed.push(IntentWithPhrases {
                id: id.clone(),
                phrases_sample: sample,
                total_phrases: total,
            });
        }
    }
    result.intents_added.sort_by(|a, b| a.id.cmp(&b.id));
    result.intents_removed.sort_by(|a, b| a.id.cmp(&b.id));

    // Phrases + metadata diffs for intents present in both
    for (id, from_path) in &from_intents {
        if let Some(to_path) = to_intents.get(id) {
            let from_json = git_show(dir, from, from_path);
            let to_json = git_show(dir, to, to_path);
            let (from_phrases, from_meta) = parse_intent(from_json.as_deref().unwrap_or("{}"));
            let (to_phrases, to_meta) = parse_intent(to_json.as_deref().unwrap_or("{}"));

            // Phrase diff per lang
            let mut all_langs: std::collections::HashSet<String> = std::collections::HashSet::new();
            all_langs.extend(from_phrases.keys().cloned());
            all_langs.extend(to_phrases.keys().cloned());
            for lang in all_langs {
                let from_set: std::collections::HashSet<&str> = from_phrases
                    .get(&lang)
                    .map(|v| v.iter().map(|s| s.as_str()).collect())
                    .unwrap_or_default();
                let to_set: std::collections::HashSet<&str> = to_phrases
                    .get(&lang)
                    .map(|v| v.iter().map(|s| s.as_str()).collect())
                    .unwrap_or_default();
                for ph in to_set.difference(&from_set) {
                    result.phrases_added.push(PhraseChange {
                        intent_id: id.clone(),
                        lang: lang.clone(),
                        phrase: ph.to_string(),
                    });
                }
                for ph in from_set.difference(&to_set) {
                    result.phrases_removed.push(PhraseChange {
                        intent_id: id.clone(),
                        lang: lang.clone(),
                        phrase: ph.to_string(),
                    });
                }
            }

            // Metadata diff
            let mut all_fields: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            all_fields.extend(from_meta.keys().cloned());
            all_fields.extend(to_meta.keys().cloned());
            for field in all_fields {
                let fv = from_meta.get(&field).map(|s| s.as_str()).unwrap_or("");
                let tv = to_meta.get(&field).map(|s| s.as_str()).unwrap_or("");
                if fv != tv {
                    result.metadata_changes.push(MetadataChange {
                        intent_id: id.clone(),
                        field,
                        from: fv.to_string(),
                        to: tv.to_string(),
                    });
                }
            }
        }
    }

    // _ns.json metadata diff
    let ns_path = format!("{}/_ns.json", ns_id);
    let from_ns = git_show(dir, from, &ns_path);
    let to_ns = git_show(dir, to, &ns_path);
    if from_ns.as_deref() != to_ns.as_deref() {
        let from_meta = parse_ns(from_ns.as_deref().unwrap_or("{}"));
        let to_meta = parse_ns(to_ns.as_deref().unwrap_or("{}"));
        let mut all_fields: std::collections::HashSet<String> = std::collections::HashSet::new();
        all_fields.extend(from_meta.keys().cloned());
        all_fields.extend(to_meta.keys().cloned());
        for field in all_fields {
            let fv = from_meta.get(&field).map(|s| s.as_str()).unwrap_or("");
            let tv = to_meta.get(&field).map(|s| s.as_str()).unwrap_or("");
            if fv != tv {
                result.metadata_changes.push(MetadataChange {
                    intent_id: "_ns".to_string(),
                    field,
                    from: fv.to_string(),
                    to: tv.to_string(),
                });
            }
        }
    }

    result
        .metadata_changes
        .sort_by(|a, b| a.intent_id.cmp(&b.intent_id).then(a.field.cmp(&b.field)));

    // L2 + L1 edge counts
    let l2_from = git_show(dir, from, &format!("{}/_l2.json", ns_id));
    let l2_to = git_show(dir, to, &format!("{}/_l2.json", ns_id));
    result.l2_edges_changed = count_edge_diff(l2_from.as_deref(), l2_to.as_deref());

    let l1_from = git_show(dir, from, &format!("{}/_l1.json", ns_id));
    let l1_to = git_show(dir, to, &format!("{}/_l1.json", ns_id));
    result.l1_edges_changed = count_edge_diff(l1_from.as_deref(), l1_to.as_deref());

    Ok(result)
}

/// Hard-reset the data dir to `sha`. Returns Err with a human-readable
/// message on failure (caller surfaces as HTTP 4xx/5xx).
pub fn rollback(dir: &Path, sha: &str) -> Result<(), String> {
    if !dir.join(".git").exists() {
        return Err("data dir is not a git repo".into());
    }
    // Reject anything that doesn't look like a hex SHA — defense in depth
    // against shell injection even though we never spawn via shell.
    if sha.is_empty() || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid sha".into());
    }
    let out = Command::new("git")
        .args(["reset", "--hard", sha])
        .current_dir(dir)
        .output()
        .map_err(|e| format!("git reset failed to spawn: {}", e))?;
    if !out.status.success() {
        return Err(format!(
            "git reset --hard {}: {}",
            sha,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(())
}
