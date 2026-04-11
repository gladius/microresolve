//! Directory-based persistence for Router.
//!
//! `save_to_dir(path)` writes one JSON file per intent, plus `_ns.json` and
//! `_domain.json` sidecar files. `load_from_dir(path)` reconstructs the router
//! from those files. The phrase-layer vector is always recomputed from phrases;
//! the learned layer is saved and restored separately.
//!
//! File layout:
//!   {ns_dir}/_ns.json                     — namespace description
//!   {ns_dir}/{intent_name}.json           — intent without domain prefix
//!   {ns_dir}/{domain}/_domain.json        — domain description
//!   {ns_dir}/{domain}/{intent_name}.json  — intent with domain prefix

use crate::*;
use crate::types::IntentType;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

impl Router {
    /// Human-readable description of this namespace/router instance.
    pub fn namespace_description(&self) -> &str {
        &self.namespace_description
    }

    /// Set the namespace description.
    pub fn set_namespace_description(&mut self, desc: &str) {
        self.namespace_description = desc.to_string();
    }

    /// Description for a specific domain prefix (e.g., "billing").
    pub fn domain_description(&self, domain: &str) -> Option<&str> {
        self.domain_descriptions.get(domain).map(|s| s.as_str())
    }

    /// Set description for a domain prefix.
    pub fn set_domain_description(&mut self, domain: &str, desc: &str) {
        self.domain_descriptions.insert(domain.to_string(), desc.to_string());
    }

    /// Remove a domain description entry (does not remove intents).
    pub fn remove_domain_description(&mut self, domain: &str) {
        self.domain_descriptions.remove(domain);
    }

    /// All domain descriptions.
    pub fn domain_descriptions(&self) -> &HashMap<String, String> {
        &self.domain_descriptions
    }

    /// Load a router from a namespace directory.
    ///
    /// Reads `_ns.json`, per-domain `_domain.json`, and per-intent `*.json` files.
    /// Phrases are re-ingested so the phrase vector is always fresh.
    /// Learned weights are restored from the saved `"learned"` field.
    pub fn load_from_dir(path: &Path) -> Result<Self, String> {
        let mut router = Self::new();

        // Namespace description
        if let Ok(json) = std::fs::read_to_string(path.join("_ns.json")) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json) {
                if let Some(desc) = val.get("description").and_then(|d| d.as_str()) {
                    router.namespace_description = desc.to_string();
                }
            }
        }

        let entries = std::fs::read_dir(path)
            .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;

        let mut domain_dirs: Vec<(String, PathBuf)> = Vec::new();

        for entry in entries.flatten() {
            let p = entry.path();
            let name = match p.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            if name.starts_with('_') { continue; }

            if p.is_dir() {
                domain_dirs.push((name, p));
            } else if p.extension().map(|e| e == "json").unwrap_or(false) {
                let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                load_intent_file(&mut router, &p, &stem);
            }
        }

        for (domain, domain_dir) in &domain_dirs {
            // Domain description
            if let Ok(json) = std::fs::read_to_string(domain_dir.join("_domain.json")) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json) {
                    if let Some(desc) = val.get("description").and_then(|d| d.as_str()) {
                        router.domain_descriptions.insert(domain.clone(), desc.to_string());
                    }
                }
            }
            // Intent files in this domain
            if let Ok(sub_entries) = std::fs::read_dir(domain_dir) {
                for sub_entry in sub_entries.flatten() {
                    let p = sub_entry.path();
                    let sub_name = match p.file_name().and_then(|n| n.to_str()) {
                        Some(n) => n.to_string(),
                        None => continue,
                    };
                    if sub_name.starts_with('_') { continue; }
                    if p.extension().map(|e| e == "json").unwrap_or(false) {
                        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                        let intent_id = format!("{}:{}", domain, stem);
                        load_intent_file(&mut router, &p, &intent_id);
                    }
                }
            }
        }

        Ok(router)
    }

    /// Save this router to a namespace directory.
    ///
    /// Writes `_ns.json`, per-domain `_domain.json`, and per-intent `*.json` files.
    /// Stale intent files from deleted intents are removed.
    pub fn save_to_dir(&self, path: &Path) -> Result<(), String> {
        std::fs::create_dir_all(path)
            .map_err(|e| format!("cannot create {}: {}", path.display(), e))?;

        // Namespace description
        let ns_meta = serde_json::json!({"description": self.namespace_description});
        std::fs::write(
            path.join("_ns.json"),
            serde_json::to_string_pretty(&ns_meta).unwrap_or_default(),
        ).map_err(|e| format!("cannot write _ns.json: {}", e))?;

        let mut written: HashSet<PathBuf> = HashSet::new();
        written.insert(path.join("_ns.json"));

        // Write domain descriptions (including explicitly-created domains with no intents)
        for (domain, desc) in &self.domain_descriptions {
            let domain_dir = path.join(domain);
            std::fs::create_dir_all(&domain_dir)
                .map_err(|e| format!("cannot create domain dir {}: {}", domain, e))?;
            let meta = serde_json::json!({"description": desc});
            let meta_path = domain_dir.join("_domain.json");
            std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap_or_default())
                .map_err(|e| format!("cannot write _domain.json for {}: {}", domain, e))?;
            written.insert(meta_path);
        }

        // Write per-intent files
        for intent_id in self.intent_ids() {
            let (domain_opt, name) = split_intent_id(&intent_id);

            let file_path = if let Some(domain) = domain_opt {
                let domain_dir = path.join(domain);
                std::fs::create_dir_all(&domain_dir)
                    .map_err(|e| format!("cannot create domain dir: {}", e))?;
                // Ensure _domain.json exists for intent-derived domains
                let meta_path = domain_dir.join("_domain.json");
                if !written.contains(&meta_path) {
                    let desc = self.domain_descriptions.get(domain).cloned().unwrap_or_default();
                    let meta = serde_json::json!({"description": desc});
                    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap_or_default()).ok();
                    written.insert(meta_path);
                }
                domain_dir.join(format!("{}.json", name))
            } else {
                path.join(format!("{}.json", name))
            };

            let learned: HashMap<String, f32> = self.vectors.get(&intent_id)
                .map(|v| v.learned_terms())
                .unwrap_or_default();

            let intent_json = serde_json::json!({
                "description": self.get_description(&intent_id),
                "type": self.get_intent_type(&intent_id),
                "phrases": self.get_training_by_lang(&intent_id).cloned().unwrap_or_default(),
                "metadata": self.get_metadata(&intent_id).cloned().unwrap_or_default(),
                "situation_patterns": self.get_situation_patterns(&intent_id).cloned().unwrap_or_default(),
                "learned": learned,
            });

            std::fs::write(&file_path, serde_json::to_string_pretty(&intent_json).unwrap_or_default())
                .map_err(|e| format!("cannot write {}: {}", file_path.display(), e))?;
            written.insert(file_path);
        }

        // Remove stale intent files
        cleanup_stale(path, &written);

        Ok(())
    }
}

/// Split "domain:name" → (Some("domain"), "name") or (None, "full_id").
fn split_intent_id(id: &str) -> (Option<&str>, &str) {
    if let Some(pos) = id.find(':') {
        (Some(&id[..pos]), &id[pos + 1..])
    } else {
        (None, id)
    }
}

/// Deserialize and register one intent file into the router.
/// Errors are logged but do not abort the load.
fn load_intent_file(router: &mut Router, path: &Path, intent_id: &str) {
    let json = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("cannot read {}: {}", path.display(), e); return; }
    };
    let val: serde_json::Value = match serde_json::from_str(&json) {
        Ok(v) => v,
        Err(e) => { eprintln!("invalid JSON in {}: {}", path.display(), e); return; }
    };

    let phrases_by_lang: HashMap<String, Vec<String>> = val.get("phrases")
        .and_then(|p| serde_json::from_value(p.clone()).ok())
        .unwrap_or_default();

    if phrases_by_lang.is_empty() {
        router.add_intent(intent_id, &[]);
    } else {
        router.add_intent_multilingual(intent_id, phrases_by_lang);
    }

    if let Some(desc) = val.get("description").and_then(|d| d.as_str()) {
        if !desc.is_empty() {
            router.set_description(intent_id, desc);
        }
    }

    if let Some(type_str) = val.get("type").and_then(|t| t.as_str()) {
        let intent_type = match type_str {
            "context" => IntentType::Context,
            _ => IntentType::Action,
        };
        router.set_intent_type(intent_id, intent_type);
    }

    if let Some(meta) = val.get("metadata").and_then(|m| m.as_object()) {
        for (key, values) in meta {
            if let Ok(vals) = serde_json::from_value::<Vec<String>>(values.clone()) {
                router.set_metadata(intent_id, key, vals);
            }
        }
    }

    if let Some(patterns) = val.get("situation_patterns")
        .and_then(|p| serde_json::from_value::<Vec<(String, f32)>>(p.clone()).ok())
    {
        if !patterns.is_empty() {
            let refs: Vec<(&str, f32)> = patterns.iter().map(|(s, w)| (s.as_str(), *w)).collect();
            router.add_situation_patterns(intent_id, &refs);
        }
    }

    // Restore learned weights (skip empty map to avoid unnecessary allocation)
    if let Some(learned) = val.get("learned")
        .and_then(|l| serde_json::from_value::<HashMap<String, f32>>(l.clone()).ok())
    {
        if !learned.is_empty() {
            if let Some(vector) = router.vectors.get_mut(intent_id) {
                vector.set_learned_terms(learned);
            }
        }
    }
}

/// Remove `*.json` files in `ns_dir` (and one level of subdirs) not in `written`.
/// Skips files/dirs starting with `_`.
fn cleanup_stale(ns_dir: &Path, written: &HashSet<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(ns_dir) else { return };
    for entry in entries.flatten() {
        let p = entry.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with('_') { continue; }

        if p.is_file() && name.ends_with(".json") && !written.contains(&p) {
            let _ = std::fs::remove_file(&p);
        } else if p.is_dir() {
            let Ok(sub_entries) = std::fs::read_dir(&p) else { continue };
            for sub_entry in sub_entries.flatten() {
                let sp = sub_entry.path();
                let sub_name = sp.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if sub_name.starts_with('_') { continue; }
                if sp.is_file() && sub_name.ends_with(".json") && !written.contains(&sp) {
                    let _ = std::fs::remove_file(&sp);
                }
            }
        }
    }
}
