//! Directory-based persistence for Resolver.
//!
//! `save_to_dir(path)` writes one JSON file per intent, plus `_ns.json` and
//! `_domain.json` sidecar files. `load_from_dir(path)` reconstructs the engine
//! from those files. The phrase-layer vector is always recomputed from phrases;
//! the learned layer is saved and restored separately.
//!
//! File layout:
//!   {ns_dir}/_ns.json                     — namespace description
//!   {ns_dir}/{intent_name}.json           — intent without domain prefix
//!   {ns_dir}/{domain}/_domain.json        — domain description
//!   {ns_dir}/{domain}/{intent_name}.json  — intent with domain prefix

use crate::types::IntentType;
use crate::*;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

impl Resolver {
    /// Read all namespace-level metadata.
    pub fn namespace_info(&self) -> NamespaceInfo {
        NamespaceInfo {
            name: self.namespace_name.clone(),
            description: self.namespace_description.clone(),
            default_threshold: self.namespace_default_threshold,
            domain_descriptions: self.domain_descriptions.clone(),
        }
    }

    /// Update one or more namespace-level metadata fields.
    ///
    /// Each field of `NamespaceEdit` is `Option<T>`: `None` leaves the value
    /// alone, `Some(_)` overwrites it. To clear `default_threshold`, pass
    /// `Some(None)`.
    ///
    /// Returns `Result` for symmetry with `update_intent` so future validation
    /// paths can surface errors without breaking callers.
    pub fn update_namespace(&mut self, edit: NamespaceEdit) -> Result<(), Error> {
        if let Some(n) = edit.name {
            self.namespace_name = n;
        }
        if let Some(d) = edit.description {
            self.namespace_description = d;
        }
        if let Some(t) = edit.default_threshold {
            self.namespace_default_threshold = t.map(|t| t.max(0.0));
        }
        if let Some(dd) = edit.domain_descriptions {
            self.domain_descriptions = dd;
        }
        Ok(())
    }

    /// Resolve the effective routing threshold using the standard cascade:
    ///   per-request override (if any) → namespace default (if set) → fallback.
    pub fn resolve_threshold(&self, request_override: Option<f32>, fallback: f32) -> f32 {
        request_override
            .or(self.namespace_default_threshold)
            .unwrap_or(fallback)
    }

    /// Description for a specific domain prefix (e.g., "billing"). `None` if not set.
    pub fn domain_description(&self, domain: &str) -> Option<&str> {
        self.domain_descriptions.get(domain).map(|s| s.as_str())
    }

    /// Set the description for a domain prefix.
    pub fn set_domain_description(&mut self, domain: &str, desc: &str) {
        self.domain_descriptions
            .insert(domain.to_string(), desc.to_string());
    }

    /// Remove a domain description (does not remove intents).
    pub fn remove_domain_description(&mut self, domain: &str) {
        self.domain_descriptions.remove(domain);
    }

    /// Load a router from a namespace directory.
    ///
    /// Reads `_ns.json`, `_l1.json`, `_l2.json`, per-domain `_domain.json`, and per-intent `*.json` files.
    /// L0 is rebuilt from L1+L2 vocabulary after load.
    pub fn load_from_dir(path: &Path) -> Result<Self, crate::Error> {
        let mut router = Self::new();

        // Namespace metadata
        if let Ok(json) = std::fs::read_to_string(path.join("_ns.json")) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json) {
                if let Some(name) = val.get("name").and_then(|d| d.as_str()) {
                    router.namespace_name = name.to_string();
                }
                if let Some(desc) = val.get("description").and_then(|d| d.as_str()) {
                    router.namespace_description = desc.to_string();
                }
                // Note: legacy "models" field on _ns.json is ignored; the model
                // registry is now instance-wide (see UiSettings::models).
                if let Some(t) = val.get("default_threshold").and_then(|t| t.as_f64()) {
                    router.namespace_default_threshold = Some(t as f32);
                }
            }
        }

        // L1 (LexicalGraph) — seed with global English morphology base, then overlay namespace-specific edges.
        let base = crate::scoring::english_morphology_base();
        for (from, edges) in base.edges {
            for edge in edges {
                let existing = router.l1.edges.entry(from.clone()).or_default();
                if !existing.iter().any(|e| e.target == edge.target) {
                    existing.push(edge);
                }
            }
        }
        if let Ok(json) = std::fs::read_to_string(path.join("_l1.json")) {
            if let Ok(g) = serde_json::from_str::<crate::scoring::LexicalGraph>(&json) {
                // Merge namespace-specific edges on top of global base (namespace wins on conflict)
                for (from, edges) in g.edges {
                    let existing = router.l1.edges.entry(from).or_default();
                    for edge in edges {
                        if let Some(e) = existing.iter_mut().find(|e| e.target == edge.target) {
                            *e = edge; // namespace overrides global
                        } else {
                            existing.push(edge);
                        }
                    }
                }
            }
        }

        // L2 (IntentIndex) — track whether L2 was pre-loaded so we can skip
        // re-indexing below.
        let l2_preloaded = if let Ok(json) = std::fs::read_to_string(path.join("_l2.json")) {
            if let Ok(ig) = serde_json::from_str::<crate::scoring::IntentIndex>(&json) {
                router.l2 = ig;
                true
            } else {
                false
            }
        } else {
            false
        };

        let entries = std::fs::read_dir(path).map_err(|e| {
            crate::Error::Persistence(format!("cannot read {}: {}", path.display(), e))
        })?;

        let mut domain_dirs: Vec<(String, PathBuf)> = Vec::new();

        for entry in entries.flatten() {
            let p = entry.path();
            let name = match p.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            if name.starts_with('_') {
                continue;
            }

            if p.is_dir() {
                domain_dirs.push((name, p));
            } else if p.extension().map(|e| e == "json").unwrap_or(false) {
                let stem = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                load_intent_file(&mut router, &p, &stem, l2_preloaded);
            }
        }

        for (domain, domain_dir) in &domain_dirs {
            // Domain description
            if let Ok(json) = std::fs::read_to_string(domain_dir.join("_domain.json")) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json) {
                    if let Some(desc) = val.get("description").and_then(|d| d.as_str()) {
                        router
                            .domain_descriptions
                            .insert(domain.clone(), desc.to_string());
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
                    if sub_name.starts_with('_') {
                        continue;
                    }
                    if p.extension().map(|e| e == "json").unwrap_or(false) {
                        let stem = p
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_string();
                        let intent_id = format!("{}:{}", domain, stem);
                        load_intent_file(&mut router, &p, &intent_id, l2_preloaded);
                    }
                }
            }
        }

        // L0 is always rebuilt once from L1+L2 vocabulary — no separate file needed.
        router.rebuild_l0();
        // Rebuild IDF cache from the loaded posting lists — O(words) once on load.
        router.l2.rebuild_idf();

        Ok(router)
    }

    /// Save this namespace to a directory.
    ///
    /// Writes `_ns.json`, per-domain `_domain.json`, and per-intent `*.json` files.
    /// Stale intent files from deleted intents are removed.
    pub fn save_to_dir(&self, path: &Path) -> Result<(), crate::Error> {
        std::fs::create_dir_all(path).map_err(|e| {
            crate::Error::Persistence(format!("cannot create {}: {}", path.display(), e))
        })?;

        // Namespace metadata
        let mut ns_meta = serde_json::json!({
            "name": self.namespace_name,
            "description": self.namespace_description,
        });
        if let Some(t) = self.namespace_default_threshold {
            ns_meta["default_threshold"] = serde_json::json!(t);
        }
        std::fs::write(
            path.join("_ns.json"),
            serde_json::to_string_pretty(&ns_meta).unwrap_or_default(),
        )
        .map_err(|e| crate::Error::Persistence(format!("cannot write _ns.json: {}", e)))?;

        let mut written: HashSet<PathBuf> = HashSet::new();
        written.insert(path.join("_ns.json"));

        // Write domain descriptions (including explicitly-created domains with no intents)
        for (domain, desc) in &self.domain_descriptions {
            let domain_dir = path.join(domain);
            std::fs::create_dir_all(&domain_dir).map_err(|e| {
                crate::Error::Persistence(format!("cannot create domain dir {}: {}", domain, e))
            })?;
            let meta = serde_json::json!({"description": desc});
            let meta_path = domain_dir.join("_domain.json");
            std::fs::write(
                &meta_path,
                serde_json::to_string_pretty(&meta).unwrap_or_default(),
            )
            .map_err(|e| {
                crate::Error::Persistence(format!(
                    "cannot write _domain.json for {}: {}",
                    domain, e
                ))
            })?;
            written.insert(meta_path);
        }

        // Write per-intent files
        for intent_id in self.intent_ids() {
            let (domain_opt, name) = split_intent_id(&intent_id);

            let file_path = if let Some(domain) = domain_opt {
                let domain_dir = path.join(domain);
                std::fs::create_dir_all(&domain_dir).map_err(|e| {
                    crate::Error::Persistence(format!("cannot create domain dir: {}", e))
                })?;
                // Ensure _domain.json exists for intent-derived domains
                let meta_path = domain_dir.join("_domain.json");
                if !written.contains(&meta_path) {
                    let desc = self
                        .domain_descriptions
                        .get(domain)
                        .cloned()
                        .unwrap_or_default();
                    let meta = serde_json::json!({"description": desc});
                    std::fs::write(
                        &meta_path,
                        serde_json::to_string_pretty(&meta).unwrap_or_default(),
                    )
                    .ok();
                    written.insert(meta_path);
                }
                domain_dir.join(format!("{}.json", name))
            } else {
                path.join(format!("{}.json", name))
            };

            let info = self.intent(&intent_id);
            let intent_json = serde_json::json!({
                "description": info.as_ref().map(|i| i.description.as_str()).unwrap_or(""),
                "type": info.as_ref().map(|i| i.intent_type).unwrap_or(IntentType::Action),
                "phrases": self.training_by_lang(&intent_id).cloned().unwrap_or_default(),
                "instructions": info.as_ref().map(|i| i.instructions.as_str()).unwrap_or(""),
                "persona": info.as_ref().map(|i| i.persona.as_str()).unwrap_or(""),
                "guardrails": info.as_ref().map(|i| i.guardrails.clone()).unwrap_or_default(),
                "source": info.as_ref().and_then(|i| i.source.clone()),
                "target": info.as_ref().and_then(|i| i.target.clone()),
                "schema": info.as_ref().and_then(|i| i.schema.clone()),
            });

            std::fs::write(
                &file_path,
                serde_json::to_string_pretty(&intent_json).unwrap_or_default(),
            )
            .map_err(|e| {
                crate::Error::Persistence(format!("cannot write {}: {}", file_path.display(), e))
            })?;
            written.insert(file_path);
        }

        // Save L1 (LexicalGraph)
        if let Ok(json) = serde_json::to_string_pretty(&self.l1) {
            let l1_path = path.join("_l1.json");
            std::fs::write(&l1_path, json)
                .map_err(|e| crate::Error::Persistence(format!("cannot write _l1.json: {}", e)))?;
            written.insert(l1_path);
        }

        // Save L2 (IntentIndex)
        if let Ok(json) = serde_json::to_string_pretty(&self.l2) {
            let l2_path = path.join("_l2.json");
            std::fs::write(&l2_path, json)
                .map_err(|e| crate::Error::Persistence(format!("cannot write _l2.json: {}", e)))?;
            written.insert(l2_path);
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

/// Deserialize and register one intent file into the engine.
/// When `skip_indexing` is true (L2 was pre-loaded from _l2.json), only store
/// training data without re-indexing — avoids O(n²) rebuild on startup.
/// Errors are logged but do not abort the load.
fn load_intent_file(router: &mut Resolver, path: &Path, intent_id: &str, skip_indexing: bool) {
    let json = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {}: {}", path.display(), e);
            return;
        }
    };
    let val: serde_json::Value = match serde_json::from_str(&json) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("invalid JSON in {}: {}", path.display(), e);
            return;
        }
    };

    let phrases_by_lang: HashMap<String, Vec<String>> = val
        .get("phrases")
        .and_then(|p| serde_json::from_value(p.clone()).ok())
        .unwrap_or_default();

    if phrases_by_lang.is_empty() {
        // Just register the intent with no phrases — no indexing needed.
        router
            .training
            .insert(intent_id.to_string(), HashMap::new());
        router.version += 1;
    } else if skip_indexing {
        // L2 pre-loaded: store training data only, skip re-indexing.
        router
            .training
            .insert(intent_id.to_string(), phrases_by_lang);
        router.version += 1;
    } else {
        // No _l2.json: index all phrases now (migration path for old namespaces).
        let _ = router.add_intent(intent_id, phrases_by_lang);
    }

    let edit = crate::IntentEdit {
        intent_type: val.get("type").and_then(|t| t.as_str()).map(|s| match s {
            "context" => IntentType::Context,
            _ => IntentType::Action,
        }),
        description: val
            .get("description")
            .and_then(|d| d.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from),
        instructions: val
            .get("instructions")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from),
        persona: val
            .get("persona")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from),
        guardrails: val
            .get("guardrails")
            .and_then(|v| v.as_array())
            .and_then(|rules| {
                let r: Vec<String> = rules
                    .iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect();
                if r.is_empty() {
                    None
                } else {
                    Some(r)
                }
            }),
        source: val
            .get("source")
            .and_then(|v| serde_json::from_value::<IntentSource>(v.clone()).ok()),
        target: val
            .get("target")
            .and_then(|v| serde_json::from_value::<IntentTarget>(v.clone()).ok()),
        schema: val.get("schema").filter(|s| !s.is_null()).cloned(),
    };
    let _ = router.update_intent(intent_id, edit);
}

/// Remove `*.json` files in `ns_dir` (and one level of subdirs) not in `written`.
/// Skips files/dirs starting with `_`.
fn cleanup_stale(ns_dir: &Path, written: &HashSet<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(ns_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with('_') {
            continue;
        }

        if p.is_file() && name.ends_with(".json") && !written.contains(&p) {
            let _ = std::fs::remove_file(&p);
        } else if p.is_dir() {
            let Ok(sub_entries) = std::fs::read_dir(&p) else {
                continue;
            };
            for sub_entry in sub_entries.flatten() {
                let sp = sub_entry.path();
                let sub_name = sp.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if sub_name.starts_with('_') {
                    continue;
                }
                if sp.is_file() && sub_name.ends_with(".json") && !written.contains(&sp) {
                    let _ = std::fs::remove_file(&sp);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Resolver;

    /// Generate a unique tmp directory under /tmp for each test.
    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::path::PathBuf::from(format!("/tmp/microresolve_test_{}_{}", tag, nanos));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn default_threshold_starts_unset() {
        let r = Resolver::new();
        assert_eq!(r.namespace_info().default_threshold, None);
    }

    #[test]
    fn set_default_threshold_persists_in_round_trip() {
        let dir = tmp_dir("threshold_set");
        let mut r = Resolver::new();
        r.update_namespace(crate::NamespaceEdit {
            name: Some("test".to_string()),
            default_threshold: Some(Some(1.30)),
            ..Default::default()
        })
        .unwrap();
        r.save_to_dir(&dir).unwrap();

        let r2 = Resolver::load_from_dir(&dir).unwrap();
        assert_eq!(r2.namespace_info().default_threshold, Some(1.30));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unset_default_threshold_omitted_from_disk() {
        let dir = tmp_dir("threshold_unset");
        let r = Resolver::new();
        r.save_to_dir(&dir).unwrap();

        // Round-trip preserves None.
        let r2 = Resolver::load_from_dir(&dir).unwrap();
        assert_eq!(r2.namespace_info().default_threshold, None);

        // The field should NOT appear in the JSON when unset.
        let json = std::fs::read_to_string(dir.join("_ns.json")).unwrap();
        assert!(
            !json.contains("default_threshold"),
            "expected _ns.json to omit default_threshold when None, got: {}",
            json
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn set_default_threshold_clamps_negative_to_zero() {
        let mut r = Resolver::new();
        r.update_namespace(crate::NamespaceEdit {
            default_threshold: Some(Some(-5.0)),
            ..Default::default()
        })
        .unwrap();
        // Negative input is clamped to 0.0; 0.0 is a valid (degenerate) setting,
        // distinct from None which means "no override."
        assert_eq!(r.namespace_info().default_threshold, Some(0.0));
    }

    #[test]
    fn clearing_default_threshold_via_none() {
        let mut r = Resolver::new();
        r.update_namespace(crate::NamespaceEdit {
            default_threshold: Some(Some(0.7)),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(r.namespace_info().default_threshold, Some(0.7));
        r.update_namespace(crate::NamespaceEdit {
            default_threshold: Some(None),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(r.namespace_info().default_threshold, None);
    }

    // ── Cascade resolution ──────────────────────────────────────────────
    // Per-request override > namespace default > caller-supplied fallback.

    #[test]
    fn cascade_request_override_wins() {
        let mut r = Resolver::new();
        r.update_namespace(crate::NamespaceEdit {
            default_threshold: Some(Some(1.30)),
            ..Default::default()
        })
        .unwrap();
        // Request explicitly asks for 0.5 — should beat the namespace 1.30.
        assert_eq!(r.resolve_threshold(Some(0.5), 0.3), 0.5);
    }

    #[test]
    fn cascade_namespace_default_used_when_no_request_override() {
        let mut r = Resolver::new();
        r.update_namespace(crate::NamespaceEdit {
            default_threshold: Some(Some(1.30)),
            ..Default::default()
        })
        .unwrap();
        // No request threshold — namespace 1.30 should win over fallback 0.3.
        assert_eq!(r.resolve_threshold(None, 0.3), 1.30);
    }

    #[test]
    fn cascade_fallback_used_when_neither_set() {
        let r = Resolver::new();
        // Nothing set anywhere — fallback wins.
        assert_eq!(r.resolve_threshold(None, 0.3), 0.3);
    }

    #[test]
    fn cascade_request_zero_explicitly_wins_over_namespace() {
        // Tricky case: caller deliberately passes Some(0.0) (accept everything).
        // This must NOT silently fall through to the namespace default.
        let mut r = Resolver::new();
        r.update_namespace(crate::NamespaceEdit {
            default_threshold: Some(Some(1.30)),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(r.resolve_threshold(Some(0.0), 0.3), 0.0);
    }

    #[test]
    fn cascade_namespace_zero_wins_over_fallback() {
        // Same principle for the namespace level: Some(0.0) is a real choice.
        let mut r = Resolver::new();
        r.update_namespace(crate::NamespaceEdit {
            default_threshold: Some(Some(0.0)),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(r.resolve_threshold(None, 0.3), 0.0);
    }

    #[test]
    fn explicit_zero_threshold_is_preserved_through_round_trip() {
        // Some(0.0) is a valid "accept everything" override and must survive
        // serialization; it must NOT collapse to None.
        let dir = tmp_dir("threshold_zero");
        let mut r = Resolver::new();
        r.update_namespace(crate::NamespaceEdit {
            default_threshold: Some(Some(0.0)),
            ..Default::default()
        })
        .unwrap();
        r.save_to_dir(&dir).unwrap();
        let r2 = Resolver::load_from_dir(&dir).unwrap();
        assert_eq!(r2.namespace_info().default_threshold, Some(0.0));
        std::fs::remove_dir_all(&dir).ok();
    }
}
