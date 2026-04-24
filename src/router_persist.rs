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
use std::collections::HashSet;
use std::path::{Path, PathBuf};

impl Router {
    /// Human-readable display name for this namespace.
    pub fn namespace_name(&self) -> &str {
        &self.namespace_name
    }

    /// Set the namespace display name.
    pub fn set_namespace_name(&mut self, name: &str) {
        self.namespace_name = name.to_string();
    }

    /// Human-readable description of this namespace/router instance.
    pub fn namespace_description(&self) -> &str {
        &self.namespace_description
    }

    /// Set the namespace description.
    pub fn set_namespace_description(&mut self, desc: &str) {
        self.namespace_description = desc.to_string();
    }

    /// Default routing threshold for this namespace, if set.
    /// `None` means callers should fall back to their own default.
    pub fn namespace_default_threshold(&self) -> Option<f32> {
        self.namespace_default_threshold
    }

    /// Set the namespace's default routing threshold.
    /// Pass `None` to clear the override.
    pub fn set_namespace_default_threshold(&mut self, threshold: Option<f32>) {
        self.namespace_default_threshold = threshold.map(|t| t.max(0.0));
    }

    /// Per-namespace entity-detection configuration, if set.
    /// `None` means the entity layer is disabled for this namespace.
    pub fn entity_config(&self) -> Option<&crate::entity::EntityConfig> {
        self.namespace_entity_config.as_ref()
    }

    /// Cached EntityLayer for this namespace, pre-built from the config.
    /// Use this in hot paths (route_multi) to avoid the per-request regex
    /// compile cost of `entity_config().build_layer()`.
    pub fn entity_layer(&self) -> Option<&crate::entity::EntityLayer> {
        self.cached_entity_layer.as_ref()
    }

    /// Set (or replace) the namespace's entity-detection configuration.
    /// Rebuilds the cached EntityLayer in lockstep so subsequent
    /// `entity_layer()` calls reflect the new config.
    pub fn set_entity_config(&mut self, config: Option<crate::entity::EntityConfig>) {
        self.cached_entity_layer = config.as_ref().map(|c| c.build_layer());
        self.namespace_entity_config = config;
    }

    /// Rebuild the cached EntityLayer from the current config.
    /// Called by load_from_dir after deserializing the config from disk.
    pub fn rebuild_entity_cache(&mut self) {
        self.cached_entity_layer = self.namespace_entity_config
            .as_ref()
            .map(|c| c.build_layer());
    }

    /// Resolve the effective routing threshold using the standard cascade:
    ///   per-request override (if any) → namespace default (if set) → fallback.
    ///
    /// Centralized here so callers (HTTP server, Node/Python bindings,
    /// embedded users) all apply the same precedence and stay in sync.
    pub fn resolve_threshold(&self, request_override: Option<f32>, fallback: f32) -> f32 {
        request_override
            .or(self.namespace_default_threshold)
            .unwrap_or(fallback)
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
    /// Reads `_ns.json`, `_l1.json`, `_l2.json`, per-domain `_domain.json`, and per-intent `*.json` files.
    /// L0 is rebuilt from L1+L2 vocabulary after load.
    pub fn load_from_dir(path: &Path) -> Result<Self, String> {
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
                if let Some(models) = val.get("models") {
                    if let Ok(m) = serde_json::from_value::<Vec<NamespaceModel>>(models.clone()) {
                        router.namespace_models = m;
                    }
                }
                if let Some(t) = val.get("default_threshold").and_then(|t| t.as_f64()) {
                    router.namespace_default_threshold = Some(t as f32);
                }
            }
        }

        // Per-namespace entity-detection config (optional).
        if let Ok(json) = std::fs::read_to_string(path.join("_entities.json")) {
            if let Ok(cfg) = serde_json::from_str::<crate::entity::EntityConfig>(&json) {
                router.namespace_entity_config = Some(cfg);
            }
        }
        // Build the EntityLayer cache from the loaded config (if any).
        router.rebuild_entity_cache();

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

        // L2 (IntentIndex) — optional, backward compat: start empty if missing.
        // Track whether L2 was pre-loaded so we can skip re-indexing below.
        let l2_preloaded = if let Ok(json) = std::fs::read_to_string(path.join("_l2.json")) {
            if let Ok(ig) = serde_json::from_str::<crate::scoring::IntentIndex>(&json) {
                router.l2 = ig;
                true
            } else { false }
        } else { false };

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
                load_intent_file(&mut router, &p, &stem, l2_preloaded);
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

    /// Save this router to a namespace directory.
    ///
    /// Writes `_ns.json`, per-domain `_domain.json`, and per-intent `*.json` files.
    /// Stale intent files from deleted intents are removed.
    pub fn save_to_dir(&self, path: &Path) -> Result<(), String> {
        std::fs::create_dir_all(path)
            .map_err(|e| format!("cannot create {}: {}", path.display(), e))?;

        // Namespace metadata
        let mut ns_meta = serde_json::json!({
            "name": self.namespace_name,
            "description": self.namespace_description,
            "models": self.namespace_models,
        });
        if let Some(t) = self.namespace_default_threshold {
            ns_meta["default_threshold"] = serde_json::json!(t);
        }
        std::fs::write(
            path.join("_ns.json"),
            serde_json::to_string_pretty(&ns_meta).unwrap_or_default(),
        ).map_err(|e| format!("cannot write _ns.json: {}", e))?;

        let mut written: HashSet<PathBuf> = HashSet::new();
        written.insert(path.join("_ns.json"));

        // Per-namespace entity config (only written when set).
        if let Some(ref cfg) = self.namespace_entity_config {
            let entities_path = path.join("_entities.json");
            std::fs::write(
                &entities_path,
                serde_json::to_string_pretty(cfg).unwrap_or_default(),
            ).map_err(|e| format!("cannot write _entities.json: {}", e))?;
            written.insert(entities_path);
        }

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

            let intent_json = serde_json::json!({
                "description": self.get_description(&intent_id),
                "type": self.get_intent_type(&intent_id),
                "phrases": self.get_training_by_lang(&intent_id).cloned().unwrap_or_default(),
                "instructions": self.get_instructions(&intent_id),
                "persona": self.get_persona(&intent_id),
                "guardrails": self.get_guardrails(&intent_id),
                "source": self.get_source(&intent_id),
                "target": self.get_target(&intent_id),
                "schema": self.get_schema(&intent_id),
            });

            std::fs::write(&file_path, serde_json::to_string_pretty(&intent_json).unwrap_or_default())
                .map_err(|e| format!("cannot write {}: {}", file_path.display(), e))?;
            written.insert(file_path);
        }

        // Save L1 (LexicalGraph)
        if let Ok(json) = serde_json::to_string_pretty(&self.l1) {
            let l1_path = path.join("_l1.json");
            std::fs::write(&l1_path, json)
                .map_err(|e| format!("cannot write _l1.json: {}", e))?;
            written.insert(l1_path);
        }

        // Save L2 (IntentIndex)
        if let Ok(json) = serde_json::to_string_pretty(&self.l2) {
            let l2_path = path.join("_l2.json");
            std::fs::write(&l2_path, json)
                .map_err(|e| format!("cannot write _l2.json: {}", e))?;
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

/// Deserialize and register one intent file into the router.
/// When `skip_indexing` is true (L2 was pre-loaded from _l2.json), only store
/// training data without re-indexing — avoids O(n²) rebuild on startup.
/// Errors are logged but do not abort the load.
fn load_intent_file(router: &mut Router, path: &Path, intent_id: &str, skip_indexing: bool) {
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
        // Just register the intent with no phrases — no indexing needed.
        router.training.insert(intent_id.to_string(), HashMap::new());
        router.version += 1;
    } else if skip_indexing {
        // L2 pre-loaded: store training data only, skip re-indexing.
        router.training.insert(intent_id.to_string(), phrases_by_lang);
        router.version += 1;
    } else {
        // No _l2.json: index all phrases now (migration path for old namespaces).
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

    if let Some(v) = val.get("instructions").and_then(|v| v.as_str()) {
        if !v.is_empty() { router.set_instructions(intent_id, v); }
    }
    if let Some(v) = val.get("persona").and_then(|v| v.as_str()) {
        if !v.is_empty() { router.set_persona(intent_id, v); }
    }
    if let Some(rules) = val.get("guardrails").and_then(|v| v.as_array()) {
        let r: Vec<String> = rules.iter().filter_map(|s| s.as_str().map(String::from)).collect();
        if !r.is_empty() { router.set_guardrails(intent_id, r); }
    }
    if let Some(src) = val.get("source").and_then(|v| serde_json::from_value::<IntentSource>(v.clone()).ok()) {
        router.set_source(intent_id, src);
    }
    if let Some(tgt) = val.get("target").and_then(|v| serde_json::from_value::<IntentTarget>(v.clone()).ok()) {
        router.set_target(intent_id, tgt);
    }
    if let Some(schema) = val.get("schema") {
        if !schema.is_null() { router.set_schema(intent_id, schema.clone()); }
    }

    // "learned" and "metadata" fields are ignored — old format.
    // "learned" weights are now managed by Hebbian L2.
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

#[cfg(test)]
mod tests {
    use crate::Router;

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
        let r = Router::new();
        assert_eq!(r.namespace_default_threshold(), None);
    }

    #[test]
    fn set_default_threshold_persists_in_round_trip() {
        let dir = tmp_dir("threshold_set");
        let mut r = Router::new();
        r.set_namespace_name("test");
        r.set_namespace_default_threshold(Some(1.30));
        r.save_to_dir(&dir).unwrap();

        let r2 = Router::load_from_dir(&dir).unwrap();
        assert_eq!(r2.namespace_default_threshold(), Some(1.30));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unset_default_threshold_omitted_from_disk() {
        let dir = tmp_dir("threshold_unset");
        let r = Router::new();
        r.save_to_dir(&dir).unwrap();

        // Round-trip preserves None.
        let r2 = Router::load_from_dir(&dir).unwrap();
        assert_eq!(r2.namespace_default_threshold(), None);

        // The field should NOT appear in the JSON when unset.
        let json = std::fs::read_to_string(dir.join("_ns.json")).unwrap();
        assert!(!json.contains("default_threshold"),
                "expected _ns.json to omit default_threshold when None, got: {}", json);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn set_default_threshold_clamps_negative_to_zero() {
        let mut r = Router::new();
        r.set_namespace_default_threshold(Some(-5.0));
        // Negative input is clamped to 0.0; 0.0 is a valid (degenerate) setting,
        // distinct from None which means "no override."
        assert_eq!(r.namespace_default_threshold(), Some(0.0));
    }

    #[test]
    fn clearing_default_threshold_via_none() {
        let mut r = Router::new();
        r.set_namespace_default_threshold(Some(0.7));
        assert_eq!(r.namespace_default_threshold(), Some(0.7));
        r.set_namespace_default_threshold(None);
        assert_eq!(r.namespace_default_threshold(), None);
    }

    // ── Cascade resolution ──────────────────────────────────────────────
    // Per-request override > namespace default > caller-supplied fallback.

    #[test]
    fn cascade_request_override_wins() {
        let mut r = Router::new();
        r.set_namespace_default_threshold(Some(1.30));
        // Request explicitly asks for 0.5 — should beat the namespace 1.30.
        assert_eq!(r.resolve_threshold(Some(0.5), 0.3), 0.5);
    }

    #[test]
    fn cascade_namespace_default_used_when_no_request_override() {
        let mut r = Router::new();
        r.set_namespace_default_threshold(Some(1.30));
        // No request threshold — namespace 1.30 should win over fallback 0.3.
        assert_eq!(r.resolve_threshold(None, 0.3), 1.30);
    }

    #[test]
    fn cascade_fallback_used_when_neither_set() {
        let r = Router::new();
        // Nothing set anywhere — fallback wins.
        assert_eq!(r.resolve_threshold(None, 0.3), 0.3);
    }

    #[test]
    fn cascade_request_zero_explicitly_wins_over_namespace() {
        // Tricky case: caller deliberately passes Some(0.0) (accept everything).
        // This must NOT silently fall through to the namespace default.
        let mut r = Router::new();
        r.set_namespace_default_threshold(Some(1.30));
        assert_eq!(r.resolve_threshold(Some(0.0), 0.3), 0.0);
    }

    #[test]
    fn cascade_namespace_zero_wins_over_fallback() {
        // Same principle for the namespace level: Some(0.0) is a real choice.
        let mut r = Router::new();
        r.set_namespace_default_threshold(Some(0.0));
        assert_eq!(r.resolve_threshold(None, 0.3), 0.0);
    }

    #[test]
    fn explicit_zero_threshold_is_preserved_through_round_trip() {
        // Some(0.0) is a valid "accept everything" override and must survive
        // serialization; it must NOT collapse to None.
        let dir = tmp_dir("threshold_zero");
        let mut r = Router::new();
        r.set_namespace_default_threshold(Some(0.0));
        r.save_to_dir(&dir).unwrap();
        let r2 = Router::load_from_dir(&dir).unwrap();
        assert_eq!(r2.namespace_default_threshold(), Some(0.0));
        std::fs::remove_dir_all(&dir).ok();
    }
}
