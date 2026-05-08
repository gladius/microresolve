//! API key storage for connected-mode auth.
//!
//! Keys live in `~/.config/microresolve/keys.json` (0600 permissions on Unix),
//! deliberately separate from the data dir so they're never git-tracked.
//! `last_used_at` is in-memory only (resets on restart) to avoid file writes
//! on every authenticated request.

use rand::Rng;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

/// Permission scope on an API key.
///
/// Two scopes today:
/// - `Admin` — full access including key management. The bootstrap key
///   the operator pastes into Studio uses this.
/// - `App` — what each application / workload uses to authenticate.
///   Read namespaces, call `/api/resolve`, write to its own audit
///   chain via `/api/audit/event`. Cannot mutate intents/namespaces or
///   manage keys. One App key per workload is the recommended pattern
///   (k8s Deployment, EC2 ASG, etc.) — the chain identity is the key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyScope {
    /// Full access — UI, libraries, key management. Default for the
    /// auto-generated bootstrap key and any explicitly-admin operator key.
    Admin,
    /// Application / workload scope. Read-only against namespace state
    /// plus write-only into its own audit chain. One App key per workload
    /// (a Deployment / service / tenant) — all replicas share it via a
    /// Secret / env var. The chain identity is the workload, not
    /// individual pods.
    App,
}

fn default_scope() -> KeyScope {
    KeyScope::Admin
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ApiKey {
    /// The actual secret. Format: "mr_<name>_<64 hex chars>".
    pub id: String,
    /// Human label, e.g. "prod-app-1".
    pub name: String,
    /// Unix seconds when this key was created.
    pub created_at: u64,
    /// Scope grant — see [`KeyScope`]. Defaults to `Admin` on legacy keys.
    #[serde(default = "default_scope")]
    pub scope: KeyScope,
}

pub struct KeyStore {
    keys: Vec<ApiKey>,
    /// In-memory last-used tracking (volatile across restarts).
    /// Map from bearer token to unix-second timestamp.
    last_used: RwLock<HashMap<String, u64>>,
    path: Option<PathBuf>,
}

impl KeyStore {
    /// Load from disk. Falls back to empty store if file missing or unreadable.
    pub fn load() -> Self {
        Self::load_from(config_path())
    }

    /// Load from a specific path. Used by tests for isolation, by the
    /// `--keys-file` CLI flag, and otherwise as the impl detail of `load()`.
    pub fn load_from(path: Option<PathBuf>) -> Self {
        let keys = path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<Vec<ApiKey>>(&s).ok())
            .unwrap_or_default();
        Self {
            keys,
            last_used: RwLock::new(HashMap::new()),
            path,
        }
    }

    /// True if any keys are configured. After v0.1.9 this is always true on
    /// a started server because boot generates a bootstrap key — kept around
    /// for the brief window during boot before bootstrap and for tests.
    pub fn is_enabled(&self) -> bool {
        !self.keys.is_empty()
    }

    /// Whether the keystore is empty — used at boot time to trigger the
    /// auto-bootstrap path.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Validate a key from a request header. Updates last-used on success.
    /// Returns the matched key's `(name, scope)` so callers can attribute
    /// requests AND eventually enforce per-scope route restrictions.
    pub fn validate(&self, provided: &str) -> Option<(String, KeyScope)> {
        let _ = parse_name(provided)?;
        let key = self.keys.iter().find(|k| k.id == provided)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.last_used
            .write()
            .unwrap()
            .insert(provided.to_string(), now);
        Some((key.name.clone(), key.scope))
    }

    /// Listing for UI: returns name, prefix, scope, created, last_used.
    /// Never returns the full key.
    pub fn list_redacted(&self) -> Vec<RedactedKey> {
        let last = self.last_used.read().unwrap();
        self.keys
            .iter()
            .map(|k| RedactedKey {
                name: k.name.clone(),
                prefix: k.id.chars().take(12).collect::<String>() + "…",
                scope: k.scope,
                created_at: k.created_at,
                last_used_at: last.get(&k.id).copied(),
            })
            .collect()
    }

    /// Generate a new key with the given name and scope. Returns the full key
    /// (caller must show it to the user once — never retrievable again).
    ///
    /// Format: `mr_<name>_<64 hex chars>`. The name is embedded so the
    /// server can attribute requests to a specific library/client without
    /// hitting the keys.json index — the random hex still provides full
    /// entropy for the secret.
    ///
    /// Name must match `[a-z0-9][a-z0-9-]{0,30}` (slug-safe, no underscore
    /// because that's the field separator in the key string).
    pub fn create(&mut self, name: &str, scope: KeyScope) -> Result<String, String> {
        validate_name(name)?;
        if self.keys.iter().any(|k| k.name == name) {
            return Err(format!("key with name '{}' already exists", name));
        }
        let mut bytes = [0u8; 32];
        rand::rng().fill_bytes(&mut bytes);
        let id = format!("mr_{}_{}", name, hex::encode(bytes));
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.keys.push(ApiKey {
            id: id.clone(),
            name: name.to_string(),
            created_at: now,
            scope,
        });
        self.save()?;
        Ok(id)
    }

    /// Bootstrap path: if the keystore is empty, generate `studio-admin`
    /// (Admin scope) and ALSO write the full key to a sibling
    /// `admin-key.txt` file (mode 0600) so the operator can `cat` it later
    /// if they lose the boot stdout. Returns the new key when it just
    /// generated one, `None` if the store already had keys.
    pub fn bootstrap_if_empty(&mut self) -> Result<Option<String>, String> {
        if !self.is_empty() {
            return Ok(None);
        }
        let id = self.create("studio-admin", KeyScope::Admin)?;
        if let Some(ref p) = self.path {
            if let Some(parent) = p.parent() {
                let admin_path = parent.join("admin-key.txt");
                if let Err(e) = std::fs::write(&admin_path, &id) {
                    eprintln!(
                        "[key_store] could not persist admin-key.txt: {} ({})",
                        admin_path.display(),
                        e
                    );
                } else {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        let _ = std::fs::set_permissions(
                            &admin_path,
                            std::fs::Permissions::from_mode(0o600),
                        );
                    }
                }
            }
        }
        Ok(Some(id))
    }

    /// Where the admin-key.txt sidecar lives — surfaced in the boot log so
    /// operators who lost stdout can `cat` it.
    pub fn admin_key_path(&self) -> Option<PathBuf> {
        self.path
            .as_ref()
            .and_then(|p| p.parent().map(|d| d.join("admin-key.txt")))
    }

    /// Revoke a key by name. Returns Err if not found.
    pub fn revoke(&mut self, name: &str) -> Result<(), String> {
        let before = self.keys.len();
        self.keys.retain(|k| k.name != name);
        if self.keys.len() == before {
            return Err(format!("key '{}' not found", name));
        }
        self.save()?;
        Ok(())
    }

    fn save(&self) -> Result<(), String> {
        let Some(ref p) = self.path else {
            return Ok(());
        };
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let json = serde_json::to_string_pretty(&self.keys).map_err(|e| e.to_string())?;
        std::fs::write(p, json).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o600));
        }
        Ok(())
    }
}

#[derive(serde::Serialize)]
pub struct RedactedKey {
    pub name: String,
    pub prefix: String,
    pub scope: KeyScope,
    pub created_at: u64,
    pub last_used_at: Option<u64>,
}

fn config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("sh", "gladius", "microresolve")
        .map(|pd| pd.config_dir().join("keys.json"))
}

/// Names embedded in keys must be slug-safe and underscore-free (since `_`
/// is the field separator). Length capped at 31 to keep the key compact.
fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("name must not be empty".into());
    }
    if name.len() > 31 {
        return Err(format!("name '{}' exceeds 31 chars", name));
    }
    if !name.chars().next().unwrap().is_ascii_alphanumeric() {
        return Err(format!("name '{}' must start with a letter or digit", name));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(format!(
            "name '{}' may only contain lowercase letters, digits, and '-'",
            name
        ));
    }
    Ok(())
}

/// Parse `mr_<name>_<hex>` and return the name slice. Returns None if the
/// key doesn't match the expected shape (wrong prefix, no second `_`,
/// missing hex tail, name fails validation).
fn parse_name(provided: &str) -> Option<String> {
    let rest = provided.strip_prefix("mr_")?;
    let underscore = rest.find('_')?;
    let (name, tail) = rest.split_at(underscore);
    let hex = tail.strip_prefix('_')?;
    // Hex tail should be 64 chars of [0-9a-f] for the random portion.
    if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    if validate_name(name).is_err() {
        return None;
    }
    Some(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_round_trip() {
        let mut s = KeyStore {
            keys: Vec::new(),
            last_used: RwLock::new(HashMap::new()),
            path: None,
        };
        let id = s.create("alex-laptop", KeyScope::App).unwrap();
        assert!(id.starts_with("mr_alex-laptop_"));
        assert_eq!(parse_name(&id).as_deref(), Some("alex-laptop"));
        let validated = s.validate(&id).expect("validates");
        assert_eq!(validated.0, "alex-laptop");
        assert_eq!(validated.1, KeyScope::App);
    }

    #[test]
    fn rejects_bad_names() {
        let mut s = KeyStore {
            keys: Vec::new(),
            last_used: RwLock::new(HashMap::new()),
            path: None,
        };
        assert!(s.create("HasUpper", KeyScope::App).is_err());
        assert!(s.create("under_score", KeyScope::App).is_err());
        assert!(s.create("", KeyScope::App).is_err());
        assert!(s.create("-leading-dash", KeyScope::App).is_err());
    }

    #[test]
    fn bootstrap_creates_admin_key_when_empty() {
        let mut s = KeyStore {
            keys: Vec::new(),
            last_used: RwLock::new(HashMap::new()),
            path: None,
        };
        let key = s.bootstrap_if_empty().unwrap();
        let key = key.expect("returns the new key");
        assert!(key.starts_with("mr_studio-admin_"));
        let validated = s.validate(&key).expect("the bootstrap key validates");
        assert_eq!(validated.0, "studio-admin");
        assert_eq!(validated.1, KeyScope::Admin);
        // Idempotent: a second call with the keystore non-empty is a no-op.
        assert!(s.bootstrap_if_empty().unwrap().is_none());
    }

    #[test]
    fn legacy_keys_default_to_admin() {
        // Keys saved before scopes existed should deserialize as Admin
        // (backwards-compat — they were trusted at creation time).
        let json = r#"[{"id":"mr_legacy_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","name":"legacy","created_at":1700000000}]"#;
        let keys: Vec<ApiKey> = serde_json::from_str(json).unwrap();
        assert_eq!(keys[0].scope, KeyScope::Admin);
    }

    #[test]
    fn rejects_malformed_keys() {
        assert!(parse_name("mr_alex_short").is_none());
        assert!(parse_name("not_a_key_at_all").is_none());
        assert!(parse_name("").is_none());
    }
}
