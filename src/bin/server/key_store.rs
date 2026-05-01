//! API key storage for connected-mode auth.
//!
//! Keys live in `~/.config/microresolve/keys.json` (0600 permissions on Unix),
//! deliberately separate from the data dir so they're never git-tracked.
//! `last_used_at` is in-memory only (resets on restart) to avoid file writes
//! on every authenticated request.

use rand::RngCore;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ApiKey {
    /// The actual secret. Format: "mr_" + 64 hex chars.
    pub id: String,
    /// Human label, e.g. "prod-app-1".
    pub name: String,
    /// Unix seconds when this key was created.
    pub created_at: u64,
}

pub struct KeyStore {
    keys: Vec<ApiKey>,
    /// In-memory last-used tracking (volatile across restarts).
    last_used: RwLock<HashMap<String, u64>>,
    path: Option<PathBuf>,
}

impl KeyStore {
    /// Load from disk. Falls back to empty store if file missing or unreadable.
    pub fn load() -> Self {
        let path = config_path();
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

    /// True if any keys are configured. Empty = open mode.
    pub fn is_enabled(&self) -> bool {
        !self.keys.is_empty()
    }

    /// Validate a key from a request header. Updates last-used on success.
    /// Returns the key's NAME on success (so callers can attribute requests).
    ///
    /// Keys are formatted `mr_<name>_<hex>` — `validate` requires this shape
    /// (no v0.1.4-and-earlier opaque-key fallback) so the name is recoverable
    /// from the wire without a hash-table lookup. The exact id is still
    /// matched against the persisted store so a leaked key prefix alone
    /// won't authenticate.
    pub fn validate(&self, provided: &str) -> Option<String> {
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
        Some(key.name.clone())
    }

    /// Listing for UI: returns name, prefix (first 12 chars), created, last_used.
    /// Never returns the full key.
    pub fn list_redacted(&self) -> Vec<RedactedKey> {
        let last = self.last_used.read().unwrap();
        self.keys
            .iter()
            .map(|k| RedactedKey {
                name: k.name.clone(),
                prefix: k.id.chars().take(12).collect::<String>() + "…",
                created_at: k.created_at,
                last_used_at: last.get(&k.id).copied(),
            })
            .collect()
    }

    /// Generate a new key with the given name. Returns the full key (caller
    /// must show it to the user once — never retrievable again).
    ///
    /// Format: `mr_<name>_<64 hex chars>`. The name is embedded so the
    /// server can attribute requests to a specific library/client without
    /// hitting the keys.json index — the random hex still provides full
    /// entropy for the secret.
    ///
    /// Name must match `[a-z0-9][a-z0-9-]{0,30}` (slug-safe, no underscore
    /// because that's the field separator in the key string).
    pub fn create(&mut self, name: &str) -> Result<String, String> {
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
        });
        self.save()?;
        Ok(id)
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
        let id = s.create("alex-laptop").unwrap();
        assert!(id.starts_with("mr_alex-laptop_"));
        assert_eq!(parse_name(&id).as_deref(), Some("alex-laptop"));
        assert_eq!(s.validate(&id).as_deref(), Some("alex-laptop"));
    }

    #[test]
    fn rejects_bad_names() {
        let mut s = KeyStore {
            keys: Vec::new(),
            last_used: RwLock::new(HashMap::new()),
            path: None,
        };
        assert!(s.create("HasUpper").is_err());
        assert!(s.create("under_score").is_err());
        assert!(s.create("").is_err());
        assert!(s.create("-leading-dash").is_err());
    }

    #[test]
    fn rejects_malformed_keys() {
        assert!(parse_name("mr_alex_short").is_none());
        assert!(parse_name("not_a_key_at_all").is_none());
        assert!(parse_name("").is_none());
    }
}
