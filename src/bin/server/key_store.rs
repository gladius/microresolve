//! API key storage for connected-mode auth.
//!
//! Keys live in `~/.config/microresolve/keys.json` (0600 permissions on Unix),
//! deliberately separate from the data dir so they're never git-tracked.
//! `last_used_at` is in-memory only (resets on restart) to avoid file writes
//! on every authenticated request.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;
use rand::RngCore;

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
        let keys = path.as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<Vec<ApiKey>>(&s).ok())
            .unwrap_or_default();
        Self { keys, last_used: RwLock::new(HashMap::new()), path }
    }

    /// True if any keys are configured. Empty = open mode.
    pub fn is_enabled(&self) -> bool { !self.keys.is_empty() }

    /// Validate a key from a request header. Updates last-used on success.
    /// Returns the key's NAME on success (so callers can attribute requests).
    pub fn validate(&self, provided: &str) -> Option<String> {
        let key = self.keys.iter().find(|k| k.id == provided)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.last_used.write().unwrap().insert(provided.to_string(), now);
        Some(key.name.clone())
    }

    /// Listing for UI: returns name, prefix (first 12 chars), created, last_used.
    /// Never returns the full key.
    pub fn list_redacted(&self) -> Vec<RedactedKey> {
        let last = self.last_used.read().unwrap();
        self.keys.iter().map(|k| RedactedKey {
            name: k.name.clone(),
            prefix: k.id.chars().take(12).collect::<String>() + "…",
            created_at: k.created_at,
            last_used_at: last.get(&k.id).copied(),
        }).collect()
    }

    /// Generate a new key with the given name. Returns the full key (caller
    /// must show it to the user once — never retrievable again).
    /// Errors if a key with that name already exists.
    pub fn create(&mut self, name: &str) -> Result<String, String> {
        if self.keys.iter().any(|k| k.name == name) {
            return Err(format!("key with name '{}' already exists", name));
        }
        let mut bytes = [0u8; 32];
        rand::rng().fill_bytes(&mut bytes);
        let id = format!("mr_{}", hex::encode(bytes));
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
        let Some(ref p) = self.path else { return Ok(()); };
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
