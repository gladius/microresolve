//! Public types for MicroResolve.

use std::fmt;
use std::path::PathBuf;

/// Global configuration for a `MicroResolve` instance.
///
/// All fields have sensible defaults. Each field is inherited by every
/// namespace unless overridden in a `NamespaceConfig`.
#[derive(Debug, Clone)]
pub struct MicroResolveConfig {
    /// Where namespace data lives. Each namespace is a subdirectory.
    /// `None` means in-memory only (nothing persisted).
    pub data_dir: Option<PathBuf>,

    /// Default resolve threshold, used as the cascade fallback.
    pub default_threshold: f32,

    /// Default languages for phrase generation.
    pub languages: Vec<String>,

    /// Default LLM config for auto-learn / phrase generation.
    /// `None` disables LLM-backed features.
    pub llm: Option<LlmConfig>,

    /// Optional server for live sync. `None` = local-only MicroResolve.
    pub server: Option<ServerConfig>,
}

impl Default for MicroResolveConfig {
    fn default() -> Self {
        Self {
            data_dir: None,
            default_threshold: 0.3,
            languages: vec!["en".to_string()],
            llm: None,
            server: None,
        }
    }
}

/// LLM provider configuration shared across namespaces.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Provider id: "anthropic" | "openai" | "gemini".
    pub provider: String,
    /// Model id, e.g. "claude-haiku-4-5".
    pub model: String,
    /// API key for the provider.
    pub api_key: String,
}

/// Server configuration for connected-mode Engines.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Server base URL, e.g. "https://microresolve.example.com".
    pub url: String,
    /// Optional API key sent as `X-Api-Key`. Required when the server has
    /// auth enabled; ignored in open mode.
    pub api_key: Option<String>,
    /// Namespace IDs to pull from the server on engine startup and keep in
    /// sync via the background tick. Each entry maps 1:1 to a namespace on
    /// the server, accessible locally as `engine.namespace(id)`.
    ///
    /// **Empty Vec = auto-subscribe to all namespaces visible on the server.**
    /// Useful for solo-dev / single-team setups; pass an explicit list when
    /// the server hosts namespaces for multiple tenants and you only want
    /// a subset.
    pub subscribe: Vec<String>,
    /// Polling interval for the background sync tick. Defaults to 30s.
    pub tick_interval_secs: u64,
    /// Maximum buffered log entries before drop-oldest kicks in. Default 500.
    pub log_buffer_max: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            api_key: None,
            subscribe: Vec::new(),
            tick_interval_secs: 30,
            log_buffer_max: 500,
        }
    }
}

/// Per-namespace configuration overrides. Each `Option::None` means
/// "inherit from `MicroResolveConfig`".
#[derive(Debug, Clone, Default)]
pub struct NamespaceConfig {
    pub default_threshold: Option<f32>,
    pub languages: Option<Vec<String>>,
    /// Overrides `EngineConfig.llm.model` when set.
    pub llm_model: Option<String>,
    /// Human-readable description of the namespace.
    pub description: String,
}

/// A routing match: an intent identifier paired with its score.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    /// The intent identifier.
    pub id: String,
    /// Match score (higher = better match).
    pub score: f32,
}

/// Tunable options for `Resolver::resolve_with`.
///
/// `Default::default()` returns sensible defaults that match the zero-arg
/// `resolve()` overload: threshold 0.3, gap 1.5.
#[derive(Debug, Clone, Copy)]
pub struct ResolveOptions {
    /// Minimum score for a match to be returned. Typical range: 0.1–0.5.
    /// Lower = more permissive (more false positives), higher = stricter.
    pub threshold: f32,
    /// Multi-intent gap cutoff. Top score divided by `gap` is the floor for
    /// secondary matches to be reported. Higher values include more matches.
    pub gap: f32,
}

impl Default for ResolveOptions {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            gap: 1.5,
        }
    }
}

/// Errors returned by the public Resolver / MicroResolve / connect API.
#[derive(Debug)]
pub enum Error {
    /// The named intent does not exist in this namespace.
    IntentNotFound(String),
    /// I/O error during persistence (load/save).
    Io(std::io::Error),
    /// Failed to parse persisted data (corrupt file, schema mismatch).
    Parse(String),
    /// Generic persistence error (directory layout, permissions, etc.).
    Persistence(String),
    /// Connected-mode transport / sync error (HTTP failure, server-side
    /// rejection, malformed sync response, etc.).
    Connect(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::IntentNotFound(id) => write!(f, "intent not found: {}", id),
            Error::Io(e) => write!(f, "I/O error: {}", e),
            Error::Parse(s) => write!(f, "parse error: {}", s),
            Error::Persistence(s) => write!(f, "persistence error: {}", s),
            Error::Connect(s) => write!(f, "connect error: {}", s),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Intent type: Action (user wants something done) or Context (supporting info).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IntentType {
    Action,
    Context,
}

/// Where an intent definition came from.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntentSource {
    /// Origin format: mcp | openapi | function | langchain | manual | dialogflow | rasa
    #[serde(rename = "type")]
    pub source_type: String,
    /// Human-readable label (e.g. server name, spec title).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// URL the definition was fetched from or spec base URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

impl IntentSource {
    pub fn new(source_type: impl Into<String>) -> Self {
        Self {
            source_type: source_type.into(),
            label: None,
            url: None,
        }
    }
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }
}

/// Where to send execution when this intent fires.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntentTarget {
    /// Destination type: mcp_server | api_endpoint | model | handler | block
    #[serde(rename = "type")]
    pub target_type: String,
    /// Server or API base URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Model identifier (e.g. "gpt-4o", "claude-opus-4-6").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Named handler in the application layer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handler: Option<String>,
}

impl IntentTarget {
    pub fn new(target_type: impl Into<String>) -> Self {
        Self {
            target_type: target_type.into(),
            url: None,
            model: None,
            handler: None,
        }
    }
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
    pub fn with_handler(mut self, handler: impl Into<String>) -> Self {
        self.handler = Some(handler.into());
        self
    }
}

/// A model entry in the namespace model registry.
/// Users define these per namespace; intents reference them by label.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NamespaceModel {
    /// Display label (e.g. "Fast", "Smart", "Vision")
    pub label: String,
    /// Model ID passed to the LLM provider (e.g. "claude-haiku-4-5", "gpt-4o")
    pub model_id: String,
}

/// Result of checking a phrase before adding it.
#[derive(Debug, Clone)]
pub struct PhraseCheckResult {
    /// Whether the phrase was added.
    pub added: bool,
    /// True if the phrase already exists in this intent.
    pub redundant: bool,
    /// Human-readable warning message, if any.
    pub warning: Option<String>,
}

/// Maximum training phrases per language per intent. Prevents overfitting.
pub const MAX_PHRASES_PER_LANGUAGE: usize = 500;

/// Seed phrases for `Resolver::add_intent`.
///
/// Accepts either a slice of strings (defaults to language `"en"`) or a map
/// from language code → phrases for multilingual seeding. `From` impls let
/// you pass the natural Rust shape directly:
///
/// ```ignore
/// resolver.add_intent("cancel", &["cancel my order", "stop my order"]);
/// resolver.add_intent("cancel", HashMap::from([
///     ("en".to_string(), vec!["cancel".to_string()]),
///     ("fr".to_string(), vec!["annuler".to_string()]),
/// ]));
/// ```
pub enum IntentSeeds {
    /// Seeds in a single language (defaults to `"en"`).
    Mono(Vec<String>),
    /// Seeds grouped by language code.
    Multi(std::collections::HashMap<String, Vec<String>>),
}

impl From<&[&str]> for IntentSeeds {
    fn from(s: &[&str]) -> Self {
        IntentSeeds::Mono(s.iter().map(|x| x.to_string()).collect())
    }
}
impl<const N: usize> From<&[&str; N]> for IntentSeeds {
    fn from(s: &[&str; N]) -> Self {
        IntentSeeds::Mono(s.iter().map(|x| x.to_string()).collect())
    }
}
impl From<Vec<String>> for IntentSeeds {
    fn from(s: Vec<String>) -> Self {
        IntentSeeds::Mono(s)
    }
}
impl From<Vec<&str>> for IntentSeeds {
    fn from(s: Vec<&str>) -> Self {
        IntentSeeds::Mono(s.into_iter().map(|x| x.to_string()).collect())
    }
}
impl From<std::collections::HashMap<String, Vec<String>>> for IntentSeeds {
    fn from(m: std::collections::HashMap<String, Vec<String>>) -> Self {
        IntentSeeds::Multi(m)
    }
}

/// Read-only view of an intent and all its metadata.
///
/// Returned by `Resolver::intent(id)`. All fields are owned (cloned from
/// internal storage) so the view is independent of the Resolver's borrow.
#[derive(Debug, Clone)]
pub struct IntentInfo {
    pub id: String,
    pub intent_type: IntentType,
    pub description: String,
    pub instructions: String,
    pub persona: String,
    pub source: Option<IntentSource>,
    pub target: Option<IntentTarget>,
    pub schema: Option<serde_json::Value>,
    pub guardrails: Vec<String>,
    /// Training phrases grouped by language code.
    pub training: std::collections::HashMap<String, Vec<String>>,
}

/// Read-only view of namespace-level metadata.
///
/// Returned by `Resolver::namespace_info()`.
#[derive(Debug, Clone)]
pub struct NamespaceInfo {
    pub name: String,
    pub description: String,
    pub default_threshold: Option<f32>,
    pub domain_descriptions: std::collections::HashMap<String, String>,
    /// L0 typo correction. Default `true`. Disable for namespaces where
    /// auto-correcting tokens is dangerous (e.g., medical / legal terms,
    /// code identifiers — anything where `cahnge` is more likely a real
    /// term than a typo of `change`).
    pub l0_enabled: bool,
    /// L1 morphological edges (`canceling` → `cancel`). Default `true`.
    pub l1_morphology: bool,
    /// L1 synonym edges (user-defined equivalences). Default `true`.
    pub l1_synonym: bool,
    /// L1 abbreviation edges (`pr` → `pull request`). Default `true`.
    /// Disable for code-search namespaces where short tokens carry literal meaning.
    pub l1_abbreviation: bool,
}

/// Patch for namespace-level metadata via `Resolver::update_namespace`.
///
/// Each field is `Option<T>`: `None` leaves the existing value alone,
/// `Some(_)` overwrites it. Empty string clears name/description.
#[derive(Debug, Clone, Default)]
pub struct NamespaceEdit {
    pub name: Option<String>,
    pub description: Option<String>,
    /// `Some(None)` clears the override; `Some(Some(_))` sets it.
    pub default_threshold: Option<Option<f32>>,
    /// Replaces the entire domain-description map. To delete a single
    /// domain, omit it from the map.
    pub domain_descriptions: Option<std::collections::HashMap<String, String>>,
    pub l0_enabled: Option<bool>,
    pub l1_morphology: Option<bool>,
    pub l1_synonym: Option<bool>,
    pub l1_abbreviation: Option<bool>,
}

/// Patch to apply to an intent's metadata via `Resolver::update_intent`.
///
/// Each field is `Option<T>`: `None` leaves the existing value alone, `Some(_)`
/// overwrites it. To clear a field, pass an empty value (`Some(String::new())`,
/// `Some(vec![])`, etc.).
#[derive(Debug, Clone, Default)]
pub struct IntentEdit {
    pub intent_type: Option<IntentType>,
    pub description: Option<String>,
    pub instructions: Option<String>,
    pub persona: Option<String>,
    pub source: Option<IntentSource>,
    pub target: Option<IntentTarget>,
    pub schema: Option<serde_json::Value>,
    pub guardrails: Option<Vec<String>>,
}
