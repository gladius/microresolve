//! Public types for the ASV Router.

/// Router configuration. Pass to `Router::with_config()`.
///
/// ```
/// use asv_router::RouterConfig;
/// let config = RouterConfig { top_k: 5, max_intents: 10, ..Default::default() };
/// ```
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Maximum results from `route()`. Default: 10.
    pub top_k: usize,
    /// Maximum intents from `route_multi()`. Default: 5.
    pub max_intents: usize,
    /// Server URL for connected mode. None = local mode.
    pub server: Option<String>,
    /// App ID for connected mode. Default: "default".
    pub app_id: String,
    /// Local file path for standalone mode. None = in-memory only.
    pub data_path: Option<String>,
    /// Sync interval in seconds (connected mode). Default: 30.
    pub sync_interval_secs: u64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            max_intents: 5,
            server: None,
            app_id: "default".to_string(),
            data_path: None,
            sync_interval_secs: 30,
        }
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
        Self { source_type: source_type.into(), label: None, url: None }
    }
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into()); self
    }
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into()); self
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
        Self { target_type: target_type.into(), url: None, model: None, handler: None }
    }
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into()); self
    }
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into()); self
    }
    pub fn with_handler(mut self, handler: impl Into<String>) -> Self {
        self.handler = Some(handler.into()); self
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

/// A routing result.
#[derive(Debug, Clone)]
pub struct RouteResult {
    /// The intent identifier.
    pub id: String,
    /// Match score (higher = better match).
    pub score: f32,
}

/// A term conflict detected by seed guard.
#[derive(Debug, Clone)]
pub struct TermConflict {
    /// The conflicting term.
    pub term: String,
    /// Intent that primarily owns this term.
    pub competing_intent: String,
    /// Discrimination ratio: what fraction of this term's total weight is in the competing intent.
    pub severity: f32,
    /// The term's weight in the competing intent.
    pub competing_weight: f32,
}

/// Result of checking a phrase before adding it.
#[derive(Debug, Clone)]
pub struct PhraseCheckResult {
    /// Whether the phrase was added.
    pub added: bool,
    /// New terms this phrase introduces (not previously in this intent).
    pub new_terms: Vec<String>,
    /// Terms that conflict with other intents.
    pub conflicts: Vec<TermConflict>,
    /// True if all content terms already exist in this intent.
    pub redundant: bool,
    /// Human-readable warning message, if any.
    pub warning: Option<String>,
}

/// Maximum training phrases per language per intent. Prevents overfitting.
pub const MAX_PHRASES_PER_LANGUAGE: usize = 500;

/// A conflict detected by the situation pattern guard.
#[derive(Debug, Clone)]
pub struct SituationConflict {
    /// The intent that already has this exact pattern.
    pub competing_intent: String,
    /// The weight the pattern has in the competing intent.
    pub competing_weight: f32,
}

/// Result of checking a situation pattern before adding it.
#[derive(Debug, Clone)]
pub struct SituationGuardResult {
    /// Whether the pattern was added.
    pub added: bool,
    /// Other intents that already own this exact pattern.
    pub conflicts: Vec<SituationConflict>,
    /// Pattern already exists in this intent (exact duplicate).
    pub duplicate: bool,
    /// Pattern is too short or generic to be a useful signal.
    pub too_generic: bool,
    /// Human-readable warning, if any.
    pub warning: Option<String>,
}
