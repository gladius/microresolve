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
pub const MAX_PHRASES_PER_LANGUAGE: usize = 20;

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
