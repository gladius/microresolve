//! Data types for connected mode: sync protocol, log shipping, metrics.

/// Configuration for connected mode.
#[derive(Debug, Clone)]
pub struct ConnectConfig {
    /// Base URL of the MicroResolve server, e.g. "http://localhost:3001".
    pub server_url: String,
    /// Optional API key sent as X-Api-Key header.
    pub api_key: Option<String>,
    /// App IDs this instance subscribes to.
    pub app_ids: Vec<String>,
    /// How often to poll server for new version (seconds). Default: 30.
    pub sync_interval_secs: u64,
    /// How often to flush log buffer (seconds). Default: 30.
    pub log_flush_secs: u64,
    /// Max log entries buffered before oldest are dropped. Default: 500.
    pub log_buffer_max: usize,
}

impl Default for ConnectConfig {
    fn default() -> Self {
        Self {
            server_url: "http://localhost:3001".to_string(),
            api_key: None,
            app_ids: vec!["default".to_string()],
            sync_interval_secs: 30,
            log_flush_secs: 30,
            log_buffer_max: 500,
        }
    }
}

/// A single routed query, buffered for batch shipping to the server.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogEntry {
    pub query: String,
    pub app_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub detected_intents: Vec<String>,
    /// "high", "medium", "low", or "none"
    pub confidence: String,
    /// "miss", "low_confidence", "false_positive", or None
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flag: Option<String>,
    pub timestamp_ms: u64,
    /// Local router version at routing time — lets server correlate results to model state.
    pub router_version: u64,
}

/// Server response to GET /api/sync.
#[derive(Debug, serde::Deserialize)]
pub struct SyncResponse {
    pub up_to_date: bool,
    pub version: u64,
    /// Full router JSON export, present only when up_to_date = false.
    #[serde(default)]
    pub export: Option<String>,
}
