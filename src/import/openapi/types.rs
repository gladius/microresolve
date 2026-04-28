//! OpenAPI parsed types
//!
//! These types represent the parsed and normalized OpenAPI specification
//! ready for import into the system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Parsed OpenAPI specification ready for import
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParsedSpec {
    pub title: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub servers: Vec<ServerInfo>,
    #[serde(default)]
    pub operations: Vec<ParsedOperation>,
    pub operation_count: usize,
    pub path_count: usize,
    #[serde(default)]
    pub security_schemes: HashMap<String, SecurityScheme>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openapi_version: Option<String>,
    /// Collection variables (from Postman or OpenAPI extensions)
    #[serde(default)]
    pub variables: HashMap<String, String>,
}

/// Server information from OpenAPI spec
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerInfo {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Parsed operation (endpoint) from OpenAPI spec
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParsedOperation {
    /// Unique operation identifier
    pub id: String,
    /// Display name for the operation
    pub name: String,
    /// Operation description
    #[serde(default)]
    pub description: String,
    /// HTTP method (GET, POST, PUT, DELETE, etc.)
    pub method: String,
    /// URL path (e.g., /users/{id})
    pub path: String,
    /// Category (usually first tag or path segment)
    pub category: String,
    /// Tags associated with this operation
    #[serde(default)]
    pub tags: Vec<String>,
    /// Request parameters (query, path, header)
    #[serde(default)]
    pub parameters: Vec<ParsedParameter>,
    /// Request body specification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_body: Option<ParsedRequestBody>,
    /// Response specifications by status code
    #[serde(default)]
    pub responses: HashMap<String, ParsedResponse>,
    /// Security requirements
    #[serde(skip_serializing_if = "Option::is_none")]
    pub security: Option<Vec<HashMap<String, Vec<String>>>>,
    /// Whether this operation is deprecated
    #[serde(default)]
    pub deprecated: bool,
    /// Short summary
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Original operationId from spec
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_id: Option<String>,
}

/// Parsed parameter from OpenAPI spec
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParsedParameter {
    /// Parameter name
    pub name: String,
    /// Location: query, path, header, cookie
    #[serde(rename = "in")]
    pub location: String,
    /// Whether the parameter is required
    #[serde(default)]
    pub required: bool,
    /// Parameter description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Parameter schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
}

/// Parsed request body from OpenAPI spec
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParsedRequestBody {
    /// Whether the body is required
    #[serde(default)]
    pub required: bool,
    /// Content type (e.g., application/json)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Description of the body
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Body schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
}

/// Parsed response from OpenAPI spec
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ParsedResponse {
    /// HTTP status code
    pub status_code: String,
    /// Response description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Content type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Response schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
}

/// Security scheme from OpenAPI spec
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SecurityScheme {
    /// Type: apiKey, http, oauth2, openIdConnect
    #[serde(rename = "type")]
    pub scheme_type: String,
    /// Description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Name of the header/query/cookie (for apiKey)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Location: header, query, cookie (for apiKey)
    #[serde(rename = "in", skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// HTTP auth scheme: bearer, basic (for http type)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheme: Option<String>,
    /// Bearer format hint (for http bearer)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bearer_format: Option<String>,
}

impl ParsedSpec {
    /// Create a new empty parsed spec
    pub fn new(title: String, version: String) -> Self {
        Self {
            title,
            version,
            description: None,
            servers: Vec::new(),
            operations: Vec::new(),
            operation_count: 0,
            path_count: 0,
            security_schemes: HashMap::new(),
            tags: Vec::new(),
            openapi_version: None,
            variables: HashMap::new(),
        }
    }

    /// Update counts after adding operations
    pub fn update_counts(&mut self) {
        self.operation_count = self.operations.len();
        let paths: std::collections::HashSet<_> = self.operations.iter().map(|op| &op.path).collect();
        self.path_count = paths.len();
    }
}

impl ParsedOperation {
    /// Generate a clean pathway name from operation
    pub fn pathway_name(&self) -> String {
        if !self.name.is_empty() {
            clean_pathway_name(&self.name)
        } else {
            clean_pathway_name(&format!("{} {}", self.method, self.path))
        }
    }
}

/// Clean name for pathway display
/// - Converts to kebab-case (lowercase alphanumeric with hyphens)
/// - Removes special characters except spaces and slashes
/// - Replaces / with -- (double hyphen)
/// - Replaces spaces with - (single hyphen)
pub fn clean_pathway_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '/')
        .collect();

    cleaned
        .trim()
        .to_lowercase()
        .replace('/', "--")
        .replace(' ', "-")
        .replace("---", "--")
        .trim_matches('-')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_pathway_name() {
        assert_eq!(clean_pathway_name("Get Users"), "get-users");
        assert_eq!(clean_pathway_name("users/list"), "users--list");
        // POST /api... -> post -api... -> "post--api--v1--users" (--- replaced with --)
        assert_eq!(clean_pathway_name("POST /api/v1/users"), "post--api--v1--users");
        assert_eq!(clean_pathway_name("  Hello World  "), "hello-world");
    }
}
