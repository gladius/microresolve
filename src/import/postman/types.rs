//! Postman Collection types
//!
//! Serde types for deserializing Postman Collection v2.0/v2.1 format.

use serde::{Deserialize, Serialize};

/// Postman Collection root
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanCollection {
    pub info: PostmanInfo,
    #[serde(default)]
    pub item: Vec<PostmanItem>,
    #[serde(default)]
    pub variable: Vec<PostmanVariable>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<PostmanAuth>,
}

/// Collection metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
}

/// Collection item (folder or request)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanItem {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Nested items (for folders)
    #[serde(default)]
    pub item: Vec<PostmanItem>,
    /// Request definition (for requests)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<PostmanRequest>,
}

impl PostmanItem {
    /// Check if this is a folder (has nested items)
    pub fn is_folder(&self) -> bool {
        !self.item.is_empty()
    }

    /// Check if this is a request
    pub fn is_request(&self) -> bool {
        self.request.is_some()
    }
}

/// Postman request definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanRequest {
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<Vec<PostmanHeader>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<PostmanBody>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<PostmanUrl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Postman URL - can be string or object
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PostmanUrl {
    String(String),
    Object(PostmanUrlObject),
}

impl PostmanUrl {
    /// Get the raw URL string
    pub fn raw(&self) -> Option<&str> {
        match self {
            PostmanUrl::String(s) => Some(s),
            PostmanUrl::Object(obj) => obj.raw.as_deref(),
        }
    }

    /// Get path segments
    pub fn path(&self) -> Option<&[String]> {
        match self {
            PostmanUrl::String(_) => None,
            PostmanUrl::Object(obj) => Some(&obj.path),
        }
    }

    /// Get query parameters
    pub fn query(&self) -> &[PostmanQueryParam] {
        match self {
            PostmanUrl::String(_) => &[],
            PostmanUrl::Object(obj) => &obj.query,
        }
    }

    /// Get path variables
    pub fn variable(&self) -> &[PostmanVariable] {
        match self {
            PostmanUrl::String(_) => &[],
            PostmanUrl::Object(obj) => &obj.variable,
        }
    }
}

/// Postman URL object format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanUrlObject {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
    #[serde(default)]
    pub host: Vec<String>,
    #[serde(default)]
    pub path: Vec<String>,
    #[serde(default)]
    pub query: Vec<PostmanQueryParam>,
    #[serde(default)]
    pub variable: Vec<PostmanVariable>,
}

/// Query parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanQueryParam {
    pub key: String,
    #[serde(default)]
    pub value: String,
    #[serde(default)]
    pub disabled: bool,
}

/// Header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanHeader {
    pub key: String,
    pub value: String,
    #[serde(default)]
    pub disabled: bool,
}

/// Request body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanBody {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
    #[serde(default)]
    pub urlencoded: Vec<PostmanFormParam>,
    #[serde(default)]
    pub formdata: Vec<PostmanFormParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<PostmanBodyOptions>,
}

/// Body options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanBodyOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<PostmanRawOptions>,
}

/// Raw body options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanRawOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

/// Form parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanFormParam {
    pub key: String,
    #[serde(default)]
    pub value: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub param_type: Option<String>,
    #[serde(default)]
    pub disabled: bool,
}

/// Collection variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanVariable {
    pub key: String,
    #[serde(default)]
    pub value: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub var_type: Option<String>,
}

/// Collection-level auth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanAuth {
    #[serde(rename = "type")]
    pub auth_type: String,
    #[serde(default)]
    pub bearer: Vec<PostmanAuthKeyValue>,
    #[serde(default)]
    pub apikey: Vec<PostmanAuthKeyValue>,
    #[serde(default)]
    pub basic: Vec<PostmanAuthKeyValue>,
}

/// Auth key-value pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostmanAuthKeyValue {
    pub key: String,
    pub value: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub value_type: Option<String>,
}
