//! Postman to ParsedSpec converter
//!
//! Converts Postman Collection v2.0/v2.1 format to our ParsedSpec format.

use std::collections::HashMap;

// No-op logging macros
macro_rules! info { ($($t:tt)*) => {} }
macro_rules! debug { ($($t:tt)*) => {} }

use super::types::*;
use crate::import::openapi::{
    clean_pathway_name, ParsedOperation, ParsedParameter, ParsedRequestBody,
    ParsedSpec, SecurityScheme,
};

/// Convert a Postman collection to our ParsedSpec format
pub fn convert_postman(collection: &PostmanCollection) -> Result<ParsedSpec, ConvertError> {
    info!(
        name = %collection.info.name,
        items = collection.item.len(),
        "📦 POSTMAN_CONVERT - Starting conversion"
    );

    let mut spec = ParsedSpec::new(
        collection.info.name.clone(),
        collection.info.version.clone().unwrap_or_else(|| "1.0.0".to_string()),
    );

    spec.description = collection.info.description.clone();

    // Extract collection variables
    for var in &collection.variable {
        spec.variables.insert(var.key.clone(), var.value.clone());
    }

    // Extract base URL from variables or first request
    if let Some(base_url) = extract_base_url(collection) {
        spec.servers.push(crate::import::openapi::ServerInfo {
            url: base_url,
            description: None,
        });
    }

    // Extract security schemes from auth
    extract_security_schemes(&collection.auth, &mut spec.security_schemes);

    // If no auth defined, add default bearer
    if spec.security_schemes.is_empty() {
        spec.security_schemes.insert(
            "bearerAuth".to_string(),
            SecurityScheme {
                scheme_type: "http".to_string(),
                description: Some("Inferred from Postman collection".to_string()),
                name: None,
                location: None,
                scheme: Some("bearer".to_string()),
                bearer_format: None,
            },
        );
    }

    // Extract operations from items
    let mut tags = Vec::new();
    extract_operations(&collection.item, &mut spec.operations, &mut tags, "", &spec.variables);

    spec.tags = tags;
    spec.update_counts();

    info!(
        operations = spec.operation_count,
        paths = spec.path_count,
        "✅ POSTMAN_CONVERT - Conversion complete"
    );

    Ok(spec)
}

/// Extract base URL from collection
fn extract_base_url(collection: &PostmanCollection) -> Option<String> {
    // Try collection variables first
    for var in &collection.variable {
        if var.key == "baseUrl" || var.key == "base_url" || var.key == "host" {
            if !var.value.is_empty() {
                return Some(var.value.clone());
            }
        }
    }

    // Try to extract from first request
    if let Some(first_request) = find_first_request(&collection.item) {
        if let Some(url) = &first_request.url {
            if let Some(raw) = url.raw() {
                // Try to extract protocol://host from URL
                if let Ok(parsed) = url::Url::parse(&raw.replace("{{", "").replace("}}", "")) {
                    return Some(format!("{}://{}", parsed.scheme(), parsed.host_str()?));
                }
            }
        }
    }

    None
}

/// Find first request in items (recursively)
fn find_first_request(items: &[PostmanItem]) -> Option<&PostmanRequest> {
    for item in items {
        if let Some(ref request) = item.request {
            return Some(request);
        }
        if let Some(found) = find_first_request(&item.item) {
            return Some(found);
        }
    }
    None
}

/// Extract security schemes from collection auth
fn extract_security_schemes(
    auth: &Option<PostmanAuth>,
    schemes: &mut HashMap<String, SecurityScheme>,
) {
    let Some(auth) = auth else { return };

    match auth.auth_type.as_str() {
        "bearer" => {
            schemes.insert(
                "bearerAuth".to_string(),
                SecurityScheme {
                    scheme_type: "http".to_string(),
                    description: None,
                    name: None,
                    location: None,
                    scheme: Some("bearer".to_string()),
                    bearer_format: None,
                },
            );
        }
        "apikey" => {
            let location = auth
                .apikey
                .iter()
                .find(|kv| kv.key == "in")
                .map(|kv| kv.value.clone())
                .unwrap_or_else(|| "header".to_string());
            let name = auth
                .apikey
                .iter()
                .find(|kv| kv.key == "key")
                .map(|kv| kv.value.clone())
                .unwrap_or_else(|| "X-API-Key".to_string());

            schemes.insert(
                "apiKeyAuth".to_string(),
                SecurityScheme {
                    scheme_type: "apiKey".to_string(),
                    description: None,
                    name: Some(name),
                    location: Some(location),
                    scheme: None,
                    bearer_format: None,
                },
            );
        }
        "basic" => {
            schemes.insert(
                "basicAuth".to_string(),
                SecurityScheme {
                    scheme_type: "http".to_string(),
                    description: None,
                    name: None,
                    location: None,
                    scheme: Some("basic".to_string()),
                    bearer_format: None,
                },
            );
        }
        _ => {
            debug!(auth_type = %auth.auth_type, "📦 POSTMAN_CONVERT - Unknown auth type");
        }
    }
}

/// Extract operations from Postman items recursively
fn extract_operations(
    items: &[PostmanItem],
    operations: &mut Vec<ParsedOperation>,
    tags: &mut Vec<String>,
    parent_tag: &str,
    variables: &HashMap<String, String>,
) {
    for item in items {
        if item.is_folder() {
            // Folder - add as tag and process children
            let tag = item.name.clone();
            if !tags.contains(&tag) {
                tags.push(tag.clone());
            }
            extract_operations(&item.item, operations, tags, &tag, variables);
        } else if let Some(ref request) = item.request {
            // Request - convert to operation
            if let Some(op) = convert_request(item, request, parent_tag, variables) {
                operations.push(op);
            }
        }
    }
}

/// Convert a Postman request to ParsedOperation
fn convert_request(
    item: &PostmanItem,
    request: &PostmanRequest,
    tag: &str,
    variables: &HashMap<String, String>,
) -> Option<ParsedOperation> {
    let method = request.method.to_uppercase();
    let path = extract_path(&request.url)?;

    // Generate operation ID
    let sanitized_path = path
        .trim_start_matches('/')
        .replace('{', "")
        .replace('}', "")
        .replace(|c: char| !c.is_alphanumeric(), "_")
        .to_lowercase();
    let operation_id = if sanitized_path.is_empty() {
        format!("{}_root", method.to_lowercase())
    } else {
        format!("{}_{}", method.to_lowercase(), sanitized_path)
    };

    // Generate clean display name
    let clean_tag = if tag.is_empty() {
        String::new()
    } else {
        clean_pathway_name(tag)
    };
    let clean_name = clean_pathway_name(&item.name);
    let display_name = if clean_tag.is_empty() {
        clean_name
    } else {
        format!("{}--{}", clean_tag, clean_name)
    };

    // Extract parameters
    let parameters = extract_parameters(request, &path, variables);

    // Extract request body
    let request_body = extract_request_body(request);

    // Check for auth header
    let has_auth = request
        .header
        .as_ref()
        .map(|h| h.iter().any(|hdr| hdr.key.to_lowercase() == "authorization"))
        .unwrap_or(false);

    Some(ParsedOperation {
        id: operation_id.clone(),
        name: display_name,
        description: item
            .description
            .clone()
            .or_else(|| request.description.clone())
            .unwrap_or_default(),
        method,
        path,
        category: if tag.is_empty() {
            "uncategorized".to_string()
        } else {
            tag.to_string()
        },
        tags: if tag.is_empty() {
            vec![]
        } else {
            vec![tag.to_string()]
        },
        parameters,
        request_body,
        responses: HashMap::new(), // Postman doesn't have response schemas
        security: if has_auth {
            Some(vec![{
                let mut m = HashMap::new();
                m.insert("bearerAuth".to_string(), vec![]);
                m
            }])
        } else {
            None
        },
        deprecated: false,
        summary: Some(item.name.clone()),
        operation_id: Some(operation_id),
    })
}

/// Extract path from Postman URL
fn extract_path(url: &Option<PostmanUrl>) -> Option<String> {
    let url = url.as_ref()?;

    match url {
        PostmanUrl::String(s) => {
            // Try to parse as URL
            let cleaned = s.replace("{{", "").replace("}}", "placeholder");
            if let Ok(parsed) = url::Url::parse(&cleaned) {
                let path = parsed.path().to_string();
                // Restore {{var}} syntax as {var} for OpenAPI
                Some(restore_path_params(s, &path))
            } else {
                // Just extract path portion
                let path = s.split('?').next().unwrap_or(s);
                let path = path
                    .trim_start_matches("http://")
                    .trim_start_matches("https://");
                let path = path.split('/').skip(1).collect::<Vec<_>>().join("/");
                Some(format!("/{}", restore_path_params(s, &path)))
            }
        }
        PostmanUrl::Object(obj) => {
            if obj.path.is_empty() {
                return obj.raw.as_ref().and_then(|r| extract_path(&Some(PostmanUrl::String(r.clone()))));
            }

            let path_parts: Vec<String> = obj
                .path
                .iter()
                .map(|p| {
                    if p.starts_with(':') {
                        // Convert :param to {param}
                        format!("{{{}}}", &p[1..])
                    } else if p.starts_with("{{") && p.ends_with("}}") {
                        // Convert {{param}} to {param}
                        format!("{{{}}}", &p[2..p.len() - 2])
                    } else {
                        p.clone()
                    }
                })
                .collect();

            Some(format!("/{}", path_parts.join("/")))
        }
    }
}

/// Restore path parameters from original URL
fn restore_path_params(original: &str, path: &str) -> String {
    // Find {{var}} patterns in original and convert to {var}
    let mut result = path.to_string();
    let re = regex::Regex::new(r"\{\{(\w+)\}\}").unwrap();

    for cap in re.captures_iter(original) {
        let var_name = &cap[1];
        // Try to find placeholder and replace
        result = result.replace("placeholder", &format!("{{{}}}", var_name));
    }

    // Also handle :param syntax
    let colon_re = regex::Regex::new(r":(\w+)").unwrap();
    for cap in colon_re.captures_iter(original) {
        let var_name = &cap[1];
        if !result.contains(&format!("{{{}}}", var_name)) {
            // Replace the first occurrence
            result = result.replacen(&format!(":{}", var_name), &format!("{{{}}}", var_name), 1);
        }
    }

    result
}

/// Extract parameters from Postman request
fn extract_parameters(
    request: &PostmanRequest,
    path: &str,
    _variables: &HashMap<String, String>,
) -> Vec<ParsedParameter> {
    let mut params = Vec::new();

    // Extract path parameters from path pattern
    let path_param_re = regex::Regex::new(r"\{(\w+)\}").unwrap();
    for cap in path_param_re.captures_iter(path) {
        let param_name = cap[1].to_string();

        // Try to find value from URL variables
        let description = if let Some(PostmanUrl::Object(obj)) = &request.url {
            obj.variable
                .iter()
                .find(|v| v.key == param_name)
                .map(|v| v.value.clone())
        } else {
            None
        };

        let schema = Some(infer_schema_from_value(description.as_deref()));
        params.push(ParsedParameter {
            name: param_name,
            location: "path".to_string(),
            required: true,
            description,
            schema,
        });
    }

    // Extract query parameters
    if let Some(url) = &request.url {
        for q in url.query() {
            if q.disabled {
                continue;
            }
            params.push(ParsedParameter {
                name: q.key.clone(),
                location: "query".to_string(),
                required: false,
                description: None,
                schema: Some(infer_schema_from_value(Some(&q.value))),
            });
        }
    }

    // Extract header parameters (excluding standard ones)
    if let Some(headers) = &request.header {
        for h in headers {
            if h.disabled {
                continue;
            }
            let key_lower = h.key.to_lowercase();
            if key_lower == "authorization" || key_lower == "content-type" || key_lower == "accept" {
                continue;
            }
            params.push(ParsedParameter {
                name: h.key.clone(),
                location: "header".to_string(),
                required: false,
                description: Some(h.value.clone()),
                schema: Some(serde_json::json!({"type": "string"})),
            });
        }
    }

    params
}

/// Infer JSON schema from example value
fn infer_schema_from_value(value: Option<&str>) -> serde_json::Value {
    let Some(value) = value else {
        return serde_json::json!({"type": "string"});
    };

    // Try integer
    if value.parse::<i64>().is_ok() {
        return serde_json::json!({"type": "integer"});
    }

    // Try number
    if value.parse::<f64>().is_ok() {
        return serde_json::json!({"type": "number"});
    }

    // Try boolean
    if value == "true" || value == "false" {
        return serde_json::json!({"type": "boolean"});
    }

    serde_json::json!({"type": "string"})
}

/// Extract request body from Postman request
fn extract_request_body(request: &PostmanRequest) -> Option<ParsedRequestBody> {
    let body = request.body.as_ref()?;
    let mode = body.mode.as_deref()?;

    match mode {
        "raw" => {
            let raw = body.raw.as_ref()?;

            // Try to parse as JSON and generate schema
            let schema = if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(raw) {
                Some(generate_schema_from_example(&parsed))
            } else {
                Some(serde_json::json!({"type": "string"}))
            };

            Some(ParsedRequestBody {
                required: true,
                content_type: Some("application/json".to_string()),
                description: None,
                schema,
            })
        }
        "urlencoded" => {
            let properties: serde_json::Map<String, serde_json::Value> = body
                .urlencoded
                .iter()
                .filter(|f| !f.disabled)
                .map(|f| {
                    (
                        f.key.clone(),
                        serde_json::json!({
                            "type": "string",
                            "description": f.value
                        }),
                    )
                })
                .collect();

            Some(ParsedRequestBody {
                required: true,
                content_type: Some("application/x-www-form-urlencoded".to_string()),
                description: None,
                schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": properties
                })),
            })
        }
        "formdata" => {
            let properties: serde_json::Map<String, serde_json::Value> = body
                .formdata
                .iter()
                .filter(|f| !f.disabled)
                .map(|f| {
                    let schema = if f.param_type.as_deref() == Some("file") {
                        serde_json::json!({
                            "type": "string",
                            "format": "binary"
                        })
                    } else {
                        serde_json::json!({
                            "type": "string",
                            "description": f.value
                        })
                    };
                    (f.key.clone(), schema)
                })
                .collect();

            Some(ParsedRequestBody {
                required: true,
                content_type: Some("multipart/form-data".to_string()),
                description: None,
                schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": properties
                })),
            })
        }
        _ => None,
    }
}

/// Generate JSON schema from example value
fn generate_schema_from_example(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Null => serde_json::json!({"type": "null"}),
        serde_json::Value::Bool(_) => serde_json::json!({"type": "boolean"}),
        serde_json::Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                serde_json::json!({"type": "integer"})
            } else {
                serde_json::json!({"type": "number"})
            }
        }
        serde_json::Value::String(_) => serde_json::json!({"type": "string"}),
        serde_json::Value::Array(arr) => {
            let items = arr
                .first()
                .map(generate_schema_from_example)
                .unwrap_or_else(|| serde_json::json!({"type": "object"}));
            serde_json::json!({
                "type": "array",
                "items": items
            })
        }
        serde_json::Value::Object(obj) => {
            let properties: serde_json::Map<String, serde_json::Value> = obj
                .iter()
                .map(|(k, v)| (k.clone(), generate_schema_from_example(v)))
                .collect();
            serde_json::json!({
                "type": "object",
                "properties": properties
            })
        }
    }
}

/// Errors during Postman conversion
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("Invalid JSON: {0}")]
    InvalidJson(String),

    #[error("Invalid Postman collection: {0}")]
    InvalidCollection(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_COLLECTION: &str = r#"
{
    "info": {
        "name": "Sample API",
        "description": "A sample API collection",
        "version": "1.0.0",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    },
    "item": [
        {
            "name": "Users",
            "item": [
                {
                    "name": "List Users",
                    "request": {
                        "method": "GET",
                        "url": {
                            "raw": "{{baseUrl}}/users",
                            "host": ["{{baseUrl}}"],
                            "path": ["users"],
                            "query": [
                                {"key": "limit", "value": "10"}
                            ]
                        }
                    }
                },
                {
                    "name": "Get User",
                    "request": {
                        "method": "GET",
                        "url": {
                            "raw": "{{baseUrl}}/users/:id",
                            "host": ["{{baseUrl}}"],
                            "path": ["users", ":id"],
                            "variable": [
                                {"key": "id", "value": "123"}
                            ]
                        }
                    }
                }
            ]
        }
    ],
    "variable": [
        {"key": "baseUrl", "value": "https://api.example.com"}
    ]
}
"#;

    #[test]
    fn test_parse_postman_collection() {
        let collection = parse_postman(SAMPLE_COLLECTION).unwrap();
        assert_eq!(collection.info.name, "Sample API");
        assert_eq!(collection.item.len(), 1);
    }

    #[test]
    fn test_convert_postman() {
        let collection = parse_postman(SAMPLE_COLLECTION).unwrap();
        let spec = convert_postman(&collection).unwrap();

        assert_eq!(spec.title, "Sample API");
        assert_eq!(spec.operation_count, 2);
        assert_eq!(spec.tags.len(), 1);
        assert!(spec.tags.contains(&"Users".to_string()));
    }
}
