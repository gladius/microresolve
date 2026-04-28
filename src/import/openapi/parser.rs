//! OpenAPI specification parser
//!
//! Parses OpenAPI 3.x and Swagger 2.0 specifications (JSON or YAML)
//! using serde_json::Value for maximum compatibility.
//!
//! This avoids typed deserialization crates (oas3, openapiv3) which
//! fail on edge cases like large integers in schema constraints.

use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

// No-op logging macros (replaces tracing dependency)
macro_rules! info { ($($t:tt)*) => {} }
macro_rules! debug { ($($t:tt)*) => {} }
macro_rules! warn { ($($t:tt)*) => {} }

use super::types::{
    ParsedOperation, ParsedParameter, ParsedRequestBody, ParsedResponse, ParsedSpec,
    SecurityScheme, ServerInfo,
};

/// Matches YAML integer values with 19+ digits that may overflow i64.
/// Only matches values after a colon (YAML key: value context).
static LARGE_INT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)(:\s+)-?(\d{19,})\b").unwrap());

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════════

/// Parse an OpenAPI specification from a string (JSON or YAML)
pub fn parse_openapi(input: &str) -> Result<ParsedSpec, ParseError> {
    info!("📄 OPENAPI_PARSE - Starting parse");

    // Parse to serde_json::Value — handles any valid YAML/JSON including
    // large integers, YAML anchors, etc.
    let root: serde_json::Value = if input.trim().starts_with('{') {
        debug!("📄 OPENAPI_PARSE - Detected JSON format");
        serde_json::from_str(input).map_err(|e| ParseError::InvalidJson(e.to_string()))?
    } else {
        debug!("📄 OPENAPI_PARSE - Detected YAML format");
        let yaml: serde_yaml::Value = match serde_yaml::from_str(input) {
            Ok(v) => v,
            Err(e) => {
                let err_msg = e.to_string();
                if err_msg.contains("i128") || err_msg.contains("integer") {
                    // Some specs (e.g. OpenAI) use integers that overflow i64 in
                    // schema min/max fields. Sanitize and retry.
                    warn!("📄 OPENAPI_PARSE - Large integer overflow, sanitizing and retrying");
                    let sanitized = LARGE_INT_RE.replace_all(input, "${1}0");
                    serde_yaml::from_str(&sanitized)
                        .map_err(|e2| ParseError::InvalidYaml(e2.to_string()))?
                } else {
                    return Err(ParseError::InvalidYaml(err_msg));
                }
            }
        };
        serde_json::to_value(yaml)
            .map_err(|e| ParseError::InvalidYaml(format!("YAML→JSON conversion failed: {}", e)))?
    };

    // Detect spec version
    let openapi_ver = str_field(&root, "openapi").unwrap_or("");
    let swagger_ver = str_field(&root, "swagger").unwrap_or("");

    let parsed = if openapi_ver.starts_with("3.") {
        info!(version = openapi_ver, "📄 OPENAPI_PARSE - Detected OpenAPI 3.x");
        convert_openapi_v3(&root)?
    } else if swagger_ver.starts_with("2.") {
        info!(version = swagger_ver, "📄 OPENAPI_PARSE - Detected Swagger 2.0");
        convert_swagger_v2(&root)?
    } else {
        return Err(ParseError::InvalidSpec(format!(
            "Unsupported spec version (openapi={}, swagger={})",
            openapi_ver, swagger_ver
        )));
    };

    info!(
        title = %parsed.title,
        version = %parsed.version,
        operations = parsed.operation_count,
        paths = parsed.path_count,
        security_schemes = parsed.security_schemes.len(),
        "✅ OPENAPI_PARSE - Parse complete"
    );

    Ok(parsed)
}

// URL fetching is handled at the server layer (requires reqwest/async)

// ═══════════════════════════════════════════════════════════════════════════════
// OPENAPI 3.x
// ═══════════════════════════════════════════════════════════════════════════════

fn convert_openapi_v3(root: &serde_json::Value) -> Result<ParsedSpec, ParseError> {
    let info = root
        .get("info")
        .ok_or_else(|| ParseError::InvalidSpec("Missing 'info' field".to_string()))?;

    let mut parsed = ParsedSpec::new(
        str_field(info, "title").unwrap_or("Untitled").to_string(),
        str_field(info, "version").unwrap_or("0.0.0").to_string(),
    );
    parsed.description = str_field(info, "description").map(|s| s.to_string());
    parsed.openapi_version = str_field(root, "openapi").map(|s| s.to_string());

    // Servers
    if let Some(servers) = root.get("servers").and_then(|v| v.as_array()) {
        parsed.servers = servers
            .iter()
            .map(|s| ServerInfo {
                url: str_field(s, "url").unwrap_or("").to_string(),
                description: str_field(s, "description").map(|d| d.to_string()),
            })
            .collect();
    }

    // Tags
    if let Some(tags) = root.get("tags").and_then(|v| v.as_array()) {
        parsed.tags = tags
            .iter()
            .filter_map(|t| str_field(t, "name").map(|n| n.to_string()))
            .collect();
    }

    // Security schemes
    if let Some(components) = root.get("components") {
        if let Some(schemes) = components.get("securitySchemes").and_then(|v| v.as_object()) {
            for (name, scheme) in schemes {
                let scheme = resolve_ref(root, scheme);
                if let Some(sec) = extract_security_scheme(scheme) {
                    parsed.security_schemes.insert(name.clone(), sec);
                }
            }
        }
    }

    // Paths → operations
    if let Some(paths) = root.get("paths").and_then(|v| v.as_object()) {
        for (path, path_item) in paths {
            let path_item = resolve_ref(root, path_item);
            extract_operations_v3(root, path, path_item, &mut parsed.operations);
        }
    }

    parsed.update_counts();

    if parsed.security_schemes.is_empty() {
        infer_security_schemes(&parsed.operations, &mut parsed.security_schemes);
    }

    Ok(parsed)
}

fn extract_operations_v3(
    root: &serde_json::Value,
    path: &str,
    path_item: &serde_json::Value,
    operations: &mut Vec<ParsedOperation>,
) {
    let obj = match path_item.as_object() {
        Some(o) => o,
        None => return,
    };

    for method in HTTP_METHODS {
        if let Some(op) = obj.get(*method) {
            // Parameters
            let parameters = extract_parameters(root, op);

            // Request body (v3 only)
            let request_body = op
                .get("requestBody")
                .map(|rb| resolve_ref(root, rb))
                .and_then(|rb| extract_request_body_v3(root, rb));

            // Responses
            let responses = extract_responses_v3(root, op);

            if let Some(parsed) =
                build_operation(root, path, method, op, parameters, request_body, responses)
            {
                operations.push(parsed);
            }
        }
    }
}

fn extract_request_body_v3(
    _root: &serde_json::Value,
    rb: &serde_json::Value,
) -> Option<ParsedRequestBody> {
    let content = rb.get("content")?.as_object()?;

    let (content_type, media_type) = content
        .get("application/json")
        .map(|mt| ("application/json", mt))
        .or_else(|| content.iter().next().map(|(ct, mt)| (ct.as_str(), mt)))?;

    Some(ParsedRequestBody {
        required: rb.get("required").and_then(|r| r.as_bool()).unwrap_or(false),
        content_type: Some(content_type.to_string()),
        description: str_field(rb, "description").map(|s| s.to_string()),
        schema: media_type.get("schema").cloned(),
    })
}

fn extract_responses_v3(
    root: &serde_json::Value,
    op: &serde_json::Value,
) -> HashMap<String, ParsedResponse> {
    let mut result = HashMap::new();

    let responses = match op.get("responses").and_then(|r| r.as_object()) {
        Some(r) => r,
        None => return result,
    };

    for (status, response) in responses {
        let response = resolve_ref(root, response);

        let (content_type, schema) = response
            .get("content")
            .and_then(|c| c.as_object())
            .and_then(|content| {
                content
                    .get("application/json")
                    .map(|mt| {
                        (
                            Some("application/json".to_string()),
                            mt.get("schema").cloned(),
                        )
                    })
                    .or_else(|| {
                        content.iter().next().map(|(ct, mt)| {
                            (Some(ct.clone()), mt.get("schema").cloned())
                        })
                    })
            })
            .unwrap_or((None, None));

        result.insert(
            status.clone(),
            ParsedResponse {
                status_code: status.clone(),
                description: str_field(response, "description").map(|s| s.to_string()),
                content_type,
                schema,
            },
        );
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// SWAGGER 2.0
// ═══════════════════════════════════════════════════════════════════════════════

fn convert_swagger_v2(root: &serde_json::Value) -> Result<ParsedSpec, ParseError> {
    let info = root
        .get("info")
        .ok_or_else(|| ParseError::InvalidSpec("Missing 'info' field".to_string()))?;

    let mut parsed = ParsedSpec::new(
        str_field(info, "title").unwrap_or("Untitled").to_string(),
        str_field(info, "version").unwrap_or("0.0.0").to_string(),
    );
    parsed.description = str_field(info, "description").map(|s| s.to_string());
    parsed.openapi_version = str_field(root, "swagger").map(|s| s.to_string());

    // Server: host + basePath + schemes
    let host = str_field(root, "host").unwrap_or("localhost");
    let base_path = str_field(root, "basePath").unwrap_or("");
    let scheme = root
        .get("schemes")
        .and_then(|s| s.as_array())
        .and_then(|a| a.first())
        .and_then(|s| s.as_str())
        .unwrap_or("https");

    parsed.servers = vec![ServerInfo {
        url: format!("{}://{}{}", scheme, host, base_path),
        description: None,
    }];

    // Tags
    if let Some(tags) = root.get("tags").and_then(|v| v.as_array()) {
        parsed.tags = tags
            .iter()
            .filter_map(|t| str_field(t, "name").map(|n| n.to_string()))
            .collect();
    }

    // Security definitions
    if let Some(defs) = root.get("securityDefinitions").and_then(|v| v.as_object()) {
        for (name, scheme) in defs {
            if let Some(sec) = extract_security_scheme_v2(scheme) {
                parsed.security_schemes.insert(name.clone(), sec);
            }
        }
    }

    // Paths → operations
    if let Some(paths) = root.get("paths").and_then(|v| v.as_object()) {
        for (path, path_item) in paths {
            let path_item = resolve_ref(root, path_item);
            extract_operations_v2(root, path, path_item, &mut parsed.operations);
        }
    }

    parsed.update_counts();

    if parsed.security_schemes.is_empty() {
        infer_security_schemes(&parsed.operations, &mut parsed.security_schemes);
    }

    Ok(parsed)
}

fn extract_operations_v2(
    root: &serde_json::Value,
    path: &str,
    path_item: &serde_json::Value,
    operations: &mut Vec<ParsedOperation>,
) {
    let obj = match path_item.as_object() {
        Some(o) => o,
        None => return,
    };

    for method in HTTP_METHODS {
        if let Some(op) = obj.get(*method) {
            // Swagger 2.0: split params into regular params + body param
            let all_params = op
                .get("parameters")
                .and_then(|p| p.as_array())
                .cloned()
                .unwrap_or_default();

            let mut parameters = Vec::new();
            let mut request_body = None;

            for param in &all_params {
                let param = resolve_ref(root, param);
                let location = str_field(param, "in").unwrap_or("");

                if location == "body" {
                    request_body = Some(ParsedRequestBody {
                        required: param
                            .get("required")
                            .and_then(|r| r.as_bool())
                            .unwrap_or(false),
                        content_type: Some("application/json".to_string()),
                        description: str_field(param, "description").map(|s| s.to_string()),
                        schema: param.get("schema").cloned(),
                    });
                } else if let Some(p) = extract_single_parameter(param) {
                    parameters.push(p);
                }
            }

            // Swagger 2.0 responses: schema is direct (no content wrapper)
            let responses = extract_responses_v2(root, op);

            if let Some(parsed) =
                build_operation(root, path, method, op, parameters, request_body, responses)
            {
                operations.push(parsed);
            }
        }
    }
}

fn extract_responses_v2(
    root: &serde_json::Value,
    op: &serde_json::Value,
) -> HashMap<String, ParsedResponse> {
    let mut result = HashMap::new();

    let responses = match op.get("responses").and_then(|r| r.as_object()) {
        Some(r) => r,
        None => return result,
    };

    for (status, response) in responses {
        let response = resolve_ref(root, response);

        result.insert(
            status.clone(),
            ParsedResponse {
                status_code: status.clone(),
                description: str_field(response, "description").map(|s| s.to_string()),
                content_type: response
                    .get("schema")
                    .map(|_| "application/json".to_string()),
                schema: response.get("schema").cloned(),
            },
        );
    }

    result
}

fn extract_security_scheme_v2(scheme: &serde_json::Value) -> Option<SecurityScheme> {
    let scheme_type = str_field(scheme, "type")?;

    match scheme_type {
        "apiKey" => Some(SecurityScheme {
            scheme_type: "apiKey".to_string(),
            description: str_field(scheme, "description").map(|s| s.to_string()),
            name: str_field(scheme, "name").map(|s| s.to_string()),
            location: str_field(scheme, "in").map(|s| s.to_string()),
            scheme: None,
            bearer_format: None,
        }),
        "basic" => Some(SecurityScheme {
            scheme_type: "http".to_string(),
            description: str_field(scheme, "description").map(|s| s.to_string()),
            name: None,
            location: None,
            scheme: Some("basic".to_string()),
            bearer_format: None,
        }),
        "oauth2" => Some(SecurityScheme {
            scheme_type: "oauth2".to_string(),
            description: str_field(scheme, "description").map(|s| s.to_string()),
            name: None,
            location: None,
            scheme: None,
            bearer_format: None,
        }),
        _ => {
            warn!(scheme_type = scheme_type, "📄 OPENAPI_PARSE - Unknown Swagger 2.0 security type");
            None
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

const HTTP_METHODS: &[&str] = &[
    "get", "post", "put", "delete", "patch", "head", "options", "trace",
];

/// Build a ParsedOperation from extracted components (shared between v2 and v3)
fn build_operation(
    _root: &serde_json::Value,
    path: &str,
    method: &str,
    op: &serde_json::Value,
    parameters: Vec<ParsedParameter>,
    request_body: Option<ParsedRequestBody>,
    responses: HashMap<String, ParsedResponse>,
) -> Option<ParsedOperation> {
    let operation_id = str_field(op, "operationId")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            let sanitized = path.replace(|c: char| !c.is_alphanumeric(), "_");
            format!("{}_{}", method, sanitized)
        });

    let tags: Vec<String> = op
        .get("tags")
        .and_then(|t| t.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let category = tags.first().cloned().unwrap_or_else(|| {
        path.split('/')
            .nth(1)
            .unwrap_or("uncategorized")
            .to_string()
    });

    let security = op
        .get("security")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    item.as_object().map(|obj| {
                        obj.iter()
                            .map(|(k, v)| {
                                let scopes: Vec<String> = v
                                    .as_array()
                                    .map(|a| {
                                        a.iter()
                                            .filter_map(|s| s.as_str().map(|s| s.to_string()))
                                            .collect()
                                    })
                                    .unwrap_or_default();
                                (k.clone(), scopes)
                            })
                            .collect::<HashMap<String, Vec<String>>>()
                    })
                })
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty());

    Some(ParsedOperation {
        id: operation_id.clone(),
        name: str_field(op, "summary")
            .or_else(|| str_field(op, "operationId"))
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{} {}", method.to_uppercase(), path)),
        description: str_field(op, "description")
            .unwrap_or("")
            .to_string(),
        method: method.to_uppercase(),
        path: path.to_string(),
        category,
        tags,
        parameters,
        request_body,
        responses,
        security,
        deprecated: op
            .get("deprecated")
            .and_then(|d| d.as_bool())
            .unwrap_or(false),
        summary: str_field(op, "summary").map(|s| s.to_string()),
        operation_id: Some(operation_id),
    })
}

/// Extract parameters from an operation (shared between v2 path params and v3)
fn extract_parameters(
    root: &serde_json::Value,
    op: &serde_json::Value,
) -> Vec<ParsedParameter> {
    op.get("parameters")
        .and_then(|p| p.as_array())
        .map(|params| {
            params
                .iter()
                .filter_map(|p| {
                    let p = resolve_ref(root, p);
                    extract_single_parameter(p)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn extract_single_parameter(param: &serde_json::Value) -> Option<ParsedParameter> {
    let name = str_field(param, "name")?.to_string();
    let location = str_field(param, "in").unwrap_or("query").to_string();

    Some(ParsedParameter {
        name,
        location,
        required: param
            .get("required")
            .and_then(|r| r.as_bool())
            .unwrap_or(false),
        description: str_field(param, "description").map(|s| s.to_string()),
        schema: param.get("schema").cloned(),
    })
}

/// Extract security scheme (OpenAPI 3.x)
fn extract_security_scheme(scheme: &serde_json::Value) -> Option<SecurityScheme> {
    let scheme_type = str_field(scheme, "type")?;

    match scheme_type {
        "apiKey" => Some(SecurityScheme {
            scheme_type: "apiKey".to_string(),
            description: str_field(scheme, "description").map(|s| s.to_string()),
            name: str_field(scheme, "name").map(|s| s.to_string()),
            location: str_field(scheme, "in").map(|s| s.to_string()),
            scheme: None,
            bearer_format: None,
        }),
        "http" => Some(SecurityScheme {
            scheme_type: "http".to_string(),
            description: str_field(scheme, "description").map(|s| s.to_string()),
            name: None,
            location: None,
            scheme: str_field(scheme, "scheme").map(|s| s.to_string()),
            bearer_format: str_field(scheme, "bearerFormat").map(|s| s.to_string()),
        }),
        "oauth2" => Some(SecurityScheme {
            scheme_type: "oauth2".to_string(),
            description: str_field(scheme, "description").map(|s| s.to_string()),
            name: None,
            location: None,
            scheme: None,
            bearer_format: None,
        }),
        "openIdConnect" => Some(SecurityScheme {
            scheme_type: "openIdConnect".to_string(),
            description: str_field(scheme, "description").map(|s| s.to_string()),
            name: None,
            location: None,
            scheme: None,
            bearer_format: None,
        }),
        "mutualTLS" => Some(SecurityScheme {
            scheme_type: "mutualTLS".to_string(),
            description: str_field(scheme, "description").map(|s| s.to_string()),
            name: None,
            location: None,
            scheme: None,
            bearer_format: None,
        }),
        _ => {
            warn!(scheme_type = scheme_type, "📄 OPENAPI_PARSE - Unknown security scheme type");
            None
        }
    }
}

/// Resolve a $ref pointer to the referenced value in the document
fn resolve_ref<'a>(root: &'a serde_json::Value, value: &'a serde_json::Value) -> &'a serde_json::Value {
    let ref_path = match value.get("$ref").and_then(|r| r.as_str()) {
        Some(r) => r,
        None => return value,
    };

    // Navigate: "#/components/schemas/Foo" → root["components"]["schemas"]["Foo"]
    let parts: Vec<&str> = ref_path.split('/').collect();
    if parts.first() != Some(&"#") {
        debug!(ref_path = ref_path, "📄 OPENAPI_PARSE - Non-local $ref, returning as-is");
        return value;
    }

    let mut current = root;
    for part in &parts[1..] {
        // JSON pointer decoding: ~1 → /, ~0 → ~
        let decoded = part.replace("~1", "/").replace("~0", "~");
        match current.get(decoded.as_str()) {
            Some(next) => current = next,
            None => {
                debug!(ref_path = ref_path, part = *part, "📄 OPENAPI_PARSE - $ref target not found");
                return value;
            }
        }
    }

    current
}

/// Get a string field from a JSON value
fn str_field<'a>(obj: &'a serde_json::Value, field: &str) -> Option<&'a str> {
    obj.get(field).and_then(|v| v.as_str())
}

/// Infer security schemes from operation security requirements
fn infer_security_schemes(
    operations: &[ParsedOperation],
    schemes: &mut HashMap<String, SecurityScheme>,
) {
    for op in operations {
        if let Some(security) = &op.security {
            for req in security {
                for scheme_name in req.keys() {
                    if !schemes.contains_key(scheme_name) {
                        let lower = scheme_name.to_lowercase();
                        let inferred = if lower.contains("bearer")
                            || lower.contains("token")
                            || lower.contains("jwt")
                        {
                            SecurityScheme {
                                scheme_type: "http".to_string(),
                                description: Some(format!(
                                    "Inferred Bearer auth from {}",
                                    op.path
                                )),
                                name: None,
                                location: None,
                                scheme: Some("bearer".to_string()),
                                bearer_format: None,
                            }
                        } else if lower.contains("apikey") || lower.contains("api_key") {
                            SecurityScheme {
                                scheme_type: "apiKey".to_string(),
                                description: Some(format!(
                                    "Inferred API Key auth from {}",
                                    op.path
                                )),
                                name: Some("Authorization".to_string()),
                                location: Some("header".to_string()),
                                scheme: None,
                                bearer_format: None,
                            }
                        } else if lower.contains("basic") {
                            SecurityScheme {
                                scheme_type: "http".to_string(),
                                description: Some(format!(
                                    "Inferred Basic auth from {}",
                                    op.path
                                )),
                                name: None,
                                location: None,
                                scheme: Some("basic".to_string()),
                                bearer_format: None,
                            }
                        } else {
                            SecurityScheme {
                                scheme_type: "http".to_string(),
                                description: Some(format!(
                                    "Inferred auth from {} (defaulted to Bearer)",
                                    op.path
                                )),
                                name: None,
                                location: None,
                                scheme: Some("bearer".to_string()),
                                bearer_format: None,
                            }
                        };

                        warn!(
                            scheme_name = %scheme_name,
                            inferred_type = %inferred.scheme_type,
                            "📄 OPENAPI_PARSE - Inferred security scheme from operation"
                        );
                        schemes.insert(scheme_name.clone(), inferred);
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ERRORS
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Invalid JSON: {0}")]
    InvalidJson(String),

    #[error("Invalid YAML: {0}")]
    InvalidYaml(String),

    #[error("Failed to fetch spec: {0}")]
    FetchFailed(String),

    #[error("Invalid OpenAPI spec: {0}")]
    InvalidSpec(String),
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_V3: &str = r#"
openapi: "3.1.0"
info:
  title: Sample API
  version: "1.0.0"
servers:
  - url: https://api.example.com
    description: Production
tags:
  - name: users
paths:
  /users:
    get:
      summary: List users
      operationId: listUsers
      tags: [users]
      responses:
        "200":
          description: Success
    post:
      summary: Create user
      operationId: createUser
      tags: [users]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                name:
                  type: string
      responses:
        "201":
          description: Created
  /users/{id}:
    get:
      summary: Get user
      operationId: getUser
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Success
"#;

    const SAMPLE_V2: &str = r#"
swagger: "2.0"
info:
  title: Swagger Petstore
  version: "1.0.0"
host: petstore.swagger.io
basePath: /v2
schemes:
  - https
tags:
  - name: pet
securityDefinitions:
  api_key:
    type: apiKey
    name: api_key
    in: header
paths:
  /pet:
    post:
      summary: Add a new pet
      operationId: addPet
      tags: [pet]
      parameters:
        - in: body
          name: body
          required: true
          schema:
            type: object
      responses:
        "200":
          description: Success
  /pet/{petId}:
    get:
      summary: Find pet by ID
      operationId: getPetById
      tags: [pet]
      parameters:
        - name: petId
          in: path
          required: true
          type: integer
      responses:
        "200":
          description: Success
          schema:
            type: object
"#;

    #[test]
    fn test_parse_openapi_v3() {
        let result = parse_openapi(SAMPLE_V3);
        assert!(result.is_ok(), "Parse failed: {:?}", result.err());

        let spec = result.unwrap();
        assert_eq!(spec.title, "Sample API");
        assert_eq!(spec.version, "1.0.0");
        assert_eq!(spec.operation_count, 3);
        assert_eq!(spec.path_count, 2);
        assert_eq!(spec.tags, vec!["users"]);
        assert_eq!(spec.servers.len(), 1);
        assert_eq!(spec.servers[0].url, "https://api.example.com");
    }

    #[test]
    fn test_parse_swagger_v2() {
        let result = parse_openapi(SAMPLE_V2);
        assert!(result.is_ok(), "Parse failed: {:?}", result.err());

        let spec = result.unwrap();
        assert_eq!(spec.title, "Swagger Petstore");
        assert_eq!(spec.version, "1.0.0");
        assert_eq!(spec.operation_count, 2);
        assert_eq!(spec.path_count, 2);
        assert_eq!(spec.tags, vec!["pet"]);
        assert_eq!(spec.servers.len(), 1);
        assert_eq!(spec.servers[0].url, "https://petstore.swagger.io/v2");
        assert!(spec.security_schemes.contains_key("api_key"));
    }

    #[test]
    fn test_v3_request_body() {
        let spec = parse_openapi(SAMPLE_V3).unwrap();
        let create_op = spec
            .operations
            .iter()
            .find(|op| op.operation_id.as_deref() == Some("createUser"))
            .unwrap();
        assert!(create_op.request_body.is_some());
        assert_eq!(
            create_op.request_body.as_ref().unwrap().content_type,
            Some("application/json".to_string())
        );
    }

    #[test]
    fn test_v2_body_parameter_becomes_request_body() {
        let spec = parse_openapi(SAMPLE_V2).unwrap();
        let add_op = spec
            .operations
            .iter()
            .find(|op| op.operation_id.as_deref() == Some("addPet"))
            .unwrap();
        assert!(add_op.request_body.is_some());
        // Body param should not appear in regular parameters
        assert!(add_op.parameters.is_empty());
    }

    #[test]
    fn test_v3_parameters() {
        let spec = parse_openapi(SAMPLE_V3).unwrap();
        let get_op = spec
            .operations
            .iter()
            .find(|op| op.operation_id.as_deref() == Some("getUser"))
            .unwrap();
        assert_eq!(get_op.parameters.len(), 1);
        assert_eq!(get_op.parameters[0].name, "id");
        assert_eq!(get_op.parameters[0].location, "path");
        assert!(get_op.parameters[0].required);
    }

    #[test]
    fn test_large_integer_handling() {
        // This is the exact pattern that breaks the oas3 crate with OpenAI's spec
        let spec_with_large_int = r#"
openapi: "3.0.0"
info:
  title: Test API
  version: "1.0.0"
paths:
  /test:
    post:
      summary: Test endpoint
      operationId: testEndpoint
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                seed:
                  type: integer
                  minimum: -9223372036854775808
                  maximum: 9223372036854775807
      responses:
        "200":
          description: Success
"#;
        let result = parse_openapi(spec_with_large_int);
        assert!(result.is_ok(), "Large integer parse failed: {:?}", result.err());

        let spec = result.unwrap();
        assert_eq!(spec.operation_count, 1);
    }

    #[test]
    fn test_json_format() {
        let json_spec = r#"{"openapi":"3.0.0","info":{"title":"JSON API","version":"1.0"},"paths":{"/test":{"get":{"summary":"Test","operationId":"test","responses":{"200":{"description":"OK"}}}}}}"#;
        let result = parse_openapi(json_spec);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().title, "JSON API");
    }

    #[test]
    fn test_ref_resolution() {
        let spec_with_refs = r#"
openapi: "3.0.0"
info:
  title: Ref Test
  version: "1.0.0"
paths:
  /test:
    get:
      summary: Test
      operationId: test
      parameters:
        - $ref: '#/components/parameters/PageParam'
      responses:
        "200":
          description: OK
components:
  parameters:
    PageParam:
      name: page
      in: query
      required: false
      schema:
        type: integer
"#;
        let result = parse_openapi(spec_with_refs);
        assert!(result.is_ok(), "Ref parse failed: {:?}", result.err());

        let spec = result.unwrap();
        let op = &spec.operations[0];
        assert_eq!(op.parameters.len(), 1);
        assert_eq!(op.parameters[0].name, "page");
        assert_eq!(op.parameters[0].location, "query");
    }
}
