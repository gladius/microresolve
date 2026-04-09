//! Spec import — OpenAPI and Postman collection parsing + intent generation.
//!
//! Parses API specs and converts each operation into a routable intent with
//! seeds derived from the operation's summary and description.

pub mod openapi;
pub mod postman;

use crate::{IntentType, Router, SeedCheckResult};
use openapi::ParsedSpec;

/// Result of importing a spec into the router.
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// Intents successfully created.
    pub created: Vec<ImportedIntent>,
    /// Operations skipped (no usable seeds).
    pub skipped: Vec<String>,
    /// Total operations in the spec.
    pub total_operations: usize,
}

/// A single imported intent.
#[derive(Debug, Clone)]
pub struct ImportedIntent {
    pub intent_id: String,
    pub seeds: Vec<String>,
    pub endpoint: String,
    pub method: String,
    pub intent_type: IntentType,
    pub seed_checks: Vec<SeedCheckResult>,
}

/// Import a parsed spec into the router, creating one intent per operation.
///
/// Seeds come from: summary + description sentences.
/// Intent type: GET/HEAD = Context, everything else = Action.
/// Metadata: endpoint (method + path), operation_id, tags.
pub fn import_spec(router: &mut Router, spec: &ParsedSpec) -> ImportResult {
    let mut created = Vec::new();
    let mut skipped = Vec::new();

    for op in &spec.operations {
        // Build intent name from operationId or path
        let intent_id = op.operation_id.as_deref()
            .unwrap_or(&op.id);
        let intent_name = to_snake_case(intent_id);

        // Build seeds from summary + description
        let mut seeds: Vec<String> = Vec::new();

        if let Some(ref summary) = op.summary {
            let s = summary.trim().to_lowercase();
            if !s.is_empty() {
                seeds.push(s);
            }
        }

        if !op.description.is_empty() {
            for sent in op.description.split(". ") {
                let s = sent.trim().to_lowercase();
                if s.len() > 10 && seeds.len() < 10 {
                    // Remove trailing period
                    let s = s.trim_end_matches('.');
                    seeds.push(s.to_string());
                }
            }
        }

        // Also add the operation name as a seed if different from summary
        let name_lower = op.name.to_lowercase();
        if !seeds.contains(&name_lower) && !name_lower.is_empty() {
            seeds.push(name_lower);
        }

        if seeds.is_empty() {
            skipped.push(intent_name);
            continue;
        }

        // Determine intent type from HTTP method
        let intent_type = match op.method.as_str() {
            "GET" | "HEAD" => IntentType::Context,
            _ => IntentType::Action,
        };

        // Create the intent
        let seed_refs: Vec<&str> = seeds.iter().map(|s| s.as_str()).collect();
        let seed_checks = router.add_intent(&intent_name, &seed_refs);
        router.set_intent_type(&intent_name, intent_type);

        // Store endpoint metadata
        let endpoint = format!("{} {}", op.method, op.path);
        router.set_metadata(&intent_name, "endpoint", vec![endpoint.clone()]);
        if let Some(ref op_id) = op.operation_id {
            router.set_metadata(&intent_name, "operation_id", vec![op_id.clone()]);
        }
        if !op.tags.is_empty() {
            router.set_metadata(&intent_name, "tags", op.tags.clone());
        }

        // Store parameter info as metadata for LLM tool calling
        if !op.parameters.is_empty() {
            let param_names: Vec<String> = op.parameters.iter()
                .map(|p| format!("{}({}{})", p.name, p.location, if p.required { ",required" } else { "" }))
                .collect();
            router.set_metadata(&intent_name, "parameters", param_names);
        }

        if op.request_body.is_some() {
            router.set_metadata(&intent_name, "has_body", vec!["true".to_string()]);
        }

        created.push(ImportedIntent {
            intent_id: intent_name,
            seeds,
            endpoint,
            method: op.method.clone(),
            intent_type,
            seed_checks,
        });
    }

    ImportResult {
        total_operations: spec.operations.len(),
        created,
        skipped,
    }
}

/// Auto-detect and parse a spec string (OpenAPI JSON/YAML or Postman JSON).
pub fn parse_spec(input: &str) -> Result<ParsedSpec, String> {
    // Try Postman first (has "info.schema" with postman URL)
    if let Ok(collection) = serde_json::from_str::<postman::PostmanCollection>(input) {
        if collection.info.schema.as_ref().map_or(false, |s| s.contains("postman")) {
            return postman::convert_postman(&collection)
                .map_err(|e| format!("Postman parse error: {}", e));
        }
    }

    // Try OpenAPI
    openapi::parse_openapi(input)
        .map_err(|e| format!("OpenAPI parse error: {}", e))
}

/// Convert a string to snake_case intent name.
pub fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            result.push('_');
        }
        if c == '-' || c == ' ' {
            result.push('_');
        } else {
            result.push(c.to_lowercase().next().unwrap_or(c));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("getOrder"), "get_order");
        assert_eq!(to_snake_case("createPaymentIntent"), "create_payment_intent");
        assert_eq!(to_snake_case("list-users"), "list_users");
        assert_eq!(to_snake_case("cancelOrder"), "cancel_order");
    }

    #[test]
    fn test_import_openapi_spec() {
        let spec_yaml = r#"
openapi: "3.0.0"
info:
  title: Test API
  version: "1.0.0"
paths:
  /orders:
    get:
      summary: List all orders
      description: Retrieve a list of all customer orders. Supports pagination and filtering.
      operationId: listOrders
      tags: [orders]
      responses:
        "200":
          description: Success
    post:
      summary: Create a new order
      description: Place a new order for the customer. Requires items and shipping address.
      operationId: createOrder
      tags: [orders]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
      responses:
        "201":
          description: Created
  /orders/{id}/cancel:
    post:
      summary: Cancel an order
      description: Cancel a pending order. Cannot cancel orders that have already shipped.
      operationId: cancelOrder
      tags: [orders]
      parameters:
        - name: id
          in: path
          required: true
      responses:
        "200":
          description: Cancelled
"#;

        let spec = openapi::parse_openapi(spec_yaml).unwrap();
        assert_eq!(spec.operations.len(), 3);

        let mut router = Router::new();
        let result = import_spec(&mut router, &spec);

        assert_eq!(result.created.len(), 3);
        assert_eq!(result.skipped.len(), 0);

        // Check intent types
        assert_eq!(router.get_intent_type("list_orders"), IntentType::Context);
        assert_eq!(router.get_intent_type("create_order"), IntentType::Action);
        assert_eq!(router.get_intent_type("cancel_order"), IntentType::Action);

        // Check routing works
        let results = router.route("cancel my order");
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "cancel_order");

        let results = router.route("show all orders");
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "list_orders");

        // Check metadata
        let meta = router.get_metadata("cancel_order").unwrap();
        assert!(meta.contains_key("endpoint"));
        assert!(meta.contains_key("parameters"));
    }

    #[test]
    fn test_parse_spec_auto_detect() {
        let openapi_json = r#"{"openapi":"3.0.0","info":{"title":"Test","version":"1.0"},"paths":{"/test":{"get":{"summary":"Test endpoint","operationId":"test","responses":{"200":{"description":"OK"}}}}}}"#;

        let spec = parse_spec(openapi_json).unwrap();
        assert_eq!(spec.title, "Test");
        assert_eq!(spec.operations.len(), 1);
    }
}
