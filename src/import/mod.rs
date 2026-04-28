//! Spec import — OpenAPI and Postman collection parsing + intent generation.
//!
//! Parses API specs and converts each operation into a routable intent with
//! phrases derived from the operation's summary and description.

pub mod openapi;
pub mod postman;

use crate::{IntentType, Resolver};
use openapi::ParsedSpec;

/// Result of importing a spec into the engine.
#[derive(Debug, Clone)]
pub struct ImportResult {
    /// Intents successfully created.
    pub created: Vec<ImportedIntent>,
    /// Operations skipped (no usable phrases).
    pub skipped: Vec<String>,
    /// Total operations in the spec.
    pub total_operations: usize,
}

/// A single imported intent.
#[derive(Debug, Clone)]
pub struct ImportedIntent {
    pub intent_id: String,
    pub phrases: Vec<String>,
    pub endpoint: String,
    pub method: String,
    pub intent_type: IntentType,
}

/// Import a parsed spec into the engine, creating one intent per operation.
///
/// Only uses operation name as minimal phrase. For proper phrases, use LLM generation
/// (server-side import_apply does this). Description is stored as metadata, not as phrases.
/// Intent type: GET/HEAD = Context, everything else = Action.
/// Metadata: endpoint (method + path), operation_id, tags, description.
pub fn import_spec(router: &mut Resolver, spec: &ParsedSpec) -> ImportResult {
    let mut created = Vec::new();
    let mut skipped = Vec::new();

    for op in &spec.operations {
        let intent_id = op.operation_id.as_deref()
            .unwrap_or(&op.id);
        let intent_name = to_snake_case(intent_id);

        // Only use operation name as minimal phrase — descriptions are metadata, not phrases
        let name_lower = op.name.to_lowercase();
        if name_lower.is_empty() {
            skipped.push(intent_name);
            continue;
        }

        // Determine intent type from HTTP method
        let intent_type = match op.method.as_str() {
            "GET" | "HEAD" => IntentType::Context,
            _ => IntentType::Action,
        };

        // Create the intent
        let _ = router.add_intent(&intent_name, &[name_lower.as_str()]);

        // Store description for LLM context (not as seed)
        let desc = op.summary.as_deref().or(Some(&op.name)).unwrap_or("");
        let _ = router.update_intent(&intent_name, crate::IntentEdit {
            intent_type: Some(intent_type),
            description: if desc.is_empty() { None } else { Some(desc.to_string()) },
            ..Default::default()
        });

        let endpoint = format!("{} {}", op.method, op.path);

        created.push(ImportedIntent {
            intent_id: intent_name,
            phrases: vec![name_lower],
            endpoint,
            method: op.method.clone(),
            intent_type,
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

        let mut router = Resolver::new();
        let result = import_spec(&mut router, &spec);

        assert_eq!(result.created.len(), 3);
        assert_eq!(result.skipped.len(), 0);

        // Check intent types
        assert_eq!(router.intent("list_orders").map(|i| i.intent_type), Some(IntentType::Context));
        assert_eq!(router.intent("create_order").map(|i| i.intent_type), Some(IntentType::Action));
        assert_eq!(router.intent("cancel_order").map(|i| i.intent_type), Some(IntentType::Action));

        // Check phrases were stored
        let cancel_phrases = router.training("cancel_order").unwrap_or_default();
        assert!(!cancel_phrases.is_empty());

        let list_phrases = router.training("list_orders").unwrap_or_default();
        assert!(!list_phrases.is_empty());
    }

    #[test]
    fn test_parse_spec_auto_detect() {
        let openapi_json = r#"{"openapi":"3.0.0","info":{"title":"Test","version":"1.0"},"paths":{"/test":{"get":{"summary":"Test endpoint","operationId":"test","responses":{"200":{"description":"OK"}}}}}}"#;

        let spec = parse_spec(openapi_json).unwrap();
        assert_eq!(spec.title, "Test");
        assert_eq!(spec.operations.len(), 1);
    }
}
