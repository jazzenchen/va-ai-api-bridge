use serde_json::{json, Map, Value};

use crate::{ToolChoice, UniversalTool};

use super::super::common::empty_extensions;

pub(crate) fn tool_choice_from_anthropic_value(value: Option<&Value>) -> Option<ToolChoice> {
    match value {
        Some(Value::Object(object)) => match object.get("type").and_then(Value::as_str) {
            Some("auto") => Some(ToolChoice::Auto),
            Some("none") => Some(ToolChoice::None),
            Some("any") => Some(ToolChoice::Required),
            Some("tool") => {
                object
                    .get("name")
                    .and_then(Value::as_str)
                    .map(|name| ToolChoice::Tool {
                        name: name.to_string(),
                    })
            }
            _ => None,
        },
        Some(Value::String(value)) => match value.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "any" | "required" => Some(ToolChoice::Required),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn tool_choice_to_anthropic(value: &ToolChoice) -> Value {
    match value {
        ToolChoice::Auto => json!({ "type": "auto" }),
        ToolChoice::None => json!({ "type": "none" }),
        ToolChoice::Required => json!({ "type": "any" }),
        ToolChoice::Tool { name } => json!({
            "type": "tool",
            "name": name
        }),
    }
}

pub(crate) fn anthropic_tool_from_value(value: &Value) -> Option<UniversalTool> {
    let object = value.as_object()?;
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())?;
    Some(UniversalTool {
        name: name.to_string(),
        description: object
            .get("description")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        input_schema: object.get("input_schema").cloned(),
        strict: None,
        extensions: empty_extensions(),
    })
}

pub(crate) fn tool_to_anthropic(tool: &UniversalTool) -> Value {
    let mut object = Map::new();
    object.insert("name".to_string(), Value::String(tool.name.clone()));
    if let Some(description) = &tool.description {
        object.insert(
            "description".to_string(),
            Value::String(description.clone()),
        );
    }
    object.insert(
        "input_schema".to_string(),
        sanitize_anthropic_input_schema(tool.input_schema.as_ref()),
    );
    Value::Object(object)
}

fn sanitize_anthropic_input_schema(input_schema: Option<&Value>) -> Value {
    let Some(input_schema) = input_schema else {
        return empty_object_schema();
    };
    match sanitize_schema_slot(input_schema, true) {
        Value::Object(object) if !object.is_empty() => Value::Object(object),
        _ => json!({
            "type": "object",
            "properties": {}
        }),
    }
}

fn sanitize_schema_slot(value: &Value, root: bool) -> Value {
    match value {
        Value::Null => {
            if root {
                empty_object_schema()
            } else {
                Value::Bool(true)
            }
        }
        Value::Bool(_) if !root => value.clone(),
        Value::Object(object) => Value::Object(sanitize_schema_object(object)),
        _ => {
            if root {
                empty_object_schema()
            } else {
                Value::Bool(true)
            }
        }
    }
}

fn empty_object_schema() -> Value {
    json!({
        "type": "object",
        "properties": {}
    })
}

fn sanitize_schema_object(object: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    for (key, value) in object {
        if let Some(sanitized) = sanitize_schema_keyword(key, value) {
            out.insert(key.clone(), sanitized);
        }
    }
    out
}

fn sanitize_schema_keyword(key: &str, value: &Value) -> Option<Value> {
    match key {
        "additionalProperties"
        | "unevaluatedProperties"
        | "additionalItems"
        | "items"
        | "contains"
        | "propertyNames"
        | "not"
        | "if"
        | "then"
        | "else" => Some(sanitize_schema_slot(value, false)),
        "properties" | "patternProperties" | "$defs" | "definitions" | "dependentSchemas" => {
            let object = value.as_object()?;
            Some(Value::Object(
                object
                    .iter()
                    .map(|(name, schema)| (name.clone(), sanitize_schema_slot(schema, false)))
                    .collect(),
            ))
        }
        "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
            let array = value.as_array()?;
            Some(Value::Array(
                array
                    .iter()
                    .map(|schema| sanitize_schema_slot(schema, false))
                    .collect(),
            ))
        }
        _ => Some(sanitize_schema_metadata(value)),
    }
}

fn sanitize_schema_metadata(value: &Value) -> Value {
    match value {
        Value::Array(array) => Value::Array(array.iter().map(sanitize_schema_metadata).collect()),
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), sanitize_schema_metadata(value)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn replaces_null_additional_properties_for_anthropic_tools() {
        let tool = UniversalTool {
            name: "mcp__codex_apps__github".to_string(),
            description: None,
            input_schema: Some(json!({
                "type": "object",
                "properties": {
                    "owner": { "type": "string" }
                },
                "additionalProperties": null
            })),
            strict: None,
            extensions: Default::default(),
        };

        let encoded = tool_to_anthropic(&tool);

        assert_eq!(encoded["input_schema"]["additionalProperties"], true);
    }

    #[test]
    fn replaces_null_property_schemas_but_keeps_enum_nulls() {
        let tool = UniversalTool {
            name: "example".to_string(),
            description: None,
            input_schema: Some(json!({
                "type": "object",
                "properties": {
                    "anything": null,
                    "nullable": {
                        "enum": ["value", null],
                        "default": null
                    }
                }
            })),
            strict: None,
            extensions: Default::default(),
        };

        let encoded = tool_to_anthropic(&tool);

        assert_eq!(encoded["input_schema"]["properties"]["anything"], true);
        assert_eq!(
            encoded["input_schema"]["properties"]["nullable"]["enum"],
            json!(["value", null])
        );
        assert_eq!(
            encoded["input_schema"]["properties"]["nullable"]["default"],
            Value::Null
        );
    }

    #[test]
    fn null_root_schema_becomes_empty_object_schema() {
        let tool = UniversalTool {
            name: "example".to_string(),
            description: None,
            input_schema: Some(Value::Null),
            strict: None,
            extensions: Default::default(),
        };

        let encoded = tool_to_anthropic(&tool);

        assert_eq!(
            encoded["input_schema"],
            json!({
                "type": "object",
                "properties": {}
            })
        );
    }

    #[test]
    fn missing_root_schema_becomes_empty_object_schema() {
        let tool = UniversalTool {
            name: "example".to_string(),
            description: None,
            input_schema: None,
            strict: None,
            extensions: Default::default(),
        };

        let encoded = tool_to_anthropic(&tool);

        assert_eq!(
            encoded["input_schema"],
            json!({
                "type": "object",
                "properties": {}
            })
        );
    }

    #[test]
    fn empty_root_schema_becomes_empty_object_schema() {
        let tool = UniversalTool {
            name: "example".to_string(),
            description: None,
            input_schema: Some(json!({})),
            strict: None,
            extensions: Default::default(),
        };

        let encoded = tool_to_anthropic(&tool);

        assert_eq!(
            encoded["input_schema"],
            json!({
                "type": "object",
                "properties": {}
            })
        );
    }

    #[test]
    fn skips_tools_with_blank_names() {
        assert!(anthropic_tool_from_value(&json!({
            "name": " ",
            "input_schema": { "type": "object" }
        }))
        .is_none());
    }

    #[test]
    fn ignores_anthropic_top_level_strict_tool_setting() {
        let tool = anthropic_tool_from_value(&json!({
            "name": "search",
            "input_schema": { "type": "object" },
            "strict": true
        }))
        .expect("tool");

        assert_eq!(tool.strict, None);
    }

    #[test]
    fn drops_strict_for_anthropic_tools() {
        let tool = UniversalTool {
            name: "search".to_string(),
            description: None,
            input_schema: Some(json!({ "type": "object" })),
            strict: Some(true),
            extensions: Default::default(),
        };

        let encoded = tool_to_anthropic(&tool);

        assert!(encoded.get("strict").is_none());
        assert!(encoded["input_schema"].get("strict").is_none());
    }
}
