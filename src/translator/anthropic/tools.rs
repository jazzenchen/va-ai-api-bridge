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
    let name = object.get("name").and_then(Value::as_str)?;
    Some(UniversalTool {
        name: name.to_string(),
        description: object
            .get("description")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        input_schema: object.get("input_schema").cloned(),
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
    if let Some(input_schema) = &tool.input_schema {
        object.insert(
            "input_schema".to_string(),
            sanitize_anthropic_input_schema(input_schema),
        );
    }
    Value::Object(object)
}

fn sanitize_anthropic_input_schema(input_schema: &Value) -> Value {
    match sanitize_schema_slot(input_schema, true) {
        Value::Object(object) => Value::Object(object),
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
                json!({
                    "type": "object",
                    "properties": {}
                })
            } else {
                Value::Bool(true)
            }
        }
        Value::Bool(_) if !root => value.clone(),
        Value::Object(object) => Value::Object(sanitize_schema_object(object)),
        _ => {
            if root {
                json!({
                    "type": "object",
                    "properties": {}
                })
            } else {
                Value::Bool(true)
            }
        }
    }
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
}
