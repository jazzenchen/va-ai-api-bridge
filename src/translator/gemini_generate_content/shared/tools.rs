use serde_json::{json, Map, Value};

use crate::translator::common;
use crate::{ToolChoice, UniversalTool};

pub(in crate::translator::gemini_generate_content) fn decode_tools(
    value: Option<&Value>,
) -> Vec<UniversalTool> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|tool| {
            tool.get("functionDeclarations")
                .or_else(|| tool.get("function_declarations"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|declaration| {
            let name = declaration.get("name")?.as_str()?.to_string();
            Some(UniversalTool {
                name,
                description: declaration
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                input_schema: declaration
                    .get("parameters")
                    .map(|schema| sanitize_gemini_parameters(Some(schema))),
                strict: None,
                extensions: common::empty_extensions(),
            })
        })
        .collect()
}

pub(in crate::translator::gemini_generate_content) fn tools_to_gemini(
    tools: &[UniversalTool],
) -> Value {
    Value::Array(vec![json!({
        "functionDeclarations": tools.iter().map(|tool| {
            let mut declaration = Map::new();
            declaration.insert("name".to_string(), Value::String(tool.name.clone()));
            if let Some(description) = &tool.description {
                declaration.insert("description".to_string(), Value::String(description.clone()));
            }
            declaration.insert(
                "parameters".to_string(),
                sanitize_gemini_parameters(tool.input_schema.as_ref()),
            );
            Value::Object(declaration)
        }).collect::<Vec<_>>()
    })])
}

pub(in crate::translator::gemini_generate_content) fn decode_tool_choice(
    value: Option<&Value>,
) -> Option<ToolChoice> {
    let value = value?;
    let config = value
        .get("functionCallingConfig")
        .or_else(|| value.get("function_calling_config"))?;
    match config.get("mode").and_then(Value::as_str) {
        Some("NONE") => Some(ToolChoice::None),
        Some("ANY") => config
            .get("allowedFunctionNames")
            .or_else(|| config.get("allowed_function_names"))
            .and_then(Value::as_array)
            .and_then(|names| names.first())
            .and_then(Value::as_str)
            .map(|name| ToolChoice::Tool {
                name: name.to_string(),
            })
            .or(Some(ToolChoice::Required)),
        Some("AUTO") => Some(ToolChoice::Auto),
        _ => None,
    }
}

pub(in crate::translator::gemini_generate_content) fn tool_choice_to_gemini(
    tool_choice: &ToolChoice,
) -> Value {
    match tool_choice {
        ToolChoice::Auto => json!({ "functionCallingConfig": { "mode": "AUTO" } }),
        ToolChoice::None => json!({ "functionCallingConfig": { "mode": "NONE" } }),
        ToolChoice::Required => json!({ "functionCallingConfig": { "mode": "ANY" } }),
        ToolChoice::Tool { name } => {
            json!({ "functionCallingConfig": { "mode": "ANY", "allowedFunctionNames": [name] } })
        }
    }
}

fn sanitize_gemini_parameters(input_schema: Option<&Value>) -> Value {
    let Some(schema) = input_schema else {
        return empty_object_schema();
    };
    match sanitize_gemini_schema(schema, true) {
        Value::Object(mut object) if !object.is_empty() => {
            object
                .entry("type".to_string())
                .or_insert_with(|| Value::String("object".to_string()));
            Value::Object(object)
        }
        _ => empty_object_schema(),
    }
}

fn sanitize_gemini_schema(schema: &Value, root: bool) -> Value {
    let Value::Object(object) = schema else {
        return if root {
            empty_object_schema()
        } else {
            Value::Object(schema_with_type("string"))
        };
    };
    let mut out = Map::new();

    for (key, value) in object {
        match key.as_str() {
            "type" => {
                if let Some(schema_type) = sanitize_schema_type(value) {
                    out.insert(key.clone(), Value::String(schema_type));
                }
            }
            "format" | "title" | "description" | "pattern" => {
                if let Some(value) = value.as_str().filter(|value| !value.is_empty()) {
                    out.insert(key.clone(), Value::String(value.to_string()));
                }
            }
            "nullable" => {
                if let Some(value) = value.as_bool() {
                    out.insert(key.clone(), Value::Bool(value));
                }
            }
            "enum" => {
                if let Some(values) = sanitize_string_array(value) {
                    out.insert(key.clone(), values);
                }
            }
            "maxItems" | "minItems" | "minProperties" | "maxProperties" | "minLength"
            | "maxLength" => {
                if let Some(value) = sanitize_int64_string(value) {
                    out.insert(key.clone(), Value::String(value));
                }
            }
            "properties" => {
                if let Some(properties) = sanitize_gemini_properties(value) {
                    out.insert(key.clone(), properties);
                }
            }
            "required" | "propertyOrdering" => {
                if let Some(values) = sanitize_string_array(value) {
                    out.insert(key.clone(), values);
                }
            }
            "example" | "default" => {
                out.insert(key.clone(), value.clone());
            }
            "anyOf" => {
                if let Some(values) = sanitize_gemini_schema_array(value) {
                    out.insert(key.clone(), values);
                }
            }
            "items" => {
                out.insert(key.clone(), sanitize_gemini_schema(value, false));
            }
            "minimum" | "maximum" => {
                if value.is_number() {
                    out.insert(key.clone(), value.clone());
                }
            }
            _ => {}
        }
    }

    if !out.contains_key("type") {
        let inferred_type = if out.contains_key("properties") {
            Some("object")
        } else if out.contains_key("items") {
            Some("array")
        } else if out.contains_key("enum") {
            Some("string")
        } else if root {
            Some("object")
        } else {
            None
        };
        if let Some(inferred_type) = inferred_type {
            out.insert("type".to_string(), Value::String(inferred_type.to_string()));
        }
    }

    if out.is_empty() && !root {
        out = schema_with_type("string");
    }

    Value::Object(out)
}

fn sanitize_schema_type(value: &Value) -> Option<String> {
    let schema_type = value.as_str()?.to_ascii_lowercase();
    matches!(
        schema_type.as_str(),
        "string" | "number" | "integer" | "boolean" | "array" | "object" | "null"
    )
    .then_some(schema_type)
}

fn sanitize_gemini_properties(value: &Value) -> Option<Value> {
    let object = value.as_object()?;
    Some(Value::Object(
        object
            .iter()
            .map(|(name, schema)| (name.clone(), sanitize_gemini_schema(schema, false)))
            .collect(),
    ))
}

fn sanitize_gemini_schema_array(value: &Value) -> Option<Value> {
    let array = value.as_array()?;
    let values = array
        .iter()
        .filter_map(|schema| match sanitize_gemini_schema(schema, false) {
            Value::Object(object) if !object.is_empty() => Some(Value::Object(object)),
            _ => None,
        })
        .collect::<Vec<_>>();
    (!values.is_empty()).then_some(Value::Array(values))
}

fn sanitize_string_array(value: &Value) -> Option<Value> {
    let values = value
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(|value| Value::String(value.to_string()))
        .collect::<Vec<_>>();
    (!values.is_empty()).then_some(Value::Array(values))
}

fn sanitize_int64_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .or_else(|| value.as_i64().map(|value| value.to_string()))
        .or_else(|| value.as_u64().map(|value| value.to_string()))
}

fn schema_with_type(schema_type: &str) -> Map<String, Value> {
    let mut object = Map::new();
    object.insert("type".to_string(), Value::String(schema_type.to_string()));
    object
}

fn empty_object_schema() -> Value {
    json!({
        "type": "object",
        "properties": {}
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignores_and_sanitizes_gemini_strict_tool_settings() {
        let raw = json!([{
            "functionDeclarations": [{
                "name": "search",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "strict": true,
                    "properties": {
                        "query": {
                            "type": "string",
                            "strict": true
                        }
                    }
                }
            }]
        }]);

        let decoded = decode_tools(Some(&raw));

        assert_eq!(decoded[0].strict, None);
        assert!(decoded[0]
            .input_schema
            .as_ref()
            .and_then(|schema| schema.get("strict"))
            .is_none());
        assert!(decoded[0]
            .input_schema
            .as_ref()
            .and_then(|schema| schema["properties"]["query"].get("strict"))
            .is_none());
    }

    #[test]
    fn strips_unsupported_json_schema_keywords_from_tool_parameters() {
        let tool = UniversalTool {
            name: "list_rules".to_string(),
            description: None,
            input_schema: Some(json!({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "filters": {
                        "type": "object",
                        "propertyNames": { "pattern": "^[a-z]+$" },
                        "additionalProperties": { "type": "string" },
                        "properties": {
                            "name": { "type": "string" }
                        }
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["filters"]
            })),
            strict: Some(true),
            extensions: Default::default(),
        };

        let encoded = tools_to_gemini(&[tool]);
        let declaration = &encoded[0]["functionDeclarations"][0];
        assert!(declaration.get("strict").is_none());
        let parameters = &encoded[0]["functionDeclarations"][0]["parameters"];

        assert!(parameters.get("$schema").is_none());
        assert!(parameters.get("strict").is_none());
        assert!(parameters.get("additionalProperties").is_none());
        assert_eq!(parameters["type"], "object");
        assert_eq!(parameters["required"], json!(["filters"]));
        assert!(parameters["properties"]["filters"]
            .get("propertyNames")
            .is_none());
        assert!(parameters["properties"]["filters"]
            .get("additionalProperties")
            .is_none());
        assert_eq!(
            parameters["properties"]["filters"]["properties"]["name"]["type"],
            "string"
        );
        assert_eq!(parameters["properties"]["limit"]["minimum"], 1);
        assert_eq!(parameters["properties"]["limit"]["maximum"], 100);
    }

    #[test]
    fn missing_gemini_tool_schema_becomes_empty_object_schema() {
        let tool = UniversalTool {
            name: "list_rules".to_string(),
            description: None,
            input_schema: None,
            strict: None,
            extensions: Default::default(),
        };

        let encoded = tools_to_gemini(&[tool]);

        assert_eq!(
            encoded[0]["functionDeclarations"][0]["parameters"],
            json!({
                "type": "object",
                "properties": {}
            })
        );
    }
}
