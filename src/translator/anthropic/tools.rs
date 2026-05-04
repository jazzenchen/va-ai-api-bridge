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
        object.insert("input_schema".to_string(), input_schema.clone());
    }
    Value::Object(object)
}
