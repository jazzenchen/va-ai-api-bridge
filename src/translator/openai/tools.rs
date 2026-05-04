use serde_json::{json, Map, Value};

use crate::{ToolChoice, UniversalTool};

use super::super::common::empty_extensions;

pub(crate) fn tool_choice_from_openai_value(value: Option<&Value>) -> Option<ToolChoice> {
    match value {
        Some(Value::String(value)) => match value.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "required" => Some(ToolChoice::Required),
            _ => None,
        },
        Some(Value::Object(object)) => object
            .get("function")
            .and_then(Value::as_object)
            .and_then(|function| function.get("name"))
            .or_else(|| object.get("name"))
            .and_then(Value::as_str)
            .map(|name| ToolChoice::Tool {
                name: name.to_string(),
            }),
        _ => None,
    }
}

pub(crate) fn tool_choice_to_openai(value: &ToolChoice) -> Value {
    match value {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool { name } => json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

pub(crate) fn tool_choice_to_openai_responses(value: &ToolChoice) -> Value {
    match value {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool { name } => json!({
            "type": "function",
            "name": name
        }),
    }
}

pub(crate) fn openai_tool_from_value(value: &Value) -> Option<UniversalTool> {
    let object = value.as_object()?;
    let function = object
        .get("function")
        .and_then(Value::as_object)
        .unwrap_or(object);
    let name = function.get("name").and_then(Value::as_str)?;
    Some(UniversalTool {
        name: name.to_string(),
        description: function
            .get("description")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        input_schema: function
            .get("parameters")
            .or_else(|| function.get("input_schema"))
            .cloned(),
        extensions: empty_extensions(),
    })
}

pub(crate) fn tool_to_openai_chat(tool: &UniversalTool) -> Value {
    let mut function = Map::new();
    function.insert("name".to_string(), Value::String(tool.name.clone()));
    if let Some(description) = &tool.description {
        function.insert(
            "description".to_string(),
            Value::String(description.clone()),
        );
    }
    if let Some(input_schema) = &tool.input_schema {
        function.insert("parameters".to_string(), input_schema.clone());
    }
    json!({
        "type": "function",
        "function": function
    })
}

pub(crate) fn tool_to_openai_responses(tool: &UniversalTool) -> Value {
    let mut object = Map::new();
    object.insert("type".to_string(), Value::String("function".to_string()));
    object.insert("name".to_string(), Value::String(tool.name.clone()));
    if let Some(description) = &tool.description {
        object.insert(
            "description".to_string(),
            Value::String(description.clone()),
        );
    }
    if let Some(input_schema) = &tool.input_schema {
        object.insert("parameters".to_string(), input_schema.clone());
    }
    Value::Object(object)
}
