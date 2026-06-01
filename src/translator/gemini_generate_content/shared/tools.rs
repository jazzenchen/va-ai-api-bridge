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
                input_schema: declaration.get("parameters").cloned(),
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
            if let Some(schema) = &tool.input_schema {
                declaration.insert("parameters".to_string(), schema.clone());
            }
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
