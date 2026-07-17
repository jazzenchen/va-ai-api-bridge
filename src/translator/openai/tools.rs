use serde_json::{json, Map, Value};

use crate::{ServerToolDeclaration, ServerToolKind, ToolChoice, UniversalTool, WireProtocol};

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

pub(crate) fn tool_choice_from_openai_responses_value(value: Option<&Value>) -> Option<ToolChoice> {
    match value {
        Some(Value::Object(object)) => object
            .get("type")
            .and_then(Value::as_str)
            .and_then(openai_server_tool_kind)
            .map(|kind| ToolChoice::ServerTool { kind })
            .or_else(|| tool_choice_from_openai_value(value)),
        _ => tool_choice_from_openai_value(value),
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
        ToolChoice::ServerTool { .. } => json!("auto"),
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
        ToolChoice::ServerTool { .. } => json!("auto"),
    }
}

pub(crate) fn openai_tool_from_value(value: &Value) -> Option<UniversalTool> {
    let object = value.as_object()?;
    if object
        .get("type")
        .and_then(Value::as_str)
        .and_then(openai_server_tool_kind)
        .is_some()
    {
        return None;
    }
    let function = object
        .get("function")
        .and_then(Value::as_object)
        .unwrap_or(object);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())?;
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
        strict: function.get("strict").and_then(Value::as_bool),
        extensions: empty_extensions(),
    })
}

pub(crate) fn openai_server_tool_from_value(value: &Value) -> Option<ServerToolDeclaration> {
    let object = value.as_object()?;
    let wire_type = object.get("type").and_then(Value::as_str)?;
    let kind = openai_server_tool_kind(wire_type)?;
    Some(ServerToolDeclaration {
        kind,
        wire_type: wire_type.to_string(),
        source_protocol: WireProtocol::OpenAiResponses,
        name: object
            .get("name")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        config: server_tool_config(object),
        raw: value.clone(),
        extensions: empty_extensions(),
    })
}

fn openai_server_tool_kind(wire_type: &str) -> Option<ServerToolKind> {
    match wire_type {
        "web_search" | "web_search_preview" => Some(ServerToolKind::WebSearch),
        "x_search" => Some(ServerToolKind::XSearch),
        "file_search" => Some(ServerToolKind::FileSearch),
        "code_interpreter" => Some(ServerToolKind::CodeInterpreter),
        "code_execution" => Some(ServerToolKind::CodeExecution),
        _ => None,
    }
}

fn server_tool_config(object: &Map<String, Value>) -> Value {
    let mut config = object.clone();
    config.remove("type");
    config.remove("name");
    if config.is_empty() {
        Value::Null
    } else {
        Value::Object(config)
    }
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
    function.insert(
        "parameters".to_string(),
        sanitize_openai_parameters(tool.input_schema.as_ref()),
    );
    if let Some(strict) = tool.strict {
        function.insert("strict".to_string(), Value::Bool(strict));
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
    object.insert(
        "parameters".to_string(),
        sanitize_openai_parameters(tool.input_schema.as_ref()),
    );
    if let Some(strict) = tool.strict {
        object.insert("strict".to_string(), Value::Bool(strict));
    }
    Value::Object(object)
}

fn sanitize_openai_parameters(input_schema: Option<&Value>) -> Value {
    let Some(Value::Object(object)) = input_schema else {
        return empty_object_schema();
    };
    if object.is_empty() {
        return empty_object_schema();
    }
    Value::Object(object.clone())
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
    fn skips_tools_with_blank_names() {
        assert!(openai_tool_from_value(&json!({
            "type": "function",
            "name": " ",
            "parameters": { "type": "object" }
        }))
        .is_none());
    }

    #[test]
    fn chat_tools_always_include_non_empty_parameters() {
        let tool = UniversalTool {
            name: "list_files".to_string(),
            description: None,
            input_schema: None,
            strict: None,
            extensions: Default::default(),
        };

        let encoded = tool_to_openai_chat(&tool);

        assert_eq!(
            encoded["function"]["parameters"],
            json!({
                "type": "object",
                "properties": {}
            })
        );
    }

    #[test]
    fn responses_tools_normalize_empty_parameter_objects() {
        let tool = UniversalTool {
            name: "list_files".to_string(),
            description: None,
            input_schema: Some(json!({})),
            strict: None,
            extensions: Default::default(),
        };

        let encoded = tool_to_openai_responses(&tool);

        assert_eq!(
            encoded["parameters"],
            json!({
                "type": "object",
                "properties": {}
            })
        );
    }

    #[test]
    fn decodes_openai_strict_tool_setting() {
        let chat_tool = openai_tool_from_value(&json!({
            "type": "function",
            "function": {
                "name": "search",
                "parameters": { "type": "object" },
                "strict": true
            }
        }))
        .expect("tool");
        let responses_tool = openai_tool_from_value(&json!({
            "type": "function",
            "name": "search",
            "parameters": { "type": "object" },
            "strict": false
        }))
        .expect("tool");

        assert_eq!(chat_tool.strict, Some(true));
        assert_eq!(responses_tool.strict, Some(false));
    }

    #[test]
    fn encodes_strict_for_openai_chat_and_responses_tools() {
        let tool = UniversalTool {
            name: "search".to_string(),
            description: None,
            input_schema: Some(json!({ "type": "object" })),
            strict: Some(true),
            extensions: Default::default(),
        };

        let chat = tool_to_openai_chat(&tool);
        let responses = tool_to_openai_responses(&tool);

        assert_eq!(chat["function"]["strict"], true);
        assert_eq!(responses["strict"], true);
    }
}
