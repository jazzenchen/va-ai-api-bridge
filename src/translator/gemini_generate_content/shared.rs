use serde_json::{json, Map, Value};

use crate::translator::common;
use crate::{ContentBlock, FinishReason, GenerationConfig, Role, ToolChoice, UniversalTool, Usage};

pub(super) const VA_MODEL_KEY: &str = "__va_model";
pub(super) const VA_STREAM_KEY: &str = "__va_stream";

pub fn attach_route_metadata(body: &mut Value, model: &str, stream: bool) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    object.insert(VA_MODEL_KEY.to_string(), Value::String(model.to_string()));
    object.insert(VA_STREAM_KEY.to_string(), Value::Bool(stream));
}

pub fn strip_route_metadata(body: &mut Value) {
    if let Some(object) = body.as_object_mut() {
        object.remove(VA_MODEL_KEY);
        object.remove(VA_STREAM_KEY);
    }
}

pub(super) fn model_from_route_segment(value: &str) -> String {
    value.strip_prefix("models/").unwrap_or(value).to_string()
}

pub(super) fn gemini_role_to_universal(role: &str) -> Role {
    match role {
        "model" => Role::Assistant,
        "function" => Role::Tool,
        _ => Role::User,
    }
}

pub(super) fn universal_role_to_gemini(role: Role) -> &'static str {
    match role {
        Role::Assistant => "model",
        Role::Tool => "function",
        Role::Developer | Role::System | Role::User => "user",
    }
}

pub(super) fn finish_reason_from_gemini(value: &str) -> FinishReason {
    match value {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => {
            FinishReason::ContentFilter
        }
        "MALFORMED_FUNCTION_CALL" => FinishReason::ToolCall,
        _ => FinishReason::Unknown,
    }
}

pub(super) fn finish_reason_to_gemini(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "STOP",
        FinishReason::Length => "MAX_TOKENS",
        FinishReason::ToolCall => "STOP",
        FinishReason::ContentFilter => "SAFETY",
        FinishReason::Error => "OTHER",
        FinishReason::Unknown => "FINISH_REASON_UNSPECIFIED",
    }
}

pub(super) fn has_finish_reason(raw: &Value) -> bool {
    raw.get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|candidate| candidate.get("finishReason").is_some())
}

pub(super) fn usage_from_gemini(value: Option<&Value>) -> Option<Usage> {
    let value = value?;
    Some(Usage {
        input_tokens: value.get("promptTokenCount").and_then(Value::as_u64),
        output_tokens: value.get("candidatesTokenCount").and_then(Value::as_u64),
        total_tokens: value.get("totalTokenCount").and_then(Value::as_u64),
    })
}

pub(super) fn usage_to_gemini(usage: &Usage) -> Value {
    let mut out = Map::new();
    if let Some(input_tokens) = usage.input_tokens {
        out.insert("promptTokenCount".to_string(), json!(input_tokens));
    }
    if let Some(output_tokens) = usage.output_tokens {
        out.insert("candidatesTokenCount".to_string(), json!(output_tokens));
    }
    if let Some(total_tokens) = usage.total_tokens {
        out.insert("totalTokenCount".to_string(), json!(total_tokens));
    }
    Value::Object(out)
}

pub(super) fn generation_from_gemini(value: Option<&Value>) -> GenerationConfig {
    let Some(object) = value.and_then(Value::as_object) else {
        return GenerationConfig::default();
    };
    GenerationConfig {
        temperature: object.get("temperature").and_then(Value::as_f64),
        top_p: field(object, "topP", "top_p").and_then(Value::as_f64),
        max_output_tokens: field(object, "maxOutputTokens", "max_output_tokens")
            .and_then(Value::as_u64),
        extensions: common::empty_extensions(),
    }
}

pub(super) fn generation_to_gemini(generation: &GenerationConfig) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(temperature) = generation.temperature {
        out.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(top_p) = generation.top_p {
        out.insert("topP".to_string(), json!(top_p));
    }
    if let Some(max_output_tokens) = generation.max_output_tokens {
        out.insert("maxOutputTokens".to_string(), json!(max_output_tokens));
    }
    out
}

pub(super) fn decode_tools(value: Option<&Value>) -> Vec<UniversalTool> {
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

pub(super) fn tools_to_gemini(tools: &[UniversalTool]) -> Value {
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

pub(super) fn decode_tool_choice(value: Option<&Value>) -> Option<ToolChoice> {
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

pub(super) fn tool_choice_to_gemini(tool_choice: &ToolChoice) -> Value {
    match tool_choice {
        ToolChoice::Auto => json!({ "functionCallingConfig": { "mode": "AUTO" } }),
        ToolChoice::None => json!({ "functionCallingConfig": { "mode": "NONE" } }),
        ToolChoice::Required => json!({ "functionCallingConfig": { "mode": "ANY" } }),
        ToolChoice::Tool { name } => {
            json!({ "functionCallingConfig": { "mode": "ANY", "allowedFunctionNames": [name] } })
        }
    }
}

pub(super) fn gemini_parts_to_blocks(value: &Value) -> Vec<ContentBlock> {
    match value {
        Value::Array(parts) => parts.iter().flat_map(gemini_part_to_blocks).collect(),
        Value::Object(_) => gemini_part_to_blocks(value),
        _ => Vec::new(),
    }
}

pub(super) fn gemini_part_to_blocks(part: &Value) -> Vec<ContentBlock> {
    if part
        .get("thought")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return vec![ContentBlock::Reasoning {
            text: part
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned),
            encrypted: gemini_thought_signature(part),
            extensions: common::empty_extensions(),
        }];
    }
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        return vec![ContentBlock::Text {
            text: text.to_string(),
        }];
    }
    if let Some(inline) = part.get("inlineData").or_else(|| part.get("inline_data")) {
        let media_type = inline
            .get("mimeType")
            .or_else(|| inline.get("mime_type"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let data = inline
            .get("data")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        return vec![data_block(media_type, None, data)];
    }
    if let Some(file) = part.get("fileData").or_else(|| part.get("file_data")) {
        let media_type = file
            .get("mimeType")
            .or_else(|| file.get("mime_type"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let url = file
            .get("fileUri")
            .or_else(|| file.get("file_uri"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        return vec![data_block(media_type, url, None)];
    }
    if let Some(function_call) = part
        .get("functionCall")
        .or_else(|| part.get("function_call"))
    {
        let name = function_call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        return vec![ContentBlock::ToolCall {
            id: gemini_function_call_id(function_call),
            name,
            arguments: function_call
                .get("args")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new())),
            extensions: common::empty_extensions(),
        }];
    }
    vec![ContentBlock::Unknown { raw: part.clone() }]
}

pub(super) fn blocks_to_gemini_parts(blocks: &[ContentBlock]) -> Vec<Value> {
    blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => json!({ "text": text }),
            ContentBlock::Image {
                media_type,
                url: Some(url),
                ..
            }
            | ContentBlock::File {
                media_type,
                url: Some(url),
                ..
            } => json!({
                "fileData": {
                    "mimeType": media_type.clone().unwrap_or_else(|| "application/octet-stream".to_string()),
                    "fileUri": url,
                }
            }),
            ContentBlock::Image {
                media_type,
                data: Some(data),
                ..
            }
            | ContentBlock::File {
                media_type,
                data: Some(data),
                ..
            } => json!({
                "inlineData": {
                    "mimeType": media_type.clone().unwrap_or_else(|| "application/octet-stream".to_string()),
                    "data": data,
                }
            }),
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
                ..
            } => function_call_part(Some(id), name, arguments.clone()),
            ContentBlock::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => function_response_part(Some(tool_call_id), None, content, *is_error),
            ContentBlock::Reasoning {
                text, encrypted, ..
            } => {
                let mut part = Map::new();
                part.insert("thought".to_string(), Value::Bool(true));
                if let Some(text) = text {
                    part.insert("text".to_string(), Value::String(text.clone()));
                }
                if let Some(encrypted) = encrypted {
                    part.insert(
                        "thoughtSignature".to_string(),
                        Value::String(encrypted.clone()),
                    );
                }
                Value::Object(part)
            }
            ContentBlock::Unknown { raw } => raw.clone(),
            ContentBlock::Image { .. } | ContentBlock::File { .. } => json!({}),
        })
        .collect()
}

pub(super) fn gemini_function_call_id(function_call: &Value) -> String {
    function_call
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| function_call.get("name").and_then(Value::as_str))
        .unwrap_or_default()
        .to_string()
}

pub(super) fn gemini_function_response_id(function_response: &Value) -> String {
    function_response
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| function_response.get("name").and_then(Value::as_str))
        .unwrap_or_default()
        .to_string()
}

pub(super) fn function_call_part(id: Option<&str>, name: &str, args: Value) -> Value {
    let mut function_call = Map::new();
    if let Some(id) = id.filter(|id| !id.is_empty()) {
        function_call.insert("id".to_string(), Value::String(id.to_string()));
    }
    function_call.insert("name".to_string(), Value::String(name.to_string()));
    function_call.insert("args".to_string(), args);
    let mut part = Map::new();
    part.insert("functionCall".to_string(), Value::Object(function_call));
    Value::Object(part)
}

pub(super) fn function_response_part(
    id: Option<&str>,
    name: Option<&str>,
    content: &[ContentBlock],
    is_error: bool,
) -> Value {
    let mut function_response = Map::new();
    if let Some(id) = id.filter(|id| !id.is_empty()) {
        function_response.insert("id".to_string(), Value::String(id.to_string()));
    }
    if let Some(name) = name.filter(|name| !name.is_empty()) {
        function_response.insert("name".to_string(), Value::String(name.to_string()));
    }
    function_response.insert(
        "response".to_string(),
        json!({
            "content": blocks_to_text(content),
            "is_error": is_error,
        }),
    );
    let mut part = Map::new();
    part.insert(
        "functionResponse".to_string(),
        Value::Object(function_response),
    );
    Value::Object(part)
}

pub(super) fn blocks_to_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn stringify_json(value: Value) -> String {
    match value {
        Value::String(value) => value,
        other => serde_json::to_string(&other).unwrap_or_default(),
    }
}

fn gemini_thought_signature(part: &Value) -> Option<String> {
    part.get("thoughtSignature")
        .or_else(|| part.get("thought_signature"))
        .and_then(Value::as_str)
        .filter(|signature| !signature.is_empty())
        .map(ToOwned::to_owned)
}

fn data_block(
    media_type: Option<String>,
    url: Option<String>,
    data: Option<String>,
) -> ContentBlock {
    if media_type
        .as_deref()
        .is_some_and(|media_type| media_type.starts_with("image/"))
    {
        ContentBlock::Image {
            media_type,
            url,
            data,
            extensions: common::empty_extensions(),
        }
    } else {
        ContentBlock::File {
            media_type,
            filename: None,
            url,
            data,
            extensions: common::empty_extensions(),
        }
    }
}

pub(super) fn field<'a>(
    object: &'a Map<String, Value>,
    camel_case: &str,
    snake_case: &str,
) -> Option<&'a Value> {
    object.get(camel_case).or_else(|| object.get(snake_case))
}
