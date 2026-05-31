use serde_json::{json, Map, Value};

use crate::translator::common;
use crate::{ContentBlock, Extensions};

use super::GEMINI_THOUGHT_SIGNATURE_KEY;

pub(in crate::translator::gemini_generate_content) fn gemini_parts_to_blocks(
    value: &Value,
) -> Vec<ContentBlock> {
    match value {
        Value::Array(parts) => parts.iter().flat_map(gemini_part_to_blocks).collect(),
        Value::Object(_) => gemini_part_to_blocks(value),
        _ => Vec::new(),
    }
}

pub(in crate::translator::gemini_generate_content) fn gemini_part_to_blocks(
    part: &Value,
) -> Vec<ContentBlock> {
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
        let mut extensions = common::empty_extensions();
        if let Some(signature) = gemini_thought_signature(part) {
            extensions.insert(GEMINI_THOUGHT_SIGNATURE_KEY.to_string(), json!(signature));
        }
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
            extensions,
        }];
    }
    vec![ContentBlock::Unknown { raw: part.clone() }]
}

pub(in crate::translator::gemini_generate_content) fn blocks_to_gemini_parts(
    blocks: &[ContentBlock],
) -> Vec<Value> {
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
                extensions,
                ..
            } => function_call_part_with_signature(
                Some(id),
                name,
                arguments.clone(),
                thought_signature_from_extensions(extensions),
            ),
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

pub(in crate::translator::gemini_generate_content) fn gemini_function_call_id(
    function_call: &Value,
) -> String {
    function_call
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| function_call.get("name").and_then(Value::as_str))
        .unwrap_or_default()
        .to_string()
}

pub(in crate::translator::gemini_generate_content) fn gemini_function_response_id(
    function_response: &Value,
) -> String {
    function_response
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| function_response.get("name").and_then(Value::as_str))
        .unwrap_or_default()
        .to_string()
}

pub(in crate::translator::gemini_generate_content) fn function_call_part(
    id: Option<&str>,
    name: &str,
    args: Value,
) -> Value {
    function_call_part_with_signature(id, name, args, None)
}

pub(in crate::translator::gemini_generate_content) fn function_call_part_with_signature(
    id: Option<&str>,
    name: &str,
    args: Value,
    thought_signature: Option<&str>,
) -> Value {
    let mut function_call = Map::new();
    if let Some(id) = id.filter(|id| !id.is_empty()) {
        function_call.insert("id".to_string(), Value::String(id.to_string()));
    }
    function_call.insert("name".to_string(), Value::String(name.to_string()));
    function_call.insert("args".to_string(), args);
    let mut part = Map::new();
    part.insert("functionCall".to_string(), Value::Object(function_call));
    if let Some(thought_signature) = thought_signature.filter(|signature| !signature.is_empty()) {
        part.insert(
            GEMINI_THOUGHT_SIGNATURE_KEY.to_string(),
            Value::String(thought_signature.to_string()),
        );
    }
    Value::Object(part)
}

pub(in crate::translator::gemini_generate_content) fn function_response_part(
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

pub(in crate::translator::gemini_generate_content) fn blocks_to_text(
    blocks: &[ContentBlock],
) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(in crate::translator::gemini_generate_content) fn stringify_json(value: Value) -> String {
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

pub(in crate::translator::gemini_generate_content) fn thought_signature_from_extensions(
    extensions: &Extensions,
) -> Option<&str> {
    extensions
        .get(GEMINI_THOUGHT_SIGNATURE_KEY)
        .or_else(|| extensions.get("thought_signature"))
        .and_then(Value::as_str)
        .filter(|signature| !signature.is_empty())
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
