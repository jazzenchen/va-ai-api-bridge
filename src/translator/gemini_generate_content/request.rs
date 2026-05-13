use serde_json::{json, Map, Value};

use crate::translator::common;
use crate::{
    ApiProxyError, ContentBlock, Result, Role, UniversalItem, UniversalRequest, WireProtocol,
};

use super::shared::{
    blocks_to_gemini_parts, decode_tool_choice, decode_tools, function_call_part,
    function_response_part, gemini_function_call_id, gemini_function_response_id,
    gemini_part_to_blocks, gemini_parts_to_blocks, gemini_role_to_universal,
    generation_from_gemini, generation_to_gemini, model_from_route_segment, stringify_json,
    tool_choice_to_gemini, tools_to_gemini, universal_role_to_gemini, VA_MODEL_KEY, VA_STREAM_KEY,
};

pub(super) fn decode_request(raw: Value) -> Result<UniversalRequest> {
    let source_raw = raw.clone();
    let object = raw
        .as_object()
        .ok_or_else(|| ApiProxyError::invalid_request("Gemini request must be a JSON object"))?;
    let mut request = UniversalRequest {
        model: object
            .get(VA_MODEL_KEY)
            .and_then(Value::as_str)
            .map(model_from_route_segment),
        stream: object
            .get(VA_STREAM_KEY)
            .and_then(Value::as_bool)
            .unwrap_or(false),
        generation: generation_from_gemini(object.get("generationConfig")),
        source: Some(common::source(
            WireProtocol::GeminiGenerateContent,
            source_raw,
        )),
        ..UniversalRequest::default()
    };
    request.instructions = decode_system_instruction(object.get("systemInstruction"));
    request.input = decode_contents(object.get("contents"))?;
    request.tools = decode_tools(object.get("tools"));
    request.tool_choice = decode_tool_choice(object.get("toolConfig"));
    Ok(request)
}

pub(super) fn encode_request(request: &UniversalRequest) -> Result<Value> {
    let mut body = Map::new();
    body.insert(
        VA_MODEL_KEY.to_string(),
        Value::String(request.model.clone().unwrap_or_default()),
    );
    body.insert(VA_STREAM_KEY.to_string(), Value::Bool(request.stream));
    if !request.instructions.is_empty() {
        body.insert(
            "systemInstruction".to_string(),
            json!({ "parts": blocks_to_gemini_parts(&request.instructions) }),
        );
    }
    body.insert(
        "contents".to_string(),
        Value::Array(items_to_contents(&request.input)),
    );
    if !request.tools.is_empty() {
        body.insert("tools".to_string(), tools_to_gemini(&request.tools));
    }
    if let Some(tool_choice) = &request.tool_choice {
        body.insert("toolConfig".to_string(), tool_choice_to_gemini(tool_choice));
    }
    let generation = generation_to_gemini(&request.generation);
    if !generation.is_empty() {
        body.insert("generationConfig".to_string(), Value::Object(generation));
    }
    Ok(Value::Object(body))
}

fn decode_system_instruction(value: Option<&Value>) -> Vec<ContentBlock> {
    value
        .and_then(|value| value.get("parts"))
        .map(gemini_parts_to_blocks)
        .unwrap_or_default()
}

fn decode_contents(value: Option<&Value>) -> Result<Vec<UniversalItem>> {
    let Some(contents) = value.and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for content in contents {
        let role = content
            .get("role")
            .and_then(Value::as_str)
            .map(gemini_role_to_universal)
            .unwrap_or(Role::User);
        let parts = content.get("parts").and_then(Value::as_array);
        let mut message_blocks = Vec::new();
        if let Some(parts) = parts {
            for part in parts {
                if let Some(function_response) = part.get("functionResponse") {
                    push_message_if_any(&mut out, role, &mut message_blocks);
                    out.push(function_response_to_tool_result(function_response));
                    continue;
                }
                if let Some(function_call) = part.get("functionCall") {
                    push_message_if_any(&mut out, role, &mut message_blocks);
                    out.push(function_call_to_tool_call(function_call));
                    continue;
                }
                message_blocks.extend(gemini_part_to_blocks(part));
            }
        }
        push_message_if_any(&mut out, role, &mut message_blocks);
    }
    Ok(out)
}

fn function_response_to_tool_result(function_response: &Value) -> UniversalItem {
    let response = function_response
        .get("response")
        .cloned()
        .unwrap_or(Value::Null);
    UniversalItem::ToolResult {
        tool_call_id: gemini_function_response_id(function_response),
        content: vec![ContentBlock::Text {
            text: stringify_json(response),
        }],
        is_error: false,
        extensions: common::empty_extensions(),
    }
}

fn function_call_to_tool_call(function_call: &Value) -> UniversalItem {
    let name = function_call
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    UniversalItem::ToolCall {
        id: gemini_function_call_id(function_call),
        name,
        arguments: function_call
            .get("args")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new())),
        extensions: common::empty_extensions(),
    }
}

fn push_message_if_any(
    out: &mut Vec<UniversalItem>,
    role: Role,
    message_blocks: &mut Vec<ContentBlock>,
) {
    if message_blocks.is_empty() {
        return;
    }
    out.push(UniversalItem::Message {
        role,
        id: None,
        content: std::mem::take(message_blocks),
        extensions: common::empty_extensions(),
    });
}

fn items_to_contents(items: &[UniversalItem]) -> Vec<Value> {
    let mut contents = Vec::new();
    for item in items {
        match item {
            UniversalItem::Message { role, content, .. } => {
                contents.push(json!({
                    "role": universal_role_to_gemini(*role),
                    "parts": blocks_to_gemini_parts(content),
                }));
            }
            UniversalItem::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                contents.push(json!({
                    "role": "function",
                    "parts": [function_response_part(Some(tool_call_id), None, content, *is_error)]
                }));
            }
            UniversalItem::ToolCall {
                id,
                name,
                arguments,
                ..
            } => {
                contents.push(json!({
                    "role": "model",
                    "parts": [function_call_part(Some(id), name, arguments.clone())]
                }));
            }
            UniversalItem::Reasoning { .. } => {}
            UniversalItem::Unknown { raw } => contents.push(raw.clone()),
        }
    }
    contents
}
