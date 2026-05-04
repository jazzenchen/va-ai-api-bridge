use serde_json::{json, Map, Value};

use crate::schema::openai::{ChatCompletionRequest, ChatMessage, ChatToolCall};
use crate::translator::{common, openai};
use crate::{ApiProxyError, Result, Role, UniversalItem, UniversalRequest, WireProtocol};

pub(super) fn decode(raw: Value) -> Result<UniversalRequest> {
    let source_raw = raw.clone();
    let request: ChatCompletionRequest = serde_json::from_value(raw)
        .map_err(|error| ApiProxyError::invalid_request(error.to_string()))?;

    let mut universal = UniversalRequest {
        model: request.model,
        tools: request
            .tools
            .iter()
            .filter_map(openai::openai_tool_from_value)
            .collect(),
        tool_choice: openai::tool_choice_from_openai_value(request.tool_choice.as_ref()),
        stream: request.stream.unwrap_or(false),
        generation: openai::generation_from_openai(
            request.temperature,
            request.top_p,
            request.max_completion_tokens.or(request.max_tokens),
        ),
        source: Some(common::source(WireProtocol::OpenAiChat, source_raw)),
        ..UniversalRequest::default()
    };

    for message in request.messages {
        decode_message(message, &mut universal);
    }

    Ok(universal)
}

pub(super) fn encode(request: &UniversalRequest) -> Result<Value> {
    let mut body = Map::new();
    if let Some(model) = &request.model {
        body.insert("model".to_string(), Value::String(model.clone()));
    }
    if request.stream {
        body.insert("stream".to_string(), Value::Bool(true));
    }
    if let Some(temperature) = request.generation.temperature {
        body.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(top_p) = request.generation.top_p {
        body.insert("top_p".to_string(), json!(top_p));
    }
    if let Some(max_output_tokens) = request.generation.max_output_tokens {
        body.insert(
            "max_completion_tokens".to_string(),
            json!(max_output_tokens),
        );
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".to_string(),
            Value::Array(
                request
                    .tools
                    .iter()
                    .map(openai::tool_to_openai_chat)
                    .collect(),
            ),
        );
    }
    if let Some(tool_choice) = &request.tool_choice {
        body.insert(
            "tool_choice".to_string(),
            openai::tool_choice_to_openai(tool_choice),
        );
    }

    let mut messages = Vec::new();
    if !request.instructions.is_empty() {
        messages.push(message_value(
            "system",
            openai::blocks_to_openai_content(&request.instructions, "text", "image_url"),
            None,
            Vec::new(),
        )?);
    }

    let mut pending_tool_calls = Vec::new();
    for item in &request.input {
        match item {
            UniversalItem::Message { role, content, .. } => {
                flush_tool_calls(&mut messages, &mut pending_tool_calls)?;
                messages.push(message_value(
                    openai::role_to_openai(*role),
                    openai::blocks_to_openai_content(content, "text", "image_url"),
                    None,
                    Vec::new(),
                )?);
            }
            UniversalItem::ToolCall {
                id,
                name,
                arguments,
                ..
            } => {
                pending_tool_calls.push(chat_tool_call_value(id, name, arguments));
            }
            UniversalItem::ToolResult {
                tool_call_id,
                content,
                ..
            } => {
                flush_tool_calls(&mut messages, &mut pending_tool_calls)?;
                messages.push(message_value(
                    "tool",
                    openai::blocks_to_openai_content(content, "text", "image_url"),
                    Some(tool_call_id),
                    Vec::new(),
                )?);
            }
            UniversalItem::Unknown { raw } => {
                flush_tool_calls(&mut messages, &mut pending_tool_calls)?;
                messages.push(raw.clone());
            }
            UniversalItem::Reasoning { .. } => {}
        }
    }
    flush_tool_calls(&mut messages, &mut pending_tool_calls)?;

    body.insert("messages".to_string(), Value::Array(messages));
    Ok(Value::Object(body))
}

fn decode_message(message: ChatMessage, request: &mut UniversalRequest) {
    let role = common::role_from_wire(&message.role);
    let blocks = openai::openai_content_to_blocks(message.content.as_ref());

    match role {
        Some(Role::System) => request.instructions.extend(blocks),
        Some(Role::Tool) => request.input.push(UniversalItem::ToolResult {
            tool_call_id: message.tool_call_id.unwrap_or_default(),
            content: blocks,
            is_error: false,
            extensions: common::empty_extensions(),
        }),
        Some(role @ (Role::User | Role::Assistant)) => {
            if !blocks.is_empty() || message.tool_calls.is_empty() {
                request.input.push(UniversalItem::Message {
                    role,
                    id: None,
                    content: blocks,
                    extensions: common::empty_extensions(),
                });
            }
            for tool_call in message.tool_calls {
                request.input.push(chat_tool_call_to_item(tool_call));
            }
        }
        None => request.input.push(UniversalItem::Unknown {
            raw: serde_json::to_value(message).unwrap_or(Value::Null),
        }),
    }
}

fn chat_tool_call_to_item(tool_call: ChatToolCall) -> UniversalItem {
    let function = tool_call.function;
    UniversalItem::ToolCall {
        id: tool_call.id.unwrap_or_default(),
        name: function
            .as_ref()
            .and_then(|function| function.name.clone())
            .unwrap_or_default(),
        arguments: common::parse_arguments(
            function
                .as_ref()
                .and_then(|function| function.arguments.as_deref()),
        ),
        extensions: common::empty_extensions(),
    }
}

fn message_value(
    role: &str,
    content: Option<crate::schema::openai::OpenAiContent>,
    tool_call_id: Option<&String>,
    tool_calls: Vec<Value>,
) -> Result<Value> {
    let mut message = Map::new();
    message.insert("role".to_string(), Value::String(role.to_string()));
    if let Some(content) = content {
        message.insert(
            "content".to_string(),
            serde_json::to_value(content)
                .map_err(|error| ApiProxyError::conversion(error.to_string()))?,
        );
    }
    if let Some(tool_call_id) = tool_call_id {
        message.insert(
            "tool_call_id".to_string(),
            Value::String(tool_call_id.clone()),
        );
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".to_string(), Value::Array(tool_calls));
    }
    Ok(Value::Object(message))
}

fn chat_tool_call_value(id: &str, name: &str, arguments: &Value) -> Value {
    json!({
        "id": id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": common::stringify_arguments(arguments)
        }
    })
}

fn flush_tool_calls(messages: &mut Vec<Value>, pending_tool_calls: &mut Vec<Value>) -> Result<()> {
    if pending_tool_calls.is_empty() {
        return Ok(());
    }
    messages.push(message_value(
        "assistant",
        None,
        None,
        std::mem::take(pending_tool_calls),
    )?);
    Ok(())
}
