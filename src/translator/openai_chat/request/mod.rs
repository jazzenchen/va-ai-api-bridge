use serde_json::{json, Map, Value};

use crate::schema::openai::{ChatCompletionRequest, ChatMessage, ChatToolCall};
use crate::translator::{common, openai};
use crate::{
    ApiBridgeError, ContentBlock, Result, Role, UniversalItem, UniversalRequest, WireProtocol,
};

pub(super) fn decode(raw: Value) -> Result<UniversalRequest> {
    let source_raw = raw.clone();
    let request: ChatCompletionRequest = serde_json::from_value(raw)
        .map_err(|error| ApiBridgeError::invalid_request(error.to_string()))?;

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
    let mut pending_tool_content = Vec::new();
    let mut pending_tool_reasoning_content = None;
    for item in &request.input {
        match item {
            UniversalItem::Message {
                role,
                content,
                extensions,
                ..
            } => {
                let chat_content = chat_compatible_content_blocks(content);
                if is_empty_assistant_message(*role, &chat_content, extensions) {
                    continue;
                }
                if *role == Role::Assistant && !pending_tool_calls.is_empty() {
                    pending_tool_content.extend(chat_content);
                    if pending_tool_reasoning_content.is_none() {
                        pending_tool_reasoning_content = reasoning_content_extension(extensions);
                    }
                    continue;
                }
                flush_tool_calls(
                    &mut messages,
                    &mut pending_tool_calls,
                    &mut pending_tool_content,
                    &mut pending_tool_reasoning_content,
                )?;
                let mut message = message_value(
                    role_to_chat_message_role(*role),
                    openai::blocks_to_openai_content(&chat_content, "text", "image_url"),
                    None,
                    Vec::new(),
                )?;
                apply_chat_message_extensions(&mut message, extensions);
                messages.push(message);
            }
            UniversalItem::ToolCall {
                id,
                name,
                arguments,
                extensions,
            } => {
                pending_tool_calls.push(chat_tool_call_value(id, name, arguments));
                if pending_tool_reasoning_content.is_none() {
                    pending_tool_reasoning_content = reasoning_content_extension(extensions);
                }
            }
            UniversalItem::ToolResult {
                tool_call_id,
                content,
                ..
            } => {
                flush_tool_calls(
                    &mut messages,
                    &mut pending_tool_calls,
                    &mut pending_tool_content,
                    &mut pending_tool_reasoning_content,
                )?;
                messages.push(message_value(
                    "tool",
                    openai::blocks_to_openai_content(content, "text", "image_url"),
                    Some(tool_call_id),
                    Vec::new(),
                )?);
            }
            UniversalItem::Unknown { raw } => {
                flush_tool_calls(
                    &mut messages,
                    &mut pending_tool_calls,
                    &mut pending_tool_content,
                    &mut pending_tool_reasoning_content,
                )?;
                if let Some(message) = unknown_chat_message(raw) {
                    messages.push(message);
                }
            }
            UniversalItem::Reasoning { .. } => {}
        }
    }
    flush_tool_calls(
        &mut messages,
        &mut pending_tool_calls,
        &mut pending_tool_content,
        &mut pending_tool_reasoning_content,
    )?;

    for message in &mut messages {
        ensure_chat_message_content(message);
    }

    body.insert("messages".to_string(), Value::Array(messages));
    Ok(Value::Object(body))
}

fn is_empty_assistant_message(
    role: Role,
    content: &[ContentBlock],
    extensions: &crate::Extensions,
) -> bool {
    role == Role::Assistant
        && extensions.is_empty()
        && content.iter().all(is_empty_message_content_block)
}

fn is_empty_message_content_block(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::Text { text } => text.trim().is_empty(),
        ContentBlock::Reasoning {
            text, encrypted, ..
        } => {
            text.as_deref().unwrap_or_default().trim().is_empty()
                && encrypted.as_deref().unwrap_or_default().trim().is_empty()
        }
        _ => false,
    }
}

fn chat_compatible_content_blocks(content: &[ContentBlock]) -> Vec<ContentBlock> {
    content
        .iter()
        .filter_map(chat_compatible_content_block)
        .collect()
}

fn chat_compatible_content_block(block: &ContentBlock) -> Option<ContentBlock> {
    match block {
        ContentBlock::Text { .. } | ContentBlock::Image { .. } | ContentBlock::File { .. } => {
            Some(block.clone())
        }
        ContentBlock::Unknown { raw } => {
            raw.as_str()
                .filter(|text| !text.trim().is_empty())
                .map(|text| ContentBlock::Text {
                    text: text.to_string(),
                })
        }
        ContentBlock::ToolCall { .. }
        | ContentBlock::ToolResult { .. }
        | ContentBlock::Reasoning { .. } => None,
    }
}

fn unknown_chat_message(raw: &Value) -> Option<Value> {
    let object = raw.as_object()?;
    if object.contains_key("type") {
        return None;
    }
    let role = object.get("role").and_then(Value::as_str)?;
    matches!(
        role,
        "system" | "developer" | "user" | "assistant" | "tool" | "function"
    )
    .then(|| raw.clone())
}

fn role_to_chat_message_role(role: Role) -> &'static str {
    match role {
        // Chat Completions-compatible providers often only accept the classic
        // role set. Preserve developer instructions as system content rather
        // than emitting a role many upstreams reject.
        Role::Developer | Role::System => "system",
        other => openai::role_to_openai(other),
    }
}

fn decode_message(message: ChatMessage, request: &mut UniversalRequest) {
    let role = common::role_from_wire(&message.role);
    let blocks = openai::openai_content_to_blocks(message.content.as_ref());
    let extensions = common::value_extensions(message.extra.clone());

    match role {
        Some(Role::Developer | Role::System) => request.instructions.extend(blocks),
        Some(Role::Tool) => request.input.push(UniversalItem::ToolResult {
            tool_call_id: message.tool_call_id.unwrap_or_default(),
            content: blocks,
            is_error: false,
            extensions,
        }),
        Some(role @ (Role::User | Role::Assistant)) => {
            if !blocks.is_empty() || message.tool_calls.is_empty() {
                request.input.push(UniversalItem::Message {
                    role,
                    id: None,
                    content: blocks,
                    extensions: extensions.clone(),
                });
            }
            for tool_call in message.tool_calls {
                request.input.push(chat_tool_call_to_item(
                    tool_call,
                    reasoning_content_extension(&extensions),
                ));
            }
        }
        None => request.input.push(UniversalItem::Unknown {
            raw: serde_json::to_value(message).unwrap_or(Value::Null),
        }),
    }
}

fn chat_tool_call_to_item(
    tool_call: ChatToolCall,
    reasoning_content: Option<String>,
) -> UniversalItem {
    let function = tool_call.function;
    let mut extensions = common::empty_extensions();
    if let Some(reasoning_content) = reasoning_content {
        extensions.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning_content),
        );
    }
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
        extensions,
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
    let content = match content {
        Some(content) => serde_json::to_value(content)
            .map_err(|error| ApiBridgeError::conversion(error.to_string()))?,
        None => Value::String(String::new()),
    };
    message.insert("content".to_string(), content);
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

fn ensure_chat_message_content(message: &mut Value) {
    let Some(object) = message.as_object_mut() else {
        return;
    };
    object
        .entry("content".to_string())
        .or_insert_with(|| Value::String(String::new()));
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

fn flush_tool_calls(
    messages: &mut Vec<Value>,
    pending_tool_calls: &mut Vec<Value>,
    pending_tool_content: &mut Vec<ContentBlock>,
    pending_tool_reasoning_content: &mut Option<String>,
) -> Result<()> {
    if pending_tool_calls.is_empty() {
        return Ok(());
    }
    let content_blocks = std::mem::take(pending_tool_content);
    let mut message = message_value(
        "assistant",
        openai::blocks_to_openai_content(&content_blocks, "text", "image_url"),
        None,
        std::mem::take(pending_tool_calls),
    )?;
    if let Some(reasoning_content) = pending_tool_reasoning_content.take() {
        if let Some(object) = message.as_object_mut() {
            object.insert(
                "reasoning_content".to_string(),
                Value::String(reasoning_content),
            );
        }
    }
    messages.push(message);
    Ok(())
}

fn apply_chat_message_extensions(message: &mut Value, extensions: &crate::Extensions) {
    let Some(reasoning_content) = reasoning_content_extension(extensions) else {
        return;
    };
    if let Some(object) = message.as_object_mut() {
        object.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning_content),
        );
    }
}

fn reasoning_content_extension(extensions: &crate::Extensions) -> Option<String> {
    extensions
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())
        .map(ToString::to_string)
}

#[cfg(test)]
mod tests;
