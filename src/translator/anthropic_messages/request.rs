use serde_json::{json, Map, Value};

use crate::schema::anthropic::{
    AnthropicContent, AnthropicContentBlock, AnthropicMessage, AnthropicMessagesRequest,
};
use crate::translator::{anthropic, common};
use crate::{
    ApiProxyError, ContentBlock, Result, Role, UniversalItem, UniversalRequest, WireProtocol,
};

pub(super) fn decode(raw: Value) -> Result<UniversalRequest> {
    let source_raw = raw.clone();
    let request: AnthropicMessagesRequest = serde_json::from_value(raw)
        .map_err(|error| ApiProxyError::invalid_request(error.to_string()))?;

    let mut universal = UniversalRequest {
        model: request.model,
        instructions: anthropic::anthropic_system_to_blocks(request.system.as_ref()),
        tools: request
            .tools
            .iter()
            .filter_map(anthropic::anthropic_tool_from_value)
            .collect(),
        tool_choice: anthropic::tool_choice_from_anthropic_value(request.tool_choice.as_ref()),
        stream: request.stream.unwrap_or(false),
        generation: anthropic::generation_from_anthropic(
            request.temperature,
            request.top_p,
            request.max_tokens,
        ),
        reasoning: request
            .thinking
            .map(anthropic::reasoning_from_anthropic_thinking),
        source: Some(common::source(WireProtocol::AnthropicMessages, source_raw)),
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
    if let Some(max_tokens) = request.generation.max_output_tokens {
        body.insert("max_tokens".to_string(), json!(max_tokens));
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
    if let Some(system) = anthropic::blocks_to_anthropic_system(&request.instructions) {
        body.insert(
            "system".to_string(),
            serde_json::to_value(system)
                .map_err(|error| ApiProxyError::conversion(error.to_string()))?,
        );
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".to_string(),
            Value::Array(
                request
                    .tools
                    .iter()
                    .map(anthropic::tool_to_anthropic)
                    .collect(),
            ),
        );
    }
    if let Some(tool_choice) = &request.tool_choice {
        body.insert(
            "tool_choice".to_string(),
            anthropic::tool_choice_to_anthropic(tool_choice),
        );
    }
    if let Some(reasoning) = &request.reasoning {
        if let Some(thinking) = anthropic::anthropic_thinking_from_reasoning(
            reasoning,
            request.generation.max_output_tokens,
        ) {
            body.insert("thinking".to_string(), thinking);
        }
    }

    let mut messages = Vec::new();
    let mut pending_tool_calls = Vec::new();
    let mut pending_tool_results = Vec::new();
    for item in &request.input {
        match item {
            UniversalItem::Message { role, content, .. } => {
                flush_anthropic_blocks(&mut messages, "assistant", &mut pending_tool_calls)?;
                flush_anthropic_blocks(&mut messages, "user", &mut pending_tool_results)?;
                messages.push(anthropic_message_value(
                    anthropic::role_to_anthropic(*role),
                    anthropic::blocks_to_anthropic_content(content),
                )?);
            }
            UniversalItem::ToolCall {
                id,
                name,
                arguments,
                ..
            } => {
                let block = ContentBlock::ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                    extensions: common::empty_extensions(),
                };
                flush_anthropic_blocks(&mut messages, "user", &mut pending_tool_results)?;
                pending_tool_calls.push(anthropic::block_to_anthropic_block(&block));
            }
            UniversalItem::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                let block = ContentBlock::ToolResult {
                    tool_call_id: tool_call_id.clone(),
                    content: content.clone(),
                    is_error: *is_error,
                    extensions: common::empty_extensions(),
                };
                flush_anthropic_blocks(&mut messages, "assistant", &mut pending_tool_calls)?;
                pending_tool_results.push(anthropic::block_to_anthropic_block(&block));
            }
            UniversalItem::Reasoning {
                text, encrypted, ..
            } => {
                flush_anthropic_blocks(&mut messages, "assistant", &mut pending_tool_calls)?;
                flush_anthropic_blocks(&mut messages, "user", &mut pending_tool_results)?;
                messages.push(anthropic_message_value(
                    "assistant",
                    AnthropicContent::Blocks(vec![AnthropicContentBlock {
                        kind: "thinking".to_string(),
                        text: None,
                        source: None,
                        id: None,
                        name: None,
                        input: None,
                        tool_use_id: None,
                        content: None,
                        thinking: text.clone(),
                        signature: encrypted.clone(),
                        extra: Default::default(),
                    }]),
                )?);
            }
            UniversalItem::Unknown { raw } => {
                flush_anthropic_blocks(&mut messages, "assistant", &mut pending_tool_calls)?;
                flush_anthropic_blocks(&mut messages, "user", &mut pending_tool_results)?;
                messages.push(raw.clone());
            }
        }
    }
    flush_anthropic_blocks(&mut messages, "assistant", &mut pending_tool_calls)?;
    flush_anthropic_blocks(&mut messages, "user", &mut pending_tool_results)?;
    body.insert("messages".to_string(), Value::Array(messages));

    Ok(Value::Object(body))
}

fn decode_message(message: AnthropicMessage, request: &mut UniversalRequest) {
    let role = common::role_from_wire(&message.role).unwrap_or(Role::User);
    let blocks = anthropic::anthropic_content_to_blocks(&message.content);
    let mut message_blocks = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
                extensions,
            } => {
                flush_message_blocks(request, role, &mut message_blocks);
                request.input.push(UniversalItem::ToolCall {
                    id,
                    name,
                    arguments,
                    extensions,
                });
            }
            ContentBlock::ToolResult {
                tool_call_id,
                content,
                is_error,
                extensions,
            } => {
                flush_message_blocks(request, role, &mut message_blocks);
                request.input.push(UniversalItem::ToolResult {
                    tool_call_id,
                    content,
                    is_error,
                    extensions,
                });
            }
            block => message_blocks.push(block),
        }
    }

    flush_message_blocks(request, role, &mut message_blocks);
}

fn flush_message_blocks(
    request: &mut UniversalRequest,
    role: Role,
    message_blocks: &mut Vec<ContentBlock>,
) {
    if message_blocks.is_empty() {
        return;
    }

    request.input.push(UniversalItem::Message {
        role,
        id: None,
        content: std::mem::take(message_blocks),
        extensions: common::empty_extensions(),
    });
}

fn anthropic_message_value(role: &str, content: AnthropicContent) -> Result<Value> {
    let mut message = Map::new();
    message.insert("role".to_string(), Value::String(role.to_string()));
    message.insert(
        "content".to_string(),
        serde_json::to_value(content)
            .map_err(|error| ApiProxyError::conversion(error.to_string()))?,
    );
    Ok(Value::Object(message))
}

fn flush_anthropic_blocks(
    messages: &mut Vec<Value>,
    role: &str,
    blocks: &mut Vec<AnthropicContentBlock>,
) -> Result<()> {
    if blocks.is_empty() {
        return Ok(());
    }
    messages.push(anthropic_message_value(
        role,
        AnthropicContent::Blocks(std::mem::take(blocks)),
    )?);
    Ok(())
}
