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

    let mut system_blocks = request.instructions.clone();
    let mut messages = Vec::new();
    let mut pending_assistant_blocks = Vec::new();
    let mut pending_tool_results = Vec::new();
    for item in &request.input {
        match item {
            UniversalItem::Message { role, content, .. } => {
                if is_empty_message_content(content) {
                    continue;
                }
                if *role == Role::Assistant {
                    flush_anthropic_blocks(&mut messages, "user", &mut pending_tool_results)?;
                    pending_assistant_blocks.extend(anthropic_blocks(content));
                } else if *role == Role::System {
                    system_blocks.extend(content.iter().cloned());
                } else {
                    flush_anthropic_blocks(
                        &mut messages,
                        "assistant",
                        &mut pending_assistant_blocks,
                    )?;
                    if *role == Role::User && !pending_tool_results.is_empty() {
                        pending_tool_results.extend(anthropic_blocks(content));
                        flush_anthropic_blocks(&mut messages, "user", &mut pending_tool_results)?;
                    } else {
                        flush_anthropic_blocks(&mut messages, "user", &mut pending_tool_results)?;
                        messages.push(anthropic_message_value(
                            anthropic::role_to_anthropic(*role),
                            anthropic::blocks_to_anthropic_content(content),
                        )?);
                    }
                }
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
                pending_assistant_blocks.push(anthropic::block_to_anthropic_block(&block));
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
                flush_anthropic_blocks(&mut messages, "assistant", &mut pending_assistant_blocks)?;
                pending_tool_results.push(anthropic::block_to_anthropic_block(&block));
            }
            UniversalItem::Reasoning {
                text, encrypted, ..
            } => {
                flush_anthropic_blocks(&mut messages, "user", &mut pending_tool_results)?;
                if text.as_deref().is_some_and(|text| !text.is_empty())
                    || encrypted
                        .as_deref()
                        .is_some_and(|encrypted| !encrypted.is_empty())
                {
                    pending_assistant_blocks.push(AnthropicContentBlock {
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
                    });
                }
            }
            UniversalItem::Unknown { raw } => {
                flush_anthropic_blocks(&mut messages, "assistant", &mut pending_assistant_blocks)?;
                flush_anthropic_blocks(&mut messages, "user", &mut pending_tool_results)?;
                messages.push(raw.clone());
            }
        }
    }
    flush_anthropic_blocks(&mut messages, "assistant", &mut pending_assistant_blocks)?;
    flush_anthropic_blocks(&mut messages, "user", &mut pending_tool_results)?;
    if let Some(system) = anthropic::blocks_to_anthropic_system(&system_blocks) {
        body.insert(
            "system".to_string(),
            serde_json::to_value(system)
                .map_err(|error| ApiProxyError::conversion(error.to_string()))?,
        );
    }
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

fn anthropic_blocks(blocks: &[ContentBlock]) -> Vec<AnthropicContentBlock> {
    blocks
        .iter()
        .map(anthropic::block_to_anthropic_block)
        .collect()
}

fn is_empty_message_content(content: &[ContentBlock]) -> bool {
    content.iter().all(|block| match block {
        ContentBlock::Text { text } => text.trim().is_empty(),
        ContentBlock::Reasoning {
            text, encrypted, ..
        } => {
            text.as_deref().unwrap_or_default().trim().is_empty()
                && encrypted.as_deref().unwrap_or_default().trim().is_empty()
        }
        _ => false,
    })
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{ContentBlock, Role, UniversalItem, UniversalRequest};

    use super::encode;

    #[test]
    fn skips_empty_assistant_message_between_tool_use_and_tool_result() {
        let request = UniversalRequest {
            model: Some("minimax".to_string()),
            input: vec![
                UniversalItem::ToolCall {
                    id: "call_1".to_string(),
                    name: "list_files".to_string(),
                    arguments: json!({ "path": "." }),
                    extensions: Default::default(),
                },
                UniversalItem::Message {
                    role: Role::Assistant,
                    id: None,
                    content: Vec::new(),
                    extensions: Default::default(),
                },
                UniversalItem::ToolResult {
                    tool_call_id: "call_1".to_string(),
                    content: vec![ContentBlock::Text {
                        text: "Cargo.toml".to_string(),
                    }],
                    is_error: false,
                    extensions: Default::default(),
                },
            ],
            ..UniversalRequest::default()
        };

        let encoded = encode(&request).expect("request encodes");
        let messages = encoded["messages"].as_array().unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"][0]["type"], "tool_use");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"][0]["type"], "tool_result");
        assert_eq!(messages[1]["content"][0]["tool_use_id"], "call_1");
    }

    #[test]
    fn combines_reasoning_and_tool_use_into_one_assistant_turn() {
        let request = UniversalRequest {
            model: Some("minimax".to_string()),
            input: vec![
                UniversalItem::Reasoning {
                    id: None,
                    text: Some("Need to inspect files.".to_string()),
                    encrypted: None,
                    extensions: Default::default(),
                },
                UniversalItem::ToolCall {
                    id: "call_1".to_string(),
                    name: "list_files".to_string(),
                    arguments: json!({ "path": "." }),
                    extensions: Default::default(),
                },
                UniversalItem::ToolResult {
                    tool_call_id: "call_1".to_string(),
                    content: vec![ContentBlock::Text {
                        text: "Cargo.toml".to_string(),
                    }],
                    is_error: false,
                    extensions: Default::default(),
                },
            ],
            ..UniversalRequest::default()
        };

        let encoded = encode(&request).expect("request encodes");
        let messages = encoded["messages"].as_array().unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["content"][0]["type"], "thinking");
        assert_eq!(messages[0]["content"][1]["type"], "tool_use");
        assert_eq!(messages[1]["content"][0]["type"], "tool_result");
    }

    #[test]
    fn combines_assistant_text_and_tool_use_into_one_assistant_turn() {
        let request = UniversalRequest {
            model: Some("minimax".to_string()),
            input: vec![
                UniversalItem::Message {
                    role: Role::Assistant,
                    id: None,
                    content: vec![ContentBlock::Text {
                        text: "I will inspect the project.".to_string(),
                    }],
                    extensions: Default::default(),
                },
                UniversalItem::ToolCall {
                    id: "call_1".to_string(),
                    name: "list_files".to_string(),
                    arguments: json!({ "path": "." }),
                    extensions: Default::default(),
                },
                UniversalItem::ToolResult {
                    tool_call_id: "call_1".to_string(),
                    content: vec![ContentBlock::Text {
                        text: "Cargo.toml".to_string(),
                    }],
                    is_error: false,
                    extensions: Default::default(),
                },
            ],
            ..UniversalRequest::default()
        };

        let encoded = encode(&request).expect("request encodes");
        let messages = encoded["messages"].as_array().unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["content"][0]["type"], "text");
        assert_eq!(messages[0]["content"][1]["type"], "tool_use");
        assert_eq!(messages[1]["content"][0]["type"], "tool_result");
    }

    #[test]
    fn places_user_text_after_pending_tool_results() {
        let request = UniversalRequest {
            model: Some("minimax".to_string()),
            input: vec![
                UniversalItem::ToolCall {
                    id: "call_1".to_string(),
                    name: "list_files".to_string(),
                    arguments: json!({ "path": "." }),
                    extensions: Default::default(),
                },
                UniversalItem::ToolResult {
                    tool_call_id: "call_1".to_string(),
                    content: vec![ContentBlock::Text {
                        text: "Cargo.toml".to_string(),
                    }],
                    is_error: false,
                    extensions: Default::default(),
                },
                UniversalItem::Message {
                    role: Role::User,
                    id: None,
                    content: vec![ContentBlock::Text {
                        text: "Continue.".to_string(),
                    }],
                    extensions: Default::default(),
                },
            ],
            ..UniversalRequest::default()
        };

        let encoded = encode(&request).expect("request encodes");
        let messages = encoded["messages"].as_array().unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"][0]["type"], "tool_result");
        assert_eq!(messages[1]["content"][1]["type"], "text");
    }

    #[test]
    fn moves_system_messages_to_top_level_system() {
        let request = UniversalRequest {
            model: Some("minimax".to_string()),
            instructions: vec![ContentBlock::Text {
                text: "Be precise.".to_string(),
            }],
            input: vec![
                UniversalItem::Message {
                    role: Role::System,
                    id: None,
                    content: vec![ContentBlock::Text {
                        text: "Prefer JSON.".to_string(),
                    }],
                    extensions: Default::default(),
                },
                UniversalItem::Message {
                    role: Role::User,
                    id: None,
                    content: vec![ContentBlock::Text {
                        text: "Ping".to_string(),
                    }],
                    extensions: Default::default(),
                },
            ],
            ..UniversalRequest::default()
        };

        let encoded = encode(&request).expect("request encodes");
        let messages = encoded["messages"].as_array().unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(encoded["system"][0]["type"], "text");
        assert_eq!(encoded["system"][0]["text"], "Be precise.");
        assert_eq!(encoded["system"][1]["text"], "Prefer JSON.");
    }
}
