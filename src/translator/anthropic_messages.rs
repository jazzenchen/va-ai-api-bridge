use serde_json::{json, Map, Value};

use crate::schema::anthropic::{
    AnthropicContent, AnthropicContentBlock, AnthropicMessage, AnthropicMessagesRequest,
    AnthropicMessagesResponse, AnthropicStreamEvent,
};
use crate::translator::common;
use crate::{
    ApiProxyError, ContentBlock, DecodeState, EncodeState, Result, Role, UniversalEvent,
    UniversalItem, UniversalRequest, WireEvent, WireProtocol,
};

use super::WireTranslator;

#[derive(Debug, Clone, Copy, Default)]
pub struct AnthropicMessagesTranslator;

impl WireTranslator for AnthropicMessagesTranslator {
    fn protocol(&self) -> WireProtocol {
        WireProtocol::AnthropicMessages
    }

    fn decode_request(&self, raw: Value) -> Result<UniversalRequest> {
        let source_raw = raw.clone();
        let request: AnthropicMessagesRequest = serde_json::from_value(raw)
            .map_err(|error| ApiProxyError::invalid_request(error.to_string()))?;

        let mut universal = UniversalRequest {
            model: request.model,
            instructions: common::anthropic_system_to_blocks(request.system.as_ref()),
            tools: request
                .tools
                .iter()
                .filter_map(common::anthropic_tool_from_value)
                .collect(),
            tool_choice: common::tool_choice_from_value(request.tool_choice.as_ref()),
            stream: request.stream.unwrap_or(false),
            generation: common::generation_from_openai(
                request.temperature,
                request.top_p,
                request.max_tokens,
            ),
            reasoning: request.thinking.map(|thinking| crate::ReasoningConfig {
                effort: thinking
                    .get("type")
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
                budget_tokens: thinking.get("budget_tokens").and_then(Value::as_u64),
                visible: None,
                extensions: common::empty_extensions(),
            }),
            source: Some(common::source(WireProtocol::AnthropicMessages, source_raw)),
            ..UniversalRequest::default()
        };

        for message in request.messages {
            decode_message(message, &mut universal);
        }

        Ok(universal)
    }

    fn encode_request(&self, request: &UniversalRequest) -> Result<Value> {
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
        if let Some(system) = common::blocks_to_anthropic_system(&request.instructions) {
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
                        .map(common::tool_to_anthropic)
                        .collect(),
                ),
            );
        }
        if let Some(tool_choice) = &request.tool_choice {
            body.insert(
                "tool_choice".to_string(),
                common::tool_choice_to_anthropic(tool_choice),
            );
        }
        if let Some(reasoning) = &request.reasoning {
            let mut thinking = Map::new();
            thinking.insert(
                "type".to_string(),
                Value::String(
                    reasoning
                        .effort
                        .clone()
                        .unwrap_or_else(|| "enabled".to_string()),
                ),
            );
            if let Some(budget_tokens) = reasoning.budget_tokens {
                thinking.insert("budget_tokens".to_string(), json!(budget_tokens));
            }
            body.insert("thinking".to_string(), Value::Object(thinking));
        }

        let mut messages = Vec::new();
        for item in &request.input {
            match item {
                UniversalItem::Message { role, content, .. } => {
                    messages.push(anthropic_message_value(
                        common::role_to_anthropic(*role),
                        common::blocks_to_anthropic_content(content),
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
                    messages.push(anthropic_message_value(
                        "assistant",
                        AnthropicContent::Blocks(vec![common::block_to_anthropic_block(&block)]),
                    )?);
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
                    messages.push(anthropic_message_value(
                        "user",
                        AnthropicContent::Blocks(vec![common::block_to_anthropic_block(&block)]),
                    )?);
                }
                UniversalItem::Reasoning {
                    text, encrypted, ..
                } => {
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
                UniversalItem::Unknown { raw } => messages.push(raw.clone()),
            }
        }
        body.insert("messages".to_string(), Value::Array(messages));

        Ok(Value::Object(body))
    }

    fn decode_response(&self, raw: Value) -> Result<Vec<UniversalEvent>> {
        let response: AnthropicMessagesResponse = serde_json::from_value(raw)
            .map_err(|error| ApiProxyError::invalid_response(error.to_string()))?;
        let usage = common::anthropic_usage_to_universal(response.usage.as_ref());
        let mut events = vec![
            UniversalEvent::ResponseStart {
                id: response.id.clone(),
                model: response.model.clone(),
                extensions: common::empty_extensions(),
            },
            UniversalEvent::MessageStart {
                id: response
                    .id
                    .clone()
                    .unwrap_or_else(|| "anthropic_message".to_string()),
                role: response
                    .role
                    .as_deref()
                    .and_then(common::role_from_wire)
                    .unwrap_or(Role::Assistant),
                extensions: common::empty_extensions(),
            },
        ];

        for (index, block) in response
            .content
            .iter()
            .map(common::anthropic_block_to_block)
            .enumerate()
        {
            common::push_block_events(&mut events, index, block);
        }
        events.push(UniversalEvent::MessageDone {
            finish_reason: common::finish_from_anthropic(response.stop_reason.as_deref()),
            usage: usage.clone(),
            extensions: common::empty_extensions(),
        });
        events.push(UniversalEvent::ResponseDone {
            usage,
            extensions: common::empty_extensions(),
        });
        Ok(events)
    }

    fn decode_stream_chunk(
        &self,
        raw: Value,
        state: &mut DecodeState,
    ) -> Result<Vec<UniversalEvent>> {
        let raw_for_unknown = raw.clone();
        let event: AnthropicStreamEvent = serde_json::from_value(raw)
            .map_err(|error| ApiProxyError::invalid_response(error.to_string()))?;
        let mut events = Vec::new();
        let kind = event.kind.as_deref().unwrap_or_default();

        match kind {
            "message_start" => {
                let message = event.message;
                common::ensure_response_start(
                    &mut events,
                    state,
                    message.as_ref().and_then(|message| message.id.clone()),
                    message.as_ref().and_then(|message| message.model.clone()),
                );
                common::ensure_message_start(
                    &mut events,
                    state,
                    message
                        .as_ref()
                        .and_then(|message| message.id.clone())
                        .unwrap_or_else(|| "anthropic_message".to_string()),
                    Role::Assistant,
                );
            }
            "content_block_start" => {
                if let Some(block) = event.content_block {
                    common::ensure_content_start(
                        &mut events,
                        state,
                        event.index.unwrap_or(0),
                        common::anthropic_block_to_block(&block),
                    );
                }
            }
            "content_block_delta" => {
                let index = event.index.unwrap_or(0);
                if let Some(delta) = event.delta {
                    match delta.kind.as_deref() {
                        Some("text_delta") => events.push(UniversalEvent::TextDelta {
                            index,
                            text: delta.text.unwrap_or_default(),
                        }),
                        Some("thinking_delta") => events.push(UniversalEvent::ReasoningDelta {
                            index,
                            text: delta.thinking.unwrap_or_default(),
                        }),
                        Some("input_json_delta") => events.push(UniversalEvent::ToolCallDelta {
                            id: format!("tool_call_{index}"),
                            name: None,
                            arguments_delta: delta.partial_json.unwrap_or_default(),
                        }),
                        _ => events.push(UniversalEvent::Unknown {
                            raw: raw_for_unknown,
                            tags: Default::default(),
                        }),
                    }
                }
            }
            "content_block_stop" => events.push(UniversalEvent::ContentDone {
                index: event.index.unwrap_or(0),
                final_block: None,
            }),
            "message_delta" => {
                let finish_reason = event
                    .delta
                    .as_ref()
                    .and_then(|delta| delta.stop_reason.as_deref());
                events.push(UniversalEvent::MessageDone {
                    finish_reason: common::finish_from_anthropic(finish_reason),
                    usage: common::anthropic_usage_to_universal(event.usage.as_ref()),
                    extensions: common::empty_extensions(),
                });
            }
            "message_stop" => {
                if common::mark_once(state, "response_done") {
                    events.push(UniversalEvent::ResponseDone {
                        usage: common::anthropic_usage_to_universal(event.usage.as_ref()),
                        extensions: common::empty_extensions(),
                    });
                }
            }
            "error" => events.push(UniversalEvent::Error {
                message: event
                    .extra
                    .get("error")
                    .and_then(|error| error.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("Anthropic stream error")
                    .to_string(),
                raw: event.extra.get("error").cloned(),
            }),
            _ => events.push(UniversalEvent::Unknown {
                raw: raw_for_unknown,
                tags: Default::default(),
            }),
        }

        Ok(events)
    }

    fn encode_events(
        &self,
        events: &[UniversalEvent],
        _state: &mut EncodeState,
    ) -> Result<Vec<WireEvent>> {
        Ok(events
            .iter()
            .map(|event| match event {
                UniversalEvent::ResponseStart { id, model, .. } => common::wire_event(json!({
                    "type": "message_start",
                    "message": {
                        "id": id,
                        "model": model,
                        "role": "assistant"
                    }
                })),
                UniversalEvent::ContentStart { index, block } => common::wire_event(json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": common::block_to_anthropic_block(block)
                })),
                UniversalEvent::TextDelta { index, text } => common::wire_event(json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "text_delta",
                        "text": text
                    }
                })),
                UniversalEvent::ReasoningDelta { index, text } => common::wire_event(json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": text
                    }
                })),
                UniversalEvent::ToolCallDelta {
                    arguments_delta, ..
                } => common::wire_event(json!({
                    "type": "content_block_delta",
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": arguments_delta
                    }
                })),
                UniversalEvent::ContentDone { index, .. } => common::wire_event(json!({
                    "type": "content_block_stop",
                    "index": index
                })),
                UniversalEvent::MessageDone {
                    finish_reason,
                    usage,
                    ..
                } => common::wire_event(json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": finish_to_anthropic(*finish_reason)
                    },
                    "usage": usage
                })),
                UniversalEvent::ResponseDone { .. } => common::wire_event(json!({
                    "type": "message_stop"
                })),
                UniversalEvent::Error { message, raw } => common::wire_event(json!({
                    "type": "error",
                    "error": {
                        "message": message,
                        "raw": raw
                    }
                })),
                UniversalEvent::Unknown { raw, .. } => common::wire_event(raw.clone()),
                UniversalEvent::MessageStart { .. } => common::wire_event(json!({
                    "type": "message_start"
                })),
            })
            .collect())
    }
}

fn decode_message(message: AnthropicMessage, request: &mut UniversalRequest) {
    let role = common::role_from_wire(&message.role).unwrap_or(Role::User);
    let blocks = common::anthropic_content_to_blocks(&message.content);
    let mut message_blocks = Vec::new();

    for block in blocks {
        match block {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
                extensions,
            } => request.input.push(UniversalItem::ToolCall {
                id,
                name,
                arguments,
                extensions,
            }),
            ContentBlock::ToolResult {
                tool_call_id,
                content,
                is_error,
                extensions,
            } => request.input.push(UniversalItem::ToolResult {
                tool_call_id,
                content,
                is_error,
                extensions,
            }),
            block => message_blocks.push(block),
        }
    }

    if !message_blocks.is_empty() {
        request.input.push(UniversalItem::Message {
            role,
            id: None,
            content: message_blocks,
            extensions: common::empty_extensions(),
        });
    }
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

fn finish_to_anthropic(reason: Option<crate::FinishReason>) -> Value {
    match reason {
        Some(crate::FinishReason::Stop) => json!("end_turn"),
        Some(crate::FinishReason::Length) => json!("max_tokens"),
        Some(crate::FinishReason::ToolCall) => json!("tool_use"),
        Some(crate::FinishReason::ContentFilter) => json!("content_filter"),
        Some(crate::FinishReason::Error) => json!("error"),
        Some(crate::FinishReason::Unknown) | None => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn decodes_anthropic_request_to_universal_items() {
        let translator = AnthropicMessagesTranslator;
        let request = translator
            .decode_request(json!({
                "model": "claude-test",
                "max_tokens": 128,
                "system": "be crisp",
                "messages": [{
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "hello" },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "sunny"
                        }
                    ]
                }],
                "tools": [{
                    "name": "lookup",
                    "input_schema": { "type": "object" }
                }]
            }))
            .unwrap();

        assert_eq!(request.model.as_deref(), Some("claude-test"));
        assert_eq!(request.instructions.len(), 1);
        assert_eq!(request.tools[0].name, "lookup");
        assert!(request
            .input
            .iter()
            .any(|item| matches!(item, UniversalItem::ToolResult { .. })));
    }

    #[test]
    fn decodes_anthropic_stream_text_delta() {
        let translator = AnthropicMessagesTranslator;
        let mut state = DecodeState::default();
        let events = translator
            .decode_stream_chunk(
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": "hi" }
                }),
                &mut state,
            )
            .unwrap();

        assert!(matches!(
            &events[0],
            UniversalEvent::TextDelta { text, .. } if text == "hi"
        ));
    }
}
