use serde_json::{json, Map, Value};

use crate::schema::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatToolCall,
};
use crate::translator::common;
use crate::{
    ApiProxyError, ContentBlock, DecodeState, EncodeState, Result, Role, UniversalEvent,
    UniversalItem, UniversalRequest, WireEvent, WireProtocol,
};

use super::WireTranslator;

#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAiChatTranslator;

impl WireTranslator for OpenAiChatTranslator {
    fn protocol(&self) -> WireProtocol {
        WireProtocol::OpenAiChat
    }

    fn decode_request(&self, raw: Value) -> Result<UniversalRequest> {
        let source_raw = raw.clone();
        let request: ChatCompletionRequest = serde_json::from_value(raw)
            .map_err(|error| ApiProxyError::invalid_request(error.to_string()))?;

        let mut universal = UniversalRequest {
            model: request.model,
            tools: request
                .tools
                .iter()
                .filter_map(common::openai_tool_from_value)
                .collect(),
            tool_choice: common::tool_choice_from_value(request.tool_choice.as_ref()),
            stream: request.stream.unwrap_or(false),
            generation: common::generation_from_openai(
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

    fn encode_request(&self, request: &UniversalRequest) -> Result<Value> {
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
                        .map(common::tool_to_openai_chat)
                        .collect(),
                ),
            );
        }
        if let Some(tool_choice) = &request.tool_choice {
            body.insert(
                "tool_choice".to_string(),
                common::tool_choice_to_openai(tool_choice),
            );
        }

        let mut messages = Vec::new();
        if !request.instructions.is_empty() {
            messages.push(message_value(
                "system",
                common::blocks_to_openai_content(&request.instructions, "text", "image_url"),
                None,
                Vec::new(),
            )?);
        }

        for item in &request.input {
            match item {
                UniversalItem::Message { role, content, .. } => {
                    messages.push(message_value(
                        common::role_to_openai(*role),
                        common::blocks_to_openai_content(content, "text", "image_url"),
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
                    messages.push(message_value(
                        "assistant",
                        None,
                        None,
                        vec![json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": common::stringify_arguments(arguments)
                            }
                        })],
                    )?);
                }
                UniversalItem::ToolResult {
                    tool_call_id,
                    content,
                    ..
                } => {
                    messages.push(message_value(
                        "tool",
                        common::blocks_to_openai_content(content, "text", "image_url"),
                        Some(tool_call_id),
                        Vec::new(),
                    )?);
                }
                UniversalItem::Unknown { raw } => messages.push(raw.clone()),
                UniversalItem::Reasoning { .. } => {}
            }
        }

        body.insert("messages".to_string(), Value::Array(messages));
        Ok(Value::Object(body))
    }

    fn decode_response(&self, raw: Value) -> Result<Vec<UniversalEvent>> {
        let response: ChatCompletionResponse = serde_json::from_value(raw)
            .map_err(|error| ApiProxyError::invalid_response(error.to_string()))?;
        let usage = common::openai_usage_to_universal(response.usage.as_ref());
        let mut events = vec![UniversalEvent::ResponseStart {
            id: response.id.clone(),
            model: response.model.clone(),
            extensions: common::empty_extensions(),
        }];

        for choice in response.choices {
            if let Some(message) = choice.message {
                let role = common::role_from_wire(&message.role).unwrap_or(Role::Assistant);
                let message_id = response_message_id(response.id.as_deref(), choice.index);
                events.push(UniversalEvent::MessageStart {
                    id: message_id,
                    role,
                    extensions: common::empty_extensions(),
                });

                let mut next_index = 0;
                for block in common::openai_content_to_blocks(message.content.as_ref()) {
                    common::push_block_events(&mut events, next_index, block);
                    next_index += 1;
                }
                for tool_call in message.tool_calls {
                    common::push_block_events(
                        &mut events,
                        next_index,
                        chat_tool_call_to_block(tool_call),
                    );
                    next_index += 1;
                }
                events.push(UniversalEvent::MessageDone {
                    finish_reason: common::finish_from_openai(choice.finish_reason.as_deref()),
                    usage: usage.clone(),
                    extensions: common::empty_extensions(),
                });
            }
        }

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
        let chunk: ChatCompletionChunk = serde_json::from_value(raw)
            .map_err(|error| ApiProxyError::invalid_response(error.to_string()))?;
        let mut events = Vec::new();
        common::ensure_response_start(&mut events, state, chunk.id.clone(), chunk.model.clone());

        let usage = common::openai_usage_to_universal(chunk.usage.as_ref());
        for choice in chunk.choices {
            let choice_index = choice.index.unwrap_or(0) as usize;
            let message_id = response_message_id(chunk.id.as_deref(), choice.index);

            if let Some(delta) = choice.delta {
                let role = delta
                    .role
                    .as_deref()
                    .and_then(common::role_from_wire)
                    .unwrap_or(Role::Assistant);
                if delta.role.is_some() || delta.content.is_some() || !delta.tool_calls.is_empty() {
                    common::ensure_message_start(&mut events, state, message_id, role);
                }

                for block in common::openai_content_to_blocks(delta.content.as_ref()) {
                    match block {
                        ContentBlock::Text { text } => {
                            common::ensure_content_start(
                                &mut events,
                                state,
                                choice_index,
                                ContentBlock::Text {
                                    text: String::new(),
                                },
                            );
                            events.push(UniversalEvent::TextDelta {
                                index: choice_index,
                                text,
                            });
                        }
                        block => common::push_block_events(&mut events, choice_index, block),
                    }
                }

                for tool_call in delta.tool_calls {
                    let id = tool_call
                        .id
                        .clone()
                        .unwrap_or_else(|| format!("tool_call_{choice_index}"));
                    let function = tool_call.function;
                    events.push(UniversalEvent::ToolCallDelta {
                        id,
                        name: function.as_ref().and_then(|function| function.name.clone()),
                        arguments_delta: function
                            .and_then(|function| function.arguments)
                            .unwrap_or_default(),
                    });
                }
            }

            if choice.finish_reason.is_some() {
                events.push(UniversalEvent::MessageDone {
                    finish_reason: common::finish_from_openai(choice.finish_reason.as_deref()),
                    usage: usage.clone(),
                    extensions: common::empty_extensions(),
                });
                if common::mark_once(state, "response_done") {
                    events.push(UniversalEvent::ResponseDone {
                        usage: usage.clone(),
                        extensions: common::empty_extensions(),
                    });
                }
            }
        }

        if usage.is_some() && common::mark_once(state, "response_done") {
            events.push(UniversalEvent::ResponseDone {
                usage,
                extensions: common::empty_extensions(),
            });
        }

        Ok(events)
    }

    fn encode_events(
        &self,
        events: &[UniversalEvent],
        state: &mut EncodeState,
    ) -> Result<Vec<WireEvent>> {
        Ok(events
            .iter()
            .map(|event| match event {
                UniversalEvent::ResponseStart { id, model, .. } => common::wire_event(json!({
                    "id": id,
                    "model": model,
                    "choices": []
                })),
                UniversalEvent::MessageStart { role, .. } => common::wire_event(json!({
                    "choices": [{
                        "index": 0,
                        "delta": { "role": common::role_to_openai(*role) }
                    }]
                })),
                UniversalEvent::TextDelta { index, text } => common::wire_event(json!({
                    "choices": [{
                        "index": index,
                        "delta": { "content": text }
                    }]
                })),
                UniversalEvent::ToolCallDelta {
                    id,
                    name,
                    arguments_delta,
                } => {
                    let index = common::encode_state_index(state);
                    common::wire_event(json!({
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": index,
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": arguments_delta
                                    }
                                }]
                            }
                        }]
                    }))
                }
                UniversalEvent::MessageDone { finish_reason, .. } => common::wire_event(json!({
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_to_openai(*finish_reason)
                    }]
                })),
                UniversalEvent::ResponseDone { usage, .. } => common::wire_event(json!({
                    "choices": [],
                    "usage": usage_to_openai_value(usage.as_ref())
                })),
                UniversalEvent::Error { message, raw } => common::wire_event(json!({
                    "error": {
                        "message": message,
                        "raw": raw
                    }
                })),
                UniversalEvent::Unknown { raw, .. } => common::wire_event(raw.clone()),
                UniversalEvent::ContentStart { .. }
                | UniversalEvent::ReasoningDelta { .. }
                | UniversalEvent::ContentDone { .. } => common::wire_event(json!({
                    "choices": []
                })),
            })
            .collect())
    }
}

fn decode_message(message: ChatMessage, request: &mut UniversalRequest) {
    let role = common::role_from_wire(&message.role);
    let blocks = common::openai_content_to_blocks(message.content.as_ref());

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

fn chat_tool_call_to_block(tool_call: ChatToolCall) -> ContentBlock {
    let function = tool_call.function;
    ContentBlock::ToolCall {
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

fn response_message_id(response_id: Option<&str>, choice_index: Option<u64>) -> String {
    format!(
        "{}:{}",
        response_id.unwrap_or("chatcmpl"),
        choice_index.unwrap_or(0)
    )
}

fn finish_to_openai(reason: Option<crate::FinishReason>) -> Value {
    match reason {
        Some(crate::FinishReason::Stop) => json!("stop"),
        Some(crate::FinishReason::Length) => json!("length"),
        Some(crate::FinishReason::ToolCall) => json!("tool_calls"),
        Some(crate::FinishReason::ContentFilter) => json!("content_filter"),
        Some(crate::FinishReason::Error) => json!("error"),
        Some(crate::FinishReason::Unknown) | None => Value::Null,
    }
}

fn usage_to_openai_value(usage: Option<&crate::Usage>) -> Value {
    match usage {
        Some(usage) => json!({
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens
        }),
        None => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn decodes_chat_request_to_universal_items() {
        let translator = OpenAiChatTranslator;
        let request = translator
            .decode_request(json!({
                "model": "gpt-test",
                "stream": true,
                "max_completion_tokens": 128,
                "messages": [
                    { "role": "system", "content": "be crisp" },
                    { "role": "user", "content": "weather?" },
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"city\":\"Paris\"}"
                            }
                        }]
                    },
                    { "role": "tool", "tool_call_id": "call_1", "content": "sunny" }
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": { "type": "object" }
                    }
                }]
            }))
            .unwrap();

        assert_eq!(request.model.as_deref(), Some("gpt-test"));
        assert!(request.stream);
        assert_eq!(request.instructions.len(), 1);
        assert_eq!(request.tools[0].name, "get_weather");
        assert!(matches!(request.input[1], UniversalItem::ToolCall { .. }));
        assert!(matches!(request.input[2], UniversalItem::ToolResult { .. }));
    }

    #[test]
    fn decodes_chat_stream_text_delta() {
        let translator = OpenAiChatTranslator;
        let mut state = DecodeState::default();
        let events = translator
            .decode_stream_chunk(
                json!({
                    "id": "chatcmpl_1",
                    "model": "gpt-test",
                    "choices": [{
                        "index": 0,
                        "delta": { "role": "assistant", "content": "hi" }
                    }]
                }),
                &mut state,
            )
            .unwrap();

        assert!(matches!(events[0], UniversalEvent::ResponseStart { .. }));
        assert!(events.iter().any(|event| matches!(
            event,
            UniversalEvent::TextDelta { text, .. } if text == "hi"
        )));
    }
}
