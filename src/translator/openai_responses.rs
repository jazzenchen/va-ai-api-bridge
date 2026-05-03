use serde_json::{json, Map, Value};

use crate::schema::openai::{
    ResponsesInput, ResponsesItem, ResponsesRequest, ResponsesResponse, ResponsesStreamEvent,
};
use crate::translator::common;
use crate::{
    ApiProxyError, ContentBlock, DecodeState, EncodeState, Result, Role, UniversalEvent,
    UniversalItem, UniversalRequest, WireEvent, WireProtocol,
};

use super::WireTranslator;

#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAiResponsesTranslator;

impl WireTranslator for OpenAiResponsesTranslator {
    fn protocol(&self) -> WireProtocol {
        WireProtocol::OpenAiResponses
    }

    fn decode_request(&self, raw: Value) -> Result<UniversalRequest> {
        let source_raw = raw.clone();
        let request: ResponsesRequest = serde_json::from_value(raw)
            .map_err(|error| ApiProxyError::invalid_request(error.to_string()))?;

        let mut universal = UniversalRequest {
            model: request.model,
            instructions: common::openai_content_to_blocks(request.instructions.as_ref()),
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
                request.max_output_tokens,
            ),
            reasoning: common::reasoning_from_openai(request.reasoning),
            source: Some(common::source(WireProtocol::OpenAiResponses, source_raw)),
            ..UniversalRequest::default()
        };

        match request.input {
            Some(ResponsesInput::Text(text)) => universal.input.push(UniversalItem::Message {
                role: Role::User,
                id: None,
                content: vec![ContentBlock::Text { text }],
                extensions: common::empty_extensions(),
            }),
            Some(ResponsesInput::Items(items)) => {
                universal
                    .input
                    .extend(items.into_iter().map(item_to_universal));
            }
            Some(ResponsesInput::Raw(raw)) => universal.input.push(UniversalItem::Unknown { raw }),
            Some(ResponsesInput::Null) | None => {}
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
            body.insert("max_output_tokens".to_string(), json!(max_output_tokens));
        }
        if let Some(reasoning) = &request.reasoning {
            let mut value = Map::new();
            if let Some(effort) = &reasoning.effort {
                value.insert("effort".to_string(), Value::String(effort.clone()));
            }
            body.insert("reasoning".to_string(), Value::Object(value));
        }
        if !request.instructions.is_empty() {
            if let Some(instructions) =
                common::blocks_to_openai_content(&request.instructions, "input_text", "input_image")
            {
                body.insert(
                    "instructions".to_string(),
                    serde_json::to_value(instructions)
                        .map_err(|error| ApiProxyError::conversion(error.to_string()))?,
                );
            }
        }
        if !request.tools.is_empty() {
            body.insert(
                "tools".to_string(),
                Value::Array(
                    request
                        .tools
                        .iter()
                        .map(common::tool_to_openai_responses)
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

        let input: Vec<Value> = request
            .input
            .iter()
            .map(universal_item_to_response_item)
            .collect::<Result<Vec<_>>>()?;
        if !input.is_empty() {
            body.insert("input".to_string(), Value::Array(input));
        }

        Ok(Value::Object(body))
    }

    fn decode_response(&self, raw: Value) -> Result<Vec<UniversalEvent>> {
        let response: ResponsesResponse = serde_json::from_value(raw)
            .map_err(|error| ApiProxyError::invalid_response(error.to_string()))?;
        let usage = common::openai_usage_to_universal(response.usage.as_ref());
        let mut events = vec![UniversalEvent::ResponseStart {
            id: response.id.clone(),
            model: response.model.clone(),
            extensions: common::empty_extensions(),
        }];

        if let Some(error) = response.error {
            events.push(UniversalEvent::Error {
                message: error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("OpenAI response error")
                    .to_string(),
                raw: Some(error),
            });
        }

        for (index, item) in response.output.into_iter().enumerate() {
            push_response_item_events(&mut events, response.id.as_deref(), index, item);
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
        let raw_for_unknown = raw.clone();
        let event: ResponsesStreamEvent = serde_json::from_value(raw)
            .map_err(|error| ApiProxyError::invalid_response(error.to_string()))?;
        let mut events = Vec::new();
        let kind = event.kind.as_deref().unwrap_or_default();

        match kind {
            "response.created" | "response.in_progress" | "response.queued" => {
                let response = event.response;
                common::ensure_response_start(
                    &mut events,
                    state,
                    response.as_ref().and_then(|response| response.id.clone()),
                    response
                        .as_ref()
                        .and_then(|response| response.model.clone()),
                );
            }
            "response.output_item.added" => {
                common::ensure_response_start(&mut events, state, None, None);
                if let Some(item) = event.item {
                    push_response_item_start(
                        &mut events,
                        state,
                        event.output_index.unwrap_or(0),
                        item,
                    );
                }
            }
            "response.content_part.added" => {
                let index = event.content_index.or(event.output_index).unwrap_or(0);
                if let Some(item) = event.item {
                    let blocks = item
                        .content
                        .as_ref()
                        .map(|content| common::openai_content_to_blocks(Some(content)))
                        .unwrap_or_default();
                    let block = blocks.into_iter().next().unwrap_or(ContentBlock::Text {
                        text: String::new(),
                    });
                    common::ensure_content_start(&mut events, state, index, block);
                }
            }
            "response.output_text.delta" => {
                let index = event.content_index.or(event.output_index).unwrap_or(0);
                common::ensure_content_start(
                    &mut events,
                    state,
                    index,
                    ContentBlock::Text {
                        text: String::new(),
                    },
                );
                events.push(UniversalEvent::TextDelta {
                    index,
                    text: value_to_string(event.delta.as_ref()),
                });
            }
            "response.reasoning_text.delta" | "response.reasoning.delta" => {
                events.push(UniversalEvent::ReasoningDelta {
                    index: event.content_index.or(event.output_index).unwrap_or(0),
                    text: value_to_string(event.delta.as_ref()),
                });
            }
            "response.function_call_arguments.delta" => {
                events.push(UniversalEvent::ToolCallDelta {
                    id: event.item_id.unwrap_or_else(|| "function_call".to_string()),
                    name: None,
                    arguments_delta: value_to_string(event.delta.as_ref()),
                });
            }
            "response.output_item.done" => {
                if let Some(item) = event.item {
                    push_response_item_events(
                        &mut events,
                        event
                            .response
                            .as_ref()
                            .and_then(|response| response.id.as_deref()),
                        event.output_index.unwrap_or(0),
                        item,
                    );
                }
            }
            "response.content_part.done" => {
                events.push(UniversalEvent::ContentDone {
                    index: event.content_index.or(event.output_index).unwrap_or(0),
                    final_block: event
                        .item
                        .as_ref()
                        .and_then(|item| item.content.as_ref())
                        .and_then(|content| {
                            common::openai_content_to_blocks(Some(content))
                                .into_iter()
                                .next()
                        }),
                });
            }
            "response.completed" => {
                let response = event.response;
                common::ensure_response_start(
                    &mut events,
                    state,
                    response.as_ref().and_then(|response| response.id.clone()),
                    response
                        .as_ref()
                        .and_then(|response| response.model.clone()),
                );
                if common::mark_once(state, "response_done") {
                    events.push(UniversalEvent::ResponseDone {
                        usage: common::openai_usage_to_universal(
                            response
                                .as_ref()
                                .and_then(|response| response.usage.as_ref()),
                        ),
                        extensions: common::empty_extensions(),
                    });
                }
            }
            "response.failed" | "response.incomplete" => {
                events.push(UniversalEvent::Error {
                    message: event
                        .error
                        .as_ref()
                        .and_then(|error| error.get("message"))
                        .and_then(Value::as_str)
                        .unwrap_or(kind)
                        .to_string(),
                    raw: event.error,
                });
            }
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
                    "type": "response.created",
                    "response": {
                        "id": id,
                        "model": model
                    }
                })),
                UniversalEvent::TextDelta { index, text } => common::wire_event(json!({
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": index,
                    "delta": text
                })),
                UniversalEvent::ReasoningDelta { index, text } => common::wire_event(json!({
                    "type": "response.reasoning_text.delta",
                    "output_index": 0,
                    "content_index": index,
                    "delta": text
                })),
                UniversalEvent::ToolCallDelta {
                    id,
                    arguments_delta,
                    ..
                } => common::wire_event(json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": id,
                    "delta": arguments_delta
                })),
                UniversalEvent::ResponseDone { usage, .. } => common::wire_event(json!({
                    "type": "response.completed",
                    "response": {
                        "usage": usage
                    }
                })),
                UniversalEvent::Error { message, raw } => common::wire_event(json!({
                    "type": "response.failed",
                    "error": {
                        "message": message,
                        "raw": raw
                    }
                })),
                UniversalEvent::Unknown { raw, .. } => common::wire_event(raw.clone()),
                UniversalEvent::MessageStart { .. }
                | UniversalEvent::ContentStart { .. }
                | UniversalEvent::ContentDone { .. }
                | UniversalEvent::MessageDone { .. } => common::wire_event(json!({
                    "type": "response.event"
                })),
            })
            .collect())
    }
}

fn item_to_universal(item: ResponsesItem) -> UniversalItem {
    match item.kind.as_deref() {
        Some("message") | None if item.role.is_some() => UniversalItem::Message {
            role: item
                .role
                .as_deref()
                .and_then(common::role_from_wire)
                .unwrap_or(Role::User),
            id: item.id,
            content: common::openai_content_to_blocks(item.content.as_ref()),
            extensions: common::empty_extensions(),
        },
        Some("function_call") => UniversalItem::ToolCall {
            id: item.call_id.or(item.id).unwrap_or_default(),
            name: item.name.unwrap_or_default(),
            arguments: item.arguments.unwrap_or(Value::Null),
            extensions: common::empty_extensions(),
        },
        Some("function_call_output") => UniversalItem::ToolResult {
            tool_call_id: item.call_id.or(item.id).unwrap_or_default(),
            content: value_to_blocks(item.output.as_ref()),
            is_error: false,
            extensions: common::empty_extensions(),
        },
        Some("reasoning") => UniversalItem::Reasoning {
            id: item.id,
            text: item
                .content
                .as_ref()
                .and_then(|content| text_from_openai_content(Some(content))),
            encrypted: None,
            extensions: common::empty_extensions(),
        },
        _ => UniversalItem::Unknown {
            raw: serde_json::to_value(item).unwrap_or(Value::Null),
        },
    }
}

fn universal_item_to_response_item(item: &UniversalItem) -> Result<Value> {
    match item {
        UniversalItem::Message {
            role, id, content, ..
        } => {
            let mut object = Map::new();
            object.insert("type".to_string(), Value::String("message".to_string()));
            object.insert(
                "role".to_string(),
                Value::String(common::role_to_openai(*role).to_string()),
            );
            if let Some(id) = id {
                object.insert("id".to_string(), Value::String(id.clone()));
            }
            if let Some(content) =
                common::blocks_to_openai_content(content, "input_text", "input_image")
            {
                object.insert(
                    "content".to_string(),
                    serde_json::to_value(content)
                        .map_err(|error| ApiProxyError::conversion(error.to_string()))?,
                );
            }
            Ok(Value::Object(object))
        }
        UniversalItem::ToolCall {
            id,
            name,
            arguments,
            ..
        } => Ok(json!({
            "type": "function_call",
            "call_id": id,
            "name": name,
            "arguments": arguments
        })),
        UniversalItem::ToolResult {
            tool_call_id,
            content,
            ..
        } => Ok(json!({
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": blocks_to_value(content)
        })),
        UniversalItem::Reasoning {
            id,
            text,
            encrypted,
            ..
        } => Ok(json!({
            "type": "reasoning",
            "id": id,
            "content": text,
            "encrypted": encrypted
        })),
        UniversalItem::Unknown { raw } => Ok(raw.clone()),
    }
}

fn push_response_item_start(
    events: &mut Vec<UniversalEvent>,
    state: &mut DecodeState,
    index: usize,
    item: ResponsesItem,
) {
    match item.kind.as_deref() {
        Some("message") | None if item.role.is_some() => common::ensure_message_start(
            events,
            state,
            item.id
                .unwrap_or_else(|| format!("response_message_{index}")),
            item.role
                .as_deref()
                .and_then(common::role_from_wire)
                .unwrap_or(Role::Assistant),
        ),
        Some("function_call") => events.push(UniversalEvent::ToolCallDelta {
            id: item.call_id.or(item.id).unwrap_or_default(),
            name: item.name,
            arguments_delta: item
                .arguments
                .as_ref()
                .map(common::stringify_arguments)
                .unwrap_or_default(),
        }),
        _ => {}
    }
}

fn push_response_item_events(
    events: &mut Vec<UniversalEvent>,
    response_id: Option<&str>,
    index: usize,
    item: ResponsesItem,
) {
    match item.kind.as_deref() {
        Some("message") | None if item.role.is_some() => {
            events.push(UniversalEvent::MessageStart {
                id: item
                    .id
                    .unwrap_or_else(|| format!("{}:{index}", response_id.unwrap_or("response"))),
                role: item
                    .role
                    .as_deref()
                    .and_then(common::role_from_wire)
                    .unwrap_or(Role::Assistant),
                extensions: common::empty_extensions(),
            });
            for (content_index, block) in common::openai_content_to_blocks(item.content.as_ref())
                .into_iter()
                .enumerate()
            {
                common::push_block_events(events, content_index, block);
            }
            events.push(UniversalEvent::MessageDone {
                finish_reason: None,
                usage: None,
                extensions: common::empty_extensions(),
            });
        }
        Some("function_call") => {
            events.push(UniversalEvent::ToolCallDelta {
                id: item.call_id.or(item.id).unwrap_or_default(),
                name: item.name,
                arguments_delta: item
                    .arguments
                    .as_ref()
                    .map(common::stringify_arguments)
                    .unwrap_or_default(),
            });
        }
        Some("reasoning") => {
            if let Some(text) = text_from_openai_content(item.content.as_ref()) {
                events.push(UniversalEvent::ReasoningDelta { index, text });
            }
        }
        _ => events.push(UniversalEvent::Unknown {
            raw: serde_json::to_value(item).unwrap_or(Value::Null),
            tags: Default::default(),
        }),
    }
}

fn value_to_string(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(value) => common::stringify_arguments(value),
        None => String::new(),
    }
}

fn text_from_openai_content(
    content: Option<&crate::schema::openai::OpenAiContent>,
) -> Option<String> {
    let text = common::openai_content_to_blocks(content)
        .into_iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text),
            ContentBlock::Reasoning { text, .. } => text,
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    (!text.is_empty()).then_some(text)
}

fn value_to_blocks(value: Option<&Value>) -> Vec<ContentBlock> {
    match value {
        Some(Value::String(text)) => vec![ContentBlock::Text { text: text.clone() }],
        Some(Value::Array(values)) => values
            .iter()
            .cloned()
            .map(|raw| ContentBlock::Unknown { raw })
            .collect(),
        Some(value) => vec![ContentBlock::Unknown { raw: value.clone() }],
        None => Vec::new(),
    }
}

fn blocks_to_value(blocks: &[ContentBlock]) -> Value {
    match blocks {
        [ContentBlock::Text { text }] => Value::String(text.clone()),
        blocks => Value::Array(
            blocks
                .iter()
                .map(|block| serde_json::to_value(block).unwrap_or(Value::Null))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn decodes_responses_request_to_universal_items() {
        let translator = OpenAiResponsesTranslator;
        let request = translator
            .decode_request(json!({
                "model": "gpt-test",
                "instructions": "be crisp",
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "hello" }]
                }],
                "tools": [{
                    "type": "function",
                    "name": "lookup",
                    "parameters": { "type": "object" }
                }]
            }))
            .unwrap();

        assert_eq!(request.model.as_deref(), Some("gpt-test"));
        assert_eq!(request.instructions.len(), 1);
        assert_eq!(request.tools[0].name, "lookup");
        assert!(matches!(request.input[0], UniversalItem::Message { .. }));
    }

    #[test]
    fn decodes_responses_text_delta() {
        let translator = OpenAiResponsesTranslator;
        let mut state = DecodeState::default();
        let events = translator
            .decode_stream_chunk(
                json!({
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "hi"
                }),
                &mut state,
            )
            .unwrap();

        assert!(events.iter().any(|event| matches!(
            event,
            UniversalEvent::TextDelta { text, .. } if text == "hi"
        )));
    }
}
