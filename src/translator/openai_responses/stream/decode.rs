use serde_json::Value;

use crate::schema::openai::{ResponsesItem, ResponsesStreamEvent};
use crate::translator::{common, openai};
use crate::{ApiProxyError, ContentBlock, DecodeState, Result, UniversalEvent};

use super::super::response::{push_response_item_events, push_response_item_start};

pub(super) fn decode_chunk(raw: Value, state: &mut DecodeState) -> Result<Vec<UniversalEvent>> {
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
            let output_index = event.output_index.unwrap_or(0);
            if let Some(item) = event.item {
                remember_response_tool_item(state, output_index, &item);
                push_response_item_start(&mut events, state, output_index, item);
            }
        }
        "response.content_part.added" => {
            let output_index = event.output_index.unwrap_or(0);
            let index = event.content_index.unwrap_or(output_index);
            mark_streamed_output(state, output_index);
            if let Some(item) = event.item {
                let blocks = item
                    .content
                    .as_ref()
                    .map(|content| openai::openai_content_to_blocks(Some(content)))
                    .unwrap_or_default();
                let block = blocks.into_iter().next().unwrap_or(ContentBlock::Text {
                    text: String::new(),
                });
                common::ensure_content_start(&mut events, state, index, block);
            }
        }
        "response.output_text.delta" => {
            let output_index = event.output_index.unwrap_or(0);
            let index = event.content_index.unwrap_or(output_index);
            mark_streamed_output(state, output_index);
            mark_text_delta(state, index);
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
        "response.output_text.done" => {
            let output_index = event.output_index.unwrap_or(0);
            let index = event.content_index.unwrap_or(output_index);
            mark_streamed_output(state, output_index);
            if !has_text_delta(state, index) {
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
                    text: value_to_string(event.extra.get("text")),
                });
            }
        }
        "response.reasoning_text.delta" | "response.reasoning.delta" => {
            events.push(UniversalEvent::ReasoningDelta {
                index: event.content_index.or(event.output_index).unwrap_or(0),
                text: value_to_string(event.delta.as_ref()),
            });
        }
        "response.function_call_arguments.delta" => {
            let item_id = event.item_id.unwrap_or_else(|| "function_call".to_string());
            mark_tool_delta(state, &item_id);
            events.push(UniversalEvent::ToolCallDelta {
                id: response_tool_call_id(state, &item_id).unwrap_or_else(|| item_id.clone()),
                name: response_tool_name(state, &item_id),
                arguments_delta: value_to_string(event.delta.as_ref()),
            });
        }
        "response.function_call_arguments.done" => {
            let item_id = event.item_id.unwrap_or_else(|| "function_call".to_string());
            mark_tool_done(state, &item_id);
            if !has_tool_delta(state, &item_id) {
                events.push(UniversalEvent::ToolCallDelta {
                    id: response_tool_call_id(state, &item_id).unwrap_or_else(|| item_id.clone()),
                    name: response_tool_name(state, &item_id),
                    arguments_delta: value_to_string(event.extra.get("arguments")),
                });
            }
        }
        "response.output_item.done" => {
            let output_index = event.output_index.unwrap_or(0);
            if let Some(item) = event.item {
                remember_response_tool_item(state, output_index, &item);
                if is_function_call_item(&item) {
                    let item_id = response_tool_item_key(state, output_index, &item);
                    if has_tool_delta(state, &item_id) || has_tool_done(state, &item_id) {
                        return Ok(events);
                    }
                    push_response_item_events(
                        &mut events,
                        event
                            .response
                            .as_ref()
                            .and_then(|response| response.id.as_deref()),
                        output_index,
                        item,
                    );
                    return Ok(events);
                }
                if has_streamed_output(state, output_index) {
                    push_message_done_once(&mut events, state, output_index, None);
                    return Ok(events);
                }
                push_response_item_events(
                    &mut events,
                    event
                        .response
                        .as_ref()
                        .and_then(|response| response.id.as_deref()),
                    output_index,
                    item,
                );
                return Ok(events);
            }
            if has_streamed_output(state, output_index) {
                push_message_done_once(&mut events, state, output_index, None);
                return Ok(events);
            }
        }
        "response.content_part.done" => {
            let output_index = event.output_index.unwrap_or(0);
            let index = event.content_index.unwrap_or(output_index);
            mark_streamed_output(state, output_index);
            push_content_done_once(
                &mut events,
                state,
                index,
                event
                    .item
                    .as_ref()
                    .and_then(|item| item.content.as_ref())
                    .and_then(|content| {
                        openai::openai_content_to_blocks(Some(content))
                            .into_iter()
                            .next()
                    }),
            );
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
                    usage: openai::openai_usage_to_universal(
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

fn value_to_string(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(value) => common::stringify_arguments(value),
        None => String::new(),
    }
}

fn mark_streamed_output(state: &mut DecodeState, output_index: usize) {
    state.extensions.insert(
        format!("stream_output_seen:{output_index}"),
        Value::Bool(true),
    );
}

fn has_streamed_output(state: &DecodeState, output_index: usize) -> bool {
    matches!(
        state
            .extensions
            .get(&format!("stream_output_seen:{output_index}")),
        Some(Value::Bool(true))
    )
}

fn mark_text_delta(state: &mut DecodeState, content_index: usize) {
    state.extensions.insert(
        format!("text_delta_seen:{content_index}"),
        Value::Bool(true),
    );
}

fn has_text_delta(state: &DecodeState, content_index: usize) -> bool {
    matches!(
        state
            .extensions
            .get(&format!("text_delta_seen:{content_index}")),
        Some(Value::Bool(true))
    )
}

fn remember_response_tool_item(state: &mut DecodeState, output_index: usize, item: &ResponsesItem) {
    if !is_function_call_item(item) {
        return;
    }
    let item_id = item
        .id
        .clone()
        .unwrap_or_else(|| format!("function_call_item_{output_index}"));
    state.extensions.insert(
        format!("response_tool_item_for_output:{output_index}"),
        Value::String(item_id.clone()),
    );
    if let Some(call_id) = item.call_id.as_ref().or(item.id.as_ref()) {
        state.extensions.insert(
            response_tool_call_key(&item_id),
            Value::String(call_id.clone()),
        );
    }
    if let Some(name) = item.name.as_ref().filter(|name| !name.is_empty()) {
        state.extensions.insert(
            response_tool_name_key(&item_id),
            Value::String(name.clone()),
        );
    }
}

fn response_tool_item_key(
    state: &DecodeState,
    output_index: usize,
    item: &ResponsesItem,
) -> String {
    item.id
        .clone()
        .or_else(|| {
            state
                .extensions
                .get(&format!("response_tool_item_for_output:{output_index}"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .or_else(|| item.call_id.clone())
        .unwrap_or_else(|| format!("function_call_item_{output_index}"))
}

fn response_tool_call_id(state: &DecodeState, item_id: &str) -> Option<String> {
    state
        .extensions
        .get(&response_tool_call_key(item_id))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn response_tool_name(state: &DecodeState, item_id: &str) -> Option<String> {
    state
        .extensions
        .get(&response_tool_name_key(item_id))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .map(ToString::to_string)
}

fn mark_tool_delta(state: &mut DecodeState, item_id: &str) {
    state
        .extensions
        .insert(response_tool_delta_key(item_id), Value::Bool(true));
}

fn has_tool_delta(state: &DecodeState, item_id: &str) -> bool {
    matches!(
        state.extensions.get(&response_tool_delta_key(item_id)),
        Some(Value::Bool(true))
    )
}

fn mark_tool_done(state: &mut DecodeState, item_id: &str) {
    state
        .extensions
        .insert(response_tool_done_key(item_id), Value::Bool(true));
}

fn has_tool_done(state: &DecodeState, item_id: &str) -> bool {
    matches!(
        state.extensions.get(&response_tool_done_key(item_id)),
        Some(Value::Bool(true))
    )
}

fn is_function_call_item(item: &ResponsesItem) -> bool {
    matches!(item.kind.as_deref(), Some("function_call"))
}

fn response_tool_call_key(item_id: &str) -> String {
    format!("response_tool_call_id:{item_id}")
}

fn response_tool_name_key(item_id: &str) -> String {
    format!("response_tool_name:{item_id}")
}

fn response_tool_delta_key(item_id: &str) -> String {
    format!("response_tool_delta_seen:{item_id}")
}

fn response_tool_done_key(item_id: &str) -> String {
    format!("response_tool_done_seen:{item_id}")
}

fn push_content_done_once(
    events: &mut Vec<UniversalEvent>,
    state: &mut DecodeState,
    index: usize,
    final_block: Option<ContentBlock>,
) {
    if common::mark_once(state, &format!("content_done:{index}")) {
        events.push(UniversalEvent::ContentDone { index, final_block });
    }
}

fn push_message_done_once(
    events: &mut Vec<UniversalEvent>,
    state: &mut DecodeState,
    index: usize,
    finish_reason: Option<crate::FinishReason>,
) {
    if common::mark_once(state, &format!("message_done:{index}")) {
        events.push(UniversalEvent::MessageDone {
            finish_reason,
            usage: None,
            extensions: common::empty_extensions(),
        });
    }
}
