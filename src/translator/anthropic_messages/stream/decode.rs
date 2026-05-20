use serde_json::Value;

use crate::schema::anthropic::AnthropicStreamEvent;
use crate::translator::{anthropic, common};
use crate::{ApiBridgeError, DecodeState, Result, Role, UniversalEvent};

pub(super) fn decode_chunk(raw: Value, state: &mut DecodeState) -> Result<Vec<UniversalEvent>> {
    let raw_for_unknown = raw.clone();
    let event: AnthropicStreamEvent = serde_json::from_value(raw)
        .map_err(|error| ApiBridgeError::invalid_response(error.to_string()))?;
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
                let index = event.index.unwrap_or(0);
                remember_tool_block(state, index, &block);
                common::ensure_content_start(
                    &mut events,
                    state,
                    index,
                    anthropic::anthropic_block_to_block(&block),
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
                    Some("input_json_delta") => {
                        let id = tool_id_for_index(state, index)
                            .unwrap_or_else(|| format!("tool_call_{index}"));
                        events.push(UniversalEvent::ToolCallDelta {
                            id,
                            name: tool_name_for_index(state, index),
                            arguments_delta: delta.partial_json.unwrap_or_default(),
                        });
                    }
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
                finish_reason: anthropic::finish_from_anthropic(finish_reason),
                usage: anthropic::anthropic_usage_to_universal(event.usage.as_ref()),
                extensions: common::empty_extensions(),
            });
        }
        "message_stop" => {
            if common::mark_once(state, "response_done") {
                events.push(UniversalEvent::ResponseDone {
                    usage: anthropic::anthropic_usage_to_universal(event.usage.as_ref()),
                    extensions: common::empty_extensions(),
                });
            }
        }
        "ping" => {}
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

fn remember_tool_block(
    state: &mut DecodeState,
    index: usize,
    block: &crate::schema::anthropic::AnthropicContentBlock,
) {
    if block.kind != "tool_use" {
        return;
    }
    if let Some(id) = block.id.as_ref().filter(|id| !id.is_empty()) {
        state
            .extensions
            .insert(tool_id_key(index), Value::String(id.clone()));
    }
    if let Some(name) = block.name.as_ref().filter(|name| !name.is_empty()) {
        state
            .extensions
            .insert(tool_name_key(index), Value::String(name.clone()));
    }
}

fn tool_id_for_index(state: &DecodeState, index: usize) -> Option<String> {
    state
        .extensions
        .get(&tool_id_key(index))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn tool_name_for_index(state: &DecodeState, index: usize) -> Option<String> {
    state
        .extensions
        .get(&tool_name_key(index))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .map(ToString::to_string)
}

fn tool_id_key(index: usize) -> String {
    format!("anthropic_tool_id:{index}")
}

fn tool_name_key(index: usize) -> String {
    format!("anthropic_tool_name:{index}")
}
