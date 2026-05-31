mod state;

use serde_json::{json, Value};

use crate::translator::{anthropic, common};
use crate::{EncodeState, Result, UniversalEvent, WireEvent};

use self::state::*;

pub(super) fn encode(events: &[UniversalEvent], state: &mut EncodeState) -> Result<Vec<WireEvent>> {
    let mut wire_events = Vec::new();
    for event in events {
        match event {
            UniversalEvent::ResponseStart { id, model, .. } => {
                wire_events.push(common::wire_event(json!({
                    "type": "message_start",
                    "message": {
                        "id": id,
                        "type": "message",
                        "model": model,
                        "role": "assistant",
                        "content": [],
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0
                        }
                    }
                })))
            }
            UniversalEvent::ContentStart { index, block } => {
                reserve_index(state, *index);
                remember_content_started(state, *index);
                match block {
                    crate::ContentBlock::ToolCall { id, name, .. } => {
                        remember_tool_id(state, id);
                        remember_tool_index(state, id, *index);
                        state
                            .extensions
                            .insert(tool_name_key(id), Value::String(name.clone()));
                        state
                            .extensions
                            .insert(tool_started_key(id), Value::Bool(true));
                        wire_events.push(common::wire_event(json!({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": {}
                            }
                        })));
                    }
                    _ => wire_events.push(common::wire_event(json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": anthropic::block_to_anthropic_block(block)
                    }))),
                }
            }
            UniversalEvent::TextDelta { index, text } => {
                wire_events.push(common::wire_event(json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "text_delta",
                        "text": text
                    }
                })))
            }
            UniversalEvent::ReasoningDelta { index, text } => {
                wire_events.push(common::wire_event(json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": text
                    }
                })))
            }
            UniversalEvent::ToolCallDelta {
                id,
                name,
                arguments_delta,
                ..
            } => {
                let index = ensure_tool_block_started(state, &mut wire_events, id, name.as_deref());
                if !arguments_delta.is_empty() {
                    wire_events.push(common::wire_event(json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arguments_delta
                        }
                    })));
                }
            }
            UniversalEvent::ContentDone { index, .. } => {
                remember_content_closed(state, *index);
                if let Some(id) = tool_id_for_index(state, *index) {
                    state
                        .extensions
                        .insert(tool_closed_key(&id), Value::Bool(true));
                }
                wire_events.push(common::wire_event(json!({
                    "type": "content_block_stop",
                    "index": index
                })));
            }
            UniversalEvent::MessageDone {
                finish_reason,
                usage,
                ..
            } => {
                remember_message_done(state, *finish_reason, usage);
            }
            UniversalEvent::ResponseDone { usage, .. } => {
                close_open_content_blocks(state, &mut wire_events);
                close_open_tool_blocks(state, &mut wire_events);
                if !message_delta_sent(state) {
                    let usage = response_usage(usage, state);
                    wire_events.push(common::wire_event(json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": finish_to_anthropic(normalize_finish_reason(
                                pending_finish_reason(state),
                                state,
                            )),
                            "stop_sequence": Value::Null
                        },
                        "usage": usage
                    })));
                    state
                        .extensions
                        .insert("anthropicMessageDeltaSent".to_string(), Value::Bool(true));
                }
                wire_events.push(common::wire_event(json!({
                "type": "message_stop"
                })));
            }
            UniversalEvent::Error { message, raw } => wire_events.push(common::wire_event(json!({
                "type": "error",
                "error": {
                    "message": message,
                    "raw": raw
                }
            }))),
            UniversalEvent::Unknown { raw, .. } => {
                wire_events.push(common::wire_event(raw.clone()))
            }
            UniversalEvent::MessageStart { .. } => {}
        }
    }
    Ok(wire_events)
}

#[cfg(test)]
mod tests;
