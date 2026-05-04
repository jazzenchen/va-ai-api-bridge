use serde_json::{json, Value};

use crate::translator::{anthropic, common};
use crate::{EncodeState, Result, UniversalEvent, Usage, WireEvent};

pub(super) fn encode(events: &[UniversalEvent], state: &mut EncodeState) -> Result<Vec<WireEvent>> {
    let mut wire_events = Vec::new();
    for event in events {
        match event {
            UniversalEvent::ResponseStart { id, model, .. } => {
                wire_events.push(common::wire_event(json!({
                    "type": "message_start",
                    "message": {
                        "id": id,
                        "model": model,
                        "role": "assistant"
                    }
                })))
            }
            UniversalEvent::ContentStart { index, block } => {
                reserve_index(state, *index);
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
                close_open_tool_blocks(state, &mut wire_events);
                if !message_delta_sent(state) {
                    let usage = response_usage(usage, state);
                    wire_events.push(common::wire_event(json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": finish_to_anthropic(normalize_finish_reason(
                                pending_finish_reason(state),
                                state,
                            ))
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

fn normalize_finish_reason(
    reason: Option<crate::FinishReason>,
    state: &EncodeState,
) -> Option<crate::FinishReason> {
    match reason {
        Some(reason) => Some(reason),
        None if has_tool_blocks(state) => Some(crate::FinishReason::ToolCall),
        None => Some(crate::FinishReason::Stop),
    }
}

fn ensure_tool_block_started(
    state: &mut EncodeState,
    wire_events: &mut Vec<WireEvent>,
    id: &str,
    name: Option<&str>,
) -> usize {
    let index = tool_block_index(state, id);
    if let Some(name) = name.filter(|name| !name.is_empty()) {
        state
            .extensions
            .insert(tool_name_key(id), Value::String(name.to_string()));
    }
    remember_tool_id(state, id);
    let previous = state
        .extensions
        .insert(tool_started_key(id), Value::Bool(true));
    if !matches!(previous, Some(Value::Bool(true))) {
        wire_events.push(common::wire_event(json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": id,
                "name": tool_name(state, id),
                "input": {}
            }
        })));
    }
    index
}

fn close_open_tool_blocks(state: &mut EncodeState, wire_events: &mut Vec<WireEvent>) {
    for id in tool_ids(state) {
        let previous = state
            .extensions
            .insert(tool_closed_key(&id), Value::Bool(true));
        if matches!(previous, Some(Value::Bool(true))) {
            continue;
        }
        wire_events.push(common::wire_event(json!({
            "type": "content_block_stop",
            "index": tool_block_index(state, &id)
        })));
    }
}

fn has_tool_blocks(state: &EncodeState) -> bool {
    !tool_ids(state).is_empty()
}

fn message_delta_sent(state: &EncodeState) -> bool {
    matches!(
        state.extensions.get("anthropicMessageDeltaSent"),
        Some(Value::Bool(true))
    )
}

fn remember_message_done(
    state: &mut EncodeState,
    finish_reason: Option<crate::FinishReason>,
    usage: &Option<Usage>,
) {
    if let Some(reason) = finish_reason {
        if let Ok(value) = serde_json::to_value(reason) {
            state
                .extensions
                .insert("anthropicPendingFinishReason".to_string(), value);
        }
    }
    if let Some(usage) = usage {
        state.extensions.insert(
            "anthropicPendingUsage".to_string(),
            usage_to_anthropic_value(usage),
        );
    }
}

fn pending_finish_reason(state: &EncodeState) -> Option<crate::FinishReason> {
    state
        .extensions
        .get("anthropicPendingFinishReason")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
}

fn pending_usage(state: &EncodeState) -> Value {
    state
        .extensions
        .get("anthropicPendingUsage")
        .cloned()
        .unwrap_or(Value::Null)
}

fn response_usage(usage: &Option<Usage>, state: &EncodeState) -> Value {
    usage
        .as_ref()
        .map(usage_to_anthropic_value)
        .unwrap_or_else(|| pending_usage(state))
}

fn usage_to_anthropic_value(usage: &Usage) -> Value {
    json!({
        "input_tokens": usage.input_tokens.unwrap_or(0),
        "output_tokens": usage.output_tokens.unwrap_or(0)
    })
}

fn reserve_index(state: &mut EncodeState, index: usize) {
    let key = "nextIndex".to_string();
    let next = state
        .extensions
        .get(&key)
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    if next <= index {
        state
            .extensions
            .insert(key, Value::Number(((index + 1) as u64).into()));
    }
}

fn tool_block_index(state: &mut EncodeState, id: &str) -> usize {
    let key = tool_index_key(id);
    if let Some(index) = state.extensions.get(&key).and_then(Value::as_u64) {
        return index as usize;
    }
    let index = common::encode_state_index(state);
    remember_tool_index(state, id, index);
    index
}

fn remember_tool_index(state: &mut EncodeState, id: &str, index: usize) {
    state
        .extensions
        .insert(tool_index_key(id), Value::Number((index as u64).into()));
    state.extensions.insert(
        format!("anthropicToolIdForIndex:{index}"),
        Value::String(id.to_string()),
    );
}

fn tool_id_for_index(state: &EncodeState, index: usize) -> Option<String> {
    state
        .extensions
        .get(&format!("anthropicToolIdForIndex:{index}"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn remember_tool_id(state: &mut EncodeState, id: &str) {
    let mut ids = tool_ids(state);
    if ids.iter().any(|existing| existing == id) {
        return;
    }
    ids.push(id.to_string());
    state.extensions.insert(
        "anthropicToolIds".to_string(),
        Value::Array(ids.into_iter().map(Value::String).collect()),
    );
}

fn tool_ids(state: &EncodeState) -> Vec<String> {
    state
        .extensions
        .get("anthropicToolIds")
        .and_then(Value::as_array)
        .map(|ids| {
            ids.iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn tool_name(state: &EncodeState, id: &str) -> String {
    state
        .extensions
        .get(&tool_name_key(id))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .unwrap_or("function")
        .to_string()
}

fn tool_index_key(id: &str) -> String {
    format!("anthropicToolIndex:{id}")
}

fn tool_started_key(id: &str) -> String {
    format!("anthropicToolStarted:{id}")
}

fn tool_closed_key(id: &str) -> String {
    format!("anthropicToolClosed:{id}")
}

fn tool_name_key(id: &str) -> String {
    format!("anthropicToolName:{id}")
}
