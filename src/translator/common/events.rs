use serde_json::Value;

use crate::{ContentBlock, DecodeState, EncodeState, Role, UniversalEvent, WireEvent};

use super::{empty_extensions, stringify_arguments};

pub(crate) fn push_block_events(
    events: &mut Vec<UniversalEvent>,
    index: usize,
    block: ContentBlock,
) {
    events.push(UniversalEvent::ContentStart {
        index,
        block: block.clone(),
    });
    match &block {
        ContentBlock::Text { text } if !text.is_empty() => {
            events.push(UniversalEvent::TextDelta {
                index,
                text: text.clone(),
            });
        }
        ContentBlock::Reasoning {
            text: Some(text), ..
        } if !text.is_empty() => {
            events.push(UniversalEvent::ReasoningDelta {
                index,
                text: text.clone(),
            });
        }
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
            ..
        } => {
            events.push(UniversalEvent::ToolCallDelta {
                id: id.clone(),
                name: Some(name.clone()),
                arguments_delta: stringify_arguments(arguments),
            });
        }
        _ => {}
    }
    events.push(UniversalEvent::ContentDone {
        index,
        final_block: Some(block),
    });
}

pub(crate) fn ensure_response_start(
    events: &mut Vec<UniversalEvent>,
    state: &mut DecodeState,
    id: Option<String>,
    model: Option<String>,
) {
    if mark_once(state, "response_start") {
        events.push(UniversalEvent::ResponseStart {
            id,
            model,
            extensions: empty_extensions(),
        });
    }
}

pub(crate) fn ensure_message_start(
    events: &mut Vec<UniversalEvent>,
    state: &mut DecodeState,
    id: String,
    role: Role,
) {
    if mark_once(state, &format!("message_start:{id}")) {
        events.push(UniversalEvent::MessageStart {
            id,
            role,
            extensions: empty_extensions(),
        });
    }
}

pub(crate) fn ensure_content_start(
    events: &mut Vec<UniversalEvent>,
    state: &mut DecodeState,
    index: usize,
    block: ContentBlock,
) {
    if mark_once(state, &format!("content_start:{index}")) {
        events.push(UniversalEvent::ContentStart { index, block });
    }
}

pub(crate) fn mark_once(state: &mut DecodeState, key: &str) -> bool {
    let previous = state.extensions.insert(key.to_string(), Value::Bool(true));
    !matches!(previous, Some(Value::Bool(true)))
}

pub(crate) fn wire_event(data: Value) -> WireEvent {
    WireEvent { event: None, data }
}

pub(crate) fn encode_state_index(state: &mut EncodeState) -> usize {
    let key = "nextIndex".to_string();
    let next = state
        .extensions
        .get(&key)
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    state
        .extensions
        .insert(key, Value::Number(((next + 1) as u64).into()));
    next
}
