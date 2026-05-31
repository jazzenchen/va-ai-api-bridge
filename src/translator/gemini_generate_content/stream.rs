use serde_json::{json, Value};

use crate::translator::WireEvent;
use crate::{ContentBlock, DecodeState, EncodeState, Result, UniversalEvent, Usage};

use super::response::decode_candidates;
use super::shared::{
    finish_reason_to_gemini, function_call_part, has_finish_reason, mark_once, usage_from_gemini,
    usage_to_gemini,
};

const TOOL_ORDER_KEY: &str = "gemini.pendingToolOrder";
const USAGE_EMITTED_KEY: &str = "gemini.responseDoneUsageEmitted";

pub(super) fn decode_stream_chunk(
    raw: Value,
    state: &mut DecodeState,
) -> Result<Vec<UniversalEvent>> {
    let mut events = Vec::new();
    if mark_once(state, "gemini.response_start") {
        events.push(UniversalEvent::ResponseStart {
            id: raw
                .get("responseId")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            model: raw
                .get("modelVersion")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            extensions: crate::translator::common::empty_extensions(),
        });
    }
    decode_candidates(&raw, &mut events, false)?;
    if has_finish_reason(&raw) {
        events.push(UniversalEvent::ResponseDone {
            usage: usage_from_gemini(raw.get("usageMetadata")),
            extensions: crate::translator::common::empty_extensions(),
        });
    }
    Ok(events)
}

pub(super) fn encode_stream_events(
    events: &[UniversalEvent],
    state: &mut EncodeState,
) -> Result<Vec<WireEvent>> {
    let mut out = Vec::new();
    for event in events {
        match event {
            UniversalEvent::TextDelta { text, .. } if !text.is_empty() => {
                out.push(WireEvent {
                    event: None,
                    data: gemini_chunk(vec![json!({ "text": text })], None, None),
                });
            }
            UniversalEvent::ReasoningDelta { text, .. } if !text.is_empty() => {
                out.push(WireEvent {
                    event: None,
                    data: gemini_chunk(vec![json!({ "text": text, "thought": true })], None, None),
                });
            }
            UniversalEvent::ContentStart {
                block:
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments: _,
                        ..
                    },
                ..
            } => remember_tool_name(state, id, name),
            UniversalEvent::ToolCallDelta {
                id,
                name,
                arguments_delta,
            } => remember_tool_delta(state, id, name.as_deref(), arguments_delta),
            UniversalEvent::ContentDone {
                final_block:
                    Some(ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                        ..
                    }),
                ..
            } => remember_final_tool_args(state, id, name, arguments),
            UniversalEvent::MessageDone {
                finish_reason,
                usage,
                ..
            } => {
                let parts = drain_tool_calls(state);
                let finish = finish_reason.map(finish_reason_to_gemini);
                if usage.is_some() {
                    mark_usage_emitted(state);
                }
                out.push(WireEvent {
                    event: None,
                    data: gemini_chunk(parts, finish, usage.as_ref()),
                });
            }
            UniversalEvent::ResponseDone { usage, .. } => {
                let parts = drain_tool_calls(state);
                if !parts.is_empty() {
                    out.push(WireEvent {
                        event: None,
                        data: gemini_chunk(parts, None, None),
                    });
                }
                if usage.is_some() && !usage_emitted(state) {
                    mark_usage_emitted(state);
                    out.push(WireEvent {
                        event: None,
                        data: gemini_chunk(Vec::new(), None, usage.as_ref()),
                    });
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn gemini_chunk(parts: Vec<Value>, finish_reason: Option<&str>, usage: Option<&Usage>) -> Value {
    let mut candidate = serde_json::Map::new();
    if !parts.is_empty() {
        candidate.insert(
            "content".to_string(),
            json!({
                "role": "model",
                "parts": parts,
            }),
        );
    }
    if let Some(finish_reason) = finish_reason {
        candidate.insert(
            "finishReason".to_string(),
            Value::String(finish_reason.to_string()),
        );
    }
    let mut out = serde_json::Map::new();
    out.insert(
        "candidates".to_string(),
        Value::Array(vec![Value::Object(candidate)]),
    );
    if let Some(usage) = usage {
        out.insert("usageMetadata".to_string(), usage_to_gemini(usage));
    }
    Value::Object(out)
}

fn remember_tool_delta(
    state: &mut EncodeState,
    id: &str,
    name: Option<&str>,
    arguments_delta: &str,
) {
    let id = normalized_tool_id(id);
    remember_tool_name(state, &id, name.unwrap_or_default());
    append_tool_args(state, &id, arguments_delta);
}

fn remember_tool_name(state: &mut EncodeState, id: &str, name: &str) {
    let id = normalized_tool_id(id);
    remember_tool_order(state, &id);
    if !name.is_empty() {
        state
            .extensions
            .insert(tool_name_key(&id), Value::String(name.to_string()));
    }
}

fn remember_final_tool_args(state: &mut EncodeState, id: &str, name: &str, arguments: &Value) {
    let id = normalized_tool_id(id);
    remember_tool_name(state, &id, name);
    if state.extensions.contains_key(&tool_args_key(&id)) {
        return;
    }
    append_tool_args(
        state,
        &id,
        &serde_json::to_string(arguments).unwrap_or_default(),
    );
}

fn remember_tool_order(state: &mut EncodeState, id: &str) {
    let mut order = state
        .extensions
        .remove(TOOL_ORDER_KEY)
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    if !order.iter().any(|value| value.as_str() == Some(id)) {
        order.push(Value::String(id.to_string()));
    }
    state
        .extensions
        .insert(TOOL_ORDER_KEY.to_string(), Value::Array(order));
}

fn append_tool_args(state: &mut EncodeState, id: &str, arguments_delta: &str) {
    if arguments_delta.is_empty() {
        return;
    }
    let key = tool_args_key(id);
    let mut current = state
        .extensions
        .remove(&key)
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .unwrap_or_default();
    current.push_str(arguments_delta);
    state.extensions.insert(key, Value::String(current));
}

fn drain_tool_calls(state: &mut EncodeState) -> Vec<Value> {
    let order = state
        .extensions
        .remove(TOOL_ORDER_KEY)
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    let mut parts = Vec::new();
    for id in order.iter().filter_map(Value::as_str) {
        let name = state
            .extensions
            .remove(&tool_name_key(id))
            .and_then(|value| value.as_str().map(ToOwned::to_owned))
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| id.to_string());
        let args = state
            .extensions
            .remove(&tool_args_key(id))
            .and_then(|value| value.as_str().map(ToOwned::to_owned))
            .unwrap_or_default();
        let args = parse_tool_args(&args);
        parts.push(function_call_part(Some(id), &name, args));
    }
    parts
}

fn parse_tool_args(value: &str) -> Value {
    if value.trim().is_empty() {
        return json!({});
    }
    serde_json::from_str(value).unwrap_or_else(|_| json!({ "value": value }))
}

fn normalized_tool_id(id: &str) -> String {
    if id.is_empty() {
        "function_call".to_string()
    } else {
        id.to_string()
    }
}

fn tool_name_key(id: &str) -> String {
    format!("gemini.toolName:{id}")
}

fn tool_args_key(id: &str) -> String {
    format!("gemini.toolArgs:{id}")
}

fn mark_usage_emitted(state: &mut EncodeState) {
    state
        .extensions
        .insert(USAGE_EMITTED_KEY.to_string(), Value::Bool(true));
}

fn usage_emitted(state: &EncodeState) -> bool {
    state
        .extensions
        .get(USAGE_EMITTED_KEY)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}
