use serde_json::{json, Value};

use crate::translator::common;
use crate::{EncodeState, WireEvent};

pub(in crate::translator::openai_responses::stream::encode) fn ensure_response_tool_started(
    state: &mut EncodeState,
    wire_events: &mut Vec<WireEvent>,
    id: &str,
    name: Option<&str>,
    content_index: Option<usize>,
) {
    remember_response_tool_id(state, id);
    if let Some(name) = name.filter(|name| !name.is_empty()) {
        state
            .extensions
            .insert(response_tool_name_key(id), Value::String(name.to_string()));
    }
    if let Some(content_index) = content_index {
        state.extensions.insert(
            format!("responseToolIdForContentIndex:{content_index}"),
            Value::String(id.to_string()),
        );
    }
    let previous = state
        .extensions
        .insert(response_tool_started_key(id), Value::Bool(true));
    if matches!(previous, Some(Value::Bool(true))) {
        return;
    }
    let output_index = assign_response_tool_output_index(state, id);
    wire_events.push(common::wire_event(json!({
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "id": response_tool_item_id(state, id),
            "type": "function_call",
            "status": "in_progress",
            "call_id": id,
            "name": response_tool_name(state, id),
            "arguments": ""
        }
    })));
}

pub(in crate::translator::openai_responses::stream::encode) fn append_response_tool_arguments(
    state: &mut EncodeState,
    id: &str,
    arguments_delta: &str,
) {
    if arguments_delta.is_empty() {
        return;
    }
    let mut current = state
        .extensions
        .get(&response_tool_arguments_key(id))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    current.push_str(arguments_delta);
    state
        .extensions
        .insert(response_tool_arguments_key(id), Value::String(current));
}

pub(in crate::translator::openai_responses::stream::encode) fn finish_all_response_tools(
    state: &mut EncodeState,
    wire_events: &mut Vec<WireEvent>,
) {
    for id in response_tool_ids(state) {
        finish_response_tool(state, wire_events, &id);
    }
}

pub(in crate::translator::openai_responses::stream::encode) fn finish_response_tool(
    state: &mut EncodeState,
    wire_events: &mut Vec<WireEvent>,
    id: &str,
) {
    if !matches!(
        state.extensions.get(&response_tool_started_key(id)),
        Some(Value::Bool(true))
    ) {
        return;
    }
    let previous = state
        .extensions
        .insert(response_tool_done_key(id), Value::Bool(true));
    if matches!(previous, Some(Value::Bool(true))) {
        return;
    }
    let output_index = response_tool_output_index(state, id);
    let item_id = response_tool_item_id(state, id);
    let arguments = response_tool_arguments(state, id);
    wire_events.push(common::wire_event(json!({
        "type": "response.function_call_arguments.done",
        "output_index": output_index,
        "item_id": item_id,
        "arguments": arguments
    })));
    wire_events.push(common::wire_event(json!({
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": response_tool_output_item(state, id)
    })));
}
pub(in crate::translator::openai_responses::stream::encode) fn response_tool_output_item(
    state: &EncodeState,
    id: &str,
) -> Value {
    json!({
        "id": response_tool_item_id(state, id),
        "type": "function_call",
        "status": "completed",
        "call_id": id,
        "name": response_tool_name(state, id),
        "arguments": response_tool_arguments(state, id)
    })
}
pub(in crate::translator::openai_responses::stream::encode) fn assign_response_tool_output_index(
    state: &mut EncodeState,
    id: &str,
) -> usize {
    let key = response_tool_output_index_key(id);
    if let Some(index) = state.extensions.get(&key).and_then(Value::as_u64) {
        return index as usize;
    }
    let index = common::encode_state_index(state);
    state
        .extensions
        .insert(key, Value::Number((index as u64).into()));
    index
}

pub(in crate::translator::openai_responses::stream::encode) fn response_tool_output_index(
    state: &EncodeState,
    id: &str,
) -> usize {
    state
        .extensions
        .get(&response_tool_output_index_key(id))
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize
}

pub(in crate::translator::openai_responses::stream::encode) fn remember_response_tool_id(
    state: &mut EncodeState,
    id: &str,
) {
    let mut ids = response_tool_ids(state);
    if ids.iter().any(|existing| existing == id) {
        return;
    }
    ids.push(id.to_string());
    state.extensions.insert(
        "responseToolIds".to_string(),
        Value::Array(ids.into_iter().map(Value::String).collect()),
    );
}

pub(in crate::translator::openai_responses::stream::encode) fn response_tool_ids(
    state: &EncodeState,
) -> Vec<String> {
    state
        .extensions
        .get("responseToolIds")
        .and_then(Value::as_array)
        .map(|ids| {
            ids.iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

pub(in crate::translator::openai_responses::stream::encode) fn response_tool_id_for_content_index(
    state: &EncodeState,
    index: usize,
) -> Option<String> {
    state
        .extensions
        .get(&format!("responseToolIdForContentIndex:{index}"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

pub(in crate::translator::openai_responses::stream::encode) fn response_tool_name(
    state: &EncodeState,
    id: &str,
) -> String {
    state
        .extensions
        .get(&response_tool_name_key(id))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .unwrap_or("function")
        .to_string()
}

pub(in crate::translator::openai_responses::stream::encode) fn response_tool_arguments(
    state: &EncodeState,
    id: &str,
) -> String {
    state
        .extensions
        .get(&response_tool_arguments_key(id))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

pub(in crate::translator::openai_responses::stream::encode) fn response_tool_item_id(
    state: &EncodeState,
    id: &str,
) -> String {
    state
        .extensions
        .get(&response_tool_item_id_key(id))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| {
            if id.starts_with("fc_") {
                id.to_string()
            } else {
                format!("fc_{id}")
            }
        })
}

pub(in crate::translator::openai_responses::stream::encode) fn response_tool_output_index_key(
    id: &str,
) -> String {
    format!("responseToolOutputIndex:{id}")
}

pub(in crate::translator::openai_responses::stream::encode) fn response_tool_started_key(
    id: &str,
) -> String {
    format!("responseToolStarted:{id}")
}

pub(in crate::translator::openai_responses::stream::encode) fn response_tool_done_key(
    id: &str,
) -> String {
    format!("responseToolDone:{id}")
}

pub(in crate::translator::openai_responses::stream::encode) fn response_tool_name_key(
    id: &str,
) -> String {
    format!("responseToolName:{id}")
}

pub(in crate::translator::openai_responses::stream::encode) fn response_tool_arguments_key(
    id: &str,
) -> String {
    format!("responseToolArguments:{id}")
}

pub(in crate::translator::openai_responses::stream::encode) fn response_tool_item_id_key(
    id: &str,
) -> String {
    format!("responseToolItemId:{id}")
}
