use serde_json::{json, Value};

use crate::translator::common;
use crate::{EncodeState, WireEvent};

pub(super) fn append_response_text(state: &mut EncodeState, text: &str) {
    let mut current = state
        .extensions
        .get("responseText")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    current.push_str(text);
    state
        .extensions
        .insert("responseText".to_string(), Value::String(current));
}

pub(super) fn append_response_reasoning_text(state: &mut EncodeState, text: &str) {
    let mut current = state
        .extensions
        .get("responseReasoningText")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    current.push_str(text);
    state
        .extensions
        .insert("responseReasoningText".to_string(), Value::String(current));
}

pub(super) fn ensure_response_reasoning_started(
    state: &mut EncodeState,
    wire_events: &mut Vec<WireEvent>,
    content_index: usize,
) {
    state.extensions.insert(
        "responseReasoningContentIndex".to_string(),
        Value::Number((content_index as u64).into()),
    );
    let previous = state
        .extensions
        .insert("responseReasoningStarted".to_string(), Value::Bool(true));
    if matches!(previous, Some(Value::Bool(true))) {
        return;
    }
    let output_index = assign_response_reasoning_output_index(state);
    wire_events.push(common::wire_event(json!({
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "id": response_reasoning_id(state),
            "type": "reasoning",
            "status": "in_progress",
            "summary": []
        }
    })));
}

pub(super) fn finish_response_reasoning(state: &mut EncodeState, wire_events: &mut Vec<WireEvent>) {
    if !matches!(
        state.extensions.get("responseReasoningStarted"),
        Some(Value::Bool(true))
    ) {
        return;
    }
    let previous = state
        .extensions
        .insert("responseReasoningDone".to_string(), Value::Bool(true));
    if matches!(previous, Some(Value::Bool(true))) {
        return;
    }
    wire_events.push(common::wire_event(json!({
        "type": "response.output_item.done",
        "output_index": response_reasoning_output_index(state),
        "item": response_reasoning_output_item(state)
    })));
}

pub(super) fn ensure_response_output_started(
    state: &mut EncodeState,
    wire_events: &mut Vec<WireEvent>,
    content_index: usize,
) {
    state.extensions.insert(
        "responseContentIndex".to_string(),
        Value::Number((content_index as u64).into()),
    );
    let previous = state
        .extensions
        .insert("responseOutputStarted".to_string(), Value::Bool(true));
    if matches!(previous, Some(Value::Bool(true))) {
        return;
    }
    let output_index = assign_response_message_output_index(state);
    let item_id = response_message_id(state);
    wire_events.push(common::wire_event(json!({
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {
            "id": item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": []
        }
    })));
    wire_events.push(common::wire_event(json!({
        "type": "response.content_part.added",
        "output_index": output_index,
        "content_index": content_index,
        "item_id": item_id,
        "part": {
            "type": "output_text",
            "text": "",
            "annotations": []
        }
    })));
}

pub(super) fn finish_response_output(state: &mut EncodeState, wire_events: &mut Vec<WireEvent>) {
    if !matches!(
        state.extensions.get("responseOutputStarted"),
        Some(Value::Bool(true))
    ) {
        return;
    }
    let previous = state
        .extensions
        .insert("responseOutputDone".to_string(), Value::Bool(true));
    if matches!(previous, Some(Value::Bool(true))) {
        return;
    }
    let item_id = response_message_id(state);
    let content_index = state
        .extensions
        .get("responseContentIndex")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let output_index = response_message_output_index(state);
    let text = response_text(state);
    let part = json!({
        "type": "output_text",
        "text": text,
        "annotations": []
    });
    wire_events.push(common::wire_event(json!({
        "type": "response.output_text.done",
        "output_index": output_index,
        "content_index": content_index,
        "item_id": item_id,
        "text": text
    })));
    wire_events.push(common::wire_event(json!({
        "type": "response.content_part.done",
        "output_index": output_index,
        "content_index": content_index,
        "item_id": item_id,
        "part": part
    })));
    wire_events.push(common::wire_event(json!({
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": response_output_item(state)
    })));
}

pub(super) fn assign_response_message_output_index(state: &mut EncodeState) -> usize {
    if let Some(index) = state
        .extensions
        .get("responseOutputIndex")
        .and_then(Value::as_u64)
    {
        return index as usize;
    }
    let index = common::encode_state_index(state);
    state.extensions.insert(
        "responseOutputIndex".to_string(),
        Value::Number((index as u64).into()),
    );
    index
}

pub(super) fn response_message_output_index(state: &EncodeState) -> usize {
    state
        .extensions
        .get("responseOutputIndex")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize
}

pub(super) fn assign_response_reasoning_output_index(state: &mut EncodeState) -> usize {
    if let Some(index) = state
        .extensions
        .get("responseReasoningOutputIndex")
        .and_then(Value::as_u64)
    {
        return index as usize;
    }
    let index = common::encode_state_index(state);
    state.extensions.insert(
        "responseReasoningOutputIndex".to_string(),
        Value::Number((index as u64).into()),
    );
    index
}

pub(super) fn response_reasoning_output_index(state: &EncodeState) -> usize {
    state
        .extensions
        .get("responseReasoningOutputIndex")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize
}

pub(super) fn ensure_response_tool_started(
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

pub(super) fn append_response_tool_arguments(
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

pub(super) fn finish_all_response_tools(state: &mut EncodeState, wire_events: &mut Vec<WireEvent>) {
    for id in response_tool_ids(state) {
        finish_response_tool(state, wire_events, &id);
    }
}

pub(super) fn finish_response_tool(
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

pub(super) fn response_message_id(state: &EncodeState) -> String {
    let id = state
        .extensions
        .get("responseId")
        .and_then(Value::as_str)
        .unwrap_or("resp_va_bridge");
    if id.starts_with("msg_") {
        id.to_string()
    } else {
        format!("msg_{id}")
    }
}

pub(super) fn response_text(state: &EncodeState) -> String {
    state
        .extensions
        .get("responseText")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

pub(super) fn response_reasoning_text(state: &EncodeState) -> String {
    state
        .extensions
        .get("responseReasoningText")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

pub(super) fn response_reasoning_id(state: &EncodeState) -> String {
    let id = state
        .extensions
        .get("responseId")
        .and_then(Value::as_str)
        .unwrap_or("resp_va_bridge");
    if id.starts_with("rs_") {
        id.to_string()
    } else {
        format!("rs_{id}")
    }
}

pub(super) fn response_output_item(state: &EncodeState) -> Value {
    json!({
        "id": response_message_id(state),
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": response_text(state),
            "annotations": []
        }]
    })
}

pub(super) fn response_reasoning_output_item(state: &EncodeState) -> Value {
    json!({
        "id": response_reasoning_id(state),
        "type": "reasoning",
        "status": "completed",
        "summary": [],
        "content": [{
            "type": "reasoning_text",
            "text": response_reasoning_text(state)
        }]
    })
}

pub(super) fn response_tool_output_item(state: &EncodeState, id: &str) -> Value {
    json!({
        "id": response_tool_item_id(state, id),
        "type": "function_call",
        "status": "completed",
        "call_id": id,
        "name": response_tool_name(state, id),
        "arguments": response_tool_arguments(state, id)
    })
}

pub(super) fn response_shell(
    state: &EncodeState,
    status: &str,
    usage: Option<&crate::Usage>,
) -> Value {
    let id = state
        .extensions
        .get("responseId")
        .and_then(Value::as_str)
        .unwrap_or("resp_va_bridge");
    let model = state
        .extensions
        .get("responseModel")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let output = if status == "completed" {
        Value::Array(response_output_items(state))
    } else {
        json!([])
    };
    json!({
        "id": id,
        "object": "response",
        "created_at": 0,
        "status": status,
        "error": Value::Null,
        "incomplete_details": Value::Null,
        "instructions": Value::Null,
        "max_output_tokens": Value::Null,
        "model": model,
        "output": output,
        "parallel_tool_calls": true,
        "previous_response_id": Value::Null,
        "reasoning": Value::Null,
        "store": false,
        "temperature": Value::Null,
        "text": Value::Null,
        "tool_choice": "auto",
        "tools": [],
        "top_p": Value::Null,
        "truncation": "disabled",
        "usage": usage_to_openai_value(usage),
        "user": Value::Null,
        "metadata": {}
    })
}

pub(super) fn response_output_items(state: &EncodeState) -> Vec<Value> {
    let mut output = Vec::new();
    if matches!(
        state.extensions.get("responseReasoningDone"),
        Some(Value::Bool(true))
    ) {
        output.push((
            response_reasoning_output_index(state),
            response_reasoning_output_item(state),
        ));
    }
    if matches!(
        state.extensions.get("responseOutputDone"),
        Some(Value::Bool(true))
    ) {
        output.push((
            response_message_output_index(state),
            response_output_item(state),
        ));
    }
    for id in response_tool_ids(state) {
        if matches!(
            state.extensions.get(&response_tool_done_key(&id)),
            Some(Value::Bool(true))
        ) {
            output.push((
                response_tool_output_index(state, &id),
                response_tool_output_item(state, &id),
            ));
        }
    }
    output.sort_by_key(|(output_index, _)| *output_index);
    output.into_iter().map(|(_, item)| item).collect()
}

pub(super) fn assign_response_tool_output_index(state: &mut EncodeState, id: &str) -> usize {
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

pub(super) fn response_tool_output_index(state: &EncodeState, id: &str) -> usize {
    state
        .extensions
        .get(&response_tool_output_index_key(id))
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize
}

pub(super) fn remember_response_tool_id(state: &mut EncodeState, id: &str) {
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

pub(super) fn response_tool_ids(state: &EncodeState) -> Vec<String> {
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

pub(super) fn response_tool_id_for_content_index(
    state: &EncodeState,
    index: usize,
) -> Option<String> {
    state
        .extensions
        .get(&format!("responseToolIdForContentIndex:{index}"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

pub(super) fn response_tool_name(state: &EncodeState, id: &str) -> String {
    state
        .extensions
        .get(&response_tool_name_key(id))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .unwrap_or("function")
        .to_string()
}

pub(super) fn response_tool_arguments(state: &EncodeState, id: &str) -> String {
    state
        .extensions
        .get(&response_tool_arguments_key(id))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

pub(super) fn response_tool_item_id(state: &EncodeState, id: &str) -> String {
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

pub(super) fn response_tool_output_index_key(id: &str) -> String {
    format!("responseToolOutputIndex:{id}")
}

pub(super) fn response_tool_started_key(id: &str) -> String {
    format!("responseToolStarted:{id}")
}

pub(super) fn response_tool_done_key(id: &str) -> String {
    format!("responseToolDone:{id}")
}

pub(super) fn response_tool_name_key(id: &str) -> String {
    format!("responseToolName:{id}")
}

pub(super) fn response_tool_arguments_key(id: &str) -> String {
    format!("responseToolArguments:{id}")
}

pub(super) fn response_tool_item_id_key(id: &str) -> String {
    format!("responseToolItemId:{id}")
}

pub(super) fn usage_to_openai_value(usage: Option<&crate::Usage>) -> Value {
    match usage {
        Some(usage) => json!({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens
        }),
        None => Value::Null,
    }
}
