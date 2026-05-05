use serde_json::{json, Value};

use crate::translator::common;
use crate::{EncodeState, Result, UniversalEvent, WireEvent};

pub(super) fn encode(events: &[UniversalEvent], state: &mut EncodeState) -> Result<Vec<WireEvent>> {
    let mut wire_events = Vec::new();
    for event in events {
        match event {
            UniversalEvent::ResponseStart { id, model, .. } => {
                if let Some(id) = id {
                    state.extensions.insert("responseId".to_string(), json!(id));
                }
                if let Some(model) = model {
                    state
                        .extensions
                        .insert("responseModel".to_string(), json!(model));
                }
                wire_events.push(common::wire_event(json!({
                    "type": "response.created",
                    "response": response_shell(state, "in_progress", None)
                })));
                wire_events.push(common::wire_event(json!({
                    "type": "response.in_progress",
                    "response": response_shell(state, "in_progress", None)
                })));
            }
            UniversalEvent::TextDelta { index, text } => {
                ensure_response_output_started(state, &mut wire_events, *index);
                let output_index = response_message_output_index(state);
                append_response_text(state, text);
                wire_events.push(common::wire_event(json!({
                    "type": "response.output_text.delta",
                    "output_index": output_index,
                    "content_index": index,
                    "item_id": response_message_id(state),
                    "delta": text
                })));
            }
            UniversalEvent::ReasoningDelta { index, text } => {
                if text.is_empty() {
                    continue;
                }
                ensure_response_reasoning_started(state, &mut wire_events, *index);
                let output_index = response_reasoning_output_index(state);
                append_response_reasoning_text(state, text);
                wire_events.push(common::wire_event(json!({
                    "type": "response.reasoning_text.delta",
                    "output_index": output_index,
                    "content_index": index,
                    "item_id": response_reasoning_id(state),
                    "delta": text
                })))
            }
            UniversalEvent::ToolCallDelta {
                id,
                name,
                arguments_delta,
            } => {
                ensure_response_tool_started(state, &mut wire_events, id, name.as_deref(), None);
                append_response_tool_arguments(state, id, arguments_delta);
                if !arguments_delta.is_empty() {
                    wire_events.push(common::wire_event(json!({
                        "type": "response.function_call_arguments.delta",
                        "output_index": response_tool_output_index(state, id),
                        "item_id": response_tool_item_id(state, id),
                        "delta": arguments_delta
                    })));
                }
            }
            UniversalEvent::ResponseDone { usage, .. } => {
                if let Some(usage) = usage {
                    state
                        .extensions
                        .insert("responseUsage".to_string(), json!(usage));
                }
                finish_response_reasoning(state, &mut wire_events);
                finish_response_output(state, &mut wire_events);
                finish_all_response_tools(state, &mut wire_events);
                wire_events.push(common::wire_event(json!({
                    "type": "response.completed",
                    "response": response_shell(state, "completed", usage.as_ref())
                })));
            }
            UniversalEvent::Error { message, raw } => wire_events.push(common::wire_event(json!({
                "type": "response.failed",
                "error": {
                    "message": message,
                    "raw": raw
                }
            }))),
            UniversalEvent::Unknown { raw, .. } => {
                wire_events.push(common::wire_event(raw.clone()));
            }
            UniversalEvent::ContentStart { index, block } => match block {
                crate::ContentBlock::Reasoning { .. } => {
                    ensure_response_reasoning_started(state, &mut wire_events, *index);
                }
                crate::ContentBlock::ToolCall { id, name, .. } => {
                    ensure_response_tool_started(
                        state,
                        &mut wire_events,
                        id,
                        Some(name),
                        Some(*index),
                    );
                }
                _ => {}
            },
            UniversalEvent::ContentDone { index, final_block } => {
                if let Some(crate::ContentBlock::ToolCall { id, .. }) = final_block {
                    finish_response_tool(state, &mut wire_events, id);
                } else if let Some(id) = response_tool_id_for_content_index(state, *index) {
                    finish_response_tool(state, &mut wire_events, &id);
                }
            }
            UniversalEvent::MessageDone { finish_reason, .. } => {
                if matches!(finish_reason, Some(crate::FinishReason::ToolCall)) {
                    finish_all_response_tools(state, &mut wire_events);
                }
            }
            UniversalEvent::MessageStart { .. } => {}
        }
    }
    Ok(wire_events)
}

fn append_response_text(state: &mut EncodeState, text: &str) {
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

fn append_response_reasoning_text(state: &mut EncodeState, text: &str) {
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

fn ensure_response_reasoning_started(
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

fn finish_response_reasoning(state: &mut EncodeState, wire_events: &mut Vec<WireEvent>) {
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

fn ensure_response_output_started(
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

fn finish_response_output(state: &mut EncodeState, wire_events: &mut Vec<WireEvent>) {
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

fn assign_response_message_output_index(state: &mut EncodeState) -> usize {
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

fn response_message_output_index(state: &EncodeState) -> usize {
    state
        .extensions
        .get("responseOutputIndex")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize
}

fn assign_response_reasoning_output_index(state: &mut EncodeState) -> usize {
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

fn response_reasoning_output_index(state: &EncodeState) -> usize {
    state
        .extensions
        .get("responseReasoningOutputIndex")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize
}

fn ensure_response_tool_started(
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

fn append_response_tool_arguments(state: &mut EncodeState, id: &str, arguments_delta: &str) {
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

fn finish_all_response_tools(state: &mut EncodeState, wire_events: &mut Vec<WireEvent>) {
    for id in response_tool_ids(state) {
        finish_response_tool(state, wire_events, &id);
    }
}

fn finish_response_tool(state: &mut EncodeState, wire_events: &mut Vec<WireEvent>, id: &str) {
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

fn response_message_id(state: &EncodeState) -> String {
    let id = state
        .extensions
        .get("responseId")
        .and_then(Value::as_str)
        .unwrap_or("resp_va_proxy");
    if id.starts_with("msg_") {
        id.to_string()
    } else {
        format!("msg_{id}")
    }
}

fn response_text(state: &EncodeState) -> String {
    state
        .extensions
        .get("responseText")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn response_reasoning_text(state: &EncodeState) -> String {
    state
        .extensions
        .get("responseReasoningText")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn response_reasoning_id(state: &EncodeState) -> String {
    let id = state
        .extensions
        .get("responseId")
        .and_then(Value::as_str)
        .unwrap_or("resp_va_proxy");
    if id.starts_with("rs_") {
        id.to_string()
    } else {
        format!("rs_{id}")
    }
}

fn response_output_item(state: &EncodeState) -> Value {
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

fn response_reasoning_output_item(state: &EncodeState) -> Value {
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

fn response_tool_output_item(state: &EncodeState, id: &str) -> Value {
    json!({
        "id": response_tool_item_id(state, id),
        "type": "function_call",
        "status": "completed",
        "call_id": id,
        "name": response_tool_name(state, id),
        "arguments": response_tool_arguments(state, id)
    })
}

fn response_shell(state: &EncodeState, status: &str, usage: Option<&crate::Usage>) -> Value {
    let id = state
        .extensions
        .get("responseId")
        .and_then(Value::as_str)
        .unwrap_or("resp_va_proxy");
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

fn response_output_items(state: &EncodeState) -> Vec<Value> {
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

fn assign_response_tool_output_index(state: &mut EncodeState, id: &str) -> usize {
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

fn response_tool_output_index(state: &EncodeState, id: &str) -> usize {
    state
        .extensions
        .get(&response_tool_output_index_key(id))
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize
}

fn remember_response_tool_id(state: &mut EncodeState, id: &str) {
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

fn response_tool_ids(state: &EncodeState) -> Vec<String> {
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

fn response_tool_id_for_content_index(state: &EncodeState, index: usize) -> Option<String> {
    state
        .extensions
        .get(&format!("responseToolIdForContentIndex:{index}"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn response_tool_name(state: &EncodeState, id: &str) -> String {
    state
        .extensions
        .get(&response_tool_name_key(id))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .unwrap_or("function")
        .to_string()
}

fn response_tool_arguments(state: &EncodeState, id: &str) -> String {
    state
        .extensions
        .get(&response_tool_arguments_key(id))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn response_tool_item_id(state: &EncodeState, id: &str) -> String {
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

fn response_tool_output_index_key(id: &str) -> String {
    format!("responseToolOutputIndex:{id}")
}

fn response_tool_started_key(id: &str) -> String {
    format!("responseToolStarted:{id}")
}

fn response_tool_done_key(id: &str) -> String {
    format!("responseToolDone:{id}")
}

fn response_tool_name_key(id: &str) -> String {
    format!("responseToolName:{id}")
}

fn response_tool_arguments_key(id: &str) -> String {
    format!("responseToolArguments:{id}")
}

fn response_tool_item_id_key(id: &str) -> String {
    format!("responseToolItemId:{id}")
}

fn usage_to_openai_value(usage: Option<&crate::Usage>) -> Value {
    match usage {
        Some(usage) => json!({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens
        }),
        None => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use crate::translator::common;
    use crate::{EncodeState, UniversalEvent};

    use super::encode;

    #[test]
    fn opens_reasoning_item_before_reasoning_delta() {
        let mut state = EncodeState::default();
        let events = encode(
            &[
                UniversalEvent::ResponseStart {
                    id: Some("resp_1".to_string()),
                    model: Some("deepseek-v4-pro".to_string()),
                    extensions: common::empty_extensions(),
                },
                UniversalEvent::ReasoningDelta {
                    index: 0,
                    text: "Need to think.".to_string(),
                },
                UniversalEvent::TextDelta {
                    index: 0,
                    text: "OK".to_string(),
                },
                UniversalEvent::ResponseDone {
                    usage: None,
                    extensions: common::empty_extensions(),
                },
            ],
            &mut state,
        )
        .expect("events encode");

        let reasoning_added_index = events
            .iter()
            .position(|event| {
                event.data["type"] == "response.output_item.added"
                    && event.data["item"]["type"] == "reasoning"
            })
            .expect("reasoning item added");
        let reasoning_delta_index = events
            .iter()
            .position(|event| event.data["type"] == "response.reasoning_text.delta")
            .expect("reasoning delta");

        assert!(reasoning_added_index < reasoning_delta_index);
        assert_eq!(
            events[reasoning_delta_index].data["item_id"],
            events[reasoning_added_index].data["item"]["id"]
        );

        let completed = events
            .iter()
            .find(|event| event.data["type"] == "response.completed")
            .expect("response completed");
        assert_eq!(completed.data["response"]["output"][0]["type"], "reasoning");
        assert_eq!(
            completed.data["response"]["output"][0]["content"][0]["text"],
            "Need to think."
        );
        assert_eq!(completed.data["response"]["output"][1]["type"], "message");
        assert_eq!(
            completed.data["response"]["output"][1]["content"][0]["text"],
            "OK"
        );
    }
}
