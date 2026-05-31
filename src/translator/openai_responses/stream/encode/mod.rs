mod response;

use serde_json::json;

use crate::translator::common;
use crate::{EncodeState, Result, UniversalEvent, WireEvent};

use self::response::*;

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

#[cfg(test)]
mod tests;
