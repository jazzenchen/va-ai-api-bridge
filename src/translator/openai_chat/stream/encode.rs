use serde_json::{json, Value};

use crate::translator::{common, openai};
use crate::{EncodeState, Result, UniversalEvent, WireEvent};

pub(super) fn encode(events: &[UniversalEvent], state: &mut EncodeState) -> Result<Vec<WireEvent>> {
    Ok(events
        .iter()
        .map(|event| match event {
            UniversalEvent::ResponseStart { id, model, .. } => common::wire_event(json!({
                "id": id,
                "model": model,
                "choices": []
            })),
            UniversalEvent::MessageStart { role, .. } => common::wire_event(json!({
                "choices": [{
                    "index": 0,
                    "delta": { "role": openai::role_to_openai(*role) }
                }]
            })),
            UniversalEvent::TextDelta { index, text } => common::wire_event(json!({
                "choices": [{
                    "index": index,
                    "delta": { "content": text }
                }]
            })),
            UniversalEvent::ToolCallDelta {
                id,
                name,
                arguments_delta,
            } => {
                let index = common::encode_state_index(state);
                common::wire_event(json!({
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments_delta
                                }
                            }]
                        }
                    }]
                }))
            }
            UniversalEvent::MessageDone { finish_reason, .. } => common::wire_event(json!({
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_to_openai(*finish_reason)
                }]
            })),
            UniversalEvent::ResponseDone { usage, .. } => common::wire_event(json!({
                "choices": [],
                "usage": usage_to_openai_value(usage.as_ref())
            })),
            UniversalEvent::Error { message, raw } => common::wire_event(json!({
                "error": {
                    "message": message,
                    "raw": raw
                }
            })),
            UniversalEvent::Unknown { raw, .. } => common::wire_event(raw.clone()),
            UniversalEvent::ContentStart { .. }
            | UniversalEvent::ReasoningDelta { .. }
            | UniversalEvent::ContentDone { .. } => common::wire_event(json!({
                "choices": []
            })),
        })
        .collect())
}

fn finish_to_openai(reason: Option<crate::FinishReason>) -> Value {
    match reason {
        Some(crate::FinishReason::Stop) => json!("stop"),
        Some(crate::FinishReason::Length) => json!("length"),
        Some(crate::FinishReason::ToolCall) => json!("tool_calls"),
        Some(crate::FinishReason::ContentFilter) => json!("content_filter"),
        Some(crate::FinishReason::Error) => json!("error"),
        Some(crate::FinishReason::Unknown) | None => Value::Null,
    }
}

fn usage_to_openai_value(usage: Option<&crate::Usage>) -> Value {
    match usage {
        Some(usage) => json!({
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens
        }),
        None => Value::Null,
    }
}
