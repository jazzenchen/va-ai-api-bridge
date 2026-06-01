use serde_json::{json, Map, Value};

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
                    "index": choice_index(state, *index),
                    "delta": { "content": text }
                }]
            })),
            UniversalEvent::ToolCallDelta {
                id,
                name,
                arguments_delta,
            } => {
                let index = tool_call_index(state, id);
                let mut function = Map::new();
                if let Some(name) = name {
                    function.insert("name".to_string(), Value::String(name.clone()));
                }
                function.insert(
                    "arguments".to_string(),
                    Value::String(arguments_delta.clone()),
                );
                common::wire_event(json!({
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "id": id,
                                "type": "function",
                                "function": Value::Object(function)
                            }]
                        }
                    }]
                }))
            }
            UniversalEvent::ReasoningDelta { index, text } => common::wire_event(json!({
                "choices": [{
                    "index": choice_index(state, *index),
                    "delta": { "reasoning_content": text }
                }]
            })),
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
            UniversalEvent::ContentStart { .. } | UniversalEvent::ContentDone { .. } => {
                common::wire_event(json!({
                "choices": []
                }))
            }
        })
        .collect())
}

fn choice_index(_state: &EncodeState, _content_index: usize) -> usize {
    0
}

fn tool_call_index(state: &mut EncodeState, id: &str) -> usize {
    let key = format!("openaiChatToolCallIndex:{id}");
    if let Some(index) = state.extensions.get(&key).and_then(Value::as_u64) {
        return index as usize;
    }
    let index = common::encode_state_index(state);
    state
        .extensions
        .insert(key, Value::Number((index as u64).into()));
    index
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

#[cfg(test)]
mod tests {
    use crate::{EncodeState, UniversalEvent};

    use super::encode;

    #[test]
    fn uses_stable_tool_call_index_for_split_deltas() {
        let mut state = EncodeState::default();
        let first = encode(
            &[UniversalEvent::ToolCallDelta {
                id: "call_123".to_string(),
                name: Some("exec_command".to_string()),
                arguments_delta: String::new(),
            }],
            &mut state,
        )
        .expect("first delta encodes");
        let second = encode(
            &[UniversalEvent::ToolCallDelta {
                id: "call_123".to_string(),
                name: None,
                arguments_delta: "{\"cmd\":\"ls\"}".to_string(),
            }],
            &mut state,
        )
        .expect("second delta encodes");

        assert_eq!(
            first[0].data["choices"][0]["delta"]["tool_calls"][0]["index"],
            0
        );
        assert_eq!(
            second[0].data["choices"][0]["delta"]["tool_calls"][0]["index"],
            0
        );
        assert_eq!(
            second[0].data["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_123"
        );
    }

    #[test]
    fn encodes_content_indexes_as_single_chat_choice() {
        let mut state = EncodeState::default();
        let wire = encode(
            &[
                UniversalEvent::TextDelta {
                    index: 2,
                    text: "answer".to_string(),
                },
                UniversalEvent::ReasoningDelta {
                    index: 1,
                    text: "thought".to_string(),
                },
            ],
            &mut state,
        )
        .expect("events encode");

        assert_eq!(wire[0].data["choices"][0]["index"], 0);
        assert_eq!(wire[1].data["choices"][0]["index"], 0);
    }
}
