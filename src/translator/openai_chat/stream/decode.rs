use serde_json::Value;

use crate::schema::openai::{ChatCompletionChunk, ChatToolCall};
use crate::translator::{common, openai};
use crate::{ApiBridgeError, ContentBlock, DecodeState, Result, Role, UniversalEvent};

use super::super::response::response_message_id;

pub(super) fn decode_chunk(raw: Value, state: &mut DecodeState) -> Result<Vec<UniversalEvent>> {
    let chunk: ChatCompletionChunk = serde_json::from_value(raw)
        .map_err(|error| ApiBridgeError::invalid_response(error.to_string()))?;
    let mut events = Vec::new();
    common::ensure_response_start(&mut events, state, chunk.id.clone(), chunk.model.clone());

    let usage = openai::openai_usage_to_universal(chunk.usage.as_ref());
    for choice in chunk.choices {
        let choice_index = choice.index.unwrap_or(0) as usize;
        let message_id = response_message_id(chunk.id.as_deref(), choice.index);

        if let Some(delta) = choice.delta {
            let role = delta
                .role
                .as_deref()
                .and_then(common::role_from_wire)
                .unwrap_or(Role::Assistant);
            if delta.role.is_some() || delta.content.is_some() || !delta.tool_calls.is_empty() {
                common::ensure_message_start(&mut events, state, message_id.clone(), role);
            }

            for block in openai::openai_content_to_blocks(delta.content.as_ref()) {
                match block {
                    ContentBlock::Text { text } => {
                        let content_index = chat_content_index(state, choice_index, "text");
                        ensure_chat_content_start(
                            &mut events,
                            state,
                            choice_index,
                            content_index,
                            ContentBlock::Text {
                                text: String::new(),
                            },
                        );
                        events.push(UniversalEvent::TextDelta {
                            index: content_index,
                            text,
                        });
                    }
                    block => {
                        let content_index = next_chat_content_index(state, choice_index);
                        common::push_block_events(&mut events, content_index, block);
                    }
                }
            }

            if let Some(reasoning_delta) = delta
                .extra
                .get("reasoning_content")
                .and_then(Value::as_str)
                .filter(|content| !content.is_empty())
            {
                common::ensure_message_start(&mut events, state, message_id.clone(), role);
                let content_index = chat_content_index(state, choice_index, "reasoning");
                ensure_chat_content_start(
                    &mut events,
                    state,
                    choice_index,
                    content_index,
                    ContentBlock::Reasoning {
                        text: None,
                        encrypted: None,
                        extensions: common::empty_extensions(),
                    },
                );
                events.push(UniversalEvent::ReasoningDelta {
                    index: content_index,
                    text: reasoning_delta.to_string(),
                });
            }

            for (fallback_index, tool_call) in delta.tool_calls.into_iter().enumerate() {
                let id = stream_tool_call_id(state, choice_index, fallback_index, &tool_call);
                let function = tool_call.function;
                events.push(UniversalEvent::ToolCallDelta {
                    id,
                    name: function.as_ref().and_then(|function| function.name.clone()),
                    arguments_delta: function
                        .and_then(|function| function.arguments)
                        .unwrap_or_default(),
                });
            }
        }

        if choice.finish_reason.is_some() {
            events.push(UniversalEvent::MessageDone {
                finish_reason: openai::finish_from_openai(choice.finish_reason.as_deref()),
                usage: usage.clone(),
                extensions: common::empty_extensions(),
            });
            if common::mark_once(state, "response_done") {
                events.push(UniversalEvent::ResponseDone {
                    usage: usage.clone(),
                    extensions: common::empty_extensions(),
                });
            }
        }
    }

    if usage.is_some() && common::mark_once(state, "response_done") {
        events.push(UniversalEvent::ResponseDone {
            usage,
            extensions: common::empty_extensions(),
        });
    }

    Ok(events)
}

fn chat_content_index(state: &mut DecodeState, choice_index: usize, kind: &str) -> usize {
    let key = format!("openaiChatContentIndex:{choice_index}:{kind}");
    if let Some(index) = state.extensions.get(&key).and_then(Value::as_u64) {
        return index as usize;
    }
    let index = next_chat_content_index(state, choice_index);
    state
        .extensions
        .insert(key, Value::Number((index as u64).into()));
    index
}

fn next_chat_content_index(state: &mut DecodeState, choice_index: usize) -> usize {
    let key = format!("openaiChatNextContentIndex:{choice_index}");
    let index = state
        .extensions
        .get(&key)
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    state
        .extensions
        .insert(key, Value::Number(((index + 1) as u64).into()));
    index
}

fn ensure_chat_content_start(
    events: &mut Vec<UniversalEvent>,
    state: &mut DecodeState,
    choice_index: usize,
    index: usize,
    block: ContentBlock,
) {
    if common::mark_once(
        state,
        &format!("openaiChatContentStart:{choice_index}:{index}"),
    ) {
        events.push(UniversalEvent::ContentStart { index, block });
    }
}

fn stream_tool_call_id(
    state: &mut DecodeState,
    choice_index: usize,
    fallback_index: usize,
    tool_call: &ChatToolCall,
) -> String {
    let tool_index = tool_call.index.unwrap_or(fallback_index as u64);
    let key = format!("openaiChatToolCallId:{choice_index}:{tool_index}");
    if let Some(id) = tool_call.id.as_deref().filter(|id| !id.is_empty()) {
        state.extensions.insert(key, Value::String(id.to_string()));
        return id.to_string();
    }
    if let Some(id) = state.extensions.get(&key).and_then(Value::as_str) {
        return id.to_string();
    }
    let fallback_id = format!("tool_call_{choice_index}_{tool_index}");
    state
        .extensions
        .insert(key, Value::String(fallback_id.clone()));
    fallback_id
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{DecodeState, UniversalEvent};

    use super::decode_chunk;

    #[test]
    fn keeps_stream_tool_call_id_across_indexed_deltas() {
        let mut state = DecodeState::default();
        let first = decode_chunk(
            json!({
                "id": "chatcmpl_1",
                "model": "deepseek-v4-pro",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "exec_command",
                                "arguments": ""
                            }
                        }]
                    }
                }]
            }),
            &mut state,
        )
        .expect("first chunk decodes");
        let second = decode_chunk(
            json!({
                "id": "chatcmpl_1",
                "model": "deepseek-v4-pro",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {
                                "arguments": "{\"cmd\":\"ls\"}"
                            }
                        }]
                    }
                }]
            }),
            &mut state,
        )
        .expect("second chunk decodes");

        let first_tool_delta = first
            .iter()
            .find_map(|event| match event {
                UniversalEvent::ToolCallDelta {
                    id,
                    name,
                    arguments_delta,
                } => Some((id.as_str(), name.as_deref(), arguments_delta.as_str())),
                _ => None,
            })
            .expect("first tool delta");
        let second_tool_delta = second
            .iter()
            .find_map(|event| match event {
                UniversalEvent::ToolCallDelta {
                    id,
                    name,
                    arguments_delta,
                } => Some((id.as_str(), name.as_deref(), arguments_delta.as_str())),
                _ => None,
            })
            .expect("second tool delta");

        assert_eq!(first_tool_delta, ("call_123", Some("exec_command"), ""));
        assert_eq!(second_tool_delta, ("call_123", None, "{\"cmd\":\"ls\"}"));
    }

    #[test]
    fn emits_reasoning_content_deltas() {
        let mut state = DecodeState::default();
        let events = decode_chunk(
            json!({
                "id": "chatcmpl_1",
                "model": "deepseek-v4-pro",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "Need to inspect files."
                    }
                }]
            }),
            &mut state,
        )
        .expect("chunk decodes");

        assert!(events.iter().any(|event| matches!(
            event,
            UniversalEvent::ReasoningDelta { text, .. } if text == "Need to inspect files."
        )));
    }

    #[test]
    fn keeps_reasoning_and_text_on_distinct_content_indexes() {
        let mut state = DecodeState::default();
        let reasoning = decode_chunk(
            json!({
                "id": "chatcmpl_1",
                "model": "deepseek-v4-pro",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "Think first."
                    }
                }]
            }),
            &mut state,
        )
        .expect("reasoning chunk decodes");
        let text = decode_chunk(
            json!({
                "id": "chatcmpl_1",
                "model": "deepseek-v4-pro",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "Then answer."
                    }
                }]
            }),
            &mut state,
        )
        .expect("text chunk decodes");

        assert!(reasoning.iter().any(|event| matches!(
            event,
            UniversalEvent::ReasoningDelta { index: 0, text } if text == "Think first."
        )));
        assert!(text.iter().any(|event| matches!(
            event,
            UniversalEvent::TextDelta { index: 1, text } if text == "Then answer."
        )));
    }
}
