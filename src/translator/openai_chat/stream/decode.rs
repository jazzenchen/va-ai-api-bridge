use serde_json::Value;

use crate::schema::openai::ChatCompletionChunk;
use crate::translator::{common, openai};
use crate::{ApiProxyError, ContentBlock, DecodeState, Result, Role, UniversalEvent};

use super::super::response::response_message_id;

pub(super) fn decode_chunk(raw: Value, state: &mut DecodeState) -> Result<Vec<UniversalEvent>> {
    let chunk: ChatCompletionChunk = serde_json::from_value(raw)
        .map_err(|error| ApiProxyError::invalid_response(error.to_string()))?;
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
                common::ensure_message_start(&mut events, state, message_id, role);
            }

            for block in openai::openai_content_to_blocks(delta.content.as_ref()) {
                match block {
                    ContentBlock::Text { text } => {
                        common::ensure_content_start(
                            &mut events,
                            state,
                            choice_index,
                            ContentBlock::Text {
                                text: String::new(),
                            },
                        );
                        events.push(UniversalEvent::TextDelta {
                            index: choice_index,
                            text,
                        });
                    }
                    block => common::push_block_events(&mut events, choice_index, block),
                }
            }

            for tool_call in delta.tool_calls {
                let id = tool_call
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("tool_call_{choice_index}"));
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
