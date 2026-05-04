use serde_json::Value;

use crate::schema::openai::{ResponsesItem, ResponsesResponse};
use crate::translator::{common, openai};
use crate::{ApiProxyError, ContentBlock, DecodeState, Result, Role, UniversalEvent};

pub(super) fn decode(raw: Value) -> Result<Vec<UniversalEvent>> {
    let response: ResponsesResponse = serde_json::from_value(raw)
        .map_err(|error| ApiProxyError::invalid_response(error.to_string()))?;
    let usage = openai::openai_usage_to_universal(response.usage.as_ref());
    let mut events = vec![UniversalEvent::ResponseStart {
        id: response.id.clone(),
        model: response.model.clone(),
        extensions: common::empty_extensions(),
    }];

    if let Some(error) = response.error {
        events.push(UniversalEvent::Error {
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("OpenAI response error")
                .to_string(),
            raw: Some(error),
        });
    }

    for (index, item) in response.output.into_iter().enumerate() {
        push_response_item_events(&mut events, response.id.as_deref(), index, item);
    }

    events.push(UniversalEvent::ResponseDone {
        usage,
        extensions: common::empty_extensions(),
    });
    Ok(events)
}

pub(super) fn push_response_item_start(
    events: &mut Vec<UniversalEvent>,
    state: &mut DecodeState,
    index: usize,
    item: ResponsesItem,
) {
    match item.kind.as_deref() {
        Some("message") | None if item.role.is_some() => common::ensure_message_start(
            events,
            state,
            item.id
                .unwrap_or_else(|| format!("response_message_{index}")),
            item.role
                .as_deref()
                .and_then(common::role_from_wire)
                .unwrap_or(Role::Assistant),
        ),
        Some("function_call") => events.push(UniversalEvent::ToolCallDelta {
            id: item.call_id.or(item.id).unwrap_or_default(),
            name: item.name,
            arguments_delta: item
                .arguments
                .as_ref()
                .map(common::stringify_arguments)
                .unwrap_or_default(),
        }),
        _ => {}
    }
}

pub(super) fn push_response_item_events(
    events: &mut Vec<UniversalEvent>,
    response_id: Option<&str>,
    index: usize,
    item: ResponsesItem,
) {
    match item.kind.as_deref() {
        Some("message") | None if item.role.is_some() => {
            events.push(UniversalEvent::MessageStart {
                id: item
                    .id
                    .unwrap_or_else(|| format!("{}:{index}", response_id.unwrap_or("response"))),
                role: item
                    .role
                    .as_deref()
                    .and_then(common::role_from_wire)
                    .unwrap_or(Role::Assistant),
                extensions: common::empty_extensions(),
            });
            for (content_index, block) in openai::openai_content_to_blocks(item.content.as_ref())
                .into_iter()
                .enumerate()
            {
                common::push_block_events(events, content_index, block);
            }
            events.push(UniversalEvent::MessageDone {
                finish_reason: None,
                usage: None,
                extensions: common::empty_extensions(),
            });
        }
        Some("function_call") => {
            events.push(UniversalEvent::ToolCallDelta {
                id: item.call_id.or(item.id).unwrap_or_default(),
                name: item.name,
                arguments_delta: item
                    .arguments
                    .as_ref()
                    .map(common::stringify_arguments)
                    .unwrap_or_default(),
            });
        }
        Some("reasoning") => {
            if let Some(text) = text_from_openai_content(item.content.as_ref()) {
                events.push(UniversalEvent::ReasoningDelta { index, text });
            }
        }
        _ => events.push(UniversalEvent::Unknown {
            raw: serde_json::to_value(item).unwrap_or(Value::Null),
            tags: Default::default(),
        }),
    }
}

fn text_from_openai_content(
    content: Option<&crate::schema::openai::OpenAiContent>,
) -> Option<String> {
    let text = openai::openai_content_to_blocks(content)
        .into_iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text),
            ContentBlock::Reasoning { text, .. } => text,
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    (!text.is_empty()).then_some(text)
}
