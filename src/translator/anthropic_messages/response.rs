use crate::schema::anthropic::AnthropicMessagesResponse;
use crate::translator::{anthropic, common};
use crate::{ApiProxyError, Result, Role, UniversalEvent};

pub(super) fn decode(raw: serde_json::Value) -> Result<Vec<UniversalEvent>> {
    let response: AnthropicMessagesResponse = serde_json::from_value(raw)
        .map_err(|error| ApiProxyError::invalid_response(error.to_string()))?;
    let usage = anthropic::anthropic_usage_to_universal(response.usage.as_ref());
    let mut events = vec![
        UniversalEvent::ResponseStart {
            id: response.id.clone(),
            model: response.model.clone(),
            extensions: common::empty_extensions(),
        },
        UniversalEvent::MessageStart {
            id: response
                .id
                .clone()
                .unwrap_or_else(|| "anthropic_message".to_string()),
            role: response
                .role
                .as_deref()
                .and_then(common::role_from_wire)
                .unwrap_or(Role::Assistant),
            extensions: common::empty_extensions(),
        },
    ];

    for (index, block) in response
        .content
        .iter()
        .map(anthropic::anthropic_block_to_block)
        .enumerate()
    {
        common::push_block_events(&mut events, index, block);
    }
    events.push(UniversalEvent::MessageDone {
        finish_reason: anthropic::finish_from_anthropic(response.stop_reason.as_deref()),
        usage: usage.clone(),
        extensions: common::empty_extensions(),
    });
    events.push(UniversalEvent::ResponseDone {
        usage,
        extensions: common::empty_extensions(),
    });
    Ok(events)
}
