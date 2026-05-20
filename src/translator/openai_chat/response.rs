use crate::schema::openai::{ChatCompletionResponse, ChatToolCall};
use crate::translator::{common, openai};
use crate::{ApiBridgeError, ContentBlock, Result, Role, UniversalEvent};

pub(super) fn decode(raw: serde_json::Value) -> Result<Vec<UniversalEvent>> {
    let response: ChatCompletionResponse = serde_json::from_value(raw)
        .map_err(|error| ApiBridgeError::invalid_response(error.to_string()))?;
    let usage = openai::openai_usage_to_universal(response.usage.as_ref());
    let mut events = vec![UniversalEvent::ResponseStart {
        id: response.id.clone(),
        model: response.model.clone(),
        extensions: common::empty_extensions(),
    }];

    for choice in response.choices {
        if let Some(message) = choice.message {
            let role = common::role_from_wire(&message.role).unwrap_or(Role::Assistant);
            let message_id = response_message_id(response.id.as_deref(), choice.index);
            events.push(UniversalEvent::MessageStart {
                id: message_id,
                role,
                extensions: common::empty_extensions(),
            });

            let mut next_index = 0;
            for block in openai::openai_content_to_blocks(message.content.as_ref()) {
                common::push_block_events(&mut events, next_index, block);
                next_index += 1;
            }
            if let Some(reasoning_content) = message
                .extra
                .get("reasoning_content")
                .and_then(serde_json::Value::as_str)
                .filter(|content| !content.is_empty())
            {
                common::push_block_events(
                    &mut events,
                    next_index,
                    ContentBlock::Reasoning {
                        text: Some(reasoning_content.to_string()),
                        encrypted: None,
                        extensions: common::empty_extensions(),
                    },
                );
                next_index += 1;
            }
            for tool_call in message.tool_calls {
                common::push_block_events(
                    &mut events,
                    next_index,
                    chat_tool_call_to_block(tool_call),
                );
                next_index += 1;
            }
            events.push(UniversalEvent::MessageDone {
                finish_reason: openai::finish_from_openai(choice.finish_reason.as_deref()),
                usage: usage.clone(),
                extensions: common::empty_extensions(),
            });
        }
    }

    events.push(UniversalEvent::ResponseDone {
        usage,
        extensions: common::empty_extensions(),
    });
    Ok(events)
}

fn chat_tool_call_to_block(tool_call: ChatToolCall) -> ContentBlock {
    let function = tool_call.function;
    ContentBlock::ToolCall {
        id: tool_call.id.unwrap_or_default(),
        name: function
            .as_ref()
            .and_then(|function| function.name.clone())
            .unwrap_or_default(),
        arguments: common::parse_arguments(
            function
                .as_ref()
                .and_then(|function| function.arguments.as_deref()),
        ),
        extensions: common::empty_extensions(),
    }
}

pub(super) fn response_message_id(response_id: Option<&str>, choice_index: Option<u64>) -> String {
    format!(
        "{}:{}",
        response_id.unwrap_or("chatcmpl"),
        choice_index.unwrap_or(0)
    )
}
