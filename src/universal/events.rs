use std::collections::BTreeMap;

use serde_json::Value;

use crate::stream::UniversalEvent;

use super::{ContentBlock, Extensions, Role, UniversalItem, UniversalResponse};

impl UniversalResponse {
    pub fn from_events(events: &[UniversalEvent]) -> Self {
        let mut response = UniversalResponse::default();
        let mut current_message: Option<PartialMessage> = None;
        let mut pending_tool_calls: Vec<PartialToolCall> = Vec::new();
        let mut pending_reasoning: BTreeMap<usize, String> = BTreeMap::new();

        for event in events {
            match event {
                UniversalEvent::ResponseStart {
                    id,
                    model,
                    extensions,
                } => {
                    response.id = id.clone();
                    response.model = model.clone();
                    response.extensions.extend(extensions.clone());
                    response
                        .status
                        .get_or_insert_with(|| "in_progress".to_string());
                }
                UniversalEvent::MessageStart {
                    id,
                    role,
                    extensions,
                } => {
                    flush_partial_message(&mut response.output, &mut current_message);
                    flush_pending_tool_calls(&mut response.output, &mut pending_tool_calls);
                    current_message = Some(PartialMessage {
                        id: Some(id.clone()),
                        role: *role,
                        content: BTreeMap::new(),
                        extensions: extensions.clone(),
                    });
                }
                UniversalEvent::ContentDone { index, final_block } => {
                    if let Some(ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                        extensions,
                    }) = final_block
                    {
                        remember_tool_call_metadata(
                            &mut pending_tool_calls,
                            id,
                            Some(name),
                            extensions,
                        );
                        fill_tool_call_arguments_if_empty(&mut pending_tool_calls, id, arguments);
                    } else if let (Some(message), Some(block)) = (&mut current_message, final_block)
                    {
                        message.content.insert(*index, block.clone());
                    }
                }
                UniversalEvent::TextDelta { index, text } => {
                    if let Some(message) = &mut current_message {
                        append_text_block(&mut message.content, *index, text);
                    }
                }
                UniversalEvent::ReasoningDelta { index, text } => {
                    if let Some(message) = &mut current_message {
                        append_reasoning_block(&mut message.content, *index, text);
                    } else {
                        pending_reasoning
                            .entry(*index)
                            .and_modify(|existing| existing.push_str(text))
                            .or_insert_with(|| text.clone());
                    }
                }
                UniversalEvent::ToolCallDelta {
                    id,
                    name,
                    arguments_delta,
                } => {
                    flush_partial_message(&mut response.output, &mut current_message);
                    append_tool_call_delta(
                        &mut pending_tool_calls,
                        id,
                        name.as_deref(),
                        arguments_delta,
                    );
                }
                UniversalEvent::MessageDone {
                    finish_reason,
                    usage,
                    extensions: _,
                } => {
                    response.finish_reason = *finish_reason;
                    if usage.is_some() {
                        response.usage = usage.clone();
                    }
                    flush_partial_message(&mut response.output, &mut current_message);
                    flush_pending_tool_calls(&mut response.output, &mut pending_tool_calls);
                }
                UniversalEvent::ResponseDone { usage, extensions } => {
                    if usage.is_some() {
                        response.usage = usage.clone();
                    }
                    flush_pending_tool_calls(&mut response.output, &mut pending_tool_calls);
                    response.status = Some("completed".to_string());
                    response.extensions.extend(extensions.clone());
                }
                UniversalEvent::Error { message, raw } => {
                    response.status = Some("failed".to_string());
                    response.extensions.insert(
                        "error".to_string(),
                        raw.clone()
                            .unwrap_or_else(|| Value::String(message.clone())),
                    );
                }
                UniversalEvent::Unknown { raw, .. } => {
                    flush_partial_message(&mut response.output, &mut current_message);
                    flush_pending_tool_calls(&mut response.output, &mut pending_tool_calls);
                    response
                        .output
                        .push(UniversalItem::Unknown { raw: raw.clone() });
                }
                UniversalEvent::ContentStart { block, .. } => {
                    if let ContentBlock::ToolCall {
                        id,
                        name,
                        extensions,
                        ..
                    } = block
                    {
                        remember_tool_call_metadata(
                            &mut pending_tool_calls,
                            id,
                            Some(name),
                            extensions,
                        );
                    }
                }
            }
        }

        flush_partial_message(&mut response.output, &mut current_message);
        flush_pending_tool_calls(&mut response.output, &mut pending_tool_calls);
        for (index, text) in pending_reasoning {
            response.output.insert(
                index.min(response.output.len()),
                UniversalItem::Reasoning {
                    id: None,
                    text: Some(text),
                    encrypted: None,
                    extensions: Extensions::new(),
                },
            );
        }
        response
    }

    pub fn to_events(&self) -> Vec<UniversalEvent> {
        let mut events = Vec::new();
        events.push(UniversalEvent::ResponseStart {
            id: self.id.clone(),
            model: self.model.clone(),
            extensions: self.extensions.clone(),
        });
        for (item_index, item) in self.output.iter().enumerate() {
            match item {
                UniversalItem::Message {
                    role,
                    id,
                    content,
                    extensions,
                } => {
                    events.push(UniversalEvent::MessageStart {
                        id: id
                            .clone()
                            .unwrap_or_else(|| format!("message_{item_index}")),
                        role: *role,
                        extensions: extensions.clone(),
                    });
                    for (content_index, block) in content.iter().cloned().enumerate() {
                        events.push(UniversalEvent::ContentStart {
                            index: content_index,
                            block: block.clone(),
                        });
                        if let ContentBlock::Text { text } = &block {
                            events.push(UniversalEvent::TextDelta {
                                index: content_index,
                                text: text.clone(),
                            });
                        }
                        if let ContentBlock::Reasoning {
                            text: Some(text), ..
                        } = &block
                        {
                            events.push(UniversalEvent::ReasoningDelta {
                                index: content_index,
                                text: text.clone(),
                            });
                        }
                        events.push(UniversalEvent::ContentDone {
                            index: content_index,
                            final_block: Some(block),
                        });
                    }
                    events.push(UniversalEvent::MessageDone {
                        finish_reason: self.finish_reason,
                        usage: self.usage.clone(),
                        extensions: Extensions::new(),
                    });
                }
                UniversalItem::ToolCall {
                    id,
                    name,
                    arguments,
                    ..
                } => events.push(UniversalEvent::ToolCallDelta {
                    id: id.clone(),
                    name: Some(name.clone()),
                    arguments_delta: match arguments {
                        Value::String(value) => value.clone(),
                        value => serde_json::to_string(value).unwrap_or_default(),
                    },
                }),
                UniversalItem::Reasoning { text, .. } => {
                    if let Some(text) = text {
                        events.push(UniversalEvent::ReasoningDelta {
                            index: item_index,
                            text: text.clone(),
                        });
                    }
                }
                UniversalItem::ToolResult { .. } | UniversalItem::Unknown { .. } => {}
            }
        }
        events.push(UniversalEvent::ResponseDone {
            usage: self.usage.clone(),
            extensions: Extensions::new(),
        });
        events
    }
}

struct PartialMessage {
    id: Option<String>,
    role: Role,
    content: BTreeMap<usize, ContentBlock>,
    extensions: Extensions,
}

struct PartialToolCall {
    id: String,
    name: Option<String>,
    arguments: String,
    saw_delta: bool,
    extensions: Extensions,
}

fn flush_partial_message(output: &mut Vec<UniversalItem>, message: &mut Option<PartialMessage>) {
    let Some(message) = message.take() else {
        return;
    };
    output.push(UniversalItem::Message {
        role: message.role,
        id: message.id,
        content: message.content.into_values().collect(),
        extensions: message.extensions,
    });
}

fn append_tool_call_delta(
    pending_tool_calls: &mut Vec<PartialToolCall>,
    id: &str,
    name: Option<&str>,
    arguments_delta: &str,
) {
    remember_tool_call_metadata(pending_tool_calls, id, name, &Extensions::new());
    let Some(tool_call) = pending_tool_calls
        .iter_mut()
        .find(|tool_call| tool_call.id == id)
    else {
        return;
    };

    tool_call.arguments.push_str(arguments_delta);
    tool_call.saw_delta = true;
}

fn remember_tool_call_metadata(
    pending_tool_calls: &mut Vec<PartialToolCall>,
    id: &str,
    name: Option<&str>,
    extensions: &Extensions,
) {
    let Some(tool_call) = pending_tool_calls
        .iter_mut()
        .find(|tool_call| tool_call.id == id)
    else {
        pending_tool_calls.push(PartialToolCall {
            id: id.to_string(),
            name: name
                .filter(|name| !name.is_empty())
                .map(ToString::to_string),
            arguments: String::new(),
            saw_delta: false,
            extensions: extensions.clone(),
        });
        return;
    };

    if let Some(name) = name.filter(|name| !name.is_empty()) {
        tool_call.name = Some(name.to_string());
    }
    tool_call.extensions.extend(extensions.clone());
}

fn fill_tool_call_arguments_if_empty(
    pending_tool_calls: &mut Vec<PartialToolCall>,
    id: &str,
    arguments: &Value,
) {
    let Some(tool_call) = pending_tool_calls
        .iter_mut()
        .find(|tool_call| tool_call.id == id)
    else {
        return;
    };
    if tool_call.saw_delta || !tool_call.arguments.is_empty() {
        return;
    }
    tool_call.arguments = stringify_tool_arguments(arguments);
}

fn flush_pending_tool_calls(
    output: &mut Vec<UniversalItem>,
    pending_tool_calls: &mut Vec<PartialToolCall>,
) {
    for tool_call in std::mem::take(pending_tool_calls) {
        output.push(UniversalItem::ToolCall {
            id: tool_call.id,
            name: tool_call.name.unwrap_or_default(),
            arguments: tool_call
                .arguments
                .parse::<Value>()
                .unwrap_or_else(|_| Value::String(tool_call.arguments)),
            extensions: tool_call.extensions,
        });
    }
}

fn stringify_tool_arguments(arguments: &Value) -> String {
    match arguments {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        value => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn append_text_block(content: &mut BTreeMap<usize, ContentBlock>, index: usize, text: &str) {
    match content.get_mut(&index) {
        Some(ContentBlock::Text { text: existing }) => existing.push_str(text),
        Some(_) => {}
        None => {
            content.insert(
                index,
                ContentBlock::Text {
                    text: text.to_string(),
                },
            );
        }
    }
}

fn append_reasoning_block(content: &mut BTreeMap<usize, ContentBlock>, index: usize, text: &str) {
    match content.get_mut(&index) {
        Some(ContentBlock::Reasoning {
            text: Some(existing),
            ..
        }) => existing.push_str(text),
        Some(_) => {}
        None => {
            content.insert(
                index,
                ContentBlock::Reasoning {
                    text: Some(text.to_string()),
                    encrypted: None,
                    extensions: Extensions::new(),
                },
            );
        }
    }
}
#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{UniversalEvent, UniversalItem, UniversalResponse};

    #[test]
    fn aggregates_split_tool_call_deltas_by_id() {
        let response = UniversalResponse::from_events(&[
            UniversalEvent::ToolCallDelta {
                id: "call_pwd".to_string(),
                name: Some("exec_command".to_string()),
                arguments_delta: "{\"cmd\"".to_string(),
            },
            UniversalEvent::ToolCallDelta {
                id: "call_pwd".to_string(),
                name: None,
                arguments_delta: ":\"pwd\"}".to_string(),
            },
        ]);

        assert_eq!(response.output.len(), 1);
        assert!(matches!(
            &response.output[0],
            UniversalItem::ToolCall {
                id,
                name,
                arguments,
                ..
            } if id == "call_pwd" && name == "exec_command" && arguments == &json!({ "cmd": "pwd" })
        ));
    }
}
