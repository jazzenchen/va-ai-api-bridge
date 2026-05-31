use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::stream::UniversalEvent;
use crate::WireProtocol;

pub type Extensions = BTreeMap<String, Value>;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UniversalRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instructions: Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input: Vec<UniversalItem>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<UniversalTool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "GenerationConfig::is_empty")]
    pub generation: GenerationConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourcePayload>,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UniversalResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output: Vec<UniversalItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourcePayload>,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

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
                    if let (Some(message), Some(block)) = (&mut current_message, final_block) {
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
                UniversalEvent::ContentStart { .. } => {}
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
    let Some(tool_call) = pending_tool_calls
        .iter_mut()
        .find(|tool_call| tool_call.id == id)
    else {
        pending_tool_calls.push(PartialToolCall {
            id: id.to_string(),
            name: name
                .filter(|name| !name.is_empty())
                .map(ToString::to_string),
            arguments: arguments_delta.to_string(),
        });
        return;
    };

    if let Some(name) = name.filter(|name| !name.is_empty()) {
        tool_call.name = Some(name.to_string());
    }
    tool_call.arguments.push_str(arguments_delta);
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
            extensions: Extensions::new(),
        });
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum UniversalItem {
    Message {
        role: Role,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        content: Vec<ContentBlock>,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    ToolResult {
        tool_call_id: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        content: Vec<ContentBlock>,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    Reasoning {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted: Option<String>,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    Unknown {
        raw: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    File {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    ToolResult {
        tool_call_id: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        content: Vec<ContentBlock>,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    Reasoning {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted: Option<String>,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    Unknown {
        raw: Value,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Developer,
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Tool { name: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UniversalTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

impl GenerationConfig {
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.max_output_tokens.is_none()
            && self.extensions.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Error,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourcePayload {
    pub protocol: WireProtocol,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

fn is_false(value: &bool) -> bool {
    !*value
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
