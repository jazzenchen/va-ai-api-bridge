use std::collections::BTreeMap;

use crate::{ContentBlock, Extensions, FinishReason, UniversalEvent};
use serde_json::{json, Value};

const TOOL_CALLS_SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
const TOOL_CALLS_SECTION_END: &str = "<|tool_calls_section_end|>";
const TOOL_CALL_BEGIN: &str = "<|tool_call_begin|>";
const TOOL_CALL_ARGUMENT_BEGIN: &str = "<|tool_call_argument_begin|>";
const TOOL_CALL_END: &str = "<|tool_call_end|>";

#[derive(Debug, Clone, Default)]
pub struct KimiBridgeAdapter {
    pending_text: BTreeMap<usize, PendingTextBlock>,
    saw_tool_call: bool,
}

#[derive(Debug, Clone)]
struct PendingTextBlock {
    index: usize,
    text: String,
}

#[derive(Debug, Clone)]
struct KimiTaggedToolCall {
    id: String,
    name: String,
    arguments: Value,
}

impl KimiBridgeAdapter {
    pub fn prepare_anthropic_request(&mut self, request: &mut Value) {
        let Some(object) = request.as_object_mut() else {
            return;
        };
        normalize_kimi_coding_model(object);
        object.insert("thinking".to_string(), json!({ "type": "disabled" }));
    }

    pub fn transform_upstream_events(&mut self, events: &mut Vec<UniversalEvent>) {
        if events.is_empty() {
            return;
        }

        let original = std::mem::take(events);
        let mut transformed = Vec::with_capacity(original.len());

        for event in original {
            match event {
                UniversalEvent::ContentStart {
                    index,
                    block: ContentBlock::Text { .. },
                } => {
                    self.pending_text.insert(
                        index,
                        PendingTextBlock {
                            index,
                            text: String::new(),
                        },
                    );
                }
                UniversalEvent::TextDelta { index, text } => {
                    if let Some(pending) = self.pending_text.get_mut(&index) {
                        pending.text.push_str(&text);
                    } else {
                        transformed.push(UniversalEvent::TextDelta { index, text });
                    }
                }
                UniversalEvent::ContentDone { index, final_block } => {
                    if let Some(mut pending) = self.pending_text.remove(&index) {
                        if pending.text.is_empty() {
                            if let Some(ContentBlock::Text { text }) = final_block.as_ref() {
                                pending.text.push_str(text);
                            }
                        }
                        self.flush_text_block(pending, final_block, &mut transformed);
                    } else {
                        transformed.push(UniversalEvent::ContentDone { index, final_block });
                    }
                }
                UniversalEvent::MessageDone {
                    finish_reason,
                    usage,
                    extensions,
                } => {
                    self.flush_all_pending_text(&mut transformed);
                    transformed.push(UniversalEvent::MessageDone {
                        finish_reason: if self.saw_tool_call {
                            Some(FinishReason::ToolCall)
                        } else {
                            finish_reason
                        },
                        usage,
                        extensions,
                    });
                    self.saw_tool_call = false;
                }
                UniversalEvent::ResponseDone { usage, extensions } => {
                    self.flush_all_pending_text(&mut transformed);
                    transformed.push(UniversalEvent::ResponseDone { usage, extensions });
                    self.saw_tool_call = false;
                }
                other => transformed.push(other),
            }
        }

        *events = transformed;
    }

    fn flush_all_pending_text(&mut self, output: &mut Vec<UniversalEvent>) {
        let pending = std::mem::take(&mut self.pending_text);
        for (_, block) in pending {
            self.flush_text_block(block, None, output);
        }
    }

    fn flush_text_block(
        &mut self,
        pending: PendingTextBlock,
        final_block: Option<ContentBlock>,
        output: &mut Vec<UniversalEvent>,
    ) {
        if let Some(tool_calls) = parse_kimi_tagged_tool_calls(&pending.text) {
            self.saw_tool_call = true;
            for (offset, tool_call) in tool_calls.into_iter().enumerate() {
                let index = pending.index + offset;
                let block = ContentBlock::ToolCall {
                    id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                    extensions: Extensions::default(),
                };
                output.push(UniversalEvent::ContentStart {
                    index,
                    block: block.clone(),
                });
                output.push(UniversalEvent::ToolCallDelta {
                    id: tool_call.id,
                    name: Some(tool_call.name),
                    arguments_delta: tool_call.arguments.to_string(),
                });
                output.push(UniversalEvent::ContentDone {
                    index,
                    final_block: Some(block),
                });
            }
            return;
        }

        output.push(UniversalEvent::ContentStart {
            index: pending.index,
            block: ContentBlock::Text {
                text: pending.text.clone(),
            },
        });
        if !pending.text.is_empty() {
            output.push(UniversalEvent::TextDelta {
                index: pending.index,
                text: pending.text.clone(),
            });
        }
        output.push(UniversalEvent::ContentDone {
            index: pending.index,
            final_block: final_block.or(Some(ContentBlock::Text { text: pending.text })),
        });
    }
}

fn normalize_kimi_coding_model(object: &mut serde_json::Map<String, Value>) {
    let Some(model) = object.get("model").and_then(Value::as_str) else {
        return;
    };
    if matches!(model, "kimi-code" | "k2p5") {
        object.insert(
            "model".to_string(),
            Value::String("kimi-for-coding".to_string()),
        );
    }
}

fn parse_kimi_tagged_tool_calls(text: &str) -> Option<Vec<KimiTaggedToolCall>> {
    let trimmed = text.trim();
    if !trimmed.starts_with(TOOL_CALLS_SECTION_BEGIN) || !trimmed.ends_with(TOOL_CALLS_SECTION_END)
    {
        return None;
    }

    let mut cursor = TOOL_CALLS_SECTION_BEGIN.len();
    let section_end = trimmed.len() - TOOL_CALLS_SECTION_END.len();
    let mut tool_calls = Vec::new();

    while cursor < section_end {
        while cursor < section_end {
            let ch = trimmed[cursor..].chars().next()?;
            if !ch.is_whitespace() {
                break;
            }
            cursor += ch.len_utf8();
        }
        if cursor >= section_end {
            break;
        }
        if !trimmed[cursor..].starts_with(TOOL_CALL_BEGIN) {
            return None;
        }

        let name_start = cursor + TOOL_CALL_BEGIN.len();
        let arg_marker = trimmed[name_start..].find(TOOL_CALL_ARGUMENT_BEGIN)? + name_start;
        if arg_marker >= section_end {
            return None;
        }

        let raw_id = trimmed[name_start..arg_marker].trim();
        if raw_id.is_empty() {
            return None;
        }

        let args_start = arg_marker + TOOL_CALL_ARGUMENT_BEGIN.len();
        let call_end = trimmed[args_start..].find(TOOL_CALL_END)? + args_start;
        if call_end > section_end {
            return None;
        }

        let raw_args = trimmed[args_start..call_end].trim();
        let arguments: Value = serde_json::from_str(raw_args).ok()?;
        if !arguments.is_object() {
            return None;
        }
        let name = strip_tagged_tool_call_counter(raw_id);
        if name.is_empty() {
            return None;
        }

        tool_calls.push(KimiTaggedToolCall {
            id: raw_id.to_string(),
            name,
            arguments,
        });
        cursor = call_end + TOOL_CALL_END.len();
    }

    if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    }
}

fn strip_tagged_tool_call_counter(value: &str) -> String {
    let trimmed = value.trim();
    match trimmed.rsplit_once(':') {
        Some((name, suffix)) if suffix.chars().all(|ch| ch.is_ascii_digit()) => {
            name.trim().to_string()
        }
        _ => trimmed.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn disables_kimi_thinking_for_anthropic_requests() {
        let mut adapter = KimiBridgeAdapter::default();
        let mut request = json!({ "model": "kimi-for-coding", "messages": [] });

        adapter.prepare_anthropic_request(&mut request);

        assert_eq!(request["thinking"], json!({ "type": "disabled" }));
    }

    #[test]
    fn normalizes_legacy_kimi_coding_model_aliases() {
        for model in ["kimi-code", "k2p5"] {
            let mut adapter = KimiBridgeAdapter::default();
            let mut request = json!({ "model": model, "messages": [] });

            adapter.prepare_anthropic_request(&mut request);

            assert_eq!(request["model"], "kimi-for-coding");
        }
    }

    #[test]
    fn rewrites_tagged_tool_calls_into_structured_events() {
        let mut adapter = KimiBridgeAdapter::default();
        let mut events = vec![
            UniversalEvent::ContentStart {
                index: 0,
                block: ContentBlock::Text {
                    text: String::new(),
                },
            },
            UniversalEvent::TextDelta {
                index: 0,
                text: " <|tool_calls_section_begin|> <|tool_call_begin|> functions.read:0 <|tool_call_argument_begin|> {\"file_path\":\"./package.json\"} <|tool_call_end|> <|tool_calls_section_end|>".to_string(),
            },
            UniversalEvent::ContentDone {
                index: 0,
                final_block: None,
            },
            UniversalEvent::MessageDone {
                finish_reason: Some(FinishReason::Stop),
                usage: None,
                extensions: Extensions::default(),
            },
        ];

        adapter.transform_upstream_events(&mut events);

        assert!(matches!(
            events.first(),
            Some(UniversalEvent::ContentStart {
                block: ContentBlock::ToolCall { name, .. },
                ..
            }) if name == "functions.read"
        ));
        assert!(matches!(
            events.get(1),
            Some(UniversalEvent::ToolCallDelta {
                id,
                arguments_delta,
                ..
            }) if id == "functions.read:0" && arguments_delta == "{\"file_path\":\"./package.json\"}"
        ));
        assert!(matches!(
            events.last(),
            Some(UniversalEvent::MessageDone {
                finish_reason: Some(FinishReason::ToolCall),
                ..
            })
        ));
    }
}
