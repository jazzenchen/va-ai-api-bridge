use std::collections::BTreeMap;

use crate::{ContentBlock, UniversalEvent};
use serde_json::{Number, Value};

#[derive(Debug, Clone, Default)]
pub struct MiniMaxBridgeAdapter {
    think_tags: MiniMaxThinkTagSplitter,
}

impl MiniMaxBridgeAdapter {
    pub fn prepare_chat_request(&mut self, chat_request: &mut Value) {
        let Some(object) = chat_request.as_object_mut() else {
            return;
        };

        normalize_system_messages(object);
        clamp_f64_setting(object, "temperature", 1.0);
        clamp_f64_setting(object, "top_p", 0.95);
        clamp_u64_setting(object, "max_completion_tokens", 2048);
    }

    pub fn transform_upstream_events(&mut self, events: &mut Vec<UniversalEvent>) {
        self.think_tags.transform(events);
    }
}

const THINK_OPEN_TAG: &str = "<think>";
const THINK_CLOSE_TAG: &str = "</think>";

#[derive(Debug, Clone, Default)]
struct MiniMaxThinkTagSplitter {
    blocks: BTreeMap<usize, ThinkBlockState>,
    passthrough_indexes: BTreeMap<usize, usize>,
    next_index: usize,
}

#[derive(Debug, Clone, Default)]
struct ThinkBlockState {
    parser: ThinkTagParser,
    current: Option<OutputBlock>,
    saw_delta: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OutputBlock {
    index: usize,
    kind: ThinkSegmentKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkSegmentKind {
    Text,
    Reasoning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ThinkSegment {
    kind: ThinkSegmentKind,
    text: String,
}

impl MiniMaxThinkTagSplitter {
    fn transform(&mut self, events: &mut Vec<UniversalEvent>) {
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
                    self.blocks.entry(index).or_default();
                }
                UniversalEvent::ContentStart { index, block } => {
                    let index = self.passthrough_index(index);
                    transformed.push(UniversalEvent::ContentStart { index, block });
                }
                UniversalEvent::TextDelta { index, text } => {
                    self.push_text(index, &text, &mut transformed);
                }
                UniversalEvent::ReasoningDelta { index, text } => {
                    let index = self.remapped_index(index);
                    transformed.push(UniversalEvent::ReasoningDelta { index, text });
                }
                UniversalEvent::ContentDone {
                    index,
                    final_block: Some(ContentBlock::Text { text }),
                } => {
                    let needs_final_text = self
                        .blocks
                        .get(&index)
                        .map(|block| !block.saw_delta && !text.is_empty())
                        .unwrap_or(false);
                    if needs_final_text {
                        self.push_text(index, &text, &mut transformed);
                    }
                    self.flush_text_index(index, &mut transformed);
                }
                UniversalEvent::ContentDone { index, final_block } => {
                    if self.blocks.contains_key(&index) {
                        self.flush_text_index(index, &mut transformed);
                    } else {
                        let index = self.remapped_index(index);
                        transformed.push(UniversalEvent::ContentDone { index, final_block });
                    }
                }
                UniversalEvent::MessageDone {
                    finish_reason,
                    usage,
                    extensions,
                } => {
                    self.flush_all_text(&mut transformed);
                    transformed.push(UniversalEvent::MessageDone {
                        finish_reason,
                        usage,
                        extensions,
                    });
                }
                UniversalEvent::ResponseDone { usage, extensions } => {
                    self.flush_all_text(&mut transformed);
                    transformed.push(UniversalEvent::ResponseDone { usage, extensions });
                }
                other => transformed.push(other),
            }
        }

        *events = transformed;
    }

    fn push_text(&mut self, index: usize, text: &str, output: &mut Vec<UniversalEvent>) {
        let segments = {
            let block = self.blocks.entry(index).or_default();
            block.saw_delta = true;
            block.parser.push(text)
        };
        self.emit_segments(index, segments, output);
    }

    fn flush_text_index(&mut self, index: usize, output: &mut Vec<UniversalEvent>) {
        let segments = match self.blocks.get_mut(&index) {
            Some(block) => block.parser.flush(),
            None => return,
        };
        self.emit_segments(index, segments, output);
        self.close_current_block(index, output);
        self.blocks.remove(&index);
    }

    fn flush_all_text(&mut self, output: &mut Vec<UniversalEvent>) {
        let indexes = self.blocks.keys().copied().collect::<Vec<_>>();
        for index in indexes {
            self.flush_text_index(index, output);
        }
    }

    fn emit_segments(
        &mut self,
        original_index: usize,
        segments: Vec<ThinkSegment>,
        output: &mut Vec<UniversalEvent>,
    ) {
        for segment in segments {
            if segment.text.is_empty() {
                continue;
            }
            let index = self.ensure_output_block(original_index, segment.kind, output);
            match segment.kind {
                ThinkSegmentKind::Text => output.push(UniversalEvent::TextDelta {
                    index,
                    text: segment.text,
                }),
                ThinkSegmentKind::Reasoning => output.push(UniversalEvent::ReasoningDelta {
                    index,
                    text: segment.text,
                }),
            }
        }
    }

    fn ensure_output_block(
        &mut self,
        original_index: usize,
        kind: ThinkSegmentKind,
        output: &mut Vec<UniversalEvent>,
    ) -> usize {
        if let Some(current) = self
            .blocks
            .get(&original_index)
            .and_then(|block| block.current)
        {
            if current.kind == kind {
                return current.index;
            }
        }

        self.close_current_block(original_index, output);
        let index = self.allocate_index();
        output.push(UniversalEvent::ContentStart {
            index,
            block: match kind {
                ThinkSegmentKind::Text => ContentBlock::Text {
                    text: String::new(),
                },
                ThinkSegmentKind::Reasoning => ContentBlock::Reasoning {
                    text: None,
                    encrypted: None,
                    extensions: Default::default(),
                },
            },
        });
        self.blocks.entry(original_index).or_default().current = Some(OutputBlock { index, kind });
        index
    }

    fn close_current_block(&mut self, original_index: usize, output: &mut Vec<UniversalEvent>) {
        let Some(current) = self
            .blocks
            .get_mut(&original_index)
            .and_then(|block| block.current.take())
        else {
            return;
        };
        output.push(UniversalEvent::ContentDone {
            index: current.index,
            final_block: None,
        });
    }

    fn allocate_index(&mut self) -> usize {
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    fn passthrough_index(&mut self, original_index: usize) -> usize {
        if let Some(index) = self.passthrough_indexes.get(&original_index) {
            return *index;
        }
        let index = self.allocate_index();
        self.passthrough_indexes.insert(original_index, index);
        index
    }

    fn remapped_index(&self, original_index: usize) -> usize {
        self.passthrough_indexes
            .get(&original_index)
            .copied()
            .unwrap_or(original_index)
    }
}

#[derive(Debug, Clone, Default)]
struct ThinkTagParser {
    inside_think: bool,
    pending: String,
}

impl ThinkTagParser {
    fn push(&mut self, text: &str) -> Vec<ThinkSegment> {
        let mut input = std::mem::take(&mut self.pending);
        input.push_str(text);
        self.parse_stable_input(&input, false)
    }

    fn flush(&mut self) -> Vec<ThinkSegment> {
        let input = std::mem::take(&mut self.pending);
        self.parse_stable_input(&input, true)
    }

    fn parse_stable_input(&mut self, input: &str, flush: bool) -> Vec<ThinkSegment> {
        let mut segments = Vec::new();
        let mut cursor = 0;

        while cursor < input.len() {
            let remaining = &input[cursor..];
            let tag = if self.inside_think {
                THINK_CLOSE_TAG
            } else {
                THINK_OPEN_TAG
            };

            if let Some(position) = remaining.find(tag) {
                self.push_segment(&mut segments, &remaining[..position]);
                cursor += position + tag.len();
                self.inside_think = !self.inside_think;
                continue;
            }

            let pending_len = if flush {
                0
            } else {
                pending_tag_suffix_len(remaining, tag)
            };
            let stable_len = remaining.len() - pending_len;
            self.push_segment(&mut segments, &remaining[..stable_len]);
            if pending_len > 0 {
                self.pending.push_str(&remaining[stable_len..]);
            }
            break;
        }

        segments
    }

    fn push_segment(&self, segments: &mut Vec<ThinkSegment>, text: &str) {
        if text.is_empty() {
            return;
        }
        let kind = if self.inside_think {
            ThinkSegmentKind::Reasoning
        } else {
            ThinkSegmentKind::Text
        };
        if let Some(previous) = segments.last_mut().filter(|segment| segment.kind == kind) {
            previous.text.push_str(text);
        } else {
            segments.push(ThinkSegment {
                kind,
                text: text.to_string(),
            });
        }
    }
}

fn pending_tag_suffix_len(text: &str, tag: &str) -> usize {
    let text = text.as_bytes();
    let tag = tag.as_bytes();
    let max_len = text.len().min(tag.len().saturating_sub(1));
    for len in (1..=max_len).rev() {
        if text.ends_with(&tag[..len]) {
            return len;
        }
    }
    0
}

fn normalize_system_messages(object: &mut serde_json::Map<String, Value>) {
    let Some(messages) = object.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };

    let mut system_parts = Vec::new();
    let mut rest = Vec::with_capacity(messages.len());

    for message in std::mem::take(messages) {
        if message.get("role").and_then(Value::as_str) == Some("system") {
            if let Some(content) = message.get("content").and_then(content_to_text) {
                if !content.is_empty() {
                    system_parts.push(content);
                }
            }
        } else {
            rest.push(message);
        }
    }

    if !system_parts.is_empty() {
        rest.insert(
            0,
            serde_json::json!({
                "role": "system",
                "content": system_parts.join("\n\n")
            }),
        );
    }

    *messages = rest;
}

fn content_to_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.trim().to_string()),
        Value::Array(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .or_else(|| part.get("input_text"))
                        .and_then(Value::as_str)
                })
                .filter(|text| !text.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            Some(text)
        }
        _ => None,
    }
}

fn clamp_f64_setting(object: &mut serde_json::Map<String, Value>, key: &str, fallback: f64) {
    let Some(value) = object.get(key) else {
        return;
    };
    let next = value
        .as_f64()
        .filter(|value| *value > 0.0 && *value <= 1.0)
        .unwrap_or(fallback);
    if let Some(number) = Number::from_f64(next) {
        object.insert(key.to_string(), Value::Number(number));
    } else {
        object.remove(key);
    }
}

fn clamp_u64_setting(object: &mut serde_json::Map<String, Value>, key: &str, max: u64) {
    let Some(value) = object.get(key) else {
        return;
    };
    let next = value
        .as_u64()
        .filter(|value| *value >= 1)
        .unwrap_or(max)
        .min(max);
    object.insert(key.to_string(), Value::Number(next.into()));
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn clamps_minimax_chat_settings_to_supported_ranges() {
        let mut adapter = MiniMaxBridgeAdapter::default();
        let mut request = json!({
            "model": "MiniMax-M2.7",
            "messages": [],
            "temperature": 0,
            "top_p": 0,
            "max_completion_tokens": 8192
        });

        adapter.prepare_chat_request(&mut request);

        assert_eq!(request["temperature"], 1.0);
        assert_eq!(request["top_p"], 0.95);
        assert_eq!(request["max_completion_tokens"], 2048);
    }

    #[test]
    fn folds_system_messages_into_one_leading_message() {
        let mut adapter = MiniMaxBridgeAdapter::default();
        let mut request = json!({
            "model": "MiniMax-M2.7",
            "messages": [
                { "role": "system", "content": "Global instructions." },
                { "role": "user", "content": "Hi" },
                { "role": "system", "content": "Developer instructions." },
                {
                    "role": "system",
                    "content": [{ "type": "text", "text": "Extra instructions." }]
                }
            ]
        });

        adapter.prepare_chat_request(&mut request);

        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(
            messages[0]["content"],
            "Global instructions.\n\nDeveloper instructions.\n\nExtra instructions."
        );
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn leaves_valid_minimax_chat_settings_unchanged() {
        let mut adapter = MiniMaxBridgeAdapter::default();
        let mut request = json!({
            "model": "MiniMax-M2.7",
            "messages": [],
            "temperature": 0.2,
            "top_p": 0.8,
            "max_completion_tokens": 1024
        });

        adapter.prepare_chat_request(&mut request);

        assert_eq!(request["temperature"], 0.2);
        assert_eq!(request["top_p"], 0.8);
        assert_eq!(request["max_completion_tokens"], 1024);
    }

    #[test]
    fn converts_minimax_think_tags_to_reasoning_events() {
        let mut adapter = MiniMaxBridgeAdapter::default();
        let mut events = vec![
            text_start(0),
            UniversalEvent::TextDelta {
                index: 0,
                text: "<think>Need ".to_string(),
            },
            UniversalEvent::TextDelta {
                index: 0,
                text: "math</think>\n\n221".to_string(),
            },
            response_done(),
        ];

        adapter.transform_upstream_events(&mut events);

        assert_eq!(joined_reasoning(&events), "Need math");
        assert_eq!(joined_text(&events), "\n\n221");
        assert!(!joined_text(&events).contains("<think>"));
        assert!(events.iter().any(|event| matches!(
            event,
            UniversalEvent::ContentStart {
                block: ContentBlock::Reasoning { .. },
                ..
            }
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            UniversalEvent::ContentStart {
                block: ContentBlock::Text { .. },
                ..
            }
        )));
    }

    #[test]
    fn handles_minimax_think_tags_split_across_stream_chunks() {
        let mut adapter = MiniMaxBridgeAdapter::default();
        let mut events = vec![
            text_start(0),
            UniversalEvent::TextDelta {
                index: 0,
                text: "<thi".to_string(),
            },
            UniversalEvent::TextDelta {
                index: 0,
                text: "nk>hidden</thi".to_string(),
            },
            UniversalEvent::TextDelta {
                index: 0,
                text: "nk>done".to_string(),
            },
            response_done(),
        ];

        adapter.transform_upstream_events(&mut events);

        assert_eq!(joined_reasoning(&events), "hidden");
        assert_eq!(joined_text(&events), "done");
    }

    #[test]
    fn preserves_plain_minimax_text_that_only_looks_like_partial_tag() {
        let mut adapter = MiniMaxBridgeAdapter::default();
        let mut events = vec![
            text_start(0),
            UniversalEvent::TextDelta {
                index: 0,
                text: "hello <thi".to_string(),
            },
            UniversalEvent::TextDelta {
                index: 0,
                text: "s is plain".to_string(),
            },
            response_done(),
        ];

        adapter.transform_upstream_events(&mut events);

        assert_eq!(joined_reasoning(&events), "");
        assert_eq!(joined_text(&events), "hello <this is plain");
    }

    #[test]
    fn parses_final_text_block_when_no_delta_was_seen() {
        let mut adapter = MiniMaxBridgeAdapter::default();
        let mut events = vec![
            text_start(0),
            UniversalEvent::ContentDone {
                index: 0,
                final_block: Some(ContentBlock::Text {
                    text: "<think>hidden</think>visible".to_string(),
                }),
            },
            response_done(),
        ];

        adapter.transform_upstream_events(&mut events);

        assert_eq!(joined_reasoning(&events), "hidden");
        assert_eq!(joined_text(&events), "visible");
    }

    #[test]
    fn remaps_following_content_blocks_after_splitting_think_tags() {
        let mut adapter = MiniMaxBridgeAdapter::default();
        let tool_block = ContentBlock::ToolCall {
            id: "call_1".to_string(),
            name: "lookup".to_string(),
            arguments: json!({}),
            extensions: Default::default(),
        };
        let mut events = vec![
            text_start(0),
            UniversalEvent::TextDelta {
                index: 0,
                text: "<think>hidden</think>visible".to_string(),
            },
            UniversalEvent::ContentDone {
                index: 0,
                final_block: Some(ContentBlock::Text {
                    text: "<think>hidden</think>visible".to_string(),
                }),
            },
            UniversalEvent::ContentStart {
                index: 1,
                block: tool_block.clone(),
            },
            UniversalEvent::ToolCallDelta {
                id: "call_1".to_string(),
                name: Some("lookup".to_string()),
                arguments_delta: "{}".to_string(),
            },
            UniversalEvent::ContentDone {
                index: 1,
                final_block: Some(tool_block),
            },
            response_done(),
        ];

        adapter.transform_upstream_events(&mut events);

        assert_eq!(content_start_indexes(&events), vec![0, 1, 2]);
        assert_eq!(content_done_indexes(&events), vec![0, 1, 2]);
    }

    fn content_start_indexes(events: &[UniversalEvent]) -> Vec<usize> {
        events
            .iter()
            .filter_map(|event| match event {
                UniversalEvent::ContentStart { index, .. } => Some(*index),
                _ => None,
            })
            .collect()
    }

    fn content_done_indexes(events: &[UniversalEvent]) -> Vec<usize> {
        events
            .iter()
            .filter_map(|event| match event {
                UniversalEvent::ContentDone { index, .. } => Some(*index),
                _ => None,
            })
            .collect()
    }

    fn text_start(index: usize) -> UniversalEvent {
        UniversalEvent::ContentStart {
            index,
            block: ContentBlock::Text {
                text: String::new(),
            },
        }
    }

    fn response_done() -> UniversalEvent {
        UniversalEvent::ResponseDone {
            usage: None,
            extensions: Default::default(),
        }
    }

    fn joined_text(events: &[UniversalEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                UniversalEvent::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn joined_reasoning(events: &[UniversalEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                UniversalEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }
}
