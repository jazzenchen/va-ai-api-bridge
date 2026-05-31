use std::collections::{hash_map::DefaultHasher, HashMap, HashSet};
use std::hash::{Hash, Hasher};

use serde_json::{json, Map, Value};

use super::reasoning_blob::decode_reasoning_content;

use super::{DeepSeekBridgeSettings, ProviderRequestSource};

const MISSING_REASONING_CONTENT_FALLBACK: &str =
    "Previous DeepSeek reasoning content is unavailable from the local bridge.";
const MISSING_TOOL_OUTPUT_FALLBACK: &str = "Tool output unavailable from the local bridge.";

#[derive(Debug, Default)]
pub(super) struct RequestReasoning {
    by_call_id: HashMap<String, String>,
    by_message: HashMap<String, String>,
}

#[derive(Debug, Default)]
struct ReasoningReplayAccumulator {
    active_reasoning: Option<String>,
    pending_call_ids: Vec<String>,
}

impl ReasoningReplayAccumulator {
    fn reset(&mut self) {
        self.active_reasoning = None;
        self.pending_call_ids.clear();
    }

    fn observe_item(
        &mut self,
        reasoning: &mut RequestReasoning,
        item: &Map<String, Value>,
    ) -> usize {
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => self.observe_reasoning(reasoning, item),
            Some("function_call") => self.observe_function_call(reasoning, item),
            Some("function_call_output") => {
                self.reset();
                0
            }
            None | Some("message") => match item.get("role").and_then(Value::as_str) {
                Some("assistant") => self.observe_assistant_message(reasoning, item),
                _ => {
                    self.reset();
                    0
                }
            },
            _ => {
                self.reset();
                0
            }
        }
    }

    fn observe_reasoning(
        &mut self,
        reasoning: &mut RequestReasoning,
        item: &Map<String, Value>,
    ) -> usize {
        let Some(reasoning_content) = reasoning_content_from_item(item) else {
            return 0;
        };

        let mut stored_count = 0usize;
        for call_id in self.pending_call_ids.drain(..) {
            store_reasoning(reasoning, &call_id, &reasoning_content);
            stored_count += 1;
        }
        self.active_reasoning = Some(reasoning_content);
        stored_count
    }

    fn observe_function_call(
        &mut self,
        reasoning: &mut RequestReasoning,
        item: &Map<String, Value>,
    ) -> usize {
        let Some(call_id) = item
            .get("call_id")
            .or_else(|| item.get("id"))
            .and_then(Value::as_str)
        else {
            return 0;
        };

        if let Some(reasoning_content) = self.active_reasoning.as_deref() {
            store_reasoning(reasoning, call_id, reasoning_content);
            1
        } else {
            self.pending_call_ids.push(call_id.to_string());
            0
        }
    }

    fn observe_assistant_message(
        &mut self,
        reasoning: &mut RequestReasoning,
        item: &Map<String, Value>,
    ) -> usize {
        let Some(reasoning_content) = self.active_reasoning.as_deref() else {
            return 0;
        };

        // OpenAI Responses history may place an empty assistant message between
        // a reasoning item and the following function_call; keep the reasoning
        // active until we see a real assistant message or another boundary.
        if store_reasoning_for_message_item(reasoning, reasoning_content, item) {
            self.reset();
            1
        } else {
            0
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeepSeekBridgeAdapter {
    settings: DeepSeekBridgeSettings,
}

impl DeepSeekBridgeAdapter {
    pub fn new(settings: DeepSeekBridgeSettings) -> Self {
        Self { settings }
    }

    pub fn prepare_chat_request(
        &mut self,
        source: ProviderRequestSource,
        original_request: &Value,
        chat_request: &mut Value,
    ) {
        if source == ProviderRequestSource::AnthropicMessages {
            strip_anthropic_reasoning_content_blocks(chat_request);
        }

        let tool_outputs = self.collect_tool_outputs(original_request, chat_request);
        repair_tool_call_history(&tool_outputs, chat_request);

        if self.should_replay_reasoning_content(source) {
            let mut reasoning = RequestReasoning::default();
            match source {
                ProviderRequestSource::OpenAiResponses => {
                    collect_reasoning_from_responses_input(&mut reasoning, original_request);
                }
                ProviderRequestSource::AnthropicMessages => {
                    collect_reasoning_from_anthropic_input(&mut reasoning, original_request);
                }
                ProviderRequestSource::GeminiGenerateContent => {
                    collect_reasoning_from_gemini_input(&mut reasoning, original_request);
                }
                ProviderRequestSource::OpenAiChat => {}
            }
            inject_reasoning_content(&reasoning, chat_request, MISSING_REASONING_CONTENT_FALLBACK);
        }

        let Some(request) = chat_request.as_object_mut() else {
            return;
        };

        request.insert(
            "thinking".to_string(),
            json!({
                "type": if self.settings.thinking {
                    "enabled"
                } else {
                    "disabled"
                },
            }),
        );
    }

    pub fn prepare_anthropic_request(&mut self, request: &mut Value) {
        repair_anthropic_thinking_tool_use_order(request);
    }

    fn should_replay_reasoning_content(&self, source: ProviderRequestSource) -> bool {
        self.settings.thinking
            && self.settings.replay_reasoning_content
            && source.supports_deepseek_reasoning_replay()
    }

    fn collect_tool_outputs(
        &self,
        original_request: &Value,
        chat_request: &Value,
    ) -> HashMap<String, String> {
        let mut outputs = HashMap::new();
        collect_tool_outputs_from_responses_input(original_request, &mut outputs);
        collect_tool_outputs_from_chat_request(chat_request, &mut outputs);
        outputs
    }
}

fn repair_anthropic_thinking_tool_use_order(request: &mut Value) {
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };

    for message in messages {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        move_first_thinking_before_leading_tool_uses(blocks);
    }
}

fn move_first_thinking_before_leading_tool_uses(blocks: &mut Vec<Value>) {
    let Some(thinking_index) = blocks.iter().position(is_anthropic_thinking_block) else {
        return;
    };
    if thinking_index == 0
        || !blocks[..thinking_index]
            .iter()
            .all(is_anthropic_tool_use_block)
    {
        return;
    }

    let thinking = blocks.remove(thinking_index);
    blocks.insert(0, thinking);
}

fn is_anthropic_thinking_block(block: &Value) -> bool {
    anthropic_block_type(block) == Some("thinking")
        && block
            .get("thinking")
            .and_then(Value::as_str)
            .is_some_and(|thinking| !thinking.is_empty())
}

fn is_anthropic_tool_use_block(block: &Value) -> bool {
    anthropic_block_type(block) == Some("tool_use")
}

fn anthropic_block_type(block: &Value) -> Option<&str> {
    block.get("type").and_then(Value::as_str)
}

pub(super) fn repair_tool_call_history(
    tool_outputs: &HashMap<String, String>,
    request: &mut Value,
) {
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };

    let original_messages = std::mem::take(messages);
    let mut repaired_messages = Vec::with_capacity(original_messages.len());
    let mut satisfied_tool_call_ids = HashSet::<String>::new();
    let mut index = 0usize;

    while index < original_messages.len() {
        let message = &original_messages[index];
        let tool_call_ids = assistant_tool_call_ids(message);
        if tool_call_ids.is_empty() {
            if is_empty_assistant_message(message) {
                index += 1;
                continue;
            }
            if let Some((tool_call_id, _)) = normalized_tool_message(message, tool_outputs) {
                satisfied_tool_call_ids.insert(tool_call_id);
                index += 1;
                continue;
            }
            repaired_messages.push(message.clone());
            index += 1;
            continue;
        }

        let expected_ids = tool_call_ids.iter().cloned().collect::<HashSet<_>>();
        let mut present_ids = HashSet::<String>::new();
        repaired_messages.push(message.clone());
        index += 1;

        while index < original_messages.len() {
            let next_message = &original_messages[index];
            if is_empty_assistant_message(next_message) {
                index += 1;
                continue;
            }
            let Some((tool_call_id, tool_message)) =
                normalized_tool_message(next_message, tool_outputs)
            else {
                break;
            };

            if expected_ids.contains(&tool_call_id) && present_ids.insert(tool_call_id.clone()) {
                repaired_messages.push(tool_message);
                satisfied_tool_call_ids.insert(tool_call_id);
            }
            index += 1;
        }

        for tool_call_id in tool_call_ids {
            if present_ids.contains(&tool_call_id) {
                continue;
            }
            let content = tool_outputs
                .get(&tool_call_id)
                .map(String::as_str)
                .unwrap_or(MISSING_TOOL_OUTPUT_FALLBACK);
            repaired_messages.push(tool_message_for_call_id(&tool_call_id, content));
            satisfied_tool_call_ids.insert(tool_call_id);
        }

        while index < original_messages.len() {
            let next_message = &original_messages[index];
            if is_empty_assistant_message(next_message) {
                index += 1;
                continue;
            }
            let Some((tool_call_id, _)) = normalized_tool_message(next_message, tool_outputs)
            else {
                break;
            };
            if satisfied_tool_call_ids.contains(&tool_call_id) {
                index += 1;
                continue;
            }
            break;
        }
    }

    *messages = repaired_messages;
}

fn assistant_tool_call_ids(message: &Value) -> Vec<String> {
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return Vec::new();
    }
    let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) else {
        return Vec::new();
    };
    tool_calls
        .iter()
        .filter_map(|tool_call| tool_call.get("id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn is_empty_assistant_message(message: &Value) -> bool {
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return false;
    }
    if message.get("tool_calls").is_some() {
        return false;
    }
    if message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .is_some_and(|content| !content.trim().is_empty())
    {
        return false;
    }
    is_empty_content(message.get("content"))
}

fn is_empty_content(content: Option<&Value>) -> bool {
    match content {
        Some(Value::String(content)) => content.trim().is_empty(),
        Some(Value::Array(parts)) => {
            parts.is_empty()
                || parts.iter().all(|part| {
                    part.as_object()
                        .and_then(|part| part.get("text"))
                        .and_then(Value::as_str)
                        .is_some_and(|text| text.trim().is_empty())
                })
        }
        Some(Value::Null) | None => true,
        Some(_) => false,
    }
}

fn normalized_tool_message(
    message: &Value,
    tool_outputs: &HashMap<String, String>,
) -> Option<(String, Value)> {
    let object = message.as_object()?;
    if object.get("role").and_then(Value::as_str) == Some("tool") {
        let tool_call_id = tool_call_id_from_tool_message(object)?;
        let content = object
            .get("content")
            .and_then(value_to_string)
            .or_else(|| tool_outputs.get(tool_call_id).cloned())
            .unwrap_or_default();
        return Some((
            tool_call_id.to_string(),
            tool_message_for_call_id(tool_call_id, &content),
        ));
    }

    if object.get("type").and_then(Value::as_str) == Some("function_call_output") {
        let tool_call_id = call_id_from_function_call_output(object)?;
        return Some((
            tool_call_id.to_string(),
            tool_message_for_call_id(tool_call_id, &tool_output_content(object)),
        ));
    }

    None
}

fn tool_call_id_from_tool_message(message: &Map<String, Value>) -> Option<&str> {
    message
        .get("tool_call_id")
        .or_else(|| message.get("call_id"))
        .or_else(|| message.get("id"))
        .and_then(Value::as_str)
}

fn call_id_from_function_call_output(item: &Map<String, Value>) -> Option<&str> {
    item.get("call_id")
        .or_else(|| item.get("tool_call_id"))
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
}

fn tool_output_content(item: &Map<String, Value>) -> String {
    item.get("output")
        .or_else(|| item.get("content"))
        .and_then(value_to_string)
        .unwrap_or_default()
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(content) => Some(content.clone()),
        Value::Null => Some(String::new()),
        other => serde_json::to_string(other).ok(),
    }
}

fn tool_message_for_call_id(tool_call_id: &str, content: &str) -> Value {
    json!({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    })
}

pub(super) fn collect_tool_outputs_from_responses_input(
    responses_request: &Value,
    outputs: &mut HashMap<String, String>,
) {
    let Some(items) = responses_request.get("input").and_then(Value::as_array) else {
        return;
    };
    for item in items {
        let Some(item) = item.as_object() else {
            continue;
        };
        if item.get("type").and_then(Value::as_str) != Some("function_call_output") {
            continue;
        }
        if let Some(call_id) = call_id_from_function_call_output(item) {
            outputs.insert(call_id.to_string(), tool_output_content(item));
        }
    }
}

pub(super) fn collect_tool_outputs_from_chat_request(
    request: &Value,
    outputs: &mut HashMap<String, String>,
) {
    let Some(messages) = request.get("messages").and_then(Value::as_array) else {
        return;
    };
    for message in messages {
        let Some(message) = message.as_object() else {
            continue;
        };
        if message.get("role").and_then(Value::as_str) == Some("tool") {
            if let Some(tool_call_id) = tool_call_id_from_tool_message(message) {
                let content = message
                    .get("content")
                    .and_then(value_to_string)
                    .unwrap_or_default();
                outputs.insert(tool_call_id.to_string(), content);
            }
            continue;
        }
        if message.get("type").and_then(Value::as_str) == Some("function_call_output") {
            if let Some(call_id) = call_id_from_function_call_output(message) {
                outputs.insert(call_id.to_string(), tool_output_content(message));
            }
        }
    }
}

pub(super) fn inject_reasoning_content(
    reasoning: &RequestReasoning,
    chat_request: &mut Value,
    missing_reasoning_content_fallback: &str,
) {
    let Some(messages) = chat_request
        .get_mut("messages")
        .and_then(Value::as_array_mut)
    else {
        return;
    };

    for message in messages {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        if message.get("reasoning_content").is_some() {
            continue;
        }
        let tool_calls = message.get("tool_calls").and_then(Value::as_array);
        let reasoning_content = if let Some(tool_calls) = tool_calls {
            tool_calls
                .iter()
                .filter_map(|tool_call| tool_call.get("id").and_then(Value::as_str))
                .find_map(|call_id| lookup_reasoning(reasoning, call_id))
                .unwrap_or_else(|| missing_reasoning_content_fallback.to_string())
        } else {
            let Some(reasoning_content) = lookup_reasoning_for_message(reasoning, message) else {
                continue;
            };
            reasoning_content
        };
        if let Some(obj) = message.as_object_mut() {
            obj.insert(
                "reasoning_content".to_string(),
                Value::String(reasoning_content),
            );
        }
    }
}

pub(super) fn collect_reasoning_from_responses_input(
    reasoning: &mut RequestReasoning,
    responses_request: &Value,
) {
    let Some(items) = responses_request.get("input").and_then(Value::as_array) else {
        return;
    };

    let mut accumulator = ReasoningReplayAccumulator::default();

    for item in items {
        let Some(obj) = item.as_object() else {
            accumulator.reset();
            continue;
        };
        accumulator.observe_item(reasoning, obj);
    }
}

pub(super) fn collect_reasoning_from_anthropic_input(
    reasoning: &mut RequestReasoning,
    anthropic_request: &Value,
) {
    let Some(messages) = anthropic_request.get("messages").and_then(Value::as_array) else {
        return;
    };

    for message in messages {
        let Some(message) = message.as_object() else {
            continue;
        };
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        collect_reasoning_from_anthropic_assistant_message(reasoning, message);
    }
}

pub(super) fn collect_reasoning_from_gemini_input(
    reasoning: &mut RequestReasoning,
    gemini_request: &Value,
) {
    let Some(contents) = gemini_request.get("contents").and_then(Value::as_array) else {
        return;
    };

    for content in contents {
        let Some(content) = content.as_object() else {
            continue;
        };
        if content.get("role").and_then(Value::as_str) != Some("model") {
            continue;
        }
        collect_reasoning_from_gemini_model_content(reasoning, content);
    }
}

fn collect_reasoning_from_gemini_model_content(
    reasoning: &mut RequestReasoning,
    content: &Map<String, Value>,
) -> usize {
    let Some(parts) = content.get("parts").and_then(Value::as_array) else {
        return 0;
    };

    let mut active_reasoning: Option<String> = None;
    let mut text_parts = Vec::new();
    let mut stored_count = 0usize;

    for part in parts {
        let Some(part) = part.as_object() else {
            continue;
        };
        if is_gemini_thought_part(part) {
            if let Some(reasoning_text) = gemini_thought_text(part) {
                active_reasoning = Some(reasoning_text);
            }
            continue;
        }
        if let Some(function_call) = part.get("functionCall").and_then(Value::as_object) {
            if let (Some(reasoning_content), Some(call_id)) = (
                active_reasoning.as_deref(),
                gemini_function_call_id(function_call),
            ) {
                store_reasoning(reasoning, call_id, reasoning_content);
                stored_count += 1;
            }
            continue;
        }
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            text_parts.push(text);
        }
    }

    if let Some(reasoning_content) = active_reasoning.as_deref() {
        let text = text_parts.join("");
        if !text.is_empty() {
            let mut message = Map::new();
            message.insert("content".to_string(), Value::String(text));
            if store_reasoning_for_message_item(reasoning, reasoning_content, &message) {
                stored_count += 1;
            }
        }
    }

    stored_count
}

fn gemini_function_call_id(function_call: &Map<String, Value>) -> Option<&str> {
    function_call
        .get("id")
        .and_then(Value::as_str)
        .or_else(|| function_call.get("name").and_then(Value::as_str))
}

fn is_gemini_thought_part(part: &Map<String, Value>) -> bool {
    part.get("thought")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn gemini_thought_text(part: &Map<String, Value>) -> Option<String> {
    part.get("text")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

fn collect_reasoning_from_anthropic_assistant_message(
    reasoning: &mut RequestReasoning,
    message: &Map<String, Value>,
) -> usize {
    let Some(blocks) = message.get("content").and_then(Value::as_array) else {
        return 0;
    };

    let mut active_reasoning: Option<String> = None;
    let mut text_parts = Vec::new();
    let mut stored_count = 0usize;

    for block in blocks {
        let Some(block) = block.as_object() else {
            continue;
        };
        match block.get("type").and_then(Value::as_str) {
            Some("thinking" | "redacted_thinking") => {
                active_reasoning = anthropic_reasoning_content_from_block(block);
            }
            Some("tool_use") => {
                if let (Some(reasoning_content), Some(call_id)) = (
                    active_reasoning.as_deref(),
                    block.get("id").and_then(Value::as_str),
                ) {
                    store_reasoning(reasoning, call_id, reasoning_content);
                    stored_count += 1;
                }
            }
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    text_parts.push(text);
                }
            }
            _ => {}
        }
    }

    if let Some(reasoning_content) = active_reasoning.as_deref() {
        let text = text_parts.join("");
        if !text.is_empty() {
            let mut message = Map::new();
            message.insert("content".to_string(), Value::String(text));
            if store_reasoning_for_message_item(reasoning, reasoning_content, &message) {
                stored_count += 1;
            }
        }
    }

    stored_count
}

fn anthropic_reasoning_content_from_block(block: &Map<String, Value>) -> Option<String> {
    block
        .get("thinking")
        .or_else(|| block.get("text"))
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())
        .map(str::to_string)
}

fn reasoning_content_from_item(item: &Map<String, Value>) -> Option<String> {
    if let Some(reasoning_content) = item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .and_then(decode_reasoning_content)
        .filter(|content| !content.is_empty())
    {
        return Some(reasoning_content);
    }

    let content = item.get("content").and_then(Value::as_array)?;
    let text = content
        .iter()
        .filter_map(|part| {
            let part = part.as_object()?;
            if part.get("type").and_then(Value::as_str) != Some("reasoning_text") {
                return None;
            }
            part.get("text").and_then(Value::as_str)
        })
        .collect::<Vec<_>>()
        .join("");
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn store_reasoning_for_message_item(
    reasoning: &mut RequestReasoning,
    reasoning_content: &str,
    message: &Map<String, Value>,
) -> bool {
    if reasoning_content.is_empty() {
        return false;
    }
    let Some(content_fingerprint) = message_content_fingerprint(message.get("content")) else {
        return false;
    };
    reasoning
        .by_message
        .insert(content_fingerprint, reasoning_content.to_string());
    true
}

fn lookup_reasoning_for_message(reasoning: &RequestReasoning, message: &Value) -> Option<String> {
    let content_fingerprint = message_content_fingerprint(message.get("content"))?;
    reasoning.by_message.get(&content_fingerprint).cloned()
}

fn message_content_fingerprint(content: Option<&Value>) -> Option<String> {
    let text = match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        Some(Value::Null) | None => String::new(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
    };
    if text.is_empty() {
        None
    } else {
        Some(message_text_fingerprint(&text))
    }
}

fn message_text_fingerprint(text: &str) -> String {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{}:{:016x}", text.len(), hasher.finish())
}

fn store_reasoning(reasoning: &mut RequestReasoning, call_id: &str, reasoning_content: &str) {
    if reasoning_content.is_empty() {
        return;
    }
    reasoning
        .by_call_id
        .insert(call_id.to_string(), reasoning_content.to_string());
}

fn lookup_reasoning(reasoning: &RequestReasoning, call_id: &str) -> Option<String> {
    reasoning.by_call_id.get(call_id).cloned()
}

pub(super) fn strip_anthropic_reasoning_content_blocks(chat_request: &mut Value) {
    let Some(messages) = chat_request
        .get_mut("messages")
        .and_then(Value::as_array_mut)
    else {
        return;
    };

    for message in messages {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        parts.retain(|part| !is_anthropic_reasoning_content_part(part));
    }
}

fn is_anthropic_reasoning_content_part(part: &Value) -> bool {
    let direct_type = part.get("type").and_then(Value::as_str);
    if matches!(
        direct_type,
        Some("thinking" | "redacted_thinking" | "reasoning")
    ) {
        return true;
    }

    direct_type == Some("unknown")
        && matches!(
            part.get("raw")
                .and_then(|raw| raw.get("type"))
                .and_then(Value::as_str),
            Some("thinking" | "redacted_thinking" | "reasoning")
        )
}

#[cfg(test)]
mod tests;
