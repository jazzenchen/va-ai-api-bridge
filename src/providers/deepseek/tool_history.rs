use std::collections::{HashMap, HashSet};

use serde_json::{json, Map, Value};

use super::MISSING_TOOL_OUTPUT_FALLBACK;

pub(in crate::providers) fn repair_tool_call_history(
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

pub(in crate::providers) fn collect_tool_outputs_from_responses_input(
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

pub(in crate::providers) fn collect_tool_outputs_from_chat_request(
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
