use serde_json::{json, Map, Value};

use crate::translator::common;
use crate::{
    ApiBridgeError, ContentBlock, Result, Role, UniversalEvent, UniversalItem, UniversalResponse,
};

use super::shared::{
    blocks_to_gemini_parts, finish_reason_from_gemini, finish_reason_to_gemini,
    function_call_part_with_signature, gemini_part_to_blocks, thought_signature_from_extensions,
    usage_from_gemini, usage_to_gemini,
};

pub fn encode_response(events: &[UniversalEvent]) -> Value {
    let response = UniversalResponse::from_events(events);
    let mut candidate = Map::new();
    candidate.insert(
        "content".to_string(),
        json!({
            "role": "model",
            "parts": response_parts(&response),
        }),
    );
    if let Some(finish_reason) = response.finish_reason {
        candidate.insert(
            "finishReason".to_string(),
            Value::String(finish_reason_to_gemini(finish_reason).to_string()),
        );
    }

    let mut out = Map::new();
    if let Some(id) = response.id {
        out.insert("responseId".to_string(), Value::String(id));
    }
    out.insert(
        "candidates".to_string(),
        Value::Array(vec![Value::Object(candidate)]),
    );
    if let Some(usage) = response.usage {
        out.insert("usageMetadata".to_string(), usage_to_gemini(&usage));
    }
    if let Some(model) = response.model {
        out.insert("modelVersion".to_string(), Value::String(model));
    }
    Value::Object(out)
}

pub(super) fn decode_response(raw: Value) -> Result<Vec<UniversalEvent>> {
    let mut events = vec![UniversalEvent::ResponseStart {
        id: raw
            .get("responseId")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        model: raw
            .get("modelVersion")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        extensions: common::empty_extensions(),
    }];
    decode_candidates(&raw, &mut events, true)?;
    events.push(UniversalEvent::ResponseDone {
        usage: usage_from_gemini(raw.get("usageMetadata")),
        extensions: common::empty_extensions(),
    });
    Ok(events)
}

pub(super) fn decode_candidates(
    raw: &Value,
    events: &mut Vec<UniversalEvent>,
    final_blocks: bool,
) -> Result<()> {
    let candidates = raw
        .get("candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiBridgeError::invalid_response("Gemini response missing candidates"))?;
    let Some(candidate) = candidates.first() else {
        return Ok(());
    };
    events.push(UniversalEvent::MessageStart {
        id: "gemini-message-0".to_string(),
        role: Role::Assistant,
        extensions: common::empty_extensions(),
    });
    let parts = candidate
        .get("content")
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for (index, part) in parts.iter().enumerate() {
        for block in gemini_part_to_blocks(part) {
            events.push(UniversalEvent::ContentStart {
                index,
                block: block.clone(),
            });
            match &block {
                ContentBlock::Text { text } => events.push(UniversalEvent::TextDelta {
                    index,
                    text: text.clone(),
                }),
                ContentBlock::Reasoning {
                    text: Some(text), ..
                } => events.push(UniversalEvent::ReasoningDelta {
                    index,
                    text: text.clone(),
                }),
                ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                    ..
                } => events.push(UniversalEvent::ToolCallDelta {
                    id: id.clone(),
                    name: Some(name.clone()),
                    arguments_delta: super::shared::stringify_json(arguments.clone()),
                }),
                _ => {}
            }
            events.push(UniversalEvent::ContentDone {
                index,
                final_block: final_blocks.then_some(block),
            });
        }
    }
    let finish_reason = candidate
        .get("finishReason")
        .and_then(Value::as_str)
        .map(finish_reason_from_gemini);
    if finish_reason.is_some() {
        events.push(UniversalEvent::MessageDone {
            finish_reason,
            usage: usage_from_gemini(raw.get("usageMetadata")),
            extensions: common::empty_extensions(),
        });
    }
    Ok(())
}

fn response_parts(response: &UniversalResponse) -> Vec<Value> {
    let mut parts = Vec::new();
    for item in &response.output {
        match item {
            UniversalItem::Message { content, .. } => parts.extend(blocks_to_gemini_parts(content)),
            UniversalItem::ToolCall {
                id,
                name,
                arguments,
                extensions,
                ..
            } => parts.push(function_call_part_with_signature(
                Some(id),
                name,
                arguments.clone(),
                thought_signature_from_extensions(extensions),
            )),
            UniversalItem::Reasoning {
                text: Some(text), ..
            } if !text.is_empty() => parts.push(json!({ "thought": true, "text": text })),
            _ => {}
        }
    }
    if parts.is_empty() {
        parts.push(json!({ "text": "" }));
    }
    parts
}
