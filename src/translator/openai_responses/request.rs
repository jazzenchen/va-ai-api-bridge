use serde_json::{json, Map, Value};

use crate::schema::openai::{ResponsesInput, ResponsesItem, ResponsesRequest};
use crate::translator::{common, openai};
use crate::{
    ApiBridgeError, ContentBlock, Result, Role, UniversalItem, UniversalRequest, WireProtocol,
};

pub(super) fn decode(raw: Value) -> Result<UniversalRequest> {
    let source_raw = raw.clone();
    let request: ResponsesRequest = serde_json::from_value(raw)
        .map_err(|error| ApiBridgeError::invalid_request(error.to_string()))?;

    let mut universal = UniversalRequest {
        model: request.model,
        instructions: openai::openai_content_to_blocks(request.instructions.as_ref()),
        tools: request
            .tools
            .iter()
            .filter_map(openai::openai_tool_from_value)
            .collect(),
        tool_choice: openai::tool_choice_from_openai_value(request.tool_choice.as_ref()),
        stream: request.stream.unwrap_or(false),
        generation: openai::generation_from_openai(
            request.temperature,
            request.top_p,
            request.max_output_tokens,
        ),
        reasoning: openai::reasoning_from_openai(request.reasoning),
        source: Some(common::source(WireProtocol::OpenAiResponses, source_raw)),
        ..UniversalRequest::default()
    };

    match request.input {
        Some(ResponsesInput::Text(text)) => universal.input.push(UniversalItem::Message {
            role: Role::User,
            id: None,
            content: vec![ContentBlock::Text { text }],
            extensions: common::empty_extensions(),
        }),
        Some(ResponsesInput::Items(items)) => {
            universal
                .input
                .extend(items.into_iter().map(item_to_universal));
        }
        Some(ResponsesInput::Raw(raw)) => universal.input.push(UniversalItem::Unknown { raw }),
        Some(ResponsesInput::Null) | None => {}
    }

    Ok(universal)
}

pub(super) fn encode(request: &UniversalRequest) -> Result<Value> {
    let mut body = Map::new();
    if let Some(model) = &request.model {
        body.insert("model".to_string(), Value::String(model.clone()));
    }
    if request.stream {
        body.insert("stream".to_string(), Value::Bool(true));
    }
    if let Some(temperature) = request.generation.temperature {
        body.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(top_p) = request.generation.top_p {
        body.insert("top_p".to_string(), json!(top_p));
    }
    if let Some(max_output_tokens) = request.generation.max_output_tokens {
        body.insert("max_output_tokens".to_string(), json!(max_output_tokens));
    }
    if let Some(reasoning) = &request.reasoning {
        let mut value = Map::new();
        if let Some(effort) = openai::openai_reasoning_effort(reasoning.effort.as_deref()) {
            value.insert("effort".to_string(), Value::String(effort.to_string()));
        }
        if !value.is_empty() {
            body.insert("reasoning".to_string(), Value::Object(value));
        }
    }
    if !request.instructions.is_empty() {
        if let Some(instructions) = openai::blocks_to_plain_text(&request.instructions) {
            body.insert("instructions".to_string(), Value::String(instructions));
        }
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".to_string(),
            Value::Array(
                request
                    .tools
                    .iter()
                    .map(openai::tool_to_openai_responses)
                    .collect(),
            ),
        );
    }
    if let Some(tool_choice) = &request.tool_choice {
        body.insert(
            "tool_choice".to_string(),
            openai::tool_choice_to_openai_responses(tool_choice),
        );
    }

    let input: Vec<Value> = request
        .input
        .iter()
        .map(universal_item_to_response_item)
        .collect::<Result<Vec<_>>>()?;
    if !input.is_empty() {
        body.insert("input".to_string(), Value::Array(input));
    }

    Ok(Value::Object(body))
}

fn item_to_universal(item: ResponsesItem) -> UniversalItem {
    match item.kind.as_deref() {
        Some("message") | None if item.role.is_some() => UniversalItem::Message {
            role: item
                .role
                .as_deref()
                .and_then(common::role_from_wire)
                .unwrap_or(Role::User),
            id: item.id,
            content: openai::openai_content_to_blocks(item.content.as_ref()),
            extensions: common::empty_extensions(),
        },
        Some("function_call") => UniversalItem::ToolCall {
            id: item.call_id.or(item.id).unwrap_or_default(),
            name: item.name.unwrap_or_default(),
            arguments: response_arguments_to_universal(item.arguments),
            extensions: common::empty_extensions(),
        },
        Some("function_call_output") => UniversalItem::ToolResult {
            tool_call_id: item.call_id.or(item.id).unwrap_or_default(),
            content: value_to_blocks(item.output.as_ref()),
            is_error: false,
            extensions: common::empty_extensions(),
        },
        Some("reasoning") => UniversalItem::Reasoning {
            id: item.id,
            text: item
                .content
                .as_ref()
                .and_then(|content| text_from_openai_content(Some(content))),
            encrypted: None,
            extensions: common::empty_extensions(),
        },
        _ => UniversalItem::Unknown {
            raw: serde_json::to_value(item).unwrap_or(Value::Null),
        },
    }
}

fn universal_item_to_response_item(item: &UniversalItem) -> Result<Value> {
    match item {
        UniversalItem::Message {
            role, id, content, ..
        } => {
            let mut object = Map::new();
            object.insert("type".to_string(), Value::String("message".to_string()));
            object.insert(
                "role".to_string(),
                Value::String(openai::role_to_openai(*role).to_string()),
            );
            if let Some(id) = id {
                object.insert("id".to_string(), Value::String(id.clone()));
            }
            let direction = if *role == Role::Assistant {
                openai::OpenAiResponsesContentDirection::Output
            } else {
                openai::OpenAiResponsesContentDirection::Input
            };
            if let Some(content) = openai::blocks_to_openai_responses_part_array(content, direction)
            {
                object.insert(
                    "content".to_string(),
                    serde_json::to_value(content)
                        .map_err(|error| ApiBridgeError::conversion(error.to_string()))?,
                );
            }
            Ok(Value::Object(object))
        }
        UniversalItem::ToolCall {
            id,
            name,
            arguments,
            ..
        } => Ok(json!({
            "type": "function_call",
            "call_id": id,
            "name": name,
            "arguments": common::stringify_arguments(arguments)
        })),
        UniversalItem::ToolResult {
            tool_call_id,
            content,
            ..
        } => Ok(json!({
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": blocks_to_value(content)
        })),
        UniversalItem::Reasoning {
            id,
            text,
            encrypted,
            ..
        } => Ok(json!({
            "type": "reasoning",
            "id": id,
            "content": text,
            "encrypted": encrypted
        })),
        UniversalItem::Unknown { raw } => Ok(raw.clone()),
    }
}

fn response_arguments_to_universal(arguments: Option<Value>) -> Value {
    match arguments {
        Some(Value::String(arguments)) => common::parse_arguments(Some(&arguments)),
        Some(arguments) => arguments,
        None => Value::Null,
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

fn value_to_blocks(value: Option<&Value>) -> Vec<ContentBlock> {
    match value {
        Some(Value::String(text)) => vec![ContentBlock::Text { text: text.clone() }],
        Some(Value::Array(values)) => values
            .iter()
            .cloned()
            .map(|raw| ContentBlock::Unknown { raw })
            .collect(),
        Some(value) => vec![ContentBlock::Unknown { raw: value.clone() }],
        None => Vec::new(),
    }
}

fn blocks_to_value(blocks: &[ContentBlock]) -> Value {
    match blocks {
        [ContentBlock::Text { text }] => Value::String(text.clone()),
        blocks => Value::Array(
            blocks
                .iter()
                .map(|block| serde_json::to_value(block).unwrap_or(Value::Null))
                .collect(),
        ),
    }
}
