use serde_json::{json, Map, Value};

use crate::schema::{anthropic, openai};
use crate::{
    ContentBlock, DecodeState, EncodeState, Extensions, FinishReason, GenerationConfig,
    ReasoningConfig, Role, SourcePayload, ToolChoice, UniversalEvent, UniversalTool, Usage,
    WireEvent, WireProtocol,
};

pub(crate) fn empty_extensions() -> Extensions {
    Extensions::new()
}

pub(crate) fn source(protocol: WireProtocol, raw: Value) -> SourcePayload {
    SourcePayload {
        protocol,
        raw: Some(raw),
    }
}

pub(crate) fn role_from_wire(role: &str) -> Option<Role> {
    match role {
        "developer" | "system" => Some(Role::System),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        "tool" => Some(Role::Tool),
        _ => None,
    }
}

pub(crate) fn role_to_openai(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

pub(crate) fn role_to_anthropic(role: Role) -> &'static str {
    match role {
        Role::System | Role::User | Role::Tool => "user",
        Role::Assistant => "assistant",
    }
}

pub(crate) fn finish_from_openai(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("stop") => Some(FinishReason::Stop),
        Some("length") => Some(FinishReason::Length),
        Some("tool_calls") | Some("function_call") => Some(FinishReason::ToolCall),
        Some("content_filter") => Some(FinishReason::ContentFilter),
        Some(_) => Some(FinishReason::Unknown),
        None => None,
    }
}

pub(crate) fn finish_from_anthropic(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("end_turn") | Some("stop_sequence") => Some(FinishReason::Stop),
        Some("max_tokens") => Some(FinishReason::Length),
        Some("tool_use") => Some(FinishReason::ToolCall),
        Some(_) => Some(FinishReason::Unknown),
        None => None,
    }
}

pub(crate) fn openai_usage_to_universal(usage: Option<&openai::OpenAiUsage>) -> Option<Usage> {
    usage.map(|usage| Usage {
        input_tokens: usage.input_tokens.or(usage.prompt_tokens),
        output_tokens: usage.output_tokens.or(usage.completion_tokens),
        total_tokens: usage.total_tokens,
    })
}

pub(crate) fn anthropic_usage_to_universal(
    usage: Option<&anthropic::AnthropicUsage>,
) -> Option<Usage> {
    usage.map(|usage| {
        let input_tokens = usage.input_tokens.map(|tokens| {
            tokens
                + usage.cache_creation_input_tokens.unwrap_or(0)
                + usage.cache_read_input_tokens.unwrap_or(0)
        });
        Usage {
            input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: input_tokens
                .zip(usage.output_tokens)
                .map(|(input, output)| input + output),
        }
    })
}

pub(crate) fn generation_from_openai(
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_output_tokens: Option<u64>,
) -> GenerationConfig {
    GenerationConfig {
        temperature,
        top_p,
        max_output_tokens,
        extensions: empty_extensions(),
    }
}

pub(crate) fn reasoning_from_openai(
    reasoning: Option<openai::OpenAiReasoning>,
) -> Option<ReasoningConfig> {
    reasoning.map(|reasoning| ReasoningConfig {
        effort: reasoning.effort,
        budget_tokens: None,
        visible: None,
        extensions: value_extensions(reasoning.extra),
    })
}

pub(crate) fn value_extensions(extra: impl IntoIterator<Item = (String, Value)>) -> Extensions {
    extra.into_iter().collect()
}

pub(crate) fn parse_arguments(arguments: Option<&str>) -> Value {
    arguments
        .filter(|arguments| !arguments.trim().is_empty())
        .and_then(|arguments| serde_json::from_str(arguments).ok())
        .unwrap_or(Value::Null)
}

pub(crate) fn stringify_arguments(arguments: &Value) -> String {
    match arguments {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        value => serde_json::to_string(value).unwrap_or_default(),
    }
}

pub(crate) fn tool_choice_from_value(value: Option<&Value>) -> Option<ToolChoice> {
    match value {
        Some(Value::String(value)) => match value.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "none" => Some(ToolChoice::None),
            "required" | "any" => Some(ToolChoice::Required),
            _ => None,
        },
        Some(Value::Object(object)) => {
            let kind = object.get("type").and_then(Value::as_str);
            if matches!(kind, Some("function") | Some("tool")) {
                object
                    .get("function")
                    .and_then(Value::as_object)
                    .and_then(|function| function.get("name"))
                    .or_else(|| object.get("name"))
                    .and_then(Value::as_str)
                    .map(|name| ToolChoice::Tool {
                        name: name.to_string(),
                    })
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(crate) fn tool_choice_to_openai(value: &ToolChoice) -> Value {
    match value {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool { name } => json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

pub(crate) fn tool_choice_to_anthropic(value: &ToolChoice) -> Value {
    match value {
        ToolChoice::Auto => json!({ "type": "auto" }),
        ToolChoice::None => json!({ "type": "none" }),
        ToolChoice::Required => json!({ "type": "any" }),
        ToolChoice::Tool { name } => json!({
            "type": "tool",
            "name": name
        }),
    }
}

pub(crate) fn openai_tool_from_value(value: &Value) -> Option<UniversalTool> {
    let object = value.as_object()?;
    let function = object
        .get("function")
        .and_then(Value::as_object)
        .unwrap_or(object);
    let name = function.get("name").and_then(Value::as_str)?;
    Some(UniversalTool {
        name: name.to_string(),
        description: function
            .get("description")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        input_schema: function
            .get("parameters")
            .or_else(|| function.get("input_schema"))
            .cloned(),
        extensions: empty_extensions(),
    })
}

pub(crate) fn anthropic_tool_from_value(value: &Value) -> Option<UniversalTool> {
    let object = value.as_object()?;
    let name = object.get("name").and_then(Value::as_str)?;
    Some(UniversalTool {
        name: name.to_string(),
        description: object
            .get("description")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        input_schema: object.get("input_schema").cloned(),
        extensions: empty_extensions(),
    })
}

pub(crate) fn tool_to_openai_chat(tool: &UniversalTool) -> Value {
    let mut function = Map::new();
    function.insert("name".to_string(), Value::String(tool.name.clone()));
    if let Some(description) = &tool.description {
        function.insert(
            "description".to_string(),
            Value::String(description.clone()),
        );
    }
    if let Some(input_schema) = &tool.input_schema {
        function.insert("parameters".to_string(), input_schema.clone());
    }
    json!({
        "type": "function",
        "function": function
    })
}

pub(crate) fn tool_to_openai_responses(tool: &UniversalTool) -> Value {
    let mut object = Map::new();
    object.insert("type".to_string(), Value::String("function".to_string()));
    object.insert("name".to_string(), Value::String(tool.name.clone()));
    if let Some(description) = &tool.description {
        object.insert(
            "description".to_string(),
            Value::String(description.clone()),
        );
    }
    if let Some(input_schema) = &tool.input_schema {
        object.insert("parameters".to_string(), input_schema.clone());
    }
    Value::Object(object)
}

pub(crate) fn tool_to_anthropic(tool: &UniversalTool) -> Value {
    let mut object = Map::new();
    object.insert("name".to_string(), Value::String(tool.name.clone()));
    if let Some(description) = &tool.description {
        object.insert(
            "description".to_string(),
            Value::String(description.clone()),
        );
    }
    if let Some(input_schema) = &tool.input_schema {
        object.insert("input_schema".to_string(), input_schema.clone());
    }
    Value::Object(object)
}

pub(crate) fn openai_content_to_blocks(
    content: Option<&openai::OpenAiContent>,
) -> Vec<ContentBlock> {
    let Some(content) = content else {
        return Vec::new();
    };
    match content {
        openai::OpenAiContent::Text(text) => vec![ContentBlock::Text { text: text.clone() }],
        openai::OpenAiContent::Parts(parts) => parts.iter().map(openai_part_to_block).collect(),
        openai::OpenAiContent::Null => Vec::new(),
        openai::OpenAiContent::Raw(raw) => vec![ContentBlock::Unknown { raw: raw.clone() }],
    }
}

fn openai_part_to_block(part: &openai::OpenAiContentPart) -> ContentBlock {
    match part.kind.as_str() {
        "text" => ContentBlock::Text {
            text: part.text.clone().unwrap_or_default(),
        },
        "input_text" => ContentBlock::Text {
            text: part
                .input_text
                .clone()
                .or_else(|| part.text.clone())
                .unwrap_or_default(),
        },
        "output_text" => ContentBlock::Text {
            text: part
                .output_text
                .clone()
                .or_else(|| part.text.clone())
                .unwrap_or_default(),
        },
        "image_url" => match &part.image_url {
            Some(image_url) => ContentBlock::Image {
                media_type: None,
                url: Some(image_url.url.clone()),
                data: None,
                extensions: empty_extensions(),
            },
            None => ContentBlock::Unknown {
                raw: serde_json::to_value(part).unwrap_or(Value::Null),
            },
        },
        "input_image" => value_to_image_or_unknown(
            part.input_image
                .as_ref()
                .unwrap_or(&Value::Object(part.extra.clone().into_iter().collect())),
        ),
        "file" | "input_file" => match &part.file {
            Some(file) => value_to_file_or_unknown(file),
            None => ContentBlock::Unknown {
                raw: serde_json::to_value(part).unwrap_or(Value::Null),
            },
        },
        _ => ContentBlock::Unknown {
            raw: serde_json::to_value(part).unwrap_or(Value::Null),
        },
    }
}

pub(crate) fn blocks_to_openai_content(
    blocks: &[ContentBlock],
    text_kind: &str,
    image_kind: &str,
) -> Option<openai::OpenAiContent> {
    match blocks {
        [] => None,
        [ContentBlock::Text { text }] => Some(openai::OpenAiContent::Text(text.clone())),
        blocks => Some(openai::OpenAiContent::Parts(
            blocks
                .iter()
                .map(|block| block_to_openai_part(block, text_kind, image_kind))
                .collect(),
        )),
    }
}

fn block_to_openai_part(
    block: &ContentBlock,
    text_kind: &str,
    image_kind: &str,
) -> openai::OpenAiContentPart {
    let mut part = openai::OpenAiContentPart {
        kind: String::new(),
        text: None,
        image_url: None,
        input_text: None,
        output_text: None,
        input_image: None,
        file: None,
        refusal: None,
        extra: Default::default(),
    };

    match block {
        ContentBlock::Text { text } => {
            part.kind = text_kind.to_string();
            match text_kind {
                "input_text" => part.input_text = Some(text.clone()),
                "output_text" => part.output_text = Some(text.clone()),
                _ => part.text = Some(text.clone()),
            }
        }
        ContentBlock::Image {
            media_type,
            url,
            data,
            extensions,
        } => {
            part.kind = image_kind.to_string();
            if image_kind == "image_url" {
                part.image_url = Some(openai::OpenAiImageUrl {
                    url: url.clone().or_else(|| data.clone()).unwrap_or_default(),
                    detail: None,
                    extra: Default::default(),
                });
            } else {
                let mut image = Map::new();
                if let Some(url) = url {
                    image.insert("image_url".to_string(), Value::String(url.clone()));
                }
                if let Some(data) = data {
                    image.insert("data".to_string(), Value::String(data.clone()));
                }
                if let Some(media_type) = media_type {
                    image.insert("media_type".to_string(), Value::String(media_type.clone()));
                }
                for (key, value) in extensions {
                    image.insert(key.clone(), value.clone());
                }
                part.input_image = Some(Value::Object(image));
            }
        }
        ContentBlock::File {
            media_type,
            filename,
            url,
            data,
            extensions,
        } => {
            part.kind = "file".to_string();
            let mut file = Map::new();
            if let Some(media_type) = media_type {
                file.insert("media_type".to_string(), Value::String(media_type.clone()));
            }
            if let Some(filename) = filename {
                file.insert("filename".to_string(), Value::String(filename.clone()));
            }
            if let Some(url) = url {
                file.insert("url".to_string(), Value::String(url.clone()));
            }
            if let Some(data) = data {
                file.insert("data".to_string(), Value::String(data.clone()));
            }
            for (key, value) in extensions {
                file.insert(key.clone(), value.clone());
            }
            part.file = Some(Value::Object(file));
        }
        ContentBlock::Unknown { raw } => {
            if let Ok(raw_part) = serde_json::from_value::<openai::OpenAiContentPart>(raw.clone()) {
                return raw_part;
            }
            part.kind = "unknown".to_string();
            part.extra.insert("raw".to_string(), raw.clone());
        }
        ContentBlock::ToolCall { .. }
        | ContentBlock::ToolResult { .. }
        | ContentBlock::Reasoning { .. } => {
            part.kind = "unknown".to_string();
            part.extra.insert(
                "raw".to_string(),
                serde_json::to_value(block).unwrap_or(Value::Null),
            );
        }
    }

    part
}

pub(crate) fn anthropic_content_to_blocks(
    content: &anthropic::AnthropicContent,
) -> Vec<ContentBlock> {
    match content {
        anthropic::AnthropicContent::Text(text) => vec![ContentBlock::Text { text: text.clone() }],
        anthropic::AnthropicContent::Blocks(blocks) => {
            blocks.iter().map(anthropic_block_to_block).collect()
        }
        anthropic::AnthropicContent::Raw(raw) => vec![ContentBlock::Unknown { raw: raw.clone() }],
    }
}

pub(crate) fn anthropic_system_to_blocks(
    system: Option<&anthropic::AnthropicSystem>,
) -> Vec<ContentBlock> {
    match system {
        Some(anthropic::AnthropicSystem::Text(text)) => {
            vec![ContentBlock::Text { text: text.clone() }]
        }
        Some(anthropic::AnthropicSystem::Blocks(blocks)) => {
            blocks.iter().map(anthropic_block_to_block).collect()
        }
        Some(anthropic::AnthropicSystem::Raw(raw)) => {
            vec![ContentBlock::Unknown { raw: raw.clone() }]
        }
        None => Vec::new(),
    }
}

pub(crate) fn anthropic_block_to_block(block: &anthropic::AnthropicContentBlock) -> ContentBlock {
    match block.kind.as_str() {
        "text" => ContentBlock::Text {
            text: block.text.clone().unwrap_or_default(),
        },
        "image" => ContentBlock::Image {
            media_type: block
                .source
                .as_ref()
                .and_then(|source| source.get("media_type"))
                .and_then(Value::as_str)
                .map(ToString::to_string),
            url: block
                .source
                .as_ref()
                .and_then(|source| source.get("url").or_else(|| source.get("image_url")))
                .and_then(Value::as_str)
                .map(ToString::to_string),
            data: block
                .source
                .as_ref()
                .and_then(|source| source.get("data"))
                .and_then(Value::as_str)
                .map(ToString::to_string),
            extensions: empty_extensions(),
        },
        "tool_use" => ContentBlock::ToolCall {
            id: block.id.clone().unwrap_or_default(),
            name: block.name.clone().unwrap_or_default(),
            arguments: block.input.clone().unwrap_or(Value::Null),
            extensions: empty_extensions(),
        },
        "tool_result" => ContentBlock::ToolResult {
            tool_call_id: block.tool_use_id.clone().unwrap_or_default(),
            content: value_to_blocks(block.content.as_ref()),
            is_error: block
                .extra
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            extensions: empty_extensions(),
        },
        "thinking" | "redacted_thinking" => ContentBlock::Reasoning {
            text: block.thinking.clone().or_else(|| block.text.clone()),
            encrypted: block.signature.clone(),
            extensions: empty_extensions(),
        },
        _ => ContentBlock::Unknown {
            raw: serde_json::to_value(block).unwrap_or(Value::Null),
        },
    }
}

pub(crate) fn block_to_anthropic_block(block: &ContentBlock) -> anthropic::AnthropicContentBlock {
    let mut content_block = anthropic::AnthropicContentBlock {
        kind: String::new(),
        text: None,
        source: None,
        id: None,
        name: None,
        input: None,
        tool_use_id: None,
        content: None,
        thinking: None,
        signature: None,
        extra: Default::default(),
    };

    match block {
        ContentBlock::Text { text } => {
            content_block.kind = "text".to_string();
            content_block.text = Some(text.clone());
        }
        ContentBlock::Image {
            media_type,
            url,
            data,
            extensions,
        } => {
            content_block.kind = "image".to_string();
            let mut source = Map::new();
            source.insert("type".to_string(), Value::String("base64".to_string()));
            if let Some(media_type) = media_type {
                source.insert("media_type".to_string(), Value::String(media_type.clone()));
            }
            if let Some(data) = data {
                source.insert("data".to_string(), Value::String(data.clone()));
            }
            if let Some(url) = url {
                source.insert("url".to_string(), Value::String(url.clone()));
            }
            for (key, value) in extensions {
                source.insert(key.clone(), value.clone());
            }
            content_block.source = Some(Value::Object(source));
        }
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
            ..
        } => {
            content_block.kind = "tool_use".to_string();
            content_block.id = Some(id.clone());
            content_block.name = Some(name.clone());
            content_block.input = Some(arguments.clone());
        }
        ContentBlock::ToolResult {
            tool_call_id,
            content,
            is_error,
            ..
        } => {
            content_block.kind = "tool_result".to_string();
            content_block.tool_use_id = Some(tool_call_id.clone());
            content_block.content = Some(blocks_to_value(content));
            if *is_error {
                content_block
                    .extra
                    .insert("is_error".to_string(), Value::Bool(true));
            }
        }
        ContentBlock::Reasoning {
            text, encrypted, ..
        } => {
            content_block.kind = "thinking".to_string();
            content_block.thinking = text.clone();
            content_block.signature = encrypted.clone();
        }
        ContentBlock::File { .. } | ContentBlock::Unknown { .. } => {
            if let ContentBlock::Unknown { raw } = block {
                if let Ok(raw_block) =
                    serde_json::from_value::<anthropic::AnthropicContentBlock>(raw.clone())
                {
                    return raw_block;
                }
            }
            content_block.kind = "unknown".to_string();
            content_block.extra.insert(
                "raw".to_string(),
                serde_json::to_value(block).unwrap_or(Value::Null),
            );
        }
    }

    content_block
}

pub(crate) fn blocks_to_anthropic_content(blocks: &[ContentBlock]) -> anthropic::AnthropicContent {
    match blocks {
        [ContentBlock::Text { text }] => anthropic::AnthropicContent::Text(text.clone()),
        blocks => anthropic::AnthropicContent::Blocks(
            blocks.iter().map(block_to_anthropic_block).collect(),
        ),
    }
}

pub(crate) fn blocks_to_anthropic_system(
    blocks: &[ContentBlock],
) -> Option<anthropic::AnthropicSystem> {
    match blocks {
        [] => None,
        [ContentBlock::Text { text }] => Some(anthropic::AnthropicSystem::Text(text.clone())),
        blocks => Some(anthropic::AnthropicSystem::Blocks(
            blocks.iter().map(block_to_anthropic_block).collect(),
        )),
    }
}

pub(crate) fn push_block_events(
    events: &mut Vec<UniversalEvent>,
    index: usize,
    block: ContentBlock,
) {
    events.push(UniversalEvent::ContentStart {
        index,
        block: block.clone(),
    });
    match &block {
        ContentBlock::Text { text } if !text.is_empty() => {
            events.push(UniversalEvent::TextDelta {
                index,
                text: text.clone(),
            });
        }
        ContentBlock::Reasoning {
            text: Some(text), ..
        } if !text.is_empty() => {
            events.push(UniversalEvent::ReasoningDelta {
                index,
                text: text.clone(),
            });
        }
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
            ..
        } => {
            events.push(UniversalEvent::ToolCallDelta {
                id: id.clone(),
                name: Some(name.clone()),
                arguments_delta: stringify_arguments(arguments),
            });
        }
        _ => {}
    }
    events.push(UniversalEvent::ContentDone {
        index,
        final_block: Some(block),
    });
}

pub(crate) fn ensure_response_start(
    events: &mut Vec<UniversalEvent>,
    state: &mut DecodeState,
    id: Option<String>,
    model: Option<String>,
) {
    if mark_once(state, "response_start") {
        events.push(UniversalEvent::ResponseStart {
            id,
            model,
            extensions: empty_extensions(),
        });
    }
}

pub(crate) fn ensure_message_start(
    events: &mut Vec<UniversalEvent>,
    state: &mut DecodeState,
    id: String,
    role: Role,
) {
    if mark_once(state, &format!("message_start:{id}")) {
        events.push(UniversalEvent::MessageStart {
            id,
            role,
            extensions: empty_extensions(),
        });
    }
}

pub(crate) fn ensure_content_start(
    events: &mut Vec<UniversalEvent>,
    state: &mut DecodeState,
    index: usize,
    block: ContentBlock,
) {
    if mark_once(state, &format!("content_start:{index}")) {
        events.push(UniversalEvent::ContentStart { index, block });
    }
}

pub(crate) fn mark_once(state: &mut DecodeState, key: &str) -> bool {
    let previous = state.extensions.insert(key.to_string(), Value::Bool(true));
    !matches!(previous, Some(Value::Bool(true)))
}

pub(crate) fn wire_event(data: Value) -> WireEvent {
    WireEvent { event: None, data }
}

pub(crate) fn encode_state_index(state: &mut EncodeState) -> usize {
    let key = "nextIndex".to_string();
    let next = state
        .extensions
        .get(&key)
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    state
        .extensions
        .insert(key, Value::Number(((next + 1) as u64).into()));
    next
}

fn value_to_image_or_unknown(value: &Value) -> ContentBlock {
    let Some(object) = value.as_object() else {
        return ContentBlock::Unknown { raw: value.clone() };
    };
    ContentBlock::Image {
        media_type: object
            .get("media_type")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        url: object
            .get("image_url")
            .or_else(|| object.get("url"))
            .and_then(Value::as_str)
            .map(ToString::to_string),
        data: object
            .get("data")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        extensions: empty_extensions(),
    }
}

fn value_to_file_or_unknown(value: &Value) -> ContentBlock {
    let Some(object) = value.as_object() else {
        return ContentBlock::Unknown { raw: value.clone() };
    };
    ContentBlock::File {
        media_type: object
            .get("media_type")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        filename: object
            .get("filename")
            .or_else(|| object.get("name"))
            .and_then(Value::as_str)
            .map(ToString::to_string),
        url: object
            .get("url")
            .or_else(|| object.get("file_url"))
            .and_then(Value::as_str)
            .map(ToString::to_string),
        data: object
            .get("data")
            .or_else(|| object.get("file_data"))
            .and_then(Value::as_str)
            .map(ToString::to_string),
        extensions: empty_extensions(),
    }
}

fn value_to_blocks(value: Option<&Value>) -> Vec<ContentBlock> {
    match value {
        Some(Value::String(text)) => vec![ContentBlock::Text { text: text.clone() }],
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                serde_json::from_value::<anthropic::AnthropicContentBlock>(item.clone())
                    .map(|block| anthropic_block_to_block(&block))
                    .unwrap_or_else(|_| ContentBlock::Unknown { raw: item.clone() })
            })
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
                .map(|block| {
                    serde_json::to_value(block_to_anthropic_block(block)).unwrap_or(Value::Null)
                })
                .collect(),
        ),
    }
}
