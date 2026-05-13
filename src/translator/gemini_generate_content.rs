use serde_json::{json, Map, Value};

use crate::translator::{common, WireEvent, WireTranslator};
use crate::{
    ApiProxyError, ContentBlock, DecodeState, EncodeState, FinishReason, GenerationConfig, Result,
    Role, ToolChoice, UniversalEvent, UniversalItem, UniversalRequest, UniversalResponse,
    UniversalTool, Usage, WireProtocol,
};

const VA_MODEL_KEY: &str = "__va_model";
const VA_STREAM_KEY: &str = "__va_stream";

pub struct GeminiGenerateContentTranslator;

impl WireTranslator for GeminiGenerateContentTranslator {
    fn protocol(&self) -> WireProtocol {
        WireProtocol::GeminiGenerateContent
    }

    fn decode_request(&self, raw: Value) -> Result<UniversalRequest> {
        decode_request(raw)
    }

    fn encode_request(&self, request: &UniversalRequest) -> Result<Value> {
        encode_request(request)
    }

    fn decode_response(&self, raw: Value) -> Result<Vec<UniversalEvent>> {
        decode_response(raw)
    }

    fn decode_stream_chunk(
        &self,
        raw: Value,
        state: &mut DecodeState,
    ) -> Result<Vec<UniversalEvent>> {
        decode_stream_chunk(raw, state)
    }

    fn encode_events(
        &self,
        events: &[UniversalEvent],
        state: &mut EncodeState,
    ) -> Result<Vec<WireEvent>> {
        encode_stream_events(events, state)
    }
}

pub fn attach_route_metadata(body: &mut Value, model: &str, stream: bool) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    object.insert(VA_MODEL_KEY.to_string(), Value::String(model.to_string()));
    object.insert(VA_STREAM_KEY.to_string(), Value::Bool(stream));
}

pub fn strip_route_metadata(body: &mut Value) {
    if let Some(object) = body.as_object_mut() {
        object.remove(VA_MODEL_KEY);
        object.remove(VA_STREAM_KEY);
    }
}

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

fn decode_request(raw: Value) -> Result<UniversalRequest> {
    let source_raw = raw.clone();
    let object = raw
        .as_object()
        .ok_or_else(|| ApiProxyError::invalid_request("Gemini request must be a JSON object"))?;
    let mut request = UniversalRequest {
        model: object
            .get(VA_MODEL_KEY)
            .and_then(Value::as_str)
            .map(model_from_route_segment),
        stream: object
            .get(VA_STREAM_KEY)
            .and_then(Value::as_bool)
            .unwrap_or(false),
        generation: generation_from_gemini(object.get("generationConfig")),
        source: Some(common::source(
            WireProtocol::GeminiGenerateContent,
            source_raw,
        )),
        ..UniversalRequest::default()
    };
    request.instructions = decode_system_instruction(object.get("systemInstruction"));
    request.input = decode_contents(object.get("contents"))?;
    request.tools = decode_tools(object.get("tools"));
    request.tool_choice = decode_tool_choice(object.get("toolConfig"));
    Ok(request)
}

fn encode_request(request: &UniversalRequest) -> Result<Value> {
    let mut body = Map::new();
    body.insert(
        VA_MODEL_KEY.to_string(),
        Value::String(request.model.clone().unwrap_or_default()),
    );
    body.insert(VA_STREAM_KEY.to_string(), Value::Bool(request.stream));
    if !request.instructions.is_empty() {
        body.insert(
            "systemInstruction".to_string(),
            json!({ "parts": blocks_to_gemini_parts(&request.instructions) }),
        );
    }
    body.insert(
        "contents".to_string(),
        Value::Array(items_to_contents(&request.input)),
    );
    if !request.tools.is_empty() {
        body.insert("tools".to_string(), tools_to_gemini(&request.tools));
    }
    if let Some(tool_choice) = &request.tool_choice {
        body.insert("toolConfig".to_string(), tool_choice_to_gemini(tool_choice));
    }
    let generation = generation_to_gemini(&request.generation);
    if !generation.is_empty() {
        body.insert("generationConfig".to_string(), Value::Object(generation));
    }
    Ok(Value::Object(body))
}

fn decode_response(raw: Value) -> Result<Vec<UniversalEvent>> {
    let mut events = vec![UniversalEvent::ResponseStart {
        id: None,
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

fn decode_stream_chunk(raw: Value, state: &mut DecodeState) -> Result<Vec<UniversalEvent>> {
    let mut events = Vec::new();
    if mark_once(state, "gemini.response_start") {
        events.push(UniversalEvent::ResponseStart {
            id: None,
            model: raw
                .get("modelVersion")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            extensions: common::empty_extensions(),
        });
    }
    decode_candidates(&raw, &mut events, false)?;
    if has_finish_reason(&raw) {
        events.push(UniversalEvent::ResponseDone {
            usage: usage_from_gemini(raw.get("usageMetadata")),
            extensions: common::empty_extensions(),
        });
    }
    Ok(events)
}

fn encode_stream_events(
    events: &[UniversalEvent],
    state: &mut EncodeState,
) -> Result<Vec<WireEvent>> {
    let mut out = Vec::new();
    for event in events {
        match event {
            UniversalEvent::TextDelta { text, .. } if !text.is_empty() => {
                out.push(WireEvent {
                    event: None,
                    data: gemini_chunk(vec![json!({ "text": text })], None, None),
                });
            }
            UniversalEvent::ReasoningDelta { text, .. } if !text.is_empty() => {
                out.push(WireEvent {
                    event: None,
                    data: gemini_chunk(vec![json!({ "text": text, "thought": true })], None, None),
                });
            }
            UniversalEvent::ToolCallDelta {
                name,
                arguments_delta,
                ..
            } => {
                let args = serde_json::from_str::<Value>(arguments_delta)
                    .unwrap_or_else(|_| json!({ "value": arguments_delta }));
                out.push(WireEvent {
                    event: None,
                    data: gemini_chunk(
                        vec![json!({
                            "functionCall": {
                                "name": name.clone().unwrap_or_default(),
                                "args": args,
                            }
                        })],
                        None,
                        None,
                    ),
                });
            }
            UniversalEvent::MessageDone {
                finish_reason,
                usage,
                ..
            } => {
                let finish = finish_reason.map(finish_reason_to_gemini);
                out.push(WireEvent {
                    event: None,
                    data: gemini_chunk(Vec::new(), finish, usage.as_ref()),
                });
            }
            UniversalEvent::ResponseDone { usage, .. } => {
                if !state
                    .extensions
                    .contains_key("gemini.response_done_usage_emitted")
                    && usage.is_some()
                {
                    state.extensions.insert(
                        "gemini.response_done_usage_emitted".to_string(),
                        Value::Bool(true),
                    );
                    out.push(WireEvent {
                        event: None,
                        data: gemini_chunk(Vec::new(), None, usage.as_ref()),
                    });
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn decode_system_instruction(value: Option<&Value>) -> Vec<ContentBlock> {
    value
        .and_then(|value| value.get("parts"))
        .map(gemini_parts_to_blocks)
        .unwrap_or_default()
}

fn decode_contents(value: Option<&Value>) -> Result<Vec<UniversalItem>> {
    let Some(contents) = value.and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for content in contents {
        let role = content
            .get("role")
            .and_then(Value::as_str)
            .map(gemini_role_to_universal)
            .unwrap_or(Role::User);
        let parts = content.get("parts").and_then(Value::as_array);
        let mut message_blocks = Vec::new();
        if let Some(parts) = parts {
            for part in parts {
                if let Some(function_response) = part.get("functionResponse") {
                    push_message_if_any(&mut out, role, &mut message_blocks);
                    out.push(function_response_to_tool_result(function_response));
                    continue;
                }
                if let Some(function_call) = part.get("functionCall") {
                    push_message_if_any(&mut out, role, &mut message_blocks);
                    out.push(function_call_to_tool_call(function_call));
                    continue;
                }
                message_blocks.extend(gemini_part_to_blocks(part));
            }
        }
        push_message_if_any(&mut out, role, &mut message_blocks);
    }
    Ok(out)
}

fn decode_tools(value: Option<&Value>) -> Vec<UniversalTool> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|tool| {
            tool.get("functionDeclarations")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|declaration| {
            let name = declaration.get("name")?.as_str()?.to_string();
            Some(UniversalTool {
                name,
                description: declaration
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                input_schema: declaration.get("parameters").cloned(),
                extensions: common::empty_extensions(),
            })
        })
        .collect()
}

fn decode_tool_choice(value: Option<&Value>) -> Option<ToolChoice> {
    let config = value?.get("functionCallingConfig")?;
    match config.get("mode").and_then(Value::as_str) {
        Some("NONE") => Some(ToolChoice::None),
        Some("ANY") => config
            .get("allowedFunctionNames")
            .and_then(Value::as_array)
            .and_then(|names| names.first())
            .and_then(Value::as_str)
            .map(|name| ToolChoice::Tool {
                name: name.to_string(),
            })
            .or(Some(ToolChoice::Required)),
        Some("AUTO") => Some(ToolChoice::Auto),
        _ => None,
    }
}

fn generation_from_gemini(value: Option<&Value>) -> GenerationConfig {
    let Some(object) = value.and_then(Value::as_object) else {
        return GenerationConfig::default();
    };
    GenerationConfig {
        temperature: object.get("temperature").and_then(Value::as_f64),
        top_p: object.get("topP").and_then(Value::as_f64),
        max_output_tokens: object.get("maxOutputTokens").and_then(Value::as_u64),
        extensions: common::empty_extensions(),
    }
}

fn gemini_parts_to_blocks(value: &Value) -> Vec<ContentBlock> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(gemini_part_to_blocks)
        .collect()
}

fn gemini_part_to_blocks(part: &Value) -> Vec<ContentBlock> {
    if part
        .get("thought")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return vec![ContentBlock::Reasoning {
            text: part
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned),
            encrypted: gemini_thought_signature(part),
            extensions: common::empty_extensions(),
        }];
    }
    if let Some(text) = part.get("text").and_then(Value::as_str) {
        return vec![ContentBlock::Text {
            text: text.to_string(),
        }];
    }
    if let Some(inline) = part.get("inlineData").or_else(|| part.get("inline_data")) {
        let media_type = inline
            .get("mimeType")
            .or_else(|| inline.get("mime_type"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let data = inline
            .get("data")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        return vec![data_block(media_type, None, data)];
    }
    if let Some(file) = part.get("fileData").or_else(|| part.get("file_data")) {
        let media_type = file
            .get("mimeType")
            .or_else(|| file.get("mime_type"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let url = file
            .get("fileUri")
            .or_else(|| file.get("file_uri"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        return vec![data_block(media_type, url, None)];
    }
    if let Some(function_call) = part.get("functionCall") {
        let name = function_call
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        return vec![ContentBlock::ToolCall {
            id: name.clone(),
            name,
            arguments: function_call
                .get("args")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new())),
            extensions: common::empty_extensions(),
        }];
    }
    vec![ContentBlock::Unknown { raw: part.clone() }]
}

fn gemini_thought_signature(part: &Value) -> Option<String> {
    part.get("thoughtSignature")
        .or_else(|| part.get("thought_signature"))
        .and_then(Value::as_str)
        .filter(|signature| !signature.is_empty())
        .map(ToOwned::to_owned)
}

fn data_block(
    media_type: Option<String>,
    url: Option<String>,
    data: Option<String>,
) -> ContentBlock {
    if media_type
        .as_deref()
        .is_some_and(|media_type| media_type.starts_with("image/"))
    {
        ContentBlock::Image {
            media_type,
            url,
            data,
            extensions: common::empty_extensions(),
        }
    } else {
        ContentBlock::File {
            media_type,
            filename: None,
            url,
            data,
            extensions: common::empty_extensions(),
        }
    }
}

fn function_response_to_tool_result(function_response: &Value) -> UniversalItem {
    let name = function_response
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let response = function_response
        .get("response")
        .cloned()
        .unwrap_or(Value::Null);
    UniversalItem::ToolResult {
        tool_call_id: name,
        content: vec![ContentBlock::Text {
            text: stringify_json(response),
        }],
        is_error: false,
        extensions: common::empty_extensions(),
    }
}

fn function_call_to_tool_call(function_call: &Value) -> UniversalItem {
    let name = function_call
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    UniversalItem::ToolCall {
        id: name.clone(),
        name,
        arguments: function_call
            .get("args")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new())),
        extensions: common::empty_extensions(),
    }
}

fn push_message_if_any(
    out: &mut Vec<UniversalItem>,
    role: Role,
    message_blocks: &mut Vec<ContentBlock>,
) {
    if message_blocks.is_empty() {
        return;
    }
    out.push(UniversalItem::Message {
        role,
        id: None,
        content: std::mem::take(message_blocks),
        extensions: common::empty_extensions(),
    });
}

fn decode_candidates(
    raw: &Value,
    events: &mut Vec<UniversalEvent>,
    final_blocks: bool,
) -> Result<()> {
    let candidates = raw
        .get("candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiProxyError::invalid_response("Gemini response missing candidates"))?;
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
                    arguments_delta: stringify_json(arguments.clone()),
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

fn items_to_contents(items: &[UniversalItem]) -> Vec<Value> {
    let mut contents = Vec::new();
    for item in items {
        match item {
            UniversalItem::Message { role, content, .. } => {
                contents.push(json!({
                    "role": universal_role_to_gemini(*role),
                    "parts": blocks_to_gemini_parts(content),
                }));
            }
            UniversalItem::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                contents.push(json!({
                    "role": "function",
                    "parts": [{
                        "functionResponse": {
                            "name": tool_call_id,
                            "response": {
                                "content": blocks_to_text(content),
                                "is_error": is_error,
                            }
                        }
                    }]
                }));
            }
            UniversalItem::ToolCall {
                name, arguments, ..
            } => {
                contents.push(json!({
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": name,
                            "args": arguments,
                        }
                    }]
                }));
            }
            UniversalItem::Reasoning { .. } => {}
            UniversalItem::Unknown { raw } => contents.push(raw.clone()),
        }
    }
    contents
}

fn blocks_to_gemini_parts(blocks: &[ContentBlock]) -> Vec<Value> {
    blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => json!({ "text": text }),
            ContentBlock::Image {
                media_type,
                url: Some(url),
                ..
            }
            | ContentBlock::File {
                media_type,
                url: Some(url),
                ..
            } => json!({
                "fileData": {
                    "mimeType": media_type.clone().unwrap_or_else(|| "application/octet-stream".to_string()),
                    "fileUri": url,
                }
            }),
            ContentBlock::Image {
                media_type,
                data: Some(data),
                ..
            }
            | ContentBlock::File {
                media_type,
                data: Some(data),
                ..
            } => json!({
                "inlineData": {
                    "mimeType": media_type.clone().unwrap_or_else(|| "application/octet-stream".to_string()),
                    "data": data,
                }
            }),
            ContentBlock::ToolCall {
                name, arguments, ..
            } => json!({ "functionCall": { "name": name, "args": arguments } }),
            ContentBlock::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => json!({
                "functionResponse": {
                    "name": tool_call_id,
                    "response": {
                        "content": blocks_to_text(content),
                        "is_error": is_error,
                    }
                }
            }),
            ContentBlock::Reasoning {
                text, encrypted, ..
            } => {
                let mut part = Map::new();
                part.insert("thought".to_string(), Value::Bool(true));
                if let Some(text) = text {
                    part.insert("text".to_string(), Value::String(text.clone()));
                }
                if let Some(encrypted) = encrypted {
                    part.insert(
                        "thoughtSignature".to_string(),
                        Value::String(encrypted.clone()),
                    );
                }
                Value::Object(part)
            }
            ContentBlock::Unknown { raw } => raw.clone(),
            ContentBlock::Image { .. } | ContentBlock::File { .. } => json!({}),
        })
        .collect()
}

fn tools_to_gemini(tools: &[UniversalTool]) -> Value {
    Value::Array(vec![json!({
        "functionDeclarations": tools.iter().map(|tool| {
            let mut declaration = Map::new();
            declaration.insert("name".to_string(), Value::String(tool.name.clone()));
            if let Some(description) = &tool.description {
                declaration.insert("description".to_string(), Value::String(description.clone()));
            }
            if let Some(schema) = &tool.input_schema {
                declaration.insert("parameters".to_string(), schema.clone());
            }
            Value::Object(declaration)
        }).collect::<Vec<_>>()
    })])
}

fn tool_choice_to_gemini(tool_choice: &ToolChoice) -> Value {
    match tool_choice {
        ToolChoice::Auto => json!({ "functionCallingConfig": { "mode": "AUTO" } }),
        ToolChoice::None => json!({ "functionCallingConfig": { "mode": "NONE" } }),
        ToolChoice::Required => json!({ "functionCallingConfig": { "mode": "ANY" } }),
        ToolChoice::Tool { name } => {
            json!({ "functionCallingConfig": { "mode": "ANY", "allowedFunctionNames": [name] } })
        }
    }
}

fn generation_to_gemini(generation: &GenerationConfig) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(temperature) = generation.temperature {
        out.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(top_p) = generation.top_p {
        out.insert("topP".to_string(), json!(top_p));
    }
    if let Some(max_output_tokens) = generation.max_output_tokens {
        out.insert("maxOutputTokens".to_string(), json!(max_output_tokens));
    }
    out
}

fn response_parts(response: &UniversalResponse) -> Vec<Value> {
    let mut parts = Vec::new();
    for item in &response.output {
        match item {
            UniversalItem::Message { content, .. } => parts.extend(blocks_to_gemini_parts(content)),
            UniversalItem::ToolCall {
                name, arguments, ..
            } => parts.push(json!({ "functionCall": { "name": name, "args": arguments } })),
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

fn gemini_chunk(parts: Vec<Value>, finish_reason: Option<&str>, usage: Option<&Usage>) -> Value {
    let mut candidate = Map::new();
    if !parts.is_empty() {
        candidate.insert(
            "content".to_string(),
            json!({
                "role": "model",
                "parts": parts,
            }),
        );
    }
    if let Some(finish_reason) = finish_reason {
        candidate.insert(
            "finishReason".to_string(),
            Value::String(finish_reason.to_string()),
        );
    }
    let mut out = Map::new();
    out.insert(
        "candidates".to_string(),
        Value::Array(vec![Value::Object(candidate)]),
    );
    if let Some(usage) = usage {
        out.insert("usageMetadata".to_string(), usage_to_gemini(usage));
    }
    Value::Object(out)
}

fn usage_from_gemini(value: Option<&Value>) -> Option<Usage> {
    let value = value?;
    Some(Usage {
        input_tokens: value.get("promptTokenCount").and_then(Value::as_u64),
        output_tokens: value.get("candidatesTokenCount").and_then(Value::as_u64),
        total_tokens: value.get("totalTokenCount").and_then(Value::as_u64),
    })
}

fn usage_to_gemini(usage: &Usage) -> Value {
    let mut out = Map::new();
    if let Some(input_tokens) = usage.input_tokens {
        out.insert("promptTokenCount".to_string(), json!(input_tokens));
    }
    if let Some(output_tokens) = usage.output_tokens {
        out.insert("candidatesTokenCount".to_string(), json!(output_tokens));
    }
    if let Some(total_tokens) = usage.total_tokens {
        out.insert("totalTokenCount".to_string(), json!(total_tokens));
    }
    Value::Object(out)
}

fn model_from_route_segment(value: &str) -> String {
    value.strip_prefix("models/").unwrap_or(value).to_string()
}

fn gemini_role_to_universal(role: &str) -> Role {
    match role {
        "model" => Role::Assistant,
        "function" => Role::Tool,
        _ => Role::User,
    }
}

fn universal_role_to_gemini(role: Role) -> &'static str {
    match role {
        Role::Assistant => "model",
        Role::Tool => "function",
        Role::Developer | Role::System | Role::User => "user",
    }
}

fn finish_reason_from_gemini(value: &str) -> FinishReason {
    match value {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => {
            FinishReason::ContentFilter
        }
        "MALFORMED_FUNCTION_CALL" => FinishReason::ToolCall,
        _ => FinishReason::Unknown,
    }
}

fn finish_reason_to_gemini(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "STOP",
        FinishReason::Length => "MAX_TOKENS",
        FinishReason::ToolCall => "MALFORMED_FUNCTION_CALL",
        FinishReason::ContentFilter => "SAFETY",
        FinishReason::Error => "OTHER",
        FinishReason::Unknown => "FINISH_REASON_UNSPECIFIED",
    }
}

fn has_finish_reason(raw: &Value) -> bool {
    raw.get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|candidate| candidate.get("finishReason").is_some())
}

fn mark_once(state: &mut DecodeState, key: &str) -> bool {
    if state.extensions.contains_key(key) {
        return false;
    }
    state.extensions.insert(key.to_string(), Value::Bool(true));
    true
}

fn blocks_to_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn stringify_json(value: Value) -> String {
    match value {
        Value::String(value) => value,
        other => serde_json::to_string(&other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{ContentBlock, Role, UniversalItem, WireTranslator};

    use super::GeminiGenerateContentTranslator;

    #[test]
    fn decodes_generate_content_request() {
        let mut body = json!({
            "contents": [
                {
                    "role": "user",
                    "parts": [{ "text": "hello" }]
                },
                {
                    "role": "model",
                    "parts": [
                        { "thought": true, "text": "Need to inspect cwd.", "thoughtSignature": "sig_123" },
                        { "functionCall": { "name": "exec_command", "args": { "cmd": "pwd" } } }
                    ]
                },
                {
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": "exec_command",
                            "response": { "output": "/tmp/project" }
                        }
                    }]
                }
            ],
            "generationConfig": { "maxOutputTokens": 32 }
        });
        super::attach_route_metadata(&mut body, "gemini-2.5-flash", false);

        let request = GeminiGenerateContentTranslator
            .decode_request(body)
            .unwrap();

        assert_eq!(request.model.as_deref(), Some("gemini-2.5-flash"));
        assert!(!request.stream);
        assert_eq!(request.generation.max_output_tokens, Some(32));
        assert!(matches!(
            request.input.first(),
            Some(UniversalItem::Message {
                role: Role::User,
                ..
            })
        ));
        assert!(matches!(
            request.input.get(1),
            Some(UniversalItem::Message {
                role: Role::Assistant,
                content,
                ..
            }) if matches!(
                content.first(),
                Some(ContentBlock::Reasoning {
                    text: Some(text),
                    encrypted: Some(signature),
                    ..
                }) if text == "Need to inspect cwd." && signature == "sig_123"
            )
        ));
        assert!(matches!(
            request.input.get(2),
            Some(UniversalItem::ToolCall {
                id,
                name,
                arguments,
                ..
            }) if id == "exec_command"
                && name == "exec_command"
                && arguments["cmd"] == "pwd"
        ));
        assert!(matches!(
            request.input.get(3),
            Some(UniversalItem::ToolResult {
                tool_call_id,
                ..
            }) if tool_call_id == "exec_command"
        ));
    }

    #[test]
    fn encodes_gemini_completion_response() {
        let events = GeminiGenerateContentTranslator
            .decode_response(json!({
                "candidates": [{
                    "content": { "role": "model", "parts": [{ "text": "pong" }] },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2
                }
            }))
            .unwrap();

        let response = super::encode_response(&events);

        assert_eq!(
            response["candidates"][0]["content"]["parts"][0]["text"],
            "pong"
        );
        assert_eq!(response["candidates"][0]["finishReason"], "STOP");
        assert_eq!(response["usageMetadata"]["totalTokenCount"], 2);
    }

    #[test]
    fn encodes_reasoning_as_gemini_thought_part() {
        let events = GeminiGenerateContentTranslator
            .decode_response(json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [
                            { "thought": true, "text": "I should inspect cwd." },
                            { "functionCall": { "name": "exec_command", "args": { "cmd": "pwd" } } }
                        ]
                    },
                    "finishReason": "MALFORMED_FUNCTION_CALL"
                }]
            }))
            .unwrap();

        let response = super::encode_response(&events);

        assert_eq!(
            response["candidates"][0]["content"]["parts"][0]["thought"],
            true
        );
        assert_eq!(
            response["candidates"][0]["content"]["parts"][0]["text"],
            "I should inspect cwd."
        );
        assert_eq!(
            response["candidates"][0]["content"]["parts"][1]["functionCall"]["name"],
            "exec_command"
        );
    }
}
