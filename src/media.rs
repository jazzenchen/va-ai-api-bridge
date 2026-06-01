use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ContentBlock, ResolvedModelSpec, Result, Role, UniversalItem, UniversalRequest};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaSanitization {
    #[serde(default, skip_serializing_if = "is_false")]
    pub image_omitted: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub file_omitted: bool,
}

impl MediaSanitization {
    pub fn changed(self) -> bool {
        self.image_omitted || self.file_omitted
    }

    fn merge(&mut self, other: Self) {
        self.image_omitted |= other.image_omitted;
        self.file_omitted |= other.file_omitted;
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MediaUsage {
    image_input: bool,
    file_input: bool,
}

impl MediaUsage {
    fn is_empty(self) -> bool {
        !self.image_input && !self.file_input
    }

    fn intersects(self, other: Self) -> bool {
        (self.image_input && other.image_input) || (self.file_input && other.file_input)
    }
}

pub fn sanitize_unsupported_media(
    request: &mut UniversalRequest,
    model: &ResolvedModelSpec,
) -> MediaSanitization {
    let usage = request_media_usage(request);
    let unsupported = MediaUsage {
        image_input: usage.image_input && !model.capabilities.supports_image_input(),
        file_input: usage.file_input && !model.capabilities.supports_file_input(),
    };
    if unsupported.is_empty() {
        return MediaSanitization::default();
    }

    sanitize_request(request, model, unsupported)
}

pub fn sanitize_unsupported_media_from_json(
    request: &mut UniversalRequest,
    model: Value,
) -> Result<MediaSanitization> {
    let model = ResolvedModelSpec::from_json(model)?;
    Ok(sanitize_unsupported_media(request, &model))
}

fn sanitize_request(
    request: &mut UniversalRequest,
    model: &ResolvedModelSpec,
    unsupported: MediaUsage,
) -> MediaSanitization {
    let mut sanitization = sanitize_blocks(&mut request.instructions, model, unsupported);
    for item in &mut request.input {
        sanitization.merge(sanitize_item(item, model, unsupported));
    }
    sanitization
}

fn sanitize_item(
    item: &mut UniversalItem,
    model: &ResolvedModelSpec,
    unsupported: MediaUsage,
) -> MediaSanitization {
    match item {
        UniversalItem::Message { content, .. } | UniversalItem::ToolResult { content, .. } => {
            sanitize_blocks(content, model, unsupported)
        }
        UniversalItem::Unknown { raw } => {
            let usage = value_media_usage(raw);
            if !usage.intersects(unsupported) {
                return MediaSanitization::default();
            }
            let omitted = omitted_usage(usage, unsupported);
            *item = UniversalItem::Message {
                role: Role::User,
                id: None,
                content: vec![omitted_content_block(omitted, model)],
                extensions: Default::default(),
            };
            sanitization_for_usage(omitted)
        }
        UniversalItem::ToolCall { .. } | UniversalItem::Reasoning { .. } => {
            MediaSanitization::default()
        }
    }
}

fn sanitize_blocks(
    blocks: &mut [ContentBlock],
    model: &ResolvedModelSpec,
    unsupported: MediaUsage,
) -> MediaSanitization {
    let mut sanitization = MediaSanitization::default();
    for block in blocks {
        sanitization.merge(sanitize_block(block, model, unsupported));
    }
    sanitization
}

fn sanitize_block(
    block: &mut ContentBlock,
    model: &ResolvedModelSpec,
    unsupported: MediaUsage,
) -> MediaSanitization {
    match block {
        ContentBlock::Image { .. } if unsupported.image_input => {
            *block = omitted_content_block(
                MediaUsage {
                    image_input: true,
                    file_input: false,
                },
                model,
            );
            MediaSanitization {
                image_omitted: true,
                file_omitted: false,
            }
        }
        ContentBlock::File { .. } if unsupported.file_input => {
            *block = omitted_content_block(
                MediaUsage {
                    image_input: false,
                    file_input: true,
                },
                model,
            );
            MediaSanitization {
                image_omitted: false,
                file_omitted: true,
            }
        }
        ContentBlock::ToolResult { content, .. } => sanitize_blocks(content, model, unsupported),
        ContentBlock::Unknown { raw } => {
            let usage = value_media_usage(raw);
            if !usage.intersects(unsupported) {
                return MediaSanitization::default();
            }
            let omitted = omitted_usage(usage, unsupported);
            *block = omitted_content_block(omitted, model);
            sanitization_for_usage(omitted)
        }
        ContentBlock::Text { .. }
        | ContentBlock::Image { .. }
        | ContentBlock::File { .. }
        | ContentBlock::ToolCall { .. }
        | ContentBlock::Reasoning { .. } => MediaSanitization::default(),
    }
}

fn request_media_usage(request: &UniversalRequest) -> MediaUsage {
    let mut usage = MediaUsage::default();
    collect_blocks_usage(&request.instructions, &mut usage);
    for item in &request.input {
        collect_item_usage(item, &mut usage);
    }
    usage
}

fn collect_item_usage(item: &UniversalItem, usage: &mut MediaUsage) {
    match item {
        UniversalItem::Message { content, .. } | UniversalItem::ToolResult { content, .. } => {
            collect_blocks_usage(content, usage);
        }
        UniversalItem::Unknown { raw } => collect_value_usage(raw, usage),
        UniversalItem::ToolCall { .. } | UniversalItem::Reasoning { .. } => {}
    }
}

fn collect_blocks_usage(blocks: &[ContentBlock], usage: &mut MediaUsage) {
    for block in blocks {
        collect_block_usage(block, usage);
    }
}

fn collect_block_usage(block: &ContentBlock, usage: &mut MediaUsage) {
    match block {
        ContentBlock::Image { .. } => usage.image_input = true,
        ContentBlock::File { .. } => usage.file_input = true,
        ContentBlock::ToolResult { content, .. } => collect_blocks_usage(content, usage),
        ContentBlock::Unknown { raw } => collect_value_usage(raw, usage),
        ContentBlock::Text { .. }
        | ContentBlock::ToolCall { .. }
        | ContentBlock::Reasoning { .. } => {}
    }
}

fn value_media_usage(value: &Value) -> MediaUsage {
    let mut usage = MediaUsage::default();
    collect_value_usage(value, &mut usage);
    usage
}

fn collect_value_usage(value: &Value, usage: &mut MediaUsage) {
    match value {
        Value::Array(values) => {
            for value in values {
                collect_value_usage(value, usage);
            }
        }
        Value::Object(object) => {
            let mut typed_image = false;
            let mut typed_file = false;
            if let Some(kind) = object.get("type").and_then(Value::as_str) {
                match kind {
                    "image" | "input_image" | "image_url" => {
                        typed_image = true;
                        usage.image_input = true;
                    }
                    "document" | "file" | "input_file" => {
                        typed_file = true;
                        usage.file_input = true;
                    }
                    _ => {}
                }
            }
            if object.contains_key("image_url") {
                usage.image_input = true;
            }
            if !typed_image
                && (typed_file
                    || object.contains_key("file_data")
                    || object.contains_key("fileData")
                    || object.contains_key("file_id")
                    || object.contains_key("file_url")
                    || object.contains_key("fileUrl")
                    || object.contains_key("filename"))
            {
                usage.file_input = true;
            }
            for value in object.values() {
                collect_value_usage(value, usage);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn omitted_usage(usage: MediaUsage, unsupported: MediaUsage) -> MediaUsage {
    MediaUsage {
        image_input: usage.image_input && unsupported.image_input,
        file_input: usage.file_input && unsupported.file_input,
    }
}

fn sanitization_for_usage(usage: MediaUsage) -> MediaSanitization {
    MediaSanitization {
        image_omitted: usage.image_input,
        file_omitted: usage.file_input,
    }
}

fn omitted_content_block(usage: MediaUsage, model: &ResolvedModelSpec) -> ContentBlock {
    ContentBlock::Text {
        text: omitted_content_message(usage, model),
    }
}

fn omitted_content_message(usage: MediaUsage, model: &ResolvedModelSpec) -> String {
    let provider = model.provider_label();
    let model = model.model_label();
    match (usage.image_input, usage.file_input) {
        (true, true) => format!(
            "[Attachment omitted: {provider} {model} does not support image or file input. Do not infer attachment contents; ask the user to describe it, paste relevant text, or switch models.]"
        ),
        (true, false) => format!(
            "[Image attachment omitted: {provider} {model} does not support image input. Do not infer image contents; ask the user to describe it or switch models.]"
        ),
        (false, true) => format!(
            "[File attachment omitted: {provider} {model} does not support file input. Do not infer file contents; ask the user to paste relevant text or switch models.]"
        ),
        (false, false) => format!(
            "[Attachment omitted: {provider} {model} does not support this attachment type. Do not infer attachment contents; ask the user to describe it or switch models.]"
        ),
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{ContentBlock, ModelCapabilities, Role, UniversalItem, UniversalRequest};

    fn text_only_model() -> ResolvedModelSpec {
        ResolvedModelSpec {
            provider_label: Some("DeepSeek".to_string()),
            model: "deepseek-v4-pro".to_string(),
            capabilities: ModelCapabilities {
                input_modalities: vec!["text".to_string()],
                ..ModelCapabilities::default()
            },
            extensions: Default::default(),
        }
    }

    #[test]
    fn replaces_unsupported_image_with_safe_text_placeholder() {
        let mut request = UniversalRequest {
            input: vec![UniversalItem::Message {
                role: Role::User,
                id: None,
                content: vec![ContentBlock::Image {
                    media_type: Some("image/png".to_string()),
                    url: Some("https://example.test/a.png".to_string()),
                    data: None,
                    extensions: Default::default(),
                }],
                extensions: Default::default(),
            }],
            ..UniversalRequest::default()
        };

        let result = sanitize_unsupported_media(&mut request, &text_only_model());

        assert!(result.changed());
        assert!(result.image_omitted);
        let UniversalItem::Message { content, .. } = &request.input[0] else {
            panic!("expected message");
        };
        let ContentBlock::Text { text } = &content[0] else {
            panic!("expected text placeholder");
        };
        assert!(text.contains("Image attachment omitted"));
        assert!(text.contains("Do not infer image contents"));
    }

    #[test]
    fn leaves_supported_image_unchanged() {
        let mut request = UniversalRequest {
            input: vec![UniversalItem::Message {
                role: Role::User,
                id: None,
                content: vec![ContentBlock::Image {
                    media_type: Some("image/png".to_string()),
                    url: Some("https://example.test/a.png".to_string()),
                    data: None,
                    extensions: Default::default(),
                }],
                extensions: Default::default(),
            }],
            ..UniversalRequest::default()
        };
        let model = ResolvedModelSpec {
            capabilities: ModelCapabilities {
                input_modalities: vec!["text".to_string(), "image".to_string()],
                ..ModelCapabilities::default()
            },
            ..ResolvedModelSpec::default()
        };

        let result = sanitize_unsupported_media(&mut request, &model);

        assert!(!result.changed());
        let UniversalItem::Message { content, .. } = &request.input[0] else {
            panic!("expected message");
        };
        assert!(matches!(content[0], ContentBlock::Image { .. }));
    }

    #[test]
    fn accepts_model_spec_json() {
        let mut request = UniversalRequest {
            input: vec![UniversalItem::Unknown {
                raw: json!({
                    "type": "input_file",
                    "file_url": "https://example.test/a.pdf"
                }),
            }],
            ..UniversalRequest::default()
        };

        let result = sanitize_unsupported_media_from_json(
            &mut request,
            json!({
                "providerLabel": "DeepSeek",
                "model": "deepseek-v4-pro",
                "capabilities": { "inputModalities": ["text"] }
            }),
        )
        .expect("json model spec is accepted");

        assert!(result.file_omitted);
        let UniversalItem::Message { content, .. } = &request.input[0] else {
            panic!("expected placeholder message");
        };
        assert!(matches!(content[0], ContentBlock::Text { .. }));
    }
}
