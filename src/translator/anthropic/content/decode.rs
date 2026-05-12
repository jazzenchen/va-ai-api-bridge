use serde_json::Value;

use crate::schema::anthropic;
use crate::ContentBlock;

use crate::translator::common::empty_extensions;

use super::media::{anthropic_source_to_file, anthropic_source_to_image};

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
        "image" => {
            let image = anthropic_source_to_image(block.source.as_ref());
            ContentBlock::Image {
                media_type: image.media_type,
                url: image.url,
                data: image.data,
                extensions: empty_extensions(),
            }
        }
        "document" => {
            let file = anthropic_source_to_file(block.source.as_ref());
            let mut extensions = empty_extensions();
            if let Some(file_id) = file.file_id {
                extensions.insert("file_id".to_string(), Value::String(file_id));
            }
            ContentBlock::File {
                media_type: file.media_type,
                filename: block
                    .extra
                    .get("title")
                    .or_else(|| block.extra.get("filename"))
                    .or_else(|| block.extra.get("name"))
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
                url: file.url,
                data: file.data,
                extensions,
            }
        }
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
        "thinking" => ContentBlock::Reasoning {
            text: block.thinking.clone().or_else(|| block.text.clone()),
            encrypted: block.signature.clone(),
            extensions: empty_extensions(),
        },
        "redacted_thinking" => ContentBlock::Unknown {
            raw: serde_json::to_value(block).unwrap_or(Value::Null),
        },
        _ => ContentBlock::Unknown {
            raw: serde_json::to_value(block).unwrap_or(Value::Null),
        },
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

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use crate::schema::anthropic::AnthropicContentBlock;
    use crate::translator::anthropic::block_to_anthropic_block;
    use crate::ContentBlock;

    use super::anthropic_block_to_block;

    #[test]
    fn preserves_redacted_thinking_as_raw_anthropic_block() {
        let mut extra: crate::schema::anthropic::ExtraFields = Default::default();
        extra.insert("data".to_string(), json!("opaque"));
        let block = AnthropicContentBlock {
            kind: "redacted_thinking".to_string(),
            text: None,
            source: None,
            id: None,
            name: None,
            input: None,
            tool_use_id: None,
            content: None,
            thinking: None,
            signature: None,
            extra,
        };

        let decoded = anthropic_block_to_block(&block);
        let ContentBlock::Unknown { raw } = &decoded else {
            panic!("redacted thinking should stay raw");
        };
        assert_eq!(raw["type"], "redacted_thinking");
        assert_eq!(raw["data"], "opaque");

        let encoded = block_to_anthropic_block(&decoded);
        assert_eq!(encoded.kind, "redacted_thinking");
        assert_eq!(
            encoded.extra.get("data"),
            Some(&Value::String("opaque".to_string()))
        );
    }

    #[test]
    fn decodes_document_file_id_as_file_block() {
        let mut extra: crate::schema::anthropic::ExtraFields = Default::default();
        extra.insert("title".to_string(), json!("report.pdf"));
        let block = AnthropicContentBlock {
            kind: "document".to_string(),
            text: None,
            source: Some(json!({
                "type": "file",
                "file_id": "file_123"
            })),
            id: None,
            name: None,
            input: None,
            tool_use_id: None,
            content: None,
            thinking: None,
            signature: None,
            extra,
        };

        let decoded = anthropic_block_to_block(&block);
        let ContentBlock::File {
            filename,
            extensions,
            ..
        } = decoded
        else {
            panic!("document should decode as file");
        };
        assert_eq!(filename.as_deref(), Some("report.pdf"));
        assert_eq!(extensions.get("file_id"), Some(&json!("file_123")));
    }

    #[test]
    fn encodes_file_data_as_document_base64_source() {
        let block = ContentBlock::File {
            media_type: None,
            filename: Some("report.pdf".to_string()),
            url: None,
            data: Some("data:application/pdf;base64,AAAA".to_string()),
            extensions: crate::translator::common::empty_extensions(),
        };

        let encoded = block_to_anthropic_block(&block);
        assert_eq!(encoded.kind, "document");
        assert_eq!(encoded.extra.get("title"), Some(&json!("report.pdf")));
        assert_eq!(
            encoded.source,
            Some(json!({
                "type": "base64",
                "media_type": "application/pdf",
                "data": "AAAA"
            }))
        );
    }

    #[test]
    fn encodes_file_url_as_document_url_source() {
        let block = ContentBlock::File {
            media_type: Some("application/pdf".to_string()),
            filename: None,
            url: Some("https://example.test/report.pdf".to_string()),
            data: None,
            extensions: crate::translator::common::empty_extensions(),
        };

        let encoded = block_to_anthropic_block(&block);
        assert_eq!(encoded.kind, "document");
        assert_eq!(
            encoded.source,
            Some(json!({
                "type": "url",
                "url": "https://example.test/report.pdf"
            }))
        );
    }
}
