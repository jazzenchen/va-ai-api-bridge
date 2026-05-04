use serde_json::Value;

use crate::schema::anthropic;
use crate::ContentBlock;

use crate::translator::common::empty_extensions;

use super::media::anthropic_source_to_image;

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
