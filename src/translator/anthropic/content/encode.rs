use serde_json::Value;

use crate::schema::anthropic;
use crate::ContentBlock;

use super::media::anthropic_image_source;

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
            content_block.source = Some(anthropic_image_source(
                media_type.as_deref(),
                url.as_deref(),
                data.as_deref(),
                extensions,
            ));
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
