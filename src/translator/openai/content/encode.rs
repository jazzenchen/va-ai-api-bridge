use serde_json::{Map, Value};

use crate::schema::openai;
use crate::ContentBlock;

use super::media::openai_media_url;

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

pub(crate) fn blocks_to_plain_text(blocks: &[ContentBlock]) -> Option<String> {
    let text = blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::Unknown { raw } => raw.as_str(),
            _ => None,
        })
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    (!text.is_empty()).then_some(text)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAiResponsesContentDirection {
    Input,
    Output,
}

pub(crate) fn blocks_to_openai_responses_part_array(
    blocks: &[ContentBlock],
    direction: OpenAiResponsesContentDirection,
) -> Option<openai::OpenAiContent> {
    match direction {
        OpenAiResponsesContentDirection::Input => {
            blocks_to_openai_part_array(blocks, "input_text", "input_image")
        }
        OpenAiResponsesContentDirection::Output => {
            if blocks.is_empty() {
                return None;
            }
            Some(openai::OpenAiContent::Parts(
                blocks
                    .iter()
                    .map(block_to_openai_response_output_part)
                    .collect(),
            ))
        }
    }
}

fn blocks_to_openai_part_array(
    blocks: &[ContentBlock],
    text_kind: &str,
    image_kind: &str,
) -> Option<openai::OpenAiContent> {
    if blocks.is_empty() {
        return None;
    }
    Some(openai::OpenAiContent::Parts(
        blocks
            .iter()
            .map(|block| block_to_openai_part(block, text_kind, image_kind))
            .collect(),
    ))
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
            part.text = Some(text.clone());
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
                    url: url
                        .clone()
                        .or_else(|| {
                            data.as_ref()
                                .map(|data| openai_media_url(media_type.as_deref(), data))
                        })
                        .unwrap_or_default(),
                    detail: None,
                    extra: Default::default(),
                });
            } else {
                if let Some(url) = url {
                    part.extra
                        .insert("image_url".to_string(), Value::String(url.clone()));
                } else if let Some(data) = data {
                    part.extra.insert(
                        "image_url".to_string(),
                        Value::String(openai_media_url(media_type.as_deref(), data)),
                    );
                }
                for (key, value) in extensions {
                    part.extra.insert(key.clone(), value.clone());
                }
            }
        }
        ContentBlock::File {
            filename,
            url,
            data,
            extensions,
            ..
        } => {
            if image_kind == "input_image" {
                part.kind = "input_file".to_string();
                if let Some(Value::String(file_id)) = extensions.get("file_id") {
                    part.extra
                        .insert("file_id".to_string(), Value::String(file_id.clone()));
                } else if let Some(url) = url {
                    part.extra
                        .insert("file_url".to_string(), Value::String(url.clone()));
                } else if let Some(data) = data {
                    if let Some(filename) = filename {
                        part.extra
                            .insert("filename".to_string(), Value::String(filename.clone()));
                    }
                    part.extra
                        .insert("file_data".to_string(), Value::String(data.clone()));
                }
                for (key, value) in extensions {
                    part.extra
                        .entry(key.clone())
                        .or_insert_with(|| value.clone());
                }
            } else {
                part.kind = "file".to_string();
                let mut file = Map::new();
                if let Some(filename) = filename {
                    file.insert("filename".to_string(), Value::String(filename.clone()));
                }
                if let Some(Value::String(file_id)) = extensions.get("file_id") {
                    file.insert("file_id".to_string(), Value::String(file_id.clone()));
                } else if let Some(data) = data.as_ref().or(url.as_ref()) {
                    file.insert("file_data".to_string(), Value::String(data.clone()));
                }
                for (key, value) in extensions {
                    file.entry(key.clone()).or_insert_with(|| value.clone());
                }
                part.file = Some(Value::Object(file));
            }
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

fn block_to_openai_response_output_part(block: &ContentBlock) -> openai::OpenAiContentPart {
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
            part.kind = "output_text".to_string();
            part.text = Some(text.clone());
        }
        ContentBlock::Unknown { raw } => {
            if let Ok(raw_part) = serde_json::from_value::<openai::OpenAiContentPart>(raw.clone()) {
                return raw_part;
            }
            part.kind = "unknown".to_string();
            part.extra.insert("raw".to_string(), raw.clone());
        }
        _ => {
            part.kind = "unknown".to_string();
            part.extra.insert(
                "raw".to_string(),
                serde_json::to_value(block).unwrap_or(Value::Null),
            );
        }
    }

    part
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use crate::translator::common::empty_extensions;
    use crate::ContentBlock;

    use super::{
        blocks_to_openai_content, blocks_to_openai_responses_part_array,
        OpenAiResponsesContentDirection,
    };

    #[test]
    fn chat_image_data_is_encoded_as_image_url_data_url() {
        let content = blocks_to_openai_content(
            &[ContentBlock::Image {
                media_type: Some("image/png".to_string()),
                url: None,
                data: Some("abc123".to_string()),
                extensions: empty_extensions(),
            }],
            "text",
            "image_url",
        );

        let value = serde_json::to_value(content).unwrap();
        assert_eq!(value[0]["type"], "image_url");
        assert_eq!(value[0]["image_url"]["url"], "data:image/png;base64,abc123");
    }

    #[test]
    fn responses_image_data_is_encoded_as_input_image() {
        let content = blocks_to_openai_responses_part_array(
            &[ContentBlock::Image {
                media_type: Some("image/png".to_string()),
                url: None,
                data: Some("abc123".to_string()),
                extensions: empty_extensions(),
            }],
            OpenAiResponsesContentDirection::Input,
        );

        let value = serde_json::to_value(content).unwrap();
        assert_eq!(value[0]["type"], "input_image");
        assert_eq!(value[0]["image_url"], "data:image/png;base64,abc123");
    }

    #[test]
    fn responses_file_url_is_encoded_as_input_file() {
        let content = blocks_to_openai_responses_part_array(
            &[ContentBlock::File {
                media_type: Some("application/pdf".to_string()),
                filename: Some("paper.pdf".to_string()),
                url: Some("https://example.test/paper.pdf".to_string()),
                data: None,
                extensions: empty_extensions(),
            }],
            OpenAiResponsesContentDirection::Input,
        );

        let value = serde_json::to_value(content).unwrap();
        assert_eq!(value[0]["type"], "input_file");
        assert_eq!(value[0]["file_url"], "https://example.test/paper.pdf");
        assert_eq!(value[0].get("file_data"), None);
    }

    #[test]
    fn chat_file_data_uses_openai_compatible_file_data_key() {
        let content = blocks_to_openai_content(
            &[ContentBlock::File {
                media_type: Some("application/pdf".to_string()),
                filename: Some("paper.pdf".to_string()),
                url: None,
                data: Some("data:application/pdf;base64,AAAA".to_string()),
                extensions: empty_extensions(),
            }],
            "text",
            "image_url",
        );

        let value = serde_json::to_value(content).unwrap();
        assert_eq!(value[0]["type"], "file");
        assert_eq!(value[0]["file"]["filename"], "paper.pdf");
        assert_eq!(
            value[0]["file"]["file_data"],
            "data:application/pdf;base64,AAAA"
        );
        assert_eq!(value[0]["file"].get("data"), None);
    }

    #[test]
    fn chat_file_id_is_preserved() {
        let mut extensions = empty_extensions();
        extensions.insert("file_id".to_string(), json!("file_123"));

        let content = blocks_to_openai_content(
            &[ContentBlock::File {
                media_type: None,
                filename: None,
                url: None,
                data: None,
                extensions,
            }],
            "text",
            "image_url",
        );

        let value = serde_json::to_value(content).unwrap();
        assert_eq!(
            value[0]["file"]["file_id"],
            Value::String("file_123".to_string())
        );
    }
}
