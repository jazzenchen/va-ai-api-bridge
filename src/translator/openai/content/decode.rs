use serde_json::Value;

use crate::schema::openai;
use crate::ContentBlock;

use crate::translator::common::empty_extensions;

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
        "file" | "input_file" => value_to_file_or_unknown(
            part.file
                .as_ref()
                .unwrap_or(&Value::Object(part.extra.clone().into_iter().collect())),
        ),
        _ => ContentBlock::Unknown {
            raw: serde_json::to_value(part).unwrap_or(Value::Null),
        },
    }
}

fn value_to_image_or_unknown(value: &Value) -> ContentBlock {
    let Some(object) = value.as_object() else {
        return ContentBlock::Unknown { raw: value.clone() };
    };
    let mut extensions = empty_extensions();
    for key in ["file_id", "detail"] {
        if let Some(value) = object.get(key) {
            extensions.insert(key.to_string(), value.clone());
        }
    }
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
        extensions,
    }
}

fn value_to_file_or_unknown(value: &Value) -> ContentBlock {
    let Some(object) = value.as_object() else {
        return ContentBlock::Unknown { raw: value.clone() };
    };
    let mut extensions = empty_extensions();
    if let Some(file_id) = object.get("file_id") {
        extensions.insert("file_id".to_string(), file_id.clone());
    }
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
        extensions,
    }
}
