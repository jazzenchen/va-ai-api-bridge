use serde_json::{Map, Value};

use crate::Extensions;

pub(super) struct AnthropicImageSource {
    pub(super) media_type: Option<String>,
    pub(super) url: Option<String>,
    pub(super) data: Option<String>,
}

pub(super) fn anthropic_source_to_image(source: Option<&Value>) -> AnthropicImageSource {
    AnthropicImageSource {
        media_type: source
            .and_then(|source| source.get("media_type"))
            .and_then(Value::as_str)
            .map(ToString::to_string),
        url: source
            .and_then(|source| source.get("url").or_else(|| source.get("image_url")))
            .and_then(Value::as_str)
            .map(ToString::to_string),
        data: source
            .and_then(|source| source.get("data"))
            .and_then(Value::as_str)
            .map(ToString::to_string),
    }
}

pub(super) fn anthropic_image_source(
    media_type: Option<&str>,
    url: Option<&str>,
    data: Option<&str>,
    extensions: &Extensions,
) -> Value {
    let mut source = Map::new();
    source.insert("type".to_string(), Value::String("base64".to_string()));
    if let Some(media_type) = media_type {
        source.insert(
            "media_type".to_string(),
            Value::String(media_type.to_string()),
        );
    }
    if let Some(data) = data {
        source.insert("data".to_string(), Value::String(data.to_string()));
    }
    if let Some(url) = url {
        source.insert("url".to_string(), Value::String(url.to_string()));
    }
    for (key, value) in extensions {
        source.insert(key.clone(), value.clone());
    }
    Value::Object(source)
}
