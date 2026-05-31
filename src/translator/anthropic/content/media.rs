use serde_json::{Map, Value};

use crate::Extensions;

pub(super) struct AnthropicImageSource {
    pub(super) media_type: Option<String>,
    pub(super) url: Option<String>,
    pub(super) data: Option<String>,
}

pub(super) struct AnthropicFileSource {
    pub(super) media_type: Option<String>,
    pub(super) url: Option<String>,
    pub(super) data: Option<String>,
    pub(super) file_id: Option<String>,
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

pub(super) fn anthropic_source_to_file(source: Option<&Value>) -> AnthropicFileSource {
    AnthropicFileSource {
        media_type: source
            .and_then(|source| source.get("media_type"))
            .and_then(Value::as_str)
            .map(ToString::to_string),
        url: source
            .and_then(|source| source.get("url").or_else(|| source.get("file_url")))
            .and_then(Value::as_str)
            .map(ToString::to_string),
        data: source
            .and_then(|source| source.get("data"))
            .and_then(Value::as_str)
            .map(ToString::to_string),
        file_id: source
            .and_then(|source| source.get("file_id"))
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
    if let Some(url) = url {
        source.insert("type".to_string(), Value::String("url".to_string()));
        source.insert("url".to_string(), Value::String(url.to_string()));
        return Value::Object(source);
    }

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
    for (key, value) in extensions {
        source.insert(key.clone(), value.clone());
    }
    Value::Object(source)
}

pub(super) fn anthropic_file_source(
    media_type: Option<&str>,
    url: Option<&str>,
    data: Option<&str>,
    extensions: &Extensions,
) -> Value {
    let mut source = Map::new();
    if let Some(Value::String(file_id)) = extensions.get("file_id") {
        source.insert("type".to_string(), Value::String("file".to_string()));
        source.insert("file_id".to_string(), Value::String(file_id.clone()));
        return Value::Object(source);
    }
    if let Some(url) = url {
        source.insert("type".to_string(), Value::String("url".to_string()));
        source.insert("url".to_string(), Value::String(url.to_string()));
        return Value::Object(source);
    }

    let (data_url_media_type, data) = data.map(split_data_url).unwrap_or((None, ""));
    source.insert("type".to_string(), Value::String("base64".to_string()));
    if let Some(media_type) = media_type.or(data_url_media_type) {
        source.insert(
            "media_type".to_string(),
            Value::String(media_type.to_string()),
        );
    }
    source.insert("data".to_string(), Value::String(data.to_string()));
    Value::Object(source)
}

fn split_data_url(data: &str) -> (Option<&str>, &str) {
    let Some(rest) = data.strip_prefix("data:") else {
        return (None, data);
    };
    let Some((metadata, payload)) = rest.split_once(',') else {
        return (None, data);
    };
    let media_type = metadata
        .split(';')
        .next()
        .filter(|media_type| !media_type.is_empty());
    (media_type, payload)
}
