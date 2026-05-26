use base64::Engine as _;
use serde_json::{json, Map, Value};

pub(super) fn convert_text_file_parts_to_text(chat_request: &mut Value) {
    let Some(messages) = chat_request
        .get_mut("messages")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    for message in messages {
        let Some(content) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in content {
            if let Some(text_part) = text_part_from_file_part(part) {
                *part = text_part;
            }
        }
    }
}

fn text_part_from_file_part(part: &Value) -> Option<Value> {
    let object = part.as_object()?;
    let kind = object.get("type").and_then(Value::as_str);
    let file_like = matches!(kind, Some("file" | "input_file"))
        || object.contains_key("file")
        || object.contains_key("file_data")
        || object.contains_key("fileData")
        || object.contains_key("filename");
    if !file_like {
        return None;
    }

    let nested = object.get("file").and_then(Value::as_object);
    let filename = string_field(object, nested, &["filename", "name"]);
    let media_type = string_field(object, nested, &["media_type", "mime_type", "mime"]);
    let data = string_field(object, nested, &["file_data", "fileData", "data"])?;
    if is_http_url(&data) {
        return None;
    }
    let text = text_from_file_data(&data, media_type.as_deref(), filename.as_deref())?;
    let label = filename.unwrap_or_else(|| "attachment".to_string());
    Some(json!({
        "type": "text",
        "text": format!("Attached file: {label}\n\n{text}")
    }))
}

fn string_field(
    object: &Map<String, Value>,
    nested: Option<&Map<String, Value>>,
    keys: &[&str],
) -> Option<String> {
    keys.iter()
        .find_map(|key| {
            object
                .get(*key)
                .or_else(|| nested.and_then(|value| value.get(*key)))
        })
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn text_from_file_data(
    value: &str,
    media_type: Option<&str>,
    filename: Option<&str>,
) -> Option<String> {
    if let Some((mime, data)) = parse_data_url(value) {
        if !is_text_like(Some(mime), filename) {
            return None;
        }
        let bytes = if data.base64 {
            base64::engine::general_purpose::STANDARD
                .decode(data.payload)
                .ok()?
        } else {
            percent_decode(data.payload)?
        };
        return String::from_utf8(bytes).ok();
    }

    if is_text_like(media_type, filename) {
        return Some(value.to_string());
    }
    None
}

struct DataUrl<'a> {
    payload: &'a str,
    base64: bool,
}

fn parse_data_url(value: &str) -> Option<(&str, DataUrl<'_>)> {
    let rest = value.strip_prefix("data:")?;
    let (metadata, payload) = rest.split_once(',')?;
    let mut parts = metadata.split(';');
    let mime = parts
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or("text/plain");
    let base64 = parts.any(|part| part.eq_ignore_ascii_case("base64"));
    Some((mime, DataUrl { payload, base64 }))
}

fn percent_decode(value: &str) -> Option<Vec<u8>> {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
                out.push(u8::from_str_radix(hex, 16).ok()?);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    Some(out)
}

fn is_text_like(media_type: Option<&str>, filename: Option<&str>) -> bool {
    let media_type = media_type
        .map(|value| {
            value
                .split(';')
                .next()
                .unwrap_or(value)
                .trim()
                .to_ascii_lowercase()
        })
        .unwrap_or_default();
    if media_type.starts_with("text/") {
        return true;
    }
    if matches!(
        media_type.as_str(),
        "application/json"
            | "application/xml"
            | "application/javascript"
            | "application/typescript"
            | "application/x-javascript"
            | "application/x-typescript"
            | "application/yaml"
            | "application/x-yaml"
            | "application/toml"
            | "application/x-toml"
    ) {
        return true;
    }

    filename
        .and_then(|filename| filename.rsplit('.').next())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "txt"
                    | "md"
                    | "markdown"
                    | "json"
                    | "jsonl"
                    | "yaml"
                    | "yml"
                    | "toml"
                    | "xml"
                    | "html"
                    | "css"
                    | "js"
                    | "jsx"
                    | "ts"
                    | "tsx"
                    | "rs"
                    | "go"
                    | "py"
                    | "java"
                    | "kt"
                    | "swift"
                    | "c"
                    | "h"
                    | "cc"
                    | "cpp"
                    | "hpp"
                    | "cs"
                    | "php"
                    | "rb"
                    | "sh"
                    | "zsh"
                    | "bash"
                    | "sql"
                    | "csv"
                    | "log"
            )
        })
        .unwrap_or(false)
}

fn is_http_url(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn converts_text_file_parts_to_text() {
        let mut request = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "input_file",
                    "filename": "hello.txt",
                    "file_data": "data:text/plain;base64,5L2g5aW9"
                }]
            }]
        });

        convert_text_file_parts_to_text(&mut request);

        assert_eq!(request["messages"][0]["content"][0]["type"], "text");
        assert!(request["messages"][0]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("你好"));
    }
}
