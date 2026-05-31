use serde_json::{Map, Value};

pub(in crate::translator::gemini_generate_content) const VA_MODEL_KEY: &str = "__va_model";
pub(in crate::translator::gemini_generate_content) const VA_STREAM_KEY: &str = "__va_stream";
pub(in crate::translator::gemini_generate_content) const GEMINI_THOUGHT_SIGNATURE_KEY: &str =
    "thoughtSignature";

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

pub(in crate::translator::gemini_generate_content) fn model_from_route_segment(
    value: &str,
) -> String {
    value.strip_prefix("models/").unwrap_or(value).to_string()
}
pub(in crate::translator::gemini_generate_content) fn field<'a>(
    object: &'a Map<String, Value>,
    camel_case: &str,
    snake_case: &str,
) -> Option<&'a Value> {
    object.get(camel_case).or_else(|| object.get(snake_case))
}
