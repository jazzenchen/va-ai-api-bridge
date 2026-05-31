use serde_json::{json, Map, Value};

use crate::translator::common;
use crate::{FinishReason, GenerationConfig, Role, Usage};

use super::field;

pub(in crate::translator::gemini_generate_content) fn gemini_role_to_universal(role: &str) -> Role {
    match role {
        "model" => Role::Assistant,
        "function" => Role::Tool,
        _ => Role::User,
    }
}

pub(in crate::translator::gemini_generate_content) fn universal_role_to_gemini(
    role: Role,
) -> &'static str {
    match role {
        Role::Assistant => "model",
        Role::Tool => "function",
        Role::Developer | Role::System | Role::User => "user",
    }
}

pub(in crate::translator::gemini_generate_content) fn finish_reason_from_gemini(
    value: &str,
) -> FinishReason {
    match value {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => {
            FinishReason::ContentFilter
        }
        "MALFORMED_FUNCTION_CALL" => FinishReason::ToolCall,
        _ => FinishReason::Unknown,
    }
}

pub(in crate::translator::gemini_generate_content) fn finish_reason_to_gemini(
    reason: FinishReason,
) -> &'static str {
    match reason {
        FinishReason::Stop => "STOP",
        FinishReason::Length => "MAX_TOKENS",
        FinishReason::ToolCall => "STOP",
        FinishReason::ContentFilter => "SAFETY",
        FinishReason::Error => "OTHER",
        FinishReason::Unknown => "FINISH_REASON_UNSPECIFIED",
    }
}

pub(in crate::translator::gemini_generate_content) fn has_finish_reason(raw: &Value) -> bool {
    raw.get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|candidate| candidate.get("finishReason").is_some())
}

pub(in crate::translator::gemini_generate_content) fn usage_from_gemini(
    value: Option<&Value>,
) -> Option<Usage> {
    let value = value?;
    Some(Usage {
        input_tokens: value.get("promptTokenCount").and_then(Value::as_u64),
        output_tokens: value.get("candidatesTokenCount").and_then(Value::as_u64),
        total_tokens: value.get("totalTokenCount").and_then(Value::as_u64),
    })
}

pub(in crate::translator::gemini_generate_content) fn usage_to_gemini(usage: &Usage) -> Value {
    let mut out = Map::new();
    if let Some(input_tokens) = usage.input_tokens {
        out.insert("promptTokenCount".to_string(), json!(input_tokens));
    }
    if let Some(output_tokens) = usage.output_tokens {
        out.insert("candidatesTokenCount".to_string(), json!(output_tokens));
    }
    if let Some(total_tokens) = usage.total_tokens {
        out.insert("totalTokenCount".to_string(), json!(total_tokens));
    }
    Value::Object(out)
}

pub(in crate::translator::gemini_generate_content) fn generation_from_gemini(
    value: Option<&Value>,
) -> GenerationConfig {
    let Some(object) = value.and_then(Value::as_object) else {
        return GenerationConfig::default();
    };
    GenerationConfig {
        temperature: object.get("temperature").and_then(Value::as_f64),
        top_p: field(object, "topP", "top_p").and_then(Value::as_f64),
        max_output_tokens: field(object, "maxOutputTokens", "max_output_tokens")
            .and_then(Value::as_u64),
        extensions: common::empty_extensions(),
    }
}

pub(in crate::translator::gemini_generate_content) fn generation_to_gemini(
    generation: &GenerationConfig,
) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(temperature) = generation.temperature {
        out.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(top_p) = generation.top_p {
        out.insert("topP".to_string(), json!(top_p));
    }
    if let Some(max_output_tokens) = generation.max_output_tokens {
        out.insert("maxOutputTokens".to_string(), json!(max_output_tokens));
    }
    out
}
