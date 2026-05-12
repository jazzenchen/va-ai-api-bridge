pub(crate) mod content;
mod reasoning;
mod tools;

pub(crate) use content::*;
pub(crate) use reasoning::*;
pub(crate) use tools::*;

use crate::schema::openai;
use crate::{FinishReason, GenerationConfig, Role, Usage};

use super::common::empty_extensions;

pub(crate) fn role_to_openai(role: Role) -> &'static str {
    match role {
        Role::Developer => "developer",
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

pub(crate) fn finish_from_openai(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("stop") => Some(FinishReason::Stop),
        Some("length") => Some(FinishReason::Length),
        Some("tool_calls") | Some("function_call") => Some(FinishReason::ToolCall),
        Some("content_filter") => Some(FinishReason::ContentFilter),
        Some(_) => Some(FinishReason::Unknown),
        None => None,
    }
}

pub(crate) fn openai_usage_to_universal(usage: Option<&openai::OpenAiUsage>) -> Option<Usage> {
    usage.map(|usage| Usage {
        input_tokens: usage.input_tokens.or(usage.prompt_tokens),
        output_tokens: usage.output_tokens.or(usage.completion_tokens),
        total_tokens: usage.total_tokens,
    })
}

pub(crate) fn generation_from_openai(
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_output_tokens: Option<u64>,
) -> GenerationConfig {
    GenerationConfig {
        temperature,
        top_p,
        max_output_tokens,
        extensions: empty_extensions(),
    }
}
