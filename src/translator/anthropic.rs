pub(crate) mod content;
mod reasoning;
mod tools;

pub(crate) use content::*;
pub(crate) use reasoning::*;
pub(crate) use tools::*;

use crate::schema::anthropic;
use crate::{FinishReason, GenerationConfig, Role, Usage};

use super::common::empty_extensions;

pub(crate) fn role_to_anthropic(role: Role) -> &'static str {
    match role {
        Role::System | Role::User | Role::Tool => "user",
        Role::Assistant => "assistant",
    }
}

pub(crate) fn finish_from_anthropic(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("end_turn") | Some("stop_sequence") => Some(FinishReason::Stop),
        Some("max_tokens") => Some(FinishReason::Length),
        Some("tool_use") => Some(FinishReason::ToolCall),
        Some(_) => Some(FinishReason::Unknown),
        None => None,
    }
}

pub(crate) fn anthropic_usage_to_universal(
    usage: Option<&anthropic::AnthropicUsage>,
) -> Option<Usage> {
    usage.map(|usage| {
        let input_tokens = usage.input_tokens.map(|tokens| {
            tokens
                + usage.cache_creation_input_tokens.unwrap_or(0)
                + usage.cache_read_input_tokens.unwrap_or(0)
        });
        Usage {
            input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: input_tokens
                .zip(usage.output_tokens)
                .map(|(input, output)| input + output),
        }
    })
}

pub(crate) fn generation_from_anthropic(
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
