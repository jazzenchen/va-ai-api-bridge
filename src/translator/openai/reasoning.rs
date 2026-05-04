use crate::schema::openai;
use crate::ReasoningConfig;

use super::super::common::value_extensions;

pub(crate) fn reasoning_from_openai(
    reasoning: Option<openai::OpenAiReasoning>,
) -> Option<ReasoningConfig> {
    reasoning.map(|reasoning| ReasoningConfig {
        effort: reasoning.effort,
        budget_tokens: None,
        visible: None,
        extensions: value_extensions(reasoning.extra),
    })
}

pub(crate) fn openai_reasoning_effort(effort: Option<&str>) -> Option<&str> {
    match effort {
        Some("none" | "minimal" | "low" | "medium" | "high" | "xhigh") => effort,
        _ => None,
    }
}
