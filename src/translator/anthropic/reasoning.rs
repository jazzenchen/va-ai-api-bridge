use serde_json::{json, Value};

use crate::ReasoningConfig;

use super::super::common::empty_extensions;

pub(crate) fn reasoning_from_anthropic_thinking(thinking: Value) -> ReasoningConfig {
    let mut extensions = empty_extensions();
    let thinking_type = thinking
        .get("type")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    if let Some(thinking_type) = &thinking_type {
        extensions.insert(
            "anthropicThinkingType".to_string(),
            Value::String(thinking_type.clone()),
        );
    }

    ReasoningConfig {
        effort: None,
        budget_tokens: thinking.get("budget_tokens").and_then(Value::as_u64),
        visible: None,
        extensions,
    }
}

pub(crate) fn anthropic_thinking_from_reasoning(
    reasoning: &ReasoningConfig,
    max_tokens: Option<u64>,
) -> Option<Value> {
    let effort = reasoning.effort.as_deref();
    if matches!(effort, Some("none" | "disabled")) {
        return None;
    }

    let wants_thinking = reasoning.budget_tokens.is_some()
        || reasoning.visible == Some(true)
        || matches!(
            effort,
            Some("minimal" | "low" | "medium" | "high" | "xhigh" | "enabled" | "adaptive")
        )
        || reasoning
            .extensions
            .get("anthropicThinkingType")
            .and_then(Value::as_str)
            .is_some_and(|kind| matches!(kind, "enabled" | "adaptive"));
    if !wants_thinking {
        return None;
    }

    let effective_max_tokens = max_tokens.or(Some(4096));
    let desired_budget = reasoning
        .budget_tokens
        .or_else(|| anthropic_budget_for_effort(effort));
    let budget_tokens = desired_budget
        .and_then(|budget| clamp_anthropic_thinking_budget(budget, effective_max_tokens))?;

    Some(json!({
        "type": "enabled",
        "budget_tokens": budget_tokens
    }))
}

fn anthropic_budget_for_effort(effort: Option<&str>) -> Option<u64> {
    Some(match effort {
        Some("minimal" | "low") => 1024,
        Some("high") => 4096,
        Some("xhigh") => 8192,
        Some("medium" | "enabled" | "adaptive") | None => 2048,
        Some(_) => return None,
    })
}

fn clamp_anthropic_thinking_budget(budget: u64, max_tokens: Option<u64>) -> Option<u64> {
    let Some(max_tokens) = max_tokens else {
        return Some(budget.max(1024));
    };
    if max_tokens <= 1024 {
        return None;
    }
    let cap = if max_tokens >= 4096 {
        max_tokens / 2
    } else {
        max_tokens - 1
    };
    Some(budget.max(1024).min(cap).min(max_tokens - 1))
}
