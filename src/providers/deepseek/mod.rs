mod anthropic_order;
mod reasoning;
mod tool_history;

use std::collections::HashMap;

use serde_json::{json, Value};

use self::anthropic_order::repair_anthropic_thinking_tool_use_order;
pub(super) use self::reasoning::{
    collect_reasoning_from_anthropic_input, collect_reasoning_from_gemini_input,
    collect_reasoning_from_responses_input, inject_reasoning_content,
    strip_anthropic_reasoning_content_blocks, RequestReasoning,
};
pub(super) use self::tool_history::{
    collect_tool_outputs_from_chat_request, collect_tool_outputs_from_responses_input,
    repair_tool_call_history,
};
use super::{DeepSeekBridgeSettings, ProviderRequestSource};

const MISSING_REASONING_CONTENT_FALLBACK: &str =
    "Previous DeepSeek reasoning content is unavailable from the local bridge.";
const MISSING_TOOL_OUTPUT_FALLBACK: &str = "Tool output unavailable from the local bridge.";

#[derive(Debug, Clone)]
pub struct DeepSeekBridgeAdapter {
    settings: DeepSeekBridgeSettings,
}

impl DeepSeekBridgeAdapter {
    pub fn new(settings: DeepSeekBridgeSettings) -> Self {
        Self { settings }
    }

    pub fn prepare_chat_request(
        &mut self,
        source: ProviderRequestSource,
        original_request: &Value,
        chat_request: &mut Value,
    ) {
        if source == ProviderRequestSource::AnthropicMessages {
            strip_anthropic_reasoning_content_blocks(chat_request);
        }

        let tool_outputs = self.collect_tool_outputs(original_request, chat_request);
        repair_tool_call_history(&tool_outputs, chat_request);

        if self.should_replay_reasoning_content(source) {
            let mut reasoning = RequestReasoning::default();
            match source {
                ProviderRequestSource::OpenAiResponses => {
                    collect_reasoning_from_responses_input(&mut reasoning, original_request);
                }
                ProviderRequestSource::AnthropicMessages => {
                    collect_reasoning_from_anthropic_input(&mut reasoning, original_request);
                }
                ProviderRequestSource::GeminiGenerateContent => {
                    collect_reasoning_from_gemini_input(&mut reasoning, original_request);
                }
                ProviderRequestSource::OpenAiChat => {}
            }
            inject_reasoning_content(&reasoning, chat_request, MISSING_REASONING_CONTENT_FALLBACK);
        }

        let Some(request) = chat_request.as_object_mut() else {
            return;
        };
        let thinking =
            self.settings.thinking && !has_forced_tool_choice(request.get("tool_choice"));

        request.insert(
            "thinking".to_string(),
            json!({
                "type": if thinking {
                    "enabled"
                } else {
                    "disabled"
                },
            }),
        );
    }

    pub fn prepare_anthropic_request(
        &mut self,
        source: ProviderRequestSource,
        request: &mut Value,
    ) {
        repair_anthropic_thinking_tool_use_order(request);
        let Some(object) = request.as_object_mut() else {
            return;
        };
        if self.anthropic_thinking_must_be_disabled(source, object) {
            object.insert(
                "thinking".to_string(),
                json!({
                    "type": "disabled"
                }),
            );
        }
    }

    fn anthropic_thinking_must_be_disabled(
        &self,
        source: ProviderRequestSource,
        request: &serde_json::Map<String, Value>,
    ) -> bool {
        if !self.anthropic_thinking_is_enabled(request) {
            return false;
        }
        has_forced_tool_choice(request.get("tool_choice"))
            || (source != ProviderRequestSource::AnthropicMessages
                && has_anthropic_native_web_search_tool(request.get("tools")))
    }

    fn anthropic_thinking_is_enabled(&self, request: &serde_json::Map<String, Value>) -> bool {
        self.settings.thinking || request_anthropic_thinking_is_enabled(request.get("thinking"))
    }

    fn should_replay_reasoning_content(&self, source: ProviderRequestSource) -> bool {
        self.settings.thinking
            && self.settings.replay_reasoning_content
            && source.supports_deepseek_reasoning_replay()
    }

    fn collect_tool_outputs(
        &self,
        original_request: &Value,
        chat_request: &Value,
    ) -> HashMap<String, String> {
        let mut outputs = HashMap::new();
        collect_tool_outputs_from_responses_input(original_request, &mut outputs);
        collect_tool_outputs_from_chat_request(chat_request, &mut outputs);
        outputs
    }
}

fn has_forced_tool_choice(tool_choice: Option<&Value>) -> bool {
    match tool_choice {
        Some(Value::String(value)) => matches!(value.as_str(), "required" | "any"),
        Some(Value::Object(object)) => object
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| matches!(kind, "function" | "tool" | "any")),
        _ => false,
    }
}

fn has_anthropic_native_web_search_tool(tools: Option<&Value>) -> bool {
    tools
        .and_then(Value::as_array)
        .is_some_and(|tools| tools.iter().any(is_anthropic_native_web_search_tool))
}

fn is_anthropic_native_web_search_tool(tool: &Value) -> bool {
    tool.get("type").and_then(Value::as_str) == Some("web_search_20250305")
}

fn request_anthropic_thinking_is_enabled(thinking: Option<&Value>) -> bool {
    thinking
        .and_then(Value::as_object)
        .and_then(|thinking| thinking.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|kind| matches!(kind, "enabled" | "adaptive"))
}

#[cfg(test)]
mod tests;
