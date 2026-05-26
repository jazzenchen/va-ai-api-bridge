use std::collections::HashMap;

use serde_json::{json, Value};

use super::deepseek::{
    collect_reasoning_from_anthropic_input, collect_reasoning_from_gemini_input,
    collect_reasoning_from_responses_input, collect_tool_outputs_from_chat_request,
    collect_tool_outputs_from_responses_input, inject_reasoning_content, repair_tool_call_history,
    strip_anthropic_reasoning_content_blocks, RequestReasoning,
};
use super::ProviderRequestSource;

const MISSING_REASONING_CONTENT_FALLBACK: &str =
    "Previous MiMo reasoning_content is unavailable from the local bridge.";

#[derive(Debug, Clone, Default)]
pub struct MimoBridgeAdapter;

impl MimoBridgeAdapter {
    pub fn prepare_chat_request(
        &mut self,
        source: ProviderRequestSource,
        original_request: &Value,
        chat_request: &mut Value,
    ) {
        if source == ProviderRequestSource::AnthropicMessages {
            strip_anthropic_reasoning_content_blocks(chat_request);
        }

        let mut tool_outputs = HashMap::new();
        collect_tool_outputs_from_responses_input(original_request, &mut tool_outputs);
        collect_tool_outputs_from_chat_request(chat_request, &mut tool_outputs);
        repair_tool_call_history(&tool_outputs, chat_request);

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

        if let Some(object) = chat_request.as_object_mut() {
            object.insert("thinking".to_string(), json!({ "type": "enabled" }));
        }
    }

    pub fn normalize_chat_response(&mut self, response: &mut Value) {
        normalize_null_tool_calls(response);
    }
}

fn normalize_null_tool_calls(value: &mut Value) {
    let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };

    for choice in choices {
        normalize_message_tool_calls(choice, "message");
        normalize_message_tool_calls(choice, "delta");
    }
}

fn normalize_message_tool_calls(choice: &mut Value, key: &str) {
    let Some(message) = choice.get_mut(key).and_then(Value::as_object_mut) else {
        return;
    };
    if matches!(message.get("tool_calls"), Some(Value::Null)) {
        message.insert("tool_calls".to_string(), Value::Array(Vec::new()));
    }
}

#[cfg(test)]
mod tests {
    use crate::{DecodeState, OpenAiChatTranslator, WireTranslator};
    use serde_json::json;

    use super::super::reasoning_blob::encode_reasoning_content;
    use super::super::ProviderRequestSource;
    use super::MimoBridgeAdapter;

    #[test]
    fn enables_mimo_thinking_and_replays_responses_reasoning_content() {
        let original_request = json!({
            "input": [
                {
                    "type": "reasoning",
                    "encrypted_content": encode_reasoning_content("Call pwd, then answer.")
                },
                {
                    "type": "function_call",
                    "call_id": "call_pwd",
                    "name": "exec_command",
                    "arguments": "{\"cmd\":\"pwd\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_pwd",
                    "output": "/tmp/project"
                }
            ]
        });
        let mut chat_request = json!({
            "model": "mimo-v2.5-pro",
            "messages": [{
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_pwd",
                    "type": "function",
                    "function": { "name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}" }
                }]
            }, {
                "role": "tool",
                "tool_call_id": "call_pwd",
                "content": "/tmp/project"
            }]
        });
        let mut adapter = MimoBridgeAdapter;

        adapter.prepare_chat_request(
            ProviderRequestSource::OpenAiResponses,
            &original_request,
            &mut chat_request,
        );

        assert_eq!(chat_request["thinking"], json!({ "type": "enabled" }));
        assert_eq!(
            chat_request["messages"][0]["reasoning_content"],
            "Call pwd, then answer."
        );
    }

    #[test]
    fn fills_missing_mimo_reasoning_content_for_tool_history() {
        let mut chat_request = json!({
            "model": "mimo-v2.5-pro",
            "messages": [{
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_old",
                    "type": "function",
                    "function": { "name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}" }
                }]
            }]
        });
        let mut adapter = MimoBridgeAdapter;

        adapter.prepare_chat_request(
            ProviderRequestSource::OpenAiChat,
            &chat_request.clone(),
            &mut chat_request,
        );

        assert_eq!(
            chat_request["messages"][0]["reasoning_content"],
            super::MISSING_REASONING_CONTENT_FALLBACK
        );
    }

    #[test]
    fn normalizes_mimo_null_tool_calls_for_chat_completion_response() {
        let mut response = json!({
            "id": "chatcmpl_mimo",
            "model": "mimo-v2.5-pro",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "OK",
                    "tool_calls": null,
                    "reasoning_content": "Answer briefly."
                }
            }]
        });
        let mut adapter = MimoBridgeAdapter;

        adapter.normalize_chat_response(&mut response);

        assert_eq!(response["choices"][0]["message"]["tool_calls"], json!([]));
        OpenAiChatTranslator
            .decode_response(response)
            .expect("normalized MiMo response decodes");
    }

    #[test]
    fn normalizes_mimo_null_tool_calls_for_chat_stream_chunk() {
        let mut chunk = json!({
            "id": "chatcmpl_mimo",
            "model": "mimo-v2.5-pro",
            "choices": [{
                "index": 0,
                "finish_reason": null,
                "delta": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": null,
                    "reasoning_content": null
                }
            }]
        });
        let mut adapter = MimoBridgeAdapter;
        let mut state = DecodeState::default();

        adapter.normalize_chat_response(&mut chunk);

        assert_eq!(chunk["choices"][0]["delta"]["tool_calls"], json!([]));
        OpenAiChatTranslator
            .decode_stream_chunk(chunk, &mut state)
            .expect("normalized MiMo stream chunk decodes");
    }
}
