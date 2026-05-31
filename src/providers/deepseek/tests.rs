use serde_json::json;

use super::super::reasoning_blob::encode_reasoning_content;
use super::{DeepSeekBridgeAdapter, DeepSeekBridgeSettings, ProviderRequestSource};

#[test]
fn default_settings_disable_thinking_for_existing_profiles() {
    let mut adapter = DeepSeekBridgeAdapter::new(DeepSeekBridgeSettings::default());
    let mut request = json!({
        "model": "deepseek-v4-flash",
        "messages": [{ "role": "user", "content": "hello" }],
    });

    adapter.prepare_chat_request(
        ProviderRequestSource::OpenAiResponses,
        &json!({ "input": "hello" }),
        &mut request,
    );

    assert_eq!(request["thinking"]["type"], "disabled");
}

#[test]
fn replays_reasoning_content_from_responses_history() {
    let settings = thinking_settings();
    let original_request = json!({
        "input": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
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
                "output": "/Users/jazzen/Development"
            }
        ]
    });
    let mut chat_request = json!({
        "model": "deepseek-v4-flash",
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
            "content": "/Users/jazzen/Development"
        }]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(settings);

    adapter.prepare_chat_request(
        ProviderRequestSource::OpenAiResponses,
        &original_request,
        &mut chat_request,
    );

    assert_eq!(
        chat_request["messages"][0]["reasoning_content"],
        "Call pwd, then answer."
    );
}

#[test]
fn replays_reasoning_text_from_responses_history() {
    let settings = thinking_settings();
    let original_request = json!({
        "input": [
            {
                "type": "reasoning",
                "content": [{
                    "type": "reasoning_text",
                    "text": "Use the tool result before answering."
                }]
            },
            {
                "type": "function_call",
                "call_id": "call_ls",
                "name": "exec_command",
                "arguments": "{\"cmd\":\"ls\"}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_ls",
                "output": "Cargo.toml"
            }
        ]
    });
    let mut chat_request = json!({
        "model": "deepseek-v4-flash",
        "messages": [{
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_ls",
                "type": "function",
                "function": { "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}" }
            }]
        }, {
            "role": "tool",
            "tool_call_id": "call_ls",
            "content": "Cargo.toml"
        }]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(settings);

    adapter.prepare_chat_request(
        ProviderRequestSource::OpenAiResponses,
        &original_request,
        &mut chat_request,
    );

    assert_eq!(
        chat_request["messages"][0]["reasoning_content"],
        "Use the tool result before answering."
    );
}

#[test]
fn adds_fallback_reasoning_content_for_existing_history() {
    let settings = thinking_settings();
    let mut chat_request = json!({
        "model": "deepseek-v4-flash",
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
    let mut adapter = DeepSeekBridgeAdapter::new(settings);

    adapter.prepare_chat_request(
        ProviderRequestSource::OpenAiResponses,
        &json!({ "input": [] }),
        &mut chat_request,
    );

    assert_eq!(
        chat_request["messages"][0]["reasoning_content"],
        super::MISSING_REASONING_CONTENT_FALLBACK
    );
}

#[test]
fn leaves_plain_assistant_history_without_synthetic_reasoning() {
    let settings = thinking_settings();
    let mut chat_request = json!({
        "model": "deepseek-v4-flash",
        "messages": [{
            "role": "assistant",
            "content": "I will check the latest docs."
        }, {
            "role": "user",
            "content": "continue"
        }]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(settings);

    adapter.prepare_chat_request(
        ProviderRequestSource::OpenAiResponses,
        &json!({ "input": [] }),
        &mut chat_request,
    );

    assert!(chat_request["messages"][0]
        .get("reasoning_content")
        .is_none());
    assert!(chat_request["messages"][1]
        .get("reasoning_content")
        .is_none());
}

#[test]
fn does_not_replay_reasoning_content_for_openai_chat_source() {
    let settings = thinking_settings();
    let mut chat_request = json!({
        "model": "deepseek-v4-flash",
        "messages": [{
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_pwd",
                "type": "function",
                "function": { "name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}" }
            }]
        }]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(settings);

    adapter.prepare_chat_request(
        ProviderRequestSource::OpenAiChat,
        &chat_request.clone(),
        &mut chat_request,
    );

    assert_eq!(chat_request["thinking"]["type"], "enabled");
    assert!(chat_request["messages"][0]
        .get("reasoning_content")
        .is_none());
}

#[test]
fn replays_reasoning_content_from_gemini_thought_tool_call() {
    let settings = thinking_settings();
    let original_request = json!({
        "contents": [{
            "role": "model",
            "parts": [
                { "thought": true, "text": "Call pwd, then answer." },
                {
                    "functionCall": {
                        "id": "call_pwd",
                        "name": "exec_command",
                        "args": { "cmd": "pwd" }
                    }
                }
            ]
        }, {
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "id": "call_pwd",
                    "name": "exec_command",
                    "response": { "output": "/tmp/project" }
                }
            }]
        }]
    });
    let mut chat_request = json!({
        "model": "deepseek-v4-flash",
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
            "content": "{\"output\":\"/tmp/project\"}"
        }]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(settings);

    adapter.prepare_chat_request(
        ProviderRequestSource::GeminiGenerateContent,
        &original_request,
        &mut chat_request,
    );

    assert_eq!(
        chat_request["messages"][0]["reasoning_content"],
        "Call pwd, then answer."
    );
}

#[test]
fn replays_reasoning_content_from_gemini_thought_text() {
    let settings = thinking_settings();
    let original_request = json!({
        "contents": [{
            "role": "model",
            "parts": [
                { "thought": true, "text": "Explain briefly." },
                { "text": "The answer is 42." }
            ]
        }]
    });
    let mut chat_request = json!({
        "model": "deepseek-v4-flash",
        "messages": [{
            "role": "assistant",
            "content": "The answer is 42."
        }]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(settings);

    adapter.prepare_chat_request(
        ProviderRequestSource::GeminiGenerateContent,
        &original_request,
        &mut chat_request,
    );

    assert_eq!(
        chat_request["messages"][0]["reasoning_content"],
        "Explain briefly."
    );
}

#[test]
fn replays_reasoning_content_from_anthropic_thinking_tool_use() {
    let settings = thinking_settings();
    let original_request = json!({
        "messages": [{
            "role": "assistant",
            "content": [
                { "type": "thinking", "thinking": "Call pwd, then answer." },
                {
                    "type": "tool_use",
                    "id": "toolu_pwd",
                    "name": "exec_command",
                    "input": { "cmd": "pwd" }
                }
            ]
        }, {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_pwd",
                "content": "/tmp/project"
            }]
        }]
    });
    let mut chat_request = json!({
        "model": "deepseek-v4-flash",
        "messages": [{
            "role": "assistant",
            "content": [{
                "type": "unknown",
                "raw": { "type": "reasoning", "text": "Call pwd, then answer." }
            }]
        }, {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "toolu_pwd",
                "type": "function",
                "function": { "name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}" }
            }]
        }, {
            "role": "tool",
            "tool_call_id": "toolu_pwd",
            "content": "/tmp/project"
        }]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(settings);

    adapter.prepare_chat_request(
        ProviderRequestSource::AnthropicMessages,
        &original_request,
        &mut chat_request,
    );

    let messages = chat_request["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["reasoning_content"], "Call pwd, then answer.");
    assert_eq!(messages[0]["tool_calls"][0]["id"], "toolu_pwd");
}

#[test]
fn replays_reasoning_content_from_anthropic_thinking_text() {
    let settings = thinking_settings();
    let original_request = json!({
        "messages": [{
            "role": "assistant",
            "content": [
                { "type": "thinking", "thinking": "Explain briefly." },
                { "type": "text", "text": "The answer is 42." }
            ]
        }]
    });
    let mut chat_request = json!({
        "model": "deepseek-v4-flash",
        "messages": [{
            "role": "assistant",
            "content": [
                {
                    "type": "unknown",
                    "raw": { "type": "reasoning", "text": "Explain briefly." }
                },
                { "type": "text", "text": "The answer is 42." }
            ]
        }]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(settings);

    adapter.prepare_chat_request(
        ProviderRequestSource::AnthropicMessages,
        &original_request,
        &mut chat_request,
    );

    assert_eq!(
        chat_request["messages"][0]["reasoning_content"],
        "Explain briefly."
    );
    assert_eq!(
        chat_request["messages"][0]["content"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn strips_anthropic_redacted_thinking_before_deepseek_chat() {
    let settings = thinking_settings();
    let original_request = json!({
        "messages": [{
            "role": "assistant",
            "content": [
                { "type": "redacted_thinking", "data": "opaque-redacted-thinking" },
                {
                    "type": "tool_use",
                    "id": "toolu_pwd_redacted",
                    "name": "exec_command",
                    "input": { "cmd": "pwd" }
                }
            ]
        }, {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_pwd_redacted",
                "content": "/tmp/project"
            }]
        }]
    });
    let mut chat_request = json!({
        "model": "deepseek-v4-flash",
        "messages": [{
            "role": "assistant",
            "content": [{ "type": "redacted_thinking", "data": "opaque-redacted-thinking" }]
        }, {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "toolu_pwd_redacted",
                "type": "function",
                "function": { "name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}" }
            }]
        }, {
            "role": "tool",
            "tool_call_id": "toolu_pwd_redacted",
            "content": "/tmp/project"
        }]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(settings);

    adapter.prepare_chat_request(
        ProviderRequestSource::AnthropicMessages,
        &original_request,
        &mut chat_request,
    );

    let messages = chat_request["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["tool_calls"][0]["id"], "toolu_pwd_redacted");
    assert!(messages[0].get("content").is_none() || messages[0]["content"].is_null());
    assert_eq!(messages[1]["role"], "tool");
    assert_eq!(messages[1]["tool_call_id"], "toolu_pwd_redacted");
}

#[test]
fn repairs_tool_history_across_empty_assistant_with_real_request_output() {
    let mut chat_request = json!({
        "model": "deepseek-v4-flash",
        "messages": [
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_ls",
                    "type": "function",
                    "function": { "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}" }
                }]
            },
            {
                "role": "assistant",
                "content": ""
            },
            {
                "role": "tool",
                "tool_call_id": "call_ls",
                "content": "Cargo.toml\nsrc"
            },
            {
                "role": "user",
                "content": "what is here?"
            }
        ]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(DeepSeekBridgeSettings::default());

    adapter.prepare_chat_request(
        ProviderRequestSource::OpenAiResponses,
        &json!({ "input": [] }),
        &mut chat_request,
    );

    let messages = chat_request["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1]["role"], "tool");
    assert_eq!(messages[1]["tool_call_id"], "call_ls");
    assert_eq!(messages[1]["content"], "Cargo.toml\nsrc");
    assert_ne!(messages[1]["content"], super::MISSING_TOOL_OUTPUT_FALLBACK);
    assert_eq!(messages[2]["role"], "user");
}

#[test]
fn repairs_tool_history_from_responses_input() {
    let original_request = json!({
        "input": [{
            "type": "function_call_output",
            "call_id": "call_pwd",
            "output": "/tmp/project"
        }]
    });
    let mut chat_request = json!({
        "model": "deepseek-v4-flash",
        "messages": [
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_pwd",
                    "type": "function",
                    "function": { "name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}" }
                }]
            },
            {
                "role": "assistant",
                "content": ""
            }
        ]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(DeepSeekBridgeSettings::default());

    adapter.prepare_chat_request(
        ProviderRequestSource::OpenAiResponses,
        &original_request,
        &mut chat_request,
    );

    assert_eq!(chat_request["messages"][1]["role"], "tool");
    assert_eq!(chat_request["messages"][1]["tool_call_id"], "call_pwd");
    assert_eq!(chat_request["messages"][1]["content"], "/tmp/project");
}

#[test]
fn moves_anthropic_thinking_before_deepseek_tool_use_history() {
    let mut request = json!({
        "model": "deepseek-v4-pro",
        "messages": [{
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_pwd",
                    "name": "exec_command",
                    "input": { "cmd": "pwd" }
                },
                { "type": "thinking", "thinking": "Call pwd, then answer." }
            ]
        }, {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "call_pwd",
                "content": "/tmp/project"
            }]
        }]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(DeepSeekBridgeSettings::default());

    adapter.prepare_anthropic_request(&mut request);

    let content = request["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["thinking"], "Call pwd, then answer.");
    assert_eq!(content[1]["type"], "tool_use");
    assert_eq!(content[1]["id"], "call_pwd");
}

#[test]
fn does_not_move_anthropic_thinking_across_text_blocks() {
    let mut request = json!({
        "model": "deepseek-v4-pro",
        "messages": [{
            "role": "assistant",
            "content": [
                { "type": "text", "text": "I will inspect the workspace." },
                {
                    "type": "tool_use",
                    "id": "call_pwd",
                    "name": "exec_command",
                    "input": { "cmd": "pwd" }
                },
                { "type": "thinking", "thinking": "Call pwd, then answer." }
            ]
        }]
    });
    let mut adapter = DeepSeekBridgeAdapter::new(DeepSeekBridgeSettings::default());

    adapter.prepare_anthropic_request(&mut request);

    let content = request["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "tool_use");
    assert_eq!(content[2]["type"], "thinking");
}

fn thinking_settings() -> DeepSeekBridgeSettings {
    DeepSeekBridgeSettings {
        thinking: true,
        replay_reasoning_content: true,
    }
}
