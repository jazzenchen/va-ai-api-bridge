use serde_json::json;

use super::{decode, encode};
use crate::{
    ContentBlock, OpenAiResponsesTranslator, Role, UniversalItem, UniversalRequest, WireTranslator,
};

#[test]
fn preserves_reasoning_content_on_assistant_tool_call_messages() {
    let universal = decode(json!({
        "model": "deepseek-v4-pro",
        "messages": [{
            "role": "assistant",
            "content": null,
            "reasoning_content": "I should inspect cwd.",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "arguments": "{\"cmd\":\"pwd\"}"
                }
            }]
        }]
    }))
    .expect("request decodes");
    let encoded = encode(&universal).expect("request encodes");

    assert_eq!(
        encoded["messages"][0]["reasoning_content"],
        "I should inspect cwd."
    );
    assert_eq!(encoded["messages"][0]["tool_calls"][0]["id"], "call_123");
}

#[test]
fn encodes_developer_messages_as_system_for_chat_compatibility() {
    let request = UniversalRequest {
        model: Some("chat-model".to_string()),
        input: vec![
            UniversalItem::Message {
                role: Role::Developer,
                id: None,
                content: vec![ContentBlock::Text {
                    text: "Follow the dashboard contract.".to_string(),
                }],
                extensions: Default::default(),
            },
            UniversalItem::Message {
                role: Role::User,
                id: None,
                content: vec![ContentBlock::Text {
                    text: "Hello".to_string(),
                }],
                extensions: Default::default(),
            },
        ],
        ..UniversalRequest::default()
    };

    let encoded = encode(&request).expect("request encodes");

    let messages = encoded["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "Follow the dashboard contract.");
    assert_eq!(messages[1]["role"], "user");
}

#[test]
fn skips_empty_assistant_message_between_tool_call_and_tool_result() {
    let request = UniversalRequest {
        model: Some("chat-model".to_string()),
        input: vec![
            UniversalItem::ToolCall {
                id: "call_pwd".to_string(),
                name: "exec_command".to_string(),
                arguments: json!({ "cmd": "pwd" }),
                extensions: Default::default(),
            },
            UniversalItem::Message {
                role: Role::Assistant,
                id: None,
                content: Vec::new(),
                extensions: Default::default(),
            },
            UniversalItem::ToolResult {
                tool_call_id: "call_pwd".to_string(),
                content: vec![ContentBlock::Text {
                    text: "/tmp/project".to_string(),
                }],
                is_error: false,
                extensions: Default::default(),
            },
        ],
        ..UniversalRequest::default()
    };

    let encoded = encode(&request).expect("request encodes");

    let messages = encoded["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["tool_calls"][0]["id"], "call_pwd");
    assert_eq!(messages[1]["role"], "tool");
    assert_eq!(messages[1]["tool_call_id"], "call_pwd");
}

#[test]
fn encodes_assistant_tool_calls_with_required_content_field() {
    let request = UniversalRequest {
        model: Some("chat-model".to_string()),
        input: vec![UniversalItem::ToolCall {
            id: "call_pwd".to_string(),
            name: "exec_command".to_string(),
            arguments: json!({ "cmd": "pwd" }),
            extensions: Default::default(),
        }],
        ..UniversalRequest::default()
    };

    let encoded = encode(&request).expect("request encodes");

    let messages = encoded["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"], "");
    assert_eq!(messages[0]["tool_calls"][0]["id"], "call_pwd");
}

#[test]
fn attaches_assistant_text_to_pending_tool_calls_before_tool_results() {
    let request = UniversalRequest {
        model: Some("chat-model".to_string()),
        input: vec![
            UniversalItem::ToolCall {
                id: "call_ls".to_string(),
                name: "exec_command".to_string(),
                arguments: json!({ "cmd": "ls" }),
                extensions: Default::default(),
            },
            UniversalItem::ToolCall {
                id: "call_pwd".to_string(),
                name: "exec_command".to_string(),
                arguments: json!({ "cmd": "pwd" }),
                extensions: Default::default(),
            },
            UniversalItem::Message {
                role: Role::Assistant,
                id: None,
                content: vec![ContentBlock::Text {
                    text: "I will inspect the project first.".to_string(),
                }],
                extensions: Default::default(),
            },
            UniversalItem::ToolResult {
                tool_call_id: "call_ls".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Cargo.toml".to_string(),
                }],
                is_error: false,
                extensions: Default::default(),
            },
            UniversalItem::ToolResult {
                tool_call_id: "call_pwd".to_string(),
                content: vec![ContentBlock::Text {
                    text: "/tmp/project".to_string(),
                }],
                is_error: false,
                extensions: Default::default(),
            },
        ],
        ..UniversalRequest::default()
    };

    let encoded = encode(&request).expect("request encodes");

    let messages = encoded["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"], "I will inspect the project first.");
    assert_eq!(messages[0]["tool_calls"].as_array().unwrap().len(), 2);
    assert_eq!(messages[1]["role"], "tool");
    assert_eq!(messages[1]["tool_call_id"], "call_ls");
    assert_eq!(messages[2]["role"], "tool");
    assert_eq!(messages[2]["tool_call_id"], "call_pwd");
}

#[test]
fn drops_encrypted_reasoning_assistant_message_for_chat_compatibility() {
    let request = UniversalRequest {
        model: Some("chat-model".to_string()),
        input: vec![UniversalItem::Message {
            role: Role::Assistant,
            id: None,
            content: vec![ContentBlock::Reasoning {
                text: None,
                encrypted: Some("opaque-reasoning".to_string()),
                extensions: Default::default(),
            }],
            extensions: Default::default(),
        }],
        ..UniversalRequest::default()
    };

    let encoded = encode(&request).expect("request encodes");

    let messages = encoded["messages"].as_array().unwrap();
    assert!(messages.is_empty());
}

#[test]
fn filters_non_chat_content_parts_from_messages() {
    let request = UniversalRequest {
        model: Some("chat-model".to_string()),
        input: vec![UniversalItem::Message {
            role: Role::Assistant,
            id: None,
            content: vec![
                ContentBlock::Text {
                    text: "Visible text.".to_string(),
                },
                ContentBlock::Reasoning {
                    text: Some("hidden reasoning".to_string()),
                    encrypted: None,
                    extensions: Default::default(),
                },
                ContentBlock::Unknown {
                    raw: json!({ "type": "reasoning_text", "text": "raw reasoning" }),
                },
            ],
            extensions: Default::default(),
        }],
        ..UniversalRequest::default()
    };

    let encoded = encode(&request).expect("request encodes");

    let messages = encoded["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"], "Visible text.");
}

#[test]
fn responses_input_image_encodes_as_chat_image_url() {
    let universal = OpenAiResponsesTranslator
        .decode_request(json!({
            "model": "qwen3.6-plus",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "What is in this image?" },
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,abc123"
                    }
                ]
            }]
        }))
        .expect("responses request decodes");

    let encoded = encode(&universal).expect("chat request encodes");

    assert_eq!(encoded["messages"][0]["content"][0]["type"], "text");
    assert_eq!(encoded["messages"][0]["content"][1]["type"], "image_url");
    assert_eq!(
        encoded["messages"][0]["content"][1]["image_url"]["url"],
        "data:image/png;base64,abc123"
    );
}

#[test]
fn drops_unknown_responses_items_in_chat_encoding() {
    let request = UniversalRequest {
        model: Some("chat-model".to_string()),
        input: vec![
            UniversalItem::Unknown {
                raw: json!({
                    "type": "reasoning",
                    "id": "rs_123",
                    "content": null,
                    "encrypted_content": "opaque"
                }),
            },
            UniversalItem::Message {
                role: Role::User,
                id: None,
                content: vec![ContentBlock::Text {
                    text: "Hello".to_string(),
                }],
                extensions: Default::default(),
            },
        ],
        ..UniversalRequest::default()
    };

    let encoded = encode(&request).expect("request encodes");

    let messages = encoded["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "Hello");
}

#[test]
fn fills_content_on_passthrough_chat_messages_without_content() {
    let request = UniversalRequest {
        model: Some("chat-model".to_string()),
        input: vec![UniversalItem::Unknown {
            raw: json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}"
                    }
                }]
            }),
        }],
        ..UniversalRequest::default()
    };

    let encoded = encode(&request).expect("request encodes");

    let messages = encoded["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"], "");
    assert_eq!(messages[0]["tool_calls"][0]["id"], "call_123");
}

#[test]
fn chat_function_named_web_search_stays_a_function_tool() {
    let request = decode(json!({
        "model": "chat-model",
        "messages": [{ "role": "user", "content": "Search the web." }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Host-provided search function.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"]
                }
            }
        }]
    }))
    .expect("request decodes");

    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, "web_search");
    assert!(request.server_tools.is_empty());
}
