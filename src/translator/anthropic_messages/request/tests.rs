
use serde_json::json;

use crate::{ContentBlock, Role, UniversalItem, UniversalRequest};

use super::encode;

#[test]
fn skips_empty_assistant_message_between_tool_use_and_tool_result() {
    let request = UniversalRequest {
        model: Some("minimax".to_string()),
        input: vec![
            UniversalItem::ToolCall {
                id: "call_1".to_string(),
                name: "list_files".to_string(),
                arguments: json!({ "path": "." }),
                extensions: Default::default(),
            },
            UniversalItem::Message {
                role: Role::Assistant,
                id: None,
                content: Vec::new(),
                extensions: Default::default(),
            },
            UniversalItem::ToolResult {
                tool_call_id: "call_1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Cargo.toml".to_string(),
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
    assert_eq!(messages[0]["content"][0]["type"], "tool_use");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"][0]["type"], "tool_result");
    assert_eq!(messages[1]["content"][0]["tool_use_id"], "call_1");
}

#[test]
fn combines_reasoning_and_tool_use_into_one_assistant_turn() {
    let request = UniversalRequest {
        model: Some("minimax".to_string()),
        input: vec![
            UniversalItem::Reasoning {
                id: None,
                text: Some("Need to inspect files.".to_string()),
                encrypted: None,
                extensions: Default::default(),
            },
            UniversalItem::ToolCall {
                id: "call_1".to_string(),
                name: "list_files".to_string(),
                arguments: json!({ "path": "." }),
                extensions: Default::default(),
            },
            UniversalItem::ToolResult {
                tool_call_id: "call_1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Cargo.toml".to_string(),
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
    assert_eq!(messages[0]["content"][0]["type"], "thinking");
    assert_eq!(messages[0]["content"][1]["type"], "tool_use");
    assert_eq!(messages[1]["content"][0]["type"], "tool_result");
}

#[test]
fn combines_assistant_text_and_tool_use_into_one_assistant_turn() {
    let request = UniversalRequest {
        model: Some("minimax".to_string()),
        input: vec![
            UniversalItem::Message {
                role: Role::Assistant,
                id: None,
                content: vec![ContentBlock::Text {
                    text: "I will inspect the project.".to_string(),
                }],
                extensions: Default::default(),
            },
            UniversalItem::ToolCall {
                id: "call_1".to_string(),
                name: "list_files".to_string(),
                arguments: json!({ "path": "." }),
                extensions: Default::default(),
            },
            UniversalItem::ToolResult {
                tool_call_id: "call_1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Cargo.toml".to_string(),
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
    assert_eq!(messages[0]["content"][0]["type"], "text");
    assert_eq!(messages[0]["content"][1]["type"], "tool_use");
    assert_eq!(messages[1]["content"][0]["type"], "tool_result");
}

#[test]
fn moves_assistant_text_after_tool_use_before_tool_use() {
    let request = UniversalRequest {
        model: Some("minimax".to_string()),
        input: vec![
            UniversalItem::ToolCall {
                id: "call_1".to_string(),
                name: "list_files".to_string(),
                arguments: json!({ "path": "." }),
                extensions: Default::default(),
            },
            UniversalItem::Message {
                role: Role::Assistant,
                id: None,
                content: vec![ContentBlock::Text {
                    text: "I will inspect the project.".to_string(),
                }],
                extensions: Default::default(),
            },
            UniversalItem::ToolResult {
                tool_call_id: "call_1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Cargo.toml".to_string(),
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
    assert_eq!(messages[0]["content"][0]["type"], "text");
    assert_eq!(messages[0]["content"][1]["type"], "tool_use");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"][0]["type"], "tool_result");
    assert_eq!(messages[1]["content"][0]["tool_use_id"], "call_1");
}

#[test]
fn moves_reasoning_after_tool_use_before_tool_use() {
    let request = UniversalRequest {
        model: Some("minimax".to_string()),
        input: vec![
            UniversalItem::ToolCall {
                id: "call_1".to_string(),
                name: "list_files".to_string(),
                arguments: json!({ "path": "." }),
                extensions: Default::default(),
            },
            UniversalItem::Reasoning {
                id: None,
                text: Some("Need to inspect files.".to_string()),
                encrypted: None,
                extensions: Default::default(),
            },
            UniversalItem::ToolResult {
                tool_call_id: "call_1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Cargo.toml".to_string(),
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
    assert_eq!(messages[0]["content"][0]["type"], "thinking");
    assert_eq!(messages[0]["content"][1]["type"], "tool_use");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"][0]["type"], "tool_result");
    assert_eq!(messages[1]["content"][0]["tool_use_id"], "call_1");
}

#[test]
fn places_user_text_after_pending_tool_results() {
    let request = UniversalRequest {
        model: Some("minimax".to_string()),
        input: vec![
            UniversalItem::ToolCall {
                id: "call_1".to_string(),
                name: "list_files".to_string(),
                arguments: json!({ "path": "." }),
                extensions: Default::default(),
            },
            UniversalItem::ToolResult {
                tool_call_id: "call_1".to_string(),
                content: vec![ContentBlock::Text {
                    text: "Cargo.toml".to_string(),
                }],
                is_error: false,
                extensions: Default::default(),
            },
            UniversalItem::Message {
                role: Role::User,
                id: None,
                content: vec![ContentBlock::Text {
                    text: "Continue.".to_string(),
                }],
                extensions: Default::default(),
            },
        ],
        ..UniversalRequest::default()
    };

    let encoded = encode(&request).expect("request encodes");
    let messages = encoded["messages"].as_array().unwrap();

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"][0]["type"], "tool_result");
    assert_eq!(messages[1]["content"][1]["type"], "text");
}

#[test]
fn moves_system_messages_to_top_level_system() {
    let request = UniversalRequest {
        model: Some("minimax".to_string()),
        instructions: vec![ContentBlock::Text {
            text: "Be precise.".to_string(),
        }],
        input: vec![
            UniversalItem::Message {
                role: Role::System,
                id: None,
                content: vec![ContentBlock::Text {
                    text: "Prefer JSON.".to_string(),
                }],
                extensions: Default::default(),
            },
            UniversalItem::Message {
                role: Role::User,
                id: None,
                content: vec![ContentBlock::Text {
                    text: "Ping".to_string(),
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
    assert_eq!(encoded["system"][0]["type"], "text");
    assert_eq!(encoded["system"][0]["text"], "Be precise.");
    assert_eq!(encoded["system"][1]["text"], "Prefer JSON.");
}
