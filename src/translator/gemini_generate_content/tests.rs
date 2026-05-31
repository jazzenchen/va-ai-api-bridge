use serde_json::json;

use crate::{
    AnthropicMessagesTranslator, ContentBlock, DecodeState, EncodeState, FinishReason, Role,
    UniversalEvent, UniversalItem, UniversalRequest, WireTranslator,
};

use super::{encode_response, GeminiGenerateContentTranslator};

#[test]
fn decodes_generate_content_request() {
    let mut body = json!({
        "contents": [
            {
                "role": "user",
                "parts": [{ "text": "hello" }]
            },
            {
                "role": "model",
                "parts": [
                    { "thought": true, "text": "Need to inspect cwd.", "thoughtSignature": "sig_123" },
                    { "functionCall": { "id": "call_pwd", "name": "exec_command", "args": { "cmd": "pwd" } } }
                ]
            },
            {
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "id": "call_pwd",
                        "name": "exec_command",
                        "response": { "output": "/tmp/project" }
                    }
                }]
            }
        ],
        "generationConfig": { "maxOutputTokens": 32 }
    });
    super::attach_route_metadata(&mut body, "gemini-2.5-flash", false);

    let request = GeminiGenerateContentTranslator
        .decode_request(body)
        .unwrap();

    assert_eq!(request.model.as_deref(), Some("gemini-2.5-flash"));
    assert!(!request.stream);
    assert_eq!(request.generation.max_output_tokens, Some(32));
    assert!(matches!(
        request.input.first(),
        Some(UniversalItem::Message {
            role: Role::User,
            ..
        })
    ));
    assert!(matches!(
        request.input.get(1),
        Some(UniversalItem::Message {
            role: Role::Assistant,
            content,
            ..
        }) if matches!(
            content.first(),
            Some(ContentBlock::Reasoning {
                text: Some(text),
                encrypted: Some(signature),
                ..
            }) if text == "Need to inspect cwd." && signature == "sig_123"
        )
    ));
    assert!(matches!(
        request.input.get(2),
        Some(UniversalItem::ToolCall {
            id,
            name,
            arguments,
            ..
        }) if id == "call_pwd"
            && name == "exec_command"
            && arguments["cmd"] == "pwd"
    ));
    assert!(matches!(
        request.input.get(3),
        Some(UniversalItem::ToolResult {
            tool_call_id,
            ..
        }) if tool_call_id == "call_pwd"
    ));
}

#[test]
fn decodes_snake_case_generate_content_request() {
    let mut body = json!({
        "system_instruction": { "parts": { "text": "Be concise." } },
        "contents": {
            "role": "user",
            "parts": { "text": "hello" }
        },
        "generation_config": { "max_output_tokens": 32, "top_p": 0.9 },
        "tools": [{
            "function_declarations": [{
                "name": "lookup",
                "parameters": { "type": "object" }
            }]
        }],
        "tool_config": {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["lookup"]
            }
        }
    });
    super::attach_route_metadata(&mut body, "models/gemini-2.5-flash", true);

    let request = GeminiGenerateContentTranslator
        .decode_request(body)
        .unwrap();

    assert_eq!(request.model.as_deref(), Some("gemini-2.5-flash"));
    assert!(request.stream);
    assert_eq!(request.instructions.len(), 1);
    assert_eq!(request.generation.max_output_tokens, Some(32));
    assert_eq!(request.generation.top_p, Some(0.9));
    assert_eq!(request.tools[0].name, "lookup");
}

#[test]
fn encodes_tool_results_as_user_function_responses_with_names() {
    let request = UniversalRequest {
        input: vec![
            UniversalItem::ToolCall {
                id: "call_pwd".to_string(),
                name: "exec_command".to_string(),
                arguments: json!({ "cmd": "pwd" }),
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

    let wire = GeminiGenerateContentTranslator
        .encode_request(&request)
        .unwrap();

    assert_eq!(wire["contents"][0]["role"], "model");
    assert_eq!(wire["contents"][1]["role"], "user");
    assert_eq!(
        wire["contents"][1]["parts"][0]["functionResponse"]["id"],
        "call_pwd"
    );
    assert_eq!(
        wire["contents"][1]["parts"][0]["functionResponse"]["name"],
        "exec_command"
    );
}

#[test]
fn preserves_function_call_thought_signature_in_request_history() {
    let request = GeminiGenerateContentTranslator
        .decode_request(json!({
            "contents": [{
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "id": "call_pwd",
                        "name": "exec_command",
                        "args": { "cmd": "pwd" }
                    },
                    "thoughtSignature": "sig_123"
                }]
            }]
        }))
        .unwrap();

    let wire = GeminiGenerateContentTranslator
        .encode_request(&request)
        .unwrap();

    assert_eq!(
        wire["contents"][0]["parts"][0]["thoughtSignature"],
        "sig_123"
    );
}

#[test]
fn encodes_gemini_completion_response() {
    let events = GeminiGenerateContentTranslator
        .decode_response(json!({
            "responseId": "resp_gemini",
            "candidates": [{
                "content": { "role": "model", "parts": [{ "text": "pong" }] },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 1,
                "totalTokenCount": 2
            }
        }))
        .unwrap();

    let response = encode_response(&events);

    assert_eq!(
        response["candidates"][0]["content"]["parts"][0]["text"],
        "pong"
    );
    assert_eq!(response["candidates"][0]["finishReason"], "STOP");
    assert_eq!(response["responseId"], "resp_gemini");
    assert_eq!(response["usageMetadata"]["totalTokenCount"], 2);
}

#[test]
fn preserves_function_call_thought_signature_in_response_roundtrip() {
    let events = GeminiGenerateContentTranslator
        .decode_response(json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "id": "call_pwd",
                            "name": "exec_command",
                            "args": { "cmd": "pwd" }
                        },
                        "thoughtSignature": "sig_123"
                    }]
                },
                "finishReason": "STOP"
            }]
        }))
        .unwrap();

    let response = encode_response(&events);

    assert_eq!(
        response["candidates"][0]["content"]["parts"][0]["thoughtSignature"],
        "sig_123"
    );
}

#[test]
fn decodes_gemini_stream_response_id() {
    let mut state = DecodeState::default();
    let events = GeminiGenerateContentTranslator
        .decode_stream_chunk(
            json!({
                "responseId": "resp_stream",
                "candidates": [{
                    "content": { "role": "model", "parts": [{ "text": "pong" }] }
                }]
            }),
            &mut state,
        )
        .unwrap();

    assert!(matches!(
        events.first(),
        Some(UniversalEvent::ResponseStart { id: Some(id), .. }) if id == "resp_stream"
    ));
}

#[test]
fn preserves_stream_reasoning_thought_signature_on_content_start() {
    let mut state = DecodeState::default();
    let events = GeminiGenerateContentTranslator
        .decode_stream_chunk(
            json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{
                            "thought": true,
                            "text": "Need to inspect cwd.",
                            "thoughtSignature": "sig_123"
                        }]
                    }
                }]
            }),
            &mut state,
        )
        .unwrap();

    assert!(events.iter().any(|event| matches!(
        event,
        UniversalEvent::ContentStart {
            block: ContentBlock::Reasoning {
                encrypted: Some(signature),
                ..
            },
            ..
        } if signature == "sig_123"
    )));
}

#[test]
fn decodes_gemini_stream_text_as_one_open_content_block() {
    let mut state = DecodeState::default();
    let first = GeminiGenerateContentTranslator
        .decode_stream_chunk(
            json!({
                "responseId": "resp_stream",
                "modelVersion": "gemini-2.5-flash",
                "candidates": [{
                    "content": { "role": "model", "parts": [{ "text": "Okay" }] }
                }]
            }),
            &mut state,
        )
        .unwrap();
    let second = GeminiGenerateContentTranslator
        .decode_stream_chunk(
            json!({
                "candidates": [{
                    "content": { "role": "model", "parts": [{ "text": ", done" }] },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 3
                }
            }),
            &mut state,
        )
        .unwrap();

    assert_eq!(
        first
            .iter()
            .filter(|event| matches!(event, UniversalEvent::ContentStart { .. }))
            .count(),
        1
    );
    assert!(first.iter().any(|event| matches!(
        event,
        UniversalEvent::TextDelta { index: 0, text } if text == "Okay"
    )));
    assert!(!first
        .iter()
        .any(|event| matches!(event, UniversalEvent::ContentDone { .. })));

    assert!(!second
        .iter()
        .any(|event| matches!(event, UniversalEvent::ContentStart { .. })));
    assert!(second.iter().any(|event| matches!(
        event,
        UniversalEvent::TextDelta { index: 0, text } if text == ", done"
    )));
    assert!(second
        .iter()
        .any(|event| matches!(event, UniversalEvent::ContentDone { index: 0, .. })));
    assert!(second.iter().any(|event| matches!(
        event,
        UniversalEvent::MessageDone {
            finish_reason: Some(FinishReason::Stop),
            ..
        }
    )));
    assert!(second
        .iter()
        .any(|event| matches!(event, UniversalEvent::ResponseDone { .. })));
}

#[test]
fn gemini_stream_to_anthropic_keeps_text_block_open_across_chunks() {
    let mut decode_state = DecodeState::default();
    let mut encode_state = EncodeState::default();
    let first_events = GeminiGenerateContentTranslator
        .decode_stream_chunk(
            json!({
                "responseId": "resp_stream",
                "modelVersion": "gemini-2.5-flash",
                "candidates": [{
                    "content": { "role": "model", "parts": [{ "text": "Hello" }] }
                }]
            }),
            &mut decode_state,
        )
        .unwrap();
    let first_wire = AnthropicMessagesTranslator
        .encode_events(&first_events, &mut encode_state)
        .unwrap();
    let second_events = GeminiGenerateContentTranslator
        .decode_stream_chunk(
            json!({
                "candidates": [{
                    "content": { "role": "model", "parts": [{ "text": " world" }] },
                    "finishReason": "STOP"
                }]
            }),
            &mut decode_state,
        )
        .unwrap();
    let second_wire = AnthropicMessagesTranslator
        .encode_events(&second_events, &mut encode_state)
        .unwrap();

    assert_eq!(
        first_wire
            .iter()
            .filter(|event| event.data["type"] == "content_block_start")
            .count(),
        1
    );
    assert!(!second_wire
        .iter()
        .any(|event| event.data["type"] == "content_block_start"));
    assert_eq!(
        second_wire
            .iter()
            .filter(|event| event.data["type"] == "content_block_stop")
            .count(),
        1
    );
}

#[test]
fn encodes_reasoning_as_gemini_thought_part() {
    let events = GeminiGenerateContentTranslator
        .decode_response(json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        { "thought": true, "text": "I should inspect cwd." },
                        { "functionCall": { "id": "call_pwd", "name": "exec_command", "args": { "cmd": "pwd" } } }
                    ]
                },
                "finishReason": "MALFORMED_FUNCTION_CALL"
            }]
        }))
        .unwrap();

    let response = encode_response(&events);

    assert_eq!(
        response["candidates"][0]["content"]["parts"][0]["thought"],
        true
    );
    assert_eq!(
        response["candidates"][0]["content"]["parts"][0]["text"],
        "I should inspect cwd."
    );
    assert_eq!(
        response["candidates"][0]["content"]["parts"][1]["functionCall"]["id"],
        "call_pwd"
    );
    assert_eq!(
        response["candidates"][0]["content"]["parts"][1]["functionCall"]["name"],
        "exec_command"
    );
    assert_eq!(response["candidates"][0]["finishReason"], "STOP");
}

#[test]
fn stream_encoder_buffers_tool_call_until_arguments_are_complete() {
    let mut state = EncodeState::default();
    let events = vec![
        UniversalEvent::ToolCallDelta {
            id: "call_pwd".to_string(),
            name: Some("exec_command".to_string()),
            arguments_delta: String::new(),
        },
        UniversalEvent::ToolCallDelta {
            id: "call_pwd".to_string(),
            name: None,
            arguments_delta: "{\"cmd\"".to_string(),
        },
        UniversalEvent::ToolCallDelta {
            id: "call_pwd".to_string(),
            name: None,
            arguments_delta: ":\"pwd\"}".to_string(),
        },
        UniversalEvent::MessageDone {
            finish_reason: Some(FinishReason::ToolCall),
            usage: None,
            extensions: Default::default(),
        },
    ];

    let wire = GeminiGenerateContentTranslator
        .encode_events(&events, &mut state)
        .unwrap();

    assert_eq!(wire.len(), 1);
    let candidate = &wire[0].data["candidates"][0];
    let function_call = &candidate["content"]["parts"][0]["functionCall"];
    assert_eq!(function_call["id"], "call_pwd");
    assert_eq!(function_call["name"], "exec_command");
    assert_eq!(function_call["args"]["cmd"], "pwd");
    assert_eq!(candidate["finishReason"], "STOP");
}
