use std::path::PathBuf;

use serde_json::Value;
use va_ai_api_proxy::{
    AnthropicMessagesTranslator, DecodeState, OpenAiChatTranslator, OpenAiResponsesTranslator,
    UniversalRequest, WireTranslator,
};

#[test]
fn openai_chat_decode_request_matches_fixture() {
    assert_decode_request(
        OpenAiChatTranslator,
        "openai_chat/decode_request.input.json",
        "openai_chat/decode_request.expected.json",
    );
}

#[test]
fn openai_chat_decode_response_matches_fixture() {
    assert_decode_response(
        OpenAiChatTranslator,
        "openai_chat/decode_response.input.json",
        "openai_chat/decode_response.expected.json",
    );
}

#[test]
fn openai_chat_decode_stream_matches_fixture() {
    assert_decode_stream(
        OpenAiChatTranslator,
        "openai_chat/decode_stream.input.json",
        "openai_chat/decode_stream.expected.json",
    );
}

#[test]
fn openai_responses_decode_request_matches_fixture() {
    assert_decode_request(
        OpenAiResponsesTranslator,
        "openai_responses/decode_request.input.json",
        "openai_responses/decode_request.expected.json",
    );
}

#[test]
fn openai_responses_decode_response_matches_fixture() {
    assert_decode_response(
        OpenAiResponsesTranslator,
        "openai_responses/decode_response.input.json",
        "openai_responses/decode_response.expected.json",
    );
}

#[test]
fn openai_responses_decode_stream_matches_fixture() {
    assert_decode_stream(
        OpenAiResponsesTranslator,
        "openai_responses/decode_stream.input.json",
        "openai_responses/decode_stream.expected.json",
    );
}

#[test]
fn anthropic_messages_decode_request_matches_fixture() {
    assert_decode_request(
        AnthropicMessagesTranslator,
        "anthropic_messages/decode_request.input.json",
        "anthropic_messages/decode_request.expected.json",
    );
}

#[test]
fn anthropic_messages_decode_response_matches_fixture() {
    assert_decode_response(
        AnthropicMessagesTranslator,
        "anthropic_messages/decode_response.input.json",
        "anthropic_messages/decode_response.expected.json",
    );
}

#[test]
fn anthropic_messages_decode_stream_matches_fixture() {
    assert_decode_stream(
        AnthropicMessagesTranslator,
        "anthropic_messages/decode_stream.input.json",
        "anthropic_messages/decode_stream.expected.json",
    );
}

fn assert_decode_request(translator: impl WireTranslator, input_path: &str, expected_path: &str) {
    let mut request = translator
        .decode_request(read_fixture(input_path))
        .expect("decode request");
    strip_source_raw(&mut request);
    assert_json_eq(
        serde_json::to_value(request).unwrap(),
        read_fixture(expected_path),
    );
}

fn assert_decode_response(translator: impl WireTranslator, input_path: &str, expected_path: &str) {
    let events = translator
        .decode_response(read_fixture(input_path))
        .expect("decode response");
    assert_json_eq(
        serde_json::to_value(events).unwrap(),
        read_fixture(expected_path),
    );
}

fn assert_decode_stream(translator: impl WireTranslator, input_path: &str, expected_path: &str) {
    let chunks = read_fixture(input_path)
        .as_array()
        .expect("stream input fixture must be an array")
        .clone();
    let mut state = DecodeState::default();
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(
            translator
                .decode_stream_chunk(chunk, &mut state)
                .expect("decode stream chunk"),
        );
    }
    assert_json_eq(
        serde_json::to_value(events).unwrap(),
        read_fixture(expected_path),
    );
}

fn strip_source_raw(request: &mut UniversalRequest) {
    if let Some(source) = &mut request.source {
        source.raw = None;
    }
}

fn read_fixture(path: &str) -> Value {
    let path = fixture_path(path);
    let body = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read fixture {}: {error}", path.display()));
    serde_json::from_str(&body)
        .unwrap_or_else(|error| panic!("parse fixture {}: {error}", path.display()))
}

fn fixture_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(path)
}

fn assert_json_eq(actual: Value, expected: Value) {
    if actual == expected {
        return;
    }

    panic!(
        "JSON mismatch\n\nactual:\n{}\n\nexpected:\n{}",
        serde_json::to_string_pretty(&actual).unwrap(),
        serde_json::to_string_pretty(&expected).unwrap()
    );
}
