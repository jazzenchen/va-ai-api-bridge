use crate::{ContentBlock, EncodeState, UniversalEvent};

use super::*;

#[test]
fn message_start_uses_complete_anthropic_shape() {
    let events = encode(
        &[UniversalEvent::ResponseStart {
            id: Some("msg_1".to_string()),
            model: Some("qwen3-coder-next".to_string()),
            extensions: Default::default(),
        }],
        &mut EncodeState::default(),
    )
    .expect("events encode");

    let data = &events[0].data;
    assert_eq!(data["type"], "message_start");
    assert_eq!(data["message"]["type"], "message");
    assert_eq!(data["message"]["role"], "assistant");
    assert_eq!(data["message"]["content"], json!([]));
    assert_eq!(data["message"]["stop_reason"], Value::Null);
    assert_eq!(data["message"]["stop_sequence"], Value::Null);
    assert_eq!(
        data["message"]["usage"],
        json!({ "input_tokens": 0, "output_tokens": 0 })
    );
}

#[test]
fn message_delta_uses_object_usage_when_upstream_omits_usage() {
    let events = encode(
        &[
            UniversalEvent::ResponseStart {
                id: Some("msg_1".to_string()),
                model: Some("qwen3-coder-next".to_string()),
                extensions: Default::default(),
            },
            UniversalEvent::ContentStart {
                index: 0,
                block: ContentBlock::Text {
                    text: String::new(),
                },
            },
            UniversalEvent::TextDelta {
                index: 0,
                text: "OK".to_string(),
            },
            UniversalEvent::ContentDone {
                index: 0,
                final_block: None,
            },
            UniversalEvent::MessageDone {
                finish_reason: Some(crate::FinishReason::Stop),
                usage: None,
                extensions: Default::default(),
            },
            UniversalEvent::ResponseDone {
                usage: None,
                extensions: Default::default(),
            },
        ],
        &mut EncodeState::default(),
    )
    .expect("events encode");

    let message_delta = events
        .iter()
        .find(|event| event.data["type"] == "message_delta")
        .expect("message_delta");
    assert_eq!(message_delta.data["delta"]["stop_reason"], "end_turn");
    assert_eq!(message_delta.data["delta"]["stop_sequence"], Value::Null);
    assert_eq!(
        message_delta.data["usage"],
        json!({ "input_tokens": 0, "output_tokens": 0 })
    );
}

#[test]
fn response_done_closes_open_text_block_before_message_delta() {
    let events = encode(
        &[
            UniversalEvent::ResponseStart {
                id: Some("msg_1".to_string()),
                model: Some("qwen3-coder-next".to_string()),
                extensions: Default::default(),
            },
            UniversalEvent::ContentStart {
                index: 0,
                block: ContentBlock::Text {
                    text: String::new(),
                },
            },
            UniversalEvent::TextDelta {
                index: 0,
                text: "OK".to_string(),
            },
            UniversalEvent::MessageDone {
                finish_reason: Some(crate::FinishReason::Stop),
                usage: None,
                extensions: Default::default(),
            },
            UniversalEvent::ResponseDone {
                usage: None,
                extensions: Default::default(),
            },
        ],
        &mut EncodeState::default(),
    )
    .expect("events encode");

    let stop_index = events
        .iter()
        .position(|event| event.data["type"] == "content_block_stop")
        .expect("content_block_stop");
    let delta_index = events
        .iter()
        .position(|event| event.data["type"] == "message_delta")
        .expect("message_delta");

    assert!(stop_index < delta_index);
    assert_eq!(events[stop_index].data["index"], 0);
}
