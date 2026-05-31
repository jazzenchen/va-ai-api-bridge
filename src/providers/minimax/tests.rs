use serde_json::json;

use super::MiniMaxBridgeAdapter;
use crate::{ContentBlock, UniversalEvent};

#[test]
fn clamps_minimax_chat_settings_to_supported_ranges() {
    let mut adapter = MiniMaxBridgeAdapter::default();
    let mut request = json!({
        "model": "MiniMax-M2.7",
        "messages": [],
        "temperature": 0,
        "top_p": 0,
        "max_completion_tokens": 8192
    });

    adapter.prepare_chat_request(&mut request);

    assert_eq!(request["temperature"], 1.0);
    assert_eq!(request["top_p"], 0.95);
    assert_eq!(request["max_completion_tokens"], 2048);
}

#[test]
fn folds_system_messages_into_one_leading_message() {
    let mut adapter = MiniMaxBridgeAdapter::default();
    let mut request = json!({
        "model": "MiniMax-M2.7",
        "messages": [
            { "role": "system", "content": "Global instructions." },
            { "role": "user", "content": "Hi" },
            { "role": "system", "content": "Developer instructions." },
            {
                "role": "system",
                "content": [{ "type": "text", "text": "Extra instructions." }]
            }
        ]
    });

    adapter.prepare_chat_request(&mut request);

    let messages = request["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(
        messages[0]["content"],
        "Global instructions.\n\nDeveloper instructions.\n\nExtra instructions."
    );
    assert_eq!(messages[1]["role"], "user");
}

#[test]
fn leaves_valid_minimax_chat_settings_unchanged() {
    let mut adapter = MiniMaxBridgeAdapter::default();
    let mut request = json!({
        "model": "MiniMax-M2.7",
        "messages": [],
        "temperature": 0.2,
        "top_p": 0.8,
        "max_completion_tokens": 1024
    });

    adapter.prepare_chat_request(&mut request);

    assert_eq!(request["temperature"], 0.2);
    assert_eq!(request["top_p"], 0.8);
    assert_eq!(request["max_completion_tokens"], 1024);
}

#[test]
fn converts_minimax_think_tags_to_reasoning_events() {
    let mut adapter = MiniMaxBridgeAdapter::default();
    let mut events = vec![
        text_start(0),
        UniversalEvent::TextDelta {
            index: 0,
            text: "<think>Need ".to_string(),
        },
        UniversalEvent::TextDelta {
            index: 0,
            text: "math</think>\n\n221".to_string(),
        },
        response_done(),
    ];

    adapter.transform_upstream_events(&mut events);

    assert_eq!(joined_reasoning(&events), "Need math");
    assert_eq!(joined_text(&events), "\n\n221");
    assert!(!joined_text(&events).contains("<think>"));
    assert!(events.iter().any(|event| matches!(
        event,
        UniversalEvent::ContentStart {
            block: ContentBlock::Reasoning { .. },
            ..
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        UniversalEvent::ContentStart {
            block: ContentBlock::Text { .. },
            ..
        }
    )));
}

#[test]
fn handles_minimax_think_tags_split_across_stream_chunks() {
    let mut adapter = MiniMaxBridgeAdapter::default();
    let mut events = vec![
        text_start(0),
        UniversalEvent::TextDelta {
            index: 0,
            text: "<thi".to_string(),
        },
        UniversalEvent::TextDelta {
            index: 0,
            text: "nk>hidden</thi".to_string(),
        },
        UniversalEvent::TextDelta {
            index: 0,
            text: "nk>done".to_string(),
        },
        response_done(),
    ];

    adapter.transform_upstream_events(&mut events);

    assert_eq!(joined_reasoning(&events), "hidden");
    assert_eq!(joined_text(&events), "done");
}

#[test]
fn preserves_plain_minimax_text_that_only_looks_like_partial_tag() {
    let mut adapter = MiniMaxBridgeAdapter::default();
    let mut events = vec![
        text_start(0),
        UniversalEvent::TextDelta {
            index: 0,
            text: "hello <thi".to_string(),
        },
        UniversalEvent::TextDelta {
            index: 0,
            text: "s is plain".to_string(),
        },
        response_done(),
    ];

    adapter.transform_upstream_events(&mut events);

    assert_eq!(joined_reasoning(&events), "");
    assert_eq!(joined_text(&events), "hello <this is plain");
}

#[test]
fn parses_final_text_block_when_no_delta_was_seen() {
    let mut adapter = MiniMaxBridgeAdapter::default();
    let mut events = vec![
        text_start(0),
        UniversalEvent::ContentDone {
            index: 0,
            final_block: Some(ContentBlock::Text {
                text: "<think>hidden</think>visible".to_string(),
            }),
        },
        response_done(),
    ];

    adapter.transform_upstream_events(&mut events);

    assert_eq!(joined_reasoning(&events), "hidden");
    assert_eq!(joined_text(&events), "visible");
}

#[test]
fn remaps_following_content_blocks_after_splitting_think_tags() {
    let mut adapter = MiniMaxBridgeAdapter::default();
    let tool_block = ContentBlock::ToolCall {
        id: "call_1".to_string(),
        name: "lookup".to_string(),
        arguments: json!({}),
        extensions: Default::default(),
    };
    let mut events = vec![
        text_start(0),
        UniversalEvent::TextDelta {
            index: 0,
            text: "<think>hidden</think>visible".to_string(),
        },
        UniversalEvent::ContentDone {
            index: 0,
            final_block: Some(ContentBlock::Text {
                text: "<think>hidden</think>visible".to_string(),
            }),
        },
        UniversalEvent::ContentStart {
            index: 1,
            block: tool_block.clone(),
        },
        UniversalEvent::ToolCallDelta {
            id: "call_1".to_string(),
            name: Some("lookup".to_string()),
            arguments_delta: "{}".to_string(),
        },
        UniversalEvent::ContentDone {
            index: 1,
            final_block: Some(tool_block),
        },
        response_done(),
    ];

    adapter.transform_upstream_events(&mut events);

    assert_eq!(content_start_indexes(&events), vec![0, 1, 2]);
    assert_eq!(content_done_indexes(&events), vec![0, 1, 2]);
}

fn content_start_indexes(events: &[UniversalEvent]) -> Vec<usize> {
    events
        .iter()
        .filter_map(|event| match event {
            UniversalEvent::ContentStart { index, .. } => Some(*index),
            _ => None,
        })
        .collect()
}

fn content_done_indexes(events: &[UniversalEvent]) -> Vec<usize> {
    events
        .iter()
        .filter_map(|event| match event {
            UniversalEvent::ContentDone { index, .. } => Some(*index),
            _ => None,
        })
        .collect()
}

fn text_start(index: usize) -> UniversalEvent {
    UniversalEvent::ContentStart {
        index,
        block: ContentBlock::Text {
            text: String::new(),
        },
    }
}

fn response_done() -> UniversalEvent {
    UniversalEvent::ResponseDone {
        usage: None,
        extensions: Default::default(),
    }
}

fn joined_text(events: &[UniversalEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            UniversalEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn joined_reasoning(events: &[UniversalEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            UniversalEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}
