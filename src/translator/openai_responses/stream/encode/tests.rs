use crate::translator::common;
use crate::{EncodeState, UniversalEvent};

use super::encode;

#[test]
fn opens_reasoning_item_before_reasoning_delta() {
    let mut state = EncodeState::default();
    let events = encode(
        &[
            UniversalEvent::ResponseStart {
                id: Some("resp_1".to_string()),
                model: Some("deepseek-v4-pro".to_string()),
                extensions: common::empty_extensions(),
            },
            UniversalEvent::ReasoningDelta {
                index: 0,
                text: "Need to think.".to_string(),
            },
            UniversalEvent::TextDelta {
                index: 0,
                text: "OK".to_string(),
            },
            UniversalEvent::ResponseDone {
                usage: None,
                extensions: common::empty_extensions(),
            },
        ],
        &mut state,
    )
    .expect("events encode");

    let reasoning_added_index = events
        .iter()
        .position(|event| {
            event.data["type"] == "response.output_item.added"
                && event.data["item"]["type"] == "reasoning"
        })
        .expect("reasoning item added");
    let reasoning_delta_index = events
        .iter()
        .position(|event| event.data["type"] == "response.reasoning_text.delta")
        .expect("reasoning delta");

    assert!(reasoning_added_index < reasoning_delta_index);
    assert_eq!(
        events[reasoning_delta_index].data["item_id"],
        events[reasoning_added_index].data["item"]["id"]
    );

    let completed = events
        .iter()
        .find(|event| event.data["type"] == "response.completed")
        .expect("response completed");
    assert_eq!(completed.data["response"]["output"][0]["type"], "reasoning");
    assert_eq!(
        completed.data["response"]["output"][0]["content"][0]["text"],
        "Need to think."
    );
    assert_eq!(completed.data["response"]["output"][1]["type"], "message");
    assert_eq!(
        completed.data["response"]["output"][1]["content"][0]["text"],
        "OK"
    );
}
