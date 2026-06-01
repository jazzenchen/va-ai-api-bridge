use va_ai_api_bridge::{
    ContentBlock, EncodeState, FinishReason, OpenAiChatTranslator, Role, UniversalEvent, Usage,
    WireTranslator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let events = vec![
        UniversalEvent::ResponseStart {
            id: Some("resp_example".to_string()),
            model: Some("example-model".to_string()),
            extensions: Default::default(),
        },
        UniversalEvent::MessageStart {
            id: "msg_example".to_string(),
            role: Role::Assistant,
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
            text: "Hello from UniversalEvent.".to_string(),
        },
        UniversalEvent::ContentDone {
            index: 0,
            final_block: Some(ContentBlock::Text {
                text: "Hello from UniversalEvent.".to_string(),
            }),
        },
        UniversalEvent::MessageDone {
            finish_reason: Some(FinishReason::Stop),
            usage: Some(Usage {
                input_tokens: Some(8),
                output_tokens: Some(4),
                total_tokens: Some(12),
            }),
            extensions: Default::default(),
        },
        UniversalEvent::ResponseDone {
            usage: None,
            extensions: Default::default(),
        },
    ];

    let mut state = EncodeState::default();
    let wire_events = OpenAiChatTranslator.encode_events(&events, &mut state)?;
    for event in wire_events {
        println!("{}", serde_json::to_string_pretty(&event)?);
    }
    Ok(())
}
