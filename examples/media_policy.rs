use va_ai_api_bridge::{
    ContentBlock, OpenAiChatTranslator, Role, UniversalItem, UniversalRequest, WireTranslator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut request = UniversalRequest {
        model: Some("deepseek-v4-pro".to_string()),
        input: vec![UniversalItem::Message {
            role: Role::User,
            id: None,
            content: vec![
                ContentBlock::Text {
                    text: "Please inspect this image.".to_string(),
                },
                ContentBlock::Image {
                    media_type: Some("image/png".to_string()),
                    url: Some("https://example.test/screenshot.png".to_string()),
                    data: None,
                    extensions: Default::default(),
                },
            ],
            extensions: Default::default(),
        }],
        ..UniversalRequest::default()
    };

    omit_unsupported_media(&mut request, "DeepSeek", "deepseek-v4-pro");

    let upstream_body = OpenAiChatTranslator.encode_request(&request)?;
    println!("{}", serde_json::to_string_pretty(&upstream_body)?);
    Ok(())
}

fn omit_unsupported_media(request: &mut UniversalRequest, provider: &str, model: &str) {
    for item in &mut request.input {
        let UniversalItem::Message { content, .. } = item else {
            continue;
        };
        for block in content {
            if matches!(block, ContentBlock::Image { .. }) {
                *block = ContentBlock::Text {
                    text: format!(
                        "[Image attachment omitted: {provider} {model} does not support image input. Do not infer image contents; ask the user to describe it or switch models.]"
                    ),
                };
            }
        }
    }
}
