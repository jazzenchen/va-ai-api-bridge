use va_ai_api_bridge::{
    sanitize_unsupported_media_from_json, ContentBlock, OpenAiChatTranslator, Role, UniversalItem,
    UniversalRequest, WireTranslator,
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

    let report = sanitize_unsupported_media_from_json(
        &mut request,
        serde_json::json!({
            "providerLabel": "DeepSeek",
            "model": "deepseek-v4-pro",
            "capabilities": {
                "inputModalities": ["text"]
            }
        }),
    )?;
    eprintln!("image omitted: {}", report.image_omitted);

    let upstream_body = OpenAiChatTranslator.encode_request(&request)?;
    println!("{}", serde_json::to_string_pretty(&upstream_body)?);
    Ok(())
}
