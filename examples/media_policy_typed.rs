use va_ai_api_bridge::{
    sanitize_unsupported_media, ContentBlock, ModelCapabilities, OpenAiChatTranslator,
    ResolvedModelSpec, Role, UniversalItem, UniversalRequest, WireTranslator,
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

    let model = ResolvedModelSpec {
        provider_label: Some("DeepSeek".to_string()),
        model: "deepseek-v4-pro".to_string(),
        capabilities: ModelCapabilities {
            input_modalities: vec!["text".to_string()],
            ..ModelCapabilities::default()
        },
        extensions: Default::default(),
    };

    let report = sanitize_unsupported_media(&mut request, &model);
    eprintln!("image omitted: {}", report.image_omitted);

    let upstream_body = OpenAiChatTranslator.encode_request(&request)?;
    println!("{}", serde_json::to_string_pretty(&upstream_body)?);
    Ok(())
}
