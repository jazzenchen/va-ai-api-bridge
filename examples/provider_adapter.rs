use serde_json::json;
use va_ai_api_bridge::{
    DeepSeekBridgeSettings, OpenAiChatTranslator, OpenAiResponsesTranslator, ProviderBridgeAdapter,
    ProviderBridgeAdapterConfig, ProviderRequestSource, WireProtocol, WireTranslator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let original_responses_body = json!({
        "model": "deepseek-v4-pro",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "Call lookup_bridge_fact, then answer." }
                ]
            }
        ],
        "tools": [{
            "type": "function",
            "name": "lookup_bridge_fact",
            "description": "Look up a bridge fact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": { "type": "string" }
                },
                "required": ["topic"]
            }
        }],
        "tool_choice": "required"
    });

    let universal = OpenAiResponsesTranslator.decode_request(original_responses_body.clone())?;
    let mut chat_body = OpenAiChatTranslator.encode_request(&universal)?;

    let mut adapter = ProviderBridgeAdapter::for_provider(
        "deepseek",
        WireProtocol::OpenAiChat,
        ProviderBridgeAdapterConfig {
            deepseek: DeepSeekBridgeSettings {
                thinking: true,
                replay_reasoning_content: true,
            },
            ..ProviderBridgeAdapterConfig::default()
        },
    );
    adapter.prepare_chat_request(
        ProviderRequestSource::OpenAiResponses,
        &original_responses_body,
        &mut chat_body,
    );

    println!("{}", serde_json::to_string_pretty(&chat_body)?);
    Ok(())
}
