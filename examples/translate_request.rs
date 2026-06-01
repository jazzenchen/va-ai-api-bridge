use serde_json::json;
use va_ai_api_bridge::{translator_for_protocol, WireProtocol};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = translator_for_protocol(WireProtocol::OpenAiChat);
    let target = translator_for_protocol(WireProtocol::AnthropicMessages);

    let universal = source.decode_request(json!({
        "model": "gpt-4.1",
        "messages": [
            { "role": "system", "content": "You are concise." },
            { "role": "user", "content": "Say hello from the bridge." }
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup_bridge_fact",
                "description": "Look up a bridge fact.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": { "type": "string" }
                    },
                    "required": ["topic"]
                }
            }
        }],
        "tool_choice": "auto"
    }))?;

    let anthropic_body = target.encode_request(&universal)?;
    println!("{}", serde_json::to_string_pretty(&anthropic_body)?);
    Ok(())
}
