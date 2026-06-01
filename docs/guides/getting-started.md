# Getting Started

This guide shows the smallest useful SDK flow: decode one protocol into IR and encode it into another protocol.

## Translate a Request

```rust
use serde_json::json;
use va_ai_api_bridge::{AnthropicMessagesTranslator, OpenAiChatTranslator, WireTranslator};

let source = OpenAiChatTranslator;
let target = AnthropicMessagesTranslator;

let universal = source.decode_request(json!({
    "model": "gpt-4.1",
    "messages": [
        { "role": "system", "content": "You are concise." },
        { "role": "user", "content": "Say hello." }
    ]
}))?;

let anthropic_body = target.encode_request(&universal)?;
```

## Dispatch Dynamically

Use `translator_for_protocol` when the host route decides protocols at runtime.

```rust
use serde_json::json;
use va_ai_api_bridge::{translator_for_protocol, WireProtocol};

let source = translator_for_protocol(WireProtocol::OpenAiChat);
let target = translator_for_protocol(WireProtocol::AnthropicMessages);

let universal = source.decode_request(json!({
    "model": "chat-model",
    "messages": [{ "role": "user", "content": "Hello" }]
}))?;
let target_body = target.encode_request(&universal)?;
```

## Run the Examples

```bash
cargo run --example translate_request
cargo run --example provider_adapter
cargo run --example stream_events
```
