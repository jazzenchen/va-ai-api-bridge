# Getting Started

This guide shows the smallest useful SDK flow: decode one protocol into IR and encode it into another protocol.

## Install

Use the published crate from crates.io:

```bash
cargo add va-ai-api-bridge serde_json
```

Or add the dependencies manually:

```toml
[dependencies]
va-ai-api-bridge = "0.1.4"
serde_json = "1"
```

The package name uses hyphens on crates.io. Rust imports use underscores:

```rust
use va_ai_api_bridge::OpenAiChatTranslator;
```

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

## Apply Media Capability Policy

Hosts should resolve the final target model before target encoding. If the host already has JSON model metadata, pass it directly:

```rust
use serde_json::json;
use va_ai_api_bridge::sanitize_unsupported_media_from_json;

let report = sanitize_unsupported_media_from_json(
    &mut universal_request,
    json!({
        "providerLabel": "DeepSeek",
        "model": "deepseek-v4-pro",
        "capabilities": {
            "inputModalities": ["text"]
        }
    }),
)?;
```

If the host has already deserialized its profile/catalog data, use the typed `ResolvedModelSpec`:

```rust
use va_ai_api_bridge::{sanitize_unsupported_media, ModelCapabilities, ResolvedModelSpec};

let model = ResolvedModelSpec {
    provider_label: Some("DeepSeek".to_string()),
    model: "deepseek-v4-pro".to_string(),
    capabilities: ModelCapabilities {
        input_modalities: vec!["text".to_string()],
        ..ModelCapabilities::default()
    },
    extensions: Default::default(),
};

let report = sanitize_unsupported_media(&mut universal_request, &model);
```

Run media policy before encoding the target protocol so unsupported `Image` or `File` blocks become safe text placeholders instead of being sent upstream.

## Run the Examples

```bash
cargo run --example translate_request
cargo run --example provider_adapter
cargo run --example media_policy
cargo run --example media_policy_typed
cargo run --example stream_events
```
