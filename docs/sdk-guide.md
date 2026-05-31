# SDK Guide

`va-ai-api-bridge` is an embeddable Rust SDK for translating AI API payloads. It is not an HTTP gateway and intentionally does not own networking, credentials, account state, retries, model routing, or chat persistence.

## Current SDK Maturity

The crate is SDK-shaped and usable by a host such as VibeAround today:

- It exposes protocol-neutral IR types, wire translators, stream events, provider adapters, errors, and provider catalog schema from `src/lib.rs`.
- It keeps the host boundary clean: callers pass `serde_json::Value` payloads in and receive translated `serde_json::Value` payloads or `UniversalEvent` streams out.
- It supports OpenAI Responses, OpenAI Chat Completions, Anthropic Messages, and Gemini Generate Content translator families.
- It includes provider-specific adapters for recurring compatibility gaps without embedding HTTP or account logic.

It is not yet a fully polished public SDK:

- The crate is still `0.x`; public APIs should be treated as evolving.
- There are no runnable examples under `examples/` yet.
- There is no generated provider catalog shipped as data.
- Live provider integration tests are external to this crate and should be promoted into an env-gated integration harness.

## Public Surface

| Area | Main exports | Responsibility |
| --- | --- | --- |
| Protocol IDs | `WireProtocol` | Stable enum for supported source and target wire protocols. |
| IR | `UniversalRequest`, `UniversalResponse`, `UniversalItem`, `ContentBlock`, `UniversalTool`, `ToolChoice` | Protocol-neutral request, response, content, tools, reasoning, and usage. |
| Streaming | `UniversalEvent`, `DecodeState`, `EncodeState`, `WireEvent` | Protocol-neutral stream lifecycle and translator state. |
| Translators | `WireTranslator`, `translator_for_protocol`, `OpenAiChatTranslator`, `OpenAiResponsesTranslator`, `AnthropicMessagesTranslator`, `GeminiGenerateContentTranslator` | Decode source wire payloads to IR and encode IR back to a target wire protocol. |
| Provider adapters | `ProviderBridgeAdapter`, `ProviderBridgeAdapterConfig`, provider-specific adapter structs | Patch provider quirks before sending requests or after receiving upstream events. |
| Catalog schema | `ProviderCatalog`, `ProviderProtocol`, `ProviderModel`, `ProviderSetting` | Serializable profile/catalog metadata for host applications. |
| Errors | `ApiBridgeError`, `Result` | SDK-level error reporting without transport concerns. |

## Host Boundary

The host application is responsible for:

- Exposing HTTP routes and selecting source/target protocols.
- Reading profile credentials and adding authorization headers.
- Choosing upstream base URLs and executing requests.
- Persisting conversation history and launch/session state.
- Framing SSE or streaming transport chunks.
- Handling retries, rate limits, quota, model fallback, and observability.

The SDK is responsible for:

- Normalizing request, response, and stream packet shapes.
- Preserving unknown fields in `extensions` or `SourcePayload` where possible.
- Keeping tool calls, tool results, reasoning, usage, and finish reasons coherent across protocols.
- Applying provider-specific package transforms that are independent from HTTP transport.

## Basic Request Translation

```rust
use serde_json::json;
use va_ai_api_bridge::{
    AnthropicMessagesTranslator, OpenAiChatTranslator, WireTranslator,
};

let source = OpenAiChatTranslator;
let target = AnthropicMessagesTranslator;

let universal = source.decode_request(json!({
    "model": "gpt-4.1",
    "messages": [
        { "role": "system", "content": "You are concise." },
        { "role": "user", "content": "Say hello." }
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": { "type": "object", "properties": {} }
        }
    }]
}))?;

let anthropic_body = target.encode_request(&universal)?;
```

Hosts that dispatch dynamically can use `translator_for_protocol(protocol)` to get a boxed translator for any supported `WireProtocol`.

## Provider Adapter Pipeline

Provider adapters should be applied after IR has been encoded into the target provider's nominal wire protocol and before the host sends the upstream request.

```rust
use va_ai_api_bridge::{
    ProviderBridgeAdapter, ProviderBridgeAdapterConfig, ProviderRequestSource, WireProtocol,
};

let mut adapter = ProviderBridgeAdapter::for_provider(
    "deepseek",
    WireProtocol::OpenAiChat,
    ProviderBridgeAdapterConfig::default(),
);

adapter.prepare_chat_request(
    ProviderRequestSource::OpenAiResponses,
    &original_source_body,
    &mut target_chat_body,
);
```

Common adapter responsibilities include:

- Disabling provider reasoning modes when they conflict with forced tool choice.
- Replaying provider-required reasoning fields in multi-turn tool history.
- Repairing tool call and tool result adjacency for stricter OpenAI-compatible endpoints.
- Splitting provider-specific `<think>` tags into reasoning events.
- Removing unsupported request fields while preserving the user's semantic intent.

## Streaming Lifecycle

Streaming is represented as a protocol-neutral event pipeline:

```text
wire stream chunk
  -> target WireTranslator::decode_stream_chunk(...)
  -> Vec<UniversalEvent>
  -> ProviderBridgeAdapter::transform_upstream_events(...)
  -> source WireTranslator::encode_events(...)
  -> Vec<WireEvent>
  -> host SSE framing
```

Use a persistent `DecodeState` per upstream stream and an `EncodeState` per downstream stream. Do not reuse those states across independent responses.

## Error Handling

SDK errors indicate translation or shape problems. Transport failures remain host errors. Recommended host behavior:

- Return upstream HTTP errors unchanged when translation was not involved.
- Attach profile, source protocol, target protocol, and provider ID to host logs.
- Avoid logging secrets, authorization headers, OAuth tokens, or raw user credentials.
- Include raw payload snippets only after redaction.

## Official Protocol References

The docs in this repository are written against the current official API references:

- OpenAI Responses API: <https://platform.openai.com/docs/api-reference/responses>
- OpenAI Chat Completions API: <https://platform.openai.com/docs/api-reference/chat/create>
- Anthropic Messages API: <https://docs.anthropic.com/en/api/messages>
- Anthropic tool use: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview>
- Gemini Generate Content API: <https://ai.google.dev/api/generate-content>
- Gemini function calling: <https://ai.google.dev/gemini-api/docs/function-calling>
- Gemini OpenAI compatibility: <https://ai.google.dev/gemini-api/docs/openai>

## Recommended Next SDK Work

- Add `examples/translate_request.rs`, `examples/stream_roundtrip.rs`, and `examples/provider_adapter.rs`.
- Promote VibeAround live checks into env-gated integration tests.
- Ship a generated provider catalog or fixtures for common provider profiles.
- Add docs.rs examples to the public traits and core IR structs.
