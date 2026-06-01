# va-ai-api-bridge (va-aab)

`va-ai-api-bridge` provides Rust SDK primitives for translating AI API request, response, and stream shapes. It is designed for hosts that need to expose one agent-facing API while sending requests to providers that use another API shape.

The crate is intentionally not an HTTP gateway. It does not perform networking, store credentials, own model routing, retry upstreams, or persist chat history. It focuses on the translation layer: wire payloads, protocol-neutral IR, streaming events, and provider-specific package transforms.

## Documentation

Start with the [documentation index](docs/README.md).

- [Getting started](docs/guides/getting-started.md): minimal request translation and dynamic translator dispatch.
- [Host integration](docs/guides/host-integration.md): where a bridge host should decode, sanitize, adapt, encode, and send requests.
- [Architecture](docs/concepts/architecture.md): crate layers and data flow.
- [Universal IR](docs/concepts/universal-ir.md): request, response, item, content, tool, reasoning, and usage model.
- [Media content](docs/concepts/media-content.md): image/file representation and safe handling when providers lack media support.
- [Provider adapters](docs/guides/provider-adapters.md): how provider-specific quirks fit around protocol translation.

## Boundary

```text
agent wire request
  -> source WireTranslator
  -> UniversalRequest
  -> host capability policy
  -> target WireTranslator
  -> ProviderBridgeAdapter request transform
  -> upstream wire request

upstream wire response / stream chunk
  -> target WireTranslator
  -> UniversalEvent
  -> ProviderBridgeAdapter event transform
  -> source WireTranslator
  -> agent wire response / stream event
```

The host application remains responsible for:

- HTTP routes and upstream requests
- authorization headers and profile credentials
- model selection and capability policy
- chat history and launch/session context
- SSE framing and transport lifecycle
- plugin loading and sandboxing

The SDK provides:

- protocol-neutral request and response types
- protocol-neutral stream events
- wire translators for supported API families
- provider adapters for package-shape quirks
- provider catalog schema types a host can serialize or extend

## Crate Layout

- `protocol`: supported wire protocol identifiers
- `schema`: provider catalog types and light serde shells for supported wire payloads
- `universal`: protocol-neutral request, content, tool, reasoning, and usage types
- `stream`: protocol-neutral streaming event types and translator state
- `translator`: traits and implementations for wire protocol translation
- `adapter`: generic adapter traits and bridge context structs
- `providers`: built-in provider adapters for common provider quirks

## Built-in Translators

- `OpenAiChatTranslator`: `/v1/chat/completions`
- `OpenAiResponsesTranslator`: `/v1/responses`
- `AnthropicMessagesTranslator`: `/v1/messages`
- `GeminiGenerateContentTranslator`: `/{version}/models/{model}:generateContent`

## Examples

Runnable examples live under `examples/`:

- `cargo run --example translate_request`
- `cargo run --example provider_adapter`
- `cargo run --example stream_events`
