# va-ai-api-bridge (va-aab)

Protocol translation primitives and SDK types for AI API request, response, and stream shapes.

`va-ai-api-bridge` (nickname: `va-aab`) is intentionally not an HTTP gateway. It does not perform networking, store credentials, manage accounts, retry upstreams, or own chat history. It provides the shared Rust types and traits hosts such as VibeAround can use to translate between API package shapes such as OpenAI Responses, OpenAI Chat Completions, Anthropic Messages, and Gemini Generate Content.

## Documentation

- [SDK guide](docs/sdk-guide.md): public API surface, host boundary, examples, and SDK maturity.
- [Architecture and IR](docs/architecture-and-ir.md): module layering, Universal IR structure, protocol mapping, and contribution workflow.
- [Provider integration guide](docs/provider-integration-guide.md): official provider references, protocol matrix, adapter notes, and VibeAround test coverage.

## Boundary

```text
agent wire request
  -> source WireTranslator
  -> UniversalRequest
  -> optional provider adapter
  -> target WireTranslator
  -> upstream wire request

upstream wire response / stream chunk
  -> target WireTranslator
  -> UniversalEvent
  -> optional provider adapter
  -> source WireTranslator
  -> agent wire response / stream event
```

The host application remains responsible for:

- HTTP routes and upstream requests
- authorization headers and profile credentials
- chat history and launch/session context
- SSE framing and transport lifecycle
- plugin loading and sandboxing

Provider adapters only transform package shapes:

- prepare a target protocol request body before the host sends it
- map provider responses or stream chunks back into universal events
- read host-supplied context such as provider settings or normalized history

## Crate Layout

- `protocol`: supported wire protocol identifiers
- `schema`: provider catalog types and light serde shells for supported wire payloads
- `universal`: protocol-neutral request, content, tool, reasoning, and usage types
- `stream`: protocol-neutral streaming event types and translator state
- `translator`: behavior traits for translating schema/value payloads to and from universal types
- `adapter`: traits and context for provider-specific package transforms
- `providers`: built-in provider adapters for common OpenAI-compatible quirks

## Built-in Translators

- `OpenAiChatTranslator`: `/v1/chat/completions`
- `OpenAiResponsesTranslator`: `/v1/responses`
- `AnthropicMessagesTranslator`: `/v1/messages`
- `GeminiGenerateContentTranslator`: `/{version}/models/{model}:generateContent`

## Status

This crate is an early API skeleton. The schema layer is deliberately permissive and keeps unknown fields so providers can evolve without breaking the bridge. Built-in translators cover the common request, response, and stream packet shapes; built-in provider adapters cover the provider-specific package transforms that can stay independent of host networking, credentials, and profile storage.

As of the current `0.x` line, the crate is suitable as an internal SDK boundary but should still add runnable examples, generated API docs, and env-gated live integration tests before being treated as a polished external SDK.
