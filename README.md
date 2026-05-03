# va-ai-api-proxy

Protocol translation primitives for AI API request and response shapes.

`va-ai-api-proxy` is intentionally not an HTTP gateway. It does not perform networking, store credentials, manage accounts, retry upstreams, or own chat history. It provides the shared Rust types and traits VibeAround can use to translate between API package shapes such as OpenAI Responses, OpenAI Chat Completions, and Anthropic Messages.

## Boundary

```text
wire API payload
  -> schema::{openai, anthropic} or serde_json::Value
  -> WireTranslator
  -> UniversalRequest / UniversalEvent
  -> optional provider adapter
  -> provider wire payload
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

## Status

This crate is an early API skeleton. The schema layer is deliberately permissive and keeps unknown fields so providers can evolve without breaking the proxy. The first production integrations should keep concrete OpenAI/Anthropic translators in host code until the golden fixtures are ready to migrate here.
