# Host Integration

A host bridge combines this crate with routing, credentials, profiles, model metadata, and transport.

## Request Pipeline

1. Resolve the agent-facing source protocol from the route.
2. Decode the incoming JSON body with the source translator.
3. Resolve the profile, upstream provider, target protocol, and final target model.
4. Apply host policy to the IR: model mapping, media capability handling, server-tool fallback, defaults, or profile-specific metadata.
5. Encode the IR with the target translator.
6. Apply the provider adapter request transform.
7. Send the JSON body to the upstream provider with host-managed auth and headers.

## Response Pipeline

1. Read the upstream response body or stream frames.
2. Decode with the target translator.
3. Apply provider adapter response/event transforms.
4. Encode events for the source protocol.
5. Return JSON or SSE using the host's HTTP framework.

## Capability Policy

Model capability policy should happen after the host knows the final target model. For media, that means building a `ResolvedModelSpec` from the host catalog/profile data, then replacing unsupported `Image` or `File` blocks with safe text placeholders before target encoding. See [Media content](../concepts/media-content.md).

The host can pass the final model metadata either as JSON or as a typed struct:

```rust
use va_ai_api_bridge::{sanitize_unsupported_media, ResolvedModelSpec, UniversalRequest};

fn apply_model_policy(request: &mut UniversalRequest, model: &ResolvedModelSpec) {
    let report = sanitize_unsupported_media(request, model);
    if report.changed() {
        // The host may log this without recording attachment bytes or user content.
    }
}
```

The resolved model spec should represent the final upstream model after alias mapping and profile overrides. Endpoint-level capabilities and model-level capabilities can be merged with `ModelCapabilities::union`.

## Server Tool Policy

`UniversalRequest.server_tools` records provider-native server tool declarations from the agent-facing request. These declarations are intentionally separate from `UniversalRequest.tools`, which contains function tools that the host or agent can execute after the model emits arguments.

The SDK preserves server-tool intent but does not execute it. Host policy should run after routing has selected the final provider:

1. If the provider and target protocol support the server tool natively, the host can preserve or re-inject the native declaration before sending upstream.
2. If the provider lacks native support but supports function tools, the host can inject a fallback function tool into `request.tools` and run the resulting tool call loop itself.
3. If neither native server tools nor function tools are available, the host should return a clear unsupported error or apply a host-specific context-injection fallback.

For web search, this means the bridge host can recognize `ServerToolKind::WebSearch`, inject a host-owned search function, execute the search provider outside this crate, append the tool result, and continue the upstream model call.

## Gemini Tool History

Gemini thinking models can require `thoughtSignature` values on replayed `functionCall` history. The SDK preserves real signatures when they are present in Gemini wire payloads. If a host routes OpenAI-compatible or other cross-protocol tool history into Gemini and no real signature exists, Gemini encoding uses `skip_thought_signature_validator` as a stateless fallback. This avoids Gemini rejecting the request, but it is not equivalent to replaying a real model-generated signature.

## What Not To Put In This Crate

Do not add host concerns to `va-ai-api-bridge`:

- API keys or OAuth tokens
- HTTP clients or retry loops
- account state or billing behavior
- database-backed history
- provider profile storage
- UI-specific launch metadata

Those belong to the embedding application.
