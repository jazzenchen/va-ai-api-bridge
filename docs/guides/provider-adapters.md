# Provider Adapters

Provider adapters handle vendor-specific package behavior that is not part of a general wire protocol.

A translator should answer: "How does OpenAI Chat map to IR?" A provider adapter should answer: "What does this OpenAI-compatible provider require that is different from the nominal protocol?"

## Request Placement

Apply request adapters after target protocol encoding and before the host sends the upstream request.

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

## Response and Stream Placement

For non-streaming responses, decode upstream JSON into `UniversalEvent`, then call `transform_upstream_events` before encoding back to the source protocol.

For streaming responses, apply provider transforms to each decoded event batch before downstream encoding. Keep adapter state for the lifetime of the response stream when the adapter type stores incremental parsing state.

## Built-in Adapter Responsibilities

| Adapter | Responsibility |
| --- | --- |
| `DeepSeekBridgeAdapter` | Thinking toggle, forced-tool compatibility, tool history repair, reasoning replay. |
| `DashScopeBridgeAdapter` | `enable_thinking`, forced tool choice normalization, unsupported reasoning field removal. |
| `KimiBridgeAdapter` | Kimi coding model aliases and tagged tool-call normalization. |
| `MimoBridgeAdapter` | Thinking enablement, reasoning replay, tool-history repair, `tool_calls: null` normalization. |
| `MiniMaxBridgeAdapter` | Chat setting clamps, system message folding, `<think>` tag splitting. |
| `XaiBridgeAdapter` | Unsupported Responses field/tool stripping and encrypted reasoning history removal. |
| `ZaiBridgeAdapter` | Reasoning-off mapping to provider thinking disablement. |

## Adapter Design Rules

- Keep adapter logic deterministic and local to payload shape.
- Do not perform network calls from adapters.
- Do not read credentials from adapters.
- Prefer preserving semantics over preserving provider quirks.
- Add adapter behavior only when the behavior is provider-specific rather than protocol-generic.
