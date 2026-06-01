# Streaming

Streaming uses `UniversalEvent` as the shared event vocabulary between source and target protocols.

## Event Lifecycle

A normal streamed response looks like this:

```text
ResponseStart
MessageStart
ContentStart
TextDelta / ReasoningDelta / ToolCallDelta
ContentDone
MessageDone
ResponseDone
```

Not every protocol emits every lifecycle marker. Translators use `DecodeState` and `EncodeState` to synthesize stable message starts, content indexes, and tool-call indexes when the wire protocol streams partial information.

## Host Responsibilities

The host owns transport details:

- reading upstream byte chunks
- parsing upstream SSE frames when the provider uses SSE
- preserving one `DecodeState` per upstream response
- preserving one `EncodeState` per downstream response
- framing `WireEvent` values as the source protocol expects
- closing the stream when the upstream sends a terminal event

Do not reuse decode or encode state across independent responses.

## Provider Event Transforms

Provider adapters may transform event sequences after target decode and before source encode. Examples include splitting provider-specific thinking tags into reasoning events or normalizing tagged tool-call text into structured tool-call events.
