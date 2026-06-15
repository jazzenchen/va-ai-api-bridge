# Universal IR

The Universal IR is the crate's shared vocabulary. Translators should map protocol-specific packet shapes into this model before any target protocol is chosen.

## `UniversalRequest`

| Field | Meaning |
| --- | --- |
| `id` | Optional request or host identifier. |
| `model` | Requested model after host/model mapping. |
| `instructions` | System/developer instructions separated from conversational input. |
| `input` | Ordered messages, tool calls, tool results, reasoning items, and unknown items. |
| `tools` | Function/tool declarations with JSON schemas plus tool-level metadata such as OpenAI strictness. |
| `tool_choice` | Tool selection policy: auto, none, required, or a specific tool. |
| `stream` | Whether the caller requested streaming. |
| `generation` | Temperature, top-p, output limit, and extension settings. |
| `reasoning` | Reasoning effort, budget, visibility, and extensions. |
| `source` | Original protocol and optional raw payload for loss-aware behavior. |
| `extensions` | Escape hatch for fields not represented by first-class IR. |

## `UniversalItem`

| Variant | Use |
| --- | --- |
| `Message` | Role-bearing text/media/tool content. |
| `ToolCall` | Model-requested function/tool invocation. |
| `ToolResult` | Host result for a prior tool call. |
| `Reasoning` | Visible or encrypted reasoning content. |
| `Unknown` | Raw item that should not be discarded yet. |

## `ContentBlock`

| Variant | Use |
| --- | --- |
| `Text` | Plain text. |
| `Image` | Image input by URL or base64 data. |
| `File` | Non-image file input by URL or base64 data. |
| `ToolCall` | Tool call embedded inside message content. |
| `ToolResult` | Tool result embedded inside message content. |
| `Reasoning` | Reasoning text or encrypted reasoning token. |
| `Unknown` | Raw content block for future or provider-specific modalities. |

## Mapping Principles

- Preserve semantic intent before preserving exact packet shape.
- Keep unknown data in `Unknown`, `extensions`, or `SourcePayload` when a lossless mapping is not available.
- Keep media as `Image` or `File`; do not collapse attachments into text unless the host has explicitly decided that a target model cannot accept them.
- Keep tool calls and tool results structurally paired so target translators can satisfy provider ordering rules.
- Use `UniversalEvent` as the common response representation for both streaming and non-streaming responses.

## Tool Strictness

`UniversalTool.strict` represents OpenAI function strict mode. It is metadata about the tool declaration, not a JSON Schema keyword inside `input_schema`.

- OpenAI Chat decodes and encodes it at `tools[].function.strict`.
- OpenAI Responses decodes and encodes it at `tools[].strict`.
- Anthropic Messages has no equivalent tool-level field, so translators ignore or drop it.
- Gemini GenerateContent has no equivalent tool-level field, so translators ignore or drop it and also sanitize unsupported `strict` keys out of Gemini `parameters` schemas.
