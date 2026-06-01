# Protocols

`WireProtocol` identifies the supported source and target wire protocol families.

| `WireProtocol` | String | Translator | Official reference |
| --- | --- | --- | --- |
| `OpenAiResponses` | `openai-responses` | `OpenAiResponsesTranslator` | <https://platform.openai.com/docs/api-reference/responses> |
| `OpenAiChat` | `openai-chat` | `OpenAiChatTranslator` | <https://platform.openai.com/docs/api-reference/chat/create> |
| `AnthropicMessages` | `anthropic-messages` | `AnthropicMessagesTranslator` | <https://docs.anthropic.com/en/api/messages> |
| `GeminiGenerateContent` | `gemini-generate-content` | `GeminiGenerateContentTranslator` | <https://ai.google.dev/api/generate-content> |

Related official references:

- Anthropic tool use: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview>
- Gemini function calling: <https://ai.google.dev/gemini-api/docs/function-calling>
- Gemini thought signatures: <https://ai.google.dev/gemini-api/docs/thought-signatures>
- Gemini OpenAI compatibility: <https://ai.google.dev/gemini-api/docs/openai>

## Gemini Thought Signatures

Gemini thinking models may attach `thoughtSignature` to `functionCall` parts and expect that value to be replayed with tool history. The translator keeps real signatures in IR extensions and writes them back when available.

When a host converts tool history from another protocol, such as OpenAI-compatible `tool_calls`, there may be no real Gemini signature to replay. In that case, Gemini request/response encoding writes `skip_thought_signature_validator` on `functionCall` parts as a stateless fallback. This skip value prevents validator failures; it does not reconstruct Gemini's hidden thinking state.

## Gemini Tool Schemas

Gemini `functionDeclarations[].parameters` uses the API `Schema` object, not a full JSON Schema draft. When encoding Gemini requests, the translator recursively keeps supported Schema fields and drops unsupported draft keywords such as `$schema`, `additionalProperties`, and `propertyNames`. This keeps cross-protocol tools from Claude or OpenAI-compatible agents within Gemini's accepted request shape.

## Protocol Alias Notes

Hosts may expose shorter route names such as `anthropic` or `gemini`, but the SDK-level enum uses explicit protocol names. Keep host route aliases outside the crate so the public SDK API remains unambiguous.
