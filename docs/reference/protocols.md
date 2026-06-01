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
- Gemini OpenAI compatibility: <https://ai.google.dev/gemini-api/docs/openai>

## Protocol Alias Notes

Hosts may expose shorter route names such as `anthropic` or `gemini`, but the SDK-level enum uses explicit protocol names. Keep host route aliases outside the crate so the public SDK API remains unambiguous.
