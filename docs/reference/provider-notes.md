# Provider Notes

This reference lists provider interfaces and adapter considerations that affect request/response shape. It is not a test report and does not describe profile-specific availability.

| Provider | Documented interfaces | SDK note |
| --- | --- | --- |
| DeepSeek | Anthropic-compatible API and thinking mode: <https://api-docs.deepseek.com/guides/anthropic_api>, <https://api-docs.deepseek.com/guides/thinking_mode> | `DeepSeekBridgeAdapter` handles thinking/tool-choice compatibility and reasoning replay for OpenAI-compatible chat targets. |
| DashScope / Qwen | OpenAI compatibility and tool calling: <https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope>, <https://www.alibabacloud.com/help/doc-detail/3016809.html> | `DashScopeBridgeAdapter` maps reasoning intent to `enable_thinking` and normalizes forced tool choice. |
| xAI | Chat Completions and Responses docs: <https://docs.x.ai/developers/rest-api-reference/inference/chat>, <https://docs.x.ai/docs/guides/chat-completions> | `XaiBridgeAdapter` strips unsupported Responses fields/tools before upstream send. |
| Google Gemini | GenerateContent, function calling, and thought signatures: <https://ai.google.dev/api/generate-content>, <https://ai.google.dev/gemini-api/docs/function-calling>, <https://ai.google.dev/gemini-api/docs/thought-signatures> | `GeminiGenerateContentTranslator` preserves real `thoughtSignature` values when present. When encoding a `functionCall` without one, it writes `skip_thought_signature_validator` as a stateless fallback. Tool schemas are sanitized to Gemini's supported `Schema` fields before upstream send. |
| MiniMax | Anthropic-compatible and OpenAI-compatible APIs: <https://platform.minimaxi.com/docs/api-reference/text-anthropic-api>, <https://platform.minimaxi.com/docs/api-reference/text-openai-api> | `MiniMaxBridgeAdapter` handles system folding, setting clamps, and `<think>` splitting. |
| Moonshot / Kimi | Kimi API overview: <https://www.kimi.com/help/kimi-api/api-overview> | `KimiBridgeAdapter` handles coding model aliases and tagged tool-call text. |
| MiMo | OpenAI-compatible Chat and Anthropic-compatible APIs: <https://platform.xiaomimimo.com/docs/en-US/api/chat/openai-api>, <https://platform.xiaomimimo.com/docs/api/chat/anthropic-api> | `MimoBridgeAdapter` handles thinking and tool-history compatibility for chat targets. |
| NVIDIA NIM | LLM API reference: <https://docs.api.nvidia.com/nim/reference/llm-apis> | Current SDK use is generic OpenAI-compatible translation unless a provider-specific quirk is identified. |
| Volcengine Ark | OpenAI-compatible Chat and compatible tool setup: <https://www.volcengine.com/docs/82379/1298454>, <https://www.volcengine.com/docs/82379/2160841> | Current SDK use is generic OpenAI/Anthropic-compatible translation unless a provider-specific quirk is identified. |

Hosts should keep provider availability, credentials, endpoint IDs, model lists, and capability flags in profile/catalog data rather than hard-coding those choices in translators.
