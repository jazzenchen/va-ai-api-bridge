# Provider Integration Guide

This document tracks provider-specific protocol support, official references, adapter behavior, and VibeAround live-test coverage.

## Canonical Protocols

| Protocol | Bridge name | Official reference |
| --- | --- | --- |
| OpenAI Responses | `openai-responses` | <https://platform.openai.com/docs/api-reference/responses> |
| OpenAI Chat Completions | `openai-chat` | <https://platform.openai.com/docs/api-reference/chat/create> |
| Anthropic Messages | `anthropic-messages` | <https://docs.anthropic.com/en/api/messages> |
| Anthropic tool use | part of `anthropic-messages` | <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview> |
| Gemini Generate Content | `gemini-generate-content` | <https://ai.google.dev/api/generate-content> |
| Gemini function calling | part of `gemini-generate-content` | <https://ai.google.dev/gemini-api/docs/function-calling> |
| Gemini OpenAI compatibility | upstream/provider option | <https://ai.google.dev/gemini-api/docs/openai> |

## Provider Matrix

Status is based on local VibeAround profiles and live checks on 2026-06-01 Asia/Shanghai.

| Provider | Official interfaces referenced | Profile target protocols tested | Adapter | Direct live status | Notes |
| --- | --- | --- | --- | --- | --- |
| DeepSeek | OpenAI-format chat, Anthropic-format Messages, thinking mode: <https://api-docs.deepseek.com/guides/anthropic_api>, <https://api-docs.deepseek.com/guides/thinking_mode> | `anthropic`, `openai-chat` | `DeepSeekBridgeAdapter` | Passed both direct targets | Supports both Anthropic and Chat in our profiles. Forced tool choice is retested on both; adapter disables thinking for forced tools and can replay reasoning content for tool history. |
| DashScope / Qwen | OpenAI-compatible Chat, model API reference, tool calling: <https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope>, <https://help.aliyun.com/zh/dashscope/api-reference>, <https://www.alibabacloud.com/help/doc-detail/3016809.html> | `openai-chat` | `DashScopeBridgeAdapter` | Passed direct target | Adapter maps reasoning intent to `enable_thinking`; forced single-tool requests disable thinking because thinking models do not support forcing a specific tool. |
| xAI | Chat Completions and Responses: <https://docs.x.ai/developers/rest-api-reference/inference/chat>, <https://docs.x.ai/docs/guides/chat-completions> | `openai-responses`, `openai-chat` | `XaiBridgeAdapter` | Passed both direct targets | Adapter strips unsupported Responses fields and encrypted reasoning history before sending to xAI Responses. |
| MiniMax | Anthropic-compatible API and OpenAI-compatible Chat docs: <https://platform.minimaxi.com/docs/api-reference/text-anthropic-api>, <https://platform.minimaxi.com/docs/api-reference/text-openai-api>, <https://platform.minimaxi.com/docs/api-reference/text-chat-openai> | `anthropic`, `openai-chat` | `MiniMaxBridgeAdapter` | Passed both direct targets | Adapter folds system messages, clamps unsupported chat settings, and splits `<think>` text into reasoning events. |
| Moonshot / Kimi | Kimi API overview: <https://www.kimi.com/help/kimi-api/api-overview>; Kimi CLI provider docs: <https://moonshotai.github.io/kimi-cli/en/configuration/providers.html> | `anthropic` | `KimiBridgeAdapter` | Passed direct target | Current profile uses Kimi coding Anthropic-style flow. Adapter normalizes coding model aliases and tagged tool-call sections. Keep watching official docs for a stable coding endpoint reference. |
| MiMo | OpenAI-compatible Chat, Anthropic-compatible Messages, quickstart: <https://platform.xiaomimimo.com/docs/en-US/api/chat/openai-api>, <https://platform.xiaomimimo.com/docs/api/chat/anthropic-api>, <https://platform.xiaomimimo.com/docs/en-US/quick-start/first-api-call> | `openai-chat` | `MimoBridgeAdapter` | Passed direct target | Adapter enables thinking, replays reasoning content for tool history, repairs tool call adjacency, and normalizes `tool_calls: null`. Profile currently targets Chat. |
| NVIDIA NIM | OpenAI-compatible Chat and experimental Responses: <https://docs.api.nvidia.com/nim/reference/llm-apis>, <https://docs.nvidia.com/nim/large-language-models/1.12.0/api-reference.html> | `openai-chat` | Generic | Passed direct target | No dedicated adapter needed for current profile. Treat provider-specific model/tool limitations as catalog data. |
| Volcengine Ark | OpenAI Chat, OpenAI/Anthropic compatible third-party tool setup, Coding Plan: <https://www.volcengine.com/docs/82379/1298454>, <https://www.volcengine.com/docs/82379/2160841>, <https://www.volcengine.com/docs/82379/1928262> | `anthropic`, `openai-chat` | Generic | Anthropic passed; OpenAI Chat failed tool-call check | Official docs expose both OpenAI-compatible and Anthropic-compatible base URLs. The current `ark-code-latest` Chat path returned no structured tool call in the direct tool test; keep as an observed upstream/model behavior until reproduced with a smaller request. |
| Gemini / Google Code Assist | Gemini Generate Content and function calling: <https://ai.google.dev/api/generate-content>, <https://ai.google.dev/gemini-api/docs/function-calling> | `gemini` | Generic | Blocked by OAuth | `/models` routing passed, but live tool conversation returned 401 because the Google account OAuth profile was not connected. |
| Custom / TokenRouter | Profile-defined, OpenAI/Anthropic-style upstream | `anthropic`, `openai-responses` | Generic | Blocked by quota | `/models` routing passed, but live calls returned 403 insufficient credit. |

## Protocol Choice by Provider

- Prefer the provider's native or explicitly documented compatibility protocol for tool-heavy coding agents.
- For Anthropic-first agents such as Claude Code, use Anthropic-compatible provider endpoints when the provider documents them.
- For OpenAI-first agents, use OpenAI Chat unless the provider's Responses API is explicitly supported and tested.
- For Gemini-native agents, keep `gemini-generate-content` as the source protocol and translate through IR only when the target is not Gemini.
- If a provider supports multiple protocols, test direct target calls and at least one cross-protocol tool round before marking it production-ready.

## Current Adapter Notes

| Adapter | Request behavior | Response/stream behavior |
| --- | --- | --- |
| `DeepSeekBridgeAdapter` | Adds `thinking` toggle, disables thinking for forced tool choice, repairs tool history, injects missing `reasoning_content` where needed. | Uses generic translators after request normalization. |
| `DashScopeBridgeAdapter` | Removes unsupported reasoning fields, converts compatible reasoning intent to `enable_thinking`, rewrites single required tool choice, disables thinking for forced tools. | Uses generic OpenAI Chat response handling. |
| `KimiBridgeAdapter` | Normalizes Kimi coding model aliases and disables thinking for Anthropic coding requests. | Converts tagged Kimi tool-call text into structured tool-call events. |
| `MimoBridgeAdapter` | Enables thinking, replays reasoning content, strips incompatible Anthropic reasoning blocks after conversion, repairs tool history. | Converts `tool_calls: null` to an empty array before OpenAI Chat decode. |
| `MiniMaxBridgeAdapter` | Folds multiple system messages, clamps `temperature`, `top_p`, and `max_completion_tokens` to supported ranges. | Splits `<think>...</think>` into reasoning events. |
| `XaiBridgeAdapter` | Removes unsupported Responses fields/tools and encrypted reasoning history. | Uses generic Responses/Chat response handling. |
| `ZaiBridgeAdapter` | Maps reasoning-off intent to provider thinking disablement. | Uses generic OpenAI Chat response handling. |

## Live Test Summary

VibeAround was used as the host because it imports this crate by local path from the sibling project.

| Test set | Result | Scope |
| --- | ---: | --- |
| `/v1/models` route matrix | 135/135 passed | All configured profiles × target protocols × agent/client protocol scopes. |
| Multi-round live tool conversations | 22/30 passed before fixes | Coverage-based cross-section across providers, agents, client APIs, target APIs. |
| Streaming smoke tests | 4/4 passed | DeepSeek Anthropic, xAI Responses, NVIDIA Chat via Anthropic client, DashScope Chat via Gemini client. |
| Direct target live tests | 11/15 passed | One native agent/client pair per provider target protocol. |
| Targeted retests after fixes | Passed | DashScope forced tool choice and DeepSeek forced tool choice on both Chat and Anthropic paths. |

Direct target failures were:

- Gemini: 401 because Google OAuth was not configured in the profile.
- Volcengine OpenAI Chat: no structured tool call returned for the direct tool test.
- Custom TokenRouter Anthropic and Responses: 403 insufficient credit.

The live tests are intentionally not a full Cartesian product of every profile, agent, client API, and target protocol because that would be expensive and rate-limit prone. The route matrix is exhaustive; live tool tests are coverage-based and should become env-gated integration tests.

## Minimum Provider Acceptance Checklist

- Official docs confirm the base URL, request path, tool-call format, stream format, and supported protocol family.
- `/v1/models` or equivalent model discovery succeeds through every configured agent/client scope.
- Direct protocol live test performs at least two turns: model calls a tool, host returns a tool result, model returns final text.
- At least one cross-protocol live test exercises source protocol conversion into the provider's target protocol.
- Streaming test observes text deltas and tool/reasoning deltas where the provider supports them.
- Provider quirks are captured in a dedicated adapter or in profile/catalog metadata, not in host route code.
- Documentation lists unsupported fields and observed provider/model failures.
