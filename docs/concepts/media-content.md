# Media Content

Media content is represented explicitly in the IR so hosts can make safe model-capability decisions before sending a request upstream.

## IR Representation

Images use `ContentBlock::Image` with optional `media_type`, `url`, `data`, and `extensions` fields. Files use `ContentBlock::File` with optional `media_type`, `filename`, `url`, `data`, and `extensions` fields.

Translators map common wire shapes into these blocks:

| Source shape | IR block |
| --- | --- |
| OpenAI Chat `image_url` content part | `ContentBlock::Image` |
| OpenAI Responses `input_image` item/part | `ContentBlock::Image` |
| Anthropic `image` content block | `ContentBlock::Image` |
| Anthropic `document` content block | `ContentBlock::File` |
| Gemini inline/file data parts | `ContentBlock::Image` or `ContentBlock::File` depending on MIME type |

## Capability Policy Belongs to the Host

The crate does not know which profile/model the host will select at runtime, so it does not reject or drop media on its own. A bridge host should compare the final target model's capabilities with the IR before encoding the upstream request.

When a target model does not support image or file input, the safe behavior is to replace the unsupported block with a text placeholder that says the attachment was omitted. The placeholder must not claim to understand the attachment contents.

Recommended image placeholder:

```text
[Image attachment omitted: <provider> <model> does not support image input. Do not infer image contents; ask the user to describe it or switch models.]
```

Recommended file placeholder:

```text
[File attachment omitted: <provider> <model> does not support file input. Do not infer file contents; ask the user to paste relevant text or switch models.]
```

This policy prevents a conversation from becoming unrecoverable when an agent keeps unsupported media in future request history. The next request can still be translated because the prior media block has become ordinary text.

## Ordering in a Host Bridge

Apply media capability policy after source decode and model mapping, but before target encode:

```text
source wire request
  -> decode to UniversalRequest
  -> choose/map target model
  -> replace unsupported Image/File blocks with text placeholders
  -> encode target wire request
  -> apply provider adapter
  -> send upstream
```

For same-protocol passthrough routes, a host can still temporarily decode to IR, sanitize unsupported media, and re-encode only when the policy changed the request.

## Safety Rules

- Do not forward `Image` or `File` blocks to a model that the host catalog marks as text-only.
- Do not summarize, OCR, caption, or infer omitted media in the placeholder.
- Preserve surrounding user text so the model can ask for clarification naturally.
- Leave supported media untouched for models whose catalog declares image/file support.
- Treat unknown raw payloads that contain recognizable media keys as media for capability policy.
