# Provider Catalog Schema

The `schema::catalog` module defines serializable metadata types that hosts can use to describe providers, endpoints, models, settings, defaults, and capabilities.

## Main Types

| Type | Purpose |
| --- | --- |
| `ProviderCatalog` | Root document for one provider. |
| `ProviderProtocol` | Supported source/target protocol pairing and endpoint defaults. |
| `ProviderModel` | Model ID, aliases, protocol support, defaults, and capabilities. |
| `ModelCapabilities` | Streaming, tools, vision, files, reasoning, and modality metadata. |
| `ResolvedModelSpec` | Final target model chosen by the host, with merged capabilities ready for request policy. |
| `ProviderSetting` | User-configurable setting metadata for host UIs. |
| `ProviderDefaults` | Default model, generation config, reasoning config, raw request extensions. |

## Usage

A host can load catalog data from JSON, merge user profile overrides, and use the result to choose models and enforce capability policy before target encoding.

The crate defines the schema but does not ship a canonical provider catalog. This keeps provider availability and product-specific profile defaults in the host application.

For request-time policy, pass the final merged model into `ResolvedModelSpec`:

```json
{
  "providerLabel": "DeepSeek",
  "model": "deepseek-v4-pro",
  "capabilities": {
    "inputModalities": ["text"]
  }
}
```

Hosts can pass this shape as JSON with `sanitize_unsupported_media_from_json`, or deserialize it into `ResolvedModelSpec` and call `sanitize_unsupported_media`.
