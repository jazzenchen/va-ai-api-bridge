# Provider Catalog Schema

The `schema::catalog` module defines serializable metadata types that hosts can use to describe providers, endpoints, models, settings, defaults, and capabilities.

## Main Types

| Type | Purpose |
| --- | --- |
| `ProviderCatalog` | Root document for one provider. |
| `ProviderProtocol` | Supported source/target protocol pairing and endpoint defaults. |
| `ProviderModel` | Model ID, aliases, protocol support, defaults, and capabilities. |
| `ModelCapabilities` | Streaming, tools, vision, files, reasoning, and modality metadata. |
| `ProviderSetting` | User-configurable setting metadata for host UIs. |
| `ProviderDefaults` | Default model, generation config, reasoning config, raw request extensions. |

## Usage

A host can load catalog data from JSON, merge user profile overrides, and use the result to choose models and enforce capability policy before target encoding.

The crate defines the schema but does not ship a canonical provider catalog. This keeps provider availability and product-specific profile defaults in the host application.
