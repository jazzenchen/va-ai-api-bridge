# Documentation

This documentation explains what the crate is, how the translation model works, and how to embed it in a host bridge. It avoids test reports and project planning notes; validation details belong in PRs, CI, and release notes.

## Start Here

- [Getting started](guides/getting-started.md) for a small request translation flow.
- [Host integration](guides/host-integration.md) for the full bridge lifecycle.
- [Architecture](concepts/architecture.md) for the crate layers and boundaries.

## Concepts

- [Architecture](concepts/architecture.md): data flow and module ownership.
- [Universal IR](concepts/universal-ir.md): the protocol-neutral model shared by translators.
- [Media content](concepts/media-content.md): image/file blocks, safety rules, and capability handling.
- [Streaming](concepts/streaming.md): `UniversalEvent`, decode state, encode state, and SSE framing responsibilities.

## Guides

- [Getting started](guides/getting-started.md): translating one request and choosing translators dynamically.
- [Host integration](guides/host-integration.md): where to place routing, model capability checks, provider adapters, and transport.
- [Provider adapters](guides/provider-adapters.md): applying provider-specific request and event transforms.

## Reference

- [Protocols](reference/protocols.md): supported wire protocols and official references.
- [Provider notes](reference/provider-notes.md): documented provider interfaces and adapter responsibilities.
- [Provider catalog schema](reference/provider-catalog.md): host-facing provider/model metadata schema.

## Examples

The `examples/` directory contains minimal runnable programs that mirror the guide flows. Use them as integration sketches, not as a full server implementation.
