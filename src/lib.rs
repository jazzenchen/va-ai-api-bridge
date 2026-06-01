#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

pub mod adapter;
pub mod error;
pub mod media;
pub mod protocol;
pub mod providers;
pub mod schema;
pub mod stream;
pub mod translator;
pub mod universal;

pub use adapter::{
    AdapterStreamState, AdapterStreamStep, BridgeContext, BridgeHistory, ProviderAdapter,
};
pub use error::{ApiBridgeError, Result};
pub use media::{
    sanitize_unsupported_media, sanitize_unsupported_media_from_json, MediaSanitization,
};
pub use protocol::WireProtocol;
pub use providers::{
    DashScopeBridgeAdapter, DeepSeekBridgeAdapter, DeepSeekBridgeSettings, KimiBridgeAdapter,
    MimoBridgeAdapter, MiniMaxBridgeAdapter, ProviderBridgeAdapter, ProviderBridgeAdapterConfig,
    ProviderRequestSource, XaiBridgeAdapter, ZaiBridgeAdapter,
};
pub use schema::{
    ModelCapabilities, ProviderCatalog, ProviderDefaults, ProviderModel, ProviderProtocol,
    ProviderSetting, ResolvedModelSpec, SettingKind, SettingOption,
    PROVIDER_CATALOG_SCHEMA_VERSION,
};
pub use stream::{DecodeState, EncodeState, UniversalEvent};
pub use translator::{
    translator_for_protocol, AnthropicMessagesTranslator, GeminiGenerateContentTranslator,
    OpenAiChatTranslator, OpenAiResponsesTranslator, WireEvent, WireTranslator,
};
pub use universal::{
    ContentBlock, Extensions, FinishReason, GenerationConfig, ReasoningConfig, Role, SourcePayload,
    ToolChoice, UniversalItem, UniversalRequest, UniversalResponse, UniversalTool, Usage,
};
