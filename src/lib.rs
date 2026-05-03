#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

pub mod adapter;
pub mod codec;
pub mod error;
pub mod protocol;
pub mod stream;
pub mod universal;

pub use adapter::{
    AdapterStreamState, AdapterStreamStep, ProviderAdapter, ProxyContext, ProxyHistory,
};
pub use codec::{WireCodec, WireEvent};
pub use error::{ApiProxyError, Result};
pub use protocol::WireProtocol;
pub use stream::{DecodeState, EncodeState, UniversalEvent};
pub use universal::{
    ContentBlock, Extensions, FinishReason, GenerationConfig, ReasoningConfig, Role, SourcePayload,
    ToolChoice, UniversalItem, UniversalRequest, UniversalTool, Usage,
};
