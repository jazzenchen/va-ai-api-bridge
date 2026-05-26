mod dashscope;
mod deepseek;
mod kimi;
mod mimo;
mod minimax;
mod reasoning_blob;
mod xai;
mod zai;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{UniversalEvent, WireProtocol};

pub use dashscope::DashScopeBridgeAdapter;
pub use deepseek::DeepSeekBridgeAdapter;
pub use kimi::KimiBridgeAdapter;
pub use mimo::MimoBridgeAdapter;
pub use minimax::MiniMaxBridgeAdapter;
pub use xai::XaiBridgeAdapter;
pub use zai::ZaiBridgeAdapter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderRequestSource {
    OpenAiResponses,
    OpenAiChat,
    AnthropicMessages,
    GeminiGenerateContent,
}

impl ProviderRequestSource {
    pub fn from_protocol(protocol: WireProtocol) -> Self {
        match protocol {
            WireProtocol::OpenAiResponses => Self::OpenAiResponses,
            WireProtocol::OpenAiChat => Self::OpenAiChat,
            WireProtocol::AnthropicMessages => Self::AnthropicMessages,
            WireProtocol::GeminiGenerateContent => Self::GeminiGenerateContent,
        }
    }

    pub(crate) fn supports_deepseek_reasoning_replay(self) -> bool {
        matches!(
            self,
            Self::OpenAiResponses | Self::AnthropicMessages | Self::GeminiGenerateContent
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeepSeekBridgeSettings {
    #[serde(default, skip_serializing_if = "is_false")]
    pub thinking: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub replay_reasoning_content: bool,
}

impl DeepSeekBridgeSettings {
    pub fn is_empty(&self) -> bool {
        !self.thinking && !self.replay_reasoning_content
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderBridgeAdapterConfig {
    #[serde(default, skip_serializing_if = "DeepSeekBridgeSettings::is_empty")]
    pub deepseek: DeepSeekBridgeSettings,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_enabled: Option<bool>,
}

#[derive(Debug, Clone)]
pub enum ProviderBridgeAdapter {
    None,
    DeepSeek(DeepSeekBridgeAdapter),
    Kimi(KimiBridgeAdapter),
    Mimo(MimoBridgeAdapter),
    MiniMax(MiniMaxBridgeAdapter),
    DashScope(DashScopeBridgeAdapter),
    Xai(XaiBridgeAdapter),
    Zai(ZaiBridgeAdapter),
}

impl ProviderBridgeAdapter {
    pub fn for_provider(
        provider_id: &str,
        _target_protocol: WireProtocol,
        config: ProviderBridgeAdapterConfig,
    ) -> Self {
        match provider_id {
            "deepseek" => Self::DeepSeek(DeepSeekBridgeAdapter::new(config.deepseek)),
            "kimi" => Self::Kimi(KimiBridgeAdapter::default()),
            "mimo" => Self::Mimo(MimoBridgeAdapter),
            "minimax" => Self::MiniMax(MiniMaxBridgeAdapter::default()),
            "dashscope" | "qwen" => {
                Self::DashScope(DashScopeBridgeAdapter::new(config.thinking_enabled))
            }
            "xai" => Self::Xai(XaiBridgeAdapter),
            "zai" => Self::Zai(ZaiBridgeAdapter::new(config.thinking_enabled)),
            _ => Self::None,
        }
    }

    pub fn prepare_responses_request(&mut self, request: &mut Value) {
        match self {
            Self::None => {}
            Self::DeepSeek(_) => {}
            Self::Kimi(_) => {}
            Self::Mimo(_) => {}
            Self::MiniMax(_) => {}
            Self::DashScope(_) => {}
            Self::Xai(adapter) => adapter.prepare_responses_request(request),
            Self::Zai(_) => {}
        }
    }

    pub fn prepare_chat_request(
        &mut self,
        source: ProviderRequestSource,
        original_request: &Value,
        chat_request: &mut Value,
    ) {
        match self {
            Self::None => {}
            Self::DeepSeek(adapter) => {
                adapter.prepare_chat_request(source, original_request, chat_request)
            }
            Self::Kimi(_) => {}
            Self::Mimo(adapter) => {
                adapter.prepare_chat_request(source, original_request, chat_request)
            }
            Self::MiniMax(adapter) => adapter.prepare_chat_request(chat_request),
            Self::DashScope(adapter) => {
                adapter.prepare_chat_request(original_request, chat_request)
            }
            Self::Xai(_) => {}
            Self::Zai(adapter) => adapter.prepare_chat_request(original_request, chat_request),
        }
    }

    pub fn prepare_anthropic_request(&mut self, request: &mut Value) {
        match self {
            Self::None => {}
            Self::DeepSeek(_) => {}
            Self::Kimi(adapter) => adapter.prepare_anthropic_request(request),
            Self::Mimo(_) => {}
            Self::MiniMax(_) => {}
            Self::DashScope(_) => {}
            Self::Xai(_) => {}
            Self::Zai(_) => {}
        }
    }

    pub fn normalize_chat_response(&mut self, response: &mut Value) {
        match self {
            Self::None => {}
            Self::DeepSeek(_) => {}
            Self::Kimi(_) => {}
            Self::Mimo(adapter) => adapter.normalize_chat_response(response),
            Self::MiniMax(_) => {}
            Self::DashScope(_) => {}
            Self::Xai(_) => {}
            Self::Zai(_) => {}
        }
    }

    pub fn transform_upstream_events(&mut self, events: &mut Vec<UniversalEvent>) {
        match self {
            Self::None => {}
            Self::DeepSeek(_) => {}
            Self::Kimi(adapter) => adapter.transform_upstream_events(events),
            Self::Mimo(_) => {}
            Self::MiniMax(adapter) => adapter.transform_upstream_events(events),
            Self::DashScope(_) => {}
            Self::Xai(_) => {}
            Self::Zai(_) => {}
        }
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}
