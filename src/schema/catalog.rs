use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Extensions, GenerationConfig, ReasoningConfig, WireProtocol};

pub const PROVIDER_CATALOG_SCHEMA_VERSION: &str = "va.ai.api.proxy.catalog.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderCatalog {
    #[serde(default = "default_catalog_schema_version")]
    pub schema_version: String,
    pub provider_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protocols: Vec<ProviderProtocol>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<ProviderModel>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub settings: Vec<ProviderSetting>,
    #[serde(default, skip_serializing_if = "ProviderDefaults::is_empty")]
    pub defaults: ProviderDefaults,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

impl ProviderCatalog {
    pub fn new(provider_id: impl Into<String>) -> Self {
        Self {
            schema_version: default_catalog_schema_version(),
            provider_id: provider_id.into(),
            display_name: None,
            version: None,
            protocols: Vec::new(),
            models: Vec::new(),
            settings: Vec::new(),
            defaults: ProviderDefaults::default(),
            extensions: Extensions::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderProtocol {
    pub source_protocol: WireProtocol,
    pub target_protocol: WireProtocol,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub streaming: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub tools: bool,
    #[serde(default, skip_serializing_if = "ProviderDefaults::is_empty")]
    pub defaults: ProviderDefaults,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderModel {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protocols: Vec<WireProtocol>,
    #[serde(default, skip_serializing_if = "ProviderDefaults::is_empty")]
    pub defaults: ProviderDefaults,
    #[serde(default, skip_serializing_if = "ModelCapabilities::is_empty")]
    pub capabilities: ModelCapabilities,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "GenerationConfig::is_empty")]
    pub generation: GenerationConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_request: Option<Value>,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

impl ProviderDefaults {
    pub fn is_empty(&self) -> bool {
        self.model.is_none()
            && self.generation.is_empty()
            && self.reasoning.is_none()
            && self.raw_request.is_none()
            && self.extensions.is_empty()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCapabilities {
    #[serde(default, skip_serializing_if = "is_false")]
    pub streaming: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub tools: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub vision: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub files: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub reasoning: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_modalities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_modalities: Vec<String>,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

impl ModelCapabilities {
    pub fn is_empty(&self) -> bool {
        !self.streaming
            && !self.tools
            && !self.vision
            && !self.files
            && !self.reasoning
            && self.input_modalities.is_empty()
            && self.output_modalities.is_empty()
            && self.extensions.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderSetting {
    pub key: String,
    pub kind: SettingKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub required: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub secret: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<SettingOption>,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingKind {
    String,
    Number,
    Integer,
    Boolean,
    Secret,
    Select,
    Object,
    Array,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingOption {
    pub value: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

fn default_catalog_schema_version() -> String {
    PROVIDER_CATALOG_SCHEMA_VERSION.to_string()
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn catalog_serializes_version_and_defaults() {
        let mut catalog = ProviderCatalog::new("deepseek");
        catalog.display_name = Some("DeepSeek".to_string());
        catalog.protocols.push(ProviderProtocol {
            source_protocol: WireProtocol::OpenAiChat,
            target_protocol: WireProtocol::OpenAiChat,
            upstream_path: Some("/chat/completions".to_string()),
            default_model: Some("deepseek-chat".to_string()),
            streaming: true,
            tools: true,
            defaults: ProviderDefaults {
                raw_request: Some(json!({ "frequency_penalty": 0 })),
                ..ProviderDefaults::default()
            },
            extensions: Extensions::new(),
        });

        let encoded = serde_json::to_value(catalog).unwrap();

        assert_eq!(encoded["schemaVersion"], PROVIDER_CATALOG_SCHEMA_VERSION);
        assert_eq!(encoded["providerId"], "deepseek");
        assert_eq!(encoded["protocols"][0]["targetProtocol"], "openai-chat");
        assert_eq!(
            encoded["protocols"][0]["defaults"]["rawRequest"]["frequency_penalty"],
            0
        );
    }
}
