use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ApiBridgeError, Extensions, GenerationConfig, ReasoningConfig, Result, WireProtocol};

pub const PROVIDER_CATALOG_SCHEMA_VERSION: &str = "va.ai.api.bridge.catalog.v1";

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
pub struct ResolvedModelSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_label: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    #[serde(default, skip_serializing_if = "ModelCapabilities::is_empty")]
    pub capabilities: ModelCapabilities,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

impl ResolvedModelSpec {
    pub fn from_json(value: Value) -> Result<Self> {
        serde_json::from_value(value).map_err(|error| {
            ApiBridgeError::invalid_request(format!("invalid resolved model spec: {error}"))
        })
    }

    pub fn provider_label(&self) -> &str {
        self.provider_label
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("Target provider")
    }

    pub fn model_label(&self) -> &str {
        self.model
            .trim()
            .is_empty()
            .then_some("selected model")
            .unwrap_or_else(|| self.model.trim())
    }
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

    pub fn supports_image_input(&self) -> bool {
        self.vision || self.has_input_modality(&["image", "images", "vision"])
    }

    pub fn supports_file_input(&self) -> bool {
        self.files || self.has_input_modality(&["file", "files", "document", "documents"])
    }

    pub fn has_input_modality(&self, aliases: &[&str]) -> bool {
        self.input_modalities
            .iter()
            .any(|modality| modality_matches(modality, aliases))
    }

    pub fn union(&self, other: &Self) -> Self {
        let mut merged = Self {
            streaming: self.streaming || other.streaming,
            tools: self.tools || other.tools,
            vision: self.vision || other.vision,
            files: self.files || other.files,
            reasoning: self.reasoning || other.reasoning,
            input_modalities: union_strings(&self.input_modalities, &other.input_modalities),
            output_modalities: union_strings(&self.output_modalities, &other.output_modalities),
            extensions: self.extensions.clone(),
        };
        merged.extensions.extend(other.extensions.clone());
        merged
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

fn modality_matches(modality: &str, aliases: &[&str]) -> bool {
    let modality = normalized_modality(modality);
    aliases
        .iter()
        .any(|alias| modality == normalized_modality(alias))
}

fn normalized_modality(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['_', ' '], "-")
}

fn union_strings(left: &[String], right: &[String]) -> Vec<String> {
    let mut out = left.to_vec();
    for value in right {
        if !out.iter().any(|existing| existing == value) {
            out.push(value.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ModelCapabilities;

    #[test]
    fn image_support_can_come_from_boolean_or_input_modality() {
        let boolean = ModelCapabilities {
            vision: true,
            ..ModelCapabilities::default()
        };
        let modality = ModelCapabilities {
            input_modalities: vec!["text".to_string(), "image".to_string()],
            ..ModelCapabilities::default()
        };

        assert!(boolean.supports_image_input());
        assert!(modality.supports_image_input());
        assert!(!ModelCapabilities::default().supports_image_input());
    }

    #[test]
    fn file_support_can_come_from_boolean_or_input_modality() {
        let boolean = ModelCapabilities {
            files: true,
            ..ModelCapabilities::default()
        };
        let modality = ModelCapabilities {
            input_modalities: vec!["text".to_string(), "document".to_string()],
            ..ModelCapabilities::default()
        };

        assert!(boolean.supports_file_input());
        assert!(modality.supports_file_input());
        assert!(!ModelCapabilities::default().supports_file_input());
    }

    #[test]
    fn union_merges_capability_flags_modalities_and_extensions() {
        let base = ModelCapabilities {
            streaming: true,
            input_modalities: vec!["text".to_string()],
            extensions: [("base".to_string(), json!(true))].into_iter().collect(),
            ..ModelCapabilities::default()
        };
        let model = ModelCapabilities {
            tools: true,
            vision: true,
            input_modalities: vec!["text".to_string(), "image".to_string()],
            extensions: [("model".to_string(), json!(true))].into_iter().collect(),
            ..ModelCapabilities::default()
        };

        let merged = base.union(&model);

        assert!(merged.streaming);
        assert!(merged.tools);
        assert!(merged.supports_image_input());
        assert_eq!(merged.input_modalities, vec!["text", "image"]);
        assert_eq!(merged.extensions["base"], json!(true));
        assert_eq!(merged.extensions["model"], json!(true));
    }

    #[test]
    fn resolved_model_spec_deserializes_from_json() {
        let spec = super::ResolvedModelSpec::from_json(json!({
            "providerLabel": "DeepSeek",
            "model": "deepseek-v4-pro",
            "capabilities": {
                "inputModalities": ["text"]
            }
        }))
        .expect("model spec deserializes");

        assert_eq!(spec.provider_label(), "DeepSeek");
        assert_eq!(spec.model_label(), "deepseek-v4-pro");
        assert!(!spec.capabilities.supports_image_input());
    }
}
