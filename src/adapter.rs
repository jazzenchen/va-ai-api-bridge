use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{DecodeState, Extensions, Result, UniversalEvent, UniversalItem, WireProtocol};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BridgeContext {
    pub profile_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launch_id: Option<String>,
    pub source_protocol: WireProtocol,
    pub target_protocol: WireProtocol,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_settings: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history: Option<BridgeHistory>,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BridgeHistory {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<UniversalItem>,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterStreamState {
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
    #[serde(default)]
    pub translator_state: DecodeState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AdapterStreamStep {
    UseTargetTranslator,
    Events(Vec<UniversalEvent>),
}

pub trait ProviderAdapter {
    fn prepare_request(&mut self, _ctx: &BridgeContext, request: Value) -> Result<Value> {
        Ok(request)
    }

    fn map_response(
        &mut self,
        _ctx: &BridgeContext,
        _response: &Value,
    ) -> Result<Option<Vec<UniversalEvent>>> {
        Ok(None)
    }

    fn map_stream_chunk(
        &mut self,
        _ctx: &BridgeContext,
        _chunk: &Value,
        _state: &mut AdapterStreamState,
    ) -> Result<AdapterStreamStep> {
        Ok(AdapterStreamStep::UseTargetTranslator)
    }
}
