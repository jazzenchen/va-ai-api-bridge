use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{DecodeState, Extensions, Result, UniversalEvent, UniversalItem, WireProtocol};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProxyContext {
    pub profile_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launch_id: Option<String>,
    pub source_protocol: WireProtocol,
    pub target_protocol: WireProtocol,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_settings: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history: Option<ProxyHistory>,
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProxyHistory {
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
    pub codec_state: DecodeState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AdapterStreamStep {
    UseTargetCodec,
    Events(Vec<UniversalEvent>),
}

pub trait ProviderAdapter {
    fn prepare_request(&mut self, _ctx: &ProxyContext, request: Value) -> Result<Value> {
        Ok(request)
    }

    fn map_response(
        &mut self,
        _ctx: &ProxyContext,
        _response: &Value,
    ) -> Result<Option<Vec<UniversalEvent>>> {
        Ok(None)
    }

    fn map_stream_chunk(
        &mut self,
        _ctx: &ProxyContext,
        _chunk: &Value,
        _state: &mut AdapterStreamState,
    ) -> Result<AdapterStreamStep> {
        Ok(AdapterStreamStep::UseTargetCodec)
    }
}
