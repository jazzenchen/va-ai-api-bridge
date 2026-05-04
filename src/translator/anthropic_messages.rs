mod request;
mod response;
mod stream;

use serde_json::Value;

use crate::{
    DecodeState, EncodeState, Result, UniversalEvent, UniversalRequest, WireEvent, WireProtocol,
};

use super::WireTranslator;

#[derive(Debug, Clone, Copy, Default)]
pub struct AnthropicMessagesTranslator;

impl WireTranslator for AnthropicMessagesTranslator {
    fn protocol(&self) -> WireProtocol {
        WireProtocol::AnthropicMessages
    }

    fn decode_request(&self, raw: Value) -> Result<UniversalRequest> {
        request::decode(raw)
    }

    fn encode_request(&self, request: &UniversalRequest) -> Result<Value> {
        request::encode(request)
    }

    fn decode_response(&self, raw: Value) -> Result<Vec<UniversalEvent>> {
        response::decode(raw)
    }

    fn decode_stream_chunk(
        &self,
        raw: Value,
        state: &mut DecodeState,
    ) -> Result<Vec<UniversalEvent>> {
        stream::decode_chunk(raw, state)
    }

    fn encode_events(
        &self,
        events: &[UniversalEvent],
        state: &mut EncodeState,
    ) -> Result<Vec<WireEvent>> {
        stream::encode(events, state)
    }
}
