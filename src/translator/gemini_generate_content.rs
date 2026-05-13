mod request;
mod response;
mod shared;
mod stream;

#[cfg(test)]
mod tests;

use serde_json::Value;

use crate::translator::{WireEvent, WireTranslator};
use crate::{DecodeState, EncodeState, Result, UniversalEvent, UniversalRequest, WireProtocol};

pub use response::encode_response;
pub use shared::{attach_route_metadata, strip_route_metadata};

pub struct GeminiGenerateContentTranslator;

impl WireTranslator for GeminiGenerateContentTranslator {
    fn protocol(&self) -> WireProtocol {
        WireProtocol::GeminiGenerateContent
    }

    fn decode_request(&self, raw: Value) -> Result<UniversalRequest> {
        request::decode_request(raw)
    }

    fn encode_request(&self, request: &UniversalRequest) -> Result<Value> {
        request::encode_request(request)
    }

    fn decode_response(&self, raw: Value) -> Result<Vec<UniversalEvent>> {
        response::decode_response(raw)
    }

    fn decode_stream_chunk(
        &self,
        raw: Value,
        state: &mut DecodeState,
    ) -> Result<Vec<UniversalEvent>> {
        stream::decode_stream_chunk(raw, state)
    }

    fn encode_events(
        &self,
        events: &[UniversalEvent],
        state: &mut EncodeState,
    ) -> Result<Vec<WireEvent>> {
        stream::encode_stream_events(events, state)
    }
}
