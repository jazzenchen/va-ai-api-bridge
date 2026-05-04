mod decode;
mod encode;

use serde_json::Value;

use crate::{DecodeState, EncodeState, Result, UniversalEvent, WireEvent};

pub(super) fn decode_chunk(raw: Value, state: &mut DecodeState) -> Result<Vec<UniversalEvent>> {
    decode::decode_chunk(raw, state)
}

pub(super) fn encode(events: &[UniversalEvent], state: &mut EncodeState) -> Result<Vec<WireEvent>> {
    encode::encode(events, state)
}
