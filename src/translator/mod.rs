pub mod anthropic_messages;
mod common;
pub mod openai_chat;
pub mod openai_responses;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{DecodeState, EncodeState, Result, UniversalEvent, UniversalRequest, WireProtocol};

pub use anthropic_messages::AnthropicMessagesTranslator;
pub use openai_chat::OpenAiChatTranslator;
pub use openai_responses::OpenAiResponsesTranslator;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WireEvent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    pub data: Value,
}

pub trait WireTranslator {
    fn protocol(&self) -> WireProtocol;

    fn decode_request(&self, raw: Value) -> Result<UniversalRequest>;

    fn encode_request(&self, request: &UniversalRequest) -> Result<Value>;

    fn decode_response(&self, raw: Value) -> Result<Vec<UniversalEvent>>;

    fn decode_stream_chunk(
        &self,
        raw: Value,
        state: &mut DecodeState,
    ) -> Result<Vec<UniversalEvent>>;

    fn encode_events(
        &self,
        events: &[UniversalEvent],
        state: &mut EncodeState,
    ) -> Result<Vec<WireEvent>>;
}
