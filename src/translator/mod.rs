mod anthropic;
pub mod anthropic_messages;
mod common;
pub mod gemini_generate_content;
mod openai;
pub mod openai_chat;
pub mod openai_responses;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{DecodeState, EncodeState, Result, UniversalEvent, UniversalRequest, WireProtocol};

pub use anthropic_messages::AnthropicMessagesTranslator;
pub use gemini_generate_content::GeminiGenerateContentTranslator;
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

pub fn translator_for_protocol(protocol: WireProtocol) -> Box<dyn WireTranslator> {
    match protocol {
        WireProtocol::OpenAiResponses => Box::new(OpenAiResponsesTranslator),
        WireProtocol::OpenAiChat => Box::new(OpenAiChatTranslator),
        WireProtocol::AnthropicMessages => Box::new(AnthropicMessagesTranslator),
        WireProtocol::GeminiGenerateContent => Box::new(GeminiGenerateContentTranslator),
    }
}

#[cfg(test)]
mod tests {
    use crate::{translator::translator_for_protocol, WireProtocol};

    #[test]
    fn creates_translator_for_each_wire_protocol() {
        for protocol in [
            WireProtocol::OpenAiResponses,
            WireProtocol::OpenAiChat,
            WireProtocol::AnthropicMessages,
            WireProtocol::GeminiGenerateContent,
        ] {
            assert_eq!(translator_for_protocol(protocol).protocol(), protocol);
        }
    }
}
