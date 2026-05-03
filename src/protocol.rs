use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{ApiProxyError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WireProtocol {
    #[serde(rename = "openai-responses")]
    OpenAiResponses,
    #[serde(rename = "openai-chat")]
    OpenAiChat,
    #[serde(rename = "anthropic-messages")]
    AnthropicMessages,
}

impl WireProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiResponses => "openai-responses",
            Self::OpenAiChat => "openai-chat",
            Self::AnthropicMessages => "anthropic-messages",
        }
    }
}

impl fmt::Display for WireProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for WireProtocol {
    type Err = ApiProxyError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "openai-responses" => Ok(Self::OpenAiResponses),
            "openai-chat" => Ok(Self::OpenAiChat),
            "anthropic-messages" => Ok(Self::AnthropicMessages),
            other => Err(ApiProxyError::unsupported_protocol(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_roundtrips_as_manifest_token() {
        let encoded = serde_json::to_string(&WireProtocol::OpenAiResponses).unwrap();
        assert_eq!(encoded, "\"openai-responses\"");
        let decoded: WireProtocol = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, WireProtocol::OpenAiResponses);
        assert_eq!(
            "openai-chat".parse::<WireProtocol>().unwrap(),
            WireProtocol::OpenAiChat
        );
    }
}
