use std::fmt;

pub type Result<T> = std::result::Result<T, ApiProxyError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiProxyError {
    UnsupportedProtocol { protocol: String },
    InvalidRequest { message: String },
    InvalidResponse { message: String },
    Conversion { message: String },
    Adapter { message: String },
}

impl ApiProxyError {
    pub fn unsupported_protocol(protocol: impl Into<String>) -> Self {
        Self::UnsupportedProtocol {
            protocol: protocol.into(),
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    pub fn invalid_response(message: impl Into<String>) -> Self {
        Self::InvalidResponse {
            message: message.into(),
        }
    }

    pub fn conversion(message: impl Into<String>) -> Self {
        Self::Conversion {
            message: message.into(),
        }
    }

    pub fn adapter(message: impl Into<String>) -> Self {
        Self::Adapter {
            message: message.into(),
        }
    }
}

impl fmt::Display for ApiProxyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedProtocol { protocol } => {
                write!(f, "unsupported protocol: {protocol}")
            }
            Self::InvalidRequest { message } => write!(f, "invalid request: {message}"),
            Self::InvalidResponse { message } => write!(f, "invalid response: {message}"),
            Self::Conversion { message } => write!(f, "conversion failed: {message}"),
            Self::Adapter { message } => write!(f, "adapter failed: {message}"),
        }
    }
}

impl std::error::Error for ApiProxyError {}
