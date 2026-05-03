use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ContentBlock, Extensions, FinishReason, Role, Usage};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecodeState {
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EncodeState {
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum UniversalEvent {
    ResponseStart {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    MessageStart {
        id: String,
        role: Role,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    ContentStart {
        index: usize,
        block: ContentBlock,
    },
    TextDelta {
        index: usize,
        text: String,
    },
    ReasoningDelta {
        index: usize,
        text: String,
    },
    ToolCallDelta {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        arguments_delta: String,
    },
    ContentDone {
        index: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        final_block: Option<ContentBlock>,
    },
    MessageDone {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        finish_reason: Option<FinishReason>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    ResponseDone {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        #[serde(default, skip_serializing_if = "Extensions::is_empty")]
        extensions: Extensions,
    },
    Error {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw: Option<Value>,
    },
    Unknown {
        raw: Value,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        tags: BTreeMap<String, String>,
    },
}
