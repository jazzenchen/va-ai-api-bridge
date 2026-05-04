mod events;

pub(crate) use events::*;

use serde_json::Value;

use crate::{Extensions, Role, SourcePayload, WireProtocol};

pub(crate) fn empty_extensions() -> Extensions {
    Extensions::new()
}

pub(crate) fn source(protocol: WireProtocol, raw: Value) -> SourcePayload {
    SourcePayload {
        protocol,
        raw: Some(raw),
    }
}

pub(crate) fn role_from_wire(role: &str) -> Option<Role> {
    match role {
        "developer" | "system" => Some(Role::System),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        "tool" => Some(Role::Tool),
        _ => None,
    }
}

pub(crate) fn value_extensions(extra: impl IntoIterator<Item = (String, Value)>) -> Extensions {
    extra.into_iter().collect()
}

pub(crate) fn parse_arguments(arguments: Option<&str>) -> Value {
    arguments
        .filter(|arguments| !arguments.trim().is_empty())
        .and_then(|arguments| serde_json::from_str(arguments).ok())
        .unwrap_or(Value::Null)
}

pub(crate) fn stringify_arguments(arguments: &Value) -> String {
    match arguments {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        value => serde_json::to_string(value).unwrap_or_default(),
    }
}
