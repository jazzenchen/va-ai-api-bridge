use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct ZaiBridgeAdapter {
    thinking_enabled: bool,
}

impl ZaiBridgeAdapter {
    pub fn new(thinking_enabled: Option<bool>) -> Self {
        let thinking_enabled = thinking_enabled.unwrap_or(true);
        Self { thinking_enabled }
    }

    pub fn prepare_chat_request(&mut self, original_request: &Value, chat_request: &mut Value) {
        let Some(object) = chat_request.as_object_mut() else {
            return;
        };

        object.remove("reasoning");
        object.remove("reasoning_effort");
        object.remove("reasoningEffort");

        let thinking_enabled =
            thinking_from_original_request(original_request).unwrap_or(self.thinking_enabled);
        if !thinking_enabled {
            object.insert("thinking".to_string(), json!({ "type": "disabled" }));
        }
    }
}

fn thinking_from_original_request(request: &Value) -> Option<bool> {
    request
        .get("reasoning")
        .and_then(|reasoning| {
            reasoning
                .get("effort")
                .or_else(|| reasoning.get("reasoning_effort"))
                .or_else(|| reasoning.get("reasoningEffort"))
                .and_then(Value::as_str)
        })
        .or_else(|| request.get("reasoning_effort").and_then(Value::as_str))
        .or_else(|| request.get("reasoningEffort").and_then(Value::as_str))
        .map(reasoning_effort_enabled)
}

fn reasoning_effort_enabled(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "off" | "none" | "disabled" | "disable" | "false"
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn disables_zai_thinking_when_reasoning_is_off() {
        let mut adapter = ZaiBridgeAdapter::new(None);
        let mut chat_request = json!({ "model": "glm-5.1", "messages": [] });

        adapter.prepare_chat_request(
            &json!({ "reasoning": { "effort": "none" } }),
            &mut chat_request,
        );

        assert_eq!(chat_request["thinking"], json!({ "type": "disabled" }));
    }

    #[test]
    fn leaves_zai_default_thinking_unpatched_when_enabled() {
        let mut adapter = ZaiBridgeAdapter::new(None);
        let mut chat_request = json!({ "model": "glm-5.1", "messages": [] });

        adapter.prepare_chat_request(
            &json!({ "reasoning": { "effort": "high" } }),
            &mut chat_request,
        );

        assert!(chat_request.get("thinking").is_none());
    }
}
