use serde_json::Value;

#[derive(Debug, Clone, Default)]
pub struct XaiBridgeAdapter;

impl XaiBridgeAdapter {
    pub fn prepare_responses_request(&mut self, request: &mut Value) {
        strip_unsupported_request_fields(request);
        strip_encrypted_reasoning_history(request);
        strip_unsupported_tools(request);
    }
}

fn strip_unsupported_request_fields(request: &mut Value) {
    match request {
        Value::Object(object) => {
            object.remove("external_web_access");
            for value in object.values_mut() {
                strip_unsupported_request_fields(value);
            }
        }
        Value::Array(items) => {
            for item in items {
                strip_unsupported_request_fields(item);
            }
        }
        _ => {}
    }
}

fn strip_unsupported_tools(request: &mut Value) {
    let Some(tools) = request.get_mut("tools").and_then(Value::as_array_mut) else {
        return;
    };

    tools.retain(|tool| {
        matches!(
            tool.get("type").and_then(Value::as_str),
            Some(
                "function"
                    | "web_search"
                    | "x_search"
                    | "collections_search"
                    | "file_search"
                    | "code_execution"
                    | "code_interpreter"
                    | "mcp"
                    | "shell"
            )
        )
    });
}

fn strip_encrypted_reasoning_history(request: &mut Value) {
    let Some(object) = request.as_object_mut() else {
        return;
    };

    if let Some(input) = object.get_mut("input").and_then(Value::as_array_mut) {
        input.retain(|item| item.get("type").and_then(Value::as_str) != Some("reasoning"));
    }

    if let Some(include) = object.get_mut("include").and_then(Value::as_array_mut) {
        include.retain(|item| item.as_str() != Some("reasoning.encrypted_content"));
        if include.is_empty() {
            object.remove("include");
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::XaiBridgeAdapter;

    #[test]
    fn normalizes_xai_responses_request() {
        let mut request = json!({
            "model": "grok-4.3",
            "external_web_access": false,
            "include": ["reasoning.encrypted_content", "file_search_call.results"],
            "input": [
                { "type": "message", "role": "user", "content": "run pwd" },
                { "type": "reasoning", "encrypted_content": "opaque-openai-blob" },
                { "type": "function_call_output", "call_id": "call_pwd", "output": "/tmp/project" }
            ],
            "tools": [
                { "type": "tool_search" },
                { "type": "custom", "name": "exec_command" },
                { "type": "web_search", "external_web_access": true },
                { "type": "shell" },
                { "type": "function", "name": "exec_command" }
            ]
        });
        let mut adapter = XaiBridgeAdapter;

        adapter.prepare_responses_request(&mut request);

        assert!(request.get("external_web_access").is_none());
        assert_eq!(request["include"], json!(["file_search_call.results"]));
        assert_eq!(
            request["input"],
            json!([
                { "type": "message", "role": "user", "content": "run pwd" },
                { "type": "function_call_output", "call_id": "call_pwd", "output": "/tmp/project" }
            ])
        );
        assert_eq!(
            request["tools"],
            json!([
                { "type": "web_search" },
                { "type": "shell" },
                { "type": "function", "name": "exec_command" }
            ])
        );
    }
}
