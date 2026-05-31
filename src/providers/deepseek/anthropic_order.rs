use serde_json::Value;

pub(super) fn repair_anthropic_thinking_tool_use_order(request: &mut Value) {
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };

    for message in messages {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        move_first_thinking_before_leading_tool_uses(blocks);
    }
}

fn move_first_thinking_before_leading_tool_uses(blocks: &mut Vec<Value>) {
    let Some(thinking_index) = blocks.iter().position(is_anthropic_thinking_block) else {
        return;
    };
    if thinking_index == 0
        || !blocks[..thinking_index]
            .iter()
            .all(is_anthropic_tool_use_block)
    {
        return;
    }

    let thinking = blocks.remove(thinking_index);
    blocks.insert(0, thinking);
}

fn is_anthropic_thinking_block(block: &Value) -> bool {
    anthropic_block_type(block) == Some("thinking")
        && block
            .get("thinking")
            .and_then(Value::as_str)
            .is_some_and(|thinking| !thinking.is_empty())
}

fn is_anthropic_tool_use_block(block: &Value) -> bool {
    anthropic_block_type(block) == Some("tool_use")
}

fn anthropic_block_type(block: &Value) -> Option<&str> {
    block.get("type").and_then(Value::as_str)
}
