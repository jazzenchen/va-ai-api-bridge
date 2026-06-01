mod content;
mod mapping;
mod route;
mod tools;

pub use route::{attach_route_metadata, strip_route_metadata};

pub(super) use content::{
    blocks_to_gemini_parts, function_call_part, function_call_part_with_signature,
    function_response_part, gemini_function_call_id, gemini_function_response_id,
    gemini_part_to_blocks, gemini_parts_to_blocks, stringify_json,
    thought_signature_from_extensions,
};
pub(super) use mapping::{
    finish_reason_from_gemini, finish_reason_to_gemini, gemini_role_to_universal,
    generation_from_gemini, generation_to_gemini, has_finish_reason, universal_role_to_gemini,
    usage_from_gemini, usage_to_gemini,
};
pub(super) use route::{
    field, model_from_route_segment, GEMINI_THOUGHT_SIGNATURE_KEY, VA_MODEL_KEY, VA_STREAM_KEY,
};
pub(super) use tools::{decode_tool_choice, decode_tools, tool_choice_to_gemini, tools_to_gemini};
