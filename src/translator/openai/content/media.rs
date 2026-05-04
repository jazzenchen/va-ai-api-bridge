pub(super) fn openai_media_url(media_type: Option<&str>, data: &str) -> String {
    if data.starts_with("data:") || data.starts_with("http://") || data.starts_with("https://") {
        return data.to_string();
    }
    media_type
        .map(|media_type| format!("data:{media_type};base64,{data}"))
        .unwrap_or_else(|| data.to_string())
}
