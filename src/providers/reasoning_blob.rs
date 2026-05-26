const REASONING_BLOB_PREFIX: &str = "vibearound.reasoning.hex.v1:";

#[cfg(test)]
pub(crate) fn encode_reasoning_content(content: &str) -> String {
    let mut out = String::with_capacity(REASONING_BLOB_PREFIX.len() + content.len() * 2);
    out.push_str(REASONING_BLOB_PREFIX);
    for byte in content.as_bytes() {
        out.push(hex_digit(byte >> 4));
        out.push(hex_digit(byte & 0x0f));
    }
    out
}

pub(crate) fn decode_reasoning_content(blob: &str) -> Option<String> {
    let hex = blob.strip_prefix(REASONING_BLOB_PREFIX)?;
    if hex.len() % 2 != 0 {
        return None;
    }

    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let raw = hex.as_bytes();
    for pair in raw.chunks_exact(2) {
        let high = from_hex_digit(pair[0])?;
        let low = from_hex_digit(pair[1])?;
        bytes.push((high << 4) | low);
    }
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
fn hex_digit(value: u8) -> char {
    match value {
        0..=9 => (b'0' + value) as char,
        10..=15 => (b'a' + value - 10) as char,
        _ => unreachable!("hex digit out of range"),
    }
}

fn from_hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_reasoning_content, encode_reasoning_content};

    #[test]
    fn reasoning_blob_roundtrips_utf8_content() {
        let content = "I should call the tool.\n然后继续。";

        let encoded = encode_reasoning_content(content);

        assert_eq!(decode_reasoning_content(&encoded).as_deref(), Some(content));
    }
}
